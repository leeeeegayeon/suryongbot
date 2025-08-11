#main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot_faiss_utils import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import re
import numpy as np
import statistics

# LangChain 관련 추가
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOpenAI

from openai import OpenAI

app = FastAPI()

# 절대 경로로 static 디렉토리 지정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_PATH = os.path.join(BASE_DIR, "app", "static")
TEMPLATE_PATH = os.path.join(BASE_DIR, "app", "templates")

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

templates = Jinja2Templates(directory=TEMPLATE_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 문서 로딩
documents = load_paragraphs("documents.txt")

# 질문 추천 인덱스 로딩
recommend_questions = load_paragraphs("question_candidates.txt")
recommend_embeddings = load_embeddings("recommend_embeddings.npy")
recommend_index = load_faiss_index("recommend_index.faiss")

class QueryRequest(BaseModel):
    query: str

# LangChain용 FAISS + retriever 설정
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = LangChainFAISS.load_local(
    "index_openai",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
llm = ChatOpenAI(model="gpt-4o", temperature=0.4)


# LLM 기반 MultiQuery retriever
retriever_multi = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

# 의미 + 키워드 기반 앙상블 리트리버
retriever = retriever_multi

# 출처 표기 기준
POST_HOC_TOP_K = 5
POST_HOC_MIN_SCORE = 0.28
POST_HOC_MIN_STD = 0.035
POST_HOC_MIN_ANSWER_LEN = 20

EMB_CACHE = {}  # key: 문서 고유ID(또는 내용 해시), val: 정규화 임베딩 np.array

def _parse_inline_source(text: str) -> str:
    """본문에 붙은 <출처: ...>만 추출. 없으면 빈 문자열."""
    m = re.search(r"<\s*출처[:：]\s*([^>]+)>", text)
    return m.group(1).strip() if m else ""

def _should_attach_citation(scores, answer_text) -> bool:
    if not scores:
        return False
    if len(answer_text.strip()) < POST_HOC_MIN_ANSWER_LEN:
        return False
    top_score = max(scores)
    if top_score < POST_HOC_MIN_SCORE:
        return False
    try:
        stdv = statistics.pstdev(scores)
    except statistics.StatisticsError:
        stdv = 0.0
    if stdv < POST_HOC_MIN_STD:
        return False
    return True


@app.post("/query")
async def handle_query(request: QueryRequest):
    query = request.query

    # 문서 검색
    relevant_docs = retriever.invoke(query)

    # 출처 정보 추출 + 모델 입력용 텍스트에서 <출처: ...> 제거
    retrieved_docs = []
    for doc in relevant_docs:
        text = doc.page_content
        text_clean = re.sub(r"<\s*출처[:：][^>]+>", "", text).strip()
        retrieved_docs.append(text_clean)

    retrieved = "\n\n".join(retrieved_docs)

    # GPT 프롬프트 구성
    prompt = f"""
                [사용자 질문]  
                {query}

                [관련 내용]  
                {retrieved}

                다음 기준에 따라 답변을 한국어로 작성해줘.

                1. 사용자 질문에 대한 답을 찾을 수 없는 경우에는, "자세한 사항은 성신여자대학교 입학처 홈페이지의 입시요강을 참고하거나, 입학처(02-920-2000)에 문의해 주세요."라는 문장을 포함시켜.
                2. 입시 관련 질문이 아니라면(예: 점심 메뉴 추천, 잡담 등), **가볍고 친근하게 스몰토크**로 답해줘.
                3. 인삿말은 매번 하지 않아도 돼.
                4. 의도를 알 수 없는 질문이나 키워드만 있을 경우에는 이렇게 답변해줘: "죄송해요, 질문이 조금 불분명해요. 구체적으로 알려주시면 더 정확하게 안내해드릴 수 있어요! 😊"
                5. 학과명이나 전형명을 언급하지 않고 모집 인원에 대한 질문을 하면, 명시해서 다시 물어보라고 안내해줘.
                6. 학생부 교과 전형의 모집인원에 관한 질문이라면 다음을 참고 해서 답해줘:
                    학생부 교과 전형으로는 간호대학(자연), 사범대학 외의 학과를 제외하고 모집하지 않아.
                7. 모집단위를 언급하지 않고 특성화고교출신자 기준학과에 대한 질문을 하면 , 명시해서 다시 물어보라고 안내해줘.
                
                """

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system",
                   "content": "너는 성신여자대학교의 입시 안내를 도와주는 챗봇 '수룡이'야. 수험생에게 친절하게 정확한 정보를 제공하는 것이 너의 역할이야. 모든 대답은 수룡의 정체성(성신여대 도우미, 친절하고 똑똑한 용 캐릭터)을 유지한 말투로 작성해줘."},
                  {"role": "user", "content": prompt}],
        temperature=0.4,
        top_p=0.95,
        presence_penalty=0.6,
        frequency_penalty=0.3
    )

    answer = chat_response.choices[0].message.content

    # 모델이 임의로 붙였을 수도 있는 본문 각주 형태 제거
    answer = re.sub(r"<\s*출처[:：][^>]+>", "", answer).strip()

    # 답변 - 문단 유사도 비교
    try:
        cand_docs = relevant_docs
        if not cand_docs:
            raise RuntimeError("No candidate docs for post-hoc matching")

        # 답변 임베딩 + 정규화
        answer_vec = embedding_model.embed_query(answer)
        q = np.array(answer_vec, dtype="float32")
        q /= (np.linalg.norm(q) + 1e-12)

        # 문서 임베딩 + 정규화 (캐시 활용)
        def _doc_key(d):
            return d.metadata.get("id") or hash(d.page_content)

        cand_embs = []
        for d in cand_docs:
            k = _doc_key(d)
            v = EMB_CACHE.get(k)
            if v is None:
                v_list = embedding_model.embed_documents([d.page_content])[0]
                v = np.array(v_list, dtype="float32")
                v /= (np.linalg.norm(v) + 1e-12)
                EMB_CACHE[k] = v
            cand_embs.append(v)

        # 코사인 유사도
        scores = [float(np.dot(q, v)) for v in cand_embs]

        # 상위 TOP_K 선별
        ranked = sorted(zip(cand_docs, scores), key=lambda x: x[1], reverse=True)
        posthoc_docs = ranked[:POST_HOC_TOP_K]
        topk_scores = [s for _, s in posthoc_docs]

        # 출처 부착 여부 판정
        if _should_attach_citation(topk_scores, answer):
            # 1) 한 줄 출처 문장 — 본문 <출처: ...>만 사용
            citation_sentence = ""
            for d, s in posthoc_docs:
                inline = _parse_inline_source(d.page_content)
                if inline:
                    citation_sentence = f"(본 내용은 2026 수시모집요강 {inline}를 참고하여 작성되었습니다.)"
                    break
            if citation_sentence:
                answer = answer.rstrip() + " " + citation_sentence

            # 2) 하단 참고 출처 블럭 (상위 k)
            citations = []
            for rank, (d, s) in enumerate(posthoc_docs, start=1):
                inline = _parse_inline_source(d.page_content)
                if not inline:
                    continue
                snippet = (d.page_content.strip().splitlines() or [""])[0]
                snippet = re.sub(r"<\s*출처[:：][^>]+>", "", snippet).strip()
                if len(snippet) > 80:
                    snippet = snippet[:80] + "..."
                citations.append(f"{rank}. 출처: {inline} | score={s:.3f} | {snippet}")
            if citations:
                answer += "\n\n—\n 참고 출처(사후 매칭 · 코사인):\n" + "\n".join(f"- {c}" for c in citations)

    except Exception:
        answer += "\n\n(참고: 사후 매칭 중 오류가 발생하여 출처 자동 첨부를 건너뛰었습니다.)"

    return {"answer": answer}


@app.post("/suggest")
async def recommend_questions_endpoint(request: QueryRequest):
    query = request.query

    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(embedding_response.data[0].embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    top_k = 10
    scores, indices = recommend_index.search(
        np.array([query_embedding], dtype=np.float32), top_k )

    THRESH = 0.35
    # 상위 3개만 반환
    pairs = [(float(scores[0][i]), int(indices[0][i])) for i in range(len(indices[0]))]
    filtered = [(s, idx) for (s, idx) in pairs if s >= THRESH]
    # 필요하면 점수 내림차순 정렬
    filtered.sort(key=lambda x: x[0], reverse=True)
    filtered = filtered[:3]

    similar_questions = [recommend_questions[idx] for (_, idx) in filtered]
    return {"results": similar_questions}


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def serve_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
