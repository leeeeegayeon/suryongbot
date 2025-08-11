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
from openai import OpenAI
# LangChain
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOpenAI

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

# 입시 정보 문서 로딩
documents = load_paragraphs("documents.txt")

# 질문 추천 문서, 인덱스 로딩
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

# 어떤 문단(hash 값)이 FAISS 인덱스에서 몇 번째(row)에 저장돼 있는지 매핑
DOCROW_BY_HASH = {}
# 인덱스의 행 번호(i)와 해당 행의 문단 ID(doc_id)를 순회
for i, doc_id in enumerate(vectorstore.index_to_docstore_id):
    doc = vectorstore.docstore.search(doc_id)
    if doc:
        # doc이 문자열이므로 그대로 해시값 계산
        DOCROW_BY_HASH[hash(doc)] = i

# 특정 문단(doc)의 임베딩 벡터를 FAISS에서 직접 꺼내기
def get_doc_vector_from_faiss(doc):
    # doc도 문자열이므로 그대로 해시값 계산
    row = DOCROW_BY_HASH.get(hash(doc))
    if row is None:
        return None
    try:
        vec = vectorstore.index.reconstruct(row)
    except Exception:
        return None
    vec = np.asarray(vec, dtype="float32")  # 넘파이 배열로 변환
    return vec / (np.linalg.norm(vec) + 1e-12)  # L2 정규화해서 반환

# MultiQuery retriever(사용자 질문과 유사한 질문 생성)
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm)

# -----------------------------------------------------------------------------
# 이 설정들은 답변을 생성한 뒤, 후보 문단들과 임베딩 유사도를 비교하여 출처를 붙일지 여부를 판단하는 데 사용됩니다.

# POST_HOC_TOP_K        : 유사도 상위 몇 개 문단을 후보로 볼지 결정
# POST_HOC_MIN_SCORE    : 후보 중 최대 코사인 유사도가 이 값보다 낮으면 출처를 붙이지 않음
# POST_HOC_MIN_STD      : 상위 k개 점수의 표준편차가 이 값보다 작으면 랜덤 매칭으로 간주하고 출처를 붙이지 않음
# POST_HOC_MIN_ANSWER_LEN: 답변 길이가 이 문자 수보다 짧으면 출처를 붙이지 않음
# QDOC_MIN/ADOC_MIN     : 질문↔문단 / 답변↔문단 최소 코사인 유사도 기준
# TOP1_MARGIN           : 최고점과 두 번째 점수 차이가 이 값 이상이어야 함
# EMB_CACHE             : 문서 임베딩 캐시. 문단을 임베딩할 때마다 API 호출을 줄이기 위해 사용
# parse_inline_source   : 문단 본문에서 <출처: ...> 형태의 출처만 추출하는 함수
# should_attach_citation: 출처를 붙일지 말지, 조건을 확인하는 함수
# -----------------------------------------------------------------------------

POST_HOC_TOP_K = 4
POST_HOC_MIN_SCORE = 0.28
POST_HOC_MIN_STD = 0.035
POST_HOC_MIN_ANSWER_LEN = 25

QDOC_MIN = 0.28
ADOC_MIN = 0.26
TOP1_MARGIN = 0.03

# 임베딩 캐시 (전역)
EMB_CACHE = {}  # key: 문서 고유ID(또는 내용 해시), value: L2 정규화된 np.array 벡터

def parse_inline_source(text: str) -> str:
    m = re.search(r"<\s*출처[:：]\s*([^>]+)>", text)
    return m.group(1).strip() if m else ""

def should_attach_citation(scores, answer_text) -> bool:
    if not scores:
        return False
    # 길이가 POST_HOC_MIN_ANSWER_LEN보다 짧으면 출처를 붙이지 않음
    if len(answer_text.strip()) < POST_HOC_MIN_ANSWER_LEN:
        return False
    # 유사도가 POST_HOC_MIN_SCORE 보다 낮으면 출처를 붙이지 않음
    top_score = max(scores)
    if top_score < POST_HOC_MIN_SCORE:
        return False
    # 표준편차가 POST_HOC_MIN_STD보다 작으면 출처를 붙이지 않음
    try:
        stdv = statistics.pstdev(scores)
    except statistics.StatisticsError:
        stdv = 0.0
    if stdv < POST_HOC_MIN_STD:
        return False
    #다 통과하면 출처를 붙임
    return True

#답변 생성 요청
@app.post("/query")
async def handle_query(request: QueryRequest):
    query = request.query

    # 문서 검색 (retriever)
    relevant_docs = retriever.invoke(query)

    # relevant_docs의 각각에 포함된 <출처: ...> 제거
    retrieved_docs = []
    for doc in relevant_docs:
        text = doc.page_content
        text_clean = re.sub(r"<\s*출처[:：][^>]+>", "", text).strip()
        retrieved_docs.append(text_clean)
    retrieved = "\n\n".join(retrieved_docs) #문단들을 합쳐서 gpt에게 보냄

    # 프롬프트
    prompt = f"""
            [사용자 질문]  
            {query}

            [관련 내용]  
            {retrieved}

            다음 기준에 따라 답변을 한국어로 작성해줘.

            1. 사용자 질문에 대한 답을 찾을 수 없다면, "자세한 사항은 성신여자대학교 입학처 홈페이지의 입시요강을 참고하거나, 입학처(02-920-2000)에 문의해 주세요."라는 문장을 포함시켜.
            2. 입시 관련 질문이 아니라면(예: 점심 메뉴 추천, 잡담 등), **가볍고 친근하게 스몰토크**로 답해줘.
            3. 인삿말은 매번 하지 않아도 돼.
            4. 의도를 알 수 없는 질문이나 키워드만 있을 경우에는 이렇게 답변해줘: "죄송해요, 질문이 조금 불분명해요. 구체적으로 알려주시면 더 정확하게 안내해드릴 수 있어요! 😊"
            5. 학과명이나 전형명을 언급하지 않고 모집 인원에 대한 질문을 하면, 명시해서 다시 물어봐달라고 해줘.
            6. 학생부 교과 전형의 모집인원에 관한 질문이라면 다음을 참고 해서 답해줘:
                학생부 교과 전형으로는 간호대학(자연), 사범대학 외의 학과를 제외하고 모집하지 않아.
            7. 모집단위를 언급하지 않고 특성화고교출신자 기준학과에 대한 질문을 하면 , 명시해서 다시 물어보라고 안내해줘.
                """

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "너는 성신여자대학교의 입시 안내를 도와주는 챗봇 '수룡이'야. 수험생에게 친절하게 정확한 정보를 제공하는 것이 너의 역할이야. 모든 대답은 수룡의 정체성(성신여대 도우미, 친절하고 똑똑한 용 캐릭터)을 유지한 말투로 작성해줘."},
                  {"role": "user", "content": prompt}],
        temperature=0.4,
        top_p=0.95,
        presence_penalty=0.6,
        frequency_penalty=0.3
    )

    # gpt 답변
    answer = chat_response.choices[0].message.content
    # gpt가 임의로 삽입했을 수 있는 <출처: ...>  제거
    answer = re.sub(r"<\s*출처[:：][^>]+>", "", answer).strip()



    # '현재 검색된 문서'와 'gpt 답변','질문' 임베딩을 비교하여 코사인 유사도가 충분히 높은 문단만 출처를 붙인다.
    try:
        cand_docs = relevant_docs  # 현재 검색된 문단만 비교 대상으로 사용
        if not cand_docs:
            raise RuntimeError("No candidate docs for post-hoc matching")


        # 답변 <-> 문단
        # 답변 임베딩 + L2 정규화(벡터의 길이를 1로)
        answer_vec = embedding_model.embed_query(answer)
        a_vec = np.array(answer_vec, dtype="float32")
        a_vec /= (np.linalg.norm(a_vec) + 1e-12)
        # 문단 벡터 가져오기 + L2 정규화
        def _doc_key(d):
            return hash(d.page_content)
        cand_embs = []
        for d in cand_docs:
            v = get_doc_vector_from_faiss(d) #FAISS 인덱스에 저장돼 있는 벡터 꺼내기
            if v is None: #못가져온 경우, EMB_CACHE에 있는지 확인
                k = _doc_key(d)
                v = EMB_CACHE.get(k)
                if v is None: #없으면 새로 임베딩, 캐시에 저장
                    v_list = embedding_model.embed_documents([d.page_content])[0]
                    v = np.array(v_list, dtype="float32")
                    v /= (np.linalg.norm(v) + 1e-12)
                    EMB_CACHE[k] = v
            cand_embs.append(v) #벡터를 cand_embs에 추가
        # 코사인 유사도: 답변 <-> 문단
        scores_ans_doc = [float(np.dot(a_vec, v)) for v in cand_embs]


        # 질문 <-> 문단
        # 질문 임베딩 + 정규화
        query_vec = embedding_model.embed_query(query)
        q_vec = np.array(query_vec, dtype="float32")
        q_vec /= (np.linalg.norm(q_vec) + 1e-12)
        # 코사인 유사도: 질문 <-> 문단
        scores_q_doc = [float(np.dot(q_vec, v)) for v in cand_embs]


        # 상위 TOP_K 문단 선택
        ranked = sorted(
            zip(cand_docs, scores_ans_doc, scores_q_doc),
            key=lambda x: x[1],
            reverse=True
        )
        posthoc = ranked[:POST_HOC_TOP_K] #상위 K만큼 잘라서 posthoc에 저장
        topk_ans = [s_ad for _, s_ad, _ in posthoc] #두 번째 값(s_ad)(답변 <-> 문단 유사도 점수)만 추출
        topk_q   = [s_qd for _, _, s_qd in posthoc] #세 번째 값(s_qd)(질문 <-> 문단 유사도 점수)만 추출

        # 답변 <-> 문단 유사도 점수들의 표준편차 계산 (클수록 후보 확실)
        try:
            stdv = statistics.pstdev(topk_ans) if len(topk_ans) > 1 else 0.0
        except statistics.StatisticsError:
            stdv = 0.0
        # 1, 2등 점수를 뽑음
        top1 = topk_ans[0] if topk_ans else 0.0
        top2 = topk_ans[1] if len(topk_ans) > 1 else 0.0
        # 둘의 차이가 TOP1_MARGIN보다 큰 지 (클수록 후보 확실)
        margin_ok = (top1 - top2) >= TOP1_MARGIN

        # 질문 <-> 문단, 답변 <-> 문단의 최고 유사도가 최소 기준 이상인지 확인
        qdoc_ok = (max(topk_q)   if topk_q   else 0.0) >= QDOC_MIN
        adoc_ok = (max(topk_ans) if topk_ans else 0.0) >= ADOC_MIN

        # 기존 기준 (답변 길이, 최고점, 분산) 만족하는지
        attach_basic = should_attach_citation(topk_ans, answer)

        # 최종 부착 여부 결정
        attach = (
                len(answer.strip()) >= POST_HOC_MIN_ANSWER_LEN and
                qdoc_ok and adoc_ok and
                (attach_basic or margin_ok)
        )

        if attach:
            # 상위 K 중 첫 번째 문단에서 출처만 추출
            top_doc, _, _ = posthoc[0]
            inline = parse_inline_source(top_doc.page_content)
            if inline:
                citation_sentence = f"(본 내용은 2026 수시모집요강 {inline}를 참고하여 작성되었습니다.)"
                answer = answer.rstrip() + " " + citation_sentence


    except Exception:
        # 사후 매칭 도중 오류가 발생하면 출처 부착을 건너뜁니다.
        answer += ""

    # 최종 답변 반환
    return {"answer": answer}


#사용자가 입력한 질문과 유사한 추천 질문 선정, 프론트에 보내기
@app.post("/suggest")
async def recommend_questions_endpoint(request: QueryRequest):
    query = request.query

    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    #사용자 질문 임베딩
    query_embedding = np.array(embedding_response.data[0].embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # 추천 질문 후보를 10개로 뽑음
    top_k = 10
    scores, indices = recommend_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    # 임계값
    THRESH = 0.35
    # (점수, 인덱스) 쌍을 만들어 임계치 이상만 남김
    pairs = [(float(scores[0][i]), int(indices[0][i])) for i in range(len(indices[0]))]
    filtered = [(s, idx) for (s, idx) in pairs if s >= THRESH]
    # 점수 순으로 정렬 후 상위 3개만 반환
    filtered.sort(key=lambda x: x[0], reverse=True)
    filtered = filtered[:3]
    similar_questions = [recommend_questions[idx] for (s, idx) in filtered]
    return {"results": similar_questions}

#메인 화면
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#채팅 화면
@app.get("/chat", response_class=HTMLResponse)
async def serve_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

#FAQ
@app.get("/jungsi_faq", response_class=HTMLResponse)
async def serve_jungsi_faq(request: Request):
    return templates.TemplateResponse("jungsi_faq.html", {"request": request})
