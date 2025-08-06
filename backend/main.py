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

# LangChain 관련 추가
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder


# OpenAI 임베딩만 별도로 쓸 거면 client 유지
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

# BM25 리트리버용 문서 변환
bm25_documents = [Document(page_content=doc) for doc in documents]
retriever_bm25 = BM25Retriever.from_documents(bm25_documents)
retriever_bm25.k = 3

# LLM 기반 MultiQuery retriever
retriever_multi = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

# 의미 + 키워드 기반 앙상블 리트리버
retriever = EnsembleRetriever(
    retrievers=[retriever_bm25, retriever_multi],
    weights=[0.4, 0.6]
)

@app.post("/query")
async def handle_query(request: QueryRequest):
    query = request.query

    # 1. 문서 검색 (앙상블 리트리버)
    relevant_docs = retriever.invoke(query)

    # 2. LongContextReorder로 순서 재정렬
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(relevant_docs)

    # 3. 출처 정보 추출
    retrieved_docs = []
    source_pages = []

    for doc in reordered_docs:
        text = doc.page_content
        match = re.search(r"<출처:\s*(.*?)>", text)
        source = match.group(1).strip() if match else "출처 미상"
        text_clean = re.sub(r"<출처:.*?>", "", text).strip()
        retrieved_docs.append(text_clean)
        source_pages.append(source)

    retrieved = "\n\n".join(retrieved_docs)
    unique_sources = sorted(set(source_pages))
    source_note = f"(위 답변은 수시모집요강 {', '.join(unique_sources)}을 참고하여 작성되었습니다.)"

    # 4. GPT 프롬프트 구성
    prompt = f"""너는 성신여자대학교의 입시 안내를 도와주는 챗봇 "수룡이"야.  
성신여대를 지원하고자 하는 수험생과 학부모에게 정확하고 친절한 정보를 제공하는 것이 너의 역할이야.

[문서 내용]  
{retrieved}

[출처 정보]  
{source_note}

[사용자 질문]  
{query}

다음 기준에 따라 답변을 작성해줘. 반드시 **한국어**로 답해.

1. 문서에 관련 정보가 있을 경우, 신뢰할 수 있도록 문서에 기반한 내용을 바탕으로 정확하게 설명해줘.
2. 문서에서 답을 찾을 수 없다면, "자세한 사항은 성신여자대학교 입학처 홈페이지의 입시요강을 참고하거나, 입학처(02-920-2000)에 문의해 주세요."라는 문장을 꼭 포함시켜.
3. 질문이 입시 관련이 아니라면(예: 점심 메뉴 추천, 잡담 등), 수룡이라는 캐릭터를 유지하면서도 **가볍고 친근하게 스몰토크**로 답해줘. 단, 너무 장황하게 늘어놓지는 말고 핵심만 짧고 유쾌하게 말해.
4. 모든 대답은 수룡의 정체성(성신여대 도우미, 친절하고 똑똑한 용 캐릭터)을 유지한 말투로 작성해줘.
5. 입시 정보에 대해 답변할 때 인삿말은 매번 하지 않아도 돼.
6. 문서에서 가져온 정보로 답변할 때는 근거로 사용된 문단의 출처가 포함되어 있다면, 출처를 따로 문장마다 넣지 말고, 답변 마지막 줄에 다음 형식으로 한 번만 요약해서 넣어줘:
(위 답변은 수시모집요강 p.16,p.17을 참고하여 작성되었습니다.)
7. 만약 질문이 단어 하나만 포함된 너무 짧은 질문이거나, 예를 들어 "모집인원"처럼 불분명한 키워드만 있을 경우에는 아래처럼 답변해줘:
죄송해요, 질문이 조금 불분명해요. 어떤 모집에 대해 궁금하신가요? 구체적으로 알려주시면 더 정확하게 안내해드릴 수 있어요! 😊
"""

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        top_p=0.95,
        presence_penalty=0.6,
        frequency_penalty=0.3
    )

    return {"answer": chat_response.choices[0].message.content}

@app.post("/suggest")
async def recommend_questions_endpoint(request: QueryRequest):
    query = request.query

    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(embedding_response.data[0].embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    top_k = 3
    scores, indices = recommend_index.search(np.array([query_embedding]), top_k)
    similar_questions = [recommend_questions[idx] for idx in indices[0]]

    return {"results": similar_questions}

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def serve_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
