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


# LLM 기반 MultiQuery retriever
retriever_multi = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

# 의미 + 키워드 기반 앙상블 리트리버
retriever = retriever_multi

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

    # 4. GPT 프롬프트 구성
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
                6. 문단마다 적혀있는 출처 정보를 포함해서 답해줘.
                7. 학생부 교과 전형의 모집인원에 관한 질문이라면 다음을 참고 해서 답해줘:
                    학생부 교과 전형으로는 간호대학(자연), 사범대학 외의 학과를 제외하고 모집하지 않아.
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
    
@app.get("/faq", response_class=HTMLResponse)
async def serve_faq(request: Request):
    return templates.TemplateResponse("common_faq.html", {"request": request})

@app.get("/susi_faq", response_class=HTMLResponse)
async def serve_susi_faq(request: Request):
    return templates.TemplateResponse("susi_faq.html", {"request": request})

@app.get("/jungsi_faq", response_class=HTMLResponse)
async def serve_jungsi_faq(request: Request):
    return templates.TemplateResponse("jungsi_faq.html", {"request": request})



