#main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot_faiss_utils import *
from openai import OpenAI
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

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

# 문서 및 인덱스 로딩
documents = load_paragraphs("documents.txt")
doc_embeddings = load_embeddings()
index = load_faiss_index()

# 질문 추천 인덱스 로딩
recommend_questions = load_paragraphs("question_candidates.txt")
recommend_embeddings = load_embeddings("recommend_embeddings.npy")
recommend_index = load_faiss_index("recommend_index.faiss")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(request: QueryRequest):
    query = request.query

    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(embedding_response.data[0].embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    top_k = 3
    scores, indices = index.search(np.array([query_embedding]), top_k)
    retrieved = "\n\n".join([documents[idx] for idx in indices[0]])

    prompt = f"""너는 성신여자대학교의 입시 안내를 도와주는 챗봇 \"수룡이\"야.  
성신여대를 지원하고자 하는 수험생과 학부모에게 정확하고 친절한 정보를 제공하는 것이 너의 역할이야.

[문서 내용]  
{retrieved}

[사용자 질문]  
{query}

다음 기준에 따라 답변을 작성해줘. 반드시 **한국어**로 답해.

1. 문서에 관련 정보가 있을 경우, 신뢰할 수 있도록 문서에 기반한 내용을 바탕으로 정확하게 설명해줘.
2. 답을 찾을 수 없다면, \"죄송해요, 해당 내용은 제가 가진 자료로는 확인할 수 없어요. 자세한 사항은 성신여자대학교 입학처 홈페이지의 입시요강을 참고하거나, 입학처(02-920-2000)에 문의해 주세요.\"라고 꼭 말해
3. 질문이 입시 관련이 아니라면(예: 점심 메뉴 추천, 잡담 등), 수룡이라는 캐릭터를 유지하면서도 **가볍고 친근하게 스몰토크**로 답해줘. 단, 너무 장황하게 늘어놓지는 말고 핵심만 짧고 유쾌하게 말해.
4. 모든 대답은 수룡의 정체성(성신여대 도우미, 친절하고 똑똑한 용 캐릭터)을 유지한 말투로 작성해줘.
5. 입시 정보에 대해 답변할 때 인삿말은 매번 하지 않아도 돼.
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
