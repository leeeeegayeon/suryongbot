#main.py
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_faiss_utils import *
from openai import OpenAI
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 환경변수에서 읽음

# 문서 및 인덱스 로딩
documents = load_paragraphs("documents.txt")
doc_embeddings = load_embeddings()
index = load_faiss_index()

# 요청 형식 정의
class QueryRequest(BaseModel):
    query: str

# POST /query 엔드포인트 정의
@app.post("/query")
async def handle_query(request: QueryRequest):
    query = request.query

    # 1. 질문 임베딩
    embedding_response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(embedding_response.data[0].embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # 2. FAISS로 유사 문단 검색
    top_k = 3
    scores, indices = index.search(np.array([query_embedding]), top_k)
    retrieved = "\n\n".join([documents[idx] for idx in indices[0]])

    # 3. GPT 프롬프트 구성
    prompt = f"""너는 성신여자대학교의 입시 안내를 도와주는 챗봇 "수룡이"야.  
성신여대를 지원하고자 하는 수험생과 학부모에게 정확하고 친절한 정보를 제공하는 것이 너의 역할이야.

[문서 내용]  
{retrieved}

[사용자 질문]  
{query}

다음 기준에 따라 답변을 작성해줘. 반드시 **한국어**로 답해.

1. 문서에 관련 정보가 있을 경우, 신뢰할 수 있도록 문서에 기반한 내용을 바탕으로 정확하게 설명해줘.
2. 문서에서 답을 찾을 수 없다면, "자세한 사항은 성신여자대학교 입학처 홈페이지의 입시요강을 참고하거나, 입학처(02-920-2000)에 문의해 주세요."라는 문장을 꼭 포함시켜.
3. 질문이 입시 관련이 아니라면(예: 점심 메뉴 추천, 잡담 등), 수룡이라는 캐릭터를 유지하면서도 **가볍고 친근하게 스몰토크**로 답해줘. 단, 너무 장황하게 늘어놓지는 말고 핵심만 짧고 유쾌하게 말해.
4. 모든 대답은 수룡의 정체성(성신여대 도우미, 친절하고 똑똑한 용 캐릭터)을 유지한 말투로 작성해줘.
"""

    # 4. GPT 응답 생성
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        top_p=0.95,
        presence_penalty=0.6,
        frequency_penalty=0.3
    )

    # 5. 프론트로 응답 반환
    return {"answer": chat_response.choices[0].message.content}
