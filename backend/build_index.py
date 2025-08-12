from langchain_core.documents import Document
from chatbot_faiss_utils import load_paragraphs
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 문단 불러오기 → Document로 변환
paragraphs = load_paragraphs("documents.txt")
documents = [Document(page_content=p) for p in paragraphs]

# 임베딩 모델
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

# 벡터스토어 생성 및 저장
vectorstore = FAISS.from_documents(documents, embedding=embedding_model)
vectorstore.save_local("index_openai")

print("LangChain용 FAISS 인덱스 저장 완료")
