#chatbot_faiss_utils.py
import openai
import os
import numpy as np
import faiss

openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. 문단 불러오기
def load_paragraphs(path="documents.txt"):
    with open(path, "r", encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    return paragraphs

# 2. OpenAI 임베딩
def embed_documents_openai(documents, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=documents,
        model=model
    )
    embeddings = [np.array(obj.embedding) for obj in response.data]
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

# 3. 임베딩 저장/불러오기
def save_embeddings(embeddings, path="doc_embeddings_openai.npy"):
    np.save(path, embeddings)

def load_embeddings(path="doc_embeddings_openai.npy"):
    return np.load(path)

# 4. FAISS 인덱스 생성/저장/불러오기
def build_faiss_index(embeddings, index_path="index_openai.faiss"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

def load_faiss_index(index_path="index_openai.faiss"):
    return faiss.read_index(index_path)
