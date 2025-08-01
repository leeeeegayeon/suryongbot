# build_index.py
# -*- coding: utf-8 -*-
from chatbot_faiss_utils import *

# 문단 불러오기
documents = load_paragraphs("documents.txt")

# 문단 임베딩 및 저장
embeddings = embed_documents_openai(documents)
save_embeddings(embeddings)
build_faiss_index(embeddings)

print("임베딩 및 FAISS 인덱스 생성 완료")
