# build_recommend_index.py
from chatbot_faiss_utils import *

questions = load_paragraphs("question_candidates.txt")
embeddings = embed_documents_openai(questions)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

save_embeddings(embeddings, "recommend_embeddings.npy")
build_faiss_index(embeddings, "recommend_index.faiss")