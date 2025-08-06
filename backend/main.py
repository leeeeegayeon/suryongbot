import os
from dotenv import load_dotenv
from openai import OpenAI
import re

from chatbot_faiss_utils import (
    load_paragraphs,
    load_embeddings,
    load_faiss_index
)

# LangChain 최신 권장 방식으로 수정
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder


# 환경 변수 로드 및 클라이언트 설정
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 문서 불러오기
documents = load_paragraphs("documents.txt")

# BM25 리트리버용 문서 구성
bm25_documents = [Document(page_content=doc) for doc in documents]
retriever_bm25 = BM25Retriever.from_documents(bm25_documents)
retriever_bm25.k = 3

# LangChain용 벡터스토어 및 리트리버 구성
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
vectorstore = LangChainFAISS.load_local(
    "index_openai",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

retriever_multi = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm
)

retriever = EnsembleRetriever(
    retrievers=[retriever_bm25, retriever_multi],
    weights=[0.4, 0.6]
)

# 사용자 질문 입력 루프
while True:
    query = input("\n질문 입력 (종료하려면 'exit'): ")
    if query.strip().lower() == "exit":
        break

    # 1. BM25
    docs_bm25 = retriever_bm25.invoke(query)
    print("\n🔍 [BM25] 선택된 문단:")
    for i, doc in enumerate(docs_bm25):
        print(f"[{i+1}] {doc.page_content[:80]}...")

    # 2. MultiQuery
    docs_multi = retriever_multi.invoke(query)
    print("\n🔍 [MultiQuery] 선택된 문단:")
    for i, doc in enumerate(docs_multi):
        print(f"[{i+1}] {doc.page_content[:80]}...")

    # 3. 앙상블 결과
    relevant_docs = retriever.invoke(query)
    print("\n📄 [Ensemble 결과 문단]:")
    for i, doc in enumerate(relevant_docs):
        print(f"{i+1}. {doc.page_content[:80]}...")

    # 4. LongContextReorder 적용
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(relevant_docs)
    print("\n📄 [LongContextReorder 적용 결과]:")
    for i, doc in enumerate(reordered_docs):
        print(f"{i+1}. {doc.page_content[:80]}...")

    # 5. GPT 프롬프트 구성 전에 출처 정리
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

    # GPT 프롬프트 구성
    retrieved = "\n\n".join([doc.page_content for doc in reordered_docs])
    prompt = f"""너는 성신여자대학교 입시를 안내하는 챗봇 "수룡이"야.  
성신여대를 지원하려는 수험생과 학부모에게 문서 기반으로 친절하고 정확한 정보를 제공하는 게 너의 역할이야.

[문서 내용]  
{retrieved}

[출처 정보]  
{source_note}

[사용자 질문]  
{query}

다음 기준에 따라 **한국어로** 답해.

1. 문서에 관련 정보가 있으면 그 내용을 바탕으로 정확하게 답해. 근거 없는 추측은 하지 마.
2. 문서에서 답을 찾을 수 없다면, 다음 문장을 꼭 포함시켜:  
"자세한 사항은 성신여자대학교 입학처 홈페이지의 입시요강을 참고하거나, 입학처(02-920-2000)에 문의해 주세요."
3. 입시 질문이 아닌 경우에는 수룡이 캐릭터를 유지해서 짧고 유쾌하게 스몰토크해.
4. 항상 수룡이라는 캐릭터의 말투(친절하고 똑똑한 용)를 유지해.
5. 문서에 출처가 포함되어 있어도, 답변 본문에는 넣지 말고, 마지막에 아래 형식으로 한 줄만 붙여줘:  
(위 답변은 수시모집요강 p.16, p.17을 참고하여 작성되었습니다.)
6. 질문이 너무 짧거나 불분명한 경우엔 이렇게 말해줘:  
"죄송해요, 질문이 조금 불분명해요. 어떤 모집에 대해 궁금하신가요? 구체적으로 알려주시면 더 정확하게 안내해드릴 수 있어요! 😊"

"""

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        top_p=0.95,
        presence_penalty=0.6,
        frequency_penalty=0.3
    )

    answer = chat_response.choices[0].message.content
    print("\n💬 수룡이의 답변:\n" + answer)
