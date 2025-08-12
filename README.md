![입시도우미 수룡이](입시도우미%20수룡이.png)
# 입시도우미 수룡이🐉🔮

## 프로젝트 소개

성신여자대학교 입시 도우미 **수룡이**입니다.
2026학년도 수시모집 입시 요강 정보를 바탕으로 수험생 질문에 정확하고 근거 있는 답을 제공하는 웹 기반 챗봇입니다.

## 핵심 기능

* 사용자 질문과 유사한 질문 추천
* 사용자 질문과 문서 간 유사도 검색(FAISS + OpenAI 임베딩)
* 검색 근거를 바탕으로 한 신뢰도 높은 답변 생성
* 웹 UI에서 실시간 질의응답

## 기술 스택

* **Backend**: Python, FastAPI
* **Retrieval**: FAISS, OpenAI Embeddings
* **LLM**: OpenAI API
* **Frontend**: HTML, CSS, JavaScript

## 레포 구조

```text
suryongbot/
├─ app/                 # 프론트엔드
│  ├─ templates/
│  └─ static/
├─ backend/             # 백엔드
│  ├─ main.py
│  └─ chatbot_faiss_utils.py
└─ README.md
```
