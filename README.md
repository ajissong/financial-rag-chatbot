# Financial RAG Chatbot (금융 용어 검색 챗봇)

ABC은행의 금융 용어 700선을 기반으로 한 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 🚀 주요 기능

- **금융 용어 검색**: 경제금융용어 700선 데이터베이스 기반 검색
- **친근한 답변**: 반말로 친근하게 답변하는 챗봇
- **Gradio UI**: 웹 기반 사용자 인터페이스
- **벡터 DB 캐싱**: 한 번 생성된 벡터 DB 재사용으로 빠른 로딩

## 🛠️ 기술 스택

- **LangChain**: 최신 LCEL(LangChain Expression Language) 사용
- **OpenAI GPT-4o-mini**: LLM 모델
- **Chroma**: 벡터 데이터베이스
- **Gradio**: 웹 인터페이스
- **PyPDF**: PDF 문서 처리

## 📦 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv finrag_env
source finrag_env/bin/activate  # Linux/Mac
# 또는
finrag_env\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 실행
```bash
python config.py
```

## 🏗️ 아키텍처

### 최신 LangChain LCEL 구조
```python
chain = (
    RunnableLambda(create_rag_chain)  # RAG 로직 처리
    | PromptTemplate                  # 프롬프트 템플릿
    | ChatOpenAI                     # LLM 호출
    | StrOutputParser()              # 출력 파싱
)
```

### 벡터 DB 최적화
- **첫 실행**: PDF 로드 → 임베딩 생성 → 벡터 DB 저장
- **재실행**: 기존 벡터 DB 로드 (빠른 시작)

## 📁 프로젝트 구조

```
finrag/
├── config.py              # 메인 애플리케이션
├── requirements.txt       # 의존성 목록
├── README.md             # 프로젝트 문서
├── .gitignore            # Git 제외 파일
├── .env                  # 환경 변수 (사용자 생성)
├── chroma_db/           # 벡터 데이터베이스 (자동 생성)
└── 2020_경제금융용어 700선_게시.pdf  # 원본 데이터
```

## 🔧 주요 함수

- `load_pdf()`: PDF 문서 로드 및 전처리
- `create_vectordb()`: 벡터 데이터베이스 생성/로드
- `create_chain()`: LCEL 기반 RAG 체인 생성
- `create_ui()`: Gradio 웹 인터페이스 생성

## 🎯 사용 예시

```
질문: "디커플링이란 무엇인가?"
답변: "디커플링은 경제에서 두 변수 간의 관계가 약해지거나 끊어지는 현상을 말해. 
       예를 들어, 경제성장과 에너지 소비 간의 관계가 약해지는 것을 의미해."
```

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈를 통해 제출해 주세요.