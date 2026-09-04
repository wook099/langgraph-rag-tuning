# 광주 여행 RAG 챗봇 서버

FastAPI + LangGraph + LangChain 기반의 RAG(Retrieval-Augmented Generation) 챗봇 백엔드입니다.
업로드된 문서를 ChromaDB에 벡터로 저장하고, 질문이 들어오면 관련 문서를 검색해 그 내용을 바탕으로 답변을 생성합니다.

## 주요 기능

- **RAG 기반 채팅** (`/chat`): LangGraph 워크플로우로 `retrieve → generate` 단계를 구성하여, 질문과 관련된 문서를 검색한 뒤 그 내용을 근거로 답변을 생성합니다. `use_rag` 플래그로 RAG 검색을 켜고 끌 수 있습니다.
- **문서 업로드 및 벡터 적재** (`/upload`): `.txt`, `.md`, `.pdf` 파일을 업로드하면 자동으로 텍스트를 추출하고 청크로 분할해 ChromaDB에 저장합니다. 마크다운 파일은 헤더(`#`, `##`, `###`) 단위로 분할하고, 그 외 파일은 `RecursiveCharacterTextSplitter`로 분할합니다.
- **DB 초기화** (`/reset-db`): ChromaDB에 저장된 모든 문서를 삭제합니다.
- **CORS 실습 API** (`/simpleparam`, `/simplejson`): Simple Request와 Preflight Request의 차이를 실습하기 위한 엔드포인트입니다.
- **RAG 성능 평가** (`eval_ragas.py`): [Ragas](https://github.com/explodinggradients/ragas) 라이브러리로 Faithfulness, Answer Relevancy, Context Recall, Context Precision 4개 지표를 측정하고 결과를 CSV/Markdown으로 저장합니다.

## 폴더 구조

```
servers/
├── main.py                    # FastAPI 서버 진입점 (RAG 워크플로우, API 라우트 정의)
├── eval_ragas.py               # Ragas를 이용한 RAG 성능 평가 스크립트
├── requirements.txt            # Python 패키지 의존성 목록
├── test_cases.json             # 평가용 질문/정답(ground_truth) 테스트 케이스
├── data/                       # RAG에 사용할 원본 문서 저장 폴더 (업로드 파일도 여기 저장됨)
│   ├── SSAFY_GUIDE.txt
│   └── 전국관광지정보_RAG용.md
├── chroma_data/                 # ChromaDB 벡터 저장소 (서버 종료 시 자동 삭제됨)
├── .env                         # 환경 변수 설정 파일 (직접 생성 필요)
└── .venv/                       # Python 가상환경
```

## 실행 방법

### 1. 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 채웁니다.

```env
API_KEY=발급받은_API_키
base_url=사용할_API_Base_URL
GPT_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 4. 서버 실행

```bash
python main.py
```

또는

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 정상적으로 실행되면 `http://localhost:8000` 에서 API를 사용할 수 있습니다.
FastAPI 자동 문서는 `http://localhost:8000/docs` 에서 확인 가능합니다.

> ⚠️ 서버 종료 시 `chroma_data/` 폴더가 자동으로 삭제되도록 구성되어 있습니다 (lifespan 종료 훅). 데이터를 영구 보존하려면 `main.py`의 종료 로직을 수정하세요.

## API 사용 예시

### 채팅 요청

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "아트빌리지 어때?", "use_rag": true}'
```

### 문서 업로드

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@./data/SSAFY_GUIDE.txt"
```

### DB 초기화

```bash
curl -X POST http://localhost:8000/reset-db
```

## RAG 성능 평가 실행

```bash
python eval_ragas.py
```

- `test_cases.json`이 있으면 해당 테스트 케이스를 사용하고, 없으면 Ragas Testset Generator로 자동 생성합니다.
- 평가 결과는 `ragas_eval_results.csv`(전체 결과)와 `ragas_eval_history.md`(누적 기록)로 저장됩니다.

## 기술 스택

- **Web Framework**: FastAPI, Uvicorn
- **RAG / LLM**: LangChain, LangGraph, OpenAI(호환 API)
- **Vector DB**: ChromaDB
- **평가**: Ragas
- **문서 파싱**: PyMuPDF(PDF), TextLoader(TXT/MD)
