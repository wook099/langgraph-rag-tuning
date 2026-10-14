from __future__ import annotations

import os
import re
import warnings
warnings.filterwarnings("ignore")

# transformers 및 huggingface 캐시 관련 경고 방지
if "HF_HOME" not in os.environ and "TRANSFORMERS_CACHE" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, TypedDict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from dotenv import load_dotenv

# LangGraph Core 임포트
from langgraph.graph import StateGraph, START, END

# .env 로드
load_dotenv()

# ==========================================
# [설정 영역]
# ==========================================
OPENAI_API_KEY = os.getenv("API_KEY", "키를 입력하세요.")   # .env의 API_KEY 로드
BASE_URL = os.getenv("base_url")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH = "./chroma_data" # ChromaDB가 파일로 저장될 로컬 디렉토리 경로
COLLECTION_NAME = "trip_docs"

# [RAG 튜닝 설정] 서버 기본값은 .env에서 읽고, 실험 시에는 그래프 실행 인자(RunnableConfig)로 덮어씀
#   rag_workflow.invoke(state, config={"configurable": {"top_k": 3, "threshold": 0.15}})
DEFAULT_RAG_CONFIG = {
    "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),          # 텍스트를 분할할 글자 수 (청크 크기)
    "overlap": int(os.getenv("OVERLAP", 50)),                 # 청크 간 겹치게 할 글자 수 (문맥 보존용)
    "md_split_mode": os.getenv("MD_SPLIT_MODE", "header"),    # .md 분할 방식: header(헤더 단위) | recursive(글자 수 단위)
    "top_k": int(os.getenv("RAG_TOP_K", 5)),                  # LLM에 전달할 최대 청크 수
    "threshold": float(os.getenv("RAG_SCORE_THRESHOLD", 0.0)),  # 코사인 유사도가 이 값 미만인 청크는 제외
    "retriever": os.getenv("RETRIEVER_MODE", "vector"),       # vector(의미 검색) | hybrid(의미 + BM25 키워드 검색)
}
RRF_K = 60   # Reciprocal Rank Fusion 상수 (일반적으로 60 사용)

# ==========================================
# [전역 변수] 서버 실행 중 메모리에 유지될 객체들
# ==========================================
llm = None                  # OpenAI 연결 클라이언트
embeddings = None           # 임베딩 모델 (텍스트 -> 벡터 변환)
vectorstore = None          # 서버용 Chroma VectorStore (실험 시에는 configurable["vectorstore"]로 교체)
prompt_template = None      # ChatPromptTemplate 객체 (RAG 전용)
prompt_template_general = None  # ChatPromptTemplate 객체 (일반 대화용)
rag_workflow = None         # 컴파일된 LangGraph RAG 에이전트 애플리케이션
_bm25_cache: dict[int, BM25Retriever] = {}   # VectorStore별 BM25 인덱스 캐시 (문서 변경 시 무효화)

print("Loaded API Key:", OPENAI_API_KEY[:5] + "..." if OPENAI_API_KEY else "None")
print("Loaded Base URL:", BASE_URL)

def resolve_rag_config(config: RunnableConfig | None = None) -> dict:
    """그래프 실행 인자의 configurable 값을 기본 설정 위에 덮어써 이번 실행의 설정을 만듭니다."""
    configurable = (config or {}).get("configurable", {})
    rag_config = {key: configurable.get(key, default) for key, default in DEFAULT_RAG_CONFIG.items()}
    rag_config["vectorstore"] = configurable.get("vectorstore") or vectorstore
    return rag_config

# ==========================================
# [LangGraph State 정의]
# ==========================================
class RAGState(TypedDict, total=False):
    question: str                   # 사용자 질문
    use_rag: bool                   # RAG 사용 여부 제어 플래그 (사용자 요청)
    answer: str                     # LLM이 생성한 최종 답변
    source: str                     # 정밀 답변 출처 태그
    rag_used: bool                  # 실제로 검색 문서를 근거로 답변했는지 (관련 문서가 없으면 False)
    retrieved_contexts: list[str]   # Threshold를 통과해 LLM에 전달된 청크 목록 (프롬프트 Context, RAGAS 평가에 공통 사용)
    retrieval_scores: list[float]   # 각 청크의 검색 점수 (vector: 코사인 유사도, hybrid: RRF 점수)
    retrieval_config: dict          # 이 답변에 실제 적용된 검색 설정(Top-K / Threshold / 검색 방식)과
                                    # 검색된 청크의 청킹 설정(분할 방식 / Chunk Size / Overlap, 청크 메타데이터에서 읽음)

# ==========================================
# [검색기] Vector / BM25 / Hybrid
# ==========================================
def tokenize_ko(text: str) -> list[str]:
    """BM25용 한국어 토크나이저: 단어 + 글자 bigram (조사가 붙은 '죽녹원은'도 '죽녹원'과 매칭되도록)"""
    words = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    tokens = []
    for word in words:
        tokens.append(word)
        tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens

def get_bm25_retriever(store: Chroma) -> BM25Retriever | None:
    """VectorStore에 저장된 문서로 BM25 인덱스를 만들고 VectorStore별로 캐싱합니다."""
    if id(store) not in _bm25_cache:
        stored = store.get(include=["documents", "metadatas"])
        if not stored["documents"]:
            return None
        docs = [Document(page_content=text, metadata=meta or {})
                for text, meta in zip(stored["documents"], stored["metadatas"])]
        _bm25_cache[id(store)] = BM25Retriever.from_documents(docs, preprocess_func=tokenize_ko)
    return _bm25_cache[id(store)]

def vector_search(store: Chroma, question: str, k: int, threshold: float) -> list[tuple[Document, float]]:
    """의미 기반 유사도 검색 + Threshold 필터링"""
    candidates = store.similarity_search_with_relevance_scores(question, k=k)
    return [(doc, score) for doc, score in candidates if score >= threshold]

def hybrid_search(store: Chroma, question: str, k: int, threshold: float) -> list[tuple[Document, float]]:
    """Vector 검색과 BM25 검색 결과를 RRF(Reciprocal Rank Fusion)로 합칩니다.
    관광지명 같은 고유명사는 BM25가, '바다 근처 가족 여행지' 같은 서술형 질문은 Vector가 강하므로 둘을 결합합니다.
    BM25는 관련 없는 질문에도 항상 결과를 반환하므로, Threshold를 통과한 Vector 결과가 없으면 '관련 문서 없음'으로 판단합니다."""
    vector_hits = [doc for doc, _ in vector_search(store, question, k * 2, threshold)]
    if not vector_hits:
        return []

    bm25 = get_bm25_retriever(store)
    bm25_hits = []
    if bm25 is not None:
        bm25.k = k * 2
        bm25_hits = bm25.invoke(question)

    fused: dict[str, list] = {}  # page_content -> [doc, rrf_score]
    for hits in (vector_hits, bm25_hits):
        for rank, doc in enumerate(hits):
            entry = fused.setdefault(doc.page_content, [doc, 0.0])
            entry[1] += 1 / (RRF_K + rank + 1)

    ranked = sorted(fused.values(), key=lambda entry: entry[1], reverse=True)
    return [(doc, score) for doc, score in ranked[:k]]

# ==========================================
# [LangGraph Node 정의]
# ==========================================
def retrieve_node(state: RAGState, config: RunnableConfig) -> RAGState:
    """[Retrieve Node]: 실행 설정(config)의 Top-K / Threshold / 검색 방식으로 관련 문서를 검색합니다."""
    rag_config = resolve_rag_config(config)
    search = hybrid_search if rag_config["retriever"] == "hybrid" else vector_search
    docs_with_scores = search(rag_config["vectorstore"], state["question"], rag_config["top_k"], rag_config["threshold"])

    docs = [doc for doc, _ in docs_with_scores]
    applied = {
        "retriever": rag_config["retriever"],
        "top_k": rag_config["top_k"],
        "threshold": rag_config["threshold"],
        # 청킹은 인덱싱 시점에 결정되므로, 현재 설정값이 아니라 검색된 청크에 기록된 값을 사용
        "chunking": [dict(c) for c in sorted({tuple(sorted(chunking_metadata(doc).items())) for doc in docs})],
    }

    # 터미널 디버그용 출력 (점수 정보 추가)
    print(f"\n--- [RAG 검색 결과] {applied} ---")
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"[{i+1}위] 점수: {score:.4f} | 출처: {doc.metadata.get('source', '알 수 없음')}")
        print(f"   내용: {doc.page_content.strip()[:80]}...")
    print("----------------------------\n")

    return {
        "source": ", ".join(sorted({doc.metadata.get("source", "알 수 없음") for doc in docs})),
        "rag_used": bool(docs),
        "retrieved_contexts": [doc.page_content for doc in docs],
        "retrieval_scores": [round(float(score), 4) for _, score in docs_with_scores],
        "retrieval_config": applied,
    }

def generate_rag_node(state: RAGState) -> RAGState:
    """[Generate RAG Node]: 검색된 청크를 참고문서로 묶어 근거 기반 답변을 생성합니다."""
    context = "\n\n".join(state["retrieved_contexts"])
    response = (prompt_template | llm).invoke({"context": context, "question": state["question"]})
    return {"answer": response.content}

def generate_general_node(state: RAGState) -> RAGState:
    """[Generate General Node]: RAG를 끄거나 관련 문서가 없을 때 일반 대화 프롬프트로 답변합니다."""
    response = (prompt_template_general | llm).invoke({"question": state["question"]})
    return {"answer": response.content, "rag_used": False, "source": ""}

# ==========================================
# [LangGraph RAG 워크플로우 빌드]
# ==========================================
def route_by_rag_flag(state: RAGState) -> str:
    """[분기 1] 사용자가 RAG를 켰는지에 따라 검색 여부를 결정"""
    return "retrieve" if state.get("use_rag", True) else "generate_general"

def route_by_retrieval(state: RAGState) -> str:
    """[분기 2] Threshold를 통과한 문서가 있으면 RAG 답변, 없으면 일반 생성형 AI로 Fallback"""
    return "generate_rag" if state.get("rag_used") else "generate_general"

def build_rag_graph():
    """
    START ─┬─ use_rag=True  → retrieve ─┬─ 문서 있음 → generate_rag ─────→ END
           │                            └─ 문서 없음 → generate_general ─→ END
           └─ use_rag=False ───────────────────────→ generate_general ─→ END
    """
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_rag", generate_rag_node)
    workflow.add_node("generate_general", generate_general_node)

    workflow.add_conditional_edges(
        START,
        route_by_rag_flag,
        {"retrieve": "retrieve", "generate_general": "generate_general"}
    )
    workflow.add_conditional_edges(
        "retrieve",
        route_by_retrieval,
        {"generate_rag": "generate_rag", "generate_general": "generate_general"}
    )
    workflow.add_edge("generate_rag", END)
    workflow.add_edge("generate_general", END)

    return workflow.compile()

# ==========================================
# [헬퍼 함수] 문서 추출, 청킹, DB 적재
# ==========================================
def extract_documents(file_path: Path) -> list[Document]:
    """파일에서 Document 객체 리스트를 추출합니다. (PyMuPDFLoader 또는 TextLoader 사용)"""
    try:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            loader = PyMuPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()
        # 모든 문서의 metadata에 source 정보로 파일명 저장
        for doc in docs:
            doc.metadata["source"] = file_path.name
        return docs
    except Exception as e:
        print(f"문서 추출 오류: {e}")
    return []

CHUNKING_KEYS = ("split_mode", "chunk_size", "chunk_overlap")

def chunking_metadata(doc: Document) -> dict:
    """청크 메타데이터에 기록된 청킹 설정을 꺼냅니다."""
    return {key: doc.metadata[key] for key in CHUNKING_KEYS if key in doc.metadata}

def chunk_documents(documents: list[Document], chunk_size: int | None = None, overlap: int | None = None,
                    md_split_mode: str | None = None) -> list[Document]:
    """마크다운 파일은 헤더 기준으로 통째 청킹하고(md_split_mode=header), 그 외 문서는 RecursiveSplitter 사용
    인자를 생략하면 DEFAULT_RAG_CONFIG(.env) 값을 사용합니다."""
    if not documents:
        return []

    chunk_size = chunk_size or DEFAULT_RAG_CONFIG["chunk_size"]
    overlap = DEFAULT_RAG_CONFIG["overlap"] if overlap is None else overlap
    md_split_mode = md_split_mode or DEFAULT_RAG_CONFIG["md_split_mode"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
        length_function=len
    )
    split_docs = []

    for doc in documents:
        source_name = doc.metadata.get("source", "")

        # 1. 마크다운(.md) 파일인 경우: 헤더 기준으로 1개 항목 = 1개 청크 분할
        if source_name.endswith(".md") and md_split_mode == "header":
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),   # 관광지 1개 레코드 단위
                ("###", "Header 3"),
            ]

            # strip_headers=False를 설정해야 청크 본문에 "## 관광지명"이 그대로 남습니다.
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )

            for chunk in md_splitter.split_text(doc.page_content):
                # 기존 문서의 metadata(filename 등)를 분할된 청크에도 복사/병합합니다.
                new_metadata = doc.metadata.copy()
                new_metadata.update(chunk.metadata)  # 헤더 정보도 메타데이터에 담김
                new_metadata["split_mode"] = "header"   # 헤더 분할은 Chunk Size / Overlap을 사용하지 않음
                split_docs.append(Document(page_content=chunk.page_content, metadata=new_metadata))

        # 2. PDF나 일반 TXT 파일(또는 recursive 모드의 .md)인 경우: RecursiveCharacterTextSplitter 사용
        else:
            for chunk in text_splitter.split_documents([doc]):
                chunk.metadata.update({"split_mode": "recursive", "chunk_size": chunk_size, "chunk_overlap": overlap})
                split_docs.append(chunk)

    return split_docs

def add_to_db(documents: list[Document], store: Chroma | None = None) -> int:
    """분할된 Document 객체 리스트를 Chroma DB에 추가합니다. (store 생략 시 서버용 vectorstore)"""
    store = store or vectorstore
    if not documents:
        return 0

    store.add_documents(documents)
    _bm25_cache.pop(id(store), None)  # 문서가 바뀌었으므로 BM25 인덱스 재생성
    return len(documents)

def create_vectorstore(collection_name: str = COLLECTION_NAME, persist_directory: str | None = CHROMA_DB_PATH) -> Chroma:
    """Chroma VectorStore 객체를 생성합니다. (persist_directory=None이면 메모리에만 생성)"""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        # 기본값(L2)은 relevance score가 음수가 될 수 있어 Threshold 해석이 어려움 → 코사인 유사도로 통일
        collection_metadata={"hnsw:space": "cosine"}
    )

def init_vectorstore():
    """서버용 VectorStore를 생성/연결합니다."""
    global vectorstore
    vectorstore = create_vectorstore()
    return vectorstore

def init_prompt_templates():
    """RAG용 및 일반 대화용 프롬프트 템플릿을 생성합니다 (Few-Shot 및 친절한 어조 적용)."""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 여행 정보와 교육 규정 등 다양한 유용한 정보를 친절하고 다정하게 안내해 주는 AI 어시스턴트입니다.\n"
            "아래의 [참고 문서] 내용을 바탕으로 사용자의 질문에 답변하세요.\n"
            "로봇처럼 딱딱하지 않고, 친구에게 이야기하듯 친근하고 부드러운 어조(~해요, ~랍니다 등)로 답변해 주세요.\n"
            "[참고문서]에 없는 내용은 '앗, 그 내용은 참고 문서에 나와 있지 않네요! 다른 궁금한 점이 있으신가요?'라고 부드럽게 답하세요.\n\n"
            "[참고문서]\n\n"
            "{context}"
        )),
        # Few-Shot 예시 추가 (질문과 다정한 답변 패턴 학습)
        ("user", "아트빌리지 어때?"),
        ("assistant", "김포아트빌리지는 모담산 자락에 안겨 있는 아주 아늑하고 자연친화적인 문화·예술 공간이에요! 전통한옥숙박체험관부터 예쁜 공방과 야외공연장까지 있어서 여유롭게 산책하며 둘러보기 참 좋답니다. 😊"),
        ("user", "주차할 수 있어?"),
        ("assistant", "네, 아트빌리지에는 총 142대 정도 주차할 수 있는 공간이 마련되어 있어서 차를 가지고 방문하시기에도 편해요!"),

        ("user", "{question}")
    ])

    general_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 언제나 밝고 친근하게 대화를 나누는 AI 어시스턴트입니다.\n"
            "로봇 같은 말투는 지우고, 일상 대화하듯 따뜻하고 다정하게 답변해주세요."
        )),
        ("user", "안녕? 반가워"),
        ("assistant", "안녕하세요! 오늘 어떤 이야기를 나누고 싶으신가요? 무엇이든 편하게 물어보세요! 😊"),
        ("user", "{question}")
    ])

    return rag_prompt, general_prompt

# ==========================================
# [FastAPI 수명주기 초기화]
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings, llm, vectorstore, prompt_template, prompt_template_general, rag_workflow
    print("1. AI 모델 및 클라이언트 초기화 중...")

    # LangChain ChatOpenAI 초기화 (base_url 지원)
    llm = ChatOpenAI(
        model=GPT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL
    )
    # embedding 모델 초기화
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL
    )

    # ChromaDB 저장소 초기화 및 연결
    init_vectorstore()

    # 프롬프트 템플릿 초기화
    prompt_template, prompt_template_general = init_prompt_templates()

    # LangGraph 워크플로우 구성 및 컴파일
    print("2. LangGraph RAG 워크플로우 빌드 중...")
    rag_workflow = build_rag_graph()
    print(f"준비 완료! 기본 RAG 설정: {DEFAULT_RAG_CONFIG}")
    yield
    print("서버가 종료됩니다.")
    # [종료 시 클린업] Chroma SQLite 연결을 닫고 물리 폴더 삭제
    if vectorstore is not None:
        if hasattr(vectorstore, "_client"):
            try:
                vectorstore._client.close()
            except Exception:
                pass
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"'{CHROMA_DB_PATH}' 폴더가 자동 정리 및 삭제되었습니다.")

def setup_cors(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(127\.0\.0\.1|localhost)(:\d+)?", # 127.0.0.1 모든 포트 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app = FastAPI(lifespan=lifespan)
setup_cors(app)

# API 요청용 스키마
class ChatReq(BaseModel):
    message: str
    use_rag: bool = False     # 기본값은 일반 대화 (클라이언트의 RAG 체크박스로 활성화)

# ==========================================
# [CORS 테스트 API] Simple vs Preflight 실습
# ==========================================
# 1. Simple Request - form 파라미터 (CORS 프리플라이트 대상 아님)
@app.post("/simpleparam")
def simple_param(message: str = Form(...)):
    print("message: ", message)
    return {
        "type": "simple_request",
        "message": f"받은 메시지: {message}",
        "preflight": "불필요 (application/x-www-form-urlencoded)"
    }

# 2. Preflight 필요 - JSON (CORS 프리플라이트 강제 대상)
@app.post("/simplejson")
def simple_json(req: ChatReq):
    print("message: ", req)
    return {
        "type": "preflight_request",
        "message": f"받은 메시지: {req.message}",
        "preflight": "필요 (application/json)"
    }

# ==========================================
# 통합 채팅 (LangGraph RAG 워크플로우 호출)
# ==========================================
@app.post("/chat")
def chat(req: ChatReq):
    final_state = rag_workflow.invoke({"question": req.message, "use_rag": req.use_rag})
    rag_used = final_state.get("rag_used", False)

    return {
        "answer": final_state["answer"],
        "source": final_state.get("source") if rag_used else None,
        "rag_used": rag_used,                                   # 실제 문서 사용 여부 (UI 표시용)
        "retrieval_scores": final_state.get("retrieval_scores", []),
        "retrieval_config": final_state.get("retrieval_config"),   # RAG를 끈 경우 None
    }

# ==========================================
# [API 2] 파일 업로드 (동적 문서 추가)
# ==========================================
@app.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    split_mode: Optional[str] = Form(None),   # (선택) header | recursive, 생략 시 .env 기본값
    chunk_size: Optional[int] = Form(None),   # (선택) recursive 분할 시 청크 크기
    overlap: Optional[int] = Form(None),      # (선택) recursive 분할 시 겹침
):
    """문서를 업로드해 청킹·임베딩 후 저장합니다. 실험에서 찾은 청킹 설정을 문서별로 지정할 수 있습니다.
    예) SSAFY 가이드에 실험 최고 설정 적용: -F split_mode=recursive -F chunk_size=700 -F overlap=100"""
    allowed_extensions = {".txt", ".md", ".pdf"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return {
            "success": False,
            "message": f"지원하지 않는 파일 형식입니다. ({', '.join(allowed_extensions)} 만 허용)"
        }
    if split_mode not in (None, "header", "recursive"):
        return {"success": False, "message": "split_mode는 header 또는 recursive만 가능합니다."}
    try:
        # 파일을 data 폴더에 백업 저장
        save_path = DATA_DIR / file.filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 백업된 파일 경로를 전달하여 Document 객체 리스트 추출
        docs = extract_documents(file_path=save_path)
        if not docs:
           return {"success": False, "message": "파일 내용이 없거나 읽을 수 없습니다."}

        # Document 청킹 및 DB 벡터 적재
        chunk_docs = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap, md_split_mode=split_mode)
        chunks_added = add_to_db(chunk_docs)
        return {
            "success": True,
            "message": f"'{file.filename}' 업로드 완료! ({chunks_added}개 청크 추가)",
            "chunks_added": chunks_added,
            "chunking": chunking_metadata(chunk_docs[0]) if chunk_docs else None,
        }
    except Exception as e:
        return {"success": False, "message": f"파일 처리 중 오류 발생: {str(e)}"}

# ==========================================
# [API 3] DB 초기화 (Chroma DB 비우기)
# ==========================================
@app.post("/reset-db")
def reset_db():
    try:
        if vectorstore is not None:
            # 컬렉션을 지우고 새로 만드는 대신, 기존 컬렉션 안의 모든 문서 ID를 삭제
            all_ids = vectorstore.get()["ids"]
            if all_ids:
                vectorstore.delete(ids=all_ids)
            _bm25_cache.pop(id(vectorstore), None)

        return {"success": True, "message": "Chroma DB 데이터가 완전히 초기화되었습니다."}
    except Exception as e:
        return {"success": False, "message": f"DB 초기화 중 오류 발생: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
