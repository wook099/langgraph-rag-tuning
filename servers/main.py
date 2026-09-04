import os
import warnings
warnings.filterwarnings("ignore")

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


# transformers 및 huggingface 캐시 관련 경고 방지
if "HF_HOME" not in os.environ and "TRANSFORMERS_CACHE" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import TypedDict, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# LangGraph Core 임포트
from langgraph.graph import StateGraph, START, END

# .env 로드
load_dotenv()

# ==========================================
# [설정 영역]
# ==========================================
# ==========================================
# TODO: 01 .env에서 사용자 환경에 맞게 수정하세요. 
# ==========================================
OPENAI_API_KEY = os.getenv("API_KEY", "키를 입력하세요.")   # .env의 API_KEY 로드
BASE_URL = os.getenv("base_url")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500               # 텍스트를 분할할 글자 수 (청크 크기)
OVERLAP = 50                   # 청크 간 겹치게 할 글자 수 (문맥 보존용)
CHROMA_DB_PATH = "./chroma_data" # ChromaDB가 파일로 저장될 로컬 디렉토리 경로

# ==========================================
# [전역 변수] 서버 실행 중 메모리에 유지될 객체들
# ==========================================
llm = None                  # OpenAI 연결 클라이언트
embeddings = None           # 임베딩 모델 (텍스트 -> 벡터 변환)
vectorstore = None          # Chroma VectorStore 객체
prompt_template = None      # ChatPromptTemplate 객체 (RAG 전용)
prompt_template_general = None  # ChatPromptTemplate 객체 (일반 대화용)
rag_workflow = None         # 컴파일된 LangGraph RAG 에이전트 애플리케이션

print("Loaded API Key:", OPENAI_API_KEY[:5] + "..." if OPENAI_API_KEY else "None")
print("Loaded Base URL:", BASE_URL)
print("줴줴")
# ==========================================
# [LangGraph State 정의]
# ==========================================
class RAGState(TypedDict):
    question: str         # 사용자 질문
    context: str          # 유사도 검색을 통해 조립된 본문 컨텍스트
    answer: str           # LLM이 생성한 최종 답변
    source: str           # 정밀 답변 출처 태그
    use_rag: bool         # RAG 사용 여부 제어 플래그

# ==========================================
# [LangGraph Node 정의]
# ==========================================
def retrieve_node(state: RAGState) -> RAGState:
    """[Retrieve Node]: 질문(question)을 받아 관련 문서를 책장에서 검색한 뒤 컨텍스트(context)를 조립합니다."""
    question = state["question"]

    # 유사도 스코어(0.0~1.0)와 함께 검색 실행
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=5,
        score_threshold=0.00
    )

    # 터미널 디버그용 출력 (점수 정보 추가)
    print("\n--- [RAG 통합 검색 결과] ---")
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"[{i+1}위] 점수: {score:.4f} | 출처: {doc.metadata.get('source', '알 수 없음')}")
        print(f"   내용: {doc.page_content.strip()}...")
    print("----------------------------\n")

    # 튜플 리스트에서 문서 객체 리스트 분리
    docs = [doc for doc, _ in docs_with_scores]

    # 검색 결과를 문자열로 포맷팅
    context = "\n\n".join([doc.page_content for doc in docs])

    # 출처 태그 조립 (기존 API 규격 유지용)
    source_tag = ", ".join(sorted({doc.metadata.get("source", "알 수 없음") for doc in docs})) if docs else "ChatGPT 일반 지식"

    return {
        "context": context,
        "source": source_tag
    }

def generate_node(state: RAGState) -> RAGState:
    """[Generate Node]: 검색된 참고문서(context)를 기반으로 사용자 질문(question)에 답변을 생성합니다."""
    question = state["question"]
    context = state.get("context", "")
    use_rag = state.get("use_rag", True)

    # RAG 사용 유무와 컨텍스트 탑재 상태에 따라 적절한 프롬프트 체인을 분기하여 호출
    if use_rag :
        # RAG 모드 프롬프트 실행
        response = (prompt_template | llm).invoke({"context": context, "question": question})
    else:
        # 일반 대화 모드 프롬프트 실행 (Chroma 검색 우회 시)
        response = (prompt_template_general | llm).invoke({"question": question})

    return {
        "answer": response.content
    }

# ==========================================
# [LangGraph RAG 워크플로우 빌드]
# ==========================================
def route_by_rag_flag(state: RAGState) -> str:
    """use_rag 플래그에 따라 다음 실행할 노드를 결정하는 조건부 라우터 함수"""
    if state.get("use_rag", True):
        return "retrieve"
    return "generate"

def build_rag_graph():
    # TODO: 03. workflow를 구성하고 compile 해서 반환하시오.
    #  workflow 생성
    #  노드 추가
    #  조건부 엣지 정의 (START에서 use_rag 분기에 따라 retrieve 혹은 generate로 바로 이동)
    #  나머지 엣지 정의
    #  컴파일 후 반환
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve",retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_conditional_edges(
        START,
        route_by_rag_flag,
        {
            "retrieve":"retrieve",
            "generate":"generate"
        }
    )

    workflow.add_edge("retrieve","generate")
    workflow.add_edge("generate",END)

    return workflow.compile()
    return None

    # END

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

def chunk_documents(documents: list[Document], size=CHUNK_SIZE, overlap=OVERLAP) -> list[Document]:
    """마크다운 파일은 헤더(###) 기준으로 통채 청킹하고, 일반 문서만 RecursiveSplitter 사용"""
    if not documents:
        return []
    
    split_docs = []

    for doc in documents:
        source_name = doc.metadata.get("source", "")
        
        # 1. 마크다운(.md) 파일인 경우: 헤더 기준으로 1개 항목 = 1개 청크 분할
        if source_name.endswith(".md"):
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),  # 관광지 1개 레코드 단위
            ]
            
            # strip_headers=False를 설정해야 청크 본문에 "### 관광지명"이 그대로 남습니다.
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )
            
            # md_splitter.split_text는 Document 객체 리스트를 반환합니다.
            md_chunks = md_splitter.split_text(doc.page_content)
            
            for chunk in md_chunks:
                # 기존 문서의 metadata(filename 등)를 분할된 청크에도 복사/병합합니다.
                new_metadata = doc.metadata.copy()
                new_metadata.update(chunk.metadata)  # 헤더 정보도 메타데이터에 담김
                
                # 메타데이터가 보강된 새 Document 객체 생성 후 추가
                split_docs.append(Document(
                    page_content=chunk.page_content,
                    metadata=new_metadata
                ))
                
        # 2. PDF나 일반 TXT 파일인 경우: 기존 RecursiveCharacterTextSplitter 사용
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ".", " "],
                length_function=len
            )
            split_docs.extend(text_splitter.split_documents([doc]))

    return split_docs

# def chunk_documents(documents: list[Document], size=CHUNK_SIZE, overlap=OVERLAP) -> list[Document]:
#     """LangChain의 RecursiveCharacterTextSplitter를 사용하여 Document 리스트를 분할합니다."""
#     if not documents:
#         return []
#     # TODO: 04. RecursiveCharacterTextSplitter를 구성해보자.
#     text_splitter=RecursiveCharacterTextSplitter(
#         chunk_size=size,
#         chunk_overlap=overlap,
#         separators=["\n\n", "\n", ".", " "],
#         length_function=len
            
#     )
#     return text_splitter.split_documents(documents)

    END



def add_to_db(documents: list[Document]) -> int:
    """분할된 Document 객체 리스트를 Chroma DB에 추가합니다."""
    if not documents:
        return 0

    vectorstore.add_documents(documents)
    return len(documents)

def init_vectorstore():
    """Chroma VectorStore 객체를 생성/초기화합니다."""
    global vectorstore
    vectorstore = Chroma(
        collection_name="ssafy_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    return vectorstore

def init_prompt_templates():
    """RAG용 및 일반 대화용 프롬프트 템플릿을 생성합니다 (Few-Shot 및 친절한 어조 적용)."""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "당신은 전국 여행 및 다양한 유용한 정보를 친절하고 다정하게 안내해 주는 AI 어시스턴트입니다.\n"
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
    print("준비 완료! 서버가 시작되었습니다.")
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
        # TODO: 02. CORS 설정을 적용합니다. (127.0.0.1의 모든 포트 허용)

        # END
    )

app = FastAPI(lifespan=lifespan)
setup_cors(app)

# API 요청용 스키마
class ChatReq(BaseModel):
    message: str
    use_rag: bool = False     # 기본값으로 RAG 사용을 활성화

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
    initial_state = {
        "question": req.message,
        "use_rag": req.use_rag,
        "context": "",
        "source": ""
    }

    final_state = rag_workflow.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "source": final_state.get("source") if req.use_rag else None
    }

# ==========================================
# [API 2] 파일 업로드 (동적 문서 추가)
# ==========================================
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".txt", ".md", ".pdf"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return {
            "success": False,
            "message": f"지원하지 않는 파일 형식입니다. ({', '.join(allowed_extensions)} 만 허용)"
        }
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
        chunk_docs = chunk_documents(docs)

        # vdctor store에 데이터 저장
        chunks_added = add_to_db(chunk_docs)
        return {
            "success": True,
            "message": f"'{file.filename}' 업로드 완료! ({chunks_added}개 청크 추가)",
            "chunks_added": chunks_added
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

        return {"success": True, "message": "Chroma DB 데이터가 완전히 초기화되었습니다."}
    except Exception as e:
        return {"success": False, "message": f"DB 초기화 중 오류 발생: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
