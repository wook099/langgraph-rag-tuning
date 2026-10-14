"""
RAGAS 기반 RAG 성능 평가 및 튜닝 실험 (LangGraph 평가 파이프라인)

    START → select_experiment → index → generate → ragas → record ─┬─ 남은 실험 있음 → select_experiment
                                    (main.rag_workflow 호출)        └─ 모든 실험 완료 → compare → END

    실험 설정(Top-K, Threshold, 검색 방식, 인덱스)은 전역 변수를 바꾸지 않고
    main.rag_workflow.invoke(state, config={"configurable": {...}}) 로 실행마다 주입한다.

사용법 (servers/ 폴더에서 실행):
    python eval_ragas.py --dataset ssafy --all           # SSAFY 규정 문서 실험 전체
    python eval_ragas.py --dataset trip --all            # 관광지 데이터 실험 전체
    python eval_ragas.py --dataset trip --exp topk_3     # 지정한 실험만 (baseline은 항상 함께 실행)
    python eval_ragas.py --dataset trip --all --limit 3  # 테스트 케이스 3개로 빠르게 동작 확인
    python eval_ragas.py --dataset ssafy --list          # 실험 목록 출력
    python eval_ragas.py --graph                         # RAG / 평가 그래프 구조(Mermaid) 출력

결과:
    eval_results/experiments_<데이터셋>.csv        실험별 설정값 + 평균 지표 (실행할 때마다 누적)
    eval_results/experiments_<데이터셋>.md         위 CSV를 비교표로 렌더링 (Overall 최고 실험 표시)
    eval_results/detail_<데이터셋>_<실험>.csv       질문별 답변, 검색 청크, 검색 점수, 지표
"""
from __future__ import annotations

import os
import argparse
import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from dotenv import load_dotenv

# gpt-5-mini의 temperature 비지원 문제 우회를 위한 openai 패치
import openai
orig_create = openai.resources.chat.completions.Completions.create
def safe_create(self, *args, **kwargs):
    if "temperature" in kwargs:
        del kwargs["temperature"]
    return orig_create(self, *args, **kwargs)
openai.resources.chat.completions.Completions.create = safe_create

orig_acreate = openai.resources.chat.completions.AsyncCompletions.create
async def safe_acreate(self, *args, **kwargs):
    if "temperature" in kwargs:
        del kwargs["temperature"]
    return await orig_acreate(self, *args, **kwargs)
openai.resources.chat.completions.AsyncCompletions.create = safe_acreate

import main
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END

load_dotenv()

RESULTS_DIR = Path("./eval_results")

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
UNANSWERABLE_TYPE = "문서에 없음"           # 이 유형은 RAGAS 대신 '근거 없는 답변 회피' 여부로 평가
REFUSAL_PHRASE = "참고 문서에 나와 있지 않"  # RAG 프롬프트에 지정한 거절 문구

# ==========================================
# [실험 설정] 데이터셋별 baseline과 실험 정의
# ==========================================
# 1) SSAFY 규정 문서: 규정과 표가 연속된 문서 → 고정 길이(recursive) 분할 기준으로 Chunk/Overlap/Top-K/Threshold 비교
SSAFY_EXPERIMENTS = {
    "baseline":        {"hypothesis": "기준 설정 (Chunk 500 / Overlap 50 / Top-5 / Threshold 0)"},
    "small_chunk":     {"chunk_size": 300, "overlap": 50,
                        "hypothesis": "작은 청크는 내용이 구체적이라 Precision은 오르지만 규정이 여러 청크로 나뉘어 Recall이 떨어진다"},
    "large_chunk":     {"chunk_size": 700, "overlap": 100,
                        "hypothesis": "큰 청크와 넓은 겹침은 규정과 예외사항을 한 청크에 담아 Recall이 오른다"},
    "retrieval_tuned": {"top_k": 3, "threshold": 0.15,
                        "hypothesis": "검색 결과를 적게, 유사도 높은 것만 전달하면 노이즈가 줄어 Precision이 오른다"},
}

# 2) 관광지 데이터: SSAFY 실험에서 두 변수를 동시에 바꾼 실험(large_chunk, retrieval_tuned)은 효과를 분리할 수 없었으므로
#    가설 하나당 변수 하나만 바꾸도록 재설계
TRIP_EXPERIMENTS = {
    "baseline":          {"hypothesis": "기준 설정 (헤더 분할 + Vector Top-5)"},
    # Chunk Size: 고정 길이 분할(overlap 50 고정)에서 크기에 따라 어떻게 달라지는가?
    "recursive_300_50":  {"md_split_mode": "recursive", "chunk_size": 300, "overlap": 50,
                          "hypothesis": "작은 청크는 한 관광지가 여러 조각으로 나뉘어 Recall이 떨어진다"},
    "recursive_500_50":  {"md_split_mode": "recursive", "chunk_size": 500, "overlap": 50,
                          "hypothesis": "고정 길이 분할은 관광지 경계를 무시해 헤더 분할보다 Precision이 떨어진다"},
    "recursive_800_50":  {"md_split_mode": "recursive", "chunk_size": 800, "overlap": 50,
                          "hypothesis": "큰 청크는 Recall은 회복되지만 여러 관광지가 섞여 Precision이 떨어진다"},
    # Overlap: 청크 크기(500) 고정, 겹침만 변경
    "recursive_500_0":   {"md_split_mode": "recursive", "chunk_size": 500, "overlap": 0,
                          "hypothesis": "겹침이 없으면 경계에서 잘린 정보가 사라져 Recall이 떨어진다"},
    "recursive_500_100": {"md_split_mode": "recursive", "chunk_size": 500, "overlap": 100,
                          "hypothesis": "겹침을 늘리면 경계 정보가 보존되어 Recall이 오르지만 중복 청크가 늘어난다"},
    # Top-K: 많이 가져올수록 Recall↑ Precision↓ 인가?
    "topk_3":            {"top_k": 3, "hypothesis": "K를 줄이면 Precision은 오르고 Recall은 떨어진다"},
    "topk_8":            {"top_k": 8, "hypothesis": "K를 늘리면 Recall은 오르고 Precision은 떨어진다"},
    # Similarity Threshold: 관련 낮은 청크를 걸러내면 Faithfulness가 오르는가?
    "threshold_0.2":     {"threshold": 0.2, "hypothesis": "약한 필터로 노이즈 청크만 제거되어 Precision이 오른다"},
    "threshold_0.3":     {"threshold": 0.3, "hypothesis": "낮은 유사도 청크 제거로 Precision/Faithfulness가 오른다"},
    "threshold_0.4":     {"threshold": 0.4, "hypothesis": "필터가 너무 강하면 필요한 청크까지 빠져 Recall이 떨어지고 Fallback이 늘어난다"},
    # 검색 방식: 고유명사(관광지명) 질문에 키워드 검색이 도움이 되는가?
    "hybrid":            {"retriever": "hybrid", "hypothesis": "BM25 결합으로 관광지명 질문의 Recall이 오른다"},
}

DATASETS = {
    "ssafy": {
        "data_file": Path("./data/SSAFY_GUIDE.md"),
        "test_cases": Path("./test_cases_ssafy.json"),
        "baseline": {"md_split_mode": "recursive", "chunk_size": 500, "overlap": 50,
                     "retriever": "vector", "top_k": 5, "threshold": 0.0},
        "experiments": SSAFY_EXPERIMENTS,
    },
    "trip": {
        "data_file": Path("./data/전국관광지정보_RAG용.md"),
        "test_cases": Path("./test_cases_trip.json"),
        "baseline": {"md_split_mode": "header", "chunk_size": 500, "overlap": 50,   # 관광지 1개 = 청크 1개
                     "retriever": "vector", "top_k": 5, "threshold": 0.0},
        "experiments": TRIP_EXPERIMENTS,
    },
}

def get_config(dataset: str, name: str) -> dict:
    ds = DATASETS[dataset]
    return {**ds["baseline"], **ds["experiments"][name]}

# ==========================================
# [초기화]
# ==========================================
def init_models():
    """서버(lifespan)와 같은 모델·프롬프트·그래프를 준비합니다. 실험 설정은 실행할 때 RunnableConfig로 주입합니다."""
    main.llm = ChatOpenAI(model=main.GPT_MODEL, api_key=main.OPENAI_API_KEY, base_url=main.BASE_URL)
    main.embeddings = OpenAIEmbeddings(model=main.EMBEDDING_MODEL_NAME, api_key=main.OPENAI_API_KEY, base_url=main.BASE_URL)
    main.prompt_template, main.prompt_template_general = main.init_prompt_templates()
    main.rag_workflow = main.build_rag_graph()

def to_runnable_config(cfg: dict, store) -> dict:
    """실험 설정을 RAG 그래프 실행 인자(RunnableConfig)로 변환합니다. 전역 설정을 바꾸지 않고 실행마다 주입합니다."""
    configurable = {key: cfg[key] for key in main.DEFAULT_RAG_CONFIG}
    configurable["vectorstore"] = store
    return {"configurable": configurable}

_index_cache: dict[tuple, object] = {}   # (데이터셋, 분할 방식, Chunk, Overlap) -> 인메모리 Chroma

def index_key(dataset: str, cfg: dict) -> tuple:
    return (dataset, cfg["md_split_mode"], cfg["chunk_size"], cfg["overlap"])

def build_index(dataset: str, cfg: dict) -> int:
    """청킹 설정별로 독립된 인메모리 Chroma 컬렉션을 만들어 실험 간 데이터가 섞이지 않게 합니다.
    청킹 설정이 같은 실험(Top-K, Threshold, Retriever만 다른 경우)은 같은 인덱스를 재사용합니다."""
    key = index_key(dataset, cfg)
    if key in _index_cache:
        return _index_cache[key]._collection.count()

    data_file = DATASETS[dataset]["data_file"]
    if not data_file.exists():
        raise FileNotFoundError(f"평가용 데이터 파일이 없습니다: {data_file}")

    store = main.create_vectorstore(collection_name="eval_{}_{}_{}_{}".format(*key), persist_directory=None)
    chunks = main.chunk_documents(main.extract_documents(data_file), chunk_size=cfg["chunk_size"],
                                  overlap=cfg["overlap"], md_split_mode=cfg["md_split_mode"])
    main.add_to_db(chunks, store)
    _index_cache[key] = store
    print(f"-> 인덱스 생성: {key} / {len(chunks)}개 청크")
    return len(chunks)

def load_test_cases(dataset: str, limit: int | None) -> list[dict]:
    with open(DATASETS[dataset]["test_cases"], "r", encoding="utf-8") as f:
        cases = json.load(f)
    return cases[:limit] if limit else cases

# ==========================================
# [평가 단계별 함수] LangGraph 노드에서 호출
# ==========================================
def collect_answers(test_cases: list[dict], run_config: dict) -> list[dict]:
    """RAG 워크플로우(main.rag_workflow)를 실험 설정(run_config)으로 질문마다 실행하고, State에 기록된 검색 결과를 수집합니다."""
    rows = []
    for case in test_cases:
        state = main.rag_workflow.invoke({"question": case["question"], "use_rag": True}, config=run_config)
        rows.append({
            "type": case.get("type", ""),
            "user_input": case["question"],
            "reference": case["ground_truth"],
            "response": state["answer"],
            "rag_used": state.get("rag_used", False),
            "retrieved_contexts": state.get("retrieved_contexts", []),
            "retrieval_scores": state.get("retrieval_scores", []),
            "retrieval_config": state.get("retrieval_config"),   # LangGraph State에 기록된 실제 적용 설정
        })
    return rows

def run_ragas(rows: list[dict]) -> list[dict]:
    """RAGAS 4개 지표로 채점합니다. 청크를 합치지 않고 리스트 그대로 넘겨야 Context Precision이 순위별로 계산됩니다."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall, ContextPrecision

    eval_dataset = EvaluationDataset.from_list([{
        "user_input": r["user_input"],
        "response": r["response"],
        "retrieved_contexts": r["retrieved_contexts"] or ["(검색된 문서 없음)"],
        "reference": r["reference"],
    } for r in rows])

    results = evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), LLMContextRecall(), ContextPrecision()],
        llm=main.llm,
        embeddings=main.embeddings,
    )
    return results.to_pandas()[METRICS].to_dict("records")

# ==========================================
# [LangGraph 평가 파이프라인]
# ==========================================
class EvalState(TypedDict, total=False):
    dataset: str                     # ssafy | trip
    test_cases: list[dict]           # 평가 질문/정답
    queue: list[str]                 # 아직 실행하지 않은 실험 이름
    experiment: str                  # 현재 실험 이름
    config: dict                     # 현재 실험의 Chunk Size / Overlap / Top-K / Threshold / 검색 방식
    n_chunks: int                    # 현재 설정으로 만든 청크 수
    rows: list[dict]                 # 질문별 답변 + 검색 결과
    scores: list[dict]               # 질문별 RAGAS 점수 (답이 문서에 있는 질문만)
    summaries: Annotated[list[dict], operator.add]   # 실험별 요약 (실험이 끝날 때마다 누적)
    best: dict                       # Overall 최고 실험

def select_experiment_node(state: EvalState) -> EvalState:
    """[1] 큐에서 다음 실험을 꺼내 설정을 결정합니다. (전역 설정은 건드리지 않음)"""
    name, *rest = state["queue"]
    cfg = get_config(state["dataset"], name)
    print(f"\n==================== [{state['dataset']} / {name}] {cfg['hypothesis']} ====================")
    return {"experiment": name, "config": cfg, "queue": rest}

def index_node(state: EvalState) -> EvalState:
    """[2] 현재 실험의 Chunk Size / Overlap으로 독립 인덱스를 준비합니다."""
    return {"n_chunks": build_index(state["dataset"], state["config"])}

def generate_node(state: EvalState) -> EvalState:
    """[3] 현재 실험의 Top-K / Threshold / 검색 방식과 인덱스를 RunnableConfig로 주입해 RAG 그래프로 답변을 생성합니다."""
    store = _index_cache[index_key(state["dataset"], state["config"])]
    return {"rows": collect_answers(state["test_cases"], to_runnable_config(state["config"], store))}

def ragas_node(state: EvalState) -> EvalState:
    """[4] 답이 문서에 있는 질문을 RAGAS로 채점합니다. ('문서에 없음' 유형은 record 단계에서 따로 평가)"""
    answerable = [r for r in state["rows"] if r["type"] != UNANSWERABLE_TYPE]
    return {"scores": run_ragas(answerable)}

def record_node(state: EvalState) -> EvalState:
    """[5] 질문별 상세 결과와 실험 요약을 저장합니다. (실험마다 저장하여 중간에 실패해도 앞선 결과 유지)"""
    dataset, name, cfg, rows = state["dataset"], state["experiment"], state["config"], state["rows"]
    answerable = [r for r in rows if r["type"] != UNANSWERABLE_TYPE]
    unanswerable = [r for r in rows if r["type"] == UNANSWERABLE_TYPE]
    df_scores = pd.DataFrame(state["scores"], columns=METRICS)

    # 근거 없는 답변 회피율: 문서에 없는 질문에 지어내지 않고 거절했거나 일반 AI 응답임을 표시했는가
    handled = [(not r["rag_used"]) or (REFUSAL_PHRASE in r["response"]) for r in unanswerable]

    detail = pd.DataFrame(rows)
    for metric in METRICS:
        detail[metric] = None
        detail.loc[detail["type"] != UNANSWERABLE_TYPE, metric] = df_scores[metric].values
    RESULTS_DIR.mkdir(exist_ok=True)
    detail.to_csv(RESULTS_DIR / f"detail_{dataset}_{name}.csv", index=False, encoding="utf-8-sig")

    averages = {metric: round(float(df_scores[metric].mean()), 4) for metric in METRICS}
    summary = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset,
        "experiment": name,
        "hypothesis": cfg["hypothesis"],
        "md_split_mode": cfg["md_split_mode"],
        "chunk_size": cfg["chunk_size"] if cfg["md_split_mode"] == "recursive" else "-",
        "overlap": cfg["overlap"] if cfg["md_split_mode"] == "recursive" else "-",
        "retriever": cfg["retriever"],
        "top_k": cfg["top_k"],
        "threshold": cfg["threshold"],
        "n_chunks": state["n_chunks"],
        "n_cases": len(answerable),
        **averages,
        "overall": round(sum(averages.values()) / len(averages), 4),
        "rag_used_rate": round(sum(r["rag_used"] for r in answerable) / max(len(answerable), 1), 4),
        "unanswerable_handled": f"{sum(handled)}/{len(handled)}" if handled else "-",
        "judge_model": main.GPT_MODEL,
    }
    print(pd.Series(summary).to_string())
    save_summary(dataset, [summary])
    return {"summaries": [summary]}

def route_next(state: EvalState) -> str:
    """남은 실험이 있으면 다음 실험으로, 없으면 최종 비교로 이동"""
    return "select_experiment" if state["queue"] else "compare"

def compare_node(state: EvalState) -> EvalState:
    """[6] 이번 실행의 실험들을 Overall 기준으로 정렬하고 baseline 대비 개선율을 계산합니다."""
    ranked = sorted(state["summaries"], key=lambda r: r["overall"], reverse=True)
    baseline = next((r for r in state["summaries"] if r["experiment"] == "baseline"), None)
    print("\n==================== 실험 비교 (Overall 순) ====================")
    for r in ranked:
        diff = f"{(r['overall'] - baseline['overall']) / baseline['overall'] * 100:+.1f}%" if baseline else "-"
        print(f"{r['experiment']:20s} overall={r['overall']:.4f}  baseline 대비 {diff}")
    return {"best": ranked[0]}

def build_eval_graph():
    """
    START → select_experiment → index → generate → ragas → record ─┬─ 남은 실험 있음 → select_experiment
                                           (main.rag_workflow 호출)  └─ 모든 실험 완료 → compare → END
    """
    graph = StateGraph(EvalState)
    graph.add_node("select_experiment", select_experiment_node)
    graph.add_node("index", index_node)
    graph.add_node("generate", generate_node)
    graph.add_node("ragas", ragas_node)
    graph.add_node("record", record_node)
    graph.add_node("compare", compare_node)

    graph.add_edge(START, "select_experiment")
    graph.add_edge("select_experiment", "index")
    graph.add_edge("index", "generate")
    graph.add_edge("generate", "ragas")
    graph.add_edge("ragas", "record")
    graph.add_conditional_edges("record", route_next, {"select_experiment": "select_experiment", "compare": "compare"})
    graph.add_edge("compare", END)
    return graph.compile()

# ==========================================
# [결과 저장] CSV 누적 + 비교표 Markdown 생성
# ==========================================
def save_summary(dataset: str, summaries: list[dict]):
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_csv = RESULTS_DIR / f"experiments_{dataset}.csv"
    summary_md = RESULTS_DIR / f"experiments_{dataset}.md"
    df_new = pd.DataFrame(summaries)
    if summary_csv.exists():
        df_all = pd.concat([pd.read_csv(summary_csv), df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # 실험별 최신 결과만 비교표로 렌더링
    order = {name: i for i, name in enumerate(DATASETS[dataset]["experiments"])}
    latest = (df_all.drop_duplicates("experiment", keep="last")
              .sort_values("experiment", key=lambda col: col.map(order))
              .reset_index(drop=True))
    best = latest["overall"].idxmax()
    baseline_overall = latest.loc[latest["experiment"] == "baseline", "overall"]

    lines = [
        f"# RAG 튜닝 실험 결과 ({dataset})",
        "",
        f"`python eval_ragas.py --dataset {dataset} --all` 실행 결과에서 자동 생성된 파일입니다. (실험별 최신 실행 기준)",
        "",
        "| 실험 | 분할 | Chunk/Overlap | 검색 | Top-K | Threshold | Faithfulness | Answer Relevancy | Context Recall | Context Precision | **Overall** | Baseline 대비 | 문서 없음 질문 처리 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, r in latest.iterrows():
        diff = f"{(r['overall'] - baseline_overall.iloc[0]) / baseline_overall.iloc[0] * 100:+.1f}%" if len(baseline_overall) else "-"
        overall = f"**{r['overall']:.4f}** ⭐" if i == best else f"{r['overall']:.4f}"
        lines.append(
            f"| {r['experiment']} | {r['md_split_mode']} | {r['chunk_size']}/{r['overlap']} | {r['retriever']} | {r['top_k']} | {r['threshold']} "
            f"| {r['faithfulness']:.4f} | {r['answer_relevancy']:.4f} | {r['context_recall']:.4f} | {r['context_precision']:.4f} "
            f"| {overall} | {diff} | {r['unanswerable_handled']} |"
        )
    lines += ["", "## 실험 가설", ""]
    lines += [f"- **{r['experiment']}**: {r['hypothesis']}" for _, r in latest.iterrows()]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n-> '{summary_csv}' 누적 저장, '{summary_md}' 비교표 갱신 완료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS 기반 RAG 튜닝 실험")
    parser.add_argument("--dataset", choices=list(DATASETS), default="trip", help="평가할 문서와 테스트셋")
    parser.add_argument("--exp", nargs="+", default=["baseline"], help="실행할 실험 이름 (--list로 확인)")
    parser.add_argument("--all", action="store_true", help="해당 데이터셋의 모든 실험 실행")
    parser.add_argument("--limit", type=int, help="테스트 케이스 수 제한 (동작 확인용)")
    parser.add_argument("--list", action="store_true", help="실험 목록 출력")
    parser.add_argument("--graph", action="store_true", help="RAG / 평가 LangGraph 구조를 Mermaid로 출력")
    args = parser.parse_args()
    experiments = DATASETS[args.dataset]["experiments"]

    if args.list:
        for name in experiments:
            cfg = get_config(args.dataset, name)
            print(f"{name:20s} chunk={cfg['chunk_size']}/{cfg['overlap']} ({cfg['md_split_mode']}) "
                  f"top_k={cfg['top_k']} threshold={cfg['threshold']} {cfg['retriever']:6s} | {cfg['hypothesis']}")
        raise SystemExit

    if args.graph:
        print("%% RAG 워크플로우 (main.py)\n" + main.build_rag_graph().get_graph().draw_mermaid())
        print("%% 평가 파이프라인 (eval_ragas.py)\n" + build_eval_graph().get_graph().draw_mermaid())
        raise SystemExit

    unknown = [name for name in args.exp if name not in experiments]
    if unknown:
        parser.error(f"'{args.dataset}' 데이터셋에 없는 실험: {unknown} (사용 가능: {list(experiments)})")

    names = list(experiments) if args.all else args.exp
    if "baseline" not in names:
        names = ["baseline"] + names  # 비교 기준이 항상 같은 실행에 포함되도록

    init_models()
    test_cases = load_test_cases(args.dataset, args.limit)
    print(f"[{args.dataset}] 테스트 케이스 {len(test_cases)}개 / 실험 {names}")

    final = build_eval_graph().invoke(
        {"dataset": args.dataset, "test_cases": test_cases, "queue": names, "summaries": []},
        config={"recursion_limit": 6 * len(names) + 10},   # 실험 1개 = 노드 5개 순환
    )
    print(f"\n-> 최고 실험: {final['best']['experiment']} (Overall {final['best']['overall']:.4f})")
