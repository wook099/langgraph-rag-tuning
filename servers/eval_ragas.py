import os
from pathlib import Path
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


# 1. 기존 main.py 모듈 및 LangChain 구성요소 로드
import main
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# .env 파일 로드
load_dotenv()

print("1. RAG 에이전트 인프라 초기화 중...")
# main의 전역 설정값을 기반으로 LLM 및 임베딩 모델 수동 초기화
main.llm = ChatOpenAI(
    model=main.GPT_MODEL,
    api_key=main.OPENAI_API_KEY,
    base_url=main.BASE_URL
)
main.embeddings = OpenAIEmbeddings(
    model=main.EMBEDDING_MODEL_NAME,
    api_key=main.OPENAI_API_KEY,
    base_url=main.BASE_URL
)

# ChromaDB 및 템플릿 초기화 (Graph 컴파일은 몽키 패치 이후에 진행)
main.init_vectorstore()
main.prompt_template, main.prompt_template_general = main.init_prompt_templates()

# 평가 시작 전 가이드 문서(SSAFY_GUIDE.md)가 DB에 없다면 자동으로 적재
print("1-2. 평가를 위한 기초 문서 데이터(SSAFY_GUIDE.md) 적재 검사...")
try:
    # 컬렉션 내 문서가 비어있는지 유사도 임의 검색으로 확인
    sample_search = main.vectorstore.similarity_search("SSAFY", k=1)
    if not sample_search:
        print("-> DB가 비어있음을 감지했습니다. SSAFY_GUIDE.md 문서를 분할하여 적재합니다...")
        guide_path = Path("./data/SSAFY_GUIDE.md")
        if guide_path.exists():
            docs = main.extract_documents(guide_path)
            chunk_docs = main.chunk_documents(docs)
            main.add_to_db(chunk_docs)
            print(f"-> {len(chunk_docs)}개 청크 적재 완료.")
        else:
            print("🚨 경고: ./data/SSAFY_GUIDE.md 파일을 찾을 수 없습니다. 평가 결과가 정확하지 않을 수 있습니다.")
    else:
        print("-> DB에 기존 문서 데이터가 존재합니다. 기존 데이터를 사용하여 평가합니다.")
except Exception as e:
    print(f"문서 사전 적재 확인 중 예외 발생 (무시하고 계속 진행): {e}")

# 최종적으로 Graph를 빌드하여 컴파일
main.rag_workflow = main.build_rag_graph()
print("-> RAG 워크플로우 컴파일 완료.")



import json

# 2. Ragas 평가용 테스트 케이스 정의 (질문 및 모범 답안)
test_cases_file = Path("./test_cases.json")
test_cases = []

if test_cases_file.exists():
    print(f"-> 기존 생성된 테스트 케이스 파일({test_cases_file})을 로드합니다.")
    try:
        with open(test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except Exception as le:
        print(f"-> 테스트 케이스 파일 로드 중 오류 발생: {le}")

if not test_cases:
    print("-> 테스트 케이스 캐시가 없거나 로드에 실패했습니다. Ragas Testset Generator로 5개의 테스트 케이스를 자동 생성합니다...")
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.testset.synthesizers.generate import TestsetGenerator

        guide_path = Path("./data/SSAFY_GUIDE.md")
        if guide_path.exists():
            docs = main.extract_documents(guide_path)
            chunk_docs = main.chunk_documents(docs)

            # Ragas 래퍼 클래스로 LLM 및 Embeddings 모델 포장
            generator_llm = LangchainLLMWrapper(main.llm)
            generator_embeddings = LangchainEmbeddingsWrapper(main.embeddings)

            generator = TestsetGenerator(
                llm=generator_llm,
                embedding_model=generator_embeddings
            )
            # Ragas가 한국어로 질문과 정답을 생성하도록 프롬프트 번역 및 언어 로컬라이징 적용
            generator.adapt_to_language("ko")


            from ragas.run_config import RunConfig
            run_config = RunConfig(
                max_workers=2,
                max_retries=10,
                timeout=180
            )

            print("-> LLM 기반 테스트셋 생성 중 (1~2분 정도 소요될 수 있습니다)...")
            testset = generator.generate_with_langchain_docs(
                documents=chunk_docs,
                testset_size=5,
                run_config=run_config,
                with_debugging_logs=False
            )

            df_testset = testset.to_pandas()
            for _, row in df_testset.iterrows():
                # Ragas 버전별 컬럼명 차이 대응 (question/ground_truth vs user_input/reference)
                q = row.get("user_input") or row.get("question")
                gt = row.get("reference") or row.get("ground_truth")
                if q and gt:
                    test_cases.append({
                        "question": q,
                        "ground_truth": gt
                    })

            # 생성된 테스트 케이스를 파일에 저장
            if test_cases:
                with open(test_cases_file, "w", encoding="utf-8") as f:
                    json.dump(test_cases, f, ensure_ascii=False, indent=4)
                print(f"-> {len(test_cases)}개의 테스트 케이스를 생성하여 '{test_cases_file}'에 캐싱 완료.")
        else:
            print("🚨 경고: ./data/SSAFY_GUIDE.md 파일이 없어 테스트셋 생성을 생략합니다.")
    except Exception as ge:
        print(f"🚨 Ragas Testset 생성 실패 (기본 Fallback 질문으로 대체): {ge}")

# 생성 및 캐싱 실패 시 최종 Fallback 질문 세트 지정
if not test_cases:
    print("-> 기본 정의된 3개의 테스트 케이스를 사용합니다.")
    test_cases = [
        {
            "question": "예비군 훈련으로 결석하는 경우는 어떻게 처리되나요?",
            "ground_truth": "공가로 처리되어 출석으로 인정되고 교육지원금도 지급됩니다. 다만 증빙 문서를 반드시 제출해야 합니다."
        },
        {
            "question": "교육 명찰을 분실하면 어떻게 해야 하고 비용은 얼마인가요?",
            "ground_truth": "교육 명찰을 분실하면 즉시 신고하고 분실 사유서 작성 후 재발급받아야 하며, 재발급 비용 약 16,500원은 본인이 부담합니다. 재발급 기간에는 임시 명찰을 패용합니다."
        },
        {
            "question": "정규 교육 시간과 점심 시간은 각각 몇 시부터 몇 시까지인가요?",
            "ground_truth": "정규 교육 시간은 평일 09:00 ~ 18:00 이며, 중식 시간은 12:00 ~ 13:00 입니다. 토요일, 일요일 및 공휴일은 미운영합니다."
        }
    ]


# 3. RAG 시스템을 호출하여 평가에 필요한 데이터(답변, 검색된 컨텍스트) 수집
print("\n2. 테스트 케이스에 대한 RAG 응답 수집 중...")
eval_data = []

for case in test_cases:
    question = case["question"]
    ground_truth = case["ground_truth"]

    print(f"질문 수행 중: {question}")
    # RAG 워크플로우 직접 실행
    initial_state = {
        "question": question,
        "use_rag": True,
        "context": "",
        "source": ""
    }
    final_state = main.rag_workflow.invoke(initial_state)

    # RAG가 찾아온 원본 문장 리스트 확보
    retrieved_contexts = [final_state["context"]] if final_state["context"] else []

    eval_data.append({
        "user_input": question,
        "response": final_state["answer"],
        "retrieved_contexts": retrieved_contexts,
        "reference": ground_truth
    })

# 4. Ragas EvaluationDataset 구축 및 평가 실행
print("\n3. Ragas 평가 시작 (LLM 심사 구동)...")
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall, ContextPrecision

# Ragas 0.4 규격에 맞게 데이터셋 로드
dataset = EvaluationDataset.from_list(eval_data)

# 평가에 사용할 지표 구성 (4대 지표)
metrics = [
    Faithfulness(llm=main.llm),
    AnswerRelevancy(llm=main.llm),
    LLMContextRecall(llm=main.llm),
    ContextPrecision(llm=main.llm)
]

# Ragas 평가 실행
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=main.llm,
    embeddings=main.embeddings
)

# 5. 결과 시각화 및 저장
print("\n================ Ragas 평가 결과 ================")
print(results)
print("=================================================")

# 판다스 데이터프레임으로 변환하여 상세 결과 출력 및 CSV/MD 저장
df_results = results.to_pandas()
print("\n[상세 리포트]")
print(df_results[["user_input", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]])

# CSV 저장 (기존 코드 유지)
output_csv = "ragas_eval_results.csv"
df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n평가 상세 결과가 '{output_csv}' 파일로 저장되었습니다.")

# MD 저장 (누적 추가)
from datetime import datetime
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
output_md = "ragas_eval_history.md"

# 평균값 계산
avg_faithfulness = df_results["faithfulness"].mean()
avg_relevancy = df_results["answer_relevancy"].mean()
avg_recall = df_results["context_recall"].mean()
avg_precision = df_results["context_precision"].mean()

# 마크다운 내용 구성
md_content = f"""
## 📊 RAG 성능 평가 결과 (실행 시각: {now_str})

### 1. 평균 점수 요약
| Faithfulness (충실도) | Answer Relevancy (관련성) | Context Recall (재현율) | Context Precision (정밀도) |
| :---: | :---: | :---: | :---: |
| {avg_faithfulness:.4f} | {avg_relevancy:.4f} | {avg_recall:.4f} | {avg_precision:.4f} |

### 2. 상세 질문별 리포트
| 질문 (user_input) | Faithfulness | Answer Relevancy | Context Recall | Context Precision |
| :--- | :---: | :---: | :---: | :---: |
"""

for _, row in df_results.iterrows():
    # Ragas 평가지표 중 None이나 NaN 값이 있을 수 있으므로 안전하게 처리
    f_val = row.get("faithfulness", 0.0)
    r_val = row.get("answer_relevancy", 0.0)
    rc_val = row.get("context_recall", 0.0)
    p_val = row.get("context_precision", 0.0)

    f_str = f"{f_val:.4f}" if pd.notna(f_val) else "N/A"
    r_str = f"{r_val:.4f}" if pd.notna(r_val) else "N/A"
    rc_str = f"{rc_val:.4f}" if pd.notna(rc_val) else "N/A"
    p_str = f"{p_val:.4f}" if pd.notna(p_val) else "N/A"

    md_content += f"| {row['user_input']} | {f_str} | {r_str} | {rc_str} | {p_str} |\n"

md_content += "\n---\n"

# 파일에 누적(Append)하여 쓰기
file_exists = os.path.exists(output_md)
with open(output_md, "a", encoding="utf-8") as f:
    if not file_exists:
        f.write("# 📈 Ragas RAG 성능 평가 누적 기록\n\n")
    f.write(md_content)

print(f"평가 결과가 누적 기록 파일 '{output_md}'에 추가되었습니다.")

