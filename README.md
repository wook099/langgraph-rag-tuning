# LangGraph RAG Tuning

LangGraph로 구성한 RAG 챗봇과, RAGAS로 **Chunk Size · Overlap · Top-K · Similarity Threshold**를 비교하는 평가 파이프라인입니다.

- **데이터**: SSAFY 교육생 가이드(규정 문서), 전국 관광지 정보(공공데이터 735곳)
- **스택**: FastAPI · LangGraph · LangChain · ChromaDB · BM25 · RAGAS · OpenAI API

## 파이프라인

### RAG 답변 그래프 (`servers/main.py`)

```mermaid
flowchart LR
    S([START]) -->|RAG on| R[retrieve]
    S -->|RAG off| G2[generate_general]
    R -->|관련 문서 있음| G1[generate_rag]
    R -->|관련 문서 없음| G2
    G1 --> E([END])
    G2 --> E
```

- **retrieve**: Vector(코사인 유사도) 또는 Hybrid(Vector + BM25, RRF 결합) 검색 후 Threshold 필터
- 관련 문서가 없으면 일반 응답으로 전환하고, 화면에 "일반 생성형 AI 응답 (관련 문서 없음)"을 표시
- Top-K · Threshold · 검색 방식은 실행할 때 `config={"configurable": {...}}`로 주입 (기본값은 `.env`)

### 평가·튜닝 그래프 (`servers/eval_ragas.py`)

```mermaid
flowchart LR
    S([START]) --> A[select_experiment] --> B[index] --> C[generate] --> D[ragas] --> F[record]
    F -->|남은 실험| A
    F -->|완료| H[compare] --> E([END])
```

실험마다 청킹 설정별 인덱스를 만들고(`index`), 답변 그래프를 그대로 실행한 뒤(`generate`), RAGAS 4개 지표로 채점해 저장합니다.

## 실험 결과

### 실험 1: SSAFY 교육 규정 (10문항)

| 실험 | Chunk / Overlap | Top-K | Threshold | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 500 / 50 | 5 | 0.00 | 0.9900 | 0.5411 | 0.9000 | 0.6672 | 0.7746 |
| small_chunk | 300 / 50 | 5 | 0.00 | 0.9550 | 0.5376 | 0.8333 | 0.8122 | 0.7845 |
| **large_chunk** | **700 / 100** | 5 | 0.00 | 0.9663 | 0.5363 | **1.0000** | **0.8567** | **0.8398** |
| retrieval_tuned | 500 / 50 | 3 | 0.15 | 0.9800 | 0.5095 | 0.8000 | 0.6833 | 0.7432 |

- **large_chunk**가 Overall **+8.4%**로 가장 높았다. 규정 본문과 예외사항이 한 청크에 함께 담겼다.
- **retrieval_tuned**는 가장 낮았다. 노이즈와 함께 필요한 Context까지 줄어 Recall이 떨어졌다.
- 한계: large_chunk와 retrieval_tuned는 변수 두 개를 동시에 바꿨다. 또 이 결과는 현재 코드로 정리하기 전 버전(L2 점수, 청크를 합쳐 평가)에서 측정했다.

### 실험 2: 전국 관광지 (14문항)

실험 1의 한계를 반영해 **각 실험이 비교 기준과 변수 하나만 다르도록** 설계했습니다.

| 변수 | 실험 | 비교 기준 |
| --- | --- | --- |
| 분할 방식 | `recursive_500_50` (헤더 분할 → 500자 분할) | `baseline` |
| Chunk Size | `recursive_300_50`, `recursive_800_50` | `recursive_500_50` |
| Overlap | `recursive_500_0`, `recursive_500_100` | `recursive_500_50` |
| Top-K | `topk_3`, `topk_8` | `baseline` |
| Threshold | `threshold_0.2`, `threshold_0.3`, `threshold_0.4` | `baseline` |
| 검색 방식 | `hybrid` | `baseline` |

"문서에 없음" 2문항은 RAGAS 대신 **근거 없는 답변을 지어내지 않았는지**로 따로 평가합니다.

설계 배경과 문제 해결 과정은 [docs/RAG_정리_보완.md](docs/RAG_정리_보완.md)에 정리했습니다.

## 실행

```bash
cd servers
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`servers/.env`

```env
API_KEY=발급받은_API_키
base_url=사용할_API_Base_URL
GPT_MODEL=gpt-5-mini
EMBEDDING_MODEL=text-embedding-3-small
```

선택 항목은 `CHUNK_SIZE`, `OVERLAP`, `MD_SPLIT_MODE`, `RAG_TOP_K`, `RAG_SCORE_THRESHOLD`, `RETRIEVER_MODE`입니다.

**서버 · 클라이언트**

```bash
python main.py                                  # http://localhost:8000 (API 문서: /docs)
cd ../clients && python -m http.server 5500     # http://localhost:5500
```

**문서 업로드** (청킹 설정은 선택)

```bash
curl -X POST http://localhost:8000/upload -F "file=@./data/SSAFY_GUIDE.md" \
  -F "split_mode=recursive" -F "chunk_size=700" -F "overlap=100"
```

**평가**

```bash
python eval_ragas.py --dataset ssafy --all    # 실험 1
python eval_ragas.py --dataset trip --all     # 실험 2
python eval_ragas.py --dataset trip --list    # 실험 목록과 비교 기준
python eval_ragas.py --graph                  # 그래프 구조(Mermaid) 출력
```

결과는 `servers/eval_results/`에 비교표(`experiments_<데이터셋>.md`)와 질문별 상세 CSV로 저장됩니다.
