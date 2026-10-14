# RAG 챗봇: 파이프라인 설계와 문제 해결 과정

> LangGraph로 RAG 파이프라인을 구성하고, 구현 과정에서 발견한 문제를 원인 분석 → 해결 → 검증 체계 구축 순서로 개선한 기록이다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 목표 | 조직 내부 규정과 공공데이터처럼 LLM이 정확히 알기 어려운 정보를 근거로 답변하는 챗봇 |
| 데이터 1 | SSAFY 교육생 가이드 (출결·교육시간·보안·장학 등 규정과 표) |
| 데이터 2 | 전국관광지정보표준데이터 (관광지 735곳) |
| 핵심 기술 | FastAPI, LangGraph, LangChain, ChromaDB, BM25, RAGAS |
| 모델 | `gpt-5-mini` (생성), `text-embedding-3-small` (임베딩) |

일반 LLM은 "예비군 훈련으로 결석하면 어떻게 처리돼?" 같은 **조직 내부 규정**이나, "죽녹원 주차 몇 대 돼?" 같은 **세부 공공데이터**를 정확히 알고 있다고 보장할 수 없다.
따라서 문서를 Vector DB에 저장하고, 질문과 관련된 부분을 검색해 그 내용만 근거로 답하도록 RAG 구조를 설계하였다.

두 데이터는 구조가 달라 **청킹 전략도 다르게** 적용하였다.

| 데이터 | 구조 | 청킹 전략 |
| --- | --- | --- |
| SSAFY 가이드 | 규정 본문, 예외사항, 표가 연속된 긴 문서 | 글자 수 기준 분할 → Chunk Size / Overlap 튜닝 |
| 관광지 정보 | 관광지 1곳 = 독립된 레코드 | 헤더 기준 분할 (관광지 1곳 = 청크 1개) |

---

## 2. 전체 아키텍처

```
[Browser] HTML / JS
    │  fetch POST /chat {message, use_rag}
    ▼
[FastAPI] main.py
    │
    ▼
[LangGraph Workflow]
    START → (분기1) → retrieve → (분기2) → generate_rag / generate_general → END
                         │
                         ▼
               [ChromaDB (cosine)] + [BM25]
                         │
                         ▼
                  [OpenAI Chat API]
    │
    ▼
JSON {answer, source, rag_used, retrieval_scores}
    │
    ▼
[Browser] 답변 + "출처: 파일명" 또는 "일반 생성형 AI 응답 (관련 문서 없음)"
```

파이프라인은 두 단계로 나뉜다.

| 단계 | 시점 | 흐름 |
| --- | --- | --- |
| **인덱싱** | 문서 업로드 시 (`/upload`) | 추출 → 청킹 → 임베딩 → ChromaDB 저장 → BM25 인덱스 무효화 |
| **질의** | 질문 시 (`/chat`) | 질문 → LangGraph(검색 → 분기 → 생성) → 응답 |

---

## 3. 데이터 준비

### 3-1. CSV → Markdown 변환

원본 CSV를 그대로 넣으면 한 줄에 값만 나열되어 **어떤 값이 무엇인지(열 이름)가 사라진다.**
그래서 관광지 1개(CSV 한 행)를 Markdown 섹션 1개로 변환하였다.

```markdown
## 24. 죽녹원

- **관광지 구분**: 관광지
- **도로명 주소**: 전라남도 담양군 담양읍 죽녹원로 119
- **주차 가능 수**: 70
- **관광지 소개**: 2003년 5월에 조성하여 약 31만㎡의 울창한 대나무숲이 ...
- **관리기관 전화번호**: 061-380-2780
```

필드명과 값이 함께 있어 **임베딩과 LLM 모두 의미를 해석하기 쉽고**, `##` 헤더가 레코드 경계 역할을 한다.
빈 값은 생략하여 불필요한 토큰을 줄였다.

### 3-2. 데이터에서 확인한 품질 문제

| 문제 | 실제 사례 | RAG에 미치는 영향 |
| --- | --- | --- |
| 의미 없는 소개 값 | `관광지 소개: Y` (12건), `관광지 소개: 관광진흥과` (부서명) | 소개를 물으면 근거가 없어 답변 불가 |
| 소개 필드 길이 편차 | `관광지 소개` 1자 ~ 1,574자 (중앙값 121자) | 소개가 긴 레코드는 임베딩이 여러 주제를 담아 유사도가 흐려짐 |
| 필드 누락 | 도로명 주소가 있는 레코드 659 / 735 | 주소 기반 질문에서 일부 누락 |
| 중복 레코드 | `114. 푸른길공원`, `419. 푸른길공원` (주소·주차 수가 서로 다름) | 같은 이름으로 다른 정보가 검색되어 답변이 충돌할 수 있음 |
| 단위 없는 수치 | `면적: 321561.8`, `공공편익시설: 106173.5` | LLM이 수치의 의미를 잘못 해석할 수 있음 |

→ RAG의 답변 품질은 LLM보다 **입력 데이터 품질에 먼저 묶여 있다**는 점을 확인하였다. (정제 계획은 9절)

---

## 4. 인덱싱 파이프라인 설계

### 4-1. 형식별 추출

| 형식 | Loader | 비고 |
| --- | --- | --- |
| `.pdf` | `PyMuPDFLoader` | 페이지 단위 추출 |
| `.txt`, `.md` | `TextLoader(encoding="utf-8")` | 한글 깨짐 방지 |

모든 청크의 `metadata["source"]`에 파일명을 저장하여 답변 출처 표시에 사용한다.

### 4-2. 청킹 전략: 문서 구조를 따른다

| 형식 | Splitter | 단위 |
| --- | --- | --- |
| `.md` (기본) | `MarkdownHeaderTextSplitter` | `##` 헤더 = 관광지 1개 = 청크 1개 (736개) |
| `.txt`, `.pdf` | `RecursiveCharacterTextSplitter` | 500자 / overlap 50, 구분자 `\n\n → \n → . → 공백` |

```python
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
    strip_headers=False,   # 본문에 "## 죽녹원"을 남겨 이름으로 묻는 질문과 매칭되도록
)
```

- `strip_headers=False`: 헤더를 본문에서 지우면 "죽녹원 어때?"처럼 **이름으로 묻는 질문의 유사도가 떨어진다.**
- 헤더 정보는 `metadata["Header 2"] = "24. 죽녹원"`으로도 보존된다.
- `MD_SPLIT_MODE=recursive`로 바꾸면 `.md`도 고정 길이로 분할되어, 두 방식을 실험으로 비교할 수 있다.

### 4-3. 저장

- **ChromaDB**: `hnsw:space=cosine`으로 컬렉션 생성 (이유는 7-3절)
- **BM25 인덱스**: Chroma에 저장된 문서로 첫 검색 시 생성하고 캐싱하며, 업로드나 초기화 시 무효화
- 서버 종료 시 `chroma_data/`를 삭제하여 실행 간 데이터가 섞이지 않게 함

---

## 5. 질의 파이프라인 설계 (LangGraph)

### 5-1. 왜 LangGraph인가

단순 체인(`retriever | prompt | llm`)은 흐름이 한 방향으로 고정되어 있다.
이 프로젝트는 **"RAG를 켰는가", "관련 문서를 찾았는가"에 따라 다른 경로**로 답해야 하므로,
State와 조건부 분기를 명시적으로 표현할 수 있는 LangGraph를 사용하였다.

### 5-2. State 설계

```python
class RAGState(TypedDict, total=False):
    question: str                   # 사용자 질문
    use_rag: bool                   # 사용자가 RAG를 켰는가 (요청)
    context: str                    # LLM에 넣을 참고 문서
    answer: str                     # 최종 답변
    source: str                     # 출처 파일명
    rag_used: bool                  # 실제로 문서를 근거로 답했는가 (결과)
    retrieved_contexts: list[str]   # LLM에 전달된 청크 목록
    retrieval_scores: list[float]   # 각 청크의 검색 점수
```

`use_rag`(요청)와 `rag_used`(결과)를 분리한 것이 핵심이다.
사용자가 RAG를 켜도 관련 문서가 없으면 `rag_used=False`가 되며, 이 값이 **분기, UI 표시, 평가**에 모두 쓰인다.

### 5-3. 그래프 구조

```
START ─┬─ use_rag=True  → retrieve ─┬─ 관련 문서 있음 → generate_rag ─────→ END
       │                            └─ 관련 문서 없음 → generate_general ─→ END
       └─ use_rag=False ─────────────────────────→ generate_general ─→ END
```

| 노드 / 분기 | 역할 |
| --- | --- |
| 분기 1 `route_by_rag_flag` | RAG를 끄면 검색을 건너뛰어 **임베딩 호출과 DB 조회 비용을 없앰** |
| `retrieve` | 검색 + Threshold 필터 → context, 출처, 점수, `rag_used` 기록 |
| 분기 2 `route_by_retrieval` | 관련 문서가 없으면 일반 생성으로 Fallback |
| `generate_rag` | 참고 문서 기반 프롬프트로 생성 |
| `generate_general` | 일반 대화 프롬프트로 생성 |

생성 노드를 둘로 나누어 **어떤 경로로 답했는지가 그래프 구조 자체에 드러나도록** 하였다.

### 5-4. 프롬프트 설계

| 구분 | System 지시 | Few-Shot |
| --- | --- | --- |
| RAG | 여행·규정 등 정보 안내 어시스턴트, `[참고문서]`만 근거, 친근한 어조 | "아트빌리지 어때?" → 소개 / "주차할 수 있어?" → 142대 |
| 일반 | 밝고 친근한 대화 | "안녕? 반가워" → 인사 |

- Few-Shot 예시는 실제 레코드 `178. 아트빌리지(한옥마을)`의 내용으로 작성하여 **답변 길이와 말투를 고정**하였다.
- 참고문서에 없는 내용은 정해진 거절 문구로 답하도록 지시하여 지어낸 답변을 막았다.
  > "앗, 그 내용은 참고 문서에 나와 있지 않네요! 다른 궁금한 점이 있으신가요?"

---

## 6. 검색 설계

| 모드 | 방식 | 강한 질문 |
| --- | --- | --- |
| `vector` | 코사인 유사도 Top-K + Threshold | "바닷물이 맑고 얕아서 가족 피서지 좋은 곳" 같은 서술형 |
| `hybrid` | Vector + BM25를 **RRF**로 결합 | "죽녹원", "심우장" 같은 고유명사 |

**RRF (Reciprocal Rank Fusion)**

```
score(doc) = Σ  1 / (60 + rank)      ← Vector 순위와 BM25 순위 각각에 대해 합산
```

두 검색기의 점수 단위가 달라 직접 더할 수 없으므로 **순위만으로 결합**한다. 두 검색 모두에서 상위에 있는 문서가 가장 높은 점수를 받는다.

서버 기본값은 `.env`로 분리하고, 실험 시에는 그래프 실행 인자(`config={"configurable": {...}}`)로 덮어쓴다.

```env
RETRIEVER_MODE=vector       # vector | hybrid
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.0     # 코사인 유사도
MD_SPLIT_MODE=header        # header | recursive
CHUNK_SIZE=500
OVERLAP=50
```

---

## 7. 문제 상황과 해결

### 7-1. 고정 길이 청킹이 관광지 경계를 깬다

| | |
| --- | --- |
| **문제** | 500자 단위로 자르면 관광지 두 곳이 한 청크에 섞이거나, 한 관광지가 두 청크로 쪼개진다. |
| **원인** | 관광지 레코드 길이가 330~1,047자(중앙값 457자)로 제각각이라 고정 길이(500자)와 레코드 경계가 맞지 않는다. |
| **해결** | `##` 헤더 기준 분할로 관광지 1개 = 청크 1개를 보장 (736청크, 고정 길이 분할은 1,088청크) |
| **검증** | `baseline`(헤더) vs `recursive_300_50` / `500_50` / `800_50` 실험으로 Context Precision/Recall 비교, Overlap은 `recursive_500_0` / `500_100`으로 비교 |

### 7-2. 관련 문서가 없어도 RAG 답변을 하고, 표시도 실제와 달랐다

| | |
| --- | --- |
| **문제** | 검색 결과가 0개여도 RAG 프롬프트로 답했고, 출처는 "ChatGPT 일반 지식"으로 표시되어 **실제 동작과 화면 표시가 어긋났다.** |
| **원인** | 그래프가 `retrieve → generate` 한 방향뿐이라 검색 결과에 따른 분기가 없었다. |
| **해결** | `retrieve` 뒤에 분기 2(`route_by_retrieval`)를 추가하고, `rag_used`를 응답에 포함해 화면에 "일반 생성형 AI 응답 (관련 문서 없음)"을 표시 |

### 7-3. Threshold가 동작하지 않음

| | |
| --- | --- |
| **문제** | `score_threshold=0.0`인데도 검색 결과가 전부 걸러지는 경우가 발생했다. |
| **원인** | Chroma 기본 거리(L2)를 relevance score로 바꾸는 식(`1 - d/√2`)은 거리가 크면 **음수**가 된다. 그래서 Threshold 0도 필터로 작동하고, 값의 의미도 해석하기 어렵다. |
| **해결** | 컬렉션을 `hnsw:space=cosine`으로 생성 → 점수 = 코사인 유사도로 통일. Threshold를 "유사도 0.3 이상"처럼 직관적으로 설정할 수 있다. |
| **확인** | 테스트 임베딩으로 L2일 때 결과 0건, cosine 전환 후 정상 검색되는 것을 확인 |

### 7-4. 고유명사 질문에 의미 검색이 약하다

| | |
| --- | --- |
| **문제** | "죽녹원 전화번호"처럼 관광지 이름이 핵심인 질문에서 이름이 다른 관광지가 상위에 올라올 수 있다. |
| **원인** | 임베딩은 의미를 압축하므로 "죽녹원"이라는 정확한 문자열 일치를 보장하지 않는다. |
| **해결** | BM25 키워드 검색을 추가하고 Vector 결과와 RRF로 결합 (`RETRIEVER_MODE=hybrid`) |
| **세부** | 한국어는 조사가 붙어("죽녹원은", "죽녹원에서") 공백 기준 토큰화로는 매칭이 안 된다 → **단어 + 글자 bigram** 토큰화로 해결 |
| **검증** | `baseline` vs `hybrid` 실험으로 Context Recall 비교 |

### 7-5. Hybrid 모드에서 Fallback이 일어나지 않음

| | |
| --- | --- |
| **문제** | 7-4 적용 후, 문서에 없는 질문에도 항상 RAG 답변 경로로 들어갔다. |
| **원인** | BM25는 관련 없는 질문에도 bigram 하나만 겹치면 **항상 결과를 반환한다.** |
| **해결** | "관련 문서 존재 여부"는 Vector 검색의 Threshold 통과 여부로만 판단하고, BM25는 순위 보강에만 사용 |

```python
vector_hits = [doc for doc, _ in vector_search(question, k * 2)]
if not vector_hits:
    return []          # 관련 문서 없음 → generate_general로 Fallback
```

### 7-6. 평가 수치가 의미 있게 측정되지 않았다

| | |
| --- | --- |
| **문제 1** | 검색된 Top-5 청크를 **문자열 하나로 합쳐서** RAGAS에 전달했다. |
| **원인** | Context Precision은 "가져온 청크 중 쓸모 있는 것의 비율과 순위"를 보는 지표인데, 청크가 1개로 합쳐지면 순위 정보가 사라진다. Top-K나 Threshold를 튜닝해도 지표에 반영되지 않는다. |
| **해결** | State의 `retrieved_contexts`(리스트)를 그대로 전달 |
| **문제 2** | 테스트셋이 SSAFY 규정 질문뿐이라, 관광지 데이터를 넣으면 질문과 문서가 맞지 않았다. |
| **해결** | 데이터셋별로 테스트셋을 분리 (`test_cases_ssafy.json` 10문항, `test_cases_trip.json` 14문항). 관광지 질문은 실제 레코드에서 정답을 뽑아 작성 (아래 표) |

| 유형 | 예시 | 목적 |
| --- | --- | --- |
| 단일 사실 | 죽녹원 관리기관 전화번호 알려줘 | 정확한 필드 추출 |
| 수치 | 주문진해변관광지 주차 몇 대? | 숫자 정확성 |
| 설명 | 심우장은 어떤 곳이야? | 소개 요약 |
| 주소 / 광주 | 길상사 주소, 조선대학교 장미원 | 지역 정보 |
| 서술형 질의 | 바닷물이 맑고 얕은 가족 피서 해변 | 이름 없이 의미로 찾기 (Vector 강점) |
| 문서에 없음 | 죽녹원 입장료는? | 지어내지 않는지 확인 |

### 7-7. "문서에 없는 질문"은 RAGAS로 평가할 수 없다

| | |
| --- | --- |
| **문제** | 정답이 "정보 없음"인 질문은 Context Recall/Precision 계산이 성립하지 않는다. |
| **해결** | 이 유형은 RAGAS에서 제외하고 **근거 없는 답변 회피** 여부를 따로 측정: 거절 문구를 출력했거나(`참고 문서에 나와 있지 않`) 일반 AI 응답으로 표시했으면 통과 |

### 7-8. 실험을 재현하고 비교할 수 없었다

| | |
| --- | --- |
| **문제** | 설정값이 코드에 하드코딩되어 있었고, 결과 파일에는 점수만 남고 **어떤 설정으로 얻은 점수인지** 남지 않았다. 같은 DB를 계속 써서 이전 실험 청크가 섞일 위험도 있었다. |
| **해결 1** | 실험을 코드로 정의: 가설 하나당 변수 하나만 baseline에서 변경 |
| **해결 2** | 청킹 설정별로 **독립된 인메모리 Chroma 컬렉션** 생성. 청킹이 같은 실험(Top-K, Threshold, 검색 방식만 다름)은 인덱스를 재사용해 임베딩 비용 절감 |
| **해결 3** | 설정값 + 4개 지표 + Overall + 회피율을 `experiments_<데이터셋>.csv`에 누적하고, 비교표 `experiments_<데이터셋>.md` 자동 생성 (Baseline 대비 증감, 최고 실험 표시) |
| **해결 4** | 검색할 때마다 적용된 Top-K / Threshold / 검색 방식과, 검색된 청크의 청킹 설정을 LangGraph State의 `retrieval_config`에 기록하여 질문별 결과에도 함께 저장. 청킹 설정은 인덱싱 시점에 청크 메타데이터로 남기고 검색 시 그 값을 읽어, 실제와 다른 값이 기록되지 않도록 함 |

### 7-9. 평가 모델 호환성

| | |
| --- | --- |
| **문제** | `gpt-5-mini`는 `temperature` 파라미터를 지원하지 않아 RAGAS 내부 호출이 실패했다. |
| **해결** | OpenAI SDK의 `create` 함수를 감싸 `temperature`를 제거하는 패치 적용 (동기/비동기 모두) |

---

## 8. 평가 체계

### 8-1. 지표

| 지표 | 평가 대상 | 질문 |
| --- | --- | --- |
| Context Recall | Retriever | 정답에 필요한 정보를 검색했는가 |
| Context Precision | Retriever | 검색한 청크 중 쓸모 있는 것이 상위에 있는가 |
| Faithfulness | Generator | 답변이 검색 문서에 근거하는가 |
| Answer Relevancy | Generator | 답변이 질문에 맞는가 |
| **Overall** | 전체 | 4개 지표 평균 |
| 근거 없는 답변 회피 | 전체 | 문서에 없는 질문에 지어내지 않았는가 |

Retriever와 Generator를 나누어 보면 **문제가 검색 단계에서 생겼는지 생성 단계에서 생겼는지** 구분할 수 있다.

### 8-1-1. 평가·튜닝 파이프라인 (LangGraph)

실험을 단순 반복문이 아니라 **LangGraph 그래프**로 설계하였다. 실험 하나가 노드 5개를 거쳐 순환하고, 모든 실험이 끝나면 비교 노드로 이동한다.

```
START → select_experiment → index → generate → ragas → record ─┬─ 남은 실험 있음 → select_experiment
                                   (답변 그래프 호출)            └─ 모든 실험 완료 → compare → END
```

| 노드 | 역할 |
| --- | --- |
| `select_experiment` | 실험 큐에서 다음 실험의 Chunk / Overlap / Top-K / Threshold 결정 |
| `index` | 현재 실험의 Chunk Size / Overlap으로 독립 인덱스 생성 (같은 청킹은 재사용) |
| `generate` | Top-K / Threshold / 검색 방식 / 인덱스를 **RunnableConfig로 주입**해 답변 그래프(`rag_workflow`)를 질문마다 실행, State의 검색 결과와 `retrieval_config` 수집 |
| `ragas` | RAGAS 4개 지표 채점 |
| `record` | 질문별 결과와 실험 요약 저장 (실험마다 저장 → 중간 실패에도 결과 유지) |
| `compare` | Overall 순위와 baseline 대비 개선율 계산 |

- 평가 State의 `summaries`는 `Annotated[list, operator.add]` 리듀서로 누적된다.
- 답변 그래프와 평가 그래프가 분리되어 있어, **서비스 코드를 수정하지 않고** 같은 파이프라인으로 실험할 수 있다.
- 처음에는 실험마다 모듈 전역 설정값(`RAG_TOP_K` 등)을 바꿔 끼웠다. 이 방식은 동시 요청 시 설정이 섞일 수 있고 노드가 숨은 전역 상태에 의존한다. 그래서 **LangGraph의 RunnableConfig(`config["configurable"]`)로 실행마다 설정을 주입**하도록 바꾸었다. 노드는 `retrieve_node(state, config)`처럼 설정을 인자로 받는다.

### 8-2. 실험 1: SSAFY 교육 규정 문서

동일한 10개 테스트 케이스로 Chunk Size, Overlap, Top-K, Similarity Threshold를 비교하였다.

| 실험 | Chunk | Overlap | Top-K | Threshold |
| --- | --- | --- | --- | --- |
| `baseline` | 500 | 50 | 5 | 0.00 |
| `small_chunk` | 300 | 50 | 5 | 0.00 |
| `large_chunk` | 700 | 100 | 5 | 0.00 |
| `retrieval_tuned` | 500 | 50 | 3 | 0.15 |

> 이 결과는 현재 평가 그래프로 정리하기 **이전 버전의 파이프라인**(Chroma 기본 L2 점수, 검색 청크를 합쳐 평가)에서 측정하였다. 현재 코드로 다시 실행하면 값이 달라질 수 있다.

| 실험 | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Overall |
| --- | --- | --- | --- | --- | --- |
| baseline (500/50) | **0.9900** | **0.5411** | 0.9000 | 0.6672 | 0.7746 |
| small_chunk (300/50) | 0.9550 | 0.5376 | 0.8333 | 0.8122 | 0.7845 |
| **large_chunk (700/100)** | 0.9663 | 0.5363 | **1.0000** | **0.8567** | **0.8398** |
| retrieval_tuned (K3, 0.15) | 0.9800 | 0.5095 | 0.8000 | 0.6833 | 0.7432 |

**분석**

- **Baseline**: Faithfulness와 Recall은 높았지만 Precision이 0.6672로 낮았다. 필요한 청크와 함께 질문과 직접 관련 없는 청크도 검색되었다.
- **Small Chunk (500 → 300)**: Precision이 0.6672 → 0.8122로 올랐다. 청크가 구체적인 내용만 담아 검색된 청크 각각의 정확도가 높아졌다. 반면 Recall은 0.9000 → 0.8333으로 떨어졌다. 하나의 규정이 여러 청크로 나뉘어 필요한 문맥 일부를 놓쳤다.
- **Large Chunk (500/50 → 700/100)**: Recall 1.0000, Precision 0.8567로 **Overall 0.7746 → 0.8398 (+8.4%)** 가장 높았다. SSAFY 가이드는 규정 본문, 예외사항, 표가 연속되어 있어 넓은 청크가 하나의 규정을 온전히 담았다. Top-K 5가 고정된 상태에서 청크마다 담긴 근거가 많아져 Recall과 Precision이 함께 올랐다.
- **Retrieval Tuned (Top-K 5 → 3, Threshold 0 → 0.15)**: Overall 0.7432로 가장 낮았다. 불필요한 청크를 줄이려던 의도와 달리 답변에 필요한 보조 Context까지 제거되어 Recall이 0.8000으로 떨어졌다. → **검색 결과를 적게 전달한다고 성능이 좋아지는 것은 아니다.**
- **Answer Relevancy**는 모든 설정에서 0.51~0.54로 비슷했다. 검색 설정과 무관하므로 원인은 생성 단계(친근한 말투, 이모지, 영어 기반 채점 프롬프트)에 있다고 판단하였다.

**한계 → 실험 2로 이어진 개선**

| 한계 | 개선 |
| --- | --- |
| `large_chunk`는 Chunk와 Overlap을, `retrieval_tuned`는 Top-K와 Threshold를 **동시에** 바꿔 개별 효과를 분리할 수 없었다. | 실험 2에서 **변수 하나만** 바꾸도록 재설계 |
| Threshold 0.15는 Chroma 기본(L2) 거리에서 변환한 점수 기준이라 값의 의미가 불분명했다 (7-3절). | 코사인 유사도로 통일 |
| 청크를 합쳐서 평가에 전달해 Precision 계산이 정확하지 않을 수 있었다 (7-6절). | 청크 리스트를 그대로 전달 |
| 실험 최고 설정(700/100)이 서비스에 반영되지 않았다. 서버는 `.md`를 모두 헤더 분할로 처리했다. | 업로드할 때 문서별 청킹 설정 지정 (`split_mode`, `chunk_size`, `overlap`) |
| 10문항, 1회 측정이라 LLM 채점 편차를 확인하지 못했다. | baseline 반복 측정으로 편차를 확인한 뒤 비교 |

```bash
cd servers
python eval_ragas.py --dataset ssafy --all
```

### 8-3. 실험 2: 전국 관광지 데이터

실험 1의 한계를 반영하여 **baseline에서 변수 하나만 바꾸도록** 설계하였다. 관광지 데이터는 헤더 분할이 기본이므로, Chunk Size와 Overlap은 고정 길이 분할로 바꿔 비교한다.

| 실험 | 변수 | 가설 |
| --- | --- | --- |
| `baseline` | - | 헤더 분할 + Vector Top-5 + Threshold 0 |
| `recursive_300_50` | Chunk Size | 작은 청크는 한 관광지가 여러 조각으로 나뉘어 Recall이 떨어진다 |
| `recursive_500_50` | Chunk Size | 고정 길이 분할은 관광지 경계를 무시해 헤더 분할보다 Precision이 떨어진다 |
| `recursive_800_50` | Chunk Size | 큰 청크는 Recall은 회복되지만 여러 관광지가 섞여 Precision이 떨어진다 |
| `recursive_500_0` | Overlap | 겹침이 없으면 경계에서 잘린 정보가 사라져 Recall이 떨어진다 |
| `recursive_500_100` | Overlap | 겹침을 늘리면 경계 정보가 보존되지만 중복 청크가 늘어난다 |
| `topk_3` | Top-K | K를 줄이면 Precision↑ Recall↓ |
| `topk_8` | Top-K | K를 늘리면 Recall↑ Precision↓ |
| `threshold_0.2` | Threshold | 약한 필터로 노이즈 청크만 제거되어 Precision이 오른다 |
| `threshold_0.3` | Threshold | 낮은 유사도 청크 제거로 Precision과 Faithfulness가 오른다 |
| `threshold_0.4` | Threshold | 필터가 너무 강하면 필요한 청크까지 빠져 Recall이 떨어지고 Fallback이 늘어난다 |
| `hybrid` | 검색 방식 | BM25 결합으로 관광지명 질문의 Recall이 오른다 |

```bash
cd servers
python eval_ragas.py --dataset trip --all
```

---

## 9. 한계와 다음 단계

| 한계 | 다음 단계 |
| --- | --- |
| **멀티턴 미지원**: Few-Shot은 "아트빌리지 어때?" → "주차할 수 있어?"처럼 이어지는 대화를 가정하지만, `/chat`은 이전 대화를 기억하지 못한다. | 대화 기록을 함께 전달하고, 검색 전에 이전 맥락을 반영해 질문을 재작성하는 노드 추가 |
| **지역 조건 검색이 약함**: "광주 관광지 추천"은 735곳 중 유사도만으로 찾기 어렵다. | 주소에서 시/도, 시/군/구를 추출해 metadata로 저장하고 검색 전 필터링 |
| **데이터 품질** (3-2절) | 무의미한 값 제거, 수치에 단위 부여, 중복 레코드 병합 |
| **순위 정밀도** | Top-10 후보를 Cross-Encoder로 재정렬하는 Reranker |
| **위·경도 미사용** | 거리 기반 "근처 관광지" 추천 |
| **평가 신뢰도**: 생성과 채점을 같은 모델이 하고, 테스트셋이 10~14문항이다. | 채점 모델 분리, 오타·짧은 질문·복합 조건 질문을 추가해 테스트셋 확대 |

---

## 10. 정리

```
[설계]  문서 구조별 청킹 (규정 문서: 글자 수 분할 / 관광지: 헤더 분할) → Chroma(cosine) + BM25
        LangGraph: use_rag 분기 → retrieve → 관련 문서 분기 → RAG / 일반 생성

[문제]  청크 경계 붕괴 · 표시와 동작 불일치 · Threshold 무력화(L2 음수 점수)
        고유명사 검색 약점 · Hybrid의 Fallback 실패 · 평가 지표 왜곡 · 실험 비재현성

[해결]  헤더 분할 · 2단계 조건부 분기 + rag_used · 코사인 공간
        BM25(bigram) + RRF · Vector 기준 관련성 판단 · 청크 리스트 평가
        실험 정의 코드화 + 독립 인덱스 + 설정·지표 누적 기록

[평가]  LangGraph 평가 그래프: 실험 선택 → 인덱싱 → 답변 그래프 실행 → RAGAS 채점 → 기록 → (반복) → 비교
[검증]  실험 1 (SSAFY 규정): Chunk/Overlap/Top-K/Threshold 4개 설정 비교
        → Large Chunk 700/100이 Overall 0.7746 → 0.8398 (+8.4%)
        실험 2 (관광지): 변수 하나씩 바꾼 12개 설정 + 근거 없는 답변 회피율
```

RAG의 성능은 LLM 하나로 결정되지 않는다.
**데이터를 어떤 단위로 저장하고, 어떤 기준으로 "관련 있다"고 판단하며, 그 판단을 어떻게 측정하느냐**가 답변 품질을 좌우한다.
이 프로젝트는 그 각 단계를 설계하고, 문제를 원인까지 추적해 고치고, 개선 효과를 수치로 확인할 수 있는 체계를 만드는 데 초점을 두었다.
