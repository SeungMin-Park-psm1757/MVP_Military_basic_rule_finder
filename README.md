# 군인의 지위 및 복무에 관한 기본법 챗봇 MVP

공개 법령 자료를 근거로 `현행 규정`, `개정 이유`, `실무 참고`를 구분해 답하는 버전 관리형 법규 RAG MVP입니다.  
일반적인 법률자문 봇이 아니라 `근거 검색 + 요약 + 원문 링크` 중심으로 설계된 데모 앱입니다.

- 저장소: [GitHub Repository](https://github.com/SeungMin-Park-psm1757/MVP_Military_basic_rule_finder)
- 실행 페이지: [http://127.0.0.1:8501](http://127.0.0.1:8501)
  로컬 기본 실행 포트 기준입니다.

## 프로젝트 목적

이 저장소는 아래 질문에 직접 답하는 법규 QA 데모를 목표로 합니다.

- `현행 규정을 찾아줘`
- `왜 바뀌었는지 설명해줘`
- `실무상 어떻게 보나`

핵심 원칙은 다음과 같습니다.

- 법률자문처럼 단정하지 않습니다.
- 검색 근거가 없는 내용은 일반론으로 메우지 않습니다.
- 답변 하단에 근거 요약과 원문 링크를 함께 제공합니다.
- `LAW_API_KEY` 없이도 데모 모드로 실행 가능합니다.

## 포함 데이터

공개 데모 코퍼스는 다음 범위를 포함합니다.

- 군인의 지위 및 복무에 관한 기본법
- 군인의 지위 및 복무에 관한 기본법 시행령
- 군인의 지위 및 복무에 관한 기본법 시행규칙
- 제·개정이유
- 제·개정문 / 신구 비교 자료
- 군인복무규율 초기본, 말기본, 제·개정이유 목록, 제·개정문 목록

## 기술 스택

- Python
- Streamlit
- Chroma PersistentClient
- Gemini 2.0 Flash
- Sentence Transformers

## 답변 방식

질문 유형에 따라 출력 구조를 다르게 구성합니다.

- 현행 규정형
  `한 줄 결론 / 주요 규정 / 적용상 참고 / 근거`
- 개정 이유형
  `한 줄 결론 / 주요 개정 이유 / 실제 제도 변화 / 해석 포인트 / 근거`
- 실무 참고형
  `한 줄 결론 / 실무적으로 보면 / 주의사항 / 근거`

질문 라우팅은 아래 기준을 우선 사용합니다.

- 현행 규정형: `현행 규정`, `현재 규정`, `찾아줘`, `무슨 내용`, `조문`
- 개정 이유형: `왜 바뀌었`, `개정 이유`, `배경`, `취지`
- 실무 참고형: `실무`, `적용`, `참고`, `주의`

## 주요 디렉터리

```text
.
├─ .agents/skills/
├─ config/
├─ data/
├─ data_manifests/
├─ docs/
├─ prompts/
├─ scripts/
├─ src/army_reg_rag/
├─ tests/
├─ AGENTS.md
├─ streamlit_app.py
├─ requirements.txt
└─ README.md
```

## 빠른 시작

### 1. 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.example`을 복사해 `.env`를 만듭니다.

```bash
copy .env.example .env
```

필수:

- `GEMINI_API_KEY`

선택:

- `LAW_API_KEY`
  향후 Open API 기반 전량 수집을 붙일 때만 필요합니다. 현재 MVP는 없어도 동작합니다.

### 3. 샘플 코퍼스 생성 및 적재

```bash
python scripts/build_sample_corpus.py
python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl
```

### 4. 앱 실행

```bash
streamlit run streamlit_app.py
```

기본 접속 주소:

- [http://127.0.0.1:8501](http://127.0.0.1:8501)

## 공개 자료 수집 흐름

### A. Public page 기준 수집

```bash
python scripts/download_sources_from_manifest.py --manifest data_manifests/public_law_sources.csv
python scripts/normalize_raw_to_jsonl.py
python scripts/ingest_to_chroma.py --input data/processed/law_corpus.jsonl
```

### B. Open API 확장

현재 MVP의 필수 범위는 아닙니다.  
추후 법령 연혁 본문 전량 수집이나 법령 간 확장 관계 분석이 필요할 때 사용합니다.

관련 가이드는 [`data_manifests/public_law_sources.csv`](data_manifests/public_law_sources.csv)에 주석성 항목으로 남겨 두었습니다.

## 검증 명령

의미 있는 수정 후에는 아래 순서로 검증합니다.

```bash
python scripts/build_sample_corpus.py
python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl
pytest
```

UI를 수정했다면 추가로 실행합니다.

```bash
streamlit run streamlit_app.py
```

## 현재 앱 특징

- 메인 화면은 챗봇 입력 중심 단일 흐름입니다.
- 자료 유형 필터는 `법령 / 개정이유 / 신구 비교 / 기타` 4개 그룹입니다.
- 예시 질문 클릭은 질문 한도 차감 없이, 최신 예시 1건만 대화 영역에 표시합니다.
- Gemini 사용량은 전역 파일 기준으로 추적하며, 한도 소진 시 `한도 소진(추가 답변이 제한)`만 표시합니다.
- 근거 영역은 `근거(하단 원문 링크 참고)` 아래에 요약줄과 `원문 링크`를 제공합니다.

## 예시 질문

- `군인의 지위 및 복무에 관한 기본법에서 휴가 관련 현행 규정을 찾아줘.`
- `왜 육아시간 관련 규정이 바뀌었는지 개정 이유 중심으로 설명해줘.`
- `휴가와 돌봄 관련 사안을 실무상 어떤 순서로 확인해야 하는지 알려줘.`
- `군인복무규율에서 군인기본법 체계로 넘어오며 무엇이 달라졌는지 설명해줘.`

## 문서

- [01_research_bibliography.md](docs/01_research_bibliography.md)
- [02_final_plan.md](docs/02_final_plan.md)
- [03_system_architecture.md](docs/03_system_architecture.md)
- [04_evaluation_framework.md](docs/04_evaluation_framework.md)
- [05_future_internal_rule_ingest_plan.md](docs/05_future_internal_rule_ingest_plan.md)
- [06_release_review.md](docs/06_release_review.md)

## 주의사항

- 이 앱은 공개 법규 기반 실무 참고용 도구입니다.
- 실제 승인, 징계, 복무 처리에는 최신 원문과 소속 부대 지침 확인이 필요합니다.
- 비공개 군 내부 규정과 개인정보는 현재 저장소 범위에 포함하지 않습니다.
