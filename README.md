# 군인의 지위 및 복무에 관한 기본법 챗봇 MVP

공개 법령 자료를 근거로 `현행 규정`, `개정 이유`, `실무 참고`를 구분해 답하는 버전 관리형 법규 RAG MVP입니다.  
일반적인 법률자문 봇이 아니라 `근거 검색 + 요약 + 원문 링크` 중심으로 설계된 데모 앱입니다.

- 저장소: [GitHub Repository](https://github.com/SeungMin-Park-psm1757/MVP_Military_basic_rule_finder)
- GitHub 실행 링크: [Open in Codespaces](https://codespaces.new/SeungMin-Park-psm1757/MVP_Military_basic_rule_finder?quickstart=1)
  GitHub Codespaces에서 저장소를 열면 앱이 자동으로 준비되고 미리보기 포트가 열립니다.
- Render 무료 배포 링크: [https://military-basic-rule-chatbot.onrender.com](https://military-basic-rule-chatbot.onrender.com)
  Render에서 free web service로 배포할 수 있는 주소 형식입니다.

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
- Docker
- Render Blueprint

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

## GitHub에서 바로 실행

이 MVP는 GitHub 저장소에서 바로 열어 실행할 수 있도록 `Codespaces` 기준 설정을 포함합니다.

### 가장 빠른 실행 방법

1. 상단의 [Open in Codespaces](https://codespaces.new/SeungMin-Park-psm1757/MVP_Military_basic_rule_finder?quickstart=1) 링크를 엽니다.
2. Codespace가 생성되면 의존성 설치와 샘플 코퍼스 적재가 자동으로 실행됩니다.
3. `postAttachCommand`가 Streamlit 앱을 자동 실행합니다.
4. GitHub가 포트 `8501` 미리보기를 열면 바로 챗봇 페이지를 확인할 수 있습니다.

### Codespaces에 포함된 설정

- [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json)
  포트 자동 공개, 미리보기, 추천 secrets, 자동 실행 명령을 담고 있습니다.
- [`scripts/setup_codespaces.sh`](scripts/setup_codespaces.sh)
  패키지 설치, 샘플 코퍼스 생성, Chroma 적재를 처리합니다.
- [`scripts/start_codespaces_webapp.sh`](scripts/start_codespaces_webapp.sh)
  Streamlit 서버를 백그라운드로 띄웁니다.

### GitHub secret 관련 주의

GitHub 저장소의 `Actions secrets`는 Codespaces 런타임에 자동 주입되지 않습니다.  
Codespaces에서 실제 챗봇 응답을 쓰려면 `Codespaces secrets`에도 `GEMINI_API_KEY`를 등록해야 합니다.

## 웹 배포

이 저장소는 로컬 시연용이 아니라 공개 웹앱으로도 바로 올릴 수 있게 정리되어 있습니다.

### 권장 배포 경로

- GitHub에서 바로 실행: Codespaces
- 외부 공개 URL 배포: Render + Docker
- 대안: Streamlit Community Cloud

### 배포에 포함된 파일

- [`render.yaml`](render.yaml)
  Render Blueprint. 현재는 `free` web service 기준으로 맞춰져 있고, `GEMINI_API_KEY`는 `sync: false`로 받아 코드에 넣지 않습니다.
- [`Dockerfile`](Dockerfile)
  Streamlit 앱을 `0.0.0.0:$PORT`로 실행하는 컨테이너 설정입니다.
- [`.streamlit/config.toml`](.streamlit/config.toml)
  웹 배포용 기본 Streamlit 설정입니다.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
  push/PR마다 샘플 코퍼스 생성, 적재, 테스트를 검증합니다.
- [`docs/07_web_deployment.md`](docs/07_web_deployment.md)
  Render와 Streamlit Community Cloud 기준 상세 배포 가이드입니다.

### 중요한 점

GitHub 저장소의 `Actions secrets`에 `GEMINI_API_KEY`를 넣어도, 실행 중인 웹앱 런타임에 자동 전달되지는 않습니다.

- GitHub Actions secret: CI나 배포 워크플로에서 사용
- GitHub Codespaces secret: GitHub Codespaces 런타임에서 사용
- Render secret / Streamlit Cloud secret: 실제 서비스 런타임에서 사용

즉, GitHub에서 Codespaces로 실행하든 외부로 배포하든, 해당 런타임에 맞는 secret 저장소에 `GEMINI_API_KEY`를 따로 등록해야 합니다.

### Render 무료 플랜 기준 메모

- 무료 배포는 가능합니다.
- 다만 free web service는 15분 유휴 시 spin-down 되며, 다시 요청이 오면 약 1분 정도 재기동될 수 있습니다.
- free web service는 영구 디스크를 지원하지 않으므로, 로컬 파일 기반 Chroma 데이터와 사용량 추적 파일은 재배포·재시작·spin-down 시 초기화될 수 있습니다.
- 이 저장소는 첫 실행 시 샘플 코퍼스를 자동 적재하도록 되어 있어, MVP 시연 자체는 무료 플랜에서도 가능합니다.

## 빠른 시작

### 1. 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 환경 변수 설정

로컬 개발용으로만 `.env`를 사용합니다. 배포 환경에서는 호스팅 플랫폼의 secrets 또는 env vars를 사용합니다.

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

### 4. 로컬 개발 실행

```bash
streamlit run streamlit_app.py
```

기본 접속 주소:

- [http://127.0.0.1:8501](http://127.0.0.1:8501)

웹 배포 환경에서는 앱이 빈 저장소일 경우 샘플 코퍼스를 자동 적재합니다.

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
- 웹 배포 환경에서는 첫 실행 시 샘플 코퍼스를 자동 적재합니다.
- Render free 플랜에서는 로컬 파일 상태가 유지되지 않을 수 있습니다.

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
- [07_web_deployment.md](docs/07_web_deployment.md)

## 주의사항

- 이 앱은 공개 법규 기반 실무 참고용 도구입니다.
- 실제 승인, 징계, 복무 처리에는 최신 원문과 소속 부대 지침 확인이 필요합니다.
- 비공개 군 내부 규정과 개인정보는 현재 저장소 범위에 포함하지 않습니다.
