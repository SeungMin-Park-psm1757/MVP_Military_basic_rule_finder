# 로컬 실행 가이드

이 로컬 앱은 `LM Studio + 로컬 모델`로 동작합니다.

## 핵심 동작

- 앱은 `LM Studio`의 공식 native API(`/api/v1/models`)를 확인해 현재 로드된 LLM을 읽습니다.
- `LM Studio`에 LLM이 하나만 로드되어 있으면 그 모델을 자동으로 사용합니다.
- 여러 개의 LLM이 동시에 로드되어 있으면 어떤 모델이 GUI에서 현재 선택되어 있는지 공식 API에 문서화된 방식으로 구분할 수 없어, 자동 추적 모드는 사용되지 않습니다.
- 이 경우에는 LLM을 하나만 남기거나 `local/.env`의 `LM_STUDIO_MODEL`에 강제 모델명을 넣으면 됩니다.

## 준비

1. `LM Studio`를 실행합니다.
2. 질문에 사용할 LLM을 하나만 로드합니다.
3. 아래 주소가 응답하는지 확인합니다.

```text
http://127.0.0.1:1234/v1/models
```

## 설정

필요하면 `local/.env.example`을 복사해 `local/.env`를 만듭니다.

```powershell
copy local\.env.example local\.env
```

기본값:

- `LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1`
- `LM_STUDIO_MODEL=`  (비워두면 자동 추적)
- `LM_STUDIO_TIMEOUT_SECONDS=120`

## 코퍼스 준비

```powershell
python scripts/build_sample_corpus.py
python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl
```

## 실행

처음 한 번은:

```powershell
setup_local_webapp.bat
```

이후에는:

```powershell
run_local_webapp.bat
```

또는 직접:

```powershell
streamlit run local/streamlit_app.py
```

기본 주소:

```text
http://127.0.0.1:8502
```

## 참고

- 로컬 버전은 질문 횟수 제한이 없습니다.
- 답변 우상단의 `DOCX 내보내기`로 현재 대화를 `.docx`로 저장할 수 있습니다.
- LM Studio가 준비되지 않으면 앱은 근거 요약 모드로 자동 전환됩니다.
