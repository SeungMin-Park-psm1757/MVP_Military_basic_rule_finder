# Codex Prompt 02 — 공개 법령 수집·정규화

다음 작업을 수행해라.

작업:
1. `data_manifests/public_law_sources.csv`의 공개 링크를 검토하고, 필요한 경우 추가적인 공개 링크를 보완해라.
2. `scripts/download_sources_from_manifest.py` 또는 `scripts/fetch_law_open_api.py`를 사용해 원문을 확보해라.
3. 수집 결과를 `data/raw/` 아래에 저장하고, 메타데이터 파일을 남겨라.
4. `scripts/normalize_raw_to_jsonl.py`를 개선해 `law_text`, `revision_reason`, `old_new_comparison`, `history_note`로 분류되도록 해라.
5. 결과를 `data/processed/law_corpus.jsonl`에 저장하고 `scripts/ingest_to_chroma.py`로 적재해라.
6. 최소 5개의 질의 예시를 실행해 검색 품질을 점검하고, 어떤 질의에서 어떤 자료유형이 잘 잡히는지 짧게 기록해라.

품질 기준:
- 조문번호, 시행일, 개정형태, source_url이 최대한 보존되어야 한다.
- 현행조문과 개정이유가 섞이지 않도록 source_type을 명확히 하라.
- 실패한 다운로드나 API 호출은 조용히 무시하지 말고 로그를 남겨라.
