---
name: legal-rag-ingest
description: 공개 법령, 개정이유, 신구비교, 연혁 자료를 raw/processed/chroma로 정리할 때 사용하는 수집·정규화 스킬
---

# 목적
버전관리형 법령 RAG의 핵심은 **현행조문 / 개정이유 / 신구비교 / 연혁**을 섞지 않고 구조화하는 것이다.  
이 스킬은 공개 법령 자료를 수집하고, 검색 가능한 JSONL 코퍼스로 정규화하는 반복 작업에 사용한다.

# 언제 사용하나
- `data_manifests/public_law_sources.csv`를 기준으로 공개 원문을 내려받을 때
- law.go.kr Open API 결과를 `data/raw/`에 저장할 때
- 여러 형식의 raw 데이터를 `law_corpus.jsonl`로 정규화할 때
- source_type 분류 정확도를 점검할 때

# 핵심 규칙
1. **raw와 processed를 절대 섞지 않는다.**
2. `source_type`은 최소한 아래 넷 중 하나여야 한다.
   - `law_text`
   - `revision_reason`
   - `old_new_comparison`
   - `history_note`
3. 조문번호, 시행일, 개정형태, source_url을 최대한 남긴다.
4. 개정이유가 없으면 임의로 생성하지 않는다.
5. 실패한 다운로드와 파싱 실패는 로그로 남긴다.

# 권장 절차
1. 매니페스트 점검
2. `scripts/download_sources_from_manifest.py` 실행
3. 필요시 `scripts/fetch_law_open_api.py` 실행
4. raw 결과 확인
5. `scripts/normalize_raw_to_jsonl.py`로 정규화
6. `scripts/ingest_to_chroma.py`로 적재
7. 샘플 질의로 검증

# 품질 체크리스트
- 현행조문이 개정이유로 잘못 들어가 있지 않은가?
- 시행일이 비어 있는 데이터가 과도하게 많지 않은가?
- 동일 문서가 중복 chunk로 과도하게 쪼개지지 않았는가?
- source_url이 모두 빠져 있지는 않은가?
- history_note를 law_text처럼 오인하지 않는가?
