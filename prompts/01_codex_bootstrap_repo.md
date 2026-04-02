# Codex Prompt 01 — 저장소 부트스트랩

이 저장소를 기준으로 아래 작업을 수행해라.

목표:
1. `README.md`, `AGENTS.md`, `.agents/skills/`를 먼저 읽고 저장소 목적을 파악할 것.
2. 현재 구조를 깨지 말고, Python + Streamlit + Chroma + Gemini 무료 티어 기반 MVP를 유지할 것.
3. 법률자문처럼 단정하는 표현을 피하고, 모든 답변은 근거자료와 출처를 우선 제시할 것.
4. 앱 운영 제한은 기본적으로 **일일 20회 질문**, **질문 1000자 제한**으로 유지할 것.
5. 변경 전후로 `python scripts/run_smoke_checks.py`, `pytest -q`를 실행해라.

작업 원칙:
- 반복 작업에는 repo-local skills를 우선 사용하라.
- 근거 없는 기능 확장은 하지 말고, 데이터 스키마와 UI 일관성을 유지하라.
- 개정이유가 없는 사안은 ‘추정’으로 표시하라.
