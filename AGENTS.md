# AGENTS.md

## Repository mission
This repo implements a **version-aware legal RAG MVP** for a policy paper on military regulation continuity.
The public demonstration corpus uses:
- 군인의 지위 및 복무에 관한 기본법
- 시행령
- 시행규칙
- revision reasons
- old/new comparison material

## Non-negotiable rules
1. Do not turn the app into a general legal-advice bot.
2. Prefer **evidence retrieval + citation cards** over fluent unsupported answers.
3. Preserve the schema in `docs/03_system_architecture.md`.
4. Keep the app runnable in **demo mode** without `LAW_API_KEY`.
5. If full collection is implemented, keep raw download and normalized jsonl separated.

## Required validations after meaningful edits
Run:
```bash
python scripts/build_sample_corpus.py
python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl
pytest
```

If UI files changed, also run:
```bash
streamlit run streamlit_app.py
```

## When to use repo-local skills
- Use `legal-rag-ingest` for corpus collection / normalization work.
- Use `streamlit-guardrails` for user limit, safety, and output UX work.
- Use `redteam-paper-mvp` before final release or major architectural changes.

## Preferred implementation style
- Small, testable functions
- Conservative error handling
- Keep prompts in dedicated files/modules
- Prefer explicit metadata over hidden assumptions