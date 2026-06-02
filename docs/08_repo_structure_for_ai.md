# Repo Structure Summary for AI Agents

This file is a practical orientation note for another AI or coding agent.
It explains what the repository currently does, which files matter most, and
where the main moving parts live.

## 1. Project Purpose

This repository implements a **version-aware legal RAG MVP** for a policy paper
about continuity of military rule interpretation and organizational memory.

The public demo corpus is centered on:

- `군인의 지위 및 복무에 관한 기본법`
- its `시행령`
- its `시행규칙`
- revision reasons
- old/new comparison material
- historical links to `군인복무규율`

This is **not** intended to become a general legal-advice bot.
The core design principle is:

- retrieve grounded evidence
- summarize conservatively
- show source links / evidence cards
- avoid unsupported fluent answers

## 2. Top-Level Layout

Important top-level paths:

- `streamlit_app.py`
  - Main web app for the default/public path.
  - Uses the shared `src/army_reg_rag` package.
  - Historically oriented toward the Gemini-backed app flow.

- `local/streamlit_app.py`
  - Local-only Streamlit app.
  - This is the main path for LM Studio / local model experiments.
  - Includes local connection probing, per-turn DOCX export, and local answer metadata UI.

- `src/army_reg_rag/`
  - Main Python package.
  - Holds retrieval, LLM clients, domain models, ingestion helpers, config loading, and DOCX export logic.

- `scripts/`
  - Corpus build / ingest / normalization utilities.
  - Also contains startup/export helper scripts.

- `data/`
  - Runtime data, Chroma persistence, processed corpora, raw downloads, sample corpus, logs.

- `docs/`
  - Architecture notes, research/paper-facing docs, deployment notes, draft chapter exports.

- `tests/`
  - Unit tests for retrieval, prompts, LM Studio client behavior, DOCX export, corpus pipeline, etc.

- `config/settings.yaml`
  - Main repo configuration.

## 3. Core Package Structure

### `src/army_reg_rag/domain/`

- `models.py`
  - Core dataclasses:
    - `DocumentChunk`
    - `SearchHit`
    - `RouteDecision`
    - `AnswerBundle`

These are the basic objects passed across retrieval, generation, and UI layers.

### `src/army_reg_rag/retrieval/`

- `router.py`
  - Lightweight query-intent routing.
  - Distinguishes things like:
    - current rule lookup
    - explain change / history
    - practical guidance

- `chroma_store.py`
  - Main retrieval backend wrapper.
  - Uses Chroma when available.
  - Falls back to a JSON-based local store if Chroma or embeddings are unavailable.
  - Also contains a lightweight embedder fallback path.

### `src/army_reg_rag/services/`

- `answer_service.py`
  - Main orchestration layer.
  - Flow:
    1. route question
    2. retrieve evidence
    3. call the active LLM client
    4. return an `AnswerBundle`
  - Also contains retrieval heuristics for:
    - law alias expansion
    - history/transition questions
    - topic keyword expansion

- `ingest_service.py`
  - Takes normalized JSONL input and writes into the store.

### `src/army_reg_rag/llm/`

- `gemini_client.py`
  - Shared answer-generation base logic and fallback behavior.
  - Also owns much of the grounded fallback summarization logic.

- `lm_studio_client.py`
  - Local LM Studio integration.
  - Important recent behavior:
    - model-specific handling for small/medium local models
    - `gpt-oss` routed through LM Studio `responses` API
    - chat-completions fallback for other local models
    - detection of “reasoning-only / no final answer” payloads
  - This file is currently the main integration hotspot for local inference issues.

- `prompts.py`
  - Prompt assembly.
  - Includes:
    - shared evidence formatting
    - compact local prompt builder
    - local system prompt

- `usage_tracker.py`
  - Token/request usage tracking persisted under `data/runtime/`.

### `src/army_reg_rag/utils/`

- `io.py`
  - JSONL / file helpers.

- `quota.py`
  - Quota-related helpers.

- `runtime_config.py`
  - Environment/runtime override access helpers.

### `src/army_reg_rag/`

- `config.py`
  - Loads `config/settings.yaml` and merges runtime overrides.

- `export_docx.py`
  - Converts conversation history into DOCX.
  - Used by the local app’s “DOCX export” feature.

## 4. UI Entry Points

## `streamlit_app.py` (root app)

Purpose:

- public/demo app
- shared retrieval/evidence UI
- default app path documented in `README.md`

Characteristics:

- uses `src/army_reg_rag` shared package
- grouped evidence cards
- quota panel
- sample corpus auto-load path
- originally centered on the default remote/gemini-style flow

## `local/streamlit_app.py` (local app)

Purpose:

- local RAG UX for LM Studio experiments
- local DOCX export
- local answer timing/token display
- local connection probing

Characteristics:

- loads `.env` from both repo root and `local/`
- creates `AnswerService` with `LMStudioAnswerClient`
- shows connection state in sidebar
- stores local per-answer metadata like:
  - `model_usage`
  - `answer_latency_ms`
  - `answer_token_usage`

Current note:

- this file is the real target when debugging local model behavior
- if a user reports “the local app is failing”, start here, not in the root app

## 5. Data Layout

### `data/raw/`

Raw public-law downloads and source artifacts.

Important subpath:

- `data/raw/public_law/`
  - raw source snapshots for:
    - basic law
    - decree
    - rule
    - military service regulation
    - reasons / revision docs

### `data/processed/`

- normalized production-like corpus output
- key file:
  - `data/processed/law_corpus.jsonl`

### `data/sample/`

- small demo/sample corpus
- key file:
  - `data/sample/processed/sample_documents.jsonl`

### `data/chroma/`

- persistent Chroma data
- also contains `_fallback_store.json` for JSON fallback behavior

### `data/runtime/`

Runtime artifacts and logs, for example:

- usage trackers
- normalization/build reports
- temporary debug files
- Streamlit startup logs

This directory is useful when debugging local behavior.

## 6. Scripts

Most important scripts:

- `scripts/build_sample_corpus.py`
  - Builds the small sample/demo JSONL corpus.

- `scripts/ingest_to_chroma.py`
  - Ingests a JSONL corpus into the configured Chroma collection.

- `scripts/build_public_corpus.py`
  - Builds the public-law corpus from processed/raw inputs.

- `scripts/normalize_raw_to_jsonl.py`
  - Normalizes raw collected materials into JSONL.

- `scripts/download_sources_from_manifest.py`
  - Pulls source files listed in the manifest.

- `scripts/fetch_law_open_api.py`
  - Open API-related fetch helper.

- `scripts/run_smoke_checks.py`
  - Smoke-check helper.

Batch helpers:

- `run_webapp.bat`
- `run_local_webapp.bat`
- `setup_local_webapp.bat`

## 7. Docs

Important docs:

- `docs/02_final_plan.md`
  - research/paper framing

- `docs/03_system_architecture.md`
  - architecture summary
  - important because repo rules explicitly say to preserve this schema

- `docs/04_evaluation_framework.md`
  - evaluation framing

- `docs/05_future_internal_rule_ingest_plan.md`
  - future extension direction for internal rules

- `docs/07_web_deployment.md`
  - deployment notes

- `docs/worked.pdf`
  - MVP/paper-related working material

## 8. Test Suite

Key tests include:

- `tests/test_lm_studio_client.py`
  - local-model behavior
  - response parsing
  - retry behavior
  - `gpt-oss` responses API path
  - reasoning-only failure detection

- `tests/test_prompts.py`
  - prompt structure expectations

- `tests/test_answer_service.py`
  - retrieval behavior
  - history-law linking

- `tests/test_export_docx.py`
  - DOCX export behavior

- `tests/test_public_law_pipeline.py`
  - corpus pipeline behavior

## 9. Current Execution Model

The repository currently has **two operational UI paths**:

1. Root app:
   - `streamlit_app.py`
   - shared/public path

2. Local app:
   - `local/streamlit_app.py`
   - local LM Studio experimentation path

The shared retrieval and answer pipeline is:

1. question enters Streamlit app
2. `AnswerService.retrieve(...)` selects intent and evidence
3. active client generates an answer or fallback summary
4. UI renders markdown answer + evidence cards + optional DOCX export

## 10. Current Local Inference Status

This is important for any future AI agent working on local inference:

- `gpt-oss-20b`
  - partially supported through LM Studio `responses` API
  - reasoning effort is intentionally lowered

- `exaone-deep-7.8b`
  - currently problematic in LM Studio
  - observed behavior: consumes reasoning tokens and sometimes fails to produce a final answer
  - the client now explicitly detects this as a “reasoning-only” failure mode

- `exaone-4.0-1.2b`
  - smaller and weaker, but can still return final text in some direct tests

- `Ollama`
  - not implemented yet in the repo at the time of writing this summary
  - if LM Studio remains unstable, Ollama is a reasonable next integration target

## 11. Commands an AI Agent Should Know First

Useful validation commands:

```bash
python scripts/build_sample_corpus.py
python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl
pytest
```

Root app:

```bash
streamlit run streamlit_app.py
```

Local app:

```bash
streamlit run local/streamlit_app.py
```

## 12. Practical Guidance for the Next AI

If the task is about:

- corpus / normalization
  - start in `scripts/` and `src/army_reg_rag/corpus/`

- retrieval quality
  - start in `src/army_reg_rag/services/answer_service.py`
  - then inspect `src/army_reg_rag/retrieval/router.py`

- Chroma / embeddings
  - start in `src/army_reg_rag/retrieval/chroma_store.py`

- local model failures
  - start in `local/streamlit_app.py`
  - then inspect `src/army_reg_rag/llm/lm_studio_client.py`

- prompt wording / answer structure
  - start in `src/army_reg_rag/llm/prompts.py`

- DOCX export
  - start in `src/army_reg_rag/export_docx.py`
  - and `local/streamlit_app.py`

## 13. Important Constraint Summary

- Do not turn the app into a general legal-advice bot.
- Prefer evidence retrieval + citation cards over fluent unsupported answers.
- Preserve the schema expected by `docs/03_system_architecture.md`.
- Keep demo mode runnable without `LAW_API_KEY`.
- Keep raw downloads and normalized JSONL separated.

This summary should be treated as a quick operational map, not as a full specification.
