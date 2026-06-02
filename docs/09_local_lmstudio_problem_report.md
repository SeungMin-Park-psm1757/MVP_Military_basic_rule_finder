# Local LM Studio Integration Problem Report

## Purpose

This report documents the local-model debugging work for the repository's LM Studio path, including:

- what was inspected
- what was patched
- which files and folders were involved
- what still fails intermittently
- why the same issue appeared to repeat

This report is intended for another AI or engineer taking over the debugging work.

## Scope

The repository has two Streamlit entry points:

1. `streamlit_app.py`
   - Public / root path
   - Not the main target for local-model debugging

2. `local/streamlit_app.py`
   - Local LM Studio experimentation path
   - This was the real target for debugging

The main local inference flow is:

- `local/streamlit_app.py`
- `src/army_reg_rag/services/answer_service.py`
- `src/army_reg_rag/llm/lm_studio_client.py`
- `src/army_reg_rag/llm/prompts.py`

## Folders Used During Debugging

- `local/`
  - Local-only Streamlit app for LM Studio

- `src/army_reg_rag/llm/`
  - LM Studio integration logic
  - Prompt construction

- `src/army_reg_rag/services/`
  - Answer orchestration

- `data/runtime/`
  - Runtime logs and diagnostic artifacts
  - New LM Studio debug artifacts are written here

- `tests/`
  - LM Studio parsing and retry tests

- `scripts/`
  - Standalone probing and validation scripts

## Files Inspected

- `local/streamlit_app.py`
- `src/army_reg_rag/services/answer_service.py`
- `src/army_reg_rag/llm/lm_studio_client.py`
- `src/army_reg_rag/llm/prompts.py`
- `src/army_reg_rag/llm/gemini_client.py`
- `src/army_reg_rag/domain/models.py`
- `tests/test_lm_studio_client.py`
- `data/runtime/*`

## Main Symptoms Observed

- LM Studio was running and GPU inference was active.
- The local app often received no final answer text.
- The UI frequently fell back to evidence-only mode.
- `gpt-oss-20b` often consumed reasoning tokens but failed to produce a final assistant answer.
- LM Studio's loaded-model inspection was inconsistent in this environment.
- The app could show `gpt-oss-20b`, but actual endpoint behavior still varied between:
  - empty final content
  - reasoning-only output
  - truncated output

## Key Root Causes

### 1. Reasoning-only model behavior

This became the primary cause.

Observed behavior:

- `gpt-oss-20b` on LM Studio sometimes returned a successful HTTP response with:
  - reasoning content present
  - final answer missing
- This happened especially on:
  - `/v1/responses`
  - sometimes `/v1/chat/completions`

This was not just a parsing bug. In many cases, the final answer field was genuinely absent.

### 2. Token budget mismatch on long app prompts

The repository prompt for local inference is much longer than a minimal direct test prompt because it includes:

- question
- intent
- short notes
- compressed evidence blocks
- formatting rules
- output section requirements

For reasoning-capable models, this led to cases where:

- most output tokens were spent on reasoning
- the final assistant message never arrived before token exhaustion

### 3. Auto-follow model detection was unreliable

LM Studio's native model listing sometimes reported empty `loaded_instances`.

That means:

- app-side automatic model tracking cannot always be trusted
- explicit model override is safer than relying on "current loaded model"

### 4. Debugging was confused by shell encoding in ad-hoc probes

Some direct one-off shell probes caused Korean text to become `????`.

That created misleading test results because:

- the model was responding to corrupted input
- the local app itself was not necessarily sending the same broken text

This made some early endpoint tests appear worse or more inconsistent than the actual app path.

## What Was Patched

### A. Local UI diagnostics

Patched file:

- `local/streamlit_app.py`

Changes:

- added small per-answer metadata display
  - latency
  - token usage
- added concise failure diagnostics in the local UI
  - model
  - endpoint
  - HTTP status
  - parsed text length
  - failure type
- added explicit `LM Studio Model Override` input in the sidebar
  - example: `gpt-oss-20b`
  - example: `exaone-4.0-1.2b`

### B. End-to-end diagnostics plumbing

Patched files:

- `src/army_reg_rag/domain/models.py`
- `src/army_reg_rag/llm/gemini_client.py`
- `src/army_reg_rag/services/answer_service.py`

Changes:

- added a `diagnostics` payload to generated answers and answer bundles
- passed diagnostics from the LM Studio client all the way to the local UI

### C. LM Studio request and parsing reliability

Patched file:

- `src/army_reg_rag/llm/lm_studio_client.py`

Changes:

- added model-specific endpoint logic
- recorded detailed attempt diagnostics:
  - endpoint used
  - model
  - HTTP status
  - finish reason
  - token usage
  - parsed text length
  - failure type
  - raw response JSON
- wrote debug artifacts to:
  - `data/runtime/lm_studio_debug/`
- added retry behavior with shorter prompts
- increased token budgets for `gpt-oss-20b`
- improved parsing coverage for:
  - `output_text`
  - `choices[0].message.content`
  - `choices[0].text`
  - `output[].content`

### D. New fallback strategy for `gpt-oss-20b`

The retry strategy was changed.

Previous behavior:

- primary: `/v1/responses`
- retry: `/v1/chat/completions`

New behavior:

- primary: `/v1/responses`
- if reasoning-only or empty: retry on native LM Studio endpoint:
  - `/api/v1/chat`

Reason:

- in direct testing, LM Studio native `/api/v1/chat` produced final assistant text in cases where OpenAI-compatible endpoints still returned reasoning-heavy or empty results

### E. Wrapped plain native-chat answers into app-friendly markdown

Because native `/api/v1/chat` often returns plain text instead of the app's preferred markdown section structure, plain final answers are now wrapped into a conservative structured markdown template instead of being discarded as "unstructured".

## New Debug Script

Added file:

- `scripts/test_lmstudio_connection.py`

Purpose:

- probe LM Studio directly outside Streamlit
- compare endpoint behavior
- print:
  - request payload
  - status code
  - raw JSON
  - parsed final text
- save results to:
  - `data/runtime/test_lmstudio_connection_results.json`

## Test Coverage Added / Updated

Patched file:

- `tests/test_lm_studio_client.py`

Coverage includes:

- response parsing
- retry behavior
- responses API path for `gpt-oss-20b`
- reasoning-only failure detection
- diagnostic artifact writing
- native chat fallback behavior

## Validation Commands Run

The required repository validations were run after the patch:

```bash
python scripts/build_sample_corpus.py
python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl
pytest
```

UI startup checks were also run:

```bash
streamlit run streamlit_app.py
streamlit run local/streamlit_app.py
```

## Current Status

### What now works better

- local failures now leave evidence
- debugging no longer depends on guessing
- the local UI shows which endpoint failed
- `gpt-oss-20b` now has a native LM Studio fallback path
- explicit model override is available in the local UI

### What still remains unstable

- `gpt-oss-20b` can still spend a large fraction of output tokens on reasoning
- final-answer generation is still model-dependent and prompt-length-sensitive
- `exaone-deep-7.8b` remains unreliable for this app flow
- LM Studio auto-follow model detection is not fully trustworthy in this environment

## Why the Issue Seemed to Repeat

The repeated failure was not caused by a single bug.

It repeated because multiple issues overlapped:

1. A valid HTTP response did not always include a final answer.
2. The app prompt was much longer than minimal direct probes.
3. Reasoning-capable models behaved differently by endpoint.
4. Some ad-hoc tests were distorted by shell encoding problems.
5. Auto-follow model detection was not stable enough to rule out model-selection ambiguity.

In other words, the bug was partly:

- endpoint behavior
- partly model behavior
- partly prompt-size sensitivity
- partly debugging-environment noise

## Most Likely Current Failure Mode

At this stage, the most likely remaining failure mode is:

- LM Studio returns a successful response
- the model spends too many tokens on internal reasoning or a long detour
- the final answer is still too weak, truncated, or not aligned with the requested structure

This is most severe with:

- `gpt-oss-20b` under long structured prompts
- `exaone-deep-7.8b`

## Recommended Next Steps

1. Always set the model explicitly in the local UI.
2. Prefer:
   - `gpt-oss-20b`
   - `exaone-4.0-1.2b`
3. Avoid:
   - `exaone-deep-7.8b`
4. When a failure happens, inspect:
   - `data/runtime/lm_studio_debug/`
   - `data/runtime/test_lmstudio_connection_results.json`
5. If instability continues, the next likely improvement would be:
   - a dedicated ultra-short prompt profile for local reasoning models
   - or a provider switch only after artifact review confirms LM Studio is the limiting factor

## Practical Handoff Summary

If another AI continues this work, start here:

- `local/streamlit_app.py`
- `src/army_reg_rag/llm/lm_studio_client.py`
- `data/runtime/lm_studio_debug/`
- `scripts/test_lmstudio_connection.py`

Do not start from the public root app unless the goal is to change the non-local experience.
