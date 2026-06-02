# Current Context Handoff

This file summarizes the active project context as of 2026-04-21 so another AI agent or a future session can continue without relying on the full chat history.

## Project Goal

This repository implements a version-aware legal RAG MVP for a Korean policy paper on military administrative continuity and organizational memory.

The paper direction is not "build a chatbot first." The core argument is that military regulation work suffers from organizational memory loss when regulation changes, revision reasons, interpretation context, and handover knowledge are not preserved across personnel turnover. The MVP should demonstrate a knowledge-management/archive structure that preserves:

- current legal articles,
- historical versions,
- revision reasons,
- old/new comparison material,
- source links,
- metadata such as promulgation date and effective date,
- and evidence cards that let users verify answers.

The local app is used as a demonstration that this structure can work in a disconnected or constrained military network environment without requiring a central GPU server.

## Repository Paths

The repository has two UI paths:

- `streamlit_app.py`: root/public path.
- `local/streamlit_app.py`: local LM Studio experimentation path.

The current local-model debugging target is:

- `local/streamlit_app.py`
- `src/army_reg_rag/services/answer_service.py`
- `src/army_reg_rag/llm/lm_studio_client.py`
- `src/army_reg_rag/llm/prompts.py`
- `src/army_reg_rag/llm/gemini_client.py`
- `src/army_reg_rag/retrieval/router.py`

Useful runtime artifacts:

- `data/runtime/lm_studio_debug/`
- `data/runtime/eval_reports/`
- Streamlit startup logs from the terminal.

## Non-Negotiable Direction

The app must not become a general legal-advice bot.

Preferred behavior:

- Evidence-first answers.
- Clear citation cards.
- Conservative uncertainty handling.
- Explicit separation between official legal grounds and practical/reference material.
- No unsupported fluent answers.
- Internal debug state must not leak into the user-facing main UI.

Public-facing phrases should be simple and safe:

- `요약 생성`
- `근거 중심 정리`
- `근거 기반 응답`

Internal statuses should stay inside diagnostics/debug:

- `reasoning_only`
- `low_information`
- `partial_salvaged`
- `garbled_text`
- `context_size_exceeded`
- `validation_failed`
- `sentence_support`
- `statute_linkage`
- `unsupported_addition`

## Current Model Setup

The user is testing LM Studio local models:

- `gpt-oss-20b`
- `exaone-4.0-1.2b`
- sometimes `exaone-deep-7.8b`

Known behavior:

- `gpt-oss-20b` often works better through LM Studio `/v1/responses`, but can spend output tokens on reasoning and return no visible final answer.
- `gpt-oss-20b` can also produce garbled `????????` output in some cases.
- `exaone-4.0-1.2b` is weak and often returns `현재 자료 기준으로 확인되는 내용은 제한적입니다.`
- `exaone-deep-7.8b` was problematic because it can consume reasoning tokens and fail to produce a final answer.

The current design should support model-agnostic handling. It should not be limited to EXAONE 1.2B. The LM Studio client now uses model capability profiles:

- `small_local`
- `medium_local`
- `default_local`
- `gpt_oss_reasoning`

These profiles are diagnostic and policy-oriented. They are not meant to hard-code behavior only for EXAONE.

## Repeated Error History

The user repeatedly saw these problems:

1. Empty LM Studio responses despite GPU activity.
2. `reasoning_only` payloads where hidden reasoning existed but final text was empty.
3. `???????` garbled text output.
4. `Context size has been exceeded` from LM Studio.
5. Overly generic fallback text such as `근거 중심으로 정리해 제공합니다.`
6. Low-value answers like `현재 자료 기준으로 확인되는 내용은 제한적입니다.`
7. Core conclusions that did not answer the actual question.
8. Sentences cut off mid-way.
9. Incorrect temporal claims, especially around whether 신고자 보호 was added in 2020.
10. Missing explicit articles such as `제44조` when the user directly asked about them.

The key insight is that forcing the LLM to complete the whole answer is not the right goal. The better goal is:

> Use LLMs for concise interpretation where they are useful, but ensure deterministic retrieval and evidence-based fallback always produce a complete answer.

## Current Architecture Direction

The desired answer pipeline is:

1. Intent classification.
2. Retrieval planning.
3. Evidence slot construction.
4. Deterministic answer skeleton.
5. Short LLM summary/interpretation.
6. Validation.
7. Fallback if validation fails.
8. UI rendering with internal diagnostics separated from public answer text.

Core intents:

- `current_rule`
- `search`
- `explain_change`
- `compare_versions`
- `practical`
- `hybrid`
- `mixed_complex`

Retrieval should prioritize source types by intent:

- Current rule/search: current law articles first.
- Explain change: revision reasons and history first, current articles as support.
- Compare versions: old/new comparison and historical documents first.
- Practical reference: current articles plus revision reasons.
- Hybrid: revision reasons plus current articles plus history.

Evidence slots should conceptually separate:

- `current_articles`
- `revision_reasons`
- `history_docs`
- `compare_docs`
- `reference_docs`

## Important Korean Legal/MVP Content

The paper and MVP currently focus on:

- `군인의 지위 및 복무에 관한 기본법`
- `군인의 지위 및 복무에 관한 기본법 시행령`
- `군인의 지위 및 복무에 관한 기본법 시행규칙`
- `군인복무규율`
- revision reasons,
- historical versions,
- old/new comparison material.

Recent corpus expansion direction:

- `군인사법`
- `군인 징계령`
- `군인 징계령 시행규칙`

This expansion is important because questions about disciplinary types and procedures cannot be fully answered from only the basic service law corpus.

## Important Answering Principles

For the question:

> 현행 「군인의 지위 및 복무에 관한 기본법」상 신고자 보호 규정을 조문번호와 함께 제시하고, 제44조의 비밀보장과 어떤 관계인지 3문장으로 설명해줘.

The answer should not collapse into generic fallback. It should retrieve both `제44조` and `제45조` if available.

Expected answer direction:

- `제44조` is the confidentiality protection layer.
- `제45조` is the protection against disadvantage layer.
- They are connected because confidentiality reduces exposure risk, while protection provisions address retaliation or disadvantage after reporting.

If `제44조` is missing from evidence, the app should explicitly diagnose `missing_explicit_refs` internally and give a conservative evidence-based answer, not pretend the relationship is fully proven.

For questions about `군인복무규율` abolished after 2016 and transition to the Basic Act:

- Explain the shift from older service/discipline-oriented regulation to a rights-protection and reporting/protection framework.
- Do not claim without evidence that only 신고의무 existed in 2016 and 신고자 보호 was added in 2020.
- Prefer conservative expressions:
  - `2016년 제정 단계에서 신고의무와 신고자 보호 취지가 함께 제도화되었고, 이후 개정을 거치며 관련 체계가 보완되었다.`
  - `확인된 근거 범위에서는`
  - `현재 자료 기준으로 보면`

For disciplinary types/procedures:

- If `군인사법`, `군인 징계령`, and related implementing rules are not in the retrieved evidence, say the current corpus cannot fully confirm that part.
- If the disciplinary law corpus is present and retrieved, answer based on those articles.

## Recent Code Changes Already Made

Recent modifications include:

- `src/army_reg_rag/llm/prompts.py`
  - Hybrid questions now prioritize revision reasons plus current law articles.
  - Local prompt rules discourage returning only `제한적입니다` when evidence exists.
  - Evidence counts are tuned by profile and intent to reduce LM Studio context-size errors.

- `src/army_reg_rag/llm/lm_studio_client.py`
  - Added/strengthened handling for:
    - `low_information`
    - `reasoning_only`
    - `garbled_text`
    - `context_size_exceeded`
    - `incomplete_sentence`
    - validation failures
  - Internal failure states are hidden from the main user UI.
  - Debug artifacts are saved under `data/runtime/lm_studio_debug/`.
  - Added `model_profile` diagnostics so future models can be compared by profile, not just exact model name.

- `src/army_reg_rag/llm/gemini_client.py`
  - Deterministic fallback/conclusion logic improved for reporter protection and confidentiality linkage.
  - Revision-reason summaries made more complete to reduce cut-off sentence artifacts.

- `src/army_reg_rag/retrieval/router.py`
  - Routing improved for hybrid questions that ask for current articles together with revision reasons/history.

- Tests added or updated:
  - `tests/test_lm_studio_client.py`
  - `tests/test_local_answer_quality.py`
  - `tests/test_router.py`
  - `tests/test_prompts.py`

Latest validation after recent model-profile generalization:

```powershell
python scripts/build_sample_corpus.py
python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl
python -m pytest -q
```

Result:

- `91 passed`
- Chroma collection count reported as `591`

## Current User Concern

The user said:

> 지금 수정이 1.2b에 한정되는 느낌이야. 우선은 범위를 좁히는건 좋은데 다른 llm 쓸 때에도 참고되었으면 해.

This means:

- The user accepts narrowing scope temporarily.
- But they want the architecture to remain useful for other local LLMs.
- Avoid making fixes that only work for `exaone-4.0-1.2b`.

Current response direction:

- Explain that the safety checks are now model-agnostic.
- Endpoint selection and token budgets can still be model-specific.
- Debugging should compare models through `model_profile`.

## LM Studio Settings Guidance

Previously reviewed LM Studio settings:

- Context length: 8192 looked acceptable.
- GPU offload enabled.
- Flash attention enabled.
- KV cache offload enabled.
- Prompt template not manually set, which is good.
- Response length/max output is mostly controlled by app request payload, not only LM Studio UI.

If errors continue:

- Restart LM Studio after changing model settings.
- Load exactly one LLM at a time.
- Confirm the local app auto-detects the intended model.
- Check latest JSON artifact in `data/runtime/lm_studio_debug/`.

## Important Caveat About Encoding

Some PowerShell output displays Korean source text as mojibake. This does not necessarily mean the file is corrupted.

When checking Korean source or tests, prefer reading with explicit UTF-8:

```powershell
python - <<'PY'
from pathlib import Path
print(Path("path/to/file.py").read_text(encoding="utf-8")[:1000])
PY
```

Avoid judging file integrity from `Get-Content` output alone.

## Current Dirty Worktree Caveat

There are many modified and untracked files, some created by prior work or other AI tools. Do not revert unrelated changes.

Relevant recently touched files include:

- `src/army_reg_rag/llm/lm_studio_client.py`
- `src/army_reg_rag/llm/prompts.py`
- `src/army_reg_rag/llm/gemini_client.py`
- `src/army_reg_rag/retrieval/router.py`
- `src/army_reg_rag/services/answer_service.py`
- `local/streamlit_app.py`
- `streamlit_app.py`
- `tests/test_lm_studio_client.py`
- `tests/test_local_answer_quality.py`
- `tests/test_router.py`
- `tests/test_prompts.py`
- `docs/09_local_lmstudio_problem_report.md`
- `docs/10_current_program_and_repeated_error_report.md`

Also present:

- `docs/worked.pdf`
- exported HTML outputs from local app in the user Downloads folder.

## Recommended Next Steps

1. Run the local app with one model loaded at a time.
2. Test four fixed golden questions:
   - Current rule: `현행 「군인의 지위 및 복무에 관한 기본법 시행령」 기준으로 군인의 휴가 종류를 조문번호와 함께 나열해줘.`
   - Reporter protection: `현행 「군인의 지위 및 복무에 관한 기본법」상 신고자 보호 규정을 조문번호와 함께 제시하고, 제44조의 비밀보장과 어떤 관계인지 3문장으로 설명해줘.`
   - Transition/history: `군인복무규율이 폐지되고 「군인의 지위 및 복무에 관한 기본법」 체계가 도입되면서, 과거의 군기·복무 중심 규정이 기본권 보호와 신고자 보호 중심으로 어떻게 재편되었는지 개정 이유와 현행 조문을 함께 근거로 설명해줘.`
   - Discipline complex: `군인복무규율이 2016년 폐지된 뒤, 복무·군기·권리구제 사항은 「군인의 지위 및 복무에 관한 기본법」 체계로 어떻게 재편되었고, 징계의 종류·절차는 현재 어떤 법령이 담당하는지 구분해서 설명해줘.`
3. For each model, compare:
   - final answer usefulness,
   - evidence card match,
   - `model_profile`,
   - `failure_type`,
   - `validator`,
   - latency and token usage.
4. If one model repeatedly fails, tune its request profile rather than changing shared retrieval/validation logic.

## Paper Implication

The MVP should be described in the paper as:

- A preliminary demonstration, not a complete legal AI system.
- Evidence that local-office execution is feasible under constrained or disconnected network conditions.
- Evidence that retrieval/data structure quality matters more than model size alone.
- A reason to prioritize document-management architecture and evidence traceability before scaling the chatbot layer.

Recommended paper phrasing:

> 예비 시연 결과, 모델 크기 자체보다 구조화된 지식기반과 검색 정책이 응답의 유용성을 더 크게 좌우하는 것으로 관찰되었다. 특히 로컬 LLM이 충분한 답변을 생성하지 못하는 경우에도 근거 카드와 규칙 기반 요약을 통해 최소한의 검증 가능한 답변 구조를 유지할 수 있었다. 이는 군 단독망 환경에서 중앙 GPU 서버만을 전제로 하기보다, 부서 단위 로컬 실행과 근거 중심 문서관리체계를 함께 검토할 필요가 있음을 시사한다.

