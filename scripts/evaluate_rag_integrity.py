from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from _bootstrap import ensure_project_src_on_path

ensure_project_src_on_path()

from army_reg_rag.config import load_settings
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.services.answer_service import AnswerService

INTERNAL_STATUS_TERMS = {
    "reasoning_only",
    "low_information",
    "partial_salvaged",
    "legacy_partial_summary",
    "context_size_exceeded",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _public_notice_for_backend(answer_backend: str, answer_notice: str, diagnostics: dict[str, Any]) -> str:
    failure_type = str(diagnostics.get("failure_type", "") or "").strip()
    if answer_backend == "quota_blocked":
        return answer_notice
    if answer_notice or (failure_type and failure_type != "ok"):
        return "근거 중심으로 정리해 제공합니다."
    return ""


def _contains_any(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term and term in text]


def _evaluate_case(service: AnswerService, case: dict[str, Any], *, live_lm: bool) -> dict[str, Any]:
    question = str(case["question"])
    answer = service.answer(question=question, allow_generation=live_lm)

    evidence_text = "\n".join(
        " ".join(
            part
            for part in [
                hit.chunk.law_name,
                hit.chunk.source_type,
                hit.chunk.article_no,
                hit.chunk.article_title,
                hit.chunk.text,
            ]
            if part
        )
        for hit in answer.evidence
    )
    source_types = sorted({hit.chunk.source_type for hit in answer.evidence if hit.chunk.source_type})
    law_names = sorted({hit.chunk.law_name for hit in answer.evidence if hit.chunk.law_name})
    public_notice = _public_notice_for_backend(answer.answer_backend, answer.answer_notice, answer.diagnostics)
    public_surface = "\n".join([answer.answer_markdown, public_notice])

    required_refs = list(case.get("required_refs", []))
    required_laws = list(case.get("required_laws", []))
    required_source_types = list(case.get("required_source_types", []))
    expected_intents = list(case.get("expected_intents", []))

    missing_required_refs = _contains_any(evidence_text, required_refs)
    missing_required_refs = [ref for ref in required_refs if ref not in missing_required_refs]
    missing_required_laws = _contains_any(evidence_text, required_laws)
    missing_required_laws = [law for law in required_laws if law not in missing_required_laws]
    missing_required_source_types = [source_type for source_type in required_source_types if source_type not in source_types]
    internal_terms_on_public_surface = _contains_any(public_surface, sorted(INTERNAL_STATUS_TERMS))

    checks = {
        "intent_ok": not expected_intents or answer.intent in expected_intents,
        "source_type_coverage_ok": not missing_required_source_types,
        "required_ref_coverage_ok": not missing_required_refs,
        "required_law_coverage_ok": not missing_required_laws,
        "public_surface_safe": not internal_terms_on_public_surface,
        "has_evidence": bool(answer.evidence),
        "has_answer": bool(answer.answer_markdown.strip()),
    }

    return {
        "id": case.get("id", ""),
        "question": question,
        "intent": answer.intent,
        "expected_intents": expected_intents,
        "answer_backend": answer.answer_backend,
        "source_types": source_types,
        "law_names": law_names,
        "evidence_count": len(answer.evidence),
        "missing_required_refs": missing_required_refs,
        "missing_required_laws": missing_required_laws,
        "missing_required_source_types": missing_required_source_types,
        "missing_explicit_refs": answer.diagnostics.get("missing_explicit_refs", []),
        "retrieval_slots": answer.diagnostics.get("retrieval_slots", {}),
        "public_notice": public_notice,
        "internal_status": answer.diagnostics.get("internal_status") or answer.diagnostics.get("failure_type") or "",
        "public_surface_internal_terms": internal_terms_on_public_surface,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# RAG Integrity Evaluation Report",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- live_lm: {report['live_lm']}",
        f"- store_count: {report['store_count']}",
        f"- passed: {report['passed_count']}/{report['case_count']}",
        "",
        "## Cases",
    ]
    for case in report["cases"]:
        lines.extend(
            [
                "",
                f"### {case['id']}",
                f"- passed: {case['passed']}",
                f"- intent: {case['intent']}",
                f"- backend: {case['answer_backend']}",
                f"- evidence_count: {case['evidence_count']}",
                f"- source_types: {', '.join(case['source_types'])}",
                f"- missing_required_refs: {', '.join(case['missing_required_refs']) or '-'}",
                f"- missing_required_laws: {', '.join(case['missing_required_laws']) or '-'}",
                f"- missing_required_source_types: {', '.join(case['missing_required_source_types']) or '-'}",
                f"- missing_explicit_refs: {', '.join(case['missing_explicit_refs']) or '-'}",
                f"- public_surface_internal_terms: {', '.join(case['public_surface_internal_terms']) or '-'}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval/generation integrity with golden questions.")
    parser.add_argument("--questions", default="tests/fixtures/golden_questions.jsonl")
    parser.add_argument("--output-dir", default="data/runtime/eval_reports")
    parser.add_argument("--live-lm", action="store_true", help="Call the configured answer client instead of deterministic fallback.")
    args = parser.parse_args()

    settings = load_settings()
    store = ChromaStore(settings)
    service = AnswerService(settings, store=store)
    question_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    cases = [_evaluate_case(service, case, live_lm=args.live_lm) for case in _load_jsonl(question_path)]
    report = {
        "generated_at": generated_at,
        "live_lm": bool(args.live_lm),
        "store_count": store.count(),
        "case_count": len(cases),
        "passed_count": sum(1 for case in cases if case["passed"]),
        "cases": cases,
    }

    json_path = output_dir / f"rag_integrity_{generated_at}.json"
    md_path = output_dir / f"rag_integrity_{generated_at}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown_report(md_path, report)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0 if report["passed_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
