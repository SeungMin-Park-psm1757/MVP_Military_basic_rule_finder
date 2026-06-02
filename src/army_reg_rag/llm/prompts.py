from __future__ import annotations

from army_reg_rag.domain.models import SearchHit

SOURCE_TYPE_LABELS = {
    "law_text": "현행 조문",
    "revision_reason": "개정이유",
    "old_new_comparison": "신구 비교",
    "history_note": "연혁 자료",
}

EVIDENCE_CHAR_LIMITS = {
    "law_text": 320,
    "revision_reason": 220,
    "old_new_comparison": 220,
    "history_note": 180,
}

LOCAL_EVIDENCE_CHAR_LIMITS = {
    "law_text": 180,
    "revision_reason": 120,
    "old_new_comparison": 120,
    "history_note": 100,
}

SYSTEM_PROMPT = """당신은 공개 법령 근거 기반의 군 복무 규정 RAG 보조 도구입니다.
아래 원칙을 반드시 지키세요.
1) 사용자의 질문을 직접 뒷받침하는 근거가 없으면 일반론으로 메우지 마세요.
2) 법률, 시행령, 시행규칙, 개정이유, 신구 비교, 연혁 자료를 구분해서 설명하세요.
3) 과거 내용이나 연혁 질문에서 근거가 군인복무규율로 이어지면 현재 기본법 및 관련 시행령·시행규칙이 그 체계에서 발전한 관계임을 명시하세요.
4) 법률 자문처럼 단정하지 말고, 근거 중심 설명과 원문 확인 안내를 제공하세요.
5) 근거가 부족하면 "현재 자료 기준으로"라고 분명히 밝히세요.
6) 빠른 맥락 메모는 검색 보조용이므로 최종 판단 근거로 사용하지 마세요.
7) 최종 답변의 모든 문장은 아래 근거 블록에서 다시 확인되는 내용만 사용하세요.
8) 조문 조각이나 목록 항목을 그대로 복사하지 말고 자연스러운 한국어 완결문으로 다시 써 주세요.
9) 작은 로컬 모델도 처리할 수 있도록 짧고 분명한 문장으로 정리하세요.
"""

LOCAL_SYSTEM_PROMPT = """당신은 공개 법령 근거만 정리하는 한국어 보조 모델입니다.
추측하지 말고, 근거가 확인되는 내용만 짧고 자연스럽게 설명하세요.
항목 조각을 그대로 복사하지 말고 완결된 한국어 문장으로 바꾸세요.
형식이 어렵더라도 제목은 그대로 유지하고, 문장은 짧게 쓰세요.
"""

LOCAL_PLAIN_SYSTEM_PROMPT = """당신은 공개 법령 근거만 짧게 정리하는 한국어 보조 모델입니다.
추측하지 말고, 근거에서 직접 확인되는 내용만 답하세요.
제목, 번호 목록, 마크다운, 코드블록은 쓰지 말고 평문만 출력하세요.
조문을 그대로 복사하지 말고 자연스러운 한국어 완결문으로 바꾸세요.
물음표를 길게 반복하거나 ??? 같은 출력을 하지 마세요.
확실하지 않으면 "현재 자료 기준으로 확인되는 내용은 제한적입니다."라고 답하세요.
"""

DISCIPLINE_PROMPT_KEYWORDS = [
    "징계",
    "징계조치",
    "징계혐의자",
    "가혹행위",
    "사적 제재",
    "신고",
    "신고자 보호",
]


def _collapse_whitespace(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def _compact_excerpt(text: str, *, limit: int) -> str:
    compact = _collapse_whitespace(text)
    if len(compact) <= limit:
        return compact

    candidate = compact[:limit]
    boundary = max(candidate.rfind(". "), candidate.rfind("다. "), candidate.rfind("; "))
    if boundary >= int(limit * 0.55):
        return candidate[: boundary + 1].rstrip(" ,")

    truncated = compact[: limit - 1]
    last_space = truncated.rfind(" ")
    if last_space >= 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,") + "..."


def _preferred_evidence_text(hit: SearchHit) -> str:
    chunk = hit.chunk
    extra = chunk.extra or {}
    return str(extra.get("display_text") or extra.get("summary_text") or chunk.text or "").strip()


def _article_ref(hit: SearchHit) -> str:
    chunk = hit.chunk
    parts = [part for part in [chunk.article_no, chunk.article_title] if part]
    return " ".join(parts).strip() or "관련 범위"


def _scope_label(hit: SearchHit) -> str:
    chunk = hit.chunk
    extra = chunk.extra or {}
    scope = str(extra.get("scope", "")).strip()
    return scope or chunk.version_label or _article_ref(hit)


def _is_timeline_question(question: str) -> bool:
    return any(keyword in question for keyword in ["연혁", "변천", "변천사", "흐름", "이어졌", "달라졌", "넘어오면서"])


def _is_discipline_question(question: str) -> bool:
    return any(keyword in question for keyword in DISCIPLINE_PROMPT_KEYWORDS)


def _select_local_evidence(
    question: str,
    intent: str,
    evidence: list[SearchHit],
    *,
    count: int,
) -> list[SearchHit]:
    if len(evidence) <= count:
        return list(evidence)

    preferred_source_order: list[str] = []
    if intent == "hybrid":
        preferred_source_order = ["revision_reason", "law_text", "law_text", "history_note", "old_new_comparison"]
    elif intent == "explain_change" and _is_timeline_question(question):
        preferred_source_order = ["revision_reason", "history_note", "history_note", "law_text", "old_new_comparison"]
    elif intent == "search" and _is_discipline_question(question):
        preferred_source_order = ["law_text", "law_text", "history_note", "revision_reason"]

    if not preferred_source_order:
        return evidence[:count]

    selected: list[SearchHit] = []
    used_ids: set[str] = set()
    for source_type in preferred_source_order:
        for hit in evidence:
            if hit.chunk.id in used_ids or hit.chunk.source_type != source_type:
                continue
            selected.append(hit)
            used_ids.add(hit.chunk.id)
            break
        if len(selected) >= count:
            return selected[:count]

    for hit in evidence:
        if hit.chunk.id in used_ids:
            continue
        selected.append(hit)
        used_ids.add(hit.chunk.id)
        if len(selected) >= count:
            break
    return selected[:count]


def _output_format(intent: str, question: str) -> str:
    if intent == "explain_change" and _is_timeline_question(question):
        return (
            "## 답변 개요\n"
            "### 핵심 결론\n"
            "## 연혁 정리\n"
            "### 시기별 변화\n"
            "## 현재 체계\n"
            "### 현재 체계와 연결\n"
            "## 근거 안내\n"
            "### 확인 방법"
        )
    if intent == "search":
        return (
            "## 답변 개요\n"
            "### 핵심 결론\n"
            "## 세부 정리\n"
            "### 주요 규정\n"
            "### 실무 참고\n"
            "## 근거 안내\n"
            "### 확인 방법"
        )
    if intent == "explain_change":
        return (
            "## 답변 개요\n"
            "### 핵심 결론\n"
            "## 세부 정리\n"
            "### 주요 개정 이유\n"
            "### 실제 제도 변화\n"
            "### 해석 시사점\n"
            "## 근거 안내\n"
            "### 확인 방법"
        )
    if intent == "practical":
        return (
            "## 답변 개요\n"
            "### 핵심 결론\n"
            "## 실무 정리\n"
            "### 실무적으로 보면\n"
            "### 주의사항\n"
            "## 근거 안내\n"
            "### 확인 방법"
        )
    return (
        "## 답변 개요\n"
        "### 핵심 결론\n"
        "## 세부 정리\n"
        "### 주요 개정 이유\n"
        "### 실무적으로 보면\n"
        "## 근거 안내\n"
        "### 확인 방법"
    )


def _context_memo_lines(question: str, intent: str, evidence: list[SearchHit]) -> list[str]:
    if not evidence:
        return []

    if intent == "search":
        return [
            f"- 직접 근거: {hit.chunk.law_name} | {_article_ref(hit)} | {_compact_excerpt(_preferred_evidence_text(hit), limit=110)}"
            for hit in evidence[:3]
        ]

    if intent == "practical":
        return [
            f"- 실무 참고: {hit.chunk.law_name} | {_article_ref(hit)} | {_compact_excerpt(_preferred_evidence_text(hit), limit=110)}"
            for hit in evidence[:3]
        ]

    if intent == "explain_change" and _is_timeline_question(question):
        ordered_hits = sorted(
            evidence,
            key=lambda hit: hit.chunk.effective_date or hit.chunk.promulgation_date or "9999-12-31",
        )
        return [
            f"- 변화축: {hit.chunk.effective_date or hit.chunk.promulgation_date or '시기 미상'} | {hit.chunk.law_name} | {_scope_label(hit)}"
            for hit in ordered_hits[:4]
        ]

    if intent == "explain_change":
        return [
            f"- 개정 단서: {hit.chunk.law_name} | {_scope_label(hit)} | {_compact_excerpt(_preferred_evidence_text(hit), limit=100)}"
            for hit in evidence[:4]
        ]

    return [f"- 참고 단서: {hit.chunk.law_name} | {_scope_label(hit)}" for hit in evidence[:3]]


def _compact_evidence_block(
    idx: int,
    hit: SearchHit,
    *,
    limit_map: dict[str, int] | None = None,
) -> str:
    chunk = hit.chunk
    excerpt = _compact_excerpt(
        _preferred_evidence_text(hit),
        limit=(limit_map or EVIDENCE_CHAR_LIMITS).get(chunk.source_type, 220),
    )
    return (
        f"[근거 {idx}]\n"
        f"- 문서명: {chunk.law_name}\n"
        f"- 자료유형: {SOURCE_TYPE_LABELS.get(chunk.source_type, chunk.source_type)}\n"
        f"- 조문/범위: {_article_ref(hit)}\n"
        f"- 공포일: {chunk.promulgation_date}\n"
        f"- 시행일: {chunk.effective_date}\n"
        f"- 개정형태: {chunk.revision_kind}\n"
        f"- 링크: {chunk.source_url}\n"
        f"- 압축 발췌: {excerpt}\n"
    )


def _instruction_for_intent(intent: str, question: str) -> str:
    guidance = {
        "search": "현행 규정 질문입니다. 현재 기준을 먼저 답하고, 필요할 때만 관련 배경을 짧게 덧붙이세요.",
        "explain_change": "개정 이유 또는 변화 흐름 질문입니다. 무엇이 바뀌었는지, 왜 바뀌었는지, 현재 체계와 어떻게 이어지는지 순서대로 설명하세요.",
        "practical": "실무 참고 질문입니다. 확인 순서와 주의사항을 먼저 정리하세요.",
        "hybrid": "개정 배경과 실무 참고가 함께 필요한 질문입니다. 변화 흐름과 현재 적용상 유의점을 함께 정리하세요.",
    }.get(intent, "질문에 직접 답하세요.")

    if intent == "explain_change" and _is_timeline_question(question):
        guidance += " 반드시 '과거 규정 -> 제도 전환 -> 현행 체계' 순서로 설명하세요."
    return guidance


def build_user_prompt(question: str, intent: str, evidence: list[SearchHit]) -> str:
    context_memo = _context_memo_lines(question, intent, evidence)
    evidence_blocks = [_compact_evidence_block(idx, hit) for idx, hit in enumerate(evidence, start=1)]

    return f"""질문: {question}
질문 유형: {intent}
작성 지침: {_instruction_for_intent(intent, question)}

[빠른 맥락 메모]
{chr(10).join(context_memo) if context_memo else "- 별도 메모 없음"}

이 메모는 검색 방향을 빠르게 파악하기 위한 보조 정보입니다.
최종 답변의 모든 주장은 아래 압축 근거 블록에서 다시 확인되는 내용만 사용하세요.

[압축 근거 블록]
{chr(10).join(evidence_blocks)}

추가 작성 규칙:
- 법률, 시행령, 시행규칙, 개정이유, 신구 비교, 연혁 자료를 섞지 말고 구분해서 설명하세요.
- 질문 주제와 직접 연결되는 근거만 사용하고, 무관한 조문은 억지로 끌어오지 마세요.
- 문장은 짧고 분명하게 쓰고, 조문 항목을 그대로 읽지 말고 완결된 한국어 문장으로 바꾸세요.
- 근거가 부족하면 "현재 자료 기준으로"라고 분명히 쓰세요.
- 제목은 아래 형식을 그대로 유지하세요.

출력 형식:
{_output_format(intent, question)}
"""


def build_local_user_prompt(
    question: str,
    intent: str,
    evidence: list[SearchHit],
    *,
    profile: str = "default",
) -> str:
    evidence_count = {"small": 3, "medium": 4}.get(profile, 5)
    selected_evidence = _select_local_evidence(question, intent, evidence, count=evidence_count)
    evidence_blocks = [
        _compact_evidence_block(idx, hit, limit_map=LOCAL_EVIDENCE_CHAR_LIMITS)
        for idx, hit in enumerate(selected_evidence, start=1)
    ]

    context_lines = _context_memo_lines(question, intent, selected_evidence)
    local_rules = [
        "아래 근거에 없는 내용은 쓰지 말 것",
        "문장은 짧은 완결문으로 쓸 것",
        "조문 조각을 그대로 복사하지 말 것",
        "확신이 없으면 '현재 자료 기준으로'라고 쓸 것",
        "제목은 아래 형식을 그대로 유지할 것",
    ]
    if profile == "small":
        local_rules.append("핵심 문장은 각 소제목마다 1~2개만 쓸 것")

    return f"""질문: {question}
질문 유형: {intent}
답변 방향: {_instruction_for_intent(intent, question)}

[짧은 메모]
{chr(10).join(context_lines) if context_lines else "- 별도 메모 없음"}

[핵심 근거]
{chr(10).join(evidence_blocks)}

반드시 지킬 규칙:
{chr(10).join(f"- {rule}" for rule in local_rules)}

출력 형식:
{_output_format(intent, question)}
"""


def build_local_plain_user_prompt(
    question: str,
    intent: str,
    evidence: list[SearchHit],
    *,
    profile: str = "small",
) -> str:
    evidence_count = {"small": 2, "medium": 3}.get(profile, 4)
    if intent == "hybrid":
        evidence_count = {"small": 3, "medium": 4}.get(profile, 4)
    selected_evidence = _select_local_evidence(question, intent, evidence, count=evidence_count)
    direction = _instruction_for_intent(intent, question)
    sentence_limit = "1~2문장" if profile == "small" else "2~3문장"
    output_rule = (
        "짧은 평문만 출력할 것"
        if profile == "small"
        else '아래 두 줄만 출력할 것: "핵심 결론: ..." 그리고 "실무 해석: ..."'
    )
    if intent == "explain_change" and _is_timeline_question(question):
        direction = f"과거에서 현재 순서로 {sentence_limit} 안에서 짧게 설명하세요."
    elif intent == "explain_change":
        direction = f"무엇이 바뀌었는지와 왜 바뀌었는지를 {sentence_limit} 안에서 짧게 설명하세요."
    elif intent == "search":
        direction = f"현재 기준 핵심 규정을 {sentence_limit}으로 짧게 설명하세요."
    elif intent == "practical":
        direction = f"실무상 먼저 확인할 점과 주의사항을 {sentence_limit}으로 짧게 설명하세요."
    elif intent == "hybrid":
        direction = f"개정 배경과 현행 조문 연결을 {sentence_limit} 안에서 함께 설명하세요."

    evidence_lines = []
    for idx, hit in enumerate(selected_evidence, start=1):
        evidence_lines.append(
            f"- 근거 {idx}: {hit.chunk.law_name} | {_article_ref(hit)} | "
            f"{_compact_excerpt(_preferred_evidence_text(hit), limit=LOCAL_EVIDENCE_CHAR_LIMITS.get(hit.chunk.source_type, 140))}"
        )

    return f"""질문: {question}
질문 유형: {intent}
답변 방향: {direction}

반드시 지킬 규칙:
- 아래 근거에 없는 내용은 쓰지 말 것
- {sentence_limit} 정도로 {output_rule}
- 제목, 번호 목록, 마크다운 기호는 쓰지 말 것
- 조문 문구를 그대로 복사하지 말 것
- ??? 같이 물음표를 길게 반복하지 말 것
- 근거가 하나라도 있으면 "제한적입니다"만 쓰지 말고 근거 내용을 한 문장으로 요약할 것
- 근거가 전혀 없을 때만 "현재 자료 기준으로 확인되는 내용은 제한적입니다."라고 답할 것

근거:
{chr(10).join(evidence_lines) if evidence_lines else "- 근거 없음"}
"""
