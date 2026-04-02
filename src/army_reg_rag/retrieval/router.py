from __future__ import annotations

from army_reg_rag.domain.models import RouteDecision


SEARCH_KEYWORDS = [
    "현행 규정",
    "현재 규정",
    "찾아줘",
    "무슨 내용",
    "조문",
]
EXPLAIN_KEYWORDS = [
    "왜 바뀌었",
    "개정 이유",
    "배경",
    "취지",
    "이유",
    "바뀌었",
    "변경",
    "개정",
]
PRACTICAL_KEYWORDS = [
    "실무",
    "적용",
    "참고",
    "주의",
    "어떻게",
    "처리",
    "승인",
    "확인해야",
]


def decide_route(question: str) -> RouteDecision:
    q = question.strip()
    has_search = any(keyword in q for keyword in SEARCH_KEYWORDS)
    has_explain = any(keyword in q for keyword in EXPLAIN_KEYWORDS)
    has_practical = any(keyword in q for keyword in PRACTICAL_KEYWORDS)

    if has_explain and has_practical:
        return RouteDecision(
            intent="hybrid",
            preferred_source_types=["revision_reason", "old_new_comparison", "law_text", "history_note"],
            rationale="개정 이유와 실무 참고 질문이 함께 있어 개정 자료와 현행 규정을 함께 조회합니다.",
        )
    if has_explain:
        return RouteDecision(
            intent="explain_change",
            preferred_source_types=["revision_reason", "old_new_comparison", "history_note", "law_text"],
            rationale="개정 이유를 먼저 설명해야 하므로 개정이유와 신구 비교 자료를 우선 조회합니다.",
        )
    if has_practical:
        return RouteDecision(
            intent="practical",
            preferred_source_types=["law_text", "revision_reason", "old_new_comparison"],
            rationale="실무 참고 질문이므로 현행 규정을 먼저 보고, 필요한 경우 개정 자료를 보조로 확인합니다.",
        )
    if has_search:
        return RouteDecision(
            intent="search",
            preferred_source_types=["law_text", "revision_reason", "old_new_comparison", "history_note"],
            rationale="현행 규정을 찾는 질문으로 보고 법률과 하위 법령을 우선 조회합니다.",
        )
    return RouteDecision(
        intent="search",
        preferred_source_types=["law_text", "revision_reason", "old_new_comparison", "history_note"],
        rationale="일반 검색 질의로 보고 현행 조문을 우선 조회합니다.",
    )
