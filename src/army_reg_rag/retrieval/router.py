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

HISTORY_KEYWORDS = [
    "과거",
    "이전",
    "예전",
    "종전",
    "연혁",
    "변천",
    "변천사",
    "흐름",
    "이어졌",
    "유래",
    "발전",
    "넘어오",
    "군인복무규율",
]

PRACTICAL_KEYWORDS = [
    "실무",
    "적용",
    "참고",
    "주의",
    "처리",
    "확인",
    "확인해야",
    "절차",
]

CURRENT_PRACTICAL_CUES = [
    "지금",
    "현재",
    "현행",
    "실무",
    "적용",
    "처리",
    "확인",
    "참고",
    "주의",
]


def decide_route(question: str) -> RouteDecision:
    q = question.strip()
    has_search = any(keyword in q for keyword in SEARCH_KEYWORDS)
    has_explain = any(keyword in q for keyword in EXPLAIN_KEYWORDS)
    has_history = any(keyword in q for keyword in HISTORY_KEYWORDS)
    has_practical = any(keyword in q for keyword in PRACTICAL_KEYWORDS)
    has_current_practical_cue = any(keyword in q for keyword in CURRENT_PRACTICAL_CUES)

    if (has_explain or has_history) and has_practical and has_current_practical_cue:
        return RouteDecision(
            intent="hybrid",
            preferred_source_types=["history_note", "revision_reason", "old_new_comparison", "law_text"],
            rationale="과거 연혁과 현재 실무 기준을 함께 묻는 질문으로 보고, 군인복무규율 연계 자료와 개정 자료를 함께 조회합니다.",
        )
    if has_history:
        return RouteDecision(
            intent="explain_change",
            preferred_source_types=["history_note", "revision_reason", "old_new_comparison", "law_text"],
            rationale="과거 내용이나 연혁을 묻는 질문으로 보고, 군인복무규율과 현재 법 체계의 연결 자료를 우선 조회합니다.",
        )
    if has_explain:
        return RouteDecision(
            intent="explain_change",
            preferred_source_types=["revision_reason", "old_new_comparison", "history_note", "law_text"],
            rationale="개정 이유를 먼저 설명해야 하는 질문으로 보고, 개정이유와 신구 비교 자료를 우선 조회합니다.",
        )
    if has_practical:
        return RouteDecision(
            intent="practical",
            preferred_source_types=["law_text", "revision_reason", "old_new_comparison"],
            rationale="실무 참고 질문으로 보고 현행 규정을 먼저 확인하고, 필요하면 개정 자료를 보조로 확인합니다.",
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
        rationale="일반 검색 질문으로 보고 현행 조문을 우선 조회합니다.",
    )
