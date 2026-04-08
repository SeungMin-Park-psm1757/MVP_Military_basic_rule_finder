from __future__ import annotations

from army_reg_rag.domain.models import SearchHit


SYSTEM_PROMPT = """당신은 공개 법규 기반의 군 복무 규정 RAG 보조 도구입니다.
반드시 아래 원칙을 지키세요.
1) 사용자 질문에 바로 답하고, 근거 없는 일반론으로 메우지 마세요.
2) 현행 법률, 시행령, 시행규칙, 개정이유, 신구비교, 연혁 메모를 구분해서 설명하세요.
3) 과거 내용이나 연혁 질문에서 근거에 군인복무규율이 나오면, 현재 기본법과 관련 시행령·시행규칙이 그 체계에서 발전한 관계임을 명시하세요.
4) 법률 자문처럼 단정하지 말고, 근거 중심의 설명과 원문 확인 안내를 유지하세요.
5) 근거가 부족하면 '현재 자료 기준'이라고 분명히 쓰세요.
"""


def _output_format(intent: str) -> str:
    if intent == "search":
        return (
            "### 핵심 결론\n"
            "### 주요 규정\n"
            "### 실무 참고\n"
            "### 근거"
        )
    if intent == "explain_change":
        return (
            "### 핵심 결론\n"
            "### 주요 개정 이유\n"
            "### 실제 제도 변화\n"
            "### 해석 시사점\n"
            "### 근거"
        )
    if intent == "practical":
        return (
            "### 핵심 결론\n"
            "### 실무적으로 보면\n"
            "### 주의사항\n"
            "### 근거"
        )
    return (
        "### 핵심 결론\n"
        "### 주요 개정 이유\n"
        "### 실무적으로 보면\n"
        "### 근거"
    )


def build_user_prompt(question: str, intent: str, evidence: list[SearchHit]) -> str:
    evidence_blocks = []
    for idx, hit in enumerate(evidence, start=1):
        chunk = hit.chunk
        evidence_blocks.append(
            f"""[근거 {idx}]
- 법령명: {chunk.law_name}
- 법령 수준: {chunk.law_level}
- 자료유형: {chunk.source_type}
- 조문: {chunk.article_no} {chunk.article_title}
- 공포일: {chunk.promulgation_date}
- 시행일: {chunk.effective_date}
- 개정형태: {chunk.revision_kind}
- 링크: {chunk.source_url}
- 본문:
{chunk.text}
"""
        )

    guidance = {
        "search": "현행 규정 질문입니다. 현행 기준을 먼저 답하고, 필요하면 하위 법령 연결을 덧붙이세요.",
        "explain_change": "개정 이유 또는 과거 연혁 질문입니다. 왜 바뀌었는지와 어떤 체계 변화가 있었는지 먼저 설명하세요.",
        "practical": "실무 참고 질문입니다. 확인 순서와 주의점을 먼저 정리하세요.",
        "hybrid": "개정 이유와 실무 참고가 함께 필요한 질문입니다. 연혁과 현재 적용 관점을 함께 정리하세요.",
    }.get(intent, "질문에 직접 답하세요.")

    return f"""질문: {question}
질문 유형: {intent}
작성 지시: {guidance}

아래 근거만 사용해 답하세요.

{chr(10).join(evidence_blocks)}

추가 작성 규칙:
- 과거 내용 또는 연혁 질문에서 근거에 군인복무규율이 나오면, 현재 기본법과 시행령·시행규칙이 그 체계에서 발전한 관계임을 명시하세요.
- 법률, 시행령, 시행규칙, 개정이유, 신구비교, 연혁 메모를 섞지 말고 구분해서 설명하세요.
- 현행 규정 질문인데 개정 배경만 길게 설명하지 마세요.
- 개정 이유 질문인데 단순 조문 요약만 하지 마세요.
- 근거가 부족하면 '현재 자료 기준'이라고 쓰세요.

출력 형식:
{_output_format(intent)}
"""
