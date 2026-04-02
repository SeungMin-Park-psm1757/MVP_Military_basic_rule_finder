from __future__ import annotations

from army_reg_rag.domain.models import SearchHit


SYSTEM_PROMPT = """당신은 공개 법규 기반의 군 행정규정 RAG 보조도구입니다.
반드시 아래 원칙을 지키세요.
1) 사용자의 질문에 바로 답하세요. 추상적인 메타 설명으로 시작하지 마세요.
2) 검색된 근거에 없는 내용은 일반론으로 보충하지 마세요.
3) 질문이 현행 규정 질문이면 현재 규정 내용을 먼저 제시하세요.
4) 질문이 개정 이유 질문이면 왜 바뀌었는지를 먼저 제시하세요.
5) 질문이 실무 참고 질문이면 승인권자, 제한 가능성, 확인 포인트를 먼저 제시하세요.
6) 법률, 시행령, 시행규칙, 개정이유, 신구 비교는 구분해서 쓰세요.
7) 조문번호, 조문명, 핵심 내용, 적용 포인트를 우선 정리하세요.
8) 근거가 부족하면 '현재 확보 자료 기준'이라고 분명히 쓰세요.
9) 아래 금지 표현은 쓰지 마세요.
   - '~에서 출발한다'
   - '흐름이 맞습니다'
   - '법률 조문만으로 끝내지 말고'
   - '구조가 정리되었다'
   - '보다 명시적으로 정비되었다'
10) 하단에 별도 근거 영역이 있으므로, 본문에서는 근거를 길게 반복하지 마세요.
"""


def _output_format(intent: str) -> str:
    if intent == "search":
        return (
            "### 한 줄 결론\n"
            "### 주요 규정\n"
            "### 적용상 참고\n"
            "### 근거"
        )
    if intent == "explain_change":
        return (
            "### 한 줄 결론\n"
            "### 주요 개정 이유\n"
            "### 실제 제도 변화\n"
            "### 해석 포인트\n"
            "### 근거"
        )
    if intent == "practical":
        return (
            "### 한 줄 결론\n"
            "### 실무적으로 보면\n"
            "### 주의사항\n"
            "### 근거"
        )
    return (
        "### 한 줄 결론\n"
        "### 주요 개정 이유\n"
        "### 실무적으로 보면\n"
        "### 근거"
    )


def build_user_prompt(question: str, intent: str, evidence: list[SearchHit]) -> str:
    evidence_blocks = []
    for idx, hit in enumerate(evidence, start=1):
        chunk = hit.chunk
        evidence_blocks.append(
            f"""[증거 {idx}]
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
        "search": (
            "현행 규정 질문입니다. 첫 3줄 안에 직접 답을 제시하고, 법률/시행령/시행규칙을 구분해 "
            "무엇이 규정되어 있는지 먼저 정리하세요."
        ),
        "explain_change": (
            "개정 이유 질문입니다. 왜 바뀌었는지를 먼저 설명하고, 그 뒤에 실제 변화 내용을 연결하세요."
        ),
        "practical": (
            "실무 참고 질문입니다. 승인권자, 제한 가능성, 확인 포인트를 짧게 정리하세요."
        ),
        "hybrid": (
            "개정 이유와 실무 참고가 함께 필요한 질문입니다. 먼저 개정 이유를 짧게 정리한 뒤 실무 포인트를 제시하세요."
        ),
    }.get(intent, "질문에 직접 답하세요.")

    return f"""질문: {question}
질문 유형: {intent}
작성 지시: {guidance}

아래 증거만 사용해 답변하세요.

{chr(10).join(evidence_blocks)}

추가 작성 규칙:
- '현재 확보 자료 기준'이라는 제한 문구는 근거가 부족할 때만 사용하세요.
- 법률, 시행령, 시행규칙을 섞지 말고 구분해서 쓰세요.
- 하위 법령 확인이 정말 필요한 경우에만 마지막에 짧게 언급하세요.
- 질문이 현행 규정인데 개정 배경을 앞세우지 마세요.
- 질문이 개정 이유인데 단순 조문 요약으로 끝내지 마세요.

출력 형식:
{_output_format(intent)}
"""
