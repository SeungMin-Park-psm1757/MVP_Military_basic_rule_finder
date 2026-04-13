from __future__ import annotations

from io import BytesIO

from docx import Document

from army_reg_rag.domain.models import DocumentChunk, SearchHit
from army_reg_rag.export_docx import build_conversation_docx, build_conversation_turns


def make_hit() -> SearchHit:
    return SearchHit(
        chunk=DocumentChunk(
            id="law-1",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            version_label="현행",
            promulgation_date="2025-01-01",
            effective_date="2026-01-01",
            article_no="제8조",
            article_title="휴가",
            revision_kind="일부개정",
            text="군인의 휴가 관련 기준을 설명한다.",
            source_url="https://example.com/law-1",
        ),
        score=0.9,
    )


def test_build_conversation_turns_groups_user_and_assistant_messages():
    history = [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "answer_markdown": "첫 답변"},
        {"role": "user", "content": "둘째 질문"},
        {"role": "assistant", "answer_markdown": "둘째 답변"},
    ]

    turns = build_conversation_turns(history)

    assert len(turns) == 2
    assert turns[0].question["content"] == "첫 질문"
    assert turns[1].answer["answer_markdown"] == "둘째 답변"


def test_build_conversation_docx_contains_question_answer_and_evidence():
    history = [
        {"role": "user", "content": "휴가 규정 알려줘"},
        {
            "role": "assistant",
            "answer_markdown": "### 핵심 결론\n- 현행 규정을 확인했습니다.",
            "evidence": [make_hit()],
        },
    ]

    content = build_conversation_docx(history)
    document = Document(BytesIO(content))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    assert "휴가 규정 알려줘" in text
    assert "핵심 결론" in text
    assert "현행 규정을 확인했습니다." in text
    assert "군인의 지위 및 복무에 관한 기본법" in text
