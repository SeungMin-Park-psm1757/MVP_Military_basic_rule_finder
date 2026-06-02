from __future__ import annotations

from army_reg_rag.domain.models import DocumentChunk, SearchHit
from army_reg_rag.llm.prompts import build_local_plain_user_prompt, build_user_prompt


def make_hit(
    *,
    source_type: str,
    article_title: str,
    text: str,
    law_name: str = "군인의 지위 및 복무에 관한 기본법",
    article_no: str = "제45조",
    display_text: str = "",
    effective_date: str = "2026-01-08",
) -> SearchHit:
    extra = {}
    if display_text:
        extra["display_text"] = display_text
    return SearchHit(
        chunk=DocumentChunk(
            id=f"{source_type}-{article_title}",
            law_name=law_name,
            law_level="법률",
            source_type=source_type,
            version_label="현행",
            promulgation_date="2025-01-07",
            effective_date=effective_date,
            article_no=article_no,
            article_title=article_title,
            revision_kind="일부개정",
            text=text,
            source_url="https://example.com/law",
            extra=extra,
        ),
        score=0.9,
    )


def test_build_user_prompt_uses_compact_evidence_blocks():
    long_text = " ".join(["긴원문"] * 400)
    display_text = "신고자 보호를 강화하고 불이익조치를 금지하는 내용이다."
    evidence = [
        make_hit(
            source_type="revision_reason",
            article_title="개정이유",
            text=long_text,
            display_text=display_text,
        )
    ]

    prompt = build_user_prompt("징계 관련 개정 이유를 알려줘.", "explain_change", evidence)

    assert "[빠른 맥락 메모]" in prompt
    assert "[압축 근거 블록]" in prompt
    assert "압축 발췌" in prompt
    assert display_text in prompt
    assert long_text not in prompt


def test_build_user_prompt_adds_timeline_context_memo():
    evidence = [
        make_hit(
            source_type="history_note",
            article_title="외출·외박 및 휴가의 보류",
            text="징계혐의자는 외출·외박 및 휴가를 일시 보류할 수 있다.",
            display_text="징계혐의자는 외출·외박 및 휴가를 일시 보류할 수 있다.",
            effective_date="1966-03-15",
        ),
        make_hit(
            source_type="law_text",
            article_title="신고자 보호",
            text="누구든지 신고등을 이유로 신고자에게 불이익조치를 하여서는 아니 된다.",
            display_text="누구든지 신고등을 이유로 신고자에게 불이익조치를 하여서는 아니 된다.",
            effective_date="2026-01-08",
        ),
    ]

    prompt = build_user_prompt("징계 관련 기본법 변천사를 알려줘.", "explain_change", evidence)

    assert "- 변화축: 1966-03-15" in prompt
    assert "- 변화축: 2026-01-08" in prompt
    assert "과거 규정 -> 제도 전환 -> 현행 체계" in prompt


def test_build_local_plain_user_prompt_selects_diverse_timeline_evidence():
    evidence = [
        make_hit(
            source_type="revision_reason",
            article_title="개정이유",
            text="기본권 침해를 줄이고 신고한 군인을 보호하려는 취지이다.",
        ),
        make_hit(
            source_type="history_note",
            article_title="사적 제재의 금지",
            text="구타와 폭언, 가혹행위 등 사적 제재를 금지하였다.",
            effective_date="2015-07-13",
        ),
        make_hit(
            source_type="history_note",
            article_title="외출ㆍ외박 및 휴가의 보류",
            text="징계혐의자는 외출ㆍ외박 및 휴가를 일시 보류할 수 있었다.",
            effective_date="1966-03-15",
        ),
        make_hit(
            source_type="law_text",
            article_title="신고자 보호",
            text="신고를 이유로 불이익조치를 해서는 아니 된다.",
        ),
    ]

    prompt = build_local_plain_user_prompt(
        "가혹행위 신고 관련 규정 변천사 알려줘.", 
        "explain_change", 
        evidence,
        profile="large",
    )

    assert "개정이유" in prompt
    assert "사적 제재의 금지" in prompt
    assert "외출ㆍ외박 및 휴가의 보류" in prompt
    assert "신고자 보호" in prompt


def test_build_local_plain_user_prompt_includes_current_law_for_hybrid_question():
    evidence = [
        make_hit(
            source_type="revision_reason",
            article_title="제정·개정이유",
            text="군 내 기본권 침해를 줄이고 신고한 군인을 보호하려는 취지이다.",
        ),
        make_hit(
            law_name="군인복무규율",
            source_type="revision_reason",
            article_title="제정·개정이유",
            text="복무자세와 군기강을 일신하고 기본권보장을 보완하려는 취지이다.",
        ),
        make_hit(
            source_type="law_text",
            article_no="제43조",
            article_title="신고의무 등",
            text="가혹행위 등을 알게 된 경우 보고하거나 신고하여야 한다.",
        ),
        make_hit(
            source_type="law_text",
            article_no="제45조",
            article_title="신고자 보호",
            text="신고등을 이유로 징계조치 등 불이익조치를 하여서는 아니 된다.",
        ),
    ]

    prompt = build_local_plain_user_prompt(
        "군인복무규율이 폐지되고 기본법 체계가 도입되면서 어떻게 재편되었는지 개정 이유와 현행 조문을 함께 근거로 설명해줘.",
        "hybrid",
        evidence,
        profile="small",
    )

    assert "제정·개정이유" in prompt
    assert "제43조 신고의무 등" in prompt or "제45조 신고자 보호" in prompt
    assert sum(1 for line in prompt.splitlines() if line.startswith("- 근거 ")) == 3
    assert "제한적입니다\"만 쓰지 말고" in prompt
