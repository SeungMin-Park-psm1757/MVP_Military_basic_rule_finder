from __future__ import annotations

from army_reg_rag.config import AppConfig, DataConfig, Settings
from army_reg_rag.domain.models import DocumentChunk, SearchHit
from army_reg_rag.llm.gemini_client import GeminiAnswerClient


def make_settings(tmp_path) -> Settings:
    settings = Settings(
        app=AppConfig(chroma_path=str(tmp_path / "chroma")),
        data=DataConfig(runtime_dir=str(tmp_path / "runtime")),
    )
    settings.ensure_runtime_dirs()
    return settings


def make_hit(
    *,
    chunk_id: str,
    law_name: str,
    law_level: str,
    source_type: str,
    article_no: str,
    article_title: str,
    text: str,
) -> SearchHit:
    return SearchHit(
        chunk=DocumentChunk(
            id=chunk_id,
            law_name=law_name,
            law_level=law_level,
            source_type=source_type,
            version_label="현행",
            promulgation_date="2025-01-01",
            effective_date="2026-01-01",
            article_no=article_no,
            article_title=article_title,
            revision_kind="일부개정",
            text=text,
            source_url="https://example.com",
        ),
        score=0.9,
    )


def test_search_fallback_uses_direct_rule_sections(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="law-1",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제18조",
            article_title="휴가 등의 보장",
            text=(
                "- 군인은 대통령령에 따라 휴가·외출·외박을 보장받는다.\n"
                "- 지휘관은 국가비상사태, 작전상황, 재난, 교육훈련·평가·검열, 징계 절차 등 사유가 있으면 제한하거나 보류할 수 있다."
            ),
        ),
        make_hit(
            chunk_id="decree-1",
            law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            law_level="시행령",
            source_type="law_text",
            article_no="제9조~제16조",
            article_title="휴가의 종류·승인범위",
            text=(
                "- 휴가는 연가, 공가, 청원휴가, 특별휴가, 정기휴가로 구분한다.\n"
                "- 하사 이상 연가는 연 21일 이내이고, 휴가 승인 범위는 부대 현재 병력의 5분의 1 이내가 원칙이다."
            ),
        ),
    ]

    answer = client.generate_answer(
        "군인의 지위 및 복무에 관한 기본법에서 휴가 관련 현행 규정을 찾아줘.",
        "search",
        evidence,
        allow_generation=False,
    )

    assert "### 주요 규정" in answer.text
    assert "법률 제18조(휴가 등의 보장)" in answer.text
    assert "시행령 제9조~제16조(휴가의 종류·승인범위)" in answer.text
    assert "흐름이 맞습니다" not in answer.text
    assert "~에서 출발" not in answer.text


def test_explain_fallback_uses_reason_first_structure(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="reason-1",
            law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            law_level="시행령",
            source_type="revision_reason",
            article_no="",
            article_title="개정이유",
            text=(
                "- 저출생 극복과 일·가정 양립 지원을 위해 육아시간 사용 범위를 확대하였다.\n"
                "- 군인의 근무 여건 개선을 위해 돌봄 관련 제도 운영 기준을 손질하였다."
            ),
        ),
        make_hit(
            chunk_id="compare-1",
            law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            law_level="시행령",
            source_type="old_new_comparison",
            article_no="제12조",
            article_title="육아시간·자녀돌봄휴가",
            text=(
                "- 육아시간 대상을 5세 이하 자녀에서 8세 이하 또는 초등학교 2학년 이하 자녀로 확대하였다.\n"
                "- 육아시간 사용기간을 24개월에서 36개월로 늘렸다."
            ),
        ),
    ]

    answer = client.generate_answer(
        "왜 육아시간 관련 규정이 바뀌었는지 개정 이유 중심으로 설명해줘.",
        "explain_change",
        evidence,
        allow_generation=False,
    )

    assert "### 주요 개정 이유" in answer.text
    assert "### 실제 제도 변화" in answer.text
    assert "저출생 대응" in answer.text or "저출생" in answer.text
    assert "8세 이하 또는 초등학교 2학년 이하" in answer.text
    assert "법률 조문만으로 끝내지 말고" not in answer.text
