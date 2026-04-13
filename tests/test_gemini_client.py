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
            article_no="제8조",
            article_title="휴가의 보장",
            text=(
                "- 군인은 대통령령에 따라 휴가·외출·외박을 보장받는다.\n"
                "- 지휘관은 국가비상사태, 작전상황, 재난, 교육훈련·평가·검열, 징계심의 대상, 환자 상태 등 사유가 있으면 제한할 수 있다."
            ),
        ),
        make_hit(
            chunk_id="decree-1",
            law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            law_level="시행령",
            source_type="law_text",
            article_no="제2조의6",
            article_title="휴가의 종류와 확인 범위",
            text=(
                "- 휴가는 연가, 공가, 청원휴가, 포상휴가, 특별휴가로 구분한다.\n"
                "- 휴가 확인 범위는 부대 현재 병력의 5분의 1 이내가 기준이다."
            ),
        ),
    ]

    answer = client.generate_answer(
        "군인의 지위 및 복무에 관한 기본법에서 휴가 관련 현행 규정을 찾아줘",
        "search",
        evidence,
        allow_generation=False,
    )

    assert "### 주요 규정" in answer.text
    assert "법률 제8조(휴가의 보장)" in answer.text
    assert "시행령 제2조의6(휴가의 종류와 확인 범위)" in answer.text
    assert "말씀드리겠습니다" not in answer.text
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
                "- 출산·양육 지원 강화를 위해 육아시간 사용 범위를 확대하였다.\n"
                "- 군인의 복무 여건 개선을 위해 돌봄 관련 운영 기준도 보완하였다."
            ),
        ),
        make_hit(
            chunk_id="compare-1",
            law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            law_level="시행령",
            source_type="old_new_comparison",
            article_no="제2조",
            article_title="육아시간과 돌봄휴가",
            text=(
                "- 육아시간 대상을 5세 이하 자녀에서 8세 이하 또는 초등학교 2학년 이하 자녀로 확대하였다.\n"
                "- 육아시간 사용기간은 24개월에서 36개월로 늘렸다."
            ),
        ),
    ]

    answer = client.generate_answer(
        "육아시간 관련 규정이 왜 바뀌었는지 개정 이유 중심으로 설명해줘.",
        "explain_change",
        evidence,
        allow_generation=False,
    )

    assert "### 주요 개정 이유" in answer.text
    assert "### 실제 제도 변화" in answer.text
    assert "출산" in answer.text
    assert "8세 이하 또는 초등학교 2학년 이하" in answer.text
    assert "법률 자문" not in answer.text


def test_explain_fallback_mentions_military_service_rules_history(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="history-1",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="history_note",
            article_no="",
            article_title="연혁 메모",
            text=(
                "- 군인복무규율 체계에서 군인의 지위 및 복무에 관한 기본법 체계로 넘어오면서 기본 원칙은 법률로 정리되었다.\n"
                "- 세부 휴가 기준은 시행령과 시행규칙으로 재구성되었다."
            ),
        ),
    ]

    answer = client.generate_answer(
        "과거 휴가 기준이 군인복무규율에서 어떻게 이어졌는지 알려줘.",
        "explain_change",
        evidence,
        allow_generation=False,
    )

    assert "군인복무규율" in answer.text
    assert "기본법" in answer.text
    assert "시행령" in answer.text


def test_timeline_fallback_uses_chronology_sections(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="legacy-history",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="history_note",
            article_no="제42조",
            article_title="외출ㆍ외박ㆍ휴가의 제한 및 보류",
            text="징계혐의자는 외출ㆍ외박 및 휴가를 일시 보류할 수 있다고 규정하였다.",
        ),
        make_hit(
            chunk_id="basic-reason",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="revision_reason",
            article_no="",
            article_title="제정·개정이유",
            text="기본권 침해를 줄이고 신고한 군인을 보호하도록 하며 징계조치 등 불이익조치를 금지하는 방향으로 제정되었다.",
        ),
        make_hit(
            chunk_id="basic-law",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제45조",
            article_title="신고자 보호",
            text="누구든지 신고를 이유로 신고자에게 징계조치 등 어떠한 신분상 불이익도 하여서는 아니 된다.",
        ),
    ]

    answer = client.generate_answer(
        "징계 관련 기본법 변천사를 알려줘.",
        "explain_change",
        evidence,
        allow_generation=False,
    )

    assert "## 연혁 정리" in answer.text
    assert "### 시기별 변화" in answer.text
    assert "####" in answer.text
    assert "### 현재 체계와 연결" in answer.text
    assert "군인복무규율" in answer.text
    assert "신고자 보호" in answer.text
