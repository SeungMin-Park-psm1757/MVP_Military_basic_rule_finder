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


def test_answer_postprocess_removes_duplicate_bullets(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    markdown = (
        "## 답변 개요\n"
        "### 핵심 결론\n"
        "- 신고자 보호는 불이익조치를 금지한다.\n"
        "- 신고자 보호는 불이익조치를 금지한다.\n"
        "- 비밀보장은 인적사항 공개를 막는 규정이다.\n"
    )

    processed = client._postprocess_answer_markdown(markdown)

    assert processed.count("신고자 보호는 불이익조치를 금지한다.") == 1
    assert "비밀보장은 인적사항 공개를 막는 규정이다." in processed


def test_search_fallback_strips_korean_list_marker_and_keeps_sentence_ending(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="decree-1",
            law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            law_level="시행령",
            source_type="law_text",
            article_no="제9조",
            article_title="휴가의 종류 등",
            text="- 가, 특별휴가 및 정기휴가로 구분한다",
        ),
    ]

    answer = client.generate_answer(
        "휴가 종류를 정리해줘",
        "search",
        evidence,
        allow_generation=False,
    )

    assert "가, 특별휴가" not in answer.text
    assert "특별휴가 및 정기휴가로 구분한다." in answer.text


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


def test_extract_points_contextualizes_fragment_lines(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="legacy-history",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="history_note",
            article_no="제42조",
            article_title="외출ㆍ외박ㆍ휴가의 제한 및 보류",
            text=(
                "허가권자는 다음 각 호의 어느 하나에 해당하는 사람에 대해서는 휴가를 보류할 수 있다.\n"
                "1. 수사 중인 사람\n"
                "2. 형사피의자와 피고인 및 징계혐의자"
            ),
        ),
    ]

    answer = client.generate_answer(
        "징계 관련 연혁을 알려줘.",
        "explain_change",
        evidence,
        allow_generation=False,
    )

    assert "형사피의자와 피고인 및 징계혐의자" in answer.text
    assert "보류 대상으로 두었습니다." in answer.text or "관련 대상으로 제시됩니다" in answer.text
    assert "- 형사피의자와 피고인 및 징계혐의자." not in answer.text


def test_timeline_fallback_rewrites_fragmentary_history_summary_as_sentence(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="history-fragment",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="history_note",
            article_no="제87조",
            article_title="외출ㆍ외박 및 휴가의 보류",
            text="2. 형사피의자와 피고인 및 징계혐의자.",
        ),
        make_hit(
            chunk_id="law-current",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제45조",
            article_title="신고자 보호",
            text="누구든지 신고를 이유로 신고자에게 징계조치 등 불이익조치를 하여서는 아니 된다.",
        ),
    ]

    answer = client.generate_answer(
        "징계 관련 기본법 변천사를 알려줘",
        "explain_change",
        evidence,
        allow_generation=False,
    )

    assert "보류 대상으로 두었습니다." in answer.text
    assert "\n- 2. 형사피의자와 피고인 및 징계혐의자." not in answer.text


def test_explain_fallback_handles_childcare_and_marks_leave_of_absence_gap(tmp_path):
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
                "- 일·가정 양립과 출산·돌봄 지원 강화를 위해 배우자 출산휴가와 청원휴가 기준을 보완하였다.\n"
                "- 가족 간호와 돌봄 사유를 보다 명확히 정비하였다."
            ),
        ),
        make_hit(
            chunk_id="history-1",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="history_note",
            article_no="제39조의4",
            article_title="청원휴가",
            text=(
                "- 허가권자는 배우자가 출산하였을 때와 직계가족 간호가 필요한 때 청원휴가를 허가할 수 있다.\n"
                "- 가족 간호와 관련된 휴가 사유가 단계적으로 정비되었다."
            ),
        ),
        make_hit(
            chunk_id="law-1",
            law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            law_level="시행령",
            source_type="law_text",
            article_no="제12조",
            article_title="청원휴가",
            text=(
                "- 지휘관은 군인이 신청한 경우 배우자 출산휴가와 가족 간호 사유에 따른 청원휴가를 승인할 수 있다.\n"
                "- 세부 일수와 승인 기준은 시행령에 따른다."
            ),
        ),
    ]

    answer = client.generate_answer(
        "군에서 육아 관련 휴가나 휴직에 대한 내용과 그 변화를 설명해줘.",
        "explain_change",
        evidence,
        allow_generation=False,
    )

    assert "배우자 출산휴가" in answer.text
    assert "가족 간호" in answer.text or "돌봄" in answer.text
    assert "휴직 기준" in answer.text


def test_search_fallback_for_discipline_query_avoids_leave_specific_guidance(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="law-1",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제45조",
            article_title="신고자 보호",
            text="누구든지 신고 등을 이유로 신고자에게 징계조치 등 어떠한 신분상 불이익을 주어서는 아니 된다.",
        ),
        make_hit(
            chunk_id="law-2",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제43조",
            article_title="신고의무 등",
            text="군인은 병영생활에서 다른 군인이 구타, 폭언, 가혹행위 및 집단 따돌림 등을 한 사실을 알게 된 경우 즉시 신고해야 한다.",
        ),
        make_hit(
            chunk_id="history-1",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="history_note",
            article_no="제87조",
            article_title="외출ㆍ외박 및 휴가의 보류",
            text="형사피의자와 피고인 및 징계혐의자는 외출ㆍ외박 및 휴가를 일시 보류할 수 있다.",
        ),
    ]

    answer = client.generate_answer(
        "징계관련 내용을 정리해서 나열해줘.",
        "search",
        evidence,
        allow_generation=False,
    )

    assert "신고자 보호" in answer.text
    assert "신고의무" in answer.text
    assert "휴가 종류" not in answer.text
    assert "### 연혁으로 같이 보이는 변화" in answer.text
    assert "징계 사유·절차·양정 전체는 현재 코퍼스만으로는 완결되지 않습니다." in answer.text


def test_search_fallback_for_discipline_query_prefers_history_support_over_irrelevant_counselor(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="discipline-law",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제45조",
            article_title="신고자 보호",
            text="신고를 이유로 징계조치 등 불이익조치를 해서는 아니 된다.",
        ),
        make_hit(
            chunk_id="discipline-history",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="history_note",
            article_no="제87조",
            article_title="외출ㆍ외박 및 휴가의 보류",
            text="형사피의자와 피고인 및 징계혐의자는 외출ㆍ외박 및 휴가를 일시 보류할 수 있다.",
        ),
        make_hit(
            chunk_id="irrelevant-counselor",
            law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            law_level="시행령",
            source_type="law_text",
            article_no="제33조",
            article_title="전문상담관의 선발",
            text="전문상담관은 서류전형과 면접시험을 거쳐 선발한다.",
        ),
    ]

    answer = client.generate_answer(
        "징계관련 내용을 정리해서 나열해줘.",
        "search",
        evidence,
        allow_generation=False,
    )

    assert "신고자 보호" in answer.text
    assert "과거 징계혐의자 관리" in answer.text
    assert "징계 사유, 절차, 양정" in answer.text
    assert "전문상담관의 선발" not in answer.text


def test_search_fallback_for_discipline_paraphrase_variant(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="discipline-law",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제43조",
            article_title="신고의무 등",
            text="군인은 구타, 폭언, 가혹행위 및 집단 따돌림 등 사적 제재를 알게 된 경우 즉시 신고하여야 한다.",
        ),
        make_hit(
            chunk_id="discipline-law-2",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제45조",
            article_title="신고자 보호",
            text="신고를 이유로 신고자에게 징계조치 등 불이익조치를 해서는 아니 된다.",
        ),
        make_hit(
            chunk_id="discipline-history",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="history_note",
            article_no="제87조",
            article_title="외출ㆍ외박 및 휴가의 보류",
            text="형사피의자와 피고인 및 징계혐의자는 외출ㆍ외박 및 휴가를 일시 보류할 수 있다.",
        ),
    ]

    answer = client.generate_answer(
        "징계 쪽 규정 위주로 요약해줘.",
        "search",
        evidence,
        allow_generation=False,
    )

    assert "신고의무" in answer.text or "사적 제재" in answer.text
    assert "신고자 보호" in answer.text


def test_transition_fallback_marks_discipline_procedure_out_of_corpus(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="repeal",
            law_name="군인복무규율",
            law_level="대통령령",
            source_type="old_new_comparison",
            article_no="부칙",
            article_title="폐지",
            text="부칙에서 군인복무규율은 폐지한다고 정하였다.",
        ),
        make_hit(
            chunk_id="reason",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="revision_reason",
            article_no="",
            article_title="제정·개정이유",
            text="군 내 기본권 침해를 줄이고 군인의 의무와 권리 보호 기준을 법률 단계에서 정비하려는 취지로 제정되었다.",
        ),
        make_hit(
            chunk_id="grievance",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제40조",
            article_title="고충 처리",
            text="군인은 고충을 제기할 수 있고 군인고충심사위원회는 고충 처리 절차를 담당한다.",
        ),
        make_hit(
            chunk_id="personnel-link",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="old_new_comparison",
            article_no="부칙",
            article_title="군인사법 경과조치",
            text="군인사법 제46조, 제47조, 제51조의3 및 제51조의4를 삭제하고 기본법 체계로 옮기는 경과조치를 둔다.",
        ),
    ]

    answer = client.generate_answer(
        "군인복무규율이 2016년 폐지된 뒤, 복무·군기·권리구제 사항은 기본법 체계로 어떻게 재편되었고, "
        "징계의 종류·절차는 현재 어떤 법령이 담당하는지 구분해서 설명해줘.",
        "hybrid",
        evidence,
        allow_generation=False,
    )

    assert "군인복무규율" in answer.text
    assert "폐지" in answer.text
    assert "군인고충심사위원회" in answer.text
    assert "징계 종류·절차" in answer.text or "징계의 종류·절차" in answer.text
    assert "현재 코퍼스에는 「군인사법」 본문이나 「군인 징계령」 계열 본문이 들어 있지 않습니다." in answer.text
    assert "확정 설명할 수 없습니다" in answer.text
    assert "종류·일수·시간" not in answer.text


def test_postprocess_repairs_common_text_artifacts(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))

    processed = client._postprocess_answer_markdown(
        "## 답변 개요\n"
        "### 핵심 결론\n"
        "청원휴 가와 왕복소일수, 관련 사항를 확인합니다.\n"
    )

    assert "청원휴가" in processed
    assert "왕복소요일수" in processed
    assert "사항을" in processed


def test_search_fallback_limits_main_rules_to_explicit_articles(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            chunk_id="law-45",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제45조",
            article_title="신고자 보호",
            text="제45조는 신고등을 이유로 신고자에게 징계조치 등 불이익조치를 하지 못하게 한다.",
        ),
        make_hit(
            chunk_id="law-41",
            law_name="군인의 지위 및 복무에 관한 기본법",
            law_level="법률",
            source_type="law_text",
            article_no="제41조",
            article_title="전문상담관",
            text="제41조는 병영생활 전문상담관을 두는 기준을 정한다.",
        ),
    ]

    answer = client.generate_answer(
        "현행 기본법 제45조 신고자 보호 규정을 조문번호와 함께 설명해줘.",
        "search",
        evidence,
        allow_generation=False,
    )

    main_rules = answer.text.split("### 주요 규정", 1)[1].split("### 실무 참고", 1)[0]
    assert "제45조" in main_rules
    assert "제41조" not in main_rules
