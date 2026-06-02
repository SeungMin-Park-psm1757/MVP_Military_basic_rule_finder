from __future__ import annotations

from army_reg_rag.config import AppConfig, DataConfig, RetrievalConfig, Settings
from army_reg_rag.domain.models import DocumentChunk, SearchHit
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.services.answer_service import AnswerService


def make_chunk(
    chunk_id: str,
    source_type: str,
    text: str,
    *,
    law_name: str = "군인의 지위 및 복무에 관한 기본법",
    article_no: str = "제1조",
    article_title: str = "테스트",
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        law_name=law_name,
        law_level="법률",
        source_type=source_type,
        version_label="현행",
        promulgation_date="2026-01-01",
        effective_date="2026-01-01",
        article_no=article_no,
        article_title=article_title,
        revision_kind="일부개정",
        text=text,
        source_url="https://example.com",
    )


def make_settings(tmp_path) -> Settings:
    settings = Settings(
        app=AppConfig(chroma_path=str(tmp_path / "chroma")),
        data=DataConfig(runtime_dir=str(tmp_path / "runtime")),
        retrieval=RetrievalConfig(top_k=6, max_evidence_per_source_type=1),
    )
    settings.ensure_runtime_dirs()
    return settings


def test_dedupe_hits_removes_duplicate_article_and_text(tmp_path):
    settings = make_settings(tmp_path)
    service = AnswerService(settings, store=ChromaStore(settings))
    hits = [
        SearchHit(
            chunk=make_chunk(
                "article-a",
                "law_text",
                "신고자 보호 조문은 신고를 이유로 한 불이익조치를 금지한다.",
                article_no="제45조",
                article_title="신고자 보호",
            ),
            score=0.9,
        ),
        SearchHit(
            chunk=make_chunk(
                "article-b",
                "law_text",
                "신고자 보호 조문은 신고를 이유로 한 불이익조치를 금지한다.",
                article_no="제45조",
                article_title="신고자 보호",
            ),
            score=0.8,
        ),
        SearchHit(
            chunk=make_chunk(
                "article-c",
                "law_text",
                "신고자라는 사정이나 인적사항을 공개하여서는 아니 된다.",
                article_no="제44조",
                article_title="신고자에 대한 비밀보장",
            ),
            score=0.7,
        ),
    ]

    deduped = service._dedupe_hits("제44조와 제45조 관계를 설명해줘", "search", hits, None)

    assert [hit.chunk.article_no for hit in deduped].count("제45조") == 1
    assert any(hit.chunk.article_no == "제44조" for hit in deduped)


def test_answer_diagnostics_records_missing_explicit_article_refs(tmp_path):
    settings = make_settings(tmp_path)
    settings.retrieval.max_evidence_per_source_type = 3
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "article-45",
                "law_text",
                "신고자 보호 조문은 신고를 이유로 한 불이익조치를 금지하고 비밀 보장을 언급한다.",
                article_no="제45조",
                article_title="신고자 보호",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    result = service.answer(
        "제44조와 제45조의 관계를 설명해줘.",
        allow_generation=False,
    )

    assert result.diagnostics["explicit_article_refs"] == ["제44조", "제45조"]
    assert "제44조" in result.diagnostics["missing_explicit_refs"]
    assert "제45조" not in result.diagnostics["missing_explicit_refs"]


def test_answer_diagnostics_marks_explicit_article_refs_covered(tmp_path):
    settings = make_settings(tmp_path)
    settings.retrieval.max_evidence_per_source_type = 3
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "article-44",
                "law_text",
                "제44조는 신고자에 대한 비밀보장을 다룬다.",
                article_no="제44조",
                article_title="신고자에 대한 비밀보장",
            ),
            make_chunk(
                "article-45",
                "law_text",
                "제45조는 신고자 보호와 불이익조치 금지를 다룬다.",
                article_no="제45조",
                article_title="신고자 보호",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    result = service.answer(
        "제44조와 제45조의 관계를 설명해줘.",
        allow_generation=False,
    )

    assert result.diagnostics["missing_explicit_refs"] == []


def test_explicit_article_refs_expand_article_ranges(tmp_path):
    settings = make_settings(tmp_path)
    service = AnswerService(settings, store=ChromaStore(settings))

    refs = service._explicit_article_refs("제43조~제45조를 중심으로 설명해줘.")

    assert refs == ["제43조", "제44조", "제45조"]


def test_retrieve_respects_selected_source_types(tmp_path):
    settings = make_settings(tmp_path)
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk("law", "law_text", "휴가 관련 현행 조문이다."),
            make_chunk("history", "history_note", "휴가 규정의 연혁과 개정 흐름을 설명한다."),
        ]
    )
    service = AnswerService(settings, store=store)

    intent, _, hits = service.retrieve("휴가 규정이 왜 바뀌었어?", source_types=["history_note"])

    assert intent == "explain_change"
    assert hits
    assert all(hit.chunk.source_type == "history_note" for hit in hits)


def test_retrieve_limits_hits_per_source_type(tmp_path):
    settings = make_settings(tmp_path)
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk("reason-1", "revision_reason", "휴가 개정 이유와 배경을 설명한다."),
            make_chunk("reason-2", "revision_reason", "휴가 개정 이유를 추가로 설명한다."),
            make_chunk("law-1", "law_text", "휴가 조문과 시행 기준을 설명한다."),
        ]
    )
    service = AnswerService(settings, store=store)

    _, _, hits = service.retrieve("휴가 규정이 왜 바뀌었고 현재 기준은 뭐야?")

    source_types = [hit.chunk.source_type for hit in hits]
    assert source_types.count("revision_reason") == 1


def test_history_question_adds_related_military_service_rules_hit(tmp_path):
    settings = make_settings(tmp_path)
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "current-history",
                "history_note",
                "군인의 지위 및 복무에 관한 기본법 체계로 정리되면서 세부 기준은 시행령으로 넘어왔다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "legacy-history",
                "history_note",
                "군인복무규율 체계에서 현재 기본법과 시행령 체계로 발전한 연혁을 설명한다.",
                law_name="군인복무규율",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    _, _, hits = service.retrieve(
        "과거 휴가 기준이 군인복무규율에서 어떻게 이어졌는지 알려줘",
        law_name="군인의 지위 및 복무에 관한 기본법",
    )

    assert any("군인복무규율" in hit.chunk.law_name or "군인복무규율" in hit.chunk.text for hit in hits)


def test_question_terms_drop_generic_request_words(tmp_path):
    settings = make_settings(tmp_path)
    service = AnswerService(settings, store=ChromaStore(settings))

    question_terms = service._question_terms("징계관련 내용을 정리해서 나열해줘.")
    topic_terms = service._topic_terms("징계관련 내용을 정리해서 나열해줘.")

    assert "내용을" not in question_terms
    assert "정리해서" not in question_terms
    assert "나열해줘" not in question_terms
    assert "징계" in topic_terms


def test_question_terms_drop_additional_request_words(tmp_path):
    settings = make_settings(tmp_path)
    service = AnswerService(settings, store=ChromaStore(settings))

    question_terms = service._question_terms("징계 사항 위주로 좀 요약해줘.")
    topic_terms = service._topic_terms("징계 사항 위주로 좀 요약해줘.")

    assert "사항" not in question_terms
    assert "위주로" not in question_terms
    assert "요약해줘" not in question_terms
    assert "징계" in topic_terms


def test_discipline_search_includes_history_note_support(tmp_path):
    settings = make_settings(tmp_path)
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "discipline-law",
                "law_text",
                "신고등을 이유로 신고자에게 징계조치 등 불이익조치를 해서는 아니 된다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "discipline-history",
                "history_note",
                "형사피의자와 피고인 및 징계혐의자는 외출ㆍ외박 및 휴가를 일시 보류할 수 있다.",
                law_name="군인복무규율",
            ),
            make_chunk(
                "discipline-reason",
                "revision_reason",
                "군 내 기본권 침해를 줄이고 신고한 군인을 보호하기 위해 관련 기준을 법률 단계에서 정비하였다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "discipline-law-2",
                "law_text",
                "군인은 구타, 폭언, 가혹행위 및 집단 따돌림 등을 알게 된 경우 즉시 신고하여야 한다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "irrelevant-counselor",
                "law_text",
                "전문상담관을 선발하려는 경우에는 서류전형과 면접시험을 거쳐 적격자를 선발해야 한다.",
                law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    _, _, hits = service.retrieve("징계관련 내용을 정리해서 나열해줘.")

    assert any(hit.chunk.source_type == "history_note" for hit in hits)
    assert any("징계혐의자" in hit.chunk.text for hit in hits if hit.chunk.source_type == "history_note")
    assert any(hit.chunk.source_type == "law_text" and "전문상담관" not in hit.chunk.text for hit in hits)


def test_discipline_search_handles_paraphrase_variants(tmp_path):
    settings = make_settings(tmp_path)
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "discipline-law",
                "law_text",
                "신고를 이유로 신고자에게 징계조치 등 불이익조치를 해서는 아니 된다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "discipline-report",
                "law_text",
                "군인은 구타, 폭언, 가혹행위 등 사적 제재를 알게 되면 즉시 신고하여야 한다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "discipline-history",
                "history_note",
                "징계혐의자는 외출ㆍ외박 및 휴가를 일시 보류할 수 있다.",
                law_name="군인복무규율",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    for question in [
        "징계사항 좀 알려줘",
        "징계 쪽 규정 위주로 요약해줘",
        "징계 관련 내용 한번 정리해줘",
    ]:
        _, _, hits = service.retrieve(question)
        combined = " ".join(hit.chunk.text for hit in hits)
        assert "징계조치" in combined or "사적 제재" in combined


def test_timeline_question_prefers_target_law_and_predecessor_hits(tmp_path):
    settings = make_settings(tmp_path)
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "basic-reason",
                "revision_reason",
                "군 내 기본권 침해를 줄이고 신고한 군인을 보호하기 위해 징계조치 등 불이익조치를 금지하는 방향으로 제정되었다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "basic-law",
                "law_text",
                "신고자에게 징계조치 등 어떠한 신분상 불이익도 주어서는 아니 된다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "legacy-history",
                "history_note",
                "군인복무규율에서는 징계혐의자를 외출·외박 및 휴가 보류 대상으로 직접 적시하였다.",
                law_name="군인복무규율",
            ),
            make_chunk(
                "irrelevant-decree",
                "law_text",
                "비상소집은 국가비상사태가 발생한 때에 발령한다.",
                law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    intent, _, hits = service.retrieve("징계 관련 기본법 변천사를 알려줘")

    assert intent == "explain_change"
    assert any(hit.chunk.law_name == "군인의 지위 및 복무에 관한 기본법" for hit in hits)
    assert any(hit.chunk.law_name == "군인복무규율" for hit in hits)
    assert all("비상소집" not in hit.chunk.text for hit in hits)


def test_history_question_keeps_revision_reason_even_when_keyword_overlap_is_weak(tmp_path):
    settings = make_settings(tmp_path)
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "history-hit",
                "history_note",
                "군인복무규율에서는 가혹행위와 관련한 군기 기준을 별도 조문에서 다루었다.",
                law_name="군인복무규율",
            ),
            make_chunk(
                "current-law",
                "law_text",
                "군인은 가혹행위 사실을 알게 되면 즉시 신고하여야 한다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "reason-hit",
                "revision_reason",
                "군 내 기본권 침해를 줄이고 권리 보호 기준을 법률 단계에서 더 분명히 하려는 취지로 제정되었다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    _, _, hits = service.retrieve("가혹행위 신고 관련 규정 변천사 알려줘")

    assert any(hit.chunk.source_type == "revision_reason" for hit in hits)


def test_childcare_change_question_prefers_family_care_evidence(tmp_path):
    settings = make_settings(tmp_path)
    settings.retrieval.max_evidence_per_source_type = 2
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "childcare-reason",
                "revision_reason",
                "일·가정 양립과 저출생 대응을 위해 자녀돌봄휴가, 배우자 출산휴가, 육아시간 지원을 확대한다.",
                law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            ),
            make_chunk(
                "childcare-law",
                "law_text",
                "청원휴가 조문에 따라 배우자 출산휴가와 자녀돌봄 사유에 관한 세부 기준을 정한다.",
                law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            ),
            make_chunk(
                "childcare-leave",
                "law_text",
                "육아휴직과 복직 절차는 별도 인사 규정과 연계하여 운영한다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "irrelevant",
                "law_text",
                "임영 및 임관 선서는 군인이 입영 후 실시한다.",
                law_name="군인의 지위 및 복무에 관한 기본법 시행령",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    intent, _, hits = service.retrieve("군에서 육아 관련 휴가나 휴직에 대한 내용과 그 변화를 설명해줘")

    combined = " ".join(hit.chunk.text for hit in hits)
    assert intent == "explain_change"
    assert "자녀돌봄휴가" in combined or "배우자 출산휴가" in combined
    assert "육아휴직" in combined or "복직" in combined
    assert "임영 및 임관 선서" not in combined


def test_service_regulation_transition_question_collects_repeal_remedy_and_personnel_act_links(tmp_path):
    settings = make_settings(tmp_path)
    settings.retrieval.max_evidence_per_source_type = 1
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "legacy-repeal",
                "old_new_comparison",
                "부칙에서 군인복무규율은 폐지한다고 정하였다.",
                law_name="군인복무규율",
            ),
            make_chunk(
                "basic-reason",
                "revision_reason",
                "군 내 기본권 침해를 줄이고 군인의 의무와 권리 보호 기준을 법률 단계에서 정비하려는 취지로 제정되었다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "grievance-law",
                "law_text",
                "군인은 고충을 제기할 수 있고 군인고충심사위원회는 고충 처리 절차를 담당한다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "personnel-link",
                "old_new_comparison",
                "군인사법 제46조, 제47조, 제51조의3 및 제51조의4를 삭제하고 기본법 체계로 옮기는 경과조치를 둔다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "discipline-law",
                "law_text",
                "신고를 이유로 신고자에게 징계조치 등 불이익조치를 해서는 아니 된다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
            ),
            make_chunk(
                "personnel-discipline",
                "law_text",
                "군인사법은 징계의 종류와 징계위원회 심의에 관한 기준을 둔다.",
                law_name="군인사법",
            ),
            make_chunk(
                "discipline-decree",
                "law_text",
                "군인 징계령은 징계 절차와 징계위원회 운영에 필요한 사항을 정한다.",
                law_name="군인 징계령",
            ),
            make_chunk(
                "discipline-rule",
                "law_text",
                "군인 징계령 시행규칙은 징계 양정 기준과 관련 서식을 정한다.",
                law_name="군인 징계령 시행규칙",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    intent, _, hits = service.retrieve(
        "군인복무규율이 2016년 폐지된 뒤, 복무·군기·권리구제 사항은 기본법 체계로 어떻게 재편되었고, "
        "징계의 종류·절차는 현재 어떤 법령이 담당하는지 구분해서 설명해줘."
    )
    combined = " ".join(hit.chunk.text for hit in hits)

    assert intent == "hybrid"
    assert "군인복무규율" in combined and "폐지" in combined
    assert "군인고충심사위원회" in combined
    assert "군인사법 제46조" in combined
    assert "징계의 종류" in combined or "징계 절차" in combined
    assert "기본권" in combined


def test_question_with_explicit_article_number_injects_that_article(tmp_path):
    settings = make_settings(tmp_path)
    settings.retrieval.max_evidence_per_source_type = 1
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "report-duty",
                "law_text",
                "군인은 가혹행위 사실을 알게 되면 즉시 신고하여야 한다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
                article_no="제43조",
                article_title="신고의무 등",
            ),
            make_chunk(
                "report-secret",
                "law_text",
                "신고자라는 사정이나 인적사항을 다른 사람에게 알려주거나 공개하여서는 아니 된다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
                article_no="제44조",
                article_title="신고자에 대한 비밀보장",
            ),
            make_chunk(
                "report-protection",
                "law_text",
                "신고등을 이유로 신고자에게 징계조치 등 불이익조치를 해서는 아니 된다.",
                law_name="군인의 지위 및 복무에 관한 기본법",
                article_no="제45조",
                article_title="신고자 보호",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    _, _, hits = service.retrieve(
        "현행 군인의 지위 및 복무에 관한 기본법상 신고자 보호 규정을 조문번호와 함께 제시하고, "
        "제44조의 비밀보장과 어떤 관계인지 3문장으로 설명해줘."
    )

    assert any(hit.chunk.article_no == "제44조" for hit in hits)
    assert any(hit.chunk.article_no == "제45조" for hit in hits)


def test_answer_body_prioritizes_explicit_article_refs(tmp_path):
    settings = make_settings(tmp_path)
    settings.retrieval.max_evidence_per_source_type = 4
    store = ChromaStore(settings)
    store.upsert(
        [
            make_chunk(
                "report-duty",
                "law_text",
                "군인은 가혹행위 등을 알게 되면 즉시 보고하거나 신고하여야 한다.",
                article_no="제43조",
                article_title="신고의무 등",
            ),
            make_chunk(
                "report-secret",
                "law_text",
                "신고자의 인적사항이나 신고자임을 알 수 있는 사실을 공개해서는 안 된다.",
                article_no="제44조",
                article_title="신고자에 대한 비밀보장",
            ),
            make_chunk(
                "report-protection",
                "law_text",
                "신고등을 이유로 신고자에게 징계조치 등 불이익조치를 해서는 안 된다.",
                article_no="제45조",
                article_title="신고자 보호",
            ),
            make_chunk(
                "private-sanction",
                "law_text",
                "군인은 구타, 폭언, 가혹행위 등 사적 제재를 하여서는 안 된다.",
                article_no="제26조",
                article_title="사적 제재 및 직권남용의 금지",
            ),
        ]
    )
    service = AnswerService(settings, store=store)

    result = service.answer(
        "현행 기본법상 신고자 보호 규정을 조문번호와 함께 제시하고, 제44조의 비밀보장과 어떤 관계인지 설명해줘.",
        allow_generation=False,
    )

    main_section = result.answer_markdown.split("### 주요 규정\n", 1)[1].split("\n\n### 실무 참고", 1)[0]
    assert "제44조" in main_section
    assert "비밀보장" in main_section
