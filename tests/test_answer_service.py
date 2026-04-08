from __future__ import annotations

from army_reg_rag.config import AppConfig, DataConfig, RetrievalConfig, Settings
from army_reg_rag.domain.models import DocumentChunk
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.services.answer_service import AnswerService


def make_chunk(
    chunk_id: str,
    source_type: str,
    text: str,
    *,
    law_name: str = "군인의 지위 및 복무에 관한 기본법",
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        law_name=law_name,
        law_level="법률",
        source_type=source_type,
        version_label="현행",
        promulgation_date="2026-01-01",
        effective_date="2026-01-01",
        article_no="제1조",
        article_title="테스트",
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
