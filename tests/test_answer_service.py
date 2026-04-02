from __future__ import annotations

from army_reg_rag.config import AppConfig, DataConfig, RetrievalConfig, Settings
from army_reg_rag.domain.models import DocumentChunk
from army_reg_rag.services.answer_service import AnswerService
from army_reg_rag.retrieval.chroma_store import ChromaStore


def make_chunk(chunk_id: str, source_type: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        law_name="군인의 지위 및 복무에 관한 기본법",
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

    intent, _, hits = service.retrieve("왜 휴가 규정이 바뀌었어?", source_types=["history_note"])

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
            make_chunk("law-1", "law_text", "휴가 조문과 시행일을 설명한다."),
        ]
    )
    service = AnswerService(settings, store=store)

    _, _, hits = service.retrieve("휴가 규정이 왜 바뀌었는지와 현재 기준을 보여줘.")

    source_types = [hit.chunk.source_type for hit in hits]
    assert source_types.count("revision_reason") == 1
