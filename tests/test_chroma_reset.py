from __future__ import annotations

from army_reg_rag.config import AppConfig, Settings
from army_reg_rag.domain.models import DocumentChunk
from army_reg_rag.retrieval.chroma_store import ChromaStore


def test_store_reset_clears_collection(tmp_path):
    settings = Settings(app=AppConfig(chroma_path=str(tmp_path / "chroma")))
    settings.ensure_runtime_dirs()
    store = ChromaStore(settings)

    store.upsert(
        [
            DocumentChunk(
                id="reset-1",
                law_name="군인의 지위 및 복무에 관한 기본법",
                law_level="법률",
                source_type="law_text",
                version_label="현행 본문",
                promulgation_date="2025-01-07",
                effective_date="2026-01-08",
                article_no="제1조",
                article_title="목적",
                revision_kind="일부개정",
                text="이 법은 목적을 정한다.",
                source_url="https://example.com",
            )
        ]
    )

    assert store.count() == 1
    store.reset()
    assert store.count() == 0
