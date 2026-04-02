from army_reg_rag.config import Settings, AppConfig
from army_reg_rag.domain.models import DocumentChunk
from army_reg_rag.retrieval.chroma_store import ChromaStore


def test_store_upsert_and_query(tmp_path):
    settings = Settings(app=AppConfig(chroma_path=str(tmp_path / "chroma")))
    settings.ensure_runtime_dirs()
    store = ChromaStore(settings)

    chunk = DocumentChunk(
        id="1",
        law_name="테스트 법령",
        law_level="법률",
        source_type="law_text",
        version_label="현행",
        promulgation_date="2026-01-01",
        effective_date="2026-01-01",
        article_no="제1조",
        article_title="목적",
        revision_kind="제정",
        text="이 조문은 휴가와 복무에 관한 내용을 담고 있다.",
        source_url="https://example.com",
    )
    store.upsert([chunk])
    hits = store.query("휴가 규정", top_k=3)
    assert hits
    assert hits[0].chunk.law_name == "테스트 법령"
