from __future__ import annotations

from types import SimpleNamespace

from army_reg_rag.config import AppConfig, DataConfig, Settings
from army_reg_rag.domain.models import DocumentChunk, SearchHit
from army_reg_rag.llm import gemini_client as gemini_module
from army_reg_rag.llm.gemini_client import (
    GeminiAnswerClient,
    PROVIDER_RATE_LIMIT_NOTICE,
    QUOTA_BLOCK_MESSAGE,
)


def make_settings(tmp_path) -> Settings:
    settings = Settings(
        app=AppConfig(chroma_path=str(tmp_path / "chroma")),
        data=DataConfig(runtime_dir=str(tmp_path / "runtime")),
    )
    settings.ensure_runtime_dirs()
    return settings


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
            article_title="휴가의 보장",
            revision_kind="일부개정",
            text=(
                "군인은 대통령령에 따라 휴가, 외출, 외박을 보장받는다. "
                "지휘관은 작전상황이나 교육훈련 등의 사유가 있으면 제한할 수 있다."
            ),
            source_url="https://example.com",
        ),
        score=0.9,
    )


def test_generate_answer_falls_back_when_local_generation_budget_is_blocked(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    client.usage_tracker.block_for_today("local generation limit")

    result = client.generate_answer(
        "휴가 관련 현행 규정을 설명해줘.",
        "search",
        [make_hit()],
        allow_generation=True,
    )

    assert result.backend == "retrieval_fallback"
    assert result.notice == QUOTA_BLOCK_MESSAGE
    assert "###" in result.text


def test_generate_answer_does_not_hard_block_day_on_provider_429(tmp_path, monkeypatch):
    class TooManyRequestsError(Exception):
        status_code = 429

    class FakeModels:
        def generate_content(self, *args, **kwargs):
            raise TooManyRequestsError("RESOURCE_EXHAUSTED")

    client = GeminiAnswerClient(make_settings(tmp_path))
    client._client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(
        gemini_module,
        "types",
        SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs),
    )

    result = client.generate_answer(
        "휴가 관련 현행 규정을 설명해줘.",
        "search",
        [make_hit()],
        allow_generation=True,
    )

    assert result.backend == "retrieval_fallback"
    assert result.notice == PROVIDER_RATE_LIMIT_NOTICE
    assert client.usage_tracker.snapshot()["hard_blocked"] is False
