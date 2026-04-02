from __future__ import annotations

from army_reg_rag.config import AppConfig, DataConfig, LLMConfig, Settings
from army_reg_rag.llm.usage_tracker import GeminiUsageTracker


def make_settings(tmp_path) -> Settings:
    settings = Settings(
        app=AppConfig(chroma_path=str(tmp_path / "chroma")),
        data=DataConfig(runtime_dir=str(tmp_path / "runtime")),
        llm=LLMConfig(
            model_name="gemini-2.0-flash",
            daily_request_budget=10,
            daily_token_budget=1000,
            budget_cutoff_ratio=0.9,
        ),
    )
    settings.ensure_runtime_dirs()
    return settings


def test_usage_tracker_stops_after_soft_limit(tmp_path):
    tracker = GeminiUsageTracker(make_settings(tmp_path))

    for _ in range(8):
        snapshot = tracker.record_success(prompt_tokens=40, candidate_tokens=10, total_tokens=50)

    assert snapshot["can_generate"] is True

    snapshot = tracker.record_success(prompt_tokens=40, candidate_tokens=10, total_tokens=50)
    assert snapshot["request_count"] == 9
    assert snapshot["can_generate"] is False


def test_usage_tracker_hard_block(tmp_path):
    tracker = GeminiUsageTracker(make_settings(tmp_path))
    snapshot = tracker.block_for_today("quota exhausted")

    assert snapshot["hard_blocked"] is True
    assert snapshot["can_generate"] is False
    assert snapshot["block_reason"] == "quota exhausted"
