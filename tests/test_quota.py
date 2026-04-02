from __future__ import annotations

from army_reg_rag.config import Settings, AppConfig, DataConfig
from army_reg_rag.utils.quota import LocalDailyQuota


def test_quota_consumes_and_resets(tmp_path):
    settings = Settings(
        app=AppConfig(daily_limit=3, chroma_path="data/chroma"),
        data=DataConfig(runtime_dir=str(tmp_path / "runtime")),
    )
    settings.ensure_runtime_dirs()
    quota = LocalDailyQuota(settings)

    assert quota.remaining() == 3
    quota.consume()
    assert quota.remaining() == 2
