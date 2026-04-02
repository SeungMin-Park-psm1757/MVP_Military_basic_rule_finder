from __future__ import annotations

from army_reg_rag.config import load_settings


def test_load_settings_accepts_runtime_overrides(monkeypatch):
    monkeypatch.setenv("APP_CHROMA_PATH", "/tmp/chroma")
    monkeypatch.setenv("APP_RUNTIME_DIR", "/tmp/runtime")
    monkeypatch.setenv("APP_ALLOW_DEBUG_TAB", "false")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "web_demo")

    settings = load_settings()

    assert settings.app.chroma_path == "/tmp/chroma"
    assert settings.data.runtime_dir == "/tmp/runtime"
    assert settings.app.allow_debug_tab is False
    assert settings.app.collection_name == "web_demo"
