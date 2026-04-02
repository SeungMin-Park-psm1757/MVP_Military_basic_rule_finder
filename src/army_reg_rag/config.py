from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "army-reg-rag-mvp"
    timezone: str = "Asia/Seoul"
    daily_limit: int = 20
    max_question_chars: int = 600
    chroma_path: str = "data/chroma"
    collection_name: str = "army_reg_rag"
    allow_debug_tab: bool = True
    default_law_filter: str = "전체"


class RetrievalConfig(BaseModel):
    top_k: int = 6
    max_evidence_per_source_type: int = 3


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fallback_dim: int = 384


class LLMConfig(BaseModel):
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.1
    max_output_tokens: int = 1400
    daily_request_budget: int = 200
    daily_token_budget: int = 1000000
    budget_cutoff_ratio: float = 0.9


class DataConfig(BaseModel):
    demo_input_path: str = "data/sample/processed/sample_documents.jsonl"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    runtime_dir: str = "data/runtime"


class UIConfig(BaseModel):
    evidence_preview_chars: int = 320
    answer_disclaimer: str = "이 답변은 법률자문이 아니라 공개 법규 기반의 실무 참고용 안내입니다."


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def chroma_path(self) -> Path:
        return self.project_root / self.app.chroma_path

    @property
    def runtime_dir(self) -> Path:
        return self.project_root / self.data.runtime_dir

    @property
    def demo_input_path(self) -> Path:
        return self.project_root / self.data.demo_input_path

    @property
    def processed_dir(self) -> Path:
        return self.project_root / self.data.processed_dir

    @property
    def raw_dir(self) -> Path:
        return self.project_root / self.data.raw_dir

    def ensure_runtime_dirs(self) -> None:
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(config_path: str | Path | None = None) -> Settings:
    default_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    path = Path(config_path) if config_path else default_path
    if not path.exists():
        settings = Settings()
        settings.ensure_runtime_dirs()
        return settings

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    env_override = {}
    if os.getenv("APP_DAILY_LIMIT"):
        env_override.setdefault("app", {})["daily_limit"] = int(os.getenv("APP_DAILY_LIMIT", "20"))
    if os.getenv("APP_MAX_QUESTION_CHARS"):
        env_override.setdefault("app", {})["max_question_chars"] = int(os.getenv("APP_MAX_QUESTION_CHARS", "600"))
    if os.getenv("GEMINI_MODEL_NAME"):
        env_override.setdefault("llm", {})["model_name"] = os.getenv("GEMINI_MODEL_NAME")
    if os.getenv("GEMINI_DAILY_REQUEST_BUDGET"):
        env_override.setdefault("llm", {})["daily_request_budget"] = int(os.getenv("GEMINI_DAILY_REQUEST_BUDGET", "20"))
    if os.getenv("GEMINI_DAILY_TOKEN_BUDGET"):
        env_override.setdefault("llm", {})["daily_token_budget"] = int(os.getenv("GEMINI_DAILY_TOKEN_BUDGET", "120000"))
    if os.getenv("GEMINI_BUDGET_CUTOFF_RATIO"):
        env_override.setdefault("llm", {})["budget_cutoff_ratio"] = float(os.getenv("GEMINI_BUDGET_CUTOFF_RATIO", "0.9"))
    if os.getenv("CHROMA_COLLECTION_NAME"):
        env_override.setdefault("app", {})["collection_name"] = os.getenv("CHROMA_COLLECTION_NAME")

    merged = _deep_merge(raw, env_override)
    settings = Settings.model_validate(merged)
    settings.ensure_runtime_dirs()
    return settings
