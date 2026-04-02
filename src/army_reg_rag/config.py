from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from army_reg_rag.utils.runtime_config import get_runtime_bool, get_runtime_value


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
    app_daily_limit = get_runtime_value("APP_DAILY_LIMIT")
    if app_daily_limit:
        env_override.setdefault("app", {})["daily_limit"] = int(app_daily_limit)

    max_question_chars = get_runtime_value("APP_MAX_QUESTION_CHARS")
    if max_question_chars:
        env_override.setdefault("app", {})["max_question_chars"] = int(max_question_chars)

    chroma_path = get_runtime_value("APP_CHROMA_PATH")
    if chroma_path:
        env_override.setdefault("app", {})["chroma_path"] = str(chroma_path)

    if get_runtime_value("APP_ALLOW_DEBUG_TAB", None) is not None:
        env_override.setdefault("app", {})["allow_debug_tab"] = get_runtime_bool("APP_ALLOW_DEBUG_TAB", True)

    default_law_filter = get_runtime_value("APP_DEFAULT_LAW_FILTER")
    if default_law_filter:
        env_override.setdefault("app", {})["default_law_filter"] = str(default_law_filter)

    runtime_dir = get_runtime_value("APP_RUNTIME_DIR")
    if runtime_dir:
        env_override.setdefault("data", {})["runtime_dir"] = str(runtime_dir)

    gemini_model_name = get_runtime_value("GEMINI_MODEL_NAME")
    if gemini_model_name:
        env_override.setdefault("llm", {})["model_name"] = str(gemini_model_name)

    request_budget = get_runtime_value("GEMINI_DAILY_REQUEST_BUDGET")
    if request_budget:
        env_override.setdefault("llm", {})["daily_request_budget"] = int(request_budget)

    token_budget = get_runtime_value("GEMINI_DAILY_TOKEN_BUDGET")
    if token_budget:
        env_override.setdefault("llm", {})["daily_token_budget"] = int(token_budget)

    budget_cutoff = get_runtime_value("GEMINI_BUDGET_CUTOFF_RATIO")
    if budget_cutoff:
        env_override.setdefault("llm", {})["budget_cutoff_ratio"] = float(budget_cutoff)

    collection_name = get_runtime_value("CHROMA_COLLECTION_NAME")
    if collection_name:
        env_override.setdefault("app", {})["collection_name"] = str(collection_name)

    merged = _deep_merge(raw, env_override)
    settings = Settings.model_validate(merged)
    settings.ensure_runtime_dirs()
    return settings
