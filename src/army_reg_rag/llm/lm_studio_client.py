from __future__ import annotations

from typing import Any

import requests

from army_reg_rag.config import Settings
from army_reg_rag.llm.gemini_client import GeminiAnswerClient, GeneratedAnswer, QUOTA_BLOCK_MESSAGE
from army_reg_rag.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from army_reg_rag.llm.usage_tracker import GeminiUsageTracker
from army_reg_rag.utils.runtime_config import get_runtime_value

DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"


class LMStudioUsageTracker(GeminiUsageTracker):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.state_path = settings.runtime_dir / "lm_studio_usage.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)


class LMStudioAnswerClient(GeminiAnswerClient):
    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float | None = None,
        api_key: str | None = None,
        enforce_limits: bool = True,
    ):
        self.settings = settings
        configured_base_url = str(
            get_runtime_value("LM_STUDIO_BASE_URL", base_url or DEFAULT_LM_STUDIO_BASE_URL)
        ).strip()
        configured_model_name = str(
            get_runtime_value("LM_STUDIO_MODEL", model_name or get_runtime_value("LM_STUDIO_MODEL_NAME", ""))
        ).strip()
        configured_timeout = get_runtime_value("LM_STUDIO_TIMEOUT_SECONDS", timeout_seconds or 120)

        self.base_url = self._normalize_base_url(configured_base_url)
        self.native_base_url = self._native_base_url(self.base_url)
        self.model_name = configured_model_name
        self.timeout_seconds = float(configured_timeout)
        self.api_key = str(get_runtime_value("LM_STUDIO_API_KEY", api_key or "")).strip()
        self.enforce_limits = enforce_limits
        self.usage_tracker = LMStudioUsageTracker(settings)
        self._client = None
        self._visible_model_cache: list[str] | None = None
        self._loaded_model_cache: list[str] | None = None
        self._last_resolved_model = ""

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = (base_url or DEFAULT_LM_STUDIO_BASE_URL).strip().rstrip("/")
        if not normalized:
            return DEFAULT_LM_STUDIO_BASE_URL
        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1"

    @staticmethod
    def _native_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            return f"{normalized[:-3]}/api/v1"
        return f"{normalized}/api/v1"

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def available_models(self, *, force_refresh: bool = False) -> list[str]:
        if self._visible_model_cache is not None and not force_refresh:
            return list(self._visible_model_cache)

        response = requests.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        raw_models = payload.get("data", [])
        models: list[str] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, dict):
                    continue
                model_id = str(item.get("id", "")).strip()
                if model_id:
                    models.append(model_id)

        self._visible_model_cache = self._dedupe(models)
        return list(self._visible_model_cache)

    def loaded_models(self, *, force_refresh: bool = False) -> list[str]:
        if self._loaded_model_cache is not None and not force_refresh:
            return list(self._loaded_model_cache)

        response = requests.get(
            f"{self.native_base_url}/models",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        raw_models = payload.get("models", [])
        loaded_models: list[str] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).strip() != "llm":
                    continue
                loaded_instances = item.get("loaded_instances", [])
                if not isinstance(loaded_instances, list) or not loaded_instances:
                    continue

                model_key = str(item.get("key", "")).strip()
                if model_key:
                    loaded_models.append(model_key)
                    continue

                for instance in loaded_instances:
                    if not isinstance(instance, dict):
                        continue
                    instance_id = str(instance.get("id", "")).strip()
                    if instance_id:
                        loaded_models.append(instance_id)
                        break

        self._loaded_model_cache = self._dedupe(loaded_models)
        return list(self._loaded_model_cache)

    def resolve_model_name(self, *, force_refresh: bool = False) -> str:
        if self.model_name:
            self._last_resolved_model = self.model_name
            return self.model_name

        loaded_models: list[str] = []
        native_error: Exception | None = None
        try:
            loaded_models = self.loaded_models(force_refresh=force_refresh)
        except requests.RequestException as exc:
            native_error = exc

        if len(loaded_models) == 1:
            self._last_resolved_model = loaded_models[0]
            return loaded_models[0]

        if len(loaded_models) > 1:
            raise RuntimeError(
                "LM Studio reports multiple loaded LLMs. The documented API exposes loaded models, "
                "but not the GUI-selected current model. Unload the others or set LM_STUDIO_MODEL."
            )

        visible_models = self.available_models(force_refresh=force_refresh)
        if len(visible_models) == 1 and native_error is not None:
            self._last_resolved_model = visible_models[0]
            return visible_models[0]

        if native_error is not None:
            raise RuntimeError(
                "LM Studio is reachable, but loaded-model inspection via /api/v1/models failed. "
                "Auto-follow mode needs exactly one loaded LLM."
            ) from native_error

        raise RuntimeError("LM Studio is reachable but there is no loaded LLM yet.")

    def describe_connection(self) -> dict[str, Any]:
        try:
            visible_models = self.available_models(force_refresh=True)
        except requests.RequestException as exc:
            return {
                "available": False,
                "message": f"LM Studio connection failed: {exc}",
                "models": [],
                "loaded_models": [],
                "base_url": self.base_url,
                "selected_model": self.model_name,
                "resolved_model": "",
            }

        loaded_models: list[str] = []
        native_error = ""
        try:
            loaded_models = self.loaded_models(force_refresh=True)
        except requests.RequestException as exc:
            native_error = str(exc)

        if self.model_name:
            return {
                "available": True,
                "message": f"LM Studio is ready with explicit override '{self.model_name}'.",
                "models": visible_models,
                "loaded_models": loaded_models,
                "base_url": self.base_url,
                "selected_model": self.model_name,
                "resolved_model": self.model_name,
                "native_error": native_error,
            }

        if len(loaded_models) == 1:
            return {
                "available": True,
                "message": "LM Studio is ready. Auto-follow mode is using the single loaded LLM.",
                "models": visible_models,
                "loaded_models": loaded_models,
                "base_url": self.base_url,
                "selected_model": "",
                "resolved_model": loaded_models[0],
                "native_error": native_error,
            }

        if len(loaded_models) > 1:
            return {
                "available": False,
                "message": (
                    "LM Studio has multiple loaded LLMs. The documented API exposes loaded models, "
                    "but not the GUI-selected current model, so auto-follow mode cannot disambiguate."
                ),
                "models": visible_models,
                "loaded_models": loaded_models,
                "base_url": self.base_url,
                "selected_model": "",
                "resolved_model": "",
                "native_error": native_error,
            }

        message = "LM Studio responded, but there is no loaded LLM yet."
        if native_error:
            message = (
                "LM Studio is reachable, but loaded-model inspection via /api/v1/models failed. "
                "Auto-follow mode needs exactly one loaded LLM."
            )
        return {
            "available": False,
            "message": message,
            "models": visible_models,
            "loaded_models": loaded_models,
            "base_url": self.base_url,
            "selected_model": "",
            "resolved_model": "",
            "native_error": native_error,
        }

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
            return "\n".join(parts).strip()
        return ""

    def _effective_snapshot(self) -> dict[str, Any]:
        snapshot = self.usage_tracker.snapshot()
        if self.enforce_limits:
            return snapshot
        return {
            **snapshot,
            "can_generate": True,
            "hard_blocked": False,
            "block_reason": "",
            "remaining_requests": -1,
            "request_soft_limit": 0,
            "request_limit": 0,
        }

    def generate_answer(
        self,
        question: str,
        intent: str,
        evidence: list,
        *,
        allow_generation: bool = True,
    ) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer(
                text=self._no_evidence_answer(intent),
                backend="retrieval_only",
                quota_snapshot=self.usage_tracker.snapshot(),
            )

        if not allow_generation:
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                quota_snapshot=self.usage_tracker.snapshot(),
            )

        snapshot = self._effective_snapshot()
        if self.enforce_limits and not snapshot["can_generate"]:
            return self._quota_block_result(snapshot)

        try:
            resolved_model = self.resolve_model_name(force_refresh=not bool(self.model_name))
            user_prompt = build_user_prompt(question=question, intent=intent, evidence=evidence)
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": resolved_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.settings.llm.temperature,
                    "max_tokens": self.settings.llm.max_output_tokens,
                    "stream": False,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice=f"LM Studio request failed, so the app switched to evidence-only summary mode. ({exc})",
                quota_snapshot=self._effective_snapshot(),
            )
        except RuntimeError as exc:
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice=f"{exc} The app switched to evidence-only summary mode.",
                quota_snapshot=self._effective_snapshot(),
            )

        final_text = self._extract_response_text(payload)
        if not final_text:
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice="LM Studio returned an empty answer, so the app switched to evidence-only summary mode.",
                quota_snapshot=self._effective_snapshot(),
            )
        if not self._is_structured_answer(question, intent, final_text):
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice="LM Studio answer did not follow the required grounded format, so the app switched to evidence-only summary mode.",
                quota_snapshot=self._effective_snapshot(),
            )

        usage = payload.get("usage", {})
        prompt_tokens = self._usage_value(usage, "prompt_tokens", "promptTokens")
        candidate_tokens = self._usage_value(usage, "completion_tokens", "completionTokens")
        total_tokens = self._usage_value(usage, "total_tokens", "totalTokens")
        updated_snapshot = self.usage_tracker.record_success(
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            total_tokens=total_tokens,
        )
        effective_snapshot = self._effective_snapshot()
        notice = QUOTA_BLOCK_MESSAGE if self.enforce_limits and not updated_snapshot["can_generate"] else ""

        return GeneratedAnswer(
            text=final_text,
            backend="lm_studio",
            notice=notice,
            quota_snapshot={
                **effective_snapshot,
                "base_url": self.base_url,
                "model_name": self._last_resolved_model or self.model_name,
            },
        )
