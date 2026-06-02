from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

import requests

from army_reg_rag.config import Settings
from army_reg_rag.llm.gemini_client import GeminiAnswerClient, GeneratedAnswer, QUOTA_BLOCK_MESSAGE
from army_reg_rag.llm.prompts import (
    LOCAL_PLAIN_SYSTEM_PROMPT,
    LOCAL_SYSTEM_PROMPT,
    build_local_plain_user_prompt,
    build_local_user_prompt,
)
from army_reg_rag.llm.usage_tracker import GeminiUsageTracker
from army_reg_rag.utils.runtime_config import get_runtime_value

DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
logger = logging.getLogger(__name__)


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
        self.debug_dir = settings.runtime_dir / "lm_studio_debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
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

    @staticmethod
    def _collect_text_fragments(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            return [cleaned] if cleaned else []
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                parts.extend(LMStudioAnswerClient._collect_text_fragments(item))
            return parts
        if isinstance(value, dict):
            parts: list[str] = []
            item_type = str(value.get("type", "")).strip()
            if item_type in {"text", "output_text"} and value.get("text"):
                parts.extend(LMStudioAnswerClient._collect_text_fragments(value.get("text")))
            for key in ("text", "output_text", "content", "value"):
                if key in value:
                    parts.extend(LMStudioAnswerClient._collect_text_fragments(value.get(key)))
            return parts
        return []

    @staticmethod
    def _dedupe_text_parts(parts: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for part in parts:
            cleaned = str(part).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return ordered

    @staticmethod
    def _model_profile_name(model_name: str) -> str:
        lowered = (model_name or "").strip().lower()
        if re.search(r"(?<!\d)(?:1(?:[._-]?2)?|2)b(?!\d)", lowered):
            return "small"
        if re.search(r"(?<!\d)(?:7(?:[._-]?8)?|8)b(?!\d)", lowered):
            return "medium"
        return "default"

    @classmethod
    def _model_profile_label(cls, model_name: str) -> str:
        lowered = (model_name or "").strip().lower()
        if "gpt-oss" in lowered:
            return "gpt_oss_reasoning"
        return f"{cls._model_profile_name(model_name)}_local"

    def _request_options(self, model_name: str, *, retry: bool) -> dict[str, Any]:
        lowered = (model_name or "").strip().lower()
        if "gpt-oss" in lowered:
            return {
                "model_profile": self._model_profile_label(model_name),
                "prompt_profile": "medium",
                "temperature": 0.0,
                "max_tokens": 520 if not retry else 900,
            }

        profile = self._model_profile_name(model_name)

        if profile == "small":
            return {
                "model_profile": self._model_profile_label(model_name),
                "prompt_profile": "small",
                "temperature": 0.0,
                "max_tokens": 180 if not retry else 140,
            }
        if profile == "medium":
            return {
                "model_profile": self._model_profile_label(model_name),
                "prompt_profile": "small",
                "temperature": min(self.settings.llm.temperature, 0.05),
                "max_tokens": 260 if not retry else 200,
            }
        return {
            "model_profile": self._model_profile_label(model_name),
            "prompt_profile": "small",
            "temperature": min(self.settings.llm.temperature, 0.1),
            "max_tokens": 320 if not retry else 220,
        }

    @staticmethod
    def _uses_responses_api(model_name: str) -> bool:
        lowered = (model_name or "").strip().lower()
        return "gpt-oss" in lowered

    @staticmethod
    def _prefers_native_chat_retry(model_name: str) -> bool:
        lowered = (model_name or "").strip().lower()
        return "gpt-oss" in lowered

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
        if len(visible_models) == 1:
            self._last_resolved_model = visible_models[0]
            return visible_models[0]

        if len(visible_models) > 1:
            raise RuntimeError(
                "LM Studio has multiple visible LLMs. Auto-follow mode cannot disambiguate."
            )

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

        if len(visible_models) == 1:
            return {
                "available": True,
                "message": "LM Studio is ready. Auto-follow mode is using the single visible LLM.",
                "models": visible_models,
                "loaded_models": loaded_models,
                "base_url": self.base_url,
                "selected_model": "",
                "resolved_model": visible_models[0],
                "native_error": native_error,
            }

        if len(visible_models) > 1:
            return {
                "available": False,
                "message": "LM Studio has multiple visible LLMs. Auto-follow mode cannot disambiguate.",
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
            return "\n".join(
                self._dedupe_text_parts(
                    self._collect_text_fragments(payload.get("output_text"))
                    + self._collect_text_fragments(payload.get("text"))
                    + self._collect_text_fragments(payload.get("content"))
                )
            ).strip()

        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        raw_candidates = [
            message.get("content") if isinstance(message, dict) else None,
            message.get("text") if isinstance(message, dict) else None,
            first_choice.get("text") if isinstance(first_choice, dict) else None,
            payload.get("output_text"),
            payload.get("text"),
            payload.get("content"),
        ]
        parts: list[str] = []
        for candidate in raw_candidates:
            parts.extend(self._collect_text_fragments(candidate))
        return "\n".join(self._dedupe_text_parts(parts)).strip()

    def _extract_responses_text(self, payload: dict[str, Any]) -> str:
        output = payload.get("output", [])
        if not isinstance(output, list):
            output = []

        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text" and content_item.get("text"):
                    parts.append(str(content_item["text"]).strip())

        if not parts:
            for candidate in (
                payload.get("output_text"),
                payload.get("text"),
                payload.get("content"),
                payload.get("response"),
            ):
                parts.extend(self._collect_text_fragments(candidate))
        return "\n".join(self._dedupe_text_parts(parts)).strip()

    def _is_reasoning_only_chat_payload(self, payload: dict[str, Any]) -> bool:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return False
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        content = ""
        if isinstance(message, dict):
            raw_content = message.get("content", "")
            content = raw_content.strip() if isinstance(raw_content, str) else ""
            reasoning_text = str(
                message.get("reasoning_content") or message.get("reasoning") or ""
            ).strip()
            return not content and bool(reasoning_text)
        return False

    def _is_reasoning_only_responses_payload(self, payload: dict[str, Any]) -> bool:
        output = payload.get("output", [])
        if not isinstance(output, list) or not output:
            return False
        has_reasoning = False
        has_message_text = False
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip()
            if item_type == "reasoning":
                has_reasoning = True
                continue
            if item_type != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text" and str(content_item.get("text", "")).strip():
                    has_message_text = True
                    break
        return has_reasoning and not has_message_text

    def _extract_native_chat_text(self, payload: dict[str, Any]) -> str:
        output = payload.get("output", [])
        if not isinstance(output, list):
            return ""
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")).strip() != "message":
                continue
            content = str(item.get("content", "") or "").strip()
            if content:
                parts.append(content)
        return "\n".join(self._dedupe_text_parts(parts)).strip()

    def _is_reasoning_only_native_payload(self, payload: dict[str, Any]) -> bool:
        output = payload.get("output", [])
        if not isinstance(output, list) or not output:
            return False
        has_reasoning = False
        has_message = False
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip()
            if item_type == "reasoning" and str(item.get("content", "") or "").strip():
                has_reasoning = True
            if item_type == "message" and str(item.get("content", "") or "").strip():
                has_message = True
        return has_reasoning and not has_message

    @staticmethod
    def _finish_reason_from_chat_payload(payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return ""
        return str(choices[0].get("finish_reason", "") or "").strip()

    @staticmethod
    def _finish_reason_from_responses_payload(payload: dict[str, Any]) -> str:
        incomplete = payload.get("incomplete_details")
        if isinstance(incomplete, dict):
            reason = str(incomplete.get("reason", "") or "").strip()
            if reason:
                return reason
        return str(payload.get("status", "") or "").strip()

    @staticmethod
    def _finish_reason_from_native_payload(payload: dict[str, Any]) -> str:
        output = payload.get("output", [])
        if not isinstance(output, list) or not output:
            return ""
        if any(isinstance(item, dict) and str(item.get("type", "")).strip() == "message" and str(item.get("content", "") or "").strip() for item in output):
            return "completed"
        if any(isinstance(item, dict) and str(item.get("type", "")).strip() == "reasoning" and str(item.get("content", "") or "").strip() for item in output):
            return "reasoning_only"
        return ""

    @staticmethod
    def _usage_payload(payload: dict[str, Any]) -> dict[str, Any]:
        usage = payload.get("usage", {})
        return usage if isinstance(usage, dict) else {}

    @staticmethod
    def _status_value(response: requests.Response | None) -> int | None:
        if response is None:
            return None
        try:
            return int(response.status_code)
        except Exception:
            return None

    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "").strip()).strip("._-") or "unknown"

    def _attempt_record(
        self,
        *,
        model_name: str,
        endpoint: str,
        request_payload: dict[str, Any],
        retry: bool,
        response: requests.Response | None = None,
        response_payload: dict[str, Any] | None = None,
        parsed_text: str = "",
        reasoning_only: bool = False,
        request_error: str = "",
    ) -> dict[str, Any]:
        payload = response_payload or {}
        if endpoint.endswith("/responses"):
            finish_reason = self._finish_reason_from_responses_payload(payload)
        elif endpoint.endswith("/api/v1/chat"):
            finish_reason = self._finish_reason_from_native_payload(payload)
        else:
            finish_reason = self._finish_reason_from_chat_payload(payload)
        return {
            "model_name": model_name,
            "model_profile": self._model_profile_label(model_name),
            "endpoint": endpoint,
            "http_status": self._status_value(response),
            "request_payload": request_payload,
            "response_payload": payload,
            "parsed_text": parsed_text,
            "parsed_text_length": len(parsed_text or ""),
            "reasoning_only": bool(reasoning_only),
            "finish_reason": finish_reason,
            "token_usage": self._usage_payload(payload),
            "retry": retry,
            "request_error": request_error,
        }

    def _write_debug_artifact(
        self,
        *,
        attempts: list[dict[str, Any]],
        summary: dict[str, Any],
        question: str,
        intent: str,
        validator: dict[str, Any] | None = None,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        model_slug = self._slugify(summary.get("model_name", "unknown"))
        endpoint_slug = self._slugify(summary.get("endpoint", "unknown"))
        failure_slug = self._slugify(summary.get("failure_type", "unknown"))
        artifact_path = self.debug_dir / f"{timestamp}_{model_slug}_{endpoint_slug}_{failure_slug}.json"
        
        summary_copy = dict(summary)
        
        artifact_payload = {
            "timestamp": timestamp,
            "question": question,
            "intent": intent,
            "summary": summary_copy,
            "validator": validator or {},
            "attempts": attempts,
        }
        artifact_path.write_text(
            json.dumps(artifact_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(artifact_path)

    def _log_failure(self, summary: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
        logger.warning(
            "LM Studio local inference failure: %s",
            json.dumps(
                {
                    **summary,
                    "attempts": attempts,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    def _failure_summary(
        self,
        *,
        attempts: list[dict[str, Any]],
        model_name: str,
        failure_type: str,
        question: str,
        intent: str,
    ) -> dict[str, Any]:
        attempt = attempts[-1] if attempts else {}
        validator = attempt.get("summary_validation", {})
        summary = {
            "model_name": model_name,
            "model_profile": self._model_profile_label(model_name),
            "endpoint": attempt.get("endpoint", ""),
            "http_status": attempt.get("http_status"),
            "finish_reason": attempt.get("finish_reason", ""),
            "parsed_text_length": int(attempt.get("parsed_text_length", 0) or 0),
            "failure_type": failure_type,
            "internal_status": failure_type,
            "fallback_reason": failure_type,
            "token_usage": attempt.get("token_usage", {}),
            "request_error": attempt.get("request_error", ""),
        }
        artifact_path = self._write_debug_artifact(
            attempts=attempts,
            summary=summary,
            question=question,
            intent=intent,
            validator=validator,
        )
        summary["artifact_path"] = artifact_path
        if validator:
            summary["validator"] = validator
        self._log_failure(summary, attempts)
        return summary

    def _request_json(
        self,
        *,
        url: str,
        request_payload: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[requests.Response | None, dict[str, Any], str]:
        try:
            response = requests.post(
                url,
                headers=self._headers(),
                json=request_payload,
                timeout=timeout_seconds,
            )
        except requests.RequestException as exc:
            return None, {}, str(exc)

        try:
            payload = response.json()
        except ValueError:
            payload = {"raw_text": response.text}
        return response, payload, ""

    @staticmethod
    def _response_error_message(attempt: dict[str, Any]) -> str:
        payload = attempt.get("response_payload") or {}
        if not isinstance(payload, dict):
            return ""
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error).strip()
        return str(error or payload.get("raw_text", "") or "").strip()

    def _is_context_size_error(self, attempt: dict[str, Any]) -> bool:
        message = self._response_error_message(attempt).lower()
        return "context size" in message or "context length" in message or "context window" in message

    @staticmethod
    def _looks_garbled_text(text: str) -> bool:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return False

        if re.search(r"[?？]{8,}", compact):
            return True

        broken_markers = compact.count("?") + compact.count("？") + compact.count("�")
        meaningful_chars = len(re.findall(r"[가-힣A-Za-z0-9]", compact))
        return broken_markers >= 24 and broken_markers > meaningful_chars

    def _should_retry_after_initial_failure(self, model_name: str, failure_reason: str) -> bool:
        if failure_reason == "garbled_text" and self._uses_responses_api(model_name):
            # A raw run of "????" means the local gpt-oss runtime produced corrupted
            # visible tokens, not that the prompt was merely too long. Retrying the
            # same model through another LM Studio endpoint repeats the failure and
            # wastes latency, so fall back to deterministic evidence immediately.
            return False
        return failure_reason in {
            "empty",
            "reasoning_only",
            "garbled_text",
            "context_size_exceeded",
            "low_information",
        }

    @staticmethod
    def _looks_low_information_text(text: str) -> bool:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return False

        exact_phrases = {
            "현재 자료 기준으로 확인되는 내용은 제한적입니다.",
            "현재 자료 기준으로 확인되는 내용은 제한적입니다",
            "현재 자료 기준으로 관련 내용을 충분히 찾지 못했습니다.",
            "현재 자료 기준으로 관련 내용을 충분히 찾지 못했습니다",
            "현재 자료 기준으로 개정 이유나 연혁을 충분히 찾지 못했습니다.",
            "현재 자료 기준으로 개정 이유나 연혁을 충분히 찾지 못했습니다",
        }
        if compact in exact_phrases:
            return True

        if len(compact) <= 40 and any(phrase in compact for phrase in ["제한적", "찾지 못", "충분히 찾지 못", "확인되는 내용은 제한적"]):
            return True
        return False

    @classmethod
    def _looks_incomplete_answer_text(cls, text: str) -> bool:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return False
        compact = cls._validation_prose(compact)
        if not compact:
            return False
        if len(compact) < 10:
            return True
        complete_sentence_end = re.search(
            r"(?:다|요|니다|습니다|됩니다|합니다|있습니다|없습니다|입니다)[.!?。]?$",
            compact,
        )
        if len(compact) < 24 and not complete_sentence_end:
            return True
        if re.search(r"(?:와|과|및|또는|이나|거나|은|는|이|가|을|를|의|로|으로|에서|에게|부터|까지)$", compact):
            return True
        if re.search(r"(?:침해와|보호와|기준과|절차와|제도와|이유는)$", compact):
            return True
        if complete_sentence_end:
            return False
        if compact.endswith((".", "!", "?", "。")):
            return False
        return True

    @staticmethod
    def _evidence_blob(evidence: list) -> str:
        parts: list[str] = []
        for hit in evidence:
            chunk = getattr(hit, "chunk", None)
            if chunk is None:
                continue
            parts.extend(
                [
                    str(getattr(chunk, "law_name", "") or ""),
                    str(getattr(chunk, "article_no", "") or ""),
                    str(getattr(chunk, "article_title", "") or ""),
                    str(getattr(chunk, "promulgation_date", "") or ""),
                    str(getattr(chunk, "effective_date", "") or ""),
                    str(getattr(chunk, "text", "") or ""),
                    str(getattr(chunk, "extra", {}) or {}),
                ]
            )
        return " ".join(parts)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return []
        pieces = re.split(r"(?<=[.!?。])\s+|(?<=다)\s+", compact)
        return [piece.strip(" -•\t") for piece in pieces if piece.strip(" -•\t")]

    @staticmethod
    def _support_terms(text: str) -> set[str]:
        stopwords = {
            "현재",
            "자료",
            "기준",
            "근거",
            "확인",
            "내용",
            "질문",
            "법령",
            "조문",
            "체계",
            "실무",
            "구분",
            "관련",
            "중심",
            "경우",
            "아래",
            "원문",
            "링크",
        }
        terms = {
            token
            for token in re.findall(r"[가-힣A-Za-z0-9]+", text or "")
            if len(token) >= 2 and token not in stopwords
        }
        return terms

    def _evidence_sentences(self, evidence: list) -> list[str]:
        sentences: list[str] = []
        for hit in evidence:
            chunk = getattr(hit, "chunk", None)
            if chunk is None:
                continue
            extra = getattr(chunk, "extra", {}) or {}
            parts = [
                str(getattr(chunk, "law_name", "") or ""),
                str(getattr(chunk, "article_no", "") or ""),
                str(getattr(chunk, "article_title", "") or ""),
                str(getattr(chunk, "promulgation_date", "") or ""),
                str(getattr(chunk, "effective_date", "") or ""),
                str(extra.get("display_text", "") or ""),
                str(extra.get("summary_text", "") or ""),
                str(getattr(chunk, "text", "") or ""),
            ]
            sentences.extend(self._split_sentences(" ".join(parts)))
        return list(dict.fromkeys(sentence for sentence in sentences if sentence))

    def _sentence_support_check(self, summary_text: str, evidence: list) -> tuple[bool, list[str]]:
        evidence_sentences = self._evidence_sentences(evidence)
        if not evidence_sentences:
            return True, []
        evidence_terms = [(sentence, self._support_terms(sentence)) for sentence in evidence_sentences]
        evidence_blob = self._evidence_blob(evidence)
        risky_markers = [
            "징계",
            "신고",
            "고충",
            "군인사법",
            "군인 징계령",
            "군인복무규율",
            "폐지",
            "개정",
            "보호",
            "절차",
            "종류",
        ]
        unsupported: list[str] = []
        for sentence in self._split_sentences(summary_text):
            if len(sentence) < 18:
                continue
            if not any(marker in sentence for marker in risky_markers) and not re.search(r"(?:19|20)\d{2}", sentence):
                continue
            sentence_terms = self._support_terms(sentence)
            if len(sentence_terms) < 2:
                continue
            sentence_years = set(re.findall(r"(?:19|20)\d{2}", sentence))
            sentence_articles = set(re.findall(r"제\d+조", sentence))
            best_overlap = 0
            for evidence_sentence, terms in evidence_terms:
                if sentence_years and not all(year in evidence_sentence for year in sentence_years):
                    continue
                if sentence_articles and not all(article in evidence_blob for article in sentence_articles):
                    continue
                best_overlap = max(best_overlap, len(sentence_terms & terms))
            threshold = 2 if len(sentence_terms) >= 4 else 1
            if best_overlap < threshold:
                unsupported.append(sentence[:120])
        return not unsupported, unsupported

    @staticmethod
    def _validation_prose(text: str) -> str:
        compact = (text or "").strip()
        if not compact:
            return ""
        if "### 핵심 결론" not in compact:
            return compact
        extracted = compact.split("### 핵심 결론", 1)[1].strip()
        next_heading_positions = [pos for pos in [extracted.find("\n## "), extracted.find("\n### ")] if pos >= 0]
        if next_heading_positions:
            extracted = extracted[: min(next_heading_positions)].strip()
        return " ".join(extracted.split()).strip()

    def _validate_summary_text(self, question: str, evidence: list, summary_text: str) -> dict[str, Any]:
        """Conservative checks before allowing LLM prose into the answer body."""
        compact = " ".join((summary_text or "").split()).strip()
        validation_text = self._validation_prose(summary_text)
        evidence_blob = self._evidence_blob(evidence)
        checks: dict[str, Any] = {
            "year_consistency": "pass",
            "statute_linkage": "pass",
            "unsupported_addition": "pass",
            "sentence_support": "pass",
            "passed": True,
            "reasons": [],
        }
        if not compact:
            checks["passed"] = False
            checks["reasons"].append("empty_summary")
            return checks

        if self._looks_incomplete_answer_text(validation_text):
            checks["completion"] = "fail"
            checks["passed"] = False
            checks["reasons"].append("incomplete_sentence")
        else:
            checks["completion"] = "pass"

        summary_years = set(re.findall(r"(?:19|20)\d{2}", validation_text))
        evidence_years = set(re.findall(r"(?:19|20)\d{2}", evidence_blob))
        unknown_years = sorted(year for year in summary_years if year not in evidence_years)
        if unknown_years:
            checks["year_consistency"] = "fail"
            checks["passed"] = False
            checks["reasons"].append(f"unsupported_years:{','.join(unknown_years)}")

        article_refs = set(re.findall(r"제\d+조", validation_text))
        missing_articles = sorted(ref for ref in article_refs if ref not in evidence_blob)
        if missing_articles:
            checks["statute_linkage"] = "fail"
            checks["passed"] = False
            checks["reasons"].append(f"unsupported_articles:{','.join(missing_articles)}")

        risky_terms = [
            "군인사법",
            "군인 징계령",
            "군인징계령",
            "징계위원회",
            "징계 절차",
            "징계의 종류",
            "감봉",
            "견책",
            "파면",
            "해임",
        ]
        unsupported_terms = sorted(term for term in risky_terms if term in validation_text and term not in evidence_blob)
        if unsupported_terms:
            checks["unsupported_addition"] = "fail"
            checks["passed"] = False
            checks["reasons"].append(f"unsupported_terms:{','.join(unsupported_terms)}")

        unsafe_transition_patterns = [
            r"2016년[^.。]*신고의무만",
            r"2020년[^.。]*(?:신고자\s*)?보호[^.。]*추가",
            r"2020년[^.。]*처음[^.。]*(?:신고자\s*)?보호",
        ]
        if any(re.search(pattern, validation_text) for pattern in unsafe_transition_patterns):
            checks["statute_linkage"] = "fail"
            checks["passed"] = False
            checks["reasons"].append("unsafe_transition_claim")

        sentence_supported, unsupported_sentences = self._sentence_support_check(validation_text, evidence)
        if not sentence_supported:
            checks["sentence_support"] = "fail"
            checks["passed"] = False
            checks["reasons"].append("unsupported_sentences")
            checks["unsupported_sentences"] = unsupported_sentences[:3]

        return checks

    def _validated_summary_answer(
        self,
        question: str,
        intent: str,
        evidence: list,
        summary_text: str,
    ) -> tuple[str, dict[str, Any]]:
        raw_text = (summary_text or "").strip()
        compact = " ".join(raw_text.split()).strip()
        validation = self._validate_summary_text(question, evidence, raw_text)
        
        if not validation.get("passed"):
            if (
                validation.get("year_consistency") == "pass"
                and validation.get("statute_linkage") == "pass"
                and validation.get("sentence_support") == "fail"
            ):
                safe_sentences = []
                unsupported_set = set(validation.get("unsupported_sentences", []))
                for sentence in self._split_sentences(self._validation_prose(raw_text) or compact):
                    if not any(sentence[:120] in us for us in unsupported_set):
                        safe_sentences.append(sentence)

                improved_text = " ".join(safe_sentences)
                if len(improved_text) >= 40:
                    improved_validation = self._validate_summary_text(question, evidence, improved_text)
                    if improved_validation.get("passed"):
                        improved_validation["salvaged"] = True
                        validation = improved_validation
                        compact = improved_text
                        
        if not validation.get("passed"):
            return "", validation
            
        return self._merge_summary_into_fallback(question, intent, evidence, self._validation_prose(raw_text) or compact), validation

    def _local_prompt_bundle(
        self,
        *,
        question: str,
        intent: str,
        evidence: list,
        profile: str,
        plain: bool,
    ) -> tuple[str, str]:
        if plain:
            return (
                LOCAL_PLAIN_SYSTEM_PROMPT,
                build_local_plain_user_prompt(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    profile=profile,
                ),
            )
        return (
            LOCAL_SYSTEM_PROMPT,
            build_local_user_prompt(
                question=question,
                intent=intent,
                evidence=evidence,
                profile=profile,
            ),
        )

    def _post_chat_completion(
        self,
        *,
        resolved_model: str,
        question: str,
        intent: str,
        evidence: list,
        retry: bool,
        plain: bool = False,
    ) -> dict[str, Any]:
        request_options = self._request_options(resolved_model, retry=retry)
        system_prompt, user_prompt = self._local_prompt_bundle(
            question=question,
            intent=intent,
            evidence=evidence,
            profile=request_options["prompt_profile"],
            plain=plain,
        )
        endpoint = f"{self.base_url}/chat/completions"
        request_payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": request_options["temperature"],
            "max_tokens": request_options["max_tokens"],
            "stream": False,
        }
        if "gpt-oss" in (resolved_model or "").lower():
            request_payload["reasoning_effort"] = "low"
            request_payload["response_format"] = {"type": "text"}
        response, payload, request_error = self._request_json(
            url=endpoint,
            request_payload=request_payload,
            timeout_seconds=self.timeout_seconds,
        )
        parsed_text = ""
        reasoning_only = False
        if not request_error and (response is not None and response.status_code < 400):
            parsed_text = self._extract_response_text(payload)
            reasoning_only = self._is_reasoning_only_chat_payload(payload)
        return self._attempt_record(
            model_name=resolved_model,
            endpoint=endpoint,
            request_payload=request_payload,
            retry=retry,
            response=response,
            response_payload=payload,
            parsed_text=parsed_text,
            reasoning_only=reasoning_only,
            request_error=request_error,
        )

    def _post_responses_completion(
        self,
        *,
        resolved_model: str,
        question: str,
        intent: str,
        evidence: list,
        retry: bool,
        plain: bool = False,
    ) -> dict[str, Any]:
        request_options = self._request_options(resolved_model, retry=retry)
        system_prompt, user_prompt = self._local_prompt_bundle(
            question=question,
            intent=intent,
            evidence=evidence,
            profile=request_options["prompt_profile"],
            plain=plain,
        )
        endpoint = f"{self.base_url}/responses"
        response_token_cap = 1600 if "gpt-oss" in (resolved_model or "").lower() and not retry else 900
        request_payload = {
            "model": resolved_model,
            "input": f"{system_prompt}\n\n{user_prompt}",
            "reasoning": {"effort": "low"},
            "temperature": request_options["temperature"],
            "max_output_tokens": max(120, min(request_options["max_tokens"], response_token_cap)),
            "text": {"format": {"type": "text"}},
            "store": False,
        }
        response, payload, request_error = self._request_json(
            url=endpoint,
            request_payload=request_payload,
            timeout_seconds=self.timeout_seconds,
        )
        parsed_text = ""
        reasoning_only = False
        if not request_error and (response is not None and response.status_code < 400):
            parsed_text = self._extract_responses_text(payload)
            reasoning_only = self._is_reasoning_only_responses_payload(payload)
        return self._attempt_record(
            model_name=resolved_model,
            endpoint=endpoint,
            request_payload=request_payload,
            retry=retry,
            response=response,
            response_payload=payload,
            parsed_text=parsed_text,
            reasoning_only=reasoning_only,
            request_error=request_error,
        )

    def _post_native_chat_completion(
        self,
        *,
        resolved_model: str,
        question: str,
        intent: str,
        evidence: list,
        retry: bool,
        plain: bool = False,
    ) -> dict[str, Any]:
        request_options = self._request_options(resolved_model, retry=retry)
        system_prompt, user_prompt = self._local_prompt_bundle(
            question=question,
            intent=intent,
            evidence=evidence,
            profile="small",
            plain=plain,
        )
        endpoint = f"{self.native_base_url}/chat"
        output_cap = 900 if "gpt-oss" in (resolved_model or "").lower() else 420
        request_payload = {
            "model": resolved_model,
            "system_prompt": system_prompt,
            "input": user_prompt,
            "temperature": request_options["temperature"],
            "max_output_tokens": max(120, min(request_options["max_tokens"], output_cap)),
            "store": False,
        }
        response, payload, request_error = self._request_json(
            url=endpoint,
            request_payload=request_payload,
            timeout_seconds=self.timeout_seconds,
        )
        parsed_text = ""
        reasoning_only = False
        if not request_error and (response is not None and response.status_code < 400):
            parsed_text = self._extract_native_chat_text(payload)
            reasoning_only = self._is_reasoning_only_native_payload(payload)
        return self._attempt_record(
            model_name=resolved_model,
            endpoint=endpoint,
            request_payload=request_payload,
            retry=retry,
            response=response,
            response_payload=payload,
            parsed_text=parsed_text,
            reasoning_only=reasoning_only,
            request_error=request_error,
        )

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

    @staticmethod
    def _wrap_plain_answer(question: str, intent: str, text: str) -> str:
        compact = " ".join((text or "").split()).strip()
        if not compact:
            return ""
        if intent == "explain_change":
            return (
                "## 답변 개요\n"
                f"### 핵심 결론\n{compact}\n\n"
                "## 세부 정리\n"
                "### 주요 개정 이유\n- 현재 자료 기준으로 세부 개정 이유는 아래 근거 카드와 함께 확인해 주세요.\n\n"
                "### 실제 제도 변화\n- 위 핵심 결론은 제공된 공개 법령 근거를 바탕으로 정리한 요약입니다.\n\n"
                "### 해석 시사점\n- 최종 판단 전에는 아래 원문 링크와 근거 카드를 함께 확인해 주세요.\n\n"
                "## 근거 안내\n"
                "### 확인 방법\n- 아래 근거 카드와 원문 링크를 확인해 주세요."
            )
        if intent == "practical":
            return (
                "## 답변 개요\n"
                f"### 핵심 결론\n{compact}\n\n"
                "## 실무 정리\n"
                "### 실무적으로 보면\n- 위 결론은 현재 자료 기준의 보조 요약입니다.\n\n"
                "### 주의사항\n- 실제 처리 전에는 최신 규정과 소속 부대 지침을 함께 확인해 주세요.\n\n"
                "## 근거 안내\n"
                "### 확인 방법\n- 아래 근거 카드와 원문 링크를 확인해 주세요."
            )
        return (
            "## 답변 개요\n"
            f"### 핵심 결론\n{compact}\n\n"
            "## 세부 정리\n"
            "### 주요 규정\n- 위 결론과 직접 연결되는 조문은 아래 근거 카드에서 확인할 수 있습니다.\n\n"
            "### 실무 참고\n- 최종 판단 전에는 아래 원문 링크를 함께 확인해 주세요.\n\n"
            "## 근거 안내\n"
            "### 확인 방법\n- 아래 근거 카드와 원문 링크를 확인해 주세요."
        )

    @staticmethod
    def _trim_partial_answer_text(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        lines = [line.rstrip() for line in stripped.splitlines()]
        while lines and (not lines[-1].strip() or lines[-1].strip().startswith("#")):
            lines.pop()
        compact = " ".join(" ".join(lines).split()).strip()
        if not compact:
            return ""
        boundary = max(compact.rfind(". "), compact.rfind("다. "), compact.rfind("? "), compact.rfind("! "))
        if boundary >= 40:
            compact = compact[: boundary + 1].strip()
        elif self._looks_incomplete_answer_text(compact):
            return ""
        return compact

    def _merge_summary_into_fallback(self, question: str, intent: str, evidence: list, summary_text: str) -> str:
        compact = " ".join((summary_text or "").split()).strip()
        if not compact:
            return ""

        parts = self._parse_summary_parts(compact)
        core_summary = parts.get("core") or compact
        interpretation = parts.get("interpretation", "")
        if self._looks_incomplete_answer_text(core_summary):
            return ""
        if interpretation and self._looks_incomplete_answer_text(interpretation):
            interpretation = ""

        fallback = self._fallback_answer(question, intent, evidence)
        updated = re.sub(
            r"(### 핵심 결론\n)(.*?)(\n\n## )",
            lambda match: f"{match.group(1)}{core_summary}{match.group(3)}",
            fallback,
            count=1,
            flags=re.S,
        )
        if interpretation:
            updated = self._prepend_interpretation(updated, interpretation)
        if updated != fallback:
            return updated
        return self._wrap_plain_answer(question, intent, core_summary)

    @staticmethod
    def _parse_summary_parts(text: str) -> dict[str, str]:
        compact = (text or "").strip()
        parts = {"core": "", "interpretation": ""}
        if not compact:
            return parts
        unlabeled: list[str] = []
        for raw_line in compact.splitlines():
            line = " ".join(raw_line.split()).strip(" -")
            if not line:
                continue
            core_match = re.match(r"^(?:핵심\s*결론|결론)\s*[:：]\s*(.+)$", line)
            if core_match:
                parts["core"] = core_match.group(1).strip()
                continue
            interpretation_match = re.match(r"^(?:실무\s*해석|해석|시사점|실무\s*참고)\s*[:：]\s*(.+)$", line)
            if interpretation_match:
                parts["interpretation"] = interpretation_match.group(1).strip()
                continue
            unlabeled.append(line)
        if not parts["core"]:
            parts["core"] = " ".join(unlabeled).strip()
        return parts

    @staticmethod
    def _prepend_interpretation(answer_markdown: str, interpretation: str) -> str:
        compact = " ".join((interpretation or "").split()).strip()
        if not compact:
            return answer_markdown
        target_headings = [
            "실무상 확인 포인트",
            "실무적으로 보면",
            "해석 시사점",
            "실무 참고",
            "주의사항",
        ]
        for heading in target_headings:
            pattern = rf"(### {re.escape(heading)}\n)(.*?)(\n\n(?:###|##) )"
            updated = re.sub(
                pattern,
                lambda match: f"{match.group(1)}- {compact}\n{match.group(2).lstrip()}",
                answer_markdown,
                count=1,
                flags=re.S,
            )
            if updated != answer_markdown:
                return updated
        return answer_markdown

    def _salvage_partial_answer(self, question: str, intent: str, evidence: list, text: str) -> str:
        compact = (text or "").strip()
        if not compact or self._looks_garbled_text(compact) or self._looks_low_information_text(compact):
            return ""

        extracted = compact
        if "### 핵심 결론" in compact:
            extracted = compact.split("### 핵심 결론", 1)[1].strip()
            next_heading_positions = [pos for pos in [extracted.find("\n## "), extracted.find("\n### ")] if pos >= 0]
            if next_heading_positions:
                extracted = extracted[: min(next_heading_positions)].strip()

        trimmed = self._trim_partial_answer_text(extracted)
        if len(trimmed) < 40:
            return ""
        return self._merge_summary_into_fallback(question, intent, evidence, trimmed)

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
                text=self._postprocess_answer_markdown(self._no_evidence_answer(intent)),
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
        except RuntimeError as exc:
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice=f"{exc} 근거 요약 모드로 전환했습니다.",
                quota_snapshot=self._effective_snapshot(),
            )

        attempts: list[dict[str, Any]] = []

        if self._uses_responses_api(resolved_model):
            first_attempt = self._post_responses_completion(
                resolved_model=resolved_model,
                question=question,
                intent=intent,
                evidence=evidence,
                retry=False,
                plain=True,
            )
        else:
            first_attempt = self._post_chat_completion(
                resolved_model=resolved_model,
                question=question,
                intent=intent,
                evidence=evidence,
                retry=False,
                plain=True,
            )
        attempts.append(first_attempt)

        if first_attempt.get("request_error"):
            diagnostics = self._failure_summary(
                attempts=attempts,
                model_name=resolved_model,
                failure_type="request_error",
                question=question,
                intent=intent,
            )
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice=f"LM Studio 요청에 실패해 근거 요약 모드로 전환했습니다. ({first_attempt['request_error']})",
                quota_snapshot=self._effective_snapshot(),
                diagnostics=diagnostics,
            )

        first_status = int(first_attempt.get("http_status") or 0)
        if first_status >= 400:
            can_retry_context_error = (
                self._uses_responses_api(resolved_model)
                and self._is_context_size_error(first_attempt)
            )
            if can_retry_context_error:
                first_attempt = {
                    **first_attempt,
                    "context_size_exceeded": True,
                }
                attempts[-1] = first_attempt
            else:
                response_payload = first_attempt.get("response_payload") or {}
                response_error = ""
                if isinstance(response_payload, dict):
                    response_error = str(response_payload.get("error", "") or "").strip()
                diagnostics = self._failure_summary(
                    attempts=attempts,
                    model_name=resolved_model,
                    failure_type="http_error",
                    question=question,
                    intent=intent,
                )
                notice_suffix = f" ({response_error})" if response_error else ""
                return self._fallback_result(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    notice=f"LM Studio가 {first_status} 응답을 반환해 근거 요약 모드로 전환했습니다.{notice_suffix}",
                    quota_snapshot=self._effective_snapshot(),
                    diagnostics=diagnostics,
                )

        final_attempt = first_attempt
        final_text = str(final_attempt.get("parsed_text", "") or "")
        first_garbled = self._looks_garbled_text(final_text)
        first_low_information = self._looks_low_information_text(final_text)
        structured = bool(final_text) and self._is_structured_answer(question, intent, final_text) and not first_garbled
        initial_failure_reason = (
            "context_size_exceeded"
            if final_attempt.get("context_size_exceeded")
            else (
                "reasoning_only"
                if final_attempt.get("reasoning_only") and not final_text
                else ("empty" if not final_text else ("garbled_text" if first_garbled else ("low_information" if first_low_information else "unstructured")))
            )
        )
        summary_validation: dict[str, Any] = {}
        if structured:
            summary_validation = self._validate_summary_text(question, evidence, final_text)
            if not summary_validation.get("passed"):
                structured = False
                final_attempt = {
                    **final_attempt,
                    "summary_validation": summary_validation,
                }
                initial_failure_reason = "validation_failed"

        if not structured and final_text and not first_garbled and not first_low_information:
            summary_answer, summary_validation = self._validated_summary_answer(question, intent, evidence, final_text)
            if summary_answer:
                final_text = summary_answer
                structured = True
                final_attempt = {
                    **final_attempt,
                    "parsed_text_length": len(final_text),
                    "summary_only": True,
                    "summary_validation": summary_validation,
                }
            else:
                final_attempt = {
                    **final_attempt,
                    "summary_validation": summary_validation,
                }
                initial_failure_reason = "validation_failed"

        if not structured:
            if not self._should_retry_after_initial_failure(resolved_model, initial_failure_reason):
                diagnostics = self._failure_summary(
                    attempts=attempts,
                    model_name=resolved_model,
                    failure_type=initial_failure_reason,
                    question=question,
                    intent=intent,
                )
                if summary_validation:
                    diagnostics["validator"] = summary_validation
                if initial_failure_reason == "garbled_text":
                    diagnostics["model_output_health"] = "garbled"
                return self._fallback_result(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    quota_snapshot=self._effective_snapshot(),
                    diagnostics=diagnostics,
                )

            if self._uses_responses_api(resolved_model) and (initial_failure_reason in {"empty", "reasoning_only", "garbled_text", "context_size_exceeded"}):
                if self._prefers_native_chat_retry(resolved_model):
                    retry_attempt = self._post_native_chat_completion(
                        resolved_model=resolved_model,
                        question=question,
                        intent=intent,
                        evidence=evidence,
                        retry=True,
                        plain=True,
                    )
                else:
                    retry_attempt = self._post_chat_completion(
                        resolved_model=resolved_model,
                        question=question,
                        intent=intent,
                        evidence=evidence,
                        retry=True,
                        plain=True,
                    )
            else:
                retry_attempt = self._post_chat_completion(
                    resolved_model=resolved_model,
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    retry=True,
                    plain=True,
                )

            attempts.append(retry_attempt)

            if retry_attempt.get("request_error"):
                diagnostics = self._failure_summary(
                    attempts=attempts,
                    model_name=resolved_model,
                    failure_type="request_error",
                    question=question,
                    intent=intent,
                )
                notice = (
                    f"LM Studio가 재시도 중 요청에 실패해 근거 요약 모드로 전환했습니다. ({retry_attempt['request_error']})"
                )
                return self._fallback_result(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    notice=notice,
                    quota_snapshot=self._effective_snapshot(),
                    diagnostics=diagnostics,
                )

            retry_status = int(retry_attempt.get("http_status") or 0)
            if retry_status >= 400:
                response_payload = retry_attempt.get("response_payload") or {}
                response_error = ""
                if isinstance(response_payload, dict):
                    response_error = str(response_payload.get("error", "") or "").strip()
                diagnostics = self._failure_summary(
                    attempts=attempts,
                    model_name=resolved_model,
                    failure_type="context_size_exceeded" if initial_failure_reason == "context_size_exceeded" else "http_error",
                    question=question,
                    intent=intent,
                )
                notice_suffix = f" ({response_error})" if response_error else ""
                return self._fallback_result(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    notice=f"LM Studio가 컨텍스트 한도를 초과해 짧은 프롬프트로 재시도했지만 실패했습니다. 근거 요약 모드로 전환했습니다.{notice_suffix}",
                    quota_snapshot=self._effective_snapshot(),
                    diagnostics=diagnostics,
                )

            retry_text = str(retry_attempt.get("parsed_text", "") or "")
            retry_garbled = self._looks_garbled_text(retry_text)
            retry_low_information = self._looks_low_information_text(retry_text)
            retry_structured = bool(retry_text) and self._is_structured_answer(question, intent, retry_text) and not retry_garbled
            if retry_structured:
                summary_validation = self._validate_summary_text(question, evidence, retry_text)
                if summary_validation.get("passed"):
                    final_attempt = {
                        **retry_attempt,
                        "summary_validation": summary_validation,
                    }
                    final_text = retry_text
                    structured = True
                else:
                    summary_answer, summary_validation = self._validated_summary_answer(
                        question, intent, evidence, retry_text
                    )
                    if summary_answer:
                        final_attempt = {
                            **retry_attempt,
                            "parsed_text_length": len(summary_answer),
                            "summary_only": True,
                            "summary_validation": summary_validation,
                        }
                        final_text = summary_answer
                        structured = True
                    else:
                        final_attempt = {
                            **retry_attempt,
                            "summary_validation": summary_validation,
                        }
                        final_reason = "validation_failed"
                        diagnostics = self._failure_summary(
                            attempts=attempts,
                            model_name=resolved_model,
                            failure_type=final_reason,
                            question=question,
                            intent=intent,
                        )
                        diagnostics["validator"] = summary_validation
                        return self._fallback_result(
                            question=question,
                            intent=intent,
                            evidence=evidence,
                            quota_snapshot=self._effective_snapshot(),
                            diagnostics=diagnostics,
                        )
            elif retry_text and not retry_garbled and not retry_low_information:
                summary_answer, summary_validation = self._validated_summary_answer(question, intent, evidence, retry_text)
                if summary_answer:
                    final_attempt = {
                        **retry_attempt,
                        "parsed_text_length": len(summary_answer),
                        "summary_only": True,
                        "summary_validation": summary_validation,
                    }
                    final_text = summary_answer
                    structured = True
                else:
                    final_attempt = {
                        **retry_attempt,
                        "summary_validation": summary_validation,
                    }
                    final_reason = "validation_failed"
                    diagnostics = self._failure_summary(
                        attempts=attempts,
                        model_name=resolved_model,
                        failure_type=final_reason,
                        question=question,
                        intent=intent,
                    )
                    diagnostics["validator"] = summary_validation
                    return self._fallback_result(
                        question=question,
                        intent=intent,
                        evidence=evidence,
                        quota_snapshot=self._effective_snapshot(),
                        diagnostics=diagnostics,
                    )
            else:
                final_attempt = retry_attempt
                final_reason = (
                    "reasoning_only"
                    if retry_attempt.get("reasoning_only") and not retry_text
                    else (
                        "empty"
                        if not retry_text
                        else ("garbled_text" if retry_garbled else ("low_information" if retry_low_information else "unstructured"))
                    )
                )
                diagnostics = self._failure_summary(
                    attempts=attempts,
                    model_name=resolved_model,
                    failure_type=final_reason,
                    question=question,
                    intent=intent,
                )
                return self._fallback_result(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    quota_snapshot=self._effective_snapshot(),
                    diagnostics=diagnostics,
                )

        payload = final_attempt.get("response_payload") or {}
        if str(final_attempt.get("endpoint", "")).endswith("/api/v1/chat"):
            stats = payload.get("stats", {})
            if not isinstance(stats, dict):
                stats = {}
            prompt_tokens = int(stats.get("input_tokens", 0) or 0)
            candidate_tokens = int(stats.get("total_output_tokens", 0) or 0)
            total_tokens = prompt_tokens + candidate_tokens
        else:
            usage = payload.get("usage", {})
            prompt_tokens = self._usage_value(usage, "prompt_tokens", "promptTokens")
            if self._uses_responses_api(resolved_model) and not prompt_tokens:
                prompt_tokens = self._usage_value(usage, "input_tokens", "inputTokens")
            candidate_tokens = self._usage_value(usage, "completion_tokens", "completionTokens")
            if self._uses_responses_api(resolved_model) and not candidate_tokens:
                candidate_tokens = self._usage_value(usage, "output_tokens", "outputTokens")
            total_tokens = self._usage_value(usage, "total_tokens", "totalTokens")
        updated_snapshot = self.usage_tracker.record_success(
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            total_tokens=total_tokens,
        )
        effective_snapshot = self._effective_snapshot()
        notice = QUOTA_BLOCK_MESSAGE if self.enforce_limits and not updated_snapshot["can_generate"] else ""

        return GeneratedAnswer(
            text=self._postprocess_answer_markdown(final_text),
            backend="lm_studio",
            notice=notice,
            quota_snapshot={
                **effective_snapshot,
                "base_url": self.base_url,
                "model_name": self._last_resolved_model or self.model_name,
            },
            diagnostics={
                "model_name": resolved_model,
                "model_profile": self._model_profile_label(resolved_model),
                "endpoint": final_attempt.get("endpoint", ""),
                "http_status": final_attempt.get("http_status"),
                "finish_reason": final_attempt.get("finish_reason", ""),
                "parsed_text_length": int(final_attempt.get("parsed_text_length", 0) or 0),
                "failure_type": "ok",
                "internal_status": "summary_ok" if final_attempt.get("summary_only") else "ok",
                "validator": final_attempt.get("summary_validation", {}),
                "token_usage": final_attempt.get("token_usage", {}),
            },
        )
