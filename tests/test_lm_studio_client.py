from __future__ import annotations

import requests

from army_reg_rag.config import AppConfig, DataConfig, Settings
from army_reg_rag.domain.models import DocumentChunk, SearchHit
from army_reg_rag.llm.lm_studio_client import LMStudioAnswerClient


class DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self):
        return self.payload


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
            article_no="제10조",
            article_title="휴가",
            revision_kind="일부개정",
            text="군인의 휴가 관련 기준과 제한 사유를 설명한다.",
            source_url="https://example.com/law-1",
        ),
        score=0.92,
    )


def fake_get_factory(*, visible_models: list[str], loaded_models: list[str]):
    def fake_get(url, headers=None, timeout=None):
        if url == "http://127.0.0.1:1234/v1/models":
            return DummyResponse({"data": [{"id": model_id} for model_id in visible_models]})
        if url == "http://127.0.0.1:1234/api/v1/models":
            return DummyResponse(
                {
                    "models": [
                        {
                            "type": "llm",
                            "key": model_id,
                            "loaded_instances": [{"id": f"{model_id}-instance"}],
                        }
                        for model_id in loaded_models
                    ]
                }
            )
        raise AssertionError(f"unexpected GET url: {url}")

    return fake_get


def test_describe_connection_uses_single_loaded_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "army_reg_rag.llm.lm_studio_client.requests.get",
        fake_get_factory(
            visible_models=["gpt-oss-20b", "hcx-seed-think-14b"],
            loaded_models=["gpt-oss-20b"],
        ),
    )

    client = LMStudioAnswerClient(make_settings(tmp_path), base_url="http://127.0.0.1:1234")
    state = client.describe_connection()

    assert state["available"] is True
    assert state["resolved_model"] == "gpt-oss-20b"
    assert state["loaded_models"] == ["gpt-oss-20b"]
    assert "single loaded LLM" in state["message"]


def test_describe_connection_marks_multiple_loaded_models_ambiguous(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "army_reg_rag.llm.lm_studio_client.requests.get",
        fake_get_factory(
            visible_models=["gpt-oss-20b", "hcx-seed-think-14b"],
            loaded_models=["gpt-oss-20b", "hcx-seed-think-14b"],
        ),
    )

    client = LMStudioAnswerClient(make_settings(tmp_path), base_url="http://127.0.0.1:1234")
    state = client.describe_connection()

    assert state["available"] is False
    assert state["resolved_model"] == ""
    assert "multiple loaded LLMs" in state["message"]


def test_generate_answer_auto_uses_single_loaded_model(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "army_reg_rag.llm.lm_studio_client.requests.get",
        fake_get_factory(
            visible_models=["gpt-oss-20b", "hcx-seed-think-14b"],
            loaded_models=["gpt-oss-20b"],
        ),
    )

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url == "http://127.0.0.1:1234/v1/chat/completions"
        assert json["model"] == "gpt-oss-20b"
        assert json["messages"][0]["role"] == "system"
        assert json["messages"][1]["role"] == "user"
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "## 답변 개요\n"
                                "### 핵심 결론\n자동 추적된 LM Studio 모델이 응답했습니다.\n\n"
                                "## 세부 정리\n"
                                "### 주요 규정\n- 근거 조문을 바탕으로 답했습니다.\n\n"
                                "### 실무 참고\n- 원문 링크를 함께 확인해 주세요.\n\n"
                                "## 근거 안내\n"
                                "### 확인 방법\n- 제공된 근거만 사용했습니다."
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
    )
    result = client.generate_answer(
        "휴가 관련 규정을 정리해줘.",
        "search",
        [make_hit()],
    )

    assert result.backend == "lm_studio"
    assert "자동 추적된 LM Studio 모델" in result.text
    assert result.quota_snapshot["model_name"] == "gpt-oss-20b"
    assert result.quota_snapshot["total_tokens"] == 60


def test_generate_answer_falls_back_when_lm_studio_is_down(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        raise requests.ConnectionError("server unavailable")

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="hcx-seed-think-14b",
    )
    result = client.generate_answer(
        "휴가 관련 규정을 정리해줘.",
        "search",
        [make_hit()],
    )

    assert result.backend == "retrieval_fallback"
    assert "LM Studio request failed" in result.notice


def test_generate_answer_ignores_soft_limit_when_limits_disabled(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "## 답변 개요\n"
                                "### 핵심 결론\n제한 없이 응답했습니다.\n\n"
                                "## 세부 정리\n"
                                "### 주요 규정\n- 현행 근거를 요약했습니다.\n\n"
                                "### 실무 참고\n- 근거 카드와 함께 읽어 주세요.\n\n"
                                "## 근거 안내\n"
                                "### 확인 방법\n- 제공된 근거만 사용했습니다."
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    settings = make_settings(tmp_path)
    settings.llm.daily_request_budget = 1
    settings.llm.budget_cutoff_ratio = 1.0

    client = LMStudioAnswerClient(
        settings,
        base_url="http://127.0.0.1:1234",
        model_name="gpt-oss-20b",
        enforce_limits=False,
    )

    first = client.generate_answer("첫 질문", "search", [make_hit()])
    second = client.generate_answer("후속 질문", "search", [make_hit()])

    assert first.backend == "lm_studio"
    assert second.backend == "lm_studio"
    assert second.quota_snapshot["can_generate"] is True
    assert second.quota_snapshot["remaining_requests"] == -1
