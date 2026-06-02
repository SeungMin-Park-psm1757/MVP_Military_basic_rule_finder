from __future__ import annotations

import json
from pathlib import Path
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


def make_hit(
    *,
    law_name: str = "군인의 지위 및 복무에 관한 기본법",
    article_no: str = "제10조",
    article_title: str = "휴가",
    text: str = "군인의 휴가 관련 기준과 제한 사유를 설명한다.",
) -> SearchHit:
    return SearchHit(
        chunk=DocumentChunk(
            id="law-1",
            law_name=law_name,
            law_level="법률",
            source_type="law_text",
            version_label="현행",
            promulgation_date="2025-01-01",
            effective_date="2026-01-01",
            article_no=article_no,
            article_title=article_title,
            revision_kind="일부개정",
            text=text,
            source_url="https://example.com/law-1",
        ),
        score=0.92,
    )


def make_structured_answer(summary: str, *, intent: str = "search") -> str:
    if intent == "explain_change":
        return (
            "## 답변 개요\n"
            f"### 핵심 결론\n{summary}\n\n"
            "## 세부 정리\n"
            "### 주요 개정 이유\n- 개정 배경을 근거 중심으로 정리했습니다.\n\n"
            "### 실제 제도 변화\n- 변경된 기준을 짧게 정리했습니다.\n\n"
            "### 해석 시사점\n- 현재 체계와 연결해 해석했습니다.\n\n"
            "## 근거 안내\n"
            "### 확인 방법\n- 제공된 근거만 사용했습니다."
        )
    return (
        "## 답변 개요\n"
        f"### 핵심 결론\n{summary}\n\n"
        "## 세부 정리\n"
        "### 주요 규정\n- 근거 조문을 바탕으로 답했습니다.\n\n"
        "### 실무 참고\n- 원문 링크를 함께 확인해 주세요.\n\n"
        "## 근거 안내\n"
        "### 확인 방법\n- 제공된 근거만 사용했습니다."
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
            visible_models=["exaone-4.0-1.2b", "hcx-seed-think-14b"],
            loaded_models=["exaone-4.0-1.2b"],
        ),
    )

    def fake_post(url, headers=None, json=None, timeout=None):
        assert url == "http://127.0.0.1:1234/v1/chat/completions"
        assert json["model"] == "exaone-4.0-1.2b"
        assert json["messages"][0]["role"] == "system"
        assert json["messages"][1]["role"] == "user"
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": make_structured_answer("자동 추적된 LM Studio 모델이 응답했습니다.")
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
    assert result.quota_snapshot["model_name"] == "exaone-4.0-1.2b"
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
    assert "LM Studio 요청에 실패" in result.notice


def test_generate_answer_ignores_soft_limit_when_limits_disabled(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": make_structured_answer("제한 없이 응답했습니다.")
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
        model_name="exaone-4.0-1.2b",
        enforce_limits=False,
    )

    first = client.generate_answer("첫 질문", "search", [make_hit()])
    second = client.generate_answer("후속 질문", "search", [make_hit()])

    assert first.backend == "lm_studio"
    assert second.backend == "lm_studio"
    assert second.quota_snapshot["can_generate"] is True
    assert second.quota_snapshot["remaining_requests"] == -1


def test_generate_answer_extracts_choice_text_payload(monkeypatch, tmp_path):
    request_payloads = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_payloads.append(json)
        return DummyResponse(
            {
                "choices": [{"text": make_structured_answer("choices[0].text 형식도 읽었습니다.")}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    settings = make_settings(tmp_path)
    client = LMStudioAnswerClient(
        settings,
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-7.8b",
    )
    result = client.generate_answer(
        "휴가 관련 규정을 정리해줘.",
        "search",
        [make_hit()],
    )

    assert result.backend == "lm_studio"
    assert "choices[0].text 형식도 읽었습니다." in result.text
    assert request_payloads[0]["max_tokens"] < settings.llm.max_output_tokens
    assert "공개 법령 근거만 짧게 정리하는 한국어 보조 모델" in request_payloads[0]["messages"][0]["content"]


def test_generate_answer_uses_responses_api_for_gpt_oss(monkeypatch, tmp_path):
    request_payloads = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_payloads.append((url, json))
        return DummyResponse(
            {
                "output": [
                    {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": "internal"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": make_structured_answer("responses API로 응답했습니다.")}],
                    },
                ],
                "usage": {"input_tokens": 30, "output_tokens": 20, "total_tokens": 50},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="gpt-oss-20b",
    )
    result = client.generate_answer(
        "휴가 관련 규정을 정리해줘.",
        "search",
        [make_hit()],
    )

    assert result.backend == "lm_studio"
    assert "responses API로 응답했습니다." in result.text
    assert request_payloads[0][0] == "http://127.0.0.1:1234/v1/responses"
    assert request_payloads[0][1]["reasoning"]["effort"] == "low"
    assert request_payloads[0][1]["max_output_tokens"] <= 520
    assert result.quota_snapshot["total_tokens"] == 50
    assert result.diagnostics["endpoint"] == "http://127.0.0.1:1234/v1/responses"
    assert result.diagnostics["failure_type"] == "ok"


def test_generate_answer_retries_with_smaller_prompt_for_exaone_1_2b(monkeypatch, tmp_path):
    request_payloads = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_payloads.append(json)
        if len(request_payloads) == 1:
            return DummyResponse({"choices": [{"message": {"content": ""}}]})
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": make_structured_answer(
                                "재시도 후 응답을 안정적으로 생성했습니다.",
                                intent="explain_change",
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 17, "completion_tokens": 12, "total_tokens": 29},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-1.2b",
    )
    result = client.generate_answer(
        "징계 기준이 어떻게 달라졌는지 설명해줘.",
        "explain_change",
        [make_hit()],
    )

    assert result.backend == "lm_studio"
    assert "재시도 후 응답을 안정적으로 생성했습니다." in result.text
    assert len(request_payloads) == 2
    assert request_payloads[1]["max_tokens"] < request_payloads[0]["max_tokens"]
    assert request_payloads[0]["temperature"] == 0.0


def test_generate_answer_retries_after_low_information_exaone_response(monkeypatch, tmp_path):
    request_payloads = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_payloads.append(json)
        if len(request_payloads) == 1:
            return DummyResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "현재 자료 기준으로 확인되는 내용은 제한적입니다.",
                                "reasoning_content": "",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 539, "completion_tokens": 14, "total_tokens": 553},
                }
            )
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "군인복무규율의 군기·복무 중심 규정은 기본법 제정 이후 기본권 침해 방지와 "
                                "신고자 보호를 법률 단계에서 확인하는 체계로 재편되었습니다."
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 620, "completion_tokens": 34, "total_tokens": 654},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-1.2b",
    )
    result = client.generate_answer(
        "군인복무규율이 폐지되고 기본법 체계가 도입되면서 어떻게 재편되었는지 설명해줘.",
        "hybrid",
        [
            make_hit(
                article_title="제정·개정이유",
                text="기본법 제정 이유는 군 내 기본권 침해 방지와 신고한 군인을 보호하려는 취지를 설명한다.",
            ),
            make_hit(
                law_name="군인복무규율",
                article_title="제정·개정이유",
                text="군인복무규율은 군인의 복무자세와 군기강을 일신하려는 취지를 설명한다.",
            ),
            make_hit(
                article_no="제45조",
                article_title="신고자 보호",
                text="제45조는 신고등을 이유로 한 징계조치 등 불이익조치를 금지한다.",
            ),
        ],
    )

    assert len(request_payloads) == 2
    assert result.backend == "lm_studio"
    assert "기본권 침해 방지" in result.text
    assert result.diagnostics["failure_type"] == "ok"


def test_generate_answer_retries_when_structured_output_is_garbled(monkeypatch, tmp_path):
    request_payloads = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_payloads.append(json)
        if len(request_payloads) == 1:
            return DummyResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": make_structured_answer(
                                    "????????????????????????????????????????????????????????",
                                    intent="explain_change",
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 18, "completion_tokens": 22, "total_tokens": 40},
                }
            )
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "현재 자료 기준으로 징계 기준은 기본법 체계에서 신고자 보호와 불이익 금지 방향이 더 분명해진 것으로 보입니다."
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 14, "total_tokens": 24},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-1.2b",
    )
    result = client.generate_answer(
        "징계 관련 기본법 변천사를 알려줘",
        "explain_change",
        [
            make_hit(
                article_no="제45조",
                article_title="신고자 보호",
                text="징계 기준은 신고자 보호와 불이익 금지 방향으로 더 분명해졌다.",
            )
        ],
    )

    assert result.backend == "lm_studio"
    assert "신고자 보호와 불이익 금지" in result.text
    assert "### 핵심 결론" in result.text
    assert "짧은 평문만 출력할 것" in request_payloads[0]["messages"][1]["content"]
    assert "마크다운 기호는 쓰지 말 것" in request_payloads[1]["messages"][1]["content"]


def test_generate_answer_falls_back_when_retry_text_is_low_information(monkeypatch, tmp_path):
    request_payloads = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_payloads.append((url, json))
        if len(request_payloads) == 1:
            return DummyResponse({"choices": [{"message": {"content": ""}}]})
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "현재 자료 기준으로 확인되는 내용은 제한적입니다."
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-1.2b",
    )
    result = client.generate_answer(
        "징계관련 내용을 정리해서 나열해줘.",
        "search",
        [make_hit()],
    )

    assert result.backend == "retrieval_fallback"
    assert result.diagnostics["failure_type"] == "low_information"
    assert result.diagnostics["model_profile"] == "small_local"
    assert result.notice == ""


def test_low_information_guard_applies_to_generic_local_model(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "현재 자료 기준으로 확인되는 내용은 제한적입니다.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="generic-13b-local",
    )
    result = client.generate_answer(
        "징계 관련 내용을 정리해서 나열해줘.",
        "search",
        [make_hit()],
    )

    assert result.backend == "retrieval_fallback"
    assert result.notice == ""
    assert result.diagnostics["failure_type"] == "low_information"
    assert result.diagnostics["model_profile"] == "default_local"


def test_generate_answer_reports_reasoning_only_failure(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "long hidden reasoning",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-deep-7.8b",
    )
    result = client.generate_answer(
        "징계 기준이 어떻게 달라졌는지 설명해줘.",
        "explain_change",
        [make_hit()],
    )

    assert result.backend == "retrieval_fallback"
    assert result.notice == ""
    assert result.diagnostics["failure_type"] == "reasoning_only"
    assert result.diagnostics["endpoint"] == "http://127.0.0.1:1234/v1/chat/completions"
    artifact_path = result.diagnostics["artifact_path"]
    assert artifact_path
    assert tmp_path.joinpath("runtime", "lm_studio_debug").exists()


def test_generate_answer_records_empty_response_diagnostics(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 0, "total_tokens": 9},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-1.2b",
    )
    result = client.generate_answer(
        "징계 관련 기본법 변천사를 알려줘.",
        "explain_change",
        [make_hit()],
    )

    assert result.backend == "retrieval_fallback"
    assert result.diagnostics["failure_type"] == "empty"
    assert result.diagnostics["parsed_text_length"] == 0
    assert result.diagnostics["http_status"] == 200
    artifact_path = Path(result.diagnostics["artifact_path"])
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert len(artifact["attempts"]) == 2
    assert artifact["summary"]["failure_type"] == "empty"


def test_generate_answer_gpt_oss_falls_back_to_native_chat_on_reasoning_only(monkeypatch, tmp_path):
    """gpt-oss가 responses API에서 reasoning_only 실패 시 /api/v1/chat으로 폴백."""
    request_log = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_log.append(url)
        if "/responses" in url:
            return DummyResponse(
                {
                    "output": [
                        {
                            "type": "reasoning",
                            "content": [{"type": "reasoning_text", "text": "internal only"}],
                        },
                    ],
                    "usage": {"input_tokens": 30, "output_tokens": 0, "total_tokens": 30},
                }
            )
        return DummyResponse(
            {
                "output": [
                    {"type": "reasoning", "content": "internal"},
                    {"type": "message", "content": "native chat 폴백으로 응답했습니다."},
                ],
                "stats": {"input_tokens": 40, "total_output_tokens": 20, "reasoning_output_tokens": 10},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="gpt-oss-20b",
    )
    result = client.generate_answer(
        "징계 관련 기본법 변천사를 알려줘",
        "search",
        [make_hit()],
    )

    assert result.backend == "lm_studio"
    assert "native chat 폴백으로 응답했습니다." in result.text
    assert request_log[0].endswith("/responses")
    assert request_log[1].endswith("/api/v1/chat")
    assert result.diagnostics["endpoint"].endswith("/api/v1/chat")


def test_generate_answer_gpt_oss_stops_after_garbled_responses_output(monkeypatch, tmp_path):
    """gpt-oss가 visible text를 ?????로 만들면 같은 모델 재시도를 중단한다."""
    request_log = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_log.append(url)
        return DummyResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "????????????????????????????????????????????????????????????????",
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 30, "output_tokens": 64, "total_tokens": 94},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="gpt-oss-20b",
    )
    result = client.generate_answer(
        "현행 기본법상 신고자 보호와 비밀보장의 관계를 설명해줘.",
        "search",
        [
            make_hit(
                article_no="제44조",
                article_title="신고자에 대한 비밀보장",
                text="제44조는 신고자의 인적사항이나 신고자임을 알 수 있는 사실을 공개하지 못하게 한다.",
            ),
            make_hit(
                article_no="제45조",
                article_title="신고자 보호",
                text="제45조는 신고등을 이유로 신고자에게 징계조치 등 불이익조치를 하지 못하게 한다.",
            ),
        ],
    )

    assert result.backend == "retrieval_fallback"
    assert len(request_log) == 1
    assert request_log[0].endswith("/responses")
    assert result.diagnostics["failure_type"] == "garbled_text"
    assert result.diagnostics["model_output_health"] == "garbled"
    assert "????" not in result.text
    assert "제44조" in result.text or "제45조" in result.text


def test_generate_answer_gpt_oss_retries_native_chat_on_context_size_error(monkeypatch, tmp_path):
    request_log = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_log.append((url, json))
        if "/responses" in url:
            return DummyResponse(
                {
                    "error": {
                        "message": "Context size has been exceeded.",
                        "type": "internal_error",
                    }
                },
                status_code=500,
            )
        return DummyResponse(
            {
                "output": [
                    {"type": "message", "content": "휴가 종류는 시행령 제9조에서 연가, 공가, 청원휴가, 특별휴가 및 정기휴가로 구분됩니다."},
                ],
                "stats": {"input_tokens": 35, "total_output_tokens": 18, "reasoning_output_tokens": 0},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="gpt-oss-20b",
    )
    result = client.generate_answer(
        "현행 시행령 기준으로 군인의 휴가 종류를 조문번호와 함께 나열해줘.",
        "practical",
        [
            make_hit(
                law_name="군인의 지위 및 복무에 관한 기본법 시행령",
                article_no="제9조",
                article_title="휴가의 종류 등",
                text="제9조는 휴가의 종류를 연가, 공가, 청원휴가, 특별휴가 및 정기휴가로 구분한다고 정한다.",
            )
        ],
    )

    assert result.backend == "lm_studio"
    assert request_log[0][0].endswith("/responses")
    assert request_log[1][0].endswith("/api/v1/chat")
    assert request_log[1][1]["input"]
    assert len(request_log[1][1]["input"]) < len(request_log[0][1]["input"])
    assert "연가, 공가, 청원휴가, 특별휴가 및 정기휴가" in result.text
    assert result.diagnostics["endpoint"].endswith("/api/v1/chat")
    assert result.diagnostics["failure_type"] == "ok"


def test_generate_answer_salvages_partial_structured_first_response(monkeypatch, tmp_path):
    request_log = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_log.append(url)
        return DummyResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "## 답변 개요\n"
                                    "### 핵심 결론\n"
                                    "군인의 징계 기준은 신고자 보호와 불이익 금지 방향으로 더 분명해졌습니다.\n\n"
                                    "## 연혁 정리\n"
                                    "### "
                                ),
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 20, "output_tokens": 18, "total_tokens": 38},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="gpt-oss-20b",
    )
    result = client.generate_answer(
        "징계 관련 기본법 변천사를 알려줘",
        "explain_change",
        [
            make_hit(
                article_no="제45조",
                article_title="신고자 보호",
                text="군인의 징계 기준은 신고자 보호와 불이익 금지 방향으로 더 분명해졌다.",
            )
        ],
    )

    assert result.backend == "lm_studio"
    assert "신고자 보호와 불이익 금지 방향" in result.text
    assert "### 시기별 변화" in result.text
    assert result.diagnostics["failure_type"] == "ok"
    assert result.diagnostics["internal_status"] == "summary_ok"
    assert len(request_log) == 1


def test_generate_answer_salvages_partial_structured_retry_response(monkeypatch, tmp_path):
    request_log = []

    def fake_post(url, headers=None, json=None, timeout=None):
        request_log.append(url)
        if len(request_log) == 1:
            return DummyResponse(
                {
                    "output": [
                        {
                            "type": "reasoning",
                            "status": "completed",
                            "content": [{"type": "reasoning_text", "text": "Thinking only."}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 20,
                        "total_tokens": 40,
                        "output_tokens_details": {"reasoning_tokens": 20},
                    },
                }
            )
        return DummyResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": (
                            "## 답변 개요\n"
                            "### 핵심 결론\n"
                            "제45조는 신고등을 이유로 한 불이익조치를 금지합니다. "
                            "군인복무규율은 폐지 후 독립 심판 절차를 포함합니다.\n\n"
                            "## 세부 정리\n"
                            "### 주요 규정\n- 근거 중심으로 정리했습니다."
                        ),
                    }
                ],
                "stats": {"input_tokens": 30, "total_output_tokens": 40, "reasoning_output_tokens": 0},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="gpt-oss-20b",
    )
    result = client.generate_answer(
        "현행 기본법 제45조 신고자 보호를 설명해줘.",
        "search",
        [
            make_hit(
                article_no="제45조",
                article_title="신고자 보호",
                text="제45조는 신고등을 이유로 신고자에게 징계조치 등 불이익조치를 하지 못하도록 한다.",
            )
        ],
    )

    assert result.backend == "retrieval_fallback"
    assert result.diagnostics["failure_type"] == "validation_failed"
    assert "독립 심판 절차" not in result.text
    assert len(request_log) == 2


def test_generate_answer_rejects_unsafe_year_transition_claim(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "2016년 제정 당시에는 신고의무만 있었고 2020년에 신고자 보호가 추가되었습니다."
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 14, "total_tokens": 26},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-1.2b",
    )
    result = client.generate_answer(
        "현행 기본법 제43조~제45조를 중심으로 신고의무와 신고자 보호의 연결 구조를 설명해줘.",
        "explain_change",
        [
            make_hit(
                article_no="제43조",
                article_title="신고의무 등",
                text="2016년 제정 이유는 군인의 신고의무와 신고자를 보호하려는 취지를 함께 설명한다.",
            ),
            make_hit(
                article_no="제45조",
                article_title="신고자 보호",
                text="신고자에게 징계조치 등 불이익조치를 하지 못하도록 하고 비밀을 보장한다.",
            ),
        ],
    )

    assert result.backend == "retrieval_fallback"
    assert result.diagnostics["failure_type"] == "validation_failed"
    assert result.diagnostics["validator"]["statute_linkage"] == "fail"
    assert "2020년에 신고자 보호가 추가" not in result.text


def test_generate_answer_uses_labeled_summary_for_core_and_practical_sections(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "핵심 결론: 군인은 시행령 제9조에 따라 연가, 공가, 청원휴가, 특별휴가 및 정기휴가로 구분되는 휴가 체계를 확인해야 합니다.\n"
                                "실무 해석: 실무자는 제9조의 종류를 먼저 확인하고 제14조의 병 정기휴가 기준을 이어서 대조하는 방식이 안전합니다."
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 24, "total_tokens": 44},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)

    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-1.2b",
    )
    result = client.generate_answer(
        "현행 시행령 기준으로 군인의 휴가 종류를 조문번호와 함께 나열해줘.",
        "practical",
        [
            make_hit(
                law_name="군인의 지위 및 복무에 관한 기본법 시행령",
                article_no="제9조",
                article_title="휴가의 종류 등",
                text="제9조는 휴가의 종류를 연가, 공가, 청원휴가, 특별휴가 및 정기휴가로 구분한다.",
            ),
            make_hit(
                law_name="군인의 지위 및 복무에 관한 기본법 시행령",
                article_no="제14조",
                article_title="병의 정기휴가 등",
                text="제14조는 병의 정기휴가 기준을 각 군별 복무기간에 따라 차등하여 실시한다고 정한다.",
            ),
        ],
    )

    assert result.backend == "lm_studio"
    assert "시행령 제9조에 따라 연가, 공가, 청원휴가, 특별휴가 및 정기휴가" in result.text
    assert "실무자는 제9조의 종류를 먼저 확인" in result.text
    assert result.diagnostics["validator"]["sentence_support"] == "pass"
