from __future__ import annotations

import requests

from army_reg_rag.config import AppConfig, DataConfig, Settings
from army_reg_rag.domain.models import DocumentChunk, SearchHit
from army_reg_rag.llm.gemini_client import GeminiAnswerClient
from army_reg_rag.llm.lm_studio_client import LMStudioAnswerClient


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> dict:
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
    article_no: str,
    article_title: str,
    text: str,
    source_type: str = "law_text",
    law_name: str = "군인의 지위 및 복무에 관한 기본법",
) -> SearchHit:
    return SearchHit(
        chunk=DocumentChunk(
            id=f"{source_type}-{article_no}-{article_title}",
            law_name=law_name,
            law_level="법률",
            source_type=source_type,
            version_label="현행",
            promulgation_date="2025-01-07",
            effective_date="2026-01-08",
            article_no=article_no,
            article_title=article_title,
            revision_kind="일부개정",
            text=text,
            source_url="https://example.com/law",
        ),
        score=0.95,
    )


def test_search_conclusion_directly_answers_reporter_protection_linkage(tmp_path):
    client = GeminiAnswerClient(make_settings(tmp_path))
    evidence = [
        make_hit(
            article_no="제44조",
            article_title="신고자에 대한 비밀보장",
            text="신고자라는 사실이나 인적사항 등을 다른 사람에게 알려주거나 공개 또는 보도해서는 아니 된다.",
        ),
        make_hit(
            article_no="제45조",
            article_title="신고자 보호",
            text="신고등을 이유로 신고자에게 징계조치 등 어떠한 신분상 불이익이나 근무조건상의 차별대우를 하여서는 아니 된다.",
        ),
    ]

    answer = client.generate_answer(
        "현행 기본법상 신고자 보호 규정을 조문번호와 함께 제시하고, 제44조의 비밀보장과 어떤 관계인지 설명해줘.",
        "search",
        evidence,
        allow_generation=False,
    )

    assert "제44조의 비밀보장" in answer.text
    assert "제45조의 신고자 보호" in answer.text
    assert "전제 장치" in answer.text
    assert "후속 보호 장치" in answer.text
    assert "징계 자체의 독립 절차" not in answer.text


def test_incomplete_lm_summary_is_rejected(tmp_path):
    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="gpt-oss-20b",
    )
    evidence = [
        make_hit(
            article_no="",
            article_title="제정·개정이유",
            source_type="revision_reason",
            text="군 내에서 계속되는 기본권 침해를 줄이고 군인의 기본권 의식을 높이려는 취지이다.",
        )
    ]

    validation = client._validate_summary_text(
        "군인복무규율 폐지 이후 기본법 체계 전환을 설명해줘.",
        evidence,
        "군인복무규율이 폐지되고 기본법이 제정된 이유는 군 내에서 계속되는 기본권 침해와",
    )

    assert validation["passed"] is False
    assert "incomplete_sentence" in validation["reasons"]


def test_low_information_fallback_notice_is_hidden(monkeypatch, tmp_path):
    def fake_post(url, headers=None, json=None, timeout=None):
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "현재 자료 기준으로 확인되는 내용은 제한적입니다.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        )

    monkeypatch.setattr("army_reg_rag.llm.lm_studio_client.requests.post", fake_post)
    client = LMStudioAnswerClient(
        make_settings(tmp_path),
        base_url="http://127.0.0.1:1234",
        model_name="exaone-4.0-1.2b",
        enforce_limits=False,
    )

    answer = client.generate_answer(
        "군인복무규율 폐지 이후 기본법 체계 전환을 설명해줘.",
        "hybrid",
        [
            make_hit(
                article_no="",
                article_title="제정·개정이유",
                source_type="revision_reason",
                text="군 내 기본권 침해를 줄이고 신고한 군인을 보호하려는 취지이다.",
            )
        ],
    )

    assert answer.backend == "retrieval_fallback"
    assert answer.notice == ""
    assert "근거 중심으로 정리해 제공합니다." not in answer.text
