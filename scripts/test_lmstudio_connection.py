from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SRC_PATH_STR = str(SRC_PATH)
if SRC_PATH_STR not in sys.path:
    sys.path.insert(0, SRC_PATH_STR)

from army_reg_rag.config import load_settings
from army_reg_rag.llm.lm_studio_client import DEFAULT_LM_STUDIO_BASE_URL, LMStudioAnswerClient


def normalize_base_url(base_url: str) -> str:
    normalized = (base_url or DEFAULT_LM_STUDIO_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def print_block(title: str, value: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(value, str):
        print(value)
        return
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run_request(url: str, payload: dict[str, Any], timeout_seconds: float) -> tuple[int | None, dict[str, Any], str]:
    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return None, {}, repr(exc)

    try:
        data = response.json()
    except ValueError:
        data = {"raw_text": response.text}
    return response.status_code, data, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the repository's LM Studio endpoints directly.")
    parser.add_argument("--base-url", default=DEFAULT_LM_STUDIO_BASE_URL, help="LM Studio OpenAI-compatible base URL.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--gpt-oss-model", default="gpt-oss-20b", help="Model id for the Responses API probe.")
    parser.add_argument("--exaone-model", default="exaone-4.0-1.2b", help="Model id for the chat/completions probe.")
    args = parser.parse_args()

    settings = load_settings()
    client = LMStudioAnswerClient(settings, base_url=args.base_url, enforce_limits=False)
    base_url = normalize_base_url(args.base_url)

    probes = [
        {
            "label": "gpt-oss-20b via /v1/responses",
            "url": f"{base_url}/responses",
            "payload": {
                "model": args.gpt_oss_model,
                "input": (
                    "질문: 징계 관련 기본법 변천사를 간단히 설명해줘.\n"
                    "답변은 한두 문장의 짧은 한국어 완결문으로만 작성해."
                ),
                "reasoning": {"effort": "low"},
                "temperature": 0.0,
                "max_output_tokens": 320,
                "text": {"format": {"type": "text"}},
                "store": False,
            },
            "parser": client._extract_responses_text,
        },
        {
            "label": "exaone-4.0-1.2b via /v1/chat/completions",
            "url": f"{base_url}/chat/completions",
            "payload": {
                "model": args.exaone_model,
                "messages": [
                    {"role": "system", "content": "짧은 한국어 한두 문장으로만 답해."},
                    {"role": "user", "content": "징계 관련 기본법 변천사를 한두 문장으로 설명해줘."},
                ],
                "temperature": 0.0,
                "max_tokens": 320,
                "stream": False,
            },
            "parser": client._extract_response_text,
        },
    ]

    results: list[dict[str, Any]] = []
    for probe in probes:
        print(f"\n##### {probe['label']} #####")
        print_block("Request Payload", probe["payload"])
        status_code, raw_json, error = run_request(probe["url"], probe["payload"], args.timeout)
        if error:
            print(f"Status Code: ERROR ({error})")
            results.append(
                {
                    "label": probe["label"],
                    "url": probe["url"],
                    "request_payload": probe["payload"],
                    "status_code": None,
                    "error": error,
                    "raw_json": {},
                    "parsed_final_text": "",
                }
            )
            continue

        parsed_final_text = probe["parser"](raw_json)
        print(f"Status Code: {status_code}")
        print_block("Raw JSON", raw_json)
        print_block("Parsed Final Text", parsed_final_text or "<empty>")
        results.append(
            {
                "label": probe["label"],
                "url": probe["url"],
                "request_payload": probe["payload"],
                "status_code": status_code,
                "error": "",
                "raw_json": raw_json,
                "parsed_final_text": parsed_final_text,
            }
        )

    output_path = Path("data/runtime/test_lmstudio_connection_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved probe results to {output_path}")


if __name__ == "__main__":
    main()
