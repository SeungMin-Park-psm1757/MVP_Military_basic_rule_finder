from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_function(module_name: str, path: Path, function_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


root_format_public_notice = _load_function(
    "root_streamlit_app_for_test",
    PROJECT_ROOT / "streamlit_app.py",
    "format_public_notice",
)
local_format_public_notice = _load_function(
    "local_streamlit_app_for_test",
    PROJECT_ROOT / "local" / "streamlit_app.py",
    "format_public_notice",
)


def test_public_notice_hides_internal_lm_studio_status_names():
    answer = {
        "answer_backend": "retrieval_fallback",
        "answer_notice": "LM Studio가 reasoning_only 상태로 실패했습니다.",
        "diagnostics": {
            "failure_type": "reasoning_only",
            "internal_status": "reasoning_only",
        },
    }

    assert root_format_public_notice(answer) == ""
    assert local_format_public_notice(answer) == ""
    assert "reasoning_only" not in root_format_public_notice(answer)
    assert "reasoning_only" not in local_format_public_notice(answer)


def test_public_notice_keeps_quota_message_visible():
    answer = {
        "answer_backend": "quota_blocked",
        "answer_notice": "오늘 사용량을 초과했습니다.",
        "diagnostics": {},
    }

    assert root_format_public_notice(answer) == "오늘 사용량을 초과했습니다."
    assert local_format_public_notice(answer) == "오늘 사용량을 초과했습니다."
