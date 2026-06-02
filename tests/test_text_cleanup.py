from __future__ import annotations

from army_reg_rag.llm.lm_studio_client import LMStudioAnswerClient
from army_reg_rag.utils.text_cleanup import repair_common_text_artifacts


def test_repair_common_text_artifacts_fixes_known_pdf_and_llm_artifacts():
    text = "청원휴 가는 왕복소일수와 관련된 사항를 확인하고, 그가신고자임을 알 수있다."

    repaired = repair_common_text_artifacts(text)

    assert "청원휴가" in repaired
    assert "왕복소요일수" in repaired
    assert "사항을" in repaired
    assert "그가 신고자임" in repaired
    assert "수 있다" in repaired


def test_lm_studio_incomplete_guard_rejects_short_core_fragments():
    assert LMStudioAnswerClient._looks_incomplete_answer_text("가혹행") is True
    assert LMStudioAnswerClient._looks_incomplete_answer_text("제45조 신고자 보호") is True
    assert LMStudioAnswerClient._looks_incomplete_answer_text("제45조는 신고자 보호를 규정합니다.") is False
