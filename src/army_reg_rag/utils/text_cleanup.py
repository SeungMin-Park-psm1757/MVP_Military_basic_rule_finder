from __future__ import annotations

import re


COMMON_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("청원휴 가", "청원휴가"),
    ("특별휴 가", "특별휴가"),
    ("정기휴 가", "정기휴가"),
    ("왕복소일수", "왕복소요일수"),
    ("왕복소 요일수", "왕복소요일수"),
    ("사항를", "사항을"),
    ("불이 익", "불이익"),
    ("권 익", "권익"),
    ("시 정", "시정"),
    ("그가신고자임", "그가 신고자임"),
)


def repair_common_text_artifacts(text: str) -> str:
    """Fix recurring PDF/LLM spacing artifacts without broad rewriting."""

    repaired = text or ""
    for source, target in COMMON_TEXT_REPLACEMENTS:
        repaired = repaired.replace(source, target)

    repaired = re.sub(r"(?<=[가-힣])할수(?=[가-힣\s.,)])", "할 수", repaired)
    repaired = re.sub(r"(?<=[가-힣])수있", "수 있", repaired)
    repaired = re.sub(r"(?<=[가-힣])수없", "수 없", repaired)
    repaired = re.sub(r"(?<=\s)수있", "수 있", repaired)
    repaired = re.sub(r"(?<=\s)수없", "수 없", repaired)
    return repaired
