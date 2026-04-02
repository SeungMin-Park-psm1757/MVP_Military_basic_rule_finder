from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from army_reg_rag.config import Settings
from army_reg_rag.domain.models import SearchHit
from army_reg_rag.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from army_reg_rag.llm.usage_tracker import GeminiUsageTracker

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None
    types = None

QUOTA_BLOCK_MESSAGE = "한도 소진(추가 답변이 제한)"

NOISY_TOKENS = [
    "데모용 요약:",
    "국가법령정보센터",
    "자바스크립트를 지원하지 않아 일부 기능을 사용할 수 없습니다.",
    "전체 제정·개정이유",
    "제정·개정문",
    "법령 > 본문 >",
    "본문목록열림",
    "부칙목록열림",
    "별표목록열림",
    "서식목록열림",
    "위로 아래로",
    "검색조문선택",
    "화면내검색",
    "입력 폼",
    "목록저장",
    "내용저장",
    "저장 닫기",
    "파일형식",
    "카카오톡",
    "페이스북",
    "트위터",
    "라인",
    "주소복사",
    "돋보기",
    "생활법령정보",
]


@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    backend: str
    notice: str = ""
    quota_snapshot: dict[str, Any] = field(default_factory=dict)


class GeminiAnswerClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._client = None
        self.usage_tracker = GeminiUsageTracker(settings)
        if self.api_key and genai is not None:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def _question_focus_terms(self, question: str) -> list[str]:
        preferred = [
            "육아시간",
            "자녀돌봄휴가",
            "임신검진 동행휴가",
            "모성보호시간",
            "출산휴가",
            "배우자 출산휴가",
            "난임치료",
            "휴가",
            "외출",
            "외박",
            "연가",
            "공가",
            "청원휴가",
            "특별휴가",
            "정기휴가",
            "승인범위",
            "제한",
        ]
        terms = [keyword for keyword in preferred if keyword in question]
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", question)
        stopwords = {
            "왜", "개정", "이유", "중심", "설명", "설명해줘", "찾아줘", "현행", "규정",
            "현재", "무슨", "내용", "조문", "실무", "참고", "적용", "주의",
            "군인의", "지위", "복무", "기본법", "시행령", "시행규칙",
        }
        for token in tokens:
            if len(token) < 2 or token in stopwords or token in terms:
                continue
            terms.append(token)
        return terms[:8]

    def _normalize_text(self, text: str) -> str:
        cleaned = text.replace("\u00a0", " ").strip()
        for token in NOISY_TOKENS:
            cleaned = cleaned.replace(token, " ")
        cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _focus_snippet(self, text: str, focus_terms: list[str], *, window: int = 260) -> str:
        if not focus_terms:
            return text
        for term in focus_terms:
            pos = text.find(term)
            if pos >= 0:
                start = max(0, pos - 70)
                end = min(len(text), pos + window)
                return text[start:end].strip(" ,.-")
        return text

    def _trim_sentence(self, text: str, *, limit: int = 150) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        truncated = compact[: limit - 1]
        last_space = truncated.rfind(" ")
        if last_space >= 0:
            truncated = truncated[:last_space]
        return truncated.rstrip(" ,") + "."

    def _extract_points(self, text: str, *, focus_terms: list[str], limit: int = 3) -> list[str]:
        raw_lines = []
        for line in text.replace("•", "\n- ").splitlines():
            candidate = line.strip()
            candidate = candidate.lstrip("-").strip()
            candidate = self._normalize_text(candidate)
            if len(candidate) < 8:
                continue
            if any(token in candidate for token in ["파일형식", "저장 닫기", "목록저장", "내용저장"]):
                continue
            raw_lines.append(candidate)

        points: list[str] = []
        for line in raw_lines:
            if line not in points:
                points.append(self._trim_sentence(line, limit=150))
            if len(points) >= limit:
                return points

        focused = self._focus_snippet(self._normalize_text(text), focus_terms)
        pieces = re.split(r"(?<=다\.)\s+|(?<=다)\s+(?=[가-힣A-Za-z0-9])|; ", focused)
        for piece in pieces:
            candidate = self._normalize_text(piece)
            if len(candidate) < 12:
                continue
            if candidate not in points:
                points.append(self._trim_sentence(candidate, limit=150))
            if len(points) >= limit:
                break
        return points[:limit]

    def _display_law_level(self, hit: SearchHit) -> str:
        if hit.chunk.law_level:
            return hit.chunk.law_level
        name = hit.chunk.law_name
        if "시행규칙" in name:
            return "시행규칙"
        if "시행령" in name:
            return "시행령"
        return "법률"

    def _article_ref(self, hit: SearchHit) -> str:
        article_no = (hit.chunk.article_no or "").strip()
        article_title = (hit.chunk.article_title or "").strip()
        if article_no and article_title:
            return f"{article_no}({article_title})"
        return article_no or article_title or "관련 조문"

    def _source_hits(self, evidence: list[SearchHit], source_type: str) -> list[SearchHit]:
        return [hit for hit in evidence if hit.chunk.source_type == source_type]

    def _line_for_hit(self, hit: SearchHit, *, focus_terms: list[str], max_points: int = 2) -> str:
        points = self._extract_points(hit.chunk.text, focus_terms=focus_terms, limit=max_points)
        body = " ".join(points) if points else "현재 확보 자료 기준으로 관련 내용을 요약하기 어렵습니다."
        return f"- {self._display_law_level(hit)} {self._article_ref(hit)}: {body}"

    def _reason_summary_points(self, reason_hits: list[SearchHit], focus_terms: list[str]) -> list[str]:
        combined = " ".join(self._normalize_text(hit.chunk.text) for hit in reason_hits)
        points: list[str] = []
        if "저출생" in combined:
            points.append("저출생 대응과 출산·돌봄 지원 확대가 주요 개정 배경으로 확인됩니다.")
        if "일ㆍ가정 양립" in combined or "일·가정 양립" in combined:
            points.append("군인의 일·가정 양립을 지원하는 방향이 개정 이유에 반복적으로 제시됩니다.")
        if "근무 여건" in combined or "복무 여건" in combined:
            points.append("군인의 근무·복무 여건을 현실화하려는 취지가 함께 확인됩니다.")
        if "복무 의욕" in combined or "고취" in combined:
            points.append("간부 복무 의욕과 인력 유지 측면의 제도 보완 의도도 드러납니다.")

        for hit in reason_hits:
            extracted = self._extract_points(hit.chunk.text, focus_terms=focus_terms, limit=2)
            for item in extracted:
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        return points[:3]

    def _change_points(self, compare_hits: list[SearchHit], law_hits: list[SearchHit], focus_terms: list[str]) -> list[str]:
        points: list[str] = []
        for hit in compare_hits:
            extracted = self._extract_points(hit.chunk.text, focus_terms=focus_terms, limit=3)
            for item in extracted:
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        for hit in law_hits:
            extracted = self._extract_points(hit.chunk.text, focus_terms=focus_terms, limit=1)
            for item in extracted:
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        return points[:3]

    def _conclusion_from_search(self, question: str, law_hits: list[SearchHit]) -> str:
        if not law_hits:
            return "현재 확보 자료 기준으로는 관련 현행 규정을 충분히 찾지 못했습니다."
        focus_terms = self._question_focus_terms(question)
        if "휴가" in focus_terms and any("제18조" in self._article_ref(hit) for hit in law_hits):
            if any(self._display_law_level(hit) == "시행령" for hit in law_hits):
                return "군인의 휴가 관련 현행 기준은 기본법 제18조와 시행령의 휴가 조문에서 확인됩니다."
        primary = law_hits[0]
        return f"현재 확보 자료 기준으로는 {primary.chunk.law_name} {self._article_ref(primary)}에서 관련 기준을 확인할 수 있습니다."

    def _conclusion_from_explain(self, question: str, reason_hits: list[SearchHit]) -> str:
        if not reason_hits:
            return "현재 확보 자료 기준으로는 개정 이유를 직접 설명할 만한 자료가 충분하지 않습니다."
        combined = " ".join(self._normalize_text(hit.chunk.text) for hit in reason_hits)
        focus_terms = self._question_focus_terms(question)
        if "저출생" in combined and ("육아시간" in focus_terms or "돌봄" in "".join(focus_terms)):
            return "현재 확보 자료 기준으로 보면, 육아시간 관련 개정은 저출생 대응과 일·가정 양립 지원 확대가 핵심입니다."
        if "일ㆍ가정 양립" in combined or "일·가정 양립" in combined:
            return "현재 확보 자료 기준으로 보면, 이번 개정은 일·가정 양립 지원을 강화하려는 취지가 중심입니다."
        return "현재 확보 자료 기준으로 보면, 관련 제도는 지원 범위 확대와 운영 기준 개선을 목적으로 개정되었습니다."

    def _conclusion_from_practical(self, law_hits: list[SearchHit]) -> str:
        if not law_hits:
            return "현재 확보 자료 기준으로는 실무 참고에 필요한 현행 조문이 충분하지 않습니다."
        if any("제18조" in self._article_ref(hit) for hit in law_hits):
            return "실무적으로는 기본법 제18조의 제한 사유와 시행령의 세부 휴가 기준을 함께 확인하는 방식이 적절합니다."
        primary = law_hits[0]
        return f"실무적으로는 {primary.chunk.law_name} {self._article_ref(primary)}를 먼저 확인하는 것이 적절합니다."

    def _build_search_answer(self, question: str, evidence: list[SearchHit]) -> str:
        law_hits = self._source_hits(evidence, "law_text")
        reason_hits = self._source_hits(evidence, "revision_reason")
        focus_terms = self._question_focus_terms(question)

        level_order = {"법률": 0, "시행령": 1, "시행규칙": 2}
        ordered_hits = sorted(
            law_hits,
            key=lambda hit: (
                level_order.get(self._display_law_level(hit), 9),
                0 if "제18조" in self._article_ref(hit) else 1,
                0 if "제9조~제16조" in self._article_ref(hit) else 1,
                0 if "제12조" in self._article_ref(hit) else 1,
            ),
        )
        main_lines = [self._line_for_hit(hit, focus_terms=focus_terms, max_points=2) for hit in ordered_hits[:3]]
        if not main_lines:
            main_lines = ["- 현재 확보 자료 기준으로는 관련 현행 조문을 충분히 찾지 못했습니다."]

        apply_points: list[str] = []
        if any("제16조" in self._article_ref(hit) or "5분의 1" in hit.chunk.text for hit in law_hits):
            apply_points.append("휴가 승인 범위는 부대 현재 병력의 5분의 1 이내를 원칙으로 하되, 부대 상황에 따라 조정될 수 있습니다.")
        if any("육아시간" in hit.chunk.text or "자녀돌봄휴가" in hit.chunk.text for hit in law_hits):
            apply_points.append("휴가 질문이라도 육아시간·자녀돌봄휴가 같이 가족돌봄 관련 세부 기준은 시행령 제12조에서 함께 확인됩니다.")
        if reason_hits and ("휴가" in focus_terms or "육아시간" in focus_terms or "돌봄" in "".join(focus_terms)):
            reason_point = self._reason_summary_points(reason_hits, focus_terms)
            if reason_point:
                apply_points.append(f"최근 개정 흐름으로는 {reason_point[0]}")
        if not apply_points:
            apply_points.append("현재 확보 자료 기준으로는 법률과 시행령에서 휴가의 종류, 일수, 제한 사유, 승인 범위를 직접 확인할 수 있습니다.")

        return (
            f"### 한 줄 결론\n{self._conclusion_from_search(question, law_hits)}\n\n"
            f"### 주요 규정\n" + "\n".join(main_lines) + "\n\n"
            f"### 적용상 참고\n" + "\n".join(f"- {point}" if not point.startswith("- ") else point for point in apply_points[:3]) + "\n\n"
            "### 근거\n- 아래 근거(하단 원문 링크 참고)를 확인하세요."
        )

    def _build_explain_answer(self, question: str, evidence: list[SearchHit]) -> str:
        reason_hits = self._source_hits(evidence, "revision_reason")
        compare_hits = self._source_hits(evidence, "old_new_comparison")
        law_hits = self._source_hits(evidence, "law_text")
        focus_terms = self._question_focus_terms(question)

        reason_points = self._reason_summary_points(reason_hits, focus_terms)
        if not reason_points:
            reason_points = ["현재 확보 자료 기준으로는 개정 이유를 직접 적은 자료가 충분하지 않습니다."]

        change_points = self._change_points(compare_hits, law_hits, focus_terms)
        if not change_points:
            change_points = ["현재 확보 자료 기준으로는 구체적인 변경 항목을 충분히 추려내기 어렵습니다."]

        interpretation_points: list[str] = []
        combined = " ".join(self._normalize_text(hit.chunk.text) for hit in reason_hits)
        if "저출생" in combined:
            interpretation_points.append("저출생 대응을 위해 출산·돌봄 관련 제도를 넓힌 개정으로 볼 수 있습니다.")
        if "일ㆍ가정 양립" in combined or "일·가정 양립" in combined:
            interpretation_points.append("군 복무와 가정 돌봄을 병행할 수 있도록 사용 대상을 넓히고 운영을 유연하게 만든 개정으로 읽힙니다.")
        if "근무 여건" in combined or "복무 여건" in combined:
            interpretation_points.append("숙련 간부의 복무 지속과 근무 여건 개선을 함께 고려한 조정으로 해석할 수 있습니다.")
        if not interpretation_points:
            interpretation_points.append("현재 확보 자료 기준으로는 지원 범위 확대와 운영 기준 보완이 핵심 방향으로 보입니다.")

        return (
            f"### 한 줄 결론\n{self._conclusion_from_explain(question, reason_hits)}\n\n"
            f"### 주요 개정 이유\n" + "\n".join(f"- {point}" for point in reason_points[:3]) + "\n\n"
            f"### 실제 제도 변화\n" + "\n".join(f"- {point}" for point in change_points[:3]) + "\n\n"
            f"### 해석 포인트\n" + "\n".join(f"- {point}" for point in interpretation_points[:3]) + "\n\n"
            "### 근거\n- 아래 근거(하단 원문 링크 참고)를 확인하세요."
        )

    def _build_practical_answer(self, question: str, evidence: list[SearchHit]) -> str:
        law_hits = self._source_hits(evidence, "law_text")
        focus_terms = self._question_focus_terms(question)

        practical_points: list[str] = []
        if law_hits:
            practical_points.append("먼저 법률 조문에서 보장 여부와 제한 사유를 확인하고, 이어서 시행령에서 휴가 종류, 일수, 시간 단위 사용 기준을 확인하는 순서가 적절합니다.")
        if any("제18조" in self._article_ref(hit) for hit in law_hits):
            practical_points.append("지휘관이 제한·보류할 수 있는 사유가 규정되어 있으므로, 작전상황·교육훈련·징계절차·환자 상태 해당 여부를 함께 점검해야 합니다.")
        if any("5분의 1" in hit.chunk.text or "승인 범위" in hit.chunk.text for hit in law_hits):
            practical_points.append("휴가 승인 범위는 부대 현재 병력의 5분의 1 이내가 원칙이므로, 부대 상황에 따른 조정 가능성을 함께 봐야 합니다.")
        if any("진단서" in hit.chunk.text or "검진" in hit.chunk.text or "신청" in hit.chunk.text for hit in law_hits):
            practical_points.append("청원휴가나 돌봄 관련 사안은 진단서, 검진 일정, 가족관계 등 신청 사유를 뒷받침하는 자료 확인이 중요합니다.")
        if not practical_points:
            practical_points.append("현재 확보 자료 기준으로는 관련 현행 조문과 시행일, 적용 대상을 먼저 대조하는 방식이 안전합니다.")

        caution_points = [
            "이 답변은 실무 참고용 요약입니다.",
            "현재 확보 자료 기준으로 정리한 것이므로 실제 승인 절차와 서식은 소속 부대 지침을 함께 확인하는 것이 안전합니다.",
        ]

        return (
            f"### 한 줄 결론\n{self._conclusion_from_practical(law_hits)}\n\n"
            f"### 실무적으로 보면\n" + "\n".join(f"- {point}" for point in practical_points[:4]) + "\n\n"
            f"### 주의사항\n" + "\n".join(f"- {point}" for point in caution_points) + "\n\n"
            "### 근거\n- 아래 근거(하단 원문 링크 참고)를 확인하세요."
        )

    def _build_hybrid_answer(self, question: str, evidence: list[SearchHit]) -> str:
        reason_hits = self._source_hits(evidence, "revision_reason")
        focus_terms = self._question_focus_terms(question)
        reason_points = self._reason_summary_points(reason_hits, focus_terms) or [
            "현재 확보 자료 기준으로는 개정 이유 자료가 충분하지 않습니다."
        ]
        practical_answer = self._build_practical_answer(question, evidence)
        practical_body = practical_answer.split("### 실무적으로 보면\n", 1)[-1].split("\n\n### 주의사항", 1)[0].strip()

        return (
            f"### 한 줄 결론\n{self._conclusion_from_explain(question, reason_hits)}\n\n"
            f"### 주요 개정 이유\n" + "\n".join(f"- {point}" for point in reason_points[:2]) + "\n\n"
            f"### 실무적으로 보면\n{practical_body}\n\n"
            "### 근거\n- 아래 근거(하단 원문 링크 참고)를 확인하세요."
        )

    def _fallback_answer(self, question: str, intent: str, evidence: list[SearchHit]) -> str:
        if intent == "search":
            return self._build_search_answer(question, evidence)
        if intent == "explain_change":
            return self._build_explain_answer(question, evidence)
        if intent == "practical":
            return self._build_practical_answer(question, evidence)
        return self._build_hybrid_answer(question, evidence)

    def _usage_value(self, usage: Any, snake_name: str, camel_name: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(snake_name, usage.get(camel_name, 0)) or 0)
        return int(getattr(usage, snake_name, getattr(usage, camel_name, 0)) or 0)

    def _fallback_result(
        self,
        *,
        question: str,
        intent: str,
        evidence: list[SearchHit],
        notice: str = "",
        quota_snapshot: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            text=self._fallback_answer(question, intent, evidence),
            backend="retrieval_fallback",
            notice=notice,
            quota_snapshot=quota_snapshot or self.usage_tracker.snapshot(),
        )

    def _quota_block_result(self, quota_snapshot: dict[str, Any] | None = None) -> GeneratedAnswer:
        return GeneratedAnswer(
            text=QUOTA_BLOCK_MESSAGE,
            backend="quota_blocked",
            notice="",
            quota_snapshot=quota_snapshot or self.usage_tracker.snapshot(),
        )

    def _no_evidence_answer(self, intent: str) -> str:
        if intent == "explain_change":
            return (
                "### 한 줄 결론\n현재 확보 자료 기준으로는 개정 이유를 충분히 찾지 못했습니다.\n\n"
                "### 주요 개정 이유\n- 관련 개정이유 또는 신구 비교 자료가 부족합니다.\n\n"
                "### 실제 제도 변화\n- 구체적인 변경 내용을 확인할 수 있는 근거가 충분하지 않습니다.\n\n"
                "### 해석 포인트\n- 질문을 더 구체화하거나 자료 유형 필터를 넓혀 다시 검색해 주세요.\n\n"
                "### 근거\n- 현재 표시할 근거가 없습니다."
            )
        if intent == "practical":
            return (
                "### 한 줄 결론\n현재 확보 자료 기준으로는 실무 참고에 필요한 근거가 부족합니다.\n\n"
                "### 실무적으로 보면\n- 우선 질문 범위를 좁혀 다시 검색하는 것이 좋습니다.\n\n"
                "### 주의사항\n- 관련 조문과 시행일을 확인할 수 있는 공개 자료가 더 필요합니다.\n\n"
                "### 근거\n- 현재 표시할 근거가 없습니다."
            )
        return (
            "### 한 줄 결론\n현재 확보 자료 기준으로는 관련 현행 규정을 충분히 찾지 못했습니다.\n\n"
            "### 주요 규정\n- 관련 조문이 충분히 검색되지 않았습니다.\n\n"
            "### 적용상 참고\n- 질문을 더 구체화하거나 자료 유형 필터를 넓혀 다시 검색해 주세요.\n\n"
            "### 근거\n- 현재 표시할 근거가 없습니다."
        )

    def generate_answer(
        self,
        question: str,
        intent: str,
        evidence: list[SearchHit],
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

        snapshot = self.usage_tracker.snapshot()
        if not snapshot["can_generate"]:
            return self._quota_block_result(snapshot)

        if self._client is None or types is None:
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice="Gemini API가 준비되지 않아 근거 검색 기반 요약 모드로 전환했습니다.",
                quota_snapshot=snapshot,
            )

        user_prompt = build_user_prompt(question=question, intent=intent, evidence=evidence)

        try:
            response = self._client.models.generate_content(
                model=self.settings.llm.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    temperature=self.settings.llm.temperature,
                    max_output_tokens=self.settings.llm.max_output_tokens,
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            text = getattr(response, "text", "") or ""
            usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
            prompt_tokens = self._usage_value(usage, "prompt_token_count", "promptTokenCount")
            candidate_tokens = self._usage_value(usage, "candidates_token_count", "candidatesTokenCount")
            total_tokens = self._usage_value(usage, "total_token_count", "totalTokenCount")
            updated_snapshot = self.usage_tracker.record_success(
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
            )

            final_text = text.strip()
            if not final_text:
                return self._fallback_result(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    notice="Gemini 응답 본문이 비어 있어 근거 검색 기반 요약으로 전환했습니다.",
                    quota_snapshot=updated_snapshot,
                )

            notice = QUOTA_BLOCK_MESSAGE if not updated_snapshot["can_generate"] else ""
            return GeneratedAnswer(
                text=final_text,
                backend="gemini",
                notice=notice,
                quota_snapshot=updated_snapshot,
            )
        except Exception as exc:
            message = str(exc)
            if getattr(exc, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in message or "Quota exceeded" in message:
                updated_snapshot = self.usage_tracker.block_for_today(QUOTA_BLOCK_MESSAGE)
                return self._quota_block_result(updated_snapshot)
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice="Gemini API 호출 오류로 인해 근거 검색 기반 요약 모드로 전환했습니다.",
                quota_snapshot=self.usage_tracker.snapshot(),
            )
