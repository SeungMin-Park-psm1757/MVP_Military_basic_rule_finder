from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from army_reg_rag.config import Settings
from army_reg_rag.domain.models import SearchHit
from army_reg_rag.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from army_reg_rag.llm.usage_tracker import GeminiUsageTracker
from army_reg_rag.utils.runtime_config import get_runtime_value

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover
    genai = None
    types = None

QUOTA_BLOCK_MESSAGE = "시도 제한(추가 응답 생성 한도)"
PROVIDER_RATE_LIMIT_NOTICE = "Gemini API 제한으로 생성 응답이 어려워 근거 기반 요약 모드로 전환했습니다."

NOISY_TOKENS = [
    "법령정보센터",
    "본문목록열기",
    "별표목록열기",
    "서식목록열기",
    "목록열기",
    "내용열기",
    "화면닫기",
    "파일형식",
    "페이지넘버",
    "주소복사",
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
        self.api_key = str(get_runtime_value("GEMINI_API_KEY", "")).strip()
        self._client = None
        self.usage_tracker = GeminiUsageTracker(settings)
        if self.api_key and genai is not None:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def _question_focus_terms(self, question: str) -> list[str]:
        preferred = [
            "휴가",
            "돌봄휴가",
            "육아시간",
            "배우자 출산휴가",
            "청원휴가",
            "공가",
            "외박",
            "외출",
            "휴직",
            "복무규율",
            "군인복무규율",
        ]
        terms = [keyword for keyword in preferred if keyword in question]
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", question)
        stopwords = {
            "개정",
            "이유",
            "설명",
            "설명해줘",
            "찾아줘",
            "현행",
            "규정",
            "현재",
            "무슨",
            "내용",
            "실무",
            "참고",
            "주의",
            "과거",
            "이전",
            "예전",
            "연혁",
            "변천",
            "발전",
            "군인의",
            "지위",
            "복무",
            "기본법",
            "시행령",
            "시행규칙",
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
            candidate = self._normalize_text(line.lstrip("-").strip())
            if len(candidate) < 8:
                continue
            raw_lines.append(candidate)

        points: list[str] = []
        for line in raw_lines:
            if line not in points:
                points.append(self._trim_sentence(line, limit=150))
            if len(points) >= limit:
                return points[:limit]

        normalized = self._normalize_text(text)
        pieces = re.split(r"(?<=[.!?])\s+|(?<=다)\s+", normalized)
        for piece in pieces:
            candidate = self._normalize_text(piece)
            if len(candidate) < 12:
                continue
            if focus_terms and not any(term in candidate for term in focus_terms) and len(points) >= 1:
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
        body = " ".join(points) if points else "현재 자료 기준으로 관련 내용을 요약하기 어렵습니다."
        return f"- {self._display_law_level(hit)} {self._article_ref(hit)}: {body}"

    def _reason_summary_points(self, reason_hits: list[SearchHit], focus_terms: list[str]) -> list[str]:
        points: list[str] = []
        combined = " ".join(self._normalize_text(hit.chunk.text) for hit in reason_hits)
        if "출산" in combined:
            points.append("출산·돌봄 지원 확대가 주요 개정 배경으로 확인됩니다.")
        if "일·가정 양립" in combined or "양립" in combined:
            points.append("일·가정 양립을 강화하려는 취지가 반복적으로 드러납니다.")
        if "근무 여건" in combined or "복무 여건" in combined:
            points.append("복무 여건과 근무 환경을 보완하려는 목적이 함께 확인됩니다.")

        for hit in reason_hits:
            for item in self._extract_points(hit.chunk.text, focus_terms=focus_terms, limit=2):
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        return points[:3]

    def _history_link_points(self, history_hits: list[SearchHit], focus_terms: list[str]) -> list[str]:
        points: list[str] = []
        combined = " ".join(self._normalize_text(hit.chunk.text) for hit in history_hits)
        if "군인복무규율" in combined:
            points.append(
                "현재 기본법과 관련 시행령·시행규칙은 군인복무규율 체계에서 발전한 관계에 있으며, 기본 원칙은 법률로, 세부 기준은 하위 법령으로 재구성되었습니다."
            )
        if "기본법 체계로 넘어오" in combined or "법체계로 넘어오" in combined:
            points.append(
                "군인복무규율 중심 체계에서 기본법·시행령·시행규칙 체계로 넘어오면서 연결성 있는 기준이 단계별로 정리되었습니다."
            )

        for hit in history_hits:
            for item in self._extract_points(hit.chunk.text, focus_terms=focus_terms, limit=2):
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        return points[:3]

    def _change_points(self, compare_hits: list[SearchHit], law_hits: list[SearchHit], focus_terms: list[str]) -> list[str]:
        points: list[str] = []
        for hit in compare_hits:
            for item in self._extract_points(hit.chunk.text, focus_terms=focus_terms, limit=3):
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        for hit in law_hits:
            for item in self._extract_points(hit.chunk.text, focus_terms=focus_terms, limit=1):
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        return points[:3]

    def _conclusion_from_search(self, question: str, law_hits: list[SearchHit]) -> str:
        if not law_hits:
            return "현재 자료 기준으로 관련 현행 규정을 충분히 찾지 못했습니다."
        focus_terms = self._question_focus_terms(question)
        if "휴가" in focus_terms and any("제8조" in self._article_ref(hit) for hit in law_hits):
            if any(self._display_law_level(hit) == "시행령" for hit in law_hits):
                return "군인의 휴가 관련 기준은 기본법 조문과 시행령 세부 기준을 함께 확인하는 구조입니다."
        primary = law_hits[0]
        return f"현재 자료 기준으로 {primary.chunk.law_name} {self._article_ref(primary)}에서 관련 기준을 확인할 수 있습니다."

    def _conclusion_from_explain(
        self,
        question: str,
        reason_hits: list[SearchHit],
        history_hits: list[SearchHit],
    ) -> str:
        if not reason_hits and not history_hits:
            return "현재 자료 기준으로 개정 이유나 연혁을 직접 설명할 만한 자료가 충분하지 않습니다."
        history_combined = " ".join(self._normalize_text(hit.chunk.text) for hit in history_hits)
        combined = " ".join(self._normalize_text(hit.chunk.text) for hit in reason_hits)
        focus_terms = self._question_focus_terms(question)
        if "군인복무규율" in history_combined:
            return "현재 자료를 보면, 군인복무규율 체계에서 기본법과 관련 시행령·시행규칙 체계로 발전한 흐름 속에서 해당 기준이 정비되었습니다."
        if "출산" in combined and ("육아시간" in focus_terms or "돌봄휴가" in "".join(focus_terms)):
            return "현재 자료를 보면, 출산·돌봄 지원 확대가 이번 개정의 핵심 배경으로 보입니다."
        if "일·가정 양립" in combined or "양립" in combined:
            return "현재 자료를 보면, 이번 개정은 일·가정 양립 지원을 강화하려는 방향으로 보입니다."
        return "현재 자료를 보면, 관련 제도의 범위와 운영 기준을 보완하려는 방향으로 개정된 것으로 보입니다."

    def _conclusion_from_practical(self, law_hits: list[SearchHit]) -> str:
        if not law_hits:
            return "현재 자료 기준으로 실무 참고에 필요한 현행 조문을 충분히 찾지 못했습니다."
        if any("제8조" in self._article_ref(hit) for hit in law_hits):
            return "실무적으로는 기본법의 제한 사유와 시행령의 세부 기준을 함께 확인하는 방식이 가장 안전합니다."
        primary = law_hits[0]
        return f"실무적으로는 {primary.chunk.law_name} {self._article_ref(primary)}부터 확인하는 것이 적절합니다."

    def _build_search_answer(self, question: str, evidence: list[SearchHit]) -> str:
        law_hits = self._source_hits(evidence, "law_text")
        reason_hits = self._source_hits(evidence, "revision_reason")
        focus_terms = self._question_focus_terms(question)

        level_order = {"법률": 0, "시행령": 1, "시행규칙": 2}
        ordered_hits = sorted(
            law_hits,
            key=lambda hit: (
                level_order.get(self._display_law_level(hit), 9),
                0 if "제8조" in self._article_ref(hit) else 1,
                0 if "제2조의6" in self._article_ref(hit) else 1,
                0 if "제2조" in self._article_ref(hit) else 1,
            ),
        )
        main_lines = [self._line_for_hit(hit, focus_terms=focus_terms, max_points=2) for hit in ordered_hits[:3]]
        if not main_lines:
            main_lines = ["- 현재 자료 기준으로 관련 현행 조문을 충분히 찾지 못했습니다."]

        apply_points: list[str] = []
        if any("제2조의6" in self._article_ref(hit) or "5분의 1" in hit.chunk.text for hit in law_hits):
            apply_points.append("휴가 확인 범위는 부대 현재 병력의 5분의 1 이내 기준이 시행령에 제시됩니다.")
        if any("육아시간" in hit.chunk.text or "돌봄휴가" in hit.chunk.text for hit in law_hits):
            apply_points.append("돌봄·육아 관련 질문은 휴가 조문뿐 아니라 시행령 제2조 계열 세부 기준까지 함께 보는 것이 안전합니다.")
        if reason_hits and ("휴가" in focus_terms or "육아시간" in focus_terms or "돌봄휴가" in focus_terms):
            reason_points = self._reason_summary_points(reason_hits, focus_terms)
            if reason_points:
                apply_points.append(f"최근 개정 흐름으로는 {reason_points[0]}")
        if not apply_points:
            apply_points.append("현행 조문과 시행령을 함께 대조해 휴가 종류, 일수, 제한 사유, 확인 범위를 직접 확인하는 방식이 적절합니다.")

        return (
            f"### 핵심 결론\n{self._conclusion_from_search(question, law_hits)}\n\n"
            f"### 주요 규정\n" + "\n".join(main_lines) + "\n\n"
            f"### 실무 참고\n" + "\n".join(f"- {point}" if not point.startswith("- ") else point for point in apply_points[:3]) + "\n\n"
            "### 근거\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _build_explain_answer(self, question: str, evidence: list[SearchHit]) -> str:
        reason_hits = self._source_hits(evidence, "revision_reason")
        compare_hits = self._source_hits(evidence, "old_new_comparison")
        history_hits = self._source_hits(evidence, "history_note")
        law_hits = self._source_hits(evidence, "law_text")
        focus_terms = self._question_focus_terms(question)

        reason_points = self._reason_summary_points(reason_hits, focus_terms)
        history_points = self._history_link_points(history_hits, focus_terms)
        for point in history_points:
            if point not in reason_points:
                reason_points.append(point)
        if not reason_points:
            reason_points = ["현재 자료 기준으로 개정 이유를 직접 적시한 자료가 충분하지 않습니다."]

        change_points = self._change_points(compare_hits, law_hits, focus_terms)
        for point in history_points:
            if point not in change_points:
                change_points.append(point)
        if not change_points:
            change_points = ["현재 자료 기준으로 구체적인 변경 내용을 충분히 추리기 어렵습니다."]

        interpretation_points: list[str] = []
        combined = " ".join(self._normalize_text(hit.chunk.text) for hit in reason_hits)
        if "출산" in combined:
            interpretation_points.append("출산·돌봄 지원을 확대하기 위한 개정으로 해석할 수 있습니다.")
        if "일·가정 양립" in combined or "양립" in combined:
            interpretation_points.append("군 복무와 가정 돌봄을 병행할 수 있도록 대상과 운영 기준을 넓힌 개정으로 볼 수 있습니다.")
        if "근무 여건" in combined or "복무 여건" in combined:
            interpretation_points.append("근무 여건과 복무 환경을 보완하려는 취지가 함께 드러납니다.")
        for point in history_points:
            if point not in interpretation_points:
                interpretation_points.append(point)
        if not interpretation_points:
            interpretation_points.append("현재 자료 기준으로는 제도 범위와 운영 기준을 보완하려는 방향으로 해석됩니다.")

        return (
            f"### 핵심 결론\n{self._conclusion_from_explain(question, reason_hits, history_hits)}\n\n"
            f"### 주요 개정 이유\n" + "\n".join(f"- {point}" for point in reason_points[:3]) + "\n\n"
            f"### 실제 제도 변화\n" + "\n".join(f"- {point}" for point in change_points[:3]) + "\n\n"
            f"### 해석 시사점\n" + "\n".join(f"- {point}" for point in interpretation_points[:3]) + "\n\n"
            "### 근거\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _build_practical_answer(self, question: str, evidence: list[SearchHit]) -> str:
        law_hits = self._source_hits(evidence, "law_text")
        focus_terms = self._question_focus_terms(question)

        practical_points: list[str] = []
        if law_hits:
            practical_points.append("먼저 법률 조문에서 보장 범위와 제한 사유를 확인하고, 이어서 시행령에서 종류·일수·시간 범위를 확인하는 순서가 적절합니다.")
        if any("제8조" in self._article_ref(hit) for hit in law_hits):
            practical_points.append("휴가 제한이나 보류 사유는 작전상황, 교육훈련, 징계·형사 절차, 부대병력 유지 필요 여부까지 함께 점검해야 합니다.")
        if any("5분의 1" in hit.chunk.text or "확인 범위" in hit.chunk.text for hit in law_hits):
            practical_points.append("휴가 확인 범위는 부대 병력 상황에 따라 조정될 수 있으므로 시행령 기준과 부대 운용 상황을 함께 봐야 합니다.")
        if any(term in "".join(focus_terms) for term in ["육아시간", "돌봄휴가", "청원휴가"]):
            practical_points.append("돌봄·육아 관련 사안은 대상 요건, 사용 기간, 시간 단위 기준을 시행령 조문까지 같이 확인하는 것이 안전합니다.")
        if not practical_points:
            practical_points.append("현행 조문과 시행령 적용 대상을 먼저 대조하는 방식이 안전합니다.")

        caution_points = [
            "이 답변은 실무 참고용 요약입니다.",
            "실제 인사·징계·복무 처리는 최신 원문과 소속 부대 지침을 함께 확인하는 것이 안전합니다.",
        ]

        return (
            f"### 핵심 결론\n{self._conclusion_from_practical(law_hits)}\n\n"
            f"### 실무적으로 보면\n" + "\n".join(f"- {point}" for point in practical_points[:4]) + "\n\n"
            f"### 주의사항\n" + "\n".join(f"- {point}" for point in caution_points) + "\n\n"
            "### 근거\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _build_hybrid_answer(self, question: str, evidence: list[SearchHit]) -> str:
        reason_hits = self._source_hits(evidence, "revision_reason")
        history_hits = self._source_hits(evidence, "history_note")
        focus_terms = self._question_focus_terms(question)
        reason_points = self._reason_summary_points(reason_hits, focus_terms)
        history_points = self._history_link_points(history_hits, focus_terms)
        for point in history_points:
            if point not in reason_points:
                reason_points.append(point)
        if not reason_points:
            reason_points = ["현재 자료 기준으로 개정 이유 자료가 충분하지 않습니다."]

        practical_answer = self._build_practical_answer(question, evidence)
        practical_body = practical_answer.split("### 실무적으로 보면\n", 1)[-1].split("\n\n### 주의사항", 1)[0].strip()

        return (
            f"### 핵심 결론\n{self._conclusion_from_explain(question, reason_hits, history_hits)}\n\n"
            f"### 주요 개정 이유\n" + "\n".join(f"- {point}" for point in reason_points[:3]) + "\n\n"
            f"### 실무적으로 보면\n{practical_body}\n\n"
            "### 근거\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
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
                "### 핵심 결론\n현재 자료 기준으로 개정 이유나 연혁을 충분히 찾지 못했습니다.\n\n"
                "### 주요 개정 이유\n- 관련 개정이유 또는 신구 비교 자료가 부족합니다.\n\n"
                "### 실제 제도 변화\n- 구체적인 변경 내용을 확인할 근거가 충분하지 않습니다.\n\n"
                "### 해석 시사점\n- 질문을 더 구체화하거나 자료 유형을 조정해 다시 검색해 주세요.\n\n"
                "### 근거\n- 현재 제시할 근거가 없습니다."
            )
        if intent == "practical":
            return (
                "### 핵심 결론\n현재 자료 기준으로 실무 참고에 필요한 근거가 부족합니다.\n\n"
                "### 실무적으로 보면\n- 질문 범위를 더 좁혀 다시 검색하는 것이 좋습니다.\n\n"
                "### 주의사항\n- 관련 조문과 시행령을 확인할 수 있는 공개 자료가 더 필요합니다.\n\n"
                "### 근거\n- 현재 제시할 근거가 없습니다."
            )
        return (
            "### 핵심 결론\n현재 자료 기준으로 관련 현행 규정을 충분히 찾지 못했습니다.\n\n"
            "### 주요 규정\n- 관련 조문이 충분히 검색되지 않았습니다.\n\n"
            "### 실무 참고\n- 질문을 더 구체화하거나 자료 유형을 조정해 다시 검색해 주세요.\n\n"
            "### 근거\n- 현재 제시할 근거가 없습니다."
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
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice=QUOTA_BLOCK_MESSAGE,
                quota_snapshot=snapshot,
            )

        if self._client is None or types is None:
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice="Gemini API가 준비되지 않아 근거 기반 요약 모드로 전환했습니다.",
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
                    notice="Gemini 응답 본문이 비어 있어 근거 기반 요약으로 전환했습니다.",
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
                return self._fallback_result(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    notice=PROVIDER_RATE_LIMIT_NOTICE,
                    quota_snapshot=self.usage_tracker.snapshot(),
                )
            return self._fallback_result(
                question=question,
                intent=intent,
                evidence=evidence,
                notice="Gemini API 호출 오류로 인해 근거 기반 요약 모드로 전환했습니다.",
                quota_snapshot=self.usage_tracker.snapshot(),
            )
