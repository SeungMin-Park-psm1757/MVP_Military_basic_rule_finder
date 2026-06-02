from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from army_reg_rag.config import Settings
from army_reg_rag.domain.models import SearchHit
from army_reg_rag.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from army_reg_rag.llm.usage_tracker import GeminiUsageTracker
from army_reg_rag.utils.runtime_config import get_runtime_value
from army_reg_rag.utils.text_cleanup import repair_common_text_artifacts

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

TIMELINE_QUESTION_KEYWORDS = ["연혁", "변천", "변천사", "흐름", "이어졌"]
GENERIC_FOCUS_STOPWORDS = {
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
    "내용을",
    "사항",
    "정리",
    "정리해줘",
    "정리해서",
    "나열",
    "나열해줘",
    "요약",
    "요약해줘",
    "실무",
    "참고",
    "주의",
    "좀",
    "한번",
    "위주",
    "위주로",
    "과거",
    "이전",
    "예전",
    "연혁",
    "변천",
    "변천사",
    "흐름",
    "군인의",
    "지위",
    "복무",
    "기본법",
    "시행령",
    "시행규칙",
    "관련",
    "알려줘",
}
FOCUS_TERM_EXPANSIONS = {
    "징계": [
        "징계",
        "징계혐의자",
        "징계조치",
        "불이익조치",
        "신고자 보호",
        "신고의무",
        "사적 제재",
        "가혹행위",
        "구타",
        "폭언",
        "상벌",
        "군기강",
        "징계 절차",
        "징계절차",
        "징계의 종류",
        "징계 종류",
        "징계위원회",
        "군인사법",
        "군인 징계령",
    ],
    "휴가": ["휴가", "외출", "외박", "연가", "청원휴가", "특별휴가", "정기휴가", "휴가 보류"],
    "육아": ["육아", "육아시간", "육아휴직", "돌봄", "자녀돌봄휴가", "배우자 출산휴가", "임신검진", "모성보호시간"],
    "돌봄": ["돌봄", "자녀돌봄휴가", "육아", "육아시간", "배우자 출산휴가", "임신검진", "모성보호시간"],
    "출산": ["출산", "배우자 출산휴가", "임신검진", "모성보호시간", "육아", "돌봄"],
    "휴직": ["휴직", "육아휴직", "복직", "육아", "돌봄"],
    "고충": ["고충", "고충 처리", "고충심사", "군인고충심사위원회", "재심청구", "군인사법", "중앙 군인사소청심사위원회"],
    "권리구제": ["권리구제", "고충", "고충 처리", "고충심사", "군인고충심사위원회", "재심청구", "군인사법", "중앙 군인사소청심사위원회"],
    "신고": ["신고", "신고자", "신고의무", "신고자 보호", "불이익조치", "가혹행위"],
}
LIST_MARKER_RE = re.compile(r"^(?:\d+[.)]|[가-하][.),]|[①-⑳]|[\(\[]\d+[\)\]])\s*")
LEADING_SYMBOL_RE = re.compile(r"^[◇◆⊙■□※]+")
NOISY_LEGAL_LINE_RE = re.compile(
    r"^(?:대통령령|법률|총리령|부령|훈령|예규|고시|영제|별표|별지|제\d+조중|다음과 같이 개정한다)"
)
NOUN_FRAGMENT_ENDINGS = ("자", "자들", "사항", "사유", "기준", "범위", "대상", "절차", "규정", "조문", "원칙")


@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    backend: str
    notice: str = ""
    quota_snapshot: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


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
            "징계",
            "신고자 보호",
            "비밀보장",
            "신고자에 대한 비밀보장",
            "가혹행위",
            "불이익조치",
            "상벌",
            "고충심사",
            "권리구제",
            "군인고충심사위원회",
            "군인사법",
            "군인 징계령",
        ]
        terms = [keyword for keyword in preferred if keyword in question]
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", question)
        for token in tokens:
            if len(token) < 2 or token in GENERIC_FOCUS_STOPWORDS or token in terms:
                continue
            terms.append(token)
        expanded_terms: list[str] = []
        for term in terms:
            if term not in expanded_terms:
                expanded_terms.append(term)
            for key, synonyms in FOCUS_TERM_EXPANSIONS.items():
                if term == key or term in synonyms:
                    for synonym in synonyms:
                        if synonym not in expanded_terms:
                            expanded_terms.append(synonym)
        return expanded_terms[:12]

    def _is_timeline_question(self, question: str) -> bool:
        return any(keyword in question for keyword in TIMELINE_QUESTION_KEYWORDS)

    def _collapse_repeated_phrase(self, text: str) -> str:
        compact = " ".join(text.split())
        parts = compact.split()
        if len(parts) >= 2 and len(parts) % 2 == 0:
            halfway = len(parts) // 2
            if parts[:halfway] == parts[halfway:]:
                return " ".join(parts[:halfway])
        return compact

    def _normalize_text(self, text: str) -> str:
        cleaned = text.replace("\u00a0", " ").strip()
        for token in NOISY_TOKENS:
            cleaned = cleaned.replace(token, " ")
        cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return self._collapse_repeated_phrase(cleaned)

    def _trim_sentence(self, text: str, *, limit: int = 150) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return self._ensure_sentence_ending(compact)
        candidate = compact[:limit]
        sentence_boundary = max(candidate.rfind(". "), candidate.rfind("다. "), candidate.rfind("; "))
        if sentence_boundary >= int(limit * 0.55):
            trimmed = candidate[: sentence_boundary + 1].rstrip(" ,")
            return self._ensure_sentence_ending(trimmed)
        truncated = compact[: limit - 1]
        last_space = truncated.rfind(" ")
        if last_space >= 0:
            truncated = truncated[:last_space]
        return truncated.rstrip(" ,") + "..."

    def _ensure_sentence_ending(self, text: str) -> str:
        compact = " ".join((text or "").split()).strip()
        if not compact or compact.endswith(("...", "…", ".", "!", "?")):
            return compact

        trailing_wrappers = "\"'”’)]】"
        base = compact.rstrip(trailing_wrappers).rstrip()
        if self._looks_like_complete_sentence(base):
            return f"{compact}."
        return compact

    def _strip_list_marker(self, text: str) -> str:
        return LIST_MARKER_RE.sub("", text or "").strip()

    def _clean_fragment_text(self, text: str) -> str:
        compact = self._normalize_text(text)
        compact = LEADING_SYMBOL_RE.sub("", compact).strip()
        compact = self._strip_list_marker(compact)
        compact = re.sub(r"<[^>]*>", " ", compact)
        compact = re.sub(r"<[^>]*$", "", compact)
        compact = re.sub(r"\s+", " ", compact).strip(" ,;:")
        return compact

    def _looks_like_noisy_legal_line(self, text: str) -> bool:
        compact = self._clean_fragment_text(text)
        if not compact:
            return True
        if NOISY_LEGAL_LINE_RE.match(compact):
            return True
        if compact.count("제") >= 4 and compact.count("조") >= 2 and len(compact) > 90:
            return True
        if compact.startswith("개정") and len(compact) < 20:
            return True
        return False

    def _looks_like_complete_sentence(self, text: str) -> bool:
        compact = " ".join((text or "").split()).rstrip()
        if not compact:
            return False
        tail = compact.rstrip(".").rstrip()
        if tail.endswith(
            (
                "다",
                "니다",
                "합니다",
                "됩니다",
                "있습니다",
                "없습니다",
                "하였다",
                "한다",
                "된다",
                "있다",
                "없다",
                "보인다",
                "확인된다",
                "규정한다",
                "정한다",
                "말한다",
                "금지한다",
                "보장한다",
                "적용한다",
            )
        ):
            return True
        return False

    def _looks_like_fragment(self, text: str) -> bool:
        compact = self._clean_fragment_text(text)
        if not compact:
            return False
        if self._looks_like_noisy_legal_line(compact):
            return True
        if self._looks_like_complete_sentence(compact):
            return False
        marker_stripped = self._strip_list_marker(compact)
        if not marker_stripped:
            return True
        if len(marker_stripped) <= 40:
            return True
        return marker_stripped.endswith(NOUN_FRAGMENT_ENDINGS)

    def _contextualize_fragment(self, lines: list[str], index: int) -> str:
        fragment = self._clean_fragment_text(lines[index])
        if not fragment:
            return ""
        previous_line = lines[index - 1] if index > 0 else ""
        previous_line = self._normalize_text(previous_line)
        if self._looks_like_complete_sentence(previous_line):
            base = previous_line.rstrip(".")
            suffix = " 등이 제시됩니다." if not fragment.endswith("등") else "이 제시됩니다."
            return self._trim_sentence(f"{base}. 대표 항목으로는 {fragment}{suffix}", limit=150)
        if len(fragment) <= 60:
            suffix = " 등이 관련 대상으로 제시됩니다." if not fragment.endswith("등") else "이 관련 대상으로 제시됩니다."
            return self._trim_sentence(f"{fragment}{suffix}", limit=150)
        return ""

    def _summary_suffix_for_hit(self, hit: SearchHit) -> str:
        article_title = (hit.chunk.article_title or "").strip()
        if "보류" in article_title:
            return "를 보류 대상으로 두었습니다."
        if "금지" in article_title:
            return "를 금지 기준으로 두었습니다."
        if "보호" in article_title:
            return "를 보호 기준으로 두었습니다."
        if "의무" in article_title:
            return "를 의무로 두었습니다."
        if "정의" in article_title:
            return "를 정의합니다."
        if "휴가" in article_title:
            return "를 휴가 기준으로 두었습니다."
        if hit.chunk.source_type == "revision_reason":
            return "를 제정·개정 이유로 제시했습니다."
        if hit.chunk.source_type == "old_new_comparison":
            return "를 개정문에 반영했습니다."
        return "를 관련 기준으로 두었습니다."

    def _metadata_summary_for_hit(self, hit: SearchHit, fragment: str) -> str:
        compact = self._clean_fragment_text(fragment).rstrip(".")
        label = f"{hit.chunk.law_name} {self._article_ref(hit)}".strip()
        when = hit.chunk.effective_date or hit.chunk.promulgation_date or ""
        intro = f"{when} {label}".strip() if when else label
        if not compact:
            if hit.chunk.source_type == "revision_reason":
                return self._trim_sentence(f"{intro}에서는 제정·개정 취지를 확인할 수 있습니다.", limit=150)
            if hit.chunk.source_type == "old_new_comparison":
                return self._trim_sentence(f"{intro}에서는 개정 전후 변화 내용을 확인할 수 있습니다.", limit=150)
            return self._trim_sentence(f"{intro}에서는 관련 기준을 확인할 수 있습니다.", limit=150)
        article_title = (hit.chunk.article_title or "").strip()
        if "비밀보장" in article_title:
            return "신고자라는 사실이나 인적사항을 다른 사람에게 알리거나 공개·보도하지 못하게 하는 비밀보장 규정입니다."
        if "신고자 보호" in article_title and any(term in compact for term in ["불이익", "징계조치", "차별대우"]):
            return "신고등을 이유로 한 징계조치나 차별대우 같은 불이익조치를 금지하는 보호 규정입니다."
        if "신고의무" in article_title and any(term in compact for term in ["신고", "보고", "군인권보호관"]):
            return "가혹행위나 성폭력 등 사적 제재 사실을 알게 된 경우 보고 또는 신고하도록 하는 의무 규정입니다."
        if hit.chunk.source_type == "revision_reason":
            if all(term in compact for term in ["기본권", "신고", "보호"]):
                return "기본권 교육, 가혹행위 신고의무, 신고자 보호를 제도화하려는 취지가 확인됩니다."
            if "군인복무규율" in hit.chunk.law_name and any(term in compact for term in ["복무자세", "군기강", "인권침해", "기본권보장"]):
                return "군인복무규율은 복무자세와 군기강 정비, 인권침해 관행 해소, 기본권 보장 보완을 개정 취지로 제시했습니다."
        if self._looks_like_complete_sentence(compact) and len(compact) <= 140 and not self._looks_like_noisy_legal_line(compact):
            return self._trim_sentence(compact, limit=150)
        suffix = self._summary_suffix_for_hit(hit)
        return self._trim_sentence(f"{intro}에서는 {compact}{suffix}", limit=150)

    def _summary_for_hit(self, hit: SearchHit, *, focus_terms: list[str], limit: int = 150) -> str:
        raw_lines = [
            self._clean_fragment_text(line.lstrip("-").strip())
            for line in self._hit_text(hit).replace("•", "\n- ").splitlines()
            if self._clean_fragment_text(line.lstrip("-").strip())
        ]
        fragment_candidates = [line for line in raw_lines if self._looks_like_fragment(line)]
        if focus_terms and fragment_candidates:
            fragment_candidates.sort(
                key=lambda line: (
                    sum(line.count(term) for term in focus_terms if term),
                    len(line),
                ),
                reverse=True,
            )
        fragment_source = fragment_candidates[0] if fragment_candidates else (raw_lines[0] if raw_lines else "")
        points = self._extract_points(self._hit_text(hit), focus_terms=focus_terms, limit=1)
        primary = points[0] if points else ""
        if primary and ("관련 대상으로 제시됩니다" in primary or "대표 항목으로는" in primary):
            return self._trim_sentence(self._metadata_summary_for_hit(hit, fragment_source), limit=limit)
        summary = self._metadata_summary_for_hit(hit, primary)
        if summary:
            return self._trim_sentence(summary, limit=limit)
        return self._trim_sentence("현재 자료 기준으로 관련 내용을 충분히 요약하기 어렵습니다.", limit=limit)

    def _hit_text(self, hit: SearchHit) -> str:
        return str(hit.chunk.extra.get("summary_text") or hit.chunk.extra.get("display_text") or hit.chunk.text)

    def _extract_points(self, text: str, *, focus_terms: list[str], limit: int = 3) -> list[str]:
        raw_lines: list[str] = []
        for line in text.replace("•", "\n- ").splitlines():
            candidate = self._clean_fragment_text(line.lstrip("-").strip())
            if len(candidate) < 8:
                continue
            if self._looks_like_noisy_legal_line(candidate):
                continue
            raw_lines.append(candidate)

        candidate_lines: list[str] = []
        for index, line in enumerate(raw_lines):
            if line not in candidate_lines:
                candidate_lines.append(line)
            if self._looks_like_fragment(line):
                contextualized = self._contextualize_fragment(raw_lines, index)
                if contextualized and contextualized not in candidate_lines:
                    candidate_lines.append(contextualized)

        def focus_score(value: str) -> tuple[int, int, int, int]:
            direct_hits = sum(1 for term in focus_terms if term and term in value)
            weighted_hits = sum(value.count(term) for term in focus_terms if term and term in value)
            has_focus = 1 if direct_hits > 0 or not focus_terms else 0
            sentence_like = 1 if self._looks_like_complete_sentence(value) else 0
            return (has_focus, sentence_like, weighted_hits, -len(value))

        points: list[str] = []
        ordered_lines = list(candidate_lines)
        if focus_terms:
            ordered_lines.sort(key=focus_score, reverse=True)
        for line in ordered_lines:
            if focus_terms and not any(term in line for term in focus_terms) and points:
                continue
            if line not in points:
                points.append(self._trim_sentence(line, limit=150))
            if len(points) >= limit:
                return points[:limit]

        normalized = self._normalize_text(text)
        pieces = re.split(r"(?<=[.!?])\s+|(?<=다)\s+", normalized)
        ordered_pieces = [self._normalize_text(piece) for piece in pieces]
        ordered_pieces = [piece for piece in ordered_pieces if len(piece) >= 12]
        if focus_terms:
            ordered_pieces.sort(key=focus_score, reverse=True)
        for candidate in ordered_pieces:
            if self._looks_like_fragment(candidate):
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

    def _scope_label(self, hit: SearchHit) -> str:
        scope = str(hit.chunk.extra.get("scope", "")).strip()
        return scope or hit.chunk.version_label or self._article_ref(hit)

    def _expected_headings(self, question: str, intent: str) -> list[str]:
        if intent == "explain_change" and self._is_timeline_question(question):
            return [
                "## 답변 개요",
                "### 핵심 결론",
                "## 연혁 정리",
                "### 시기별 변화",
                "## 현재 체계",
                "### 현재 체계와 연결",
                "## 근거 안내",
                "### 확인 방법",
            ]
        if intent == "search":
            return ["## 답변 개요", "### 핵심 결론", "## 세부 정리", "### 주요 규정", "### 실무 참고", "## 근거 안내", "### 확인 방법"]
        if intent == "explain_change":
            return [
                "## 답변 개요",
                "### 핵심 결론",
                "## 세부 정리",
                "### 주요 개정 이유",
                "### 실제 제도 변화",
                "### 해석 시사점",
                "## 근거 안내",
                "### 확인 방법",
            ]
        if intent == "practical":
            return ["## 답변 개요", "### 핵심 결론", "## 실무 정리", "### 실무적으로 보면", "### 주의사항", "## 근거 안내", "### 확인 방법"]
        return ["## 답변 개요", "### 핵심 결론", "## 세부 정리", "### 주요 개정 이유", "### 실무적으로 보면", "## 근거 안내", "### 확인 방법"]

    def _is_structured_answer(self, question: str, intent: str, text: str) -> bool:
        expected = self._expected_headings(question, intent)
        return all(heading in text for heading in expected)

    def _line_for_hit(self, hit: SearchHit, *, focus_terms: list[str], max_points: int = 2) -> str:
        body = self._summary_for_hit(hit, focus_terms=focus_terms)
        return f"- {self._display_law_level(hit)} {self._article_ref(hit)}: {body}"

    def _reason_summary_points(self, reason_hits: list[SearchHit], focus_terms: list[str]) -> list[str]:
        points: list[str] = []
        combined = " ".join(self._normalize_text(self._hit_text(hit)) for hit in reason_hits)
        if "출산" in combined:
            points.append("출산·돌봄 지원 확대가 주요 개정 배경으로 확인됩니다.")
        if "육아" in combined or "자녀돌봄휴가" in combined or "육아시간" in combined:
            points.append("육아시간, 자녀돌봄휴가 등 가족돌봄 관련 제도를 넓히려는 흐름이 보입니다.")
        if "휴직" in combined:
            points.append("휴가뿐 아니라 휴직·복직과 연결되는 지원 기준을 함께 보완하려는 방향이 확인됩니다.")
        if "일·가정 양립" in combined or "양립" in combined:
            points.append("일·가정 양립을 강화하려는 취지가 반복적으로 드러납니다.")
        if "근무 여건" in combined or "복무 여건" in combined:
            points.append("복무 여건과 근무 환경을 보완하려는 목적이 함께 확인됩니다.")

        for hit in reason_hits:
            for item in self._extract_points(self._hit_text(hit), focus_terms=focus_terms, limit=2):
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        return points[:3]

    def _history_link_points(self, history_hits: list[SearchHit], focus_terms: list[str]) -> list[str]:
        points: list[str] = []
        combined = " ".join(self._normalize_text(self._hit_text(hit)) for hit in history_hits)
        if "군인복무규율" in combined:
            points.append(
                "현재 기본법과 관련 시행령·시행규칙은 군인복무규율 체계에서 발전한 관계에 있으며, 기본 원칙은 법률로, 세부 기준은 하위 법령으로 재구성되었습니다."
            )
        if "기본법 체계로 넘어오" in combined or "법체계로 넘어오" in combined:
            points.append(
                "군인복무규율 중심 체계에서 기본법·시행령·시행규칙 체계로 넘어오면서 연결성 있는 기준이 단계별로 정리되었습니다."
            )

        for hit in history_hits:
            for item in self._extract_points(self._hit_text(hit), focus_terms=focus_terms, limit=2):
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        return points[:3]

    def _change_points(self, compare_hits: list[SearchHit], law_hits: list[SearchHit], focus_terms: list[str]) -> list[str]:
        points: list[str] = []
        for hit in compare_hits:
            for item in self._extract_points(self._hit_text(hit), focus_terms=focus_terms, limit=3):
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        for hit in law_hits:
            for item in self._extract_points(self._hit_text(hit), focus_terms=focus_terms, limit=1):
                if item not in points:
                    points.append(item)
                if len(points) >= 3:
                    return points[:3]
        return points[:3]

    def _timeline_entry_markdown(self, hit: SearchHit, *, focus_terms: list[str]) -> str:
        when = hit.chunk.effective_date or hit.chunk.promulgation_date or "시기 미상"
        summary = self._summary_for_hit(hit, focus_terms=focus_terms)
        title = f"{when} | {hit.chunk.law_name} | {self._scope_label(hit)}"
        return f"#### {title}\n- {summary}"

    def _current_link_markdown(self, hit: SearchHit, *, focus_terms: list[str]) -> str:
        summary = self._summary_for_hit(hit, focus_terms=focus_terms)
        title = f"{hit.chunk.law_name} {self._article_ref(hit)}"
        return f"#### {title}\n- {summary}"

    def _conclusion_from_search(self, question: str, law_hits: list[SearchHit]) -> str:
        if not law_hits:
            return "현재 자료 기준으로 관련 현행 규정을 충분히 찾지 못했습니다."
        focus_terms = self._question_focus_terms(question)
        question_blob = self._normalize_text(question)
        combined = " ".join(self._normalize_text(self._hit_text(hit)) for hit in law_hits)
        article_blob = " ".join(f"{self._article_ref(hit)} {hit.chunk.article_title}" for hit in law_hits)
        asks_reporter_protection = any(term in question_blob for term in ["신고자 보호", "비밀보장", "제44조", "제45조"])
        has_confidentiality = "제44조" in article_blob or "비밀보장" in article_blob
        has_protection = "제45조" in article_blob or "신고자 보호" in article_blob
        if asks_reporter_protection and has_protection:
            if has_confidentiality:
                return "제44조의 비밀보장은 신고자 신분과 인적사항이 드러나지 않게 하는 전제 장치이고, 제45조의 신고자 보호는 신고를 이유로 한 징계조치 등 불이익조치를 금지하는 후속 보호 장치입니다."
            return "제45조의 신고자 보호는 신고를 이유로 한 징계조치 등 불이익조치를 금지하는 규정이며, 제44조와의 관계는 현재 검색된 근거 카드에서 제44조 원문을 함께 확인해야 합니다."
        if any(term in focus_terms for term in ["징계", "징계조치", "징계혐의자", "신고", "신고자 보호", "불이익조치", "가혹행위"]):
            if any(keyword in combined for keyword in ["징계조치", "불이익조치", "신고", "가혹행위"]):
                return "현재 자료 기준으로 징계 자체의 독립 절차보다는 신고자 보호, 가혹행위 대응, 불이익조치 금지처럼 연결된 기준이 먼저 확인됩니다."
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
        law_hits: list[SearchHit],
    ) -> str:
        if not reason_hits and not history_hits and not law_hits:
            return "현재 자료 기준으로 개정 이유나 연혁을 직접 설명할 만한 자료가 충분하지 않습니다."
        history_combined = " ".join(self._normalize_text(self._hit_text(hit)) for hit in history_hits)
        combined = " ".join(self._normalize_text(self._hit_text(hit)) for hit in reason_hits + law_hits)
        focus_terms = self._question_focus_terms(question)
        asks_childcare = any(term in focus_terms for term in ["육아", "돌봄", "출산", "자녀돌봄휴가", "배우자 출산휴가"])
        asks_leave_of_absence = any(term in focus_terms for term in ["휴직", "육아휴직"])
        has_childcare_evidence = any(keyword in combined for keyword in ["배우자가 출산", "출산휴가", "자녀돌봄", "육아시간", "청원휴가"])
        has_leave_of_absence_evidence = any(keyword in combined for keyword in ["육아휴직", "휴직", "복직"])
        if "징계" in focus_terms and ("징계혐의자" in combined or "불이익조치" in combined or "신고자 보호" in combined):
            return (
                "현재 자료 기준으로 보면, 과거 군인복무규율의 군기·징계혐의자 관리 중심 표현에서 "
                "기본법 제정 이후에는 기본권 침해 방지와 신고자 보호까지 법률 단계에서 직접 다루는 방향으로 이동했습니다."
            )
        if asks_childcare and has_childcare_evidence:
            if asks_leave_of_absence and not has_leave_of_absence_evidence:
                return (
                    "현재 자료를 보면 육아·돌봄 관련 기준은 군인복무규율의 휴가 체계에서 "
                    "배우자 출산휴가, 가족 간호, 청원휴가 세부 기준을 넓히는 방향으로 이어졌습니다. "
                    "다만 별도 휴직 기준은 현재 확보한 공개 코퍼스만으로는 충분히 설명되지 않아 연계 인사 규정을 함께 볼 필요가 있습니다."
                )
            return (
                "현재 자료를 보면 육아·돌봄 관련 기준은 군인복무규율의 휴가 체계에서 "
                "배우자 출산휴가와 가족 간호, 청원휴가 세부 기준을 더 구체화하는 방향으로 이어졌습니다."
            )
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

    @staticmethod
    def _join_labels(labels: list[str]) -> str:
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]}와 {labels[1]}"
        return ", ".join(labels[:-1]) + f", 그리고 {labels[-1]}"

    def _is_broad_overview_request(self, question: str) -> bool:
        if re.search(r"제\s*\d+조", question):
            return False
        return any(
            keyword in question
            for keyword in ["정리", "나열", "요약", "위주", "사항", "내용", "전반", "전체", "뭐가", "무엇"]
        )

    @staticmethod
    def _explicit_article_refs(question: str) -> list[str]:
        refs: list[str] = []
        for match in re.finditer(r"제\s*(\d+)조\s*(?:~|∼|-|부터)\s*(?:제\s*)?(\d+)조", question):
            start = int(match.group(1))
            end = int(match.group(2))
            if start <= end and end - start <= 20:
                refs.extend(f"제{article_no}조" for article_no in range(start, end + 1))

        refs.extend(
            match.group(0).replace(" ", "")
            for match in re.finditer(r"제\s*\d+조(?:의\s*\d+)?", question)
        )
        return list(dict.fromkeys(refs))

    def _first_hit_with_terms(
        self,
        hits: list[SearchHit],
        terms: list[str],
        *,
        exclude_ids: set[str] | None = None,
    ) -> SearchHit | None:
        excluded = exclude_ids or set()
        for hit in hits:
            if hit.chunk.id in excluded:
                continue
            haystack = self._normalize_text(
                " ".join(
                    [
                        hit.chunk.law_name,
                        self._article_ref(hit),
                        hit.chunk.article_title,
                        self._hit_text(hit),
                    ]
                )
            )
            if any(term in haystack for term in terms):
                return hit
        return None

    def _is_service_regulation_transition_question(self, question: str) -> bool:
        if "군인복무규율" not in question:
            return False
        has_transition = any(keyword in question for keyword in ["폐지", "재편", "체계", "넘어오", "이어졌", "변천", "연혁"])
        asks_scope = any(keyword in question for keyword in ["현재", "현행", "담당", "구분", "종류", "절차", "권리구제"])
        return has_transition and asks_scope

    def _has_direct_discipline_procedure_law(self, evidence: list[SearchHit]) -> bool:
        direct_law_names = {"군인사법", "군인 징계령", "군인징계령", "군인 징계령 시행규칙"}
        direct_terms = ["징계위원회", "징계의 종류", "징계 절차", "감봉", "견책"]
        for hit in evidence:
            if hit.chunk.law_name in direct_law_names:
                return True
            haystack = self._normalize_text(" ".join([hit.chunk.law_name, self._article_ref(hit), self._hit_text(hit)]))
            if any(term in haystack for term in direct_terms):
                return True
        return False

    def _transition_line(
        self,
        label: str,
        hits: list[SearchHit],
        terms: list[str],
        *,
        focus_terms: list[str],
        fallback: str,
    ) -> str:
        hit = None
        for candidate in hits:
            haystack = self._normalize_text(
                " ".join([candidate.chunk.law_name, self._article_ref(candidate), candidate.chunk.article_title, self._hit_text(candidate)])
            )
            if all(term in haystack for term in terms):
                hit = candidate
                break
        if hit is None:
            return f"- {label}: {fallback}"
        if label == "폐지 근거":
            when = hit.chunk.effective_date or "2016-06-30"
            return f"- {label}: {when} 시행된 기본법 시행령 부칙에서 군인복무규율 폐지를 확인할 수 있습니다."
        if label == "권리구제 축":
            return f"- {label}: 기본법 {self._article_ref(hit)}에서 군인고충심사위원회를 통한 고충 심사 청구와 불이익 금지 기준을 확인할 수 있습니다."
        if label == "군인사법 연계 단서":
            return "- 군인사법 연계 단서: 기본법 부칙에서 종전 「군인사법」의 휴가·복무·고충 관련 조항을 기본법 체계로 옮기는 경과조치가 확인됩니다."
        return f"- {label}: {self._summary_for_hit(hit, focus_terms=focus_terms, limit=190)}"

    def _build_service_regulation_transition_answer(self, question: str, evidence: list[SearchHit]) -> str:
        focus_terms = self._question_focus_terms(question)
        reason_hits = self._source_hits(evidence, "revision_reason")
        compare_hits = self._source_hits(evidence, "old_new_comparison")
        history_hits = self._source_hits(evidence, "history_note")
        law_hits = self._source_hits(evidence, "law_text")
        all_hits = reason_hits + compare_hits + history_hits + law_hits

        has_direct_discipline_law = self._has_direct_discipline_procedure_law(evidence)
        conclusion = (
            "현재 자료 기준으로 군인복무규율은 2016년 기본법 시행령 부칙에서 폐지되었고, "
            "복무·군기·권리구제 중 일부는 기본법과 시행령 체계로 재배치된 흐름이 확인됩니다. "
            "다만 징계의 종류·절차를 직접 담당하는 현행 법령 본문은 현재 코퍼스에 포함되어 있지 않아 이 앱 안에서는 확정 설명할 수 없습니다."
        )
        if has_direct_discipline_law:
            conclusion = (
                "현재 자료 기준으로 군인복무규율 폐지 이후 복무·군기·권리구제 사항은 기본법과 시행령 체계로 재배치되었고, "
                "징계 종류·절차도 별도 징계·인사 법령 근거와 함께 구분해 확인할 수 있습니다."
            )

        transition_points = [
            self._transition_line(
                "폐지 근거",
                compare_hits + history_hits,
                ["군인복무규율", "폐지"],
                focus_terms=focus_terms,
                fallback="군인복무규율 폐지 근거는 현재 검색된 자료에서 충분히 특정되지 않았습니다.",
            ),
            self._transition_line(
                "기본법 제정 취지",
                reason_hits,
                ["기본권"],
                focus_terms=focus_terms,
                fallback="기본법 제정 이유 자료가 제한적으로만 확인됩니다.",
            ),
        ]
        reorganization_points = [
            self._transition_line(
                "권리구제 축",
                law_hits + compare_hits,
                ["군인고충심사위원회"],
                focus_terms=focus_terms,
                fallback="고충 처리나 권리구제 조항은 현재 근거에서 충분히 확인되지 않았습니다.",
            ),
            self._transition_line(
                "군인사법 연계 단서",
                compare_hits + law_hits,
                ["군인사법"],
                focus_terms=focus_terms,
                fallback="군인사법과의 조문 이동·삭제 단서는 현재 근거에서 제한적으로만 확인됩니다.",
            ),
        ]
        discipline_points = []
        if has_direct_discipline_law:
            discipline_points.append(
                self._transition_line(
                    "징계 절차 근거",
                    all_hits,
                    ["징계"],
                    focus_terms=focus_terms,
                    fallback="징계 절차 근거는 아래 근거 카드에서 원문을 확인해야 합니다.",
                )
            )
        else:
            discipline_points.extend(
                [
                    "- 징계 종류·절차: 현재 코퍼스에는 「군인사법」 본문이나 「군인 징계령」 계열 본문이 들어 있지 않습니다.",
                    "- 답변 범위: 따라서 이 앱은 복무규율 폐지와 기본법 체계 재편 단서는 설명할 수 있지만, 징계의 종류·절차 담당 법령을 확정해서 말하지 않습니다.",
                    "- 후속 보완: 해당 질문을 완결하려면 「군인사법」, 「군인사법 시행령」, 「군인 징계령」 계열 자료를 별도 코퍼스로 추가해야 합니다.",
                ]
            )

        practical_points = [
            "기본법은 복무·기본권·고충 처리의 현재 축을 확인하는 출발점으로 보고, 폐지된 군인복무규율은 연혁·배경 근거로 구분해 보는 것이 안전합니다.",
            (
                "징계 종류·절차·양정은 군인사법, 군인 징계령, 군인 징계령 시행규칙 근거를 나누어 확인하는 것이 안전합니다."
                if has_direct_discipline_law
                else "징계 종류·절차·양정은 현재 코퍼스의 신고자 보호나 고충 처리 조항만으로 판단하지 말고, 별도 징계·인사 법령 원문을 추가 확인해야 합니다."
            ),
        ]

        return (
            "## 답변 개요\n"
            f"### 핵심 결론\n{conclusion}\n\n"
            "## 세부 정리\n"
            "### 2016년 체계 전환\n"
            + "\n".join(transition_points)
            + "\n\n"
            + "### 복무·군기·권리구제의 재편\n"
            + "\n".join(reorganization_points)
            + "\n\n"
            + "### 징계 종류·절차 담당 법령\n"
            + "\n".join(discipline_points)
            + "\n\n"
            + "### 실무적으로 보면\n"
            + "\n".join(f"- {point}" for point in practical_points)
            + "\n\n"
            + "## 근거 안내\n"
            + "### 확인 방법\n- 아래 근거 카드에서 폐지 부칙, 기본법 제정이유, 고충 처리 조항, 군인사법 연계 단서를 나누어 확인해 주세요."
        )

    def _discipline_overview_line(self, label: str, hit: SearchHit, *, focus_terms: list[str]) -> str:
        summary = self._summary_for_hit(hit, focus_terms=focus_terms)
        if not summary:
            summary = f"{self._article_ref(hit)}에서 관련 기준을 확인할 수 있습니다."
        return f"- {label}: {summary}"

    def _build_discipline_overview_answer(self, question: str, evidence: list[SearchHit]) -> str:
        focus_terms = self._question_focus_terms(question)
        law_hits = self._source_hits(evidence, "law_text")
        history_hits = self._source_hits(evidence, "history_note")
        reason_hits = self._source_hits(evidence, "revision_reason")

        used_ids: set[str] = set()
        current_points: list[str] = []
        current_labels: list[str] = []
        current_specs = [
            ("신고의무", ["신고의무", "즉시 신고", "신고하여야", "보고하거나", "군인권보호관"]),
            ("신고자 보호", ["신고자 보호", "불이익조치", "징계조치", "원상회복", "비밀을 보장"]),
            ("사적 제재 금지", ["사적 제재", "구타", "폭언", "가혹행위", "집단 따돌림", "성폭력"]),
        ]
        current_candidates = law_hits + reason_hits
        for label, terms in current_specs:
            hit = self._first_hit_with_terms(current_candidates, terms, exclude_ids=used_ids)
            if hit is None:
                continue
            used_ids.add(hit.chunk.id)
            current_labels.append(label)
            current_points.append(self._discipline_overview_line(label, hit, focus_terms=focus_terms))

        history_points: list[str] = []
        history_labels: list[str] = []
        history_specs = [
            ("과거 징계혐의자 관리", ["징계혐의자", "휴가의 보류", "외출ㆍ외박", "휴가를 일시 보류"]),
            ("군기·인권침해 정비 흐름", ["군기강", "인권침해", "구시대적 관행", "기본권보장"]),
        ]
        for label, terms in history_specs:
            hit = self._first_hit_with_terms(history_hits + reason_hits, terms, exclude_ids=used_ids)
            if hit is None:
                continue
            used_ids.add(hit.chunk.id)
            history_labels.append(label)
            history_points.append(self._discipline_overview_line(label, hit, focus_terms=focus_terms))

        if not current_points and law_hits:
            for hit in law_hits[:2]:
                current_points.append(
                    f"- {self._article_ref(hit)}: {self._summary_for_hit(hit, focus_terms=focus_terms)}"
                )
                current_labels.append(self._article_ref(hit))

        if not history_points and history_hits:
            history_points.append(
                f"- 연혁 연결: {self._summary_for_hit(history_hits[0], focus_terms=focus_terms)}"
            )

        conclusion_labels = current_labels[:]
        if history_labels and "과거 징계혐의자 관리" in history_labels and "과거 징계혐의자 관리" not in conclusion_labels:
            conclusion_labels.append("과거 징계혐의자 관리")
        joined_labels = self._join_labels(conclusion_labels[:4])
        conclusion = (
            f"공개 법령 코퍼스 기준으로 징계 관련해서 먼저 확인되는 축은 {joined_labels}입니다. "
            "다만 징계 사유·절차·양정 전체는 현재 코퍼스만으로는 완결되지 않습니다."
            if joined_labels
            else "현재 자료 기준으로 징계 관련 공개 법령 근거를 충분히 모으지 못했습니다."
        )

        practical_points = []
        if any("신고" in label for label in current_labels):
            practical_points.append("신고 사안을 검토할 때는 신고의무, 신고자 보호, 불이익조치 금지를 한 묶음으로 같이 확인하는 편이 안전합니다.")
        if history_points:
            practical_points.append("군인복무규율 연혁은 배경 설명용으로 보고, 실제 적용 판단은 현행 기본법과 현재 운용 규정을 우선 확인해야 합니다.")
        practical_points.append("징계 사유, 절차, 양정까지 판단하려면 현재 공개 코퍼스 밖의 별도 징계 규정과 소속 부대 지침을 함께 확인해야 합니다.")

        return (
            "## 답변 개요\n"
            f"### 핵심 결론\n{conclusion}\n\n"
            "## 세부 정리\n"
            "### 주요 규정\n"
            + "\n".join(current_points[:4] or ["- 현재 자료 기준으로 관련 현행 조문을 충분히 찾지 못했습니다."])
            + "\n\n"
            + "### 연혁으로 같이 보이는 변화\n"
            + "\n".join(history_points[:3] or ["- 현재 자료 기준으로 징계 관련 연혁 근거는 제한적입니다."])
            + "\n\n"
            + "### 실무 참고\n"
            + "\n".join(f"- {point}" for point in practical_points[:3])
            + "\n\n"
            + "## 근거 안내\n"
            + "### 확인 방법\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _build_search_answer(self, question: str, evidence: list[SearchHit]) -> str:
        if self._is_service_regulation_transition_question(question):
            return self._build_service_regulation_transition_answer(question, evidence)

        law_hits = self._source_hits(evidence, "law_text")
        history_hits = self._source_hits(evidence, "history_note")
        reason_hits = self._source_hits(evidence, "revision_reason")
        focus_terms = self._question_focus_terms(question)
        leave_focus_terms = {"휴가", "연가", "청원휴가", "특별휴가", "정기휴가", "육아시간", "돌봄휴가", "배우자 출산휴가"}
        discipline_focus_terms = {
            "징계",
            "징계조치",
            "징계혐의자",
            "신고",
            "신고자",
            "신고자 보호",
            "비밀보장",
            "신고자에 대한 비밀보장",
            "불이익조치",
            "가혹행위",
        }
        leave_focused = any(term in focus_terms for term in leave_focus_terms)
        discipline_focused = any(term in focus_terms for term in discipline_focus_terms)

        if discipline_focused and self._is_broad_overview_request(question):
            return self._build_discipline_overview_answer(question, evidence)

        level_order = {"법률": 0, "시행령": 1, "시행규칙": 2}
        explicit_article_refs = self._explicit_article_refs(question)

        def focus_overlap(hit: SearchHit) -> int:
            haystack = self._normalize_text(" ".join([self._article_ref(hit), self._hit_text(hit), hit.chunk.law_name]))
            return sum(haystack.count(term) for term in focus_terms if term)

        def explicit_article_priority(hit: SearchHit) -> int:
            article_ref = self._article_ref(hit).replace(" ", "")
            if not explicit_article_refs:
                return 1
            return 0 if any(ref in article_ref for ref in explicit_article_refs) else 1

        candidate_hits = law_hits
        if discipline_focused:
            candidate_hits = [hit for hit in evidence if hit.chunk.source_type in {"law_text", "history_note"}]
            focused_hits = [
                hit
                for hit in candidate_hits
                if focus_overlap(hit) > 0 or explicit_article_priority(hit) == 0
            ]
            if focused_hits:
                candidate_hits = focused_hits

        ordered_hits = sorted(
            candidate_hits,
            key=lambda hit: (
                explicit_article_priority(hit),
                0 if leave_focused and "제8조" in self._article_ref(hit) else 1,
                0 if leave_focused and "제2조의6" in self._article_ref(hit) else 1,
                0 if leave_focused and "제2조" in self._article_ref(hit) else 1,
                0 if discipline_focused and hit.chunk.source_type == "law_text" else 1,
                -focus_overlap(hit),
                level_order.get(self._display_law_level(hit), 9),
            ),
        )
        main_source_hits = ordered_hits
        if explicit_article_refs:
            explicit_hits = [hit for hit in ordered_hits if explicit_article_priority(hit) == 0]
            title_focus_terms = [
                term
                for term in focus_terms
                if len(term) >= 4 and term in question and term not in {"신고자", "가혹행위"}
            ]
            title_focus_hits = [
                hit
                for hit in ordered_hits
                if hit.chunk.source_type == "law_text"
                and any(term in self._article_ref(hit) for term in title_focus_terms)
            ]
            focused_explicit_hits = list({hit.chunk.id: hit for hit in [*explicit_hits, *title_focus_hits]}.values())
            if focused_explicit_hits:
                main_source_hits = focused_explicit_hits

        main_lines = [
            self._line_for_hit(
                hit,
                focus_terms=focus_terms,
                max_points=1 if discipline_focused else 2,
            )
            for hit in main_source_hits[:3]
        ]
        if not main_lines:
            main_lines = ["- 현재 자료 기준으로 관련 현행 조문을 충분히 찾지 못했습니다."]

        apply_points: list[str] = []
        if any("제2조의6" in self._article_ref(hit) or "5분의 1" in self._hit_text(hit) for hit in law_hits):
            apply_points.append("휴가 확인 범위는 부대 현재 병력의 5분의 1 이내 기준이 시행령에 제시됩니다.")
        if any("육아시간" in self._hit_text(hit) or "돌봄휴가" in self._hit_text(hit) for hit in law_hits):
            apply_points.append("돌봄·육아 관련 질문은 휴가 조문뿐 아니라 시행령 제2조 계열 세부 기준까지 함께 보는 것이 안전합니다.")
        if reason_hits and ("휴가" in focus_terms or "육아시간" in focus_terms or "돌봄휴가" in focus_terms):
            reason_points = self._reason_summary_points(reason_hits, focus_terms)
            if reason_points:
                apply_points.append(f"최근 개정 흐름으로는 {reason_points[0]}")
        if discipline_focused:
            if history_hits:
                apply_points.append("공개 코퍼스에서는 징계 절차 전체보다 신고자 보호, 사적 제재 금지, 징계혐의자 관리처럼 연결되는 기준이 먼저 포착됩니다.")
            apply_points.append("징계 관련 질문은 징계조치 자체만이 아니라 신고의무, 신고자 보호, 가혹행위 금지처럼 연결되는 조문을 함께 보는 것이 안전합니다.")
        if not apply_points:
            apply_points.append("현행 조문과 관련 개정 자료를 함께 대조해 적용 범위와 제한 사유를 직접 확인하는 방식이 적절합니다.")

        return (
            "## 답변 개요\n"
            f"### 핵심 결론\n{self._conclusion_from_search(question, law_hits)}\n\n"
            "## 세부 정리\n"
            f"### 주요 규정\n" + "\n".join(main_lines) + "\n\n"
            f"### 실무 참고\n" + "\n".join(f"- {point}" if not point.startswith("- ") else point for point in apply_points[:3]) + "\n\n"
            "## 근거 안내\n"
            "### 확인 방법\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _build_explain_answer(self, question: str, evidence: list[SearchHit]) -> str:
        if self._is_service_regulation_transition_question(question):
            return self._build_service_regulation_transition_answer(question, evidence)

        reason_hits = self._source_hits(evidence, "revision_reason")
        compare_hits = self._source_hits(evidence, "old_new_comparison")
        history_hits = self._source_hits(evidence, "history_note")
        law_hits = self._source_hits(evidence, "law_text")
        focus_terms = self._question_focus_terms(question)
        if self._is_timeline_question(question):
            return self._build_timeline_answer(question, evidence)

        reason_points = self._reason_summary_points(reason_hits, focus_terms)
        history_points = self._history_link_points(history_hits, focus_terms)
        combined_law_text = " ".join(self._normalize_text(self._hit_text(hit)) for hit in law_hits + history_hits)
        if any(term in focus_terms for term in ["육아", "돌봄", "출산", "자녀돌봄휴가", "배우자 출산휴가"]):
            childcare_reason_points: list[str] = []
            if "배우자가 출산" in combined_law_text or "출산휴가" in combined_law_text:
                childcare_reason_points.append("배우자 출산과 가족 돌봄 상황에 맞는 청원휴가 기준을 더 세밀하게 정비하려는 흐름이 보입니다.")
            if "육아시간" in combined_law_text:
                childcare_reason_points.append("단순 일수 보장보다 육아시간 같은 시간 단위 지원까지 넓히려는 방향이 확인됩니다.")
            for point in reversed(childcare_reason_points):
                if point not in reason_points:
                    reason_points.insert(0, point)
        for point in history_points:
            if point not in reason_points:
                reason_points.append(point)
        if not reason_points:
            reason_points = ["현재 자료 기준으로 개정 이유를 직접 적시한 자료가 충분하지 않습니다."]

        change_points = self._change_points(compare_hits, law_hits, focus_terms)
        for point in history_points:
            if point not in change_points:
                change_points.append(point)
        if any(term in focus_terms for term in ["육아", "돌봄", "출산", "자녀돌봄휴가", "배우자 출산휴가"]):
            childcare_change_points: list[str] = []
            if "배우자가 출산" in combined_law_text or "출산휴가" in combined_law_text:
                childcare_change_points.append("청원휴가 체계 안에서 배우자 출산휴가 기준이 구체화된 흐름이 확인됩니다.")
            if "직계가족" in combined_law_text or "간호" in combined_law_text or "자녀돌봄" in combined_law_text:
                childcare_change_points.append("본인 치료뿐 아니라 가족 간호·돌봄 사유를 휴가 사유로 넓혀 가는 방향이 보입니다.")
            if "육아시간" in combined_law_text:
                childcare_change_points.append("최근 기준으로 갈수록 단순 휴가일수뿐 아니라 육아시간처럼 시간 단위 지원까지 함께 보완되는 흐름이 보입니다.")
            for point in reversed(childcare_change_points):
                if point not in change_points:
                    change_points.insert(0, point)
        if not change_points:
            change_points = ["현재 자료 기준으로 구체적인 변경 내용을 충분히 추리기 어렵습니다."]

        interpretation_points: list[str] = []
        combined = " ".join(self._normalize_text(self._hit_text(hit)) for hit in reason_hits)
        if "출산" in combined:
            interpretation_points.append("출산·돌봄 지원을 확대하기 위한 개정으로 해석할 수 있습니다.")
        if "일·가정 양립" in combined or "양립" in combined:
            interpretation_points.append("군 복무와 가정 돌봄을 병행할 수 있도록 대상과 운영 기준을 넓힌 개정으로 볼 수 있습니다.")
        if "근무 여건" in combined or "복무 여건" in combined:
            interpretation_points.append("근무 여건과 복무 환경을 보완하려는 취지가 함께 드러납니다.")
        if "휴직" in "".join(focus_terms) and not any(keyword in combined_law_text for keyword in ["육아휴직", "휴직", "복직"]):
            interpretation_points.append("현재 확보한 공개 코퍼스에서는 육아 관련 휴가 근거가 더 뚜렷하고, 별도 휴직 기준은 군인사법 등 연계 인사 규정을 함께 확인해야 합니다.")
        for point in history_points:
            if point not in interpretation_points:
                interpretation_points.append(point)
        if not interpretation_points:
            interpretation_points.append("현재 자료 기준으로는 제도 범위와 운영 기준을 보완하려는 방향으로 해석됩니다.")

        return (
            "## 답변 개요\n"
            f"### 핵심 결론\n{self._conclusion_from_explain(question, reason_hits, history_hits, law_hits)}\n\n"
            "## 세부 정리\n"
            f"### 주요 개정 이유\n" + "\n".join(f"- {point}" for point in reason_points[:3]) + "\n\n"
            f"### 실제 제도 변화\n" + "\n".join(f"- {point}" for point in change_points[:3]) + "\n\n"
            f"### 해석 시사점\n" + "\n".join(f"- {point}" for point in interpretation_points[:3]) + "\n\n"
            "## 근거 안내\n"
            "### 확인 방법\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _build_practical_answer(self, question: str, evidence: list[SearchHit]) -> str:
        law_hits = self._source_hits(evidence, "law_text")
        focus_terms = self._question_focus_terms(question)
        discipline_or_procedure = any(
            term in focus_terms
            for term in [
                "징계",
                "징계조치",
                "징계혐의자",
                "징계 절차",
                "징계절차",
                "징계의 종류",
                "징계 종류",
                "징계위원회",
                "군인사법",
                "군인 징계령",
            ]
        )

        practical_points: list[str] = []
        combined_law_text = " ".join(self._normalize_text(self._hit_text(hit)) for hit in law_hits)
        if "휴가" in "".join(focus_terms) and all(term in combined_law_text for term in ["연가", "공가", "청원휴가", "특별휴가", "정기휴가"]):
            practical_points.append("시행령 제9조 기준 휴가 종류는 연가, 공가, 청원휴가, 특별휴가 및 정기휴가로 구분됩니다.")
        if law_hits and discipline_or_procedure:
            practical_points.append("징계 관련 실무 판단은 현재 확보 근거에서 확인되는 신고자 보호·고충 처리 축과, 별도 징계 절차 법령의 확인 필요성을 구분해 보는 것이 안전합니다.")
        elif law_hits:
            practical_points.append("먼저 법률 조문에서 보장 범위와 제한 사유를 확인하고, 이어서 시행령에서 종류·일수·시간 범위를 확인하는 순서가 적절합니다.")
        if any("제8조" in self._article_ref(hit) for hit in law_hits):
            practical_points.append("휴가 제한이나 보류 사유는 작전상황, 교육훈련, 징계·형사 절차, 부대병력 유지 필요 여부까지 함께 점검해야 합니다.")
        if any("5분의 1" in self._hit_text(hit) or "확인 범위" in self._hit_text(hit) for hit in law_hits):
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
            "## 답변 개요\n"
            f"### 핵심 결론\n{self._conclusion_from_practical(law_hits)}\n\n"
            "## 실무 정리\n"
            f"### 실무적으로 보면\n" + "\n".join(f"- {point}" for point in practical_points[:4]) + "\n\n"
            f"### 주의사항\n" + "\n".join(f"- {point}" for point in caution_points) + "\n\n"
            "## 근거 안내\n"
            "### 확인 방법\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _build_hybrid_answer(self, question: str, evidence: list[SearchHit]) -> str:
        if self._is_service_regulation_transition_question(question):
            return self._build_service_regulation_transition_answer(question, evidence)

        reason_hits = self._source_hits(evidence, "revision_reason")
        history_hits = self._source_hits(evidence, "history_note")
        law_hits = self._source_hits(evidence, "law_text")
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
            "## 답변 개요\n"
            f"### 핵심 결론\n{self._conclusion_from_explain(question, reason_hits, history_hits, law_hits)}\n\n"
            "## 세부 정리\n"
            f"### 주요 개정 이유\n" + "\n".join(f"- {point}" for point in reason_points[:3]) + "\n\n"
            f"### 실무적으로 보면\n{practical_body}\n\n"
            "## 근거 안내\n"
            "### 확인 방법\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _build_timeline_answer(self, question: str, evidence: list[SearchHit]) -> str:
        focus_terms = self._question_focus_terms(question)
        relevant_hits = sorted(
            evidence,
            key=lambda hit: (
                hit.chunk.effective_date or hit.chunk.promulgation_date or "9999-12-31",
                0 if hit.chunk.law_name == "군인복무규율" else 1,
                0 if hit.chunk.source_type == "revision_reason" else 1,
            ),
        )

        chronology_blocks: list[str] = []
        seen_scopes: set[tuple[str, str, str]] = set()
        for hit in relevant_hits:
            summary_points = self._extract_points(self._hit_text(hit), focus_terms=focus_terms, limit=1)
            if not summary_points:
                continue
            key = (hit.chunk.law_name, self._scope_label(hit), hit.chunk.source_type)
            if key in seen_scopes:
                continue
            seen_scopes.add(key)
            chronology_blocks.append(self._timeline_entry_markdown(hit, focus_terms=focus_terms))
            if len(chronology_blocks) >= 4:
                break
        if not chronology_blocks:
            chronology_blocks = ["- 현재 자료 기준으로 시기별 변화를 직접 추적할 근거가 충분하지 않습니다."]

        current_lines: list[str] = []
        history_hits = self._source_hits(evidence, "history_note")
        law_hits = [
            hit
            for hit in self._source_hits(evidence, "law_text")
            if hit.chunk.law_name == "군인의 지위 및 복무에 관한 기본법"
        ]
        reason_hits = [
            hit
            for hit in self._source_hits(evidence, "revision_reason")
            if hit.chunk.law_name == "군인의 지위 및 복무에 관한 기본법"
        ]

        history_combined = " ".join(self._normalize_text(self._hit_text(hit)) for hit in history_hits)
        if "징계" in focus_terms and "군인복무규율" in history_combined and law_hits:
            current_lines.append(
                "#### 연결 요약\n- 군인복무규율에서 보이던 징계혐의자·휴가 보류 중심 기준이 현재는 기본법상 신고자 보호, "
                "가혹행위 대응, 상담 지원 조문으로 재구성되어 권리보호 기준이 법률 단계에서 직접 드러납니다."
            )
        elif "군인복무규율" in history_combined and law_hits:
            current_lines.append(
                "#### 연결 요약\n- 군인복무규율의 복무관리 기준이 현재 기본법 조문과 하위 법령 체계로 재배치되면서, "
                "핵심 원칙은 법률에 두고 세부 운영은 하위 법령으로 나누는 구조가 더 분명해졌습니다."
            )

        prioritized_current_hits = law_hits + reason_hits
        for hit in prioritized_current_hits:
            summary_points = self._extract_points(self._hit_text(hit), focus_terms=focus_terms, limit=1)
            if not summary_points:
                continue
            line = self._current_link_markdown(hit, focus_terms=focus_terms)
            if line not in current_lines:
                current_lines.append(line)
            if len(current_lines) >= 3:
                break
        if not current_lines:
            current_lines.append("- 현재 자료 기준으로 기본법 단계에서 직접 확인되는 조문은 제한적입니다.")

        return (
            "## 답변 개요\n"
            f"### 핵심 결론\n{self._conclusion_from_explain(question, reason_hits, history_hits, law_hits)}\n\n"
            "## 연혁 정리\n"
            f"### 시기별 변화\n" + "\n\n".join(chronology_blocks) + "\n\n"
            "## 현재 체계\n"
            f"### 현재 체계와 연결\n" + "\n\n".join(current_lines[:3]) + "\n\n"
            "## 근거 안내\n"
            "### 확인 방법\n- 아래 근거 카드와 원문 링크를 함께 확인해 주세요."
        )

    def _fallback_answer(self, question: str, intent: str, evidence: list[SearchHit]) -> str:
        if intent == "search":
            return self._build_search_answer(question, evidence)
        if intent == "explain_change":
            return self._build_explain_answer(question, evidence)
        if intent == "practical":
            return self._build_practical_answer(question, evidence)
        return self._build_hybrid_answer(question, evidence)

    @staticmethod
    def _normalize_answer_line(line: str) -> str:
        normalized = line.strip()
        normalized = re.sub(r"^[-*•]\s*", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.rstrip(".。").strip()

    def _postprocess_answer_markdown(self, markdown: str) -> str:
        seen_bullets: set[str] = set()
        output_lines: list[str] = []
        previous_blank = False

        for raw_line in (markdown or "").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                if not previous_blank:
                    output_lines.append("")
                previous_blank = True
                continue

            previous_blank = False
            if stripped.startswith(("- ", "* ", "• ")):
                key = self._normalize_answer_line(stripped)
                if key and key in seen_bullets:
                    continue
                seen_bullets.add(key)

            output_lines.append(repair_common_text_artifacts(line))

        return repair_common_text_artifacts("\n".join(output_lines)).strip()

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
        diagnostics: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            text=self._postprocess_answer_markdown(self._fallback_answer(question, intent, evidence)),
            backend="retrieval_fallback",
            notice=notice,
            quota_snapshot=quota_snapshot or self.usage_tracker.snapshot(),
            diagnostics=diagnostics or {},
        )

    def _quota_block_result(
        self,
        quota_snapshot: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            text=QUOTA_BLOCK_MESSAGE,
            backend="quota_blocked",
            notice="",
            quota_snapshot=quota_snapshot or self.usage_tracker.snapshot(),
            diagnostics=diagnostics or {},
        )

    def _no_evidence_answer(self, intent: str) -> str:
        if intent == "explain_change":
            return (
                "## 답변 개요\n"
                "### 핵심 결론\n현재 자료 기준으로 개정 이유나 연혁을 충분히 찾지 못했습니다.\n\n"
                "## 세부 정리\n"
                "### 주요 개정 이유\n- 관련 개정이유 또는 신구 비교 자료가 부족합니다.\n\n"
                "### 실제 제도 변화\n- 구체적인 변경 내용을 확인할 근거가 충분하지 않습니다.\n\n"
                "### 해석 시사점\n- 질문을 더 구체화하거나 자료 유형을 조정해 다시 검색해 주세요.\n\n"
                "## 근거 안내\n"
                "### 확인 방법\n- 현재 제시할 근거가 없습니다."
            )
        if intent == "practical":
            return (
                "## 답변 개요\n"
                "### 핵심 결론\n현재 자료 기준으로 실무 참고에 필요한 근거가 부족합니다.\n\n"
                "## 실무 정리\n"
                "### 실무적으로 보면\n- 질문 범위를 더 좁혀 다시 검색하는 것이 좋습니다.\n\n"
                "### 주의사항\n- 관련 조문과 시행령을 확인할 수 있는 공개 자료가 더 필요합니다.\n\n"
                "## 근거 안내\n"
                "### 확인 방법\n- 현재 제시할 근거가 없습니다."
            )
        return (
            "## 답변 개요\n"
            "### 핵심 결론\n현재 자료 기준으로 관련 현행 규정을 충분히 찾지 못했습니다.\n\n"
            "## 세부 정리\n"
            "### 주요 규정\n- 관련 조문이 충분히 검색되지 않았습니다.\n\n"
            "### 실무 참고\n- 질문을 더 구체화하거나 자료 유형을 조정해 다시 검색해 주세요.\n\n"
            "## 근거 안내\n"
            "### 확인 방법\n- 현재 제시할 근거가 없습니다."
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
            if not self._is_structured_answer(question, intent, final_text):
                return self._fallback_result(
                    question=question,
                    intent=intent,
                    evidence=evidence,
                    notice="Gemini 응답이 지정한 형식을 충분히 따르지 않아 근거 기반 요약으로 전환했습니다.",
                    quota_snapshot=updated_snapshot,
                )

            notice = QUOTA_BLOCK_MESSAGE if not updated_snapshot["can_generate"] else ""
            return GeneratedAnswer(
                text=self._postprocess_answer_markdown(final_text),
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
