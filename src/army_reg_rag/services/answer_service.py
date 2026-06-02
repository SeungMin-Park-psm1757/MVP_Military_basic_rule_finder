from __future__ import annotations

from collections import OrderedDict
import re

from army_reg_rag.config import Settings
from army_reg_rag.domain.models import AnswerBundle, SearchHit
from army_reg_rag.llm.gemini_client import GeminiAnswerClient
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.retrieval.router import decide_route

HISTORY_REQUEST_KEYWORDS = {
    "과거",
    "이전",
    "예전",
    "종전",
    "연혁",
    "변천",
    "변천사",
    "유래",
    "발전",
    "흐름",
    "이어졌",
    "넘어오",
    "넘어오면서",
    "체계로 오",
    "군인복무규율",
}

TIMELINE_REQUEST_KEYWORDS = {
    "연혁",
    "변천",
    "변천사",
    "흐름",
    "어떻게 이어졌",
    "어떻게 바뀌",
    "어떻게 달라졌",
    "넘어오면서",
}

LAW_NAME_ALIASES = {
    "군인의 지위 및 복무에 관한 기본법": {
        "군인의 지위 및 복무에 관한 기본법",
        "군인기본법",
        "기본법",
    },
    "군인의 지위 및 복무에 관한 기본법 시행령": {
        "군인의 지위 및 복무에 관한 기본법 시행령",
        "기본법 시행령",
        "시행령",
    },
    "군인의 지위 및 복무에 관한 기본법 시행규칙": {
        "군인의 지위 및 복무에 관한 기본법 시행규칙",
        "기본법 시행규칙",
        "시행규칙",
    },
    "군인복무규율": {
        "군인복무규율",
        "복무규율",
    },
    "군인사법": {
        "군인사법",
        "인사법",
    },
    "군인 징계령": {
        "군인 징계령",
        "군인징계령",
        "징계령",
    },
    "군인 징계령 시행규칙": {
        "군인 징계령 시행규칙",
        "군인징계령 시행규칙",
        "징계령 시행규칙",
    },
}

RELATED_LAW_MAP = {
    "군인의 지위 및 복무에 관한 기본법": [
        "군인복무규율",
    ],
    "군인의 지위 및 복무에 관한 기본법 시행령": [
        "군인의 지위 및 복무에 관한 기본법",
        "군인복무규율",
    ],
    "군인의 지위 및 복무에 관한 기본법 시행규칙": [
        "군인의 지위 및 복무에 관한 기본법 시행령",
        "군인의 지위 및 복무에 관한 기본법",
        "군인복무규율",
    ],
    "군인복무규율": [
        "군인의 지위 및 복무에 관한 기본법",
        "군인사법",
        "군인 징계령",
        "군인 징계령 시행규칙",
    ],
    "군인사법": [
        "군인의 지위 및 복무에 관한 기본법",
        "군인 징계령",
        "군인 징계령 시행규칙",
    ],
    "군인 징계령": [
        "군인사법",
        "군인 징계령 시행규칙",
    ],
    "군인 징계령 시행규칙": [
        "군인 징계령",
        "군인사법",
    ],
}

RELATED_LAW_FAMILY = set(LAW_NAME_ALIASES)
for related_names in RELATED_LAW_MAP.values():
    RELATED_LAW_FAMILY.update(related_names)

TOPIC_EXPANSION_MAP = {
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
    "신고": ["신고", "신고자", "신고의무", "신고자 보호", "불이익조치", "가혹행위"],
    "휴가": ["휴가", "외출", "외박", "연가", "청원휴가", "특별휴가", "정기휴가", "휴가 보류"],
    "육아": ["육아", "육아시간", "육아휴직", "돌봄", "자녀", "자녀돌봄휴가", "배우자 출산휴가", "모성보호시간", "임신검진", "청원휴가"],
    "돌봄": ["돌봄", "자녀돌봄휴가", "육아", "육아시간", "배우자 출산휴가", "임신검진", "모성보호시간"],
    "출산": ["출산", "배우자 출산휴가", "임신검진", "모성보호시간", "청원휴가", "육아", "돌봄"],
    "휴직": ["휴직", "육아휴직", "육아", "돌봄", "복직"],
    "고충": ["고충", "고충 처리", "고충심사", "군인고충심사위원회", "재심청구", "군인사법", "중앙 군인사소청심사위원회"],
    "권리구제": ["권리구제", "고충", "고충 처리", "고충심사", "군인고충심사위원회", "재심청구", "군인사법", "중앙 군인사소청심사위원회"],
    "기본권": ["기본권", "인권", "가혹행위", "신고자 보호"],
}

QUESTION_STOPWORDS = {
    "무엇",
    "관련",
    "사항",
    "규정",
    "내용",
    "내용을",
    "정리",
    "정리해서",
    "정리해줘",
    "나열",
    "나열해줘",
    "요약",
    "요약해줘",
    "설명",
    "설명해줘",
    "알려줘",
    "좀",
    "한번",
    "위주",
    "위주로",
    "찾아줘",
    "현행",
    "기준",
    "내용",
    "바뀌었어",
    "바뀌었지",
    "실무",
    "어떻게",
    "해야",
    "해줘",
    "현재",
    "조문",
    "무슨",
    "참고",
    "주의",
    "과거",
    "연혁",
    "예전",
    "이전",
    "종전",
    "변천",
    "변천사",
    "흐름",
}

GENERIC_LAW_TOKENS = {
    "군인",
    "복무",
    "지위",
    "기본법",
    "시행령",
    "시행규칙",
    "복무규율",
    "군인복무규율",
    "법령",
}

DISCIPLINE_TOPIC_TERMS = {
    "징계",
    "징계조치",
    "징계혐의자",
    "징계 절차",
    "징계절차",
    "징계의 종류",
    "징계 종류",
    "징계위원회",
    "신고의무",
    "신고자 보호",
    "불이익조치",
    "사적 제재",
    "가혹행위",
    "구타",
    "폭언",
    "군인사법",
    "군인 징계령",
}

DISCIPLINE_PROCEDURE_TERMS = {
    "징계 절차",
    "징계절차",
    "징계의 종류",
    "징계 종류",
    "징계위원회",
    "군인사법",
    "군인 징계령",
    "군인징계령",
    "군인 징계령 시행규칙",
}

SERVICE_REGULATION_TRANSITION_TERMS = {
    "군인복무규율",
    "폐지",
    "재편",
    "체계",
    "기본법",
}

DISCIPLINE_OVERVIEW_KEYWORDS = {
    "정리",
    "나열",
    "요약",
    "위주",
    "사항",
    "내용",
    "전반",
    "전체",
    "무엇",
    "뭐가",
}


class AnswerService:
    def __init__(
        self,
        settings: Settings,
        *,
        store: ChromaStore | None = None,
        client: GeminiAnswerClient | None = None,
    ):
        self.settings = settings
        self.store = store or ChromaStore(settings)
        self.client = client or GeminiAnswerClient(settings)

    @staticmethod
    def _intent_alias(intent: str) -> str:
        return {
            "search": "current_rule",
            "explain_change": "explain_change",
            "practical": "practical_reference",
            "hybrid": "mixed_complex",
        }.get(intent, intent)

    @staticmethod
    def _evidence_slots(evidence: list[SearchHit]) -> dict[str, list[str]]:
        slots = {
            "current_articles": [],
            "revision_reasons": [],
            "history_docs": [],
            "compare_docs": [],
        }
        source_to_slot = {
            "law_text": "current_articles",
            "revision_reason": "revision_reasons",
            "history_note": "history_docs",
            "old_new_comparison": "compare_docs",
        }
        for hit in evidence:
            slot_name = source_to_slot.get(hit.chunk.source_type)
            if not slot_name:
                continue
            label = " ".join(
                part
                for part in [hit.chunk.law_name, hit.chunk.article_no, hit.chunk.article_title]
                if part
            ).strip()
            if label:
                slots[slot_name].append(label)
        return slots

    def _question_tokens(self, question: str) -> list[str]:
        return [token.strip() for token in re.findall(r"[가-힣A-Za-z0-9]+", question) if token.strip()]

    def _mentioned_laws(self, question: str, law_name_filter: str | None = None) -> list[str]:
        mentioned: list[str] = []
        if law_name_filter and law_name_filter != "전체":
            mentioned.append(law_name_filter)

        for canonical_name, aliases in LAW_NAME_ALIASES.items():
            if canonical_name in question or any(alias in question for alias in aliases):
                if canonical_name not in mentioned:
                    mentioned.append(canonical_name)
        return mentioned

    def _primary_target_laws(self, question: str, law_name_filter: str | None = None) -> list[str]:
        if law_name_filter and law_name_filter != "전체":
            return [law_name_filter]
        mentioned = self._mentioned_laws(question)
        return mentioned[:1]

    def _question_terms(self, question: str) -> list[str]:
        terms: list[str] = []
        for token in self._question_tokens(question):
            if len(token) < 2:
                continue
            if token in QUESTION_STOPWORDS or token in GENERIC_LAW_TOKENS:
                continue
            if token not in terms:
                terms.append(token)
        return terms[:6]

    def _topic_terms(self, question: str) -> list[str]:
        terms = self._question_terms(question)
        expanded: list[str] = []
        for topic_key, synonyms in TOPIC_EXPANSION_MAP.items():
            topic_family = [topic_key, *synonyms]
            if any(keyword in question for keyword in topic_family):
                for keyword in topic_family:
                    if keyword not in expanded:
                        expanded.append(keyword)
        for term in terms:
            if term not in expanded:
                expanded.append(term)
            for topic_key, synonyms in TOPIC_EXPANSION_MAP.items():
                if term == topic_key or term in synonyms:
                    for synonym in synonyms:
                        if synonym not in expanded:
                            expanded.append(synonym)
        return expanded[:12]

    def _is_history_request(self, question: str, intent: str) -> bool:
        return intent in {"explain_change", "hybrid"} or any(keyword in question for keyword in HISTORY_REQUEST_KEYWORDS)

    def _is_timeline_request(self, question: str) -> bool:
        return any(keyword in question for keyword in TIMELINE_REQUEST_KEYWORDS)

    def _is_discipline_request(self, question: str) -> bool:
        haystack = " ".join(self._topic_terms(question)) + " " + question
        return any(term in haystack for term in DISCIPLINE_TOPIC_TERMS)

    def _is_discipline_overview_request(self, question: str) -> bool:
        if not self._is_discipline_request(question):
            return False
        if re.search(r"제\s*\d+조", question):
            return False
        return any(keyword in question for keyword in DISCIPLINE_OVERVIEW_KEYWORDS)

    def _is_service_regulation_transition_request(self, question: str, intent: str) -> bool:
        if "군인복무규율" not in question:
            return False
        has_transition = any(keyword in question for keyword in ["폐지", "재편", "체계", "넘어오", "이어졌", "변천", "연혁"])
        asks_current_scope = any(keyword in question for keyword in ["현재", "현행", "담당", "구분", "종류", "절차", "권리구제"])
        return has_transition and (asks_current_scope or intent == "hybrid")

    def _is_discipline_hit(self, hit: SearchHit) -> bool:
        haystack = " ".join(
            part
            for part in [
                hit.chunk.law_name,
                hit.chunk.article_no,
                hit.chunk.article_title,
                hit.chunk.text,
                str(hit.chunk.extra.get("scope", "")),
            ]
            if part
        )
        return any(term in haystack for term in DISCIPLINE_TOPIC_TERMS)

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

    @classmethod
    def _missing_explicit_article_refs(cls, question: str, evidence: list[SearchHit]) -> list[str]:
        refs = cls._explicit_article_refs(question)
        if not refs:
            return []

        found: set[str] = set()
        for hit in evidence:
            haystack = " ".join(
                part
                for part in [
                    hit.chunk.article_no,
                    hit.chunk.article_title,
                    hit.chunk.text,
                    str(hit.chunk.extra.get("scope", "")),
                ]
                if part
            ).replace(" ", "")
            for ref in refs:
                if ref in haystack:
                    found.add(ref)
        return [ref for ref in refs if ref not in found]

    def _hit_contains_all(self, hit: SearchHit, terms: list[str]) -> bool:
        haystack = " ".join(
            part
            for part in [
                hit.chunk.law_name,
                hit.chunk.article_no,
                hit.chunk.article_title,
                hit.chunk.text,
                str(hit.chunk.extra.get("scope", "")),
            ]
            if part
        )
        return all(term in haystack for term in terms)

    def _discipline_hit_count(self, hits: list[SearchHit], source_type: str) -> int:
        return sum(
            1
            for hit in hits
            if hit.chunk.source_type == source_type and self._is_discipline_hit(hit)
        )

    def _law_matches_related_history(self, hit: SearchHit) -> bool:
        haystack = " ".join(
            part
            for part in [
                hit.chunk.law_name,
                hit.chunk.article_title,
                hit.chunk.text,
                str(hit.chunk.extra.get("scope", "")),
            ]
            if part
        )
        return any(name in haystack for name in RELATED_LAW_FAMILY)

    def _keyword_overlap(self, question: str, hit: SearchHit) -> tuple[int, int]:
        haystack = " ".join(
            part
            for part in [
                hit.chunk.law_name,
                hit.chunk.article_no,
                hit.chunk.article_title,
                hit.chunk.text,
                str(hit.chunk.extra.get("scope", "")),
            ]
            if part
        )
        terms = self._topic_terms(question) or self._question_terms(question)
        direct_hits = sum(1 for term in terms if term in haystack)
        weighted_hits = sum(haystack.count(term) for term in terms if term in haystack)
        return direct_hits, weighted_hits

    def _law_target_priority(self, question: str, hit: SearchHit, law_name_filter: str | None) -> int:
        target_laws = self._mentioned_laws(question, law_name_filter)
        if not target_laws:
            return 1

        hit_law_name = (hit.chunk.law_name or "").strip()
        if hit_law_name in target_laws:
            return 0

        for target_law in target_laws:
            if hit_law_name in RELATED_LAW_MAP.get(target_law, []):
                return 1

        if hit_law_name in RELATED_LAW_FAMILY:
            return 2
        return 3

    def _source_priority(self, question: str, intent: str, hit: SearchHit) -> int:
        if self._is_history_request(question, intent):
            priorities = {
                "revision_reason": 0,
                "history_note": 1,
                "old_new_comparison": 2,
                "law_text": 3,
            }
        else:
            priorities = {
                "search": {
                    "law_text": 0,
                    "history_note": 1 if self._is_discipline_request(question) else 3,
                    "revision_reason": 2 if self._is_discipline_request(question) else 1,
                    "old_new_comparison": 3 if self._is_discipline_request(question) else 2,
                },
                "explain_change": {
                    "revision_reason": 0,
                    "old_new_comparison": 1,
                    "history_note": 2,
                    "law_text": 3,
                },
                "practical": {
                    "law_text": 0,
                    "revision_reason": 1,
                    "old_new_comparison": 2,
                    "history_note": 3,
                },
                "hybrid": {
                    "revision_reason": 0,
                    "law_text": 1,
                    "old_new_comparison": 2,
                    "history_note": 3,
                },
            }.get(intent, {})
        return priorities.get(hit.chunk.source_type or "", 9)

    def _law_level_priority(self, hit: SearchHit) -> int:
        level = (hit.chunk.law_level or "").strip()
        if level in {"법률"}:
            return 0
        if level in {"시행령", "대통령령"}:
            return 1
        if level in {"시행규칙", "부령", "총리령"}:
            return 2
        return 3

    def _history_link_priority(self, question: str, intent: str, hit: SearchHit) -> int:
        if not self._is_history_request(question, intent):
            return 1
        return 0 if self._law_matches_related_history(hit) else 1

    def _required_history_related_laws(self, question: str, law_name_filter: str | None) -> list[str]:
        target_laws = self._primary_target_laws(question, law_name_filter)
        related_targets: list[str] = []
        for target_law in target_laws:
            for related_law in RELATED_LAW_MAP.get(target_law, []):
                if related_law not in related_targets:
                    related_targets.append(related_law)
        return related_targets

    def _sort_key(self, question: str, intent: str, hit: SearchHit, law_name_filter: str | None) -> tuple:
        direct_hits, weighted_hits = self._keyword_overlap(question, hit)
        article_priority = 0 if (hit.chunk.article_no or hit.chunk.article_title) else 1
        
        explicit_article_match = 0
        explicit_refs = self._explicit_article_refs(question)
        if explicit_refs and hit.chunk.article_no in explicit_refs:
            explicit_article_match = -1
            
        return (
            explicit_article_match,
            self._law_target_priority(question, hit, law_name_filter),
            self._source_priority(question, intent, hit),
            self._history_link_priority(question, intent, hit),
            -direct_hits,
            -weighted_hits,
            self._law_level_priority(hit),
            article_priority,
            -hit.score,
            hit.chunk.id,
        )

    def _dedupe_hits(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        merged: OrderedDict[str, SearchHit] = OrderedDict()
        seen_keys: set[tuple[str, ...]] = set()
        for hit in sorted(hits, key=lambda item: self._sort_key(question, intent, item, law_name_filter)):
            keys = self._hit_dedupe_keys(hit)
            if any(key in seen_keys for key in keys):
                continue
            merged.setdefault(hit.chunk.id, hit)
            seen_keys.update(keys)
        return list(merged.values())

    @staticmethod
    def _canonical_hit_text(text: str, *, limit: int = 220) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        return normalized[:limit]

    def _hit_dedupe_keys(self, hit: SearchHit) -> list[tuple[str, ...]]:
        chunk = hit.chunk
        keys: list[tuple[str, ...]] = [("id", chunk.id)]

        if chunk.article_no or chunk.article_title:
            keys.append(
                (
                    "article",
                    chunk.law_name or "",
                    chunk.source_type or "",
                    chunk.article_no or "",
                    chunk.article_title or "",
                    self._canonical_hit_text(chunk.text),
                )
            )

        text_key = self._canonical_hit_text(chunk.text)
        if text_key:
            keys.append(("text", chunk.law_name or "", chunk.source_type or "", text_key))

        return keys

    def _resolve_source_types(self, preferred: list[str], selected: list[str] | None) -> list[str]:
        if not selected:
            return list(dict.fromkeys(preferred))
        preferred_selected = [source_type for source_type in preferred if source_type in selected]
        if preferred_selected:
            return preferred_selected
        return list(dict.fromkeys(selected))

    def _expand_query_variants(self, question: str, intent: str, law_name_filter: str | None) -> list[str]:
        queries = [question]
        topic_terms = self._topic_terms(question)
        target_laws = self._mentioned_laws(question, law_name_filter)
        prioritized_terms: list[str] = []

        direct_terms = sorted(
            [term for term in topic_terms if term in question],
            key=lambda term: (-len(term), topic_terms.index(term)),
        )
        for term in direct_terms:
            if term not in prioritized_terms:
                prioritized_terms.append(term)

        supporting_terms = sorted(
            [term for term in topic_terms if term not in prioritized_terms and len(term) >= 3],
            key=lambda term: (-len(term), topic_terms.index(term)),
        )
        for term in supporting_terms:
            if term not in prioritized_terms:
                prioritized_terms.append(term)

        for term in topic_terms:
            if term not in prioritized_terms:
                prioritized_terms.append(term)

        for topic_term in prioritized_terms[:5]:
            queries.append(topic_term)
            if self._is_history_request(question, intent):
                queries.append(f"{topic_term} 변천사")
                queries.append(f"{topic_term} 개정 이유")
            for target_law in target_laws[:2]:
                queries.append(f"{topic_term} {target_law}")
                if self._is_history_request(question, intent):
                    queries.append(f"{topic_term} {target_law} 개정 이유")

        if self._is_history_request(question, intent):
            for target_law in target_laws[:2]:
                queries.append(f"{target_law} 연혁")
            if "군인복무규율" not in " ".join(queries):
                queries.append(f"{question} 군인복무규율")

        if self._is_service_regulation_transition_request(question, intent):
            queries.extend(
                [
                    "군인복무규율 폐지",
                    "군인의 지위 및 복무에 관한 기본법 제정이유",
                    "군인의 지위 및 복무에 관한 기본법 제정 개정문 군인사법",
                    "군인고충심사위원회",
                    "고충 처리 군인사법",
                    "군인사법 제46조 제47조 제51조의3",
                    "징계 절차 군인사법",
                    "군인 징계령",
                ]
            )

        if self._is_discipline_request(question):
            for discipline_term in [
                "징계",
                "신고",
                "신고자 보호",
                "가혹행위",
                "사적 제재",
                "징계혐의자",
                "징계 절차",
                "징계의 종류",
                "징계위원회",
                "군인사법",
                "군인 징계령",
            ]:
                queries.append(discipline_term)
                for target_law in target_laws[:2]:
                    queries.append(f"{discipline_term} {target_law}")

        return list(dict.fromkeys(query for query in queries if query.strip()))

    def _expand_law_filters(self, question: str, intent: str, law_name: str | None) -> list[str | None]:
        filters: list[str | None] = []
        target_laws = self._mentioned_laws(question, law_name)

        if target_laws:
            filters.extend(target_laws)
        elif not self._is_history_request(question, intent):
            filters.append(None)

        if not self._is_history_request(question, intent):
            return filters or [None]

        if not target_laws:
            filters.append(None)

        related_laws: list[str] = []
        for target_law in target_laws:
            related_laws.extend(RELATED_LAW_MAP.get(target_law, []))

        if not target_laws:
            related_laws.append("군인복무규율")

        for related_law in related_laws:
            if related_law not in filters:
                filters.append(related_law)
        return filters or [None]

    def _filter_topic_relevance(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        if not self._topic_terms(question):
            return hits

        relevant_hits = [hit for hit in hits if self._keyword_overlap(question, hit)[0] > 0]
        if not relevant_hits:
            return hits
        return self._dedupe_hits(question, intent, relevant_hits, law_name_filter)

    def _limit_per_source_type(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        if self._is_history_request(question, intent):
            configured_limit = max(1, self.settings.retrieval.max_evidence_per_source_type)
            per_type_limits = {
                "revision_reason": min(2, configured_limit),
                "old_new_comparison": min(1, configured_limit),
                "law_text": min(2, configured_limit),
                "history_note": min(2, configured_limit),
            }
        else:
            per_type_limits = {
                "law_text": max(1, self.settings.retrieval.max_evidence_per_source_type),
                "revision_reason": max(1, self.settings.retrieval.max_evidence_per_source_type),
                "old_new_comparison": max(1, self.settings.retrieval.max_evidence_per_source_type),
                "history_note": max(1, self.settings.retrieval.max_evidence_per_source_type),
            }

        limited: list[SearchHit] = []
        counts: dict[str, int] = {}
        for hit in sorted(hits, key=lambda item: self._sort_key(question, intent, item, law_name_filter)):
            source_type = hit.chunk.source_type or "unknown"
            if counts.get(source_type, 0) >= per_type_limits.get(source_type, 1):
                continue
            limited.append(hit)
            counts[source_type] = counts.get(source_type, 0) + 1
            if len(limited) >= self.settings.retrieval.top_k:
                break
        return limited

    def _inject_candidate(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        candidate: SearchHit,
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        if any(existing.chunk.id == candidate.chunk.id for existing in hits):
            return hits

        ordered = sorted(hits, key=lambda item: self._sort_key(question, intent, item, law_name_filter))
        if len(ordered) < self.settings.retrieval.top_k:
            ordered.append(candidate)
        else:
            ordered[-1] = candidate
        return self._dedupe_hits(question, intent, ordered, law_name_filter)

    def _ensure_history_link(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        candidates: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        if not self._is_history_request(question, intent):
            return hits

        related_targets = self._required_history_related_laws(question, law_name_filter)
        if related_targets:
            if any(hit.chunk.law_name in related_targets for hit in hits):
                return hits
            for candidate in sorted(candidates, key=lambda item: self._sort_key(question, intent, item, law_name_filter)):
                if candidate.chunk.law_name in related_targets:
                    return self._inject_candidate(question, intent, hits, candidate, law_name_filter)
            return hits

        if any(self._law_matches_related_history(hit) for hit in hits):
            return hits

        for candidate in sorted(candidates, key=lambda item: self._sort_key(question, intent, item, law_name_filter)):
            if self._law_matches_related_history(candidate):
                return self._inject_candidate(question, intent, hits, candidate, law_name_filter)
        return hits

    def _ensure_revision_reason_support(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        candidates: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        if intent != "explain_change":
            return hits
        if any(hit.chunk.source_type == "revision_reason" for hit in hits):
            return hits

        for candidate in sorted(candidates, key=lambda item: self._sort_key(question, intent, item, law_name_filter)):
            if candidate.chunk.source_type != "revision_reason":
                continue
            return self._inject_candidate(question, intent, hits, candidate, law_name_filter)
        return hits

    def _ensure_timeline_coverage(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        candidates: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        if not self._is_timeline_request(question):
            return hits

        target_laws = self._mentioned_laws(question, law_name_filter)
        if not target_laws:
            return hits

        augmented_hits = list(hits)
        related_targets = [law for target_law in target_laws for law in RELATED_LAW_MAP.get(target_law, [])]

        if not any(hit.chunk.law_name in target_laws for hit in augmented_hits):
            for candidate in sorted(candidates, key=lambda item: self._sort_key(question, intent, item, law_name_filter)):
                if candidate.chunk.law_name in target_laws:
                    augmented_hits = self._inject_candidate(question, intent, augmented_hits, candidate, law_name_filter)
                    break

        if related_targets and not any(hit.chunk.law_name in related_targets for hit in augmented_hits):
            for candidate in sorted(candidates, key=lambda item: self._sort_key(question, intent, item, law_name_filter)):
                if candidate.chunk.law_name in related_targets:
                    augmented_hits = self._inject_candidate(question, intent, augmented_hits, candidate, law_name_filter)
                    break

        return augmented_hits

    def _ensure_service_regulation_transition_support(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        candidates: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        if not self._is_service_regulation_transition_request(question, intent):
            return hits

        augmented_hits = list(hits)
        supplemental_candidates = list(candidates)
        specs = [
            {
                "source_type": "old_new_comparison",
                "law_name": None,
                "required_terms": ["군인복무규율", "폐지"],
                "queries": [["군인복무규율", "폐지"], ["군인복무규율은 폐지한다"]],
            },
            {
                "source_type": "revision_reason",
                "law_name": "군인의 지위 및 복무에 관한 기본법",
                "required_terms": ["기본권"],
                "queries": [["군인의 지위 및 복무에 관한 기본법", "기본권"], ["군인의 의무", "기본권"]],
            },
            {
                "source_type": "law_text",
                "law_name": "군인의 지위 및 복무에 관한 기본법",
                "required_terms": ["군인고충심사위원회"],
                "queries": [["고충", "군인고충심사위원회"], ["고충 처리", "군인사법"]],
            },
            {
                "source_type": "old_new_comparison",
                "law_name": "군인의 지위 및 복무에 관한 기본법",
                "required_terms": ["군인사법", "제46조"],
                "queries": [["군인사법", "제46조"], ["군인사법", "제51조의3"], ["군인사법", "고충"]],
            },
            {
                "source_type": "law_text",
                "law_name": "군인사법",
                "required_terms": ["징계"],
                "queries": [["군인사법", "징계"], ["징계의 종류", "군인사법"], ["징계위원회", "군인사법"]],
            },
            {
                "source_type": "law_text",
                "law_name": "군인 징계령",
                "required_terms": ["징계"],
                "queries": [["군인 징계령", "징계"], ["징계 절차", "군인 징계령"], ["징계위원회", "군인 징계령"]],
            },
            {
                "source_type": "law_text",
                "law_name": "군인 징계령 시행규칙",
                "required_terms": ["징계"],
                "queries": [["군인 징계령 시행규칙", "징계"], ["징계 양정", "군인 징계령 시행규칙"], ["별표", "징계"]],
            },
        ]

        for spec in specs:
            source_type = str(spec["source_type"])
            law_name = spec["law_name"]
            required_terms = list(spec["required_terms"])
            if any(
                hit.chunk.source_type == source_type
                and (not law_name or hit.chunk.law_name == law_name)
                and self._hit_contains_all(hit, required_terms)
                for hit in augmented_hits
            ):
                continue

            for query_terms in spec["queries"]:
                supplemental_candidates.extend(
                    self.store.lexical_query(
                        list(query_terms),
                        top_k=max(self.settings.retrieval.top_k + 4, 10),
                        law_name=law_name,
                        source_type=source_type,
                    )
                )

            ordered_candidates = sorted(
                self._dedupe_hits(question, intent, supplemental_candidates, law_name_filter),
                key=lambda item: self._sort_key(question, intent, item, law_name_filter),
            )
            for candidate in ordered_candidates:
                if candidate.chunk.source_type != source_type:
                    continue
                if law_name and candidate.chunk.law_name != law_name:
                    continue
                if not self._hit_contains_all(candidate, required_terms):
                    continue
                augmented_hits.append(candidate)
                augmented_hits = self._dedupe_hits(question, intent, augmented_hits, law_name_filter)
                break

        return augmented_hits

    def _ensure_explicit_article_support(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        candidates: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        article_refs = self._explicit_article_refs(question)
        if not article_refs:
            return hits

        augmented_hits = list(hits)
        target_laws = self._mentioned_laws(question, law_name_filter)
        if law_name_filter and law_name_filter != "전체" and law_name_filter not in target_laws:
            target_laws.append(law_name_filter)
        law_filters = target_laws or [None]
        supplemental_candidates = list(candidates)

        for article_ref in article_refs:
            if any(hit.chunk.article_no == article_ref for hit in augmented_hits):
                continue

            for law_filter in law_filters:
                supplemental_candidates.extend(
                    self.store.lexical_query(
                        [article_ref],
                        top_k=max(self.settings.retrieval.top_k + 4, 10),
                        law_name=law_filter,
                        source_type="law_text",
                    )
                )

            ordered_candidates = sorted(
                self._dedupe_hits(question, intent, supplemental_candidates, law_name_filter),
                key=lambda item: self._sort_key(question, intent, item, law_name_filter),
            )
            for candidate in ordered_candidates:
                if candidate.chunk.source_type != "law_text":
                    continue
                if candidate.chunk.article_no != article_ref:
                    continue
                if target_laws and candidate.chunk.law_name not in target_laws:
                    continue
                augmented_hits = self._inject_candidate(question, intent, augmented_hits, candidate, law_name_filter)
                break

        return augmented_hits

    def _ensure_discipline_support(
        self,
        question: str,
        intent: str,
        hits: list[SearchHit],
        candidates: list[SearchHit],
        law_name_filter: str | None,
    ) -> list[SearchHit]:
        if not self._is_discipline_request(question):
            return hits

        augmented_hits = list(hits)
        supplemental_candidates = list(candidates)
        required_source_types = {
            "law_text": 2,
            "history_note": 1,
            "revision_reason": 1 if (self._is_history_request(question, intent) or self._is_discipline_overview_request(question)) else 0,
        }

        law_filters = self._expand_law_filters(question, intent, law_name_filter)
        supplemental_terms = ["징계", "신고", "신고자 보호", "가혹행위", "사적 제재", "징계혐의자"]
        for source_type, minimum_count in required_source_types.items():
            if minimum_count <= 0:
                continue
            current_count = self._discipline_hit_count(augmented_hits, source_type)
            if current_count >= minimum_count:
                continue

            if source_type in {"law_text", "revision_reason", "history_note"}:
                for query_text in supplemental_terms:
                    for law_filter in law_filters:
                        supplemental_candidates.extend(
                            self.store.lexical_query(
                                [query_text],
                                top_k=max(self.settings.retrieval.top_k + 2, 8),
                                law_name=law_filter,
                                source_type=source_type,
                            )
                        )

            ordered_candidates = sorted(
                self._dedupe_hits(question, intent, supplemental_candidates, law_name_filter),
                key=lambda item: self._sort_key(question, intent, item, law_name_filter),
            )
            for candidate in ordered_candidates:
                if candidate.chunk.source_type != source_type:
                    continue
                if not self._is_discipline_hit(candidate):
                    continue
                augmented_hits = self._inject_candidate(question, intent, augmented_hits, candidate, law_name_filter)
                current_count += 1
                if current_count >= minimum_count:
                    break

        return augmented_hits

    def retrieve(
        self,
        question: str,
        *,
        law_name: str | None = None,
        source_types: list[str] | None = None,
    ) -> tuple[str, str, list[SearchHit]]:
        route = decide_route(question)
        active_source_types = self._resolve_source_types(route.preferred_source_types, source_types)
        if self._is_history_request(question, route.intent) and not source_types:
            for required_source_type in ["revision_reason", "old_new_comparison", "law_text", "history_note"]:
                if required_source_type not in active_source_types:
                    active_source_types.append(required_source_type)

        query_variants = self._expand_query_variants(question, route.intent, law_name)
        law_filters = self._expand_law_filters(question, route.intent, law_name)
        hits: list[SearchHit] = []

        for source_type in active_source_types:
            for query_text in query_variants:
                for law_filter in law_filters:
                    hits.extend(
                        self.store.query(
                            query_text,
                            top_k=max(self.settings.retrieval.top_k + 2, 8),
                            law_name=law_filter,
                            source_type=source_type,
                        )
                    )

        deduped_hits = self._dedupe_hits(question, route.intent, hits, law_name)
        relevant_hits = self._filter_topic_relevance(question, route.intent, deduped_hits, law_name)
        limited_hits = self._limit_per_source_type(question, route.intent, relevant_hits, law_name)
        limited_hits = self._ensure_revision_reason_support(question, route.intent, limited_hits, deduped_hits, law_name)
        limited_hits = self._ensure_timeline_coverage(question, route.intent, limited_hits, relevant_hits, law_name)
        limited_hits = self._ensure_history_link(question, route.intent, limited_hits, relevant_hits, law_name)
        limited_hits = self._ensure_explicit_article_support(question, route.intent, limited_hits, deduped_hits, law_name)
        limited_hits = self._ensure_discipline_support(question, route.intent, limited_hits, deduped_hits, law_name)
        limited_hits = self._ensure_service_regulation_transition_support(question, route.intent, limited_hits, deduped_hits, law_name)
        return route.intent, route.rationale, limited_hits

    def answer(
        self,
        question: str,
        *,
        law_name: str | None = None,
        source_types: list[str] | None = None,
        allow_generation: bool = True,
    ) -> AnswerBundle:
        intent, route_rationale, evidence = self.retrieve(
            question=question,
            law_name=law_name,
            source_types=source_types,
        )
        answer_result = self.client.generate_answer(
            question=question,
            intent=intent,
            evidence=evidence,
            allow_generation=allow_generation,
        )
        diagnostics = {
            **(answer_result.diagnostics or {}),
            "intent": self._intent_alias(intent),
            "route_rationale": route_rationale,
            "retrieval_slots": self._evidence_slots(evidence),
            "explicit_article_refs": self._explicit_article_refs(question),
            "missing_explicit_refs": self._missing_explicit_article_refs(question, evidence),
        }
        return AnswerBundle(
            question=question,
            intent=intent,
            route_rationale=route_rationale,
            answer_markdown=answer_result.text,
            evidence=evidence,
            answer_backend=answer_result.backend,
            answer_notice=answer_result.notice,
            quota_snapshot=answer_result.quota_snapshot,
            diagnostics=diagnostics,
        )
