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
    "군인복무규율",
}

TIMELINE_REQUEST_KEYWORDS = {
    "연혁",
    "변천",
    "변천사",
    "흐름",
    "어떻게 이어졌",
    "어떻게 바뀌",
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
    ],
}

RELATED_LAW_FAMILY = set(LAW_NAME_ALIASES)
for related_names in RELATED_LAW_MAP.values():
    RELATED_LAW_FAMILY.update(related_names)

TOPIC_EXPANSION_MAP = {
    "징계": ["징계", "징계혐의자", "징계조치", "불이익조치", "신고자 보호", "가혹행위", "상벌", "군기강"],
    "신고": ["신고", "신고자", "신고자 보호", "불이익조치", "가혹행위"],
    "휴가": ["휴가", "외출", "외박", "연가", "청원휴가", "특별휴가", "정기휴가", "휴가 보류"],
    "고충": ["고충", "고충심사", "재심청구"],
    "기본권": ["기본권", "인권", "가혹행위", "신고자 보호"],
}

QUESTION_STOPWORDS = {
    "무엇",
    "관련",
    "규정",
    "설명",
    "설명해줘",
    "알려줘",
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
                    "revision_reason": 1,
                    "old_new_comparison": 2,
                    "history_note": 3,
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
        return (
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
        for hit in sorted(hits, key=lambda item: self._sort_key(question, intent, item, law_name_filter)):
            merged.setdefault(hit.chunk.id, hit)
        return list(merged.values())

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

        for topic_term in topic_terms[:3]:
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
        limited_hits = self._ensure_timeline_coverage(question, route.intent, limited_hits, relevant_hits, law_name)
        limited_hits = self._ensure_history_link(question, route.intent, limited_hits, relevant_hits, law_name)
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
        return AnswerBundle(
            question=question,
            intent=intent,
            route_rationale=route_rationale,
            answer_markdown=answer_result.text,
            evidence=evidence,
            answer_backend=answer_result.backend,
            answer_notice=answer_result.notice,
            quota_snapshot=answer_result.quota_snapshot,
        )
