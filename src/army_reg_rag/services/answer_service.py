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
    "유래",
    "발전",
    "넘어오",
    "군인복무규율",
}

RELATED_LAW_MAP = {
    "군인의 지위 및 복무에 관한 기본법": [
        "군인복무규율",
    ],
    "군인의 지위 및 복무에 관한 기본법 시행령": [
        "군인복무규율",
    ],
    "군인의 지위 및 복무에 관한 기본법 시행규칙": [
        "군인복무규율",
    ],
    "군인복무규율": [
        "군인의 지위 및 복무에 관한 기본법",
        "군인의 지위 및 복무에 관한 기본법 시행령",
        "군인의 지위 및 복무에 관한 기본법 시행규칙",
    ],
}

RELATED_LAW_FAMILY = set(RELATED_LAW_MAP)
for related_names in RELATED_LAW_MAP.values():
    RELATED_LAW_FAMILY.update(related_names)


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

    def _question_terms(self, question: str) -> list[str]:
        stopwords = {
            "무엇",
            "관련",
            "규정",
            "중심",
            "중심으로",
            "설명",
            "설명해줘",
            "찾아줘",
            "현행",
            "기준",
            "내용",
            "바뀌었어",
            "바뀌었지",
            "개정",
            "이유",
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
        }
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", question)
        terms = []
        for token in tokens:
            stripped = token.strip()
            if len(stripped) < 2 or stripped in stopwords:
                continue
            if stripped not in terms:
                terms.append(stripped)
        return terms[:6]

    def _is_history_request(self, question: str, intent: str) -> bool:
        return intent in {"explain_change", "hybrid"} or any(keyword in question for keyword in HISTORY_REQUEST_KEYWORDS)

    def _law_matches_related_history(self, hit: SearchHit) -> bool:
        haystack = " ".join(
            part
            for part in [
                hit.chunk.law_name,
                hit.chunk.article_title,
                hit.chunk.text,
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
            ]
            if part
        )
        terms = self._question_terms(question)
        direct_hits = sum(1 for term in terms if term in haystack)
        weighted_hits = sum(haystack.count(term) for term in terms if term in haystack)
        return direct_hits, weighted_hits

    def _source_priority(self, question: str, intent: str, hit: SearchHit) -> int:
        if self._is_history_request(question, intent):
            priorities = {
                "history_note": 0,
                "revision_reason": 1,
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
                    "law_text": 2,
                    "history_note": 3,
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
        name = hit.chunk.law_name
        if "시행규칙" in name:
            return 2
        if "시행령" in name:
            return 1
        return 0

    def _history_link_priority(self, question: str, intent: str, hit: SearchHit) -> int:
        if not self._is_history_request(question, intent):
            return 1
        return 0 if self._law_matches_related_history(hit) else 1

    def _sort_key(self, question: str, intent: str, hit: SearchHit) -> tuple:
        direct_hits, weighted_hits = self._keyword_overlap(question, hit)
        article_priority = 0 if (hit.chunk.article_no or hit.chunk.article_title) else 1
        return (
            self._source_priority(question, intent, hit),
            self._history_link_priority(question, intent, hit),
            -direct_hits,
            -weighted_hits,
            self._law_level_priority(hit),
            article_priority,
            -hit.score,
            hit.chunk.id,
        )

    def _dedupe_hits(self, question: str, intent: str, hits: list[SearchHit]) -> list[SearchHit]:
        merged: OrderedDict[str, SearchHit] = OrderedDict()
        for hit in sorted(hits, key=lambda item: self._sort_key(question, intent, item)):
            merged.setdefault(hit.chunk.id, hit)
        return list(merged.values())

    def _resolve_source_types(self, preferred: list[str], selected: list[str] | None) -> list[str]:
        if not selected:
            return preferred
        preferred_selected = [source_type for source_type in preferred if source_type in selected]
        if preferred_selected:
            return preferred_selected
        return list(dict.fromkeys(selected))

    def _expand_query_variants(self, question: str, intent: str) -> list[str]:
        queries = [question]
        if self._is_history_request(question, intent):
            queries.append(f"{question} 군인복무규율 연혁")
        return list(dict.fromkeys(queries))

    def _mentioned_related_laws(self, question: str) -> list[str]:
        mentioned = [law_name for law_name in RELATED_LAW_FAMILY if law_name in question]
        return list(dict.fromkeys(mentioned))

    def _expand_law_filters(self, question: str, intent: str, law_name: str | None) -> list[str | None]:
        filters: list[str | None] = []
        if law_name and law_name != "전체":
            filters.append(law_name)
        else:
            filters.append(None)

        if not self._is_history_request(question, intent):
            return filters

        related_laws: list[str] = []
        if law_name and law_name in RELATED_LAW_MAP:
            related_laws.extend(RELATED_LAW_MAP[law_name])
        for mentioned in self._mentioned_related_laws(question):
            related_laws.extend(RELATED_LAW_MAP.get(mentioned, []))
            related_laws.append(mentioned)

        if not related_laws:
            related_laws.append("군인복무규율")

        for related_law in related_laws:
            if related_law not in filters:
                filters.append(related_law)
        return filters

    def _limit_per_source_type(self, question: str, intent: str, hits: list[SearchHit]) -> list[SearchHit]:
        limited: list[SearchHit] = []
        counts: dict[str, int] = {}
        max_per_type = max(1, self.settings.retrieval.max_evidence_per_source_type)
        for hit in sorted(hits, key=lambda item: self._sort_key(question, intent, item)):
            source_type = hit.chunk.source_type or "unknown"
            if counts.get(source_type, 0) >= max_per_type:
                continue
            limited.append(hit)
            counts[source_type] = counts.get(source_type, 0) + 1
            if len(limited) >= self.settings.retrieval.top_k:
                break
        return limited

    def _ensure_history_link(self, question: str, intent: str, hits: list[SearchHit], candidates: list[SearchHit]) -> list[SearchHit]:
        if not self._is_history_request(question, intent):
            return hits
        if any(self._law_matches_related_history(hit) for hit in hits):
            return hits

        for candidate in sorted(candidates, key=lambda item: self._sort_key(question, intent, item)):
            if not self._law_matches_related_history(candidate):
                continue
            augmented = list(hits)
            if len(augmented) < self.settings.retrieval.top_k:
                augmented.append(candidate)
            else:
                augmented[-1] = candidate
            return self._dedupe_hits(question, intent, augmented)
        return hits

    def retrieve(
        self,
        question: str,
        *,
        law_name: str | None = None,
        source_types: list[str] | None = None,
    ) -> tuple[str, str, list[SearchHit]]:
        route = decide_route(question)
        hits: list[SearchHit] = []
        active_source_types = self._resolve_source_types(route.preferred_source_types, source_types)
        if self._is_history_request(question, route.intent):
            for required_source_type in ["history_note", "revision_reason"]:
                if required_source_type not in active_source_types:
                    active_source_types.append(required_source_type)
        query_variants = self._expand_query_variants(question, route.intent)
        law_filters = self._expand_law_filters(question, route.intent, law_name)

        for source_type in active_source_types:
            for query_text in query_variants:
                for law_filter in law_filters:
                    hits.extend(
                        self.store.query(
                            query_text,
                            top_k=max(self.settings.retrieval.top_k, 4),
                            law_name=law_filter,
                            source_type=source_type,
                        )
                    )

        deduped_hits = self._dedupe_hits(question, route.intent, hits)
        limited_hits = self._limit_per_source_type(question, route.intent, deduped_hits)
        limited_hits = self._ensure_history_link(question, route.intent, limited_hits, deduped_hits)
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
