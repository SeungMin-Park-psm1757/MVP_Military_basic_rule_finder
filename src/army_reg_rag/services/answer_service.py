from __future__ import annotations

from collections import OrderedDict
import re

from army_reg_rag.config import Settings
from army_reg_rag.domain.models import AnswerBundle, SearchHit
from army_reg_rag.llm.gemini_client import GeminiAnswerClient
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.retrieval.router import decide_route


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
            "왜", "무엇", "관련", "규정", "규정이", "중심", "중심으로", "설명", "설명해줘",
            "찾아줘", "현행", "기준", "내용", "바뀌었는지", "바뀌었어", "개정", "이유",
            "실무", "어떻게", "해야", "해줘", "현재", "조문", "무슨", "참고", "주의",
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

    def _source_priority(self, intent: str, hit: SearchHit) -> int:
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
        }
        return priorities.get(intent, {}).get(hit.chunk.source_type or "", 9)

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

    def _sort_key(self, question: str, intent: str, hit: SearchHit) -> tuple:
        direct_hits, weighted_hits = self._keyword_overlap(question, hit)
        article_priority = 0 if (hit.chunk.article_no or hit.chunk.article_title) else 1
        return (
            self._source_priority(intent, hit),
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
        for source_type in active_source_types:
            hits.extend(
                self.store.query(
                    question,
                    top_k=max(self.settings.retrieval.top_k, 4),
                    law_name=law_name,
                    source_type=source_type,
                )
            )
        hits = self._limit_per_source_type(question, route.intent, self._dedupe_hits(question, route.intent, hits))
        return route.intent, route.rationale, hits

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
