from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    id: str
    law_name: str
    law_level: str
    source_type: str
    version_label: str
    promulgation_date: str
    effective_date: str
    article_no: str
    article_title: str
    revision_kind: str
    text: str
    source_url: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        data = {
            "law_name": self.law_name,
            "law_level": self.law_level,
            "source_type": self.source_type,
            "version_label": self.version_label,
            "promulgation_date": self.promulgation_date,
            "effective_date": self.effective_date,
            "article_no": self.article_no,
            "article_title": self.article_title,
            "revision_kind": self.revision_kind,
            "source_url": self.source_url,
        }
        for key, value in self.extra.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                data[key] = value if value is not None else ""
            else:
                data[key] = str(value)
        return data

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DocumentChunk":
        core = {k: record.get(k, "") for k in [
            "id", "law_name", "law_level", "source_type", "version_label",
            "promulgation_date", "effective_date", "article_no", "article_title",
            "revision_kind", "text", "source_url"
        ]}
        extra = {k: v for k, v in record.items() if k not in core}
        return cls(**core, extra=extra)

    def to_record(self) -> dict[str, Any]:
        base = {
            "id": self.id,
            "law_name": self.law_name,
            "law_level": self.law_level,
            "source_type": self.source_type,
            "version_label": self.version_label,
            "promulgation_date": self.promulgation_date,
            "effective_date": self.effective_date,
            "article_no": self.article_no,
            "article_title": self.article_title,
            "revision_kind": self.revision_kind,
            "text": self.text,
            "source_url": self.source_url,
        }
        return {**base, **self.extra}


@dataclass(slots=True)
class SearchHit:
    chunk: DocumentChunk
    score: float


@dataclass(slots=True)
class RouteDecision:
    intent: str
    preferred_source_types: list[str]
    rationale: str


@dataclass(slots=True)
class AnswerBundle:
    question: str
    intent: str
    route_rationale: str
    answer_markdown: str
    evidence: list[SearchHit]
    answer_backend: str = "retrieval_fallback"
    answer_notice: str = ""
    quota_snapshot: dict[str, Any] = field(default_factory=dict)
