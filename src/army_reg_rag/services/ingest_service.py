from __future__ import annotations

from typing import Iterable

from army_reg_rag.domain.models import DocumentChunk
from army_reg_rag.retrieval.chroma_store import ChromaStore
from army_reg_rag.utils.io import read_jsonl


def load_chunks_from_jsonl(path: str) -> list[DocumentChunk]:
    return [DocumentChunk.from_record(record) for record in read_jsonl(path)]


def ingest_jsonl(path: str, store: ChromaStore) -> int:
    chunks = load_chunks_from_jsonl(path)
    return store.upsert(chunks)
