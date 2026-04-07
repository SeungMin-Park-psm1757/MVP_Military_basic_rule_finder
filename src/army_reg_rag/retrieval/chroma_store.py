from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency fallback
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency fallback
    SentenceTransformer = None

from army_reg_rag.config import Settings
from army_reg_rag.domain.models import DocumentChunk, SearchHit


class HybridTextEmbedder:
    def __init__(self, model_name: str, fallback_dim: int = 384):
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self._model = None
        self._model_load_attempted = False

    def _ensure_model(self):
        if self._model is not None or self._model_load_attempted:
            return self._model
        self._model_load_attempted = True
        if SentenceTransformer is None:
            return None
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None
        return self._model

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[\w가-힣]+", text.lower())

    def _fallback_embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * self.fallback_dim
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.fallback_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        if model is not None:
            vectors = model.encode(texts, normalize_embeddings=True)
            return [vector.tolist() for vector in vectors]
        return [self._fallback_embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    return sum(a * b for a, b in zip(vec1, vec2))


class _JsonFallbackStore:
    def __init__(self, path: Path):
        self.path = path / "_fallback_store.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.records = raw
            except Exception:
                self.records = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.records, ensure_ascii=False), encoding="utf-8")

    def count(self) -> int:
        return len(self.records)

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        for doc_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.records[doc_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }
        self._save()

    def query(self, query_embedding: list[float], n_results: int, where: dict | None) -> dict:
        rows = []
        for doc_id, payload in self.records.items():
            metadata = payload.get("metadata", {})
            if where:
                ok = True
                for key, value in where.items():
                    if metadata.get(key) != value:
                        ok = False
                        break
                if not ok:
                    continue
            score = _cosine_similarity(query_embedding, payload.get("embedding", []))
            rows.append((score, doc_id, payload.get("document", ""), metadata))
        rows.sort(key=lambda item: item[0], reverse=True)
        rows = rows[:n_results]
        return {
            "ids": [[row[1] for row in rows]],
            "documents": [[row[2] for row in rows]],
            "metadatas": [[row[3] for row in rows]],
            "distances": [[1.0 - float(row[0]) for row in rows]],
        }


class ChromaStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = HybridTextEmbedder(
            model_name=settings.embedding.model_name,
            fallback_dim=settings.embedding.fallback_dim,
        )
        self._backend = "chroma" if chromadb is not None else "json_fallback"

        if chromadb is not None:
            self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
            self.collection = self.client.get_or_create_collection(
                name=settings.app.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self.client = None
            self.collection = _JsonFallbackStore(settings.chroma_path)

    @property
    def backend_name(self) -> str:
        return self._backend

    def count(self) -> int:
        return self.collection.count()

    def upsert(self, chunks: Iterable[DocumentChunk]) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        ids = [chunk.id for chunk in chunk_list]
        documents = [chunk.text for chunk in chunk_list]
        metadatas = [chunk.to_metadata() for chunk in chunk_list]
        embeddings = self.embedder.embed_texts(documents)

        if self._backend == "chroma":
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return len(chunk_list)

    def query(
        self,
        query_text: str,
        *,
        top_k: int,
        law_name: str | None = None,
        source_type: str | None = None,
    ) -> list[SearchHit]:
        where = {}
        if law_name and law_name != "전체":
            where["law_name"] = law_name
        if source_type:
            where["source_type"] = source_type

        if self._backend == "chroma":
            result = self.collection.query(
                query_embeddings=[self.embedder.embed_query(query_text)],
                n_results=top_k,
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        else:
            result = self.collection.query(
                query_embedding=self.embedder.embed_query(query_text),
                n_results=top_k,
                where=where if where else None,
            )

        hits: list[SearchHit] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc_id, document, metadata, distance in zip(ids, docs, metadatas, distances):
            record = {
                "id": doc_id,
                "text": document,
                "law_name": metadata.get("law_name", ""),
                "law_level": metadata.get("law_level", ""),
                "source_type": metadata.get("source_type", ""),
                "version_label": metadata.get("version_label", ""),
                "promulgation_date": metadata.get("promulgation_date", ""),
                "effective_date": metadata.get("effective_date", ""),
                "article_no": metadata.get("article_no", ""),
                "article_title": metadata.get("article_title", ""),
                "revision_kind": metadata.get("revision_kind", ""),
                "source_url": metadata.get("source_url", ""),
            }
            extra = {k: v for k, v in metadata.items() if k not in record}
            record.update(extra)
            score = 1.0 / (1.0 + float(distance or 0.0))
            hits.append(SearchHit(chunk=DocumentChunk.from_record(record), score=score))
        return hits
