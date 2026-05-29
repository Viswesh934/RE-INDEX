from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class _NumpyIndexFlatIP:
    def __init__(self, d: int):
        self.d = d
        self.ntotal = 0
        self._vectors = np.empty((0, d), dtype="float32")

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.d:
            raise ValueError(f"Vector dim mismatch: expected {self.d}, got {vectors.shape[1]}")
        self._vectors = np.vstack([self._vectors, vectors])
        self.ntotal = self._vectors.shape[0]

    def search(self, query: np.ndarray, top_k: int):
        query = np.asarray(query, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if self._vectors.size == 0:
            return np.zeros((1, top_k), dtype="float32"), np.full((1, top_k), -1, dtype="int64")
        scores = query @ self._vectors.T
        order = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        return sorted_scores.astype("float32"), order.astype("int64")


def _new_index(dim: int):
    if faiss is not None:
        return faiss.IndexFlatIP(dim)
    return _NumpyIndexFlatIP(dim)


def _write_index(index, path: str) -> None:
    if faiss is not None:
        faiss.write_index(index, path)
    else:
        payload = {
            "d": index.d,
            "vectors": index._vectors.tolist(),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _read_index(path: str):
    if faiss is not None:
        return faiss.read_index(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    idx = _NumpyIndexFlatIP(payload["d"])
    vecs = np.asarray(payload["vectors"], dtype="float32")
    if vecs.size:
        idx.add(vecs)
    return idx


from .storage import ChunkRecord, LocalStore


@dataclass
class SearchHit:
    chunk: ChunkRecord
    semantic_score: float
    keyword_score: float
    combined_score: float


class LocalFaissIndex:
    def __init__(self, store: LocalStore, vector_dim: int | None = None):
        self.store = store
        self.vector_dim = vector_dim
        self.index = None

    def load(self) -> None:
        if self.store.paths.faiss_index.exists():
            self.index = _read_index(str(self.store.paths.faiss_index))
            self.vector_dim = self.index.d
        else:
            self.index = None

    def _ensure_index(self, vector_dim: int) -> None:
        if self.index is None:
            self.vector_dim = vector_dim
            self.index = _new_index(vector_dim)
        elif self.index.d != vector_dim:
            raise ValueError(f"Vector dim mismatch: existing={self.index.d}, new={vector_dim}")

    def rebuild(self, embeddings: np.ndarray) -> None:
        if embeddings.size == 0:
            self.index = _new_index(self.vector_dim or 384)
            _write_index(self.index, str(self.store.paths.faiss_index))
            return
        self._ensure_index(embeddings.shape[1])
        self.index = _new_index(embeddings.shape[1])
        self.index.add(embeddings)
        _write_index(self.index, str(self.store.paths.faiss_index))

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.size == 0:
            return
        self._ensure_index(embeddings.shape[1])
        self.index.add(embeddings)
        _write_index(self.index, str(self.store.paths.faiss_index))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            self.load()
        if self.index is None or self.index.ntotal == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        scores, ids = self.index.search(query_embedding.reshape(1, -1).astype("float32"), top_k)
        return scores, ids


def save_chunks_jsonl(store: LocalStore, chunks: list[ChunkRecord]) -> None:
    store.append_chunks(chunks)


def keyword_score(query: str, text: str) -> float:
    q_terms = [t for t in _tokens(query) if len(t) > 2]
    if not q_terms:
        return 0.0
    t_terms = set(_tokens(text))
    overlap = 0
    for q in q_terms:
        if q in t_terms:
            overlap += 1
            continue
        if any(t.startswith(q) or q.startswith(t) for t in t_terms):
            overlap += 1
    return overlap / len(q_terms)


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in __import__("re").findall(r"[A-Za-z0-9]+", text)]


def rerank_hits(
    chunks: list[ChunkRecord],
    query: str,
    semantic_scores: list[float],
    use_keyword_search: bool = True,
    semantic_weight: float = 0.75,
    keyword_weight: float = 0.25,
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for chunk, sem in zip(chunks, semantic_scores):
        kw = keyword_score(query, chunk.chunk_text) if use_keyword_search else 0.0
        combined = semantic_weight * float(sem) + keyword_weight * float(kw)
        chunk.semantic_score = float(sem)
        chunk.keyword_score = float(kw)
        hits.append(SearchHit(chunk=chunk, semantic_score=float(sem), keyword_score=float(kw), combined_score=float(combined)))
    hits.sort(key=lambda h: h.combined_score, reverse=True)
    return hits
