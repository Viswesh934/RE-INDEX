from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .embeddings import embed_query
from .index import LocalFaissIndex, SearchHit, rerank_hits
from .storage import ChunkRecord, LocalStore


@dataclass
class RetrievalResult:
    query: str
    hits: list[SearchHit]


class Retriever:
    def __init__(self, store: LocalStore, faiss_index: LocalFaissIndex, model_name: str):
        self.store = store
        self.faiss_index = faiss_index
        self.model_name = model_name
        self.chunks = self.store.load_chunks()

    def refresh(self) -> None:
        self.chunks = self.store.load_chunks()
        self.faiss_index.load()

    def retrieve(self, query: str, top_k: int = 5, use_keyword_search: bool = True) -> RetrievalResult:
        self.refresh()
        if not self.chunks:
            return RetrievalResult(query=query, hits=[])

        q_emb = embed_query(query, model_name=self.model_name)
        scores, ids = self.faiss_index.search(q_emb, top_k=min(top_k, len(self.chunks)))
        if ids.size == 0:
            return RetrievalResult(query=query, hits=[])

        idxs = [int(i) for i in ids[0] if int(i) >= 0]
        candidate_chunks = [self.chunks[i] for i in idxs]
        candidate_scores = [float(s) for s in scores[0][: len(candidate_chunks)]]
        hits = rerank_hits(candidate_chunks, query, candidate_scores, use_keyword_search=use_keyword_search)
        return RetrievalResult(query=query, hits=hits[:top_k])
