from pathlib import Path
import numpy as np

from app.index import LocalFaissIndex
from app.retrieve import Retriever
from app.storage import ChunkRecord, LocalStore


def test_retrieval_rerank(tmp_path, monkeypatch):
    store = LocalStore(tmp_path)
    chunks = [
        ChunkRecord("d", "doc.pdf", 1, "c1", "Apples are fruit and sweet.", "pdf_text", 1.0),
        ChunkRecord("d", "doc.pdf", 2, "c2", "Refunds are processed within 30 days.", "pdf_text", 1.0),
    ]
    store.append_chunks(chunks)

    class DummyIndex(LocalFaissIndex):
        def __init__(self, store):
            super().__init__(store)
        def load(self):
            self.index = None
        def search(self, query_embedding, top_k=5):
            return np.array([[0.9, 0.8]], dtype="float32"), np.array([[0, 1]], dtype="int64")

    idx = DummyIndex(store)
    retriever = Retriever(store, idx, model_name="dummy")

    monkeypatch.setattr("app.retrieve.embed_query", lambda text, model_name: np.zeros(384, dtype="float32"))
    result = retriever.retrieve("refund policy", top_k=2, use_keyword_search=True)
    assert result.hits
    assert result.hits[0].chunk.page_number == 2
