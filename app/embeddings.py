from __future__ import annotations

from functools import lru_cache
import hashlib
from typing import Iterable

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FALLBACK_DIM = 384


try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class _FallbackEmbedder:
    def __init__(self, dim: int = DEFAULT_FALLBACK_DIM):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vectors = np.vstack([self._encode_one(t) for t in texts]).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        tokens = [tok.lower() for tok in __import__("re").findall(r"[A-Za-z0-9]+", text)]
        for tok in tokens:
            digest = hashlib.sha1(tok.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            vec[idx] += 1.0
        return vec


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    if SentenceTransformer is None:
        return _FallbackEmbedder()
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return _FallbackEmbedder()


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    model = get_embedding_model(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return vectors.astype("float32")


def embed_query(text: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    return embed_texts([text], model_name=model_name)[0]
