"""Pluggable embedding backends.

- HashingEmbedder: dependency-free lexical embedding (hashed bag-of-words with
  sublinear TF weighting). Works fully offline; used for tests/CI and as a
  fallback retriever signal.
- SentenceTransformerEmbedder: semantic embeddings via sentence-transformers
  (all-MiniLM-L6-v2 by default). Recommended for real deployments.

Both expose the same interface so the vector store is backend-agnostic.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, offline hashed bag-of-words embedder (feature hashing)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for tok in self._tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                counts[h % self.dim] = counts.get(h % self.dim, 0.0) + 1.0
            for j, c in counts.items():
                out[i, j] = 1.0 + math.log(c)  # sublinear TF
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    """Semantic embeddings via sentence-transformers (requires model download)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(backend: str = "hashing") -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedding backend: {backend!r}")
