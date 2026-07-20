"""Lightweight persistent vector store with cosine-similarity search.

Stores chunk metadata as JSON and embeddings as .npz. The interface mirrors
common vector databases (add / query / persist / load) so it can be swapped
for Chroma, pgvector, or Pinecone without touching the agent layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .schemas import DocumentChunk, RetrievalResult


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[DocumentChunk] = []
        self.vectors: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)

    # ---------------------------------------------------------------- ingest
    def add(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        vecs = self.embedder.embed([c.text for c in chunks])
        self.vectors = np.vstack([self.vectors, vecs]) if self.vectors.size else vecs
        self.chunks.extend(chunks)

    # ----------------------------------------------------------------- query
    def query(self, text: str, top_k: int = 4) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        q = self.embedder.embed([text])[0]
        scores = self.vectors @ q  # rows and q are L2-normalized -> cosine sim
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in order
        ]

    # ------------------------------------------------------------ persistence
    def persist(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "vectors.npz", vectors=self.vectors)
        meta = [c.model_dump() for c in self.chunks]
        (directory / "chunks.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> "VectorStore":
        directory = Path(directory)
        store = cls(embedder)
        store.vectors = np.load(directory / "vectors.npz")["vectors"]
        meta = json.loads((directory / "chunks.json").read_text())
        store.chunks = [DocumentChunk.model_validate(m) for m in meta]
        if store.vectors.shape[0] != len(store.chunks):
            raise ValueError("vector/chunk count mismatch — re-run ingestion")
        return store
