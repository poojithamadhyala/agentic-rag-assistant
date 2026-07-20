"""Document loading and chunking for the RAG ingestion pipeline.

Splits markdown/text documents into overlapping chunks sized for embedding,
preserving paragraph boundaries where possible.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .schemas import DocumentChunk


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:12]


def chunk_text(
    text: str,
    doc_id: str,
    source_path: str,
    max_chars: int = 900,
    overlap_chars: int = 150,
) -> list[DocumentChunk]:
    """Split text into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[DocumentChunk] = []
    buf = ""

    def flush(buffer: str) -> None:
        if not buffer.strip():
            return
        idx = len(chunks)
        chunks.append(
            DocumentChunk(
                chunk_id=_hash_id(doc_id, str(idx), buffer[:64]),
                doc_id=doc_id,
                source_path=source_path,
                text=buffer.strip(),
                chunk_index=idx,
            )
        )

    for para in paragraphs:
        if len(buf) + len(para) + 2 <= max_chars:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            flush(buf)
            # carry overlap from the tail of the previous buffer for context continuity
            tail = buf[-overlap_chars:] if overlap_chars and buf else ""
            buf = f"{tail}\n\n{para}" if tail else para
            # hard-split any single paragraph longer than max_chars
            while len(buf) > max_chars:
                flush(buf[:max_chars])
                buf = buf[max_chars - overlap_chars :]
    flush(buf)
    return chunks


def load_and_chunk_directory(docs_dir: str | Path) -> list[DocumentChunk]:
    """Load every .md/.txt file in a directory and return its chunks."""
    docs_dir = Path(docs_dir)
    all_chunks: list[DocumentChunk] = []
    for path in sorted(docs_dir.glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, doc_id=doc_id, source_path=str(path)))
    return all_chunks
