"""Build (or rebuild) the vector index from data/docs.

Usage: python scripts/ingest.py [--docs data/docs] [--out .index] [--embedder hashing]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.chunking import load_and_chunk_directory
from rag_agent.embeddings import get_embedder
from rag_agent.vectorstore import VectorStore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--out", default=".index")
    p.add_argument("--embedder", default="hashing",
                   choices=["hashing", "sentence-transformers"])
    args = p.parse_args()

    chunks = load_and_chunk_directory(args.docs)
    store = VectorStore(get_embedder(args.embedder))
    store.add(chunks)
    store.persist(args.out)
    print(f"Indexed {len(chunks)} chunks from {args.docs} -> {args.out}")


if __name__ == "__main__":
    main()
