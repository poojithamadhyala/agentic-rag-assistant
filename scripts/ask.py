"""CLI entry point: ask the agent a question against the built index.

Usage: ANTHROPIC_API_KEY=... python scripts/ask.py "What is the UR5e payload?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.agent import RagAgent
from rag_agent.embeddings import get_embedder
from rag_agent.vectorstore import VectorStore


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('usage: python scripts/ask.py "your question"')
    store = VectorStore.load(".index", get_embedder("hashing"))
    agent = RagAgent(store)
    result = agent.ask(sys.argv[1])
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
