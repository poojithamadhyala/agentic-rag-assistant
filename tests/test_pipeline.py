"""Unit tests for chunking, vector store, tools, and the agent loop (mocked LLM)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.agent import RagAgent
from rag_agent.chunking import chunk_text, load_and_chunk_directory
from rag_agent.embeddings import HashingEmbedder
from rag_agent.tools import make_tool_executors
from rag_agent.vectorstore import VectorStore

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"


def build_store() -> VectorStore:
    store = VectorStore(HashingEmbedder(dim=512))
    store.add(load_and_chunk_directory(DOCS_DIR))
    return store


# ------------------------------------------------------------------ chunking
def test_chunking_respects_max_chars_and_ids_unique():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 120 for i in range(20))
    chunks = chunk_text(text, doc_id="d", source_path="d.md", max_chars=400)
    assert all(len(c.text) <= 400 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# --------------------------------------------------------------- vector store
def test_retrieval_finds_relevant_doc():
    store = build_store()
    results = store.query("default MuJoCo timestep", top_k=3)
    assert results and results[0].chunk.doc_id == "mujoco_basics"


def test_persist_and_load_roundtrip(tmp_path):
    store = build_store()
    store.persist(tmp_path)
    loaded = VectorStore.load(tmp_path, HashingEmbedder(dim=512))
    assert len(loaded.chunks) == len(store.chunks)
    assert loaded.query("UR5e payload", top_k=1)[0].chunk.doc_id == "ros2_digital_twin"


# --------------------------------------------------------------------- tools
def test_calculate_tool_safe_eval():
    executors = make_tool_executors(build_store())
    assert json.loads(executors["calculate"]({"expression": "1/500"}))["result"] == 0.002
    assert "error" in json.loads(
        executors["calculate"]({"expression": "__import__('os')"})
    )


# ------------------------------------------------------------- agent (mocked)
class FakeClient:
    """Mocks the Anthropic client: one tool_use turn, then a final JSON answer."""

    def __init__(self):
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            block = SimpleNamespace(
                type="tool_use",
                id="tu_1",
                name="search_docs",
                input={"query": "UR5e payload"},
            )
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        final = {
            "answer": "The UR5e payload is 5 kg.",
            "citations": [{"doc_id": "ros2_digital_twin", "chunk_id": "abc123def456"}],
            "confidence": "high",
        }
        block = SimpleNamespace(type="text", text=json.dumps(final))
        return SimpleNamespace(stop_reason="end_turn", content=[block])


def test_agent_loop_with_mocked_llm():
    agent = RagAgent(build_store(), client=FakeClient(), model="test-model")
    result = agent.ask("What payload can the UR5e handle?")
    assert "5 kg" in result.answer
    assert result.confidence == "high"
    assert result.tools_used == ["search_docs"]
    assert result.citations[0].doc_id == "ros2_digital_twin"
