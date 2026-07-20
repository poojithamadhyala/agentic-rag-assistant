"""FastAPI service exposing the agentic RAG assistant over REST.

Run:  uvicorn rag_agent.api:app --reload
Env:  ANTHROPIC_API_KEY, RAG_INDEX_DIR (default .index), RAG_EMBEDDER (hashing|sentence-transformers)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .agent import RagAgent
from .embeddings import get_embedder
from .schemas import AskRequest, AskResponse
from .vectorstore import VectorStore

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_dir = os.environ.get("RAG_INDEX_DIR", ".index")
    embedder = get_embedder(os.environ.get("RAG_EMBEDDER", "hashing"))
    store = VectorStore.load(index_dir, embedder)
    _state["store"] = store
    _state["agent"] = RagAgent(store)
    yield
    _state.clear()


app = FastAPI(title="Agentic RAG Assistant", version="1.0.0", lifespan=lifespan)


_UI_PATH = Path(__file__).resolve().parents[2] / "static" / "index.html"


@app.get("/", include_in_schema=False)
def demo_ui() -> FileResponse:
    """Serve the demo console (answer + retrieval telemetry side by side)."""
    return FileResponse(_UI_PATH, media_type="text/html")


@app.get("/health")
def health() -> dict:
    store: VectorStore = _state["store"]
    return {"status": "ok", "chunks_indexed": len(store.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    store: VectorStore = _state["store"]
    agent: RagAgent = _state["agent"]
    try:
        result = agent.ask(req.question)
    except Exception as exc:  # surface a clean 502 instead of a stack trace
        raise HTTPException(status_code=502, detail=f"agent error: {exc}") from exc
    retrieved = store.query(req.question, top_k=req.top_k)
    return AskResponse(question=req.question, result=result, retrieved=retrieved)
