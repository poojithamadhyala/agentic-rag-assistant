"""Tool definitions and executors for the agentic loop.

Each tool has a JSON schema (sent to the LLM for tool calling) and a Python
executor. Adding a tool = add a schema + an executor entry.
"""
from __future__ import annotations

import ast
import json
import operator
from typing import Any, Callable

from .vectorstore import VectorStore

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic Messages API tool format)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": (
            "Search the robotics/simulation knowledge base for passages relevant "
            "to a query. Returns the top matching chunks with ids and scores. "
            "Use this before answering any factual question about the docs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Safely evaluate an arithmetic expression (e.g. control-loop timing, "
            "torque, gear-ratio math). Supports + - * / ** ( )."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def make_tool_executors(store: VectorStore) -> dict[str, Callable[[dict], str]]:
    """Bind tools to a vector store instance and return name -> executor map."""

    def search_docs(args: dict) -> str:
        results = store.query(args["query"], top_k=int(args.get("top_k", 4)))
        payload = [
            {
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "score": round(r.score, 4),
                "text": r.chunk.text,
            }
            for r in results
        ]
        return json.dumps(payload)

    def calculate(args: dict) -> str:
        try:
            value = _safe_eval(ast.parse(args["expression"], mode="eval"))
            return json.dumps({"result": value})
        except Exception as exc:  # noqa: BLE001 — report back to the model
            return json.dumps({"error": str(exc)})

    return {"search_docs": search_docs, "calculate": calculate}
