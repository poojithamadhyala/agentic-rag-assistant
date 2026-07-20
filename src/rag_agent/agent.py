"""Agent orchestration: an LLM tool-use loop over the RAG knowledge base.

Flow per question:
  1. Send the question + tool schemas to the LLM.
  2. While the model requests tools, execute them and return tool_result blocks.
  3. When the model produces a final answer, parse and validate it into an
     AgentAnswer (Pydantic) so downstream consumers get structured output.

The Anthropic client is injected, so tests can pass a mock and the model name
is configurable via env (RAG_AGENT_MODEL).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .schemas import AgentAnswer, Citation
from .tools import TOOL_SCHEMAS, make_tool_executors
from .vectorstore import VectorStore

SYSTEM_PROMPT = """\
You are a robotics-simulation documentation assistant. Answer ONLY using
evidence retrieved with the search_docs tool; use the calculate tool for any
arithmetic. If the docs do not contain the answer, say so plainly.

After gathering evidence, respond with ONLY a JSON object (no markdown fences):
{"answer": "<concise answer>",
 "citations": [{"doc_id": "...", "chunk_id": "..."}],
 "confidence": "high" | "medium" | "low"}
"""

MAX_TOOL_TURNS = 6


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the model's final JSON, tolerating stray fences or prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


class RagAgent:
    def __init__(self, store: VectorStore, client: Any = None, model: str | None = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.client = client
        self.model = model or os.environ.get("RAG_AGENT_MODEL", "claude-sonnet-4-6")
        self.store = store
        self.executors = make_tool_executors(store)

    def ask(self, question: str) -> AgentAnswer:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        tools_used: list[str] = []

        for _ in range(MAX_TOOL_TURNS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    tools_used.append(block.name)
                    executor = self.executors.get(block.name)
                    output = (
                        executor(block.input)
                        if executor
                        else json.dumps({"error": f"unknown tool {block.name}"})
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            # Final answer turn
            text = "".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            raw = _extract_json(text)
            answer = AgentAnswer(
                answer=raw.get("answer", ""),
                citations=[Citation.model_validate(c) for c in raw.get("citations", [])],
                confidence=raw.get("confidence", "medium"),
                tools_used=sorted(set(tools_used)),
            )
            return answer

        return AgentAnswer(
            answer="I could not complete the request within the tool-call budget.",
            confidence="low",
            tools_used=sorted(set(tools_used)),
        )
