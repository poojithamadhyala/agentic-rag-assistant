# Agentic RAG Assistant for Robotics Simulation Docs

An **agentic AI** system that answers questions over a robotics/simulation knowledge base using **LLM tool calling**, a **retrieval-augmented generation (RAG)** pipeline, **vector storage**, **Pydantic schema validation**, an **evaluation harness**, and a **FastAPI REST service**. Built with the Anthropic Messages API (model-configurable).

```
question ──> FastAPI ──> Agent loop (Claude + tools)
                            │  search_docs ──> VectorStore (cosine sim, persisted)
                            │  calculate   ──> safe arithmetic evaluator
                            ▼
                    validated AgentAnswer (JSON: answer + citations + confidence)
```

## Features

- **Agent orchestration** — multi-turn tool-use loop: the LLM plans, calls `search_docs` / `calculate`, reasons over results, and emits a final structured answer with citations.
- **RAG pipeline** — paragraph-aware chunking with overlap → embeddings → top-k cosine retrieval; grounded answers only, "I don't know" when evidence is missing.
- **Pluggable embeddings** — offline `HashingEmbedder` (deterministic, CI-friendly) or `sentence-transformers/all-MiniLM-L6-v2` for semantic retrieval; the store interface mirrors Chroma/pgvector so backends are swappable.
- **Schema validation** — every boundary (API request/response, chunks, agent output, eval cases) is a Pydantic model; malformed LLM output is caught, never propagated.
- **Evaluation tooling** — labeled eval set + harness reporting **Hit@k** and **MRR**, with a CI-failing threshold to catch retrieval regressions.
- **Production hygiene** — dependency-injected LLM client (fully mocked unit tests, no API key needed in CI), tool-call budget, clean error surfaces, Dockerized.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/ingest.py                      # build the vector index
python -m pytest tests/ -q                    # 5 tests, no API key required
python eval/run_eval.py --k 4                 # retrieval metrics

export ANTHROPIC_API_KEY=sk-ant-...
python scripts/ask.py "What payload can the UR5e handle?"

uvicorn rag_agent.api:app --app-dir src       # REST service
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "What is the default MuJoCo timestep?"}'
```

Docker: `docker build -t rag-agent . && docker run -e ANTHROPIC_API_KEY=... -p 8000:8000 rag-agent`

## Current eval results

| Metric | Value |
|---|---|
| Hit@4 | 1.00 |
| MRR | 1.00 |
| Eval cases | 8 |
| Unit tests | 5 passing |

## Swap in your own docs

Drop `.md`/`.txt` files into `data/docs/`, re-run `scripts/ingest.py`, and extend `eval/eval_set.jsonl` with labeled questions. For semantic retrieval: `python scripts/ingest.py --embedder sentence-transformers` (requires `pip install sentence-transformers`).

## Design notes

- **Why dependency injection for the LLM client?** The agent loop is the riskiest logic; mocking the client lets tests exercise the full tool-calling path deterministically.
- **Why structured JSON output?** Downstream consumers (UI, logging, evals) need machine-readable answers with citations — free-text answers can't be audited.
- **Why a tool-call budget?** Autonomous loops need hard stops; `MAX_TOOL_TURNS` bounds cost and latency and forces a graceful low-confidence answer.

## Roadmap

- Hybrid retrieval (BM25 + dense) with reciprocal rank fusion
- LLM-as-judge answer grading in the eval harness
- Streaming responses and OpenTelemetry tracing on the API
- Kubernetes manifest + horizontal scaling of the retrieval tier
