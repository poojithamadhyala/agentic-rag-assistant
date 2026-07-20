"""Retrieval evaluation harness.

Measures Hit@k and MRR for the retriever against a labeled eval set
(eval/eval_set.jsonl). Run after any change to chunking, embeddings, or the
store to catch retrieval regressions:

    python eval/run_eval.py --index .index --k 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.embeddings import get_embedder
from rag_agent.schemas import EvalCase
from rag_agent.vectorstore import VectorStore


def load_cases(path: Path) -> list[EvalCase]:
    return [
        EvalCase.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def evaluate(store: VectorStore, cases: list[EvalCase], k: int) -> dict:
    hits, rr_sum = 0, 0.0
    per_case = []
    for case in cases:
        results = store.query(case.question, top_k=k)
        ranked_docs = [r.chunk.doc_id for r in results]
        rank = next(
            (i + 1 for i, d in enumerate(ranked_docs) if d in case.expected_doc_ids),
            None,
        )
        hit = rank is not None
        hits += int(hit)
        rr_sum += (1.0 / rank) if rank else 0.0
        per_case.append({"question": case.question, "hit": hit, "rank": rank})
    n = len(cases)
    return {
        "n_cases": n,
        f"hit@{k}": round(hits / n, 3),
        "mrr": round(rr_sum / n, 3),
        "per_case": per_case,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=".index")
    p.add_argument("--eval-set", default="eval/eval_set.jsonl")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--embedder", default="hashing")
    args = p.parse_args()

    store = VectorStore.load(args.index, get_embedder(args.embedder))
    cases = load_cases(Path(args.eval_set))
    report = evaluate(store, cases, args.k)
    print(json.dumps(report, indent=2))
    if report[f"hit@{args.k}"] < 0.8:
        raise SystemExit("FAIL: hit rate below 0.8 threshold")


if __name__ == "__main__":
    main()
