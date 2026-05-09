"""Demo script for Stage 6 LangGraph generation workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generation.langgraph_workflow import GenerationWorkflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage 6 generation with LangGraph")
    p.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--adapter-path", type=str, default=None)
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--clauses-path", type=Path, default=Path("data/processed/clauses.jsonl"))
    p.add_argument("--top-k", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = [json.loads(line) for line in args.clauses_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    clauses = []
    for row in rows[: args.top_k]:
        clauses.append(
            {
                "clause_id": row.get("clause_id", "unknown"),
                "page_number": max(1, int(row.get("char_count", 0)) // 3000 + 1),
                "char_span": [0, min(int(row.get("char_count", 200)), 500)],
                "clause_text": row.get("clause_text", ""),
            }
        )

    workflow = GenerationWorkflow(model_name=args.model_name, adapter_path=args.adapter_path)
    result = workflow.run(query=args.query, clauses=clauses)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
