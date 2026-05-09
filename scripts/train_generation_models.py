"""Train Stage 6 generator model candidates with LoRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generation.train_generator import (
    DEFAULT_MODELS,
    build_generator_dataset_from_clauses,
    train_model_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple Stage 6 generation models with LoRA")
    parser.add_argument("--clauses-path", type=Path, default=Path("data/processed/clauses.jsonl"))
    parser.add_argument("--train-out", type=Path, default=Path("data/processed/generation_train.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/processed/generation_eval.jsonl"))
    parser.add_argument("--models-out", type=Path, default=Path("data/processed/generation_models"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/processed/generation_benchmark"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--model-name",
        action="append",
        dest="model_names",
        default=None,
        help="Repeatable. Base model to train; if omitted, uses default candidate list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_n, eval_n = build_generator_dataset_from_clauses(
        clauses_path=args.clauses_path,
        output_train=args.train_out,
        output_eval=args.eval_out,
    )

    model_names = args.model_names or DEFAULT_MODELS
    results = train_model_candidates(
        train_file=args.train_out,
        eval_file=args.eval_out,
        output_root=args.models_out,
        model_names=model_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.benchmark_dir / "generation_training_summary.json"
    summary = {
        "train_examples": train_n,
        "eval_examples": eval_n,
        "model_count": len(model_names),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved generation training summary -> {summary_path}")


if __name__ == "__main__":
    main()
