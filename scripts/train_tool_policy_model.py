"""Train the Stage 4 tool-policy classifier from CUAD clause data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.tool_policy_model import (
    benchmark_tool_policy_models,
    build_dataset_from_clauses,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark and train ContractSense tool-policy models."
    )
    parser.add_argument("--clauses-path", type=Path, default=Path("data/processed/clauses.jsonl"))
    parser.add_argument("--data-path", type=Path, default=Path("data/processed/tool_policy_train.jsonl"))
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument(
        "--candidate-models",
        nargs="*",
        default=["google/electra-small-discriminator", "prajjwal1/bert-tiny"],
        help="Additional model backbones to compare against the base model.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/processed/tool_policy_benchmark"),
        help="Directory where each trained model and comparison report will be saved.",
    )
    parser.add_argument("--max-clauses", type=int, default=300)
    parser.add_argument("--examples-per-tool", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["group_contract", "random"],
        default="group_contract",
        help="Data split mode. group_contract is safer for generalization checks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    build_dataset_from_clauses(
        clauses_path=args.clauses_path,
        output_path=args.data_path,
        max_clauses=args.max_clauses,
        examples_per_tool=args.examples_per_tool,
        seed=args.seed,
    )

    benchmark = benchmark_tool_policy_models(
        data_path=args.data_path,
        benchmark_dir=args.benchmark_dir,
        base_model=args.base_model,
        candidate_models=args.candidate_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        split_strategy=args.split_strategy,
        seed=args.seed,
    )

    print("Tool policy benchmarking complete")
    print(f"Base model: {benchmark['base_model']}")
    print(f"Best model: {benchmark['best_model']}")
    print(f"Best model path: {benchmark['best_model_path']}")
    print(f"Split strategy: {benchmark.get('split_strategy', args.split_strategy)}")
    print(f"Best f1_macro: {benchmark['best_f1_macro']:.4f}")
    print(f"Best accuracy: {benchmark['best_accuracy']:.4f}")
    print(f"Comparison report: {args.benchmark_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    main()