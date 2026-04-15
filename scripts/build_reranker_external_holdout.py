"""Create strict external holdout files for reranker training and evaluation."""

from __future__ import annotations

import argparse
import json

from src.reranking.external_holdout import build_external_holdout_pair_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contract/family-isolated external holdout reranker datasets.",
    )
    parser.add_argument(
        "--clauses-path",
        type=str,
        default="data/processed/clauses.jsonl",
        help="Input clauses JSONL path.",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default="data/processed/reranker_train_external.jsonl",
        help="Output JSONL for reranker training pairs.",
    )
    parser.add_argument(
        "--holdout-out",
        type=str,
        default="data/processed/reranker_holdout_external.jsonl",
        help="Output JSONL for strict external holdout pairs.",
    )
    parser.add_argument(
        "--metadata-out",
        type=str,
        default="data/processed/reranker_external_holdout_metadata.json",
        help="Output JSON metadata path.",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of families to assign to holdout when no explicit test split exists.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=3,
        help="How many negative candidates to sample per positive pair.",
    )
    parser.add_argument(
        "--max-queries-per-contract",
        type=int,
        default=20,
        help="Maximum positive query clauses selected per contract.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--family-regex",
        type=str,
        default=None,
        help="Optional regex removed from contract_id to build family_id for stricter grouping.",
    )
    parser.add_argument(
        "--disable-explicit-test-split",
        action="store_true",
        help="Ignore split=test rows and always sample holdout groups randomly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_external_holdout_pair_files(
        clauses_path=args.clauses_path,
        train_output_path=args.train_out,
        holdout_output_path=args.holdout_out,
        metadata_output_path=args.metadata_out,
        holdout_ratio=args.holdout_ratio,
        negatives_per_positive=args.negatives_per_positive,
        max_queries_per_contract=args.max_queries_per_contract,
        seed=args.seed,
        family_regex=args.family_regex,
        prefer_explicit_test_split=not args.disable_explicit_test_split,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()