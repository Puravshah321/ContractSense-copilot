"""Benchmark baseline vs LoRA generation outputs and create plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generation.benchmark_generation import build_overall_leaderboard, compare_baseline_vs_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 6 generation model benchmark")
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("data/processed/generation_benchmark/generation_training_summary.json"),
    )
    parser.add_argument(
        "--holdout-path",
        type=Path,
        default=Path("data/processed/generation_eval.jsonl"),
        help="Holdout JSONL containing query/clause/citation metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/generation_benchmark"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    records = summary["results"]

    df = compare_baseline_vs_lora(
        model_training_records=records,
        holdout_path=args.holdout_path,
        output_dir=args.output_dir,
    )

    leaderboard = build_overall_leaderboard(df)
    leaderboard_path = args.output_dir / "generation_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    finetuned_only = leaderboard[leaderboard["variant"] == "lora_finetuned"].copy()
    best_finetuned = finetuned_only.head(1)
    best_finetuned_record = best_finetuned.to_dict(orient="records")[0] if not best_finetuned.empty else None

    winner_path = args.output_dir / "best_generation_model.json"
    winner_path.write_text(json.dumps(best_finetuned_record, indent=2), encoding="utf-8")

    summary_path = args.output_dir / "generation_best_model_summary.json"
    summary = {
        "tested_base_models": sorted(df["base_model"].unique().tolist()),
        "best_base_model": (
            leaderboard[leaderboard["variant"] == "baseline"].sort_values(by="final_score", ascending=False).head(1).to_dict(orient="records")[0]
            if not leaderboard[leaderboard["variant"] == "baseline"].empty
            else None
        ),
        "best_finetuned_model": best_finetuned_record,
        "winner_rule": "Best finetuned model ranked by final_score, then citation_recall, risk_salience_score, and actionability_score.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved generation comparison -> {args.output_dir / 'generation_model_comparison.csv'}")
    print(f"Saved generation leaderboard -> {leaderboard_path}")
    print(f"Saved winner summary -> {summary_path}")
    print(f"Saved winner -> {winner_path}")


if __name__ == "__main__":
    main()
