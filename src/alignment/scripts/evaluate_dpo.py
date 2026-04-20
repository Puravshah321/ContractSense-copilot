"""
evaluate_dpo.py (entry point script)
Runs evaluation + failure analysis to produce final results and charts:
1. Evaluate baseline vs LoRA vs DPO metrics
2. Run failure analysis and error classification
3. Generate comparison samples
4. Create all visualization plots
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.config import (
    VALIDATED_DATASET_PATH, AUGMENTED_DATASET_PATH,
    EVAL_REPORT_PATH, FAILURE_REPORT_PATH,
    METRIC_COMPARISON_PLOT, ERROR_DISTRIBUTION_PLOT,
    COMPARISON_SAMPLES_PATH, EVAL_SAMPLE_SIZE,
)
from alignment.evaluation.evaluator import evaluate_from_dataset
from alignment.failure_analysis.failure_analysis_report import analyze_failures
from alignment.failure_analysis.compare_outputs import compare_from_dataset
from alignment.evaluation.visualization import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Evaluation Pipeline")
    parser.add_argument("--dataset", default=None,
                        help="Path to dataset for evaluation")
    parser.add_argument("--sample-size", type=int, default=EVAL_SAMPLE_SIZE,
                        help="Number of samples to evaluate")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-failure", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = args.dataset
    if dataset_path is None:
        for candidate in [AUGMENTED_DATASET_PATH, VALIDATED_DATASET_PATH]:
            if Path(candidate).exists():
                dataset_path = candidate
                break

    if dataset_path is None or not Path(dataset_path).exists():
        print("❌ No dataset found for evaluation!")
        print("   Run train_dpo.py first (with --skip-train to just build the dataset)")
        sys.exit(1)

    print("=" * 60)
    print("ContractSense DPO Evaluation Pipeline")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Sample size: {args.sample_size}")

    # ── Step 1: Evaluate metrics ──
    if not args.skip_eval:
        print("\n📊 STEP 1: Running metric evaluation...")
        evaluate_from_dataset(
            dataset_path,
            EVAL_REPORT_PATH,
            sample_size=args.sample_size,
        )

    # ── Step 2: Failure analysis ──
    if not args.skip_failure:
        print("\n🔍 STEP 2: Running failure analysis...")
        analyze_failures(
            dataset_path,
            FAILURE_REPORT_PATH,
            sample_size=args.sample_size,
        )

        print("\n📋 STEP 2b: Generating comparison samples...")
        compare_from_dataset(
            dataset_path,
            COMPARISON_SAMPLES_PATH,
            sample_size=50,
        )

    # ── Step 3: Generate plots ──
    if not args.skip_plots:
        print("\n📈 STEP 3: Generating visualization plots...")
        eval_dir = str(Path(EVAL_REPORT_PATH).parent)
        generate_all_plots(
            EVAL_REPORT_PATH,
            FAILURE_REPORT_PATH,
            eval_dir,
        )

    print("\n" + "=" * 60)
    print("✅ Evaluation pipeline complete!")
    print("=" * 60)

    print("\nGenerated outputs:")
    for path, desc in [
        (EVAL_REPORT_PATH, "Evaluation report"),
        (FAILURE_REPORT_PATH, "Failure analysis"),
        (COMPARISON_SAMPLES_PATH, "Comparison samples"),
        (METRIC_COMPARISON_PLOT, "Metric comparison plot"),
        (ERROR_DISTRIBUTION_PLOT, "Error distribution plot"),
    ]:
        status = "✅" if Path(path).exists() else "❌"
        print(f"  {status} {desc}: {Path(path).name}")


if __name__ == "__main__":
    main()
