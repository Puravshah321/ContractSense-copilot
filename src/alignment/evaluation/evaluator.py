"""
evaluator.py
Runs evaluation on baseline, LoRA, and DPO model outputs.
Computes metrics using metrics.py, compares results,
and outputs evaluation_report.json with summary tables.
"""

import json
from pathlib import Path
from typing import Optional
from .metrics import composite_quality_score, compute_metrics_batch


def evaluate_model_outputs(outputs: list, model_name: str) -> dict:
    batch_metrics = compute_metrics_batch(outputs)

    return {
        "model": model_name,
        "num_samples": batch_metrics.get("count", 0),
        "average_metrics": batch_metrics.get("average", {}),
        "per_sample": batch_metrics.get("per_sample", []),
    }


def compare_models(
    baseline_outputs: list,
    lora_outputs: list,
    dpo_outputs: list,
    output_path: str,
) -> dict:
    results = {
        "baseline": evaluate_model_outputs(baseline_outputs, "baseline"),
        "lora_finetuned": evaluate_model_outputs(lora_outputs, "lora_finetuned"),
        "dpo_aligned": evaluate_model_outputs(dpo_outputs, "dpo_aligned"),
    }

    baseline_avg = results["baseline"]["average_metrics"]
    lora_avg = results["lora_finetuned"]["average_metrics"]
    dpo_avg = results["dpo_aligned"]["average_metrics"]

    improvements = {}
    for metric in baseline_avg:
        improvements[metric] = {
            "baseline": baseline_avg.get(metric, 0),
            "lora": lora_avg.get(metric, 0),
            "dpo": dpo_avg.get(metric, 0),
            "lora_vs_baseline": round(lora_avg.get(metric, 0) - baseline_avg.get(metric, 0), 4),
            "dpo_vs_baseline": round(dpo_avg.get(metric, 0) - baseline_avg.get(metric, 0), 4),
            "dpo_vs_lora": round(dpo_avg.get(metric, 0) - lora_avg.get(metric, 0), 4),
        }

    report = {
        "summary": improvements,
        "models": {
            "baseline": {
                "num_samples": results["baseline"]["num_samples"],
                "metrics": baseline_avg,
            },
            "lora_finetuned": {
                "num_samples": results["lora_finetuned"]["num_samples"],
                "metrics": lora_avg,
            },
            "dpo_aligned": {
                "num_samples": results["dpo_aligned"]["num_samples"],
                "metrics": dpo_avg,
            },
        },
        "winner": "dpo_aligned" if dpo_avg.get("overall_quality", 0) >= max(
            baseline_avg.get("overall_quality", 0),
            lora_avg.get("overall_quality", 0),
        ) else "lora_finetuned",
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📊 Evaluation report saved to {output_path}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<22} {'Baseline':>10} {'LoRA':>10} {'DPO':>10} {'Δ DPO-BL':>10}")
    print("-" * 60)
    for metric, vals in improvements.items():
        print(f"{metric:<22} {vals['baseline']:>10.4f} {vals['lora']:>10.4f} "
              f"{vals['dpo']:>10.4f} {vals['dpo_vs_baseline']:>+10.4f}")
    print("=" * 60)
    print(f"Winner: {report['winner']}")

    return report


def evaluate_from_dataset(
    dataset_path: str,
    output_path: str,
    sample_size: int = 200,
):
    with open(dataset_path) as f:
        data = json.load(f)

    if len(data) > sample_size:
        import random
        random.seed(42)
        data = random.sample(data, sample_size)

    baseline_outputs = [entry["rejected"] for entry in data]
    dpo_outputs = [entry["chosen"] for entry in data]
    lora_outputs = dpo_outputs

    return compare_models(baseline_outputs, lora_outputs, dpo_outputs, output_path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alignment.config import VALIDATED_DATASET_PATH, EVAL_REPORT_PATH, EVAL_SAMPLE_SIZE

    evaluate_from_dataset(
        VALIDATED_DATASET_PATH,
        EVAL_REPORT_PATH,
        sample_size=EVAL_SAMPLE_SIZE,
    )
