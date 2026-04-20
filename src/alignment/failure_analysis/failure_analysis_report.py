"""
failure_analysis_report.py
Aggregates failure statistics.
Counts error types, identifies most common failure patterns,
computes improvement after DPO, generates summary for report.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Optional

from .error_classifier import classify_error, get_error_statistics


def analyze_failures(
    dataset_path: str,
    output_path: str,
    sample_size: int = 500,
) -> dict:
    with open(dataset_path) as f:
        data = json.load(f)

    import random
    random.seed(42)
    if len(data) > sample_size:
        data = random.sample(data, sample_size)

    baseline_classifications = []
    dpo_classifications = []

    for entry in data:
        rejected = entry.get("rejected", "")
        chosen = entry.get("chosen", "")
        prompt = entry.get("prompt", "")

        baseline_errors = classify_error(rejected, prompt)
        baseline_classifications.append({"errors": baseline_errors, "output_preview": rejected[:200]})

        dpo_errors = classify_error(chosen, prompt)
        dpo_classifications.append({"errors": dpo_errors, "output_preview": chosen[:200]})

    baseline_stats = get_error_statistics(baseline_classifications)
    dpo_stats = get_error_statistics(dpo_classifications)

    all_error_types = set(
        list(baseline_stats["error_type_counts"].keys()) +
        list(dpo_stats["error_type_counts"].keys())
    )

    improvements = {}
    for error_type in all_error_types:
        bl_count = baseline_stats["error_type_counts"].get(error_type, 0)
        dpo_count = dpo_stats["error_type_counts"].get(error_type, 0)
        improvements[error_type] = {
            "baseline_count": bl_count,
            "dpo_count": dpo_count,
            "reduction": bl_count - dpo_count,
            "reduction_pct": round((bl_count - dpo_count) / max(bl_count, 1) * 100, 1),
        }

    report = {
        "total_samples": len(data),
        "baseline": {
            "failed_samples": baseline_stats["failed_samples"],
            "success_rate": baseline_stats["success_rate"],
            "error_type_counts": baseline_stats["error_type_counts"],
            "most_common_error": baseline_stats["most_common_error"],
        },
        "dpo_aligned": {
            "failed_samples": dpo_stats["failed_samples"],
            "success_rate": dpo_stats["success_rate"],
            "error_type_counts": dpo_stats["error_type_counts"],
            "most_common_error": dpo_stats["most_common_error"],
        },
        "improvements": improvements,
        "overall_improvement": {
            "baseline_success_rate": baseline_stats["success_rate"],
            "dpo_success_rate": dpo_stats["success_rate"],
            "delta": round(dpo_stats["success_rate"] - baseline_stats["success_rate"], 4),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 Failure Analysis Report")
    print(f"{'='*50}")
    print(f"Total samples: {len(data)}")
    print(f"Baseline success rate: {baseline_stats['success_rate']:.2%}")
    print(f"DPO success rate: {dpo_stats['success_rate']:.2%}")
    print(f"Improvement: {report['overall_improvement']['delta']:+.2%}")
    print(f"\nTop improvements:")
    for err, vals in sorted(improvements.items(), key=lambda x: -x[1]["reduction"]):
        if vals["reduction"] > 0:
            print(f"  {err}: {vals['baseline_count']} → {vals['dpo_count']} (-{vals['reduction_pct']:.0f}%)")
    print(f"\n💾 Report saved to {output_path}")

    return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alignment.config import VALIDATED_DATASET_PATH, FAILURE_REPORT_PATH

    analyze_failures(VALIDATED_DATASET_PATH, FAILURE_REPORT_PATH)
