"""
visualization.py
Generates plots for the DPO alignment pipeline evaluation:
- Metric comparison bar chart (Baseline vs LoRA vs DPO)
- Error distribution charts
- Training curves
"""

import json
from pathlib import Path
from typing import Optional


def plot_metric_comparison(
    report_path: str,
    output_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(report_path) as f:
        report = json.load(f)

    summary = report["summary"]
    metrics = list(summary.keys())
    baseline_vals = [summary[m]["baseline"] for m in metrics]
    lora_vals = [summary[m]["lora"] for m in metrics]
    dpo_vals = [summary[m]["dpo"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#95A5A6", "#3498DB", "#2ECC71"]
    bars1 = ax.bar(x - width, baseline_vals, width, label="Baseline", color=colors[0], alpha=0.85)
    bars2 = ax.bar(x, lora_vals, width, label="LoRA", color=colors[1], alpha=0.85)
    bars3 = ax.bar(x + width, dpo_vals, width, label="DPO", color=colors[2], alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: Baseline vs LoRA vs DPO", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Metric comparison plot saved to {output_path}")


def plot_improvement_heatmap(
    report_path: str,
    output_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(report_path) as f:
        report = json.load(f)

    summary = report["summary"]
    metrics = list(summary.keys())
    comparisons = ["lora_vs_baseline", "dpo_vs_baseline", "dpo_vs_lora"]

    data = []
    for metric in metrics:
        row = [summary[metric].get(c, 0) for c in comparisons]
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.RdYlGn

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(["LoRA vs BL", "DPO vs BL", "DPO vs LoRA"], fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace("_", " ").title() for m in metrics], fontsize=10)

    for i in range(len(metrics)):
        for j in range(len(comparisons)):
            text = ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center",
                          color="black", fontsize=9, fontweight="bold")

    ax.set_title("Improvement Heatmap", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Delta")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Improvement heatmap saved to {output_path}")


def plot_error_distribution(
    failure_report_path: str,
    output_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(failure_report_path) as f:
        report = json.load(f)

    improvements = report.get("improvements", {})
    baseline_counts = report.get("baseline", {}).get("error_type_counts", {})

    if not improvements and not baseline_counts:
        print("⚠️  No error data to plot")
        return

    if improvements:
        categories = [c for c in improvements.keys() if c != "none"]
        bl_counts = [improvements[c].get("baseline_count", improvements[c].get("baseline", 0)) for c in categories]
        dpo_counts = [improvements[c].get("dpo_count", improvements[c].get("dpo", 0)) for c in categories]
    else:
        categories = list(baseline_counts.keys())
        bl_counts = list(baseline_counts.values())
        dpo_counts = [0] * len(categories)

    if not categories:
        print("⚠️  No error categories to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    y = np.arange(len(categories))
    ax1.barh(y - 0.2, bl_counts, 0.4, label="Baseline", color="#E74C3C", alpha=0.8)
    ax1.barh(y + 0.2, dpo_counts, 0.4, label="DPO", color="#2ECC71", alpha=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(categories)
    ax1.set_xlabel("Count")
    ax1.set_title("Error Distribution: Baseline vs DPO", fontweight="bold")
    ax1.legend()
    ax1.grid(True, axis="x", alpha=0.3)

    reductions = [improvements.get(c, {}).get("reduction_pct", 0) for c in categories]
    colors = ["#2ECC71" if r > 0 else "#E74C3C" for r in reductions]
    ax2.barh(categories, reductions, color=colors, alpha=0.8)
    ax2.set_xlabel("Reduction %")
    ax2.set_title("Error Reduction after DPO", fontweight="bold")
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Error distribution plot saved to {output_path}")


def generate_all_plots(
    eval_report_path: str,
    failure_report_path: str,
    output_dir: str,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if Path(eval_report_path).exists():
        plot_metric_comparison(
            eval_report_path,
            str(Path(output_dir) / "metric_comparison.png"),
        )
        plot_improvement_heatmap(
            eval_report_path,
            str(Path(output_dir) / "improvement_heatmap.png"),
        )

    if Path(failure_report_path).exists():
        plot_error_distribution(
            failure_report_path,
            str(Path(output_dir) / "error_distribution.png"),
        )

    print(f"✅ All plots generated in {output_dir}")
