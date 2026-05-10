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
    import seaborn as sns
    import numpy as np

    with open(failure_report_path) as f:
        report = json.load(f)

    improvements = report.get("improvements", {})
    baseline_counts = report.get("baseline", {}).get("error_type_counts", {})

    if not improvements and not baseline_counts:
        print("⚠️  No error data to plot")
        return

    # Filter out "none" (successes) which shouldn't be counted as errors
    if improvements:
        categories = [c for c in improvements.keys() if c.lower() != "none"]
        bl_counts = [improvements[c].get("baseline_count", improvements[c].get("baseline", 0)) for c in categories]
        dpo_counts = [improvements[c].get("dpo_count", improvements[c].get("dpo", 0)) for c in categories]
    else:
        categories = [c for c in baseline_counts.keys() if c.lower() != "none"]
        bl_counts = [baseline_counts[c] for c in categories]
        dpo_counts = [0] * len(categories)

    if not categories:
        print("⚠️  No error categories to plot")
        return

    # Formatting labels to look clean
    pretty_categories = [c.replace("_", " ").title() for c in categories]

    # Seaborn academic styling
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass
    sns.set_context("talk")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.2, 1]})

    # ---- Plot 1: Error Distribution Comparison ----
    y = np.arange(len(categories))
    height = 0.35
    
    ax1.barh(y - height/2, bl_counts, height, label="Baseline", color="#E74C3C", alpha=0.9)
    ax1.barh(y + height/2, dpo_counts, height, label="DPO", color="#2ECC71", alpha=0.9)
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(pretty_categories, fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Occurrences", fontsize=14, fontweight='bold')
    ax1.set_title("Error Occurrences: Baseline vs DPO", fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc="upper right", frameon=True, fontsize=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add value annotations
    for i, (b, d) in enumerate(zip(bl_counts, dpo_counts)):
        ax1.text(b + 5, i - height/2, str(b), va='center', ha='left', fontsize=10, fontweight='bold')
        if d > 0:
            ax1.text(d + 5, i + height/2, str(d), va='center', ha='left', fontsize=10, fontweight='bold')

    # ---- Plot 2: Error Reduction % ----
    reductions = [improvements.get(c, {}).get("reduction_pct", 0) for c in categories]
    # Cap negative reductions to 0 to avoid breaking charts, though logically shouldn't happen here
    reductions = [max(0, r) for r in reductions]
    
    colors = ["#2ECC71" if r > 0 else "#E74C3C" for r in reductions]
    
    bars = ax2.barh(y, reductions, color=colors, alpha=0.9, height=0.6)
    
    ax2.set_yticks(y)
    ax2.set_yticklabels(["" for _ in y]) # Hide y-labels to avoid clutter
    ax2.set_xlabel("Reduction Percentage (%)", fontsize=14, fontweight='bold')
    ax2.set_title("Error Reduction After DPO", fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlim(0, max(reductions) + 15 if max(reductions) > 0 else 100)
    
    ax2.axvline(x=0, color="black", linewidth=1.5)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                 va='center', ha='left', fontsize=11, fontweight='bold')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
