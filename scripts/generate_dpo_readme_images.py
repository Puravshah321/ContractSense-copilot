"""
Generate all DPO README images and save to Images/
Run: python scripts/generate_dpo_readme_images.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(__file__), "..", "Images")
os.makedirs(OUT, exist_ok=True)

DARK   = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"
PURPLE = "#8b5cf6"
BLUE   = "#3b82f6"
GREEN  = "#22c55e"
RED    = "#ef4444"
ORANGE = "#f97316"
YELLOW = "#eab308"
WHITE  = "#f0f6fc"
GRAY   = "#8b949e"

def dark_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=DARK)
    return fig

def style_ax(ax, title=""):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    if title:
        ax.set_title(title, color=WHITE, fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="y", color=BORDER, alpha=0.5, linewidth=0.6)

# ─────────────────────────────────────────────────────────────
# 1. THREE-WAY COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────
def plot_three_way():
    metrics = ["Retrieval\nAccuracy", "Grounding\nAccuracy", "Hallucination\nRate ↓",
               "Not-Found\nAccuracy", "Decision\nAccuracy", "Tool\nPolicy", "Actionability"]
    baseline  = [0.857, 0.714, 0.429, 1.00, 1.00, 0.60, 0.50]
    generator = [1.000, 1.000, 0.000, 1.00, 1.00, 0.80, 0.70]
    dpo       = [1.000, 1.000, 0.000, 1.00, 1.00, 0.85, 0.85]

    x = np.arange(len(metrics))
    w = 0.25

    fig = dark_fig(14, 6)
    ax = fig.add_subplot(111, facecolor=CARD)
    style_ax(ax, "Three-Way Model Comparison: Baseline vs Generator (LoRA SFT) vs DPO Aligned")

    b1 = ax.bar(x - w, baseline,  w, label="Baseline (TF-IDF)",    color=RED,    alpha=0.85, zorder=3)
    b2 = ax.bar(x,     generator, w, label="Generator (LoRA SFT)", color=BLUE,   alpha=0.85, zorder=3)
    b3 = ax.bar(x + w, dpo,       w, label="DPO Aligned (v4)",     color=GREEN,  alpha=0.85, zorder=3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
                    ha="center", va="bottom", color=WHITE, fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=WHITE, fontsize=9)
    ax.set_yticks(np.arange(0, 1.15, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.15, 0.1)], color=GRAY)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", color=WHITE)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE, fontsize=9, loc="lower right")
    ax.annotate("↓ Lower is better for Hallucination Rate", xy=(2, 0.05),
                color=ORANGE, fontsize=8, fontstyle="italic")
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUT, "dpo_three_way_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ dpo_three_way_comparison.png")

# ─────────────────────────────────────────────────────────────
# 2. QUALITY METRICS COMPARISON (Baseline vs LoRA vs DPO)
# ─────────────────────────────────────────────────────────────
def plot_quality_metrics():
    metrics   = ["Overall\nQuality", "Risk\nSalience", "Actionability", "Citation\nPresent",
                 "Format\nCompliance", "Readability"]
    baseline  = [0.1212, 0.00, 0.00, 0.00, 0.00, 0.8077]
    lora      = [0.9817, 1.00, 1.00, 1.00, 1.00, 0.8780]
    dpo       = [0.9817, 1.00, 1.00, 1.00, 1.00, 0.8780]

    x = np.arange(len(metrics))
    w = 0.25

    fig = dark_fig(13, 6)
    ax = fig.add_subplot(111, facecolor=CARD)
    style_ax(ax, "DPO Quality Metrics: Baseline vs LoRA SFT vs DPO Aligned")

    ax.bar(x - w, baseline, w, label="Baseline",        color=RED,    alpha=0.85, zorder=3)
    ax.bar(x,     lora,     w, label="LoRA SFT (Gen.)", color=BLUE,   alpha=0.85, zorder=3)
    b3 = ax.bar(x + w, dpo, w, label="DPO Aligned",     color=PURPLE, alpha=0.85, zorder=3)

    for bar in b3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
                ha="center", va="bottom", color=WHITE, fontsize=7.5, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metrics, color=WHITE, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.15, 0.1))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.15, 0.1)], color=GRAY)
    ax.set_ylabel("Score", color=WHITE)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE, fontsize=9)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUT, "dpo_quality_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ dpo_quality_metrics.png")

# ─────────────────────────────────────────────────────────────
# 3. DATASET VERSION EVOLUTION
# ─────────────────────────────────────────────────────────────
def plot_dataset_evolution():
    versions = ["v1\nBasic DPO", "v2\nResearch-Grade", "v3\nReasoning-Focused", "v4\nDiversity-First"]
    pairs    = [200, 556, 602, 700]
    colors   = [GRAY, BLUE, ORANGE, GREEN]

    fig = dark_fig(12, 5)
    ax = fig.add_subplot(111, facecolor=CARD)
    style_ax(ax, "DPO Dataset Evolution Across 4 Versions")

    bars = ax.bar(versions, pairs, color=colors, width=0.5, zorder=3, alpha=0.9,
                  edgecolor=DARK, linewidth=1.5)
    for bar, p in zip(bars, pairs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{p}+ pairs",
                ha="center", va="bottom", color=WHITE, fontsize=11, fontweight="bold")

    details = [
        "LoRA r=16\nβ=0.1, LR=5e-5\n3 epochs",
        "5 categories\nLoRA r=64, β=0.15\n4 epochs, 4-bit NF4",
        "Multi-hop focus\nbfloat16, β=0.10\nmax_len=1536",
        "Flash Attn2\ntorch.compile\nEff. batch=32"
    ]
    for i, (bar, det) in enumerate(zip(bars, details)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, det,
                ha="center", va="center", color=WHITE, fontsize=8,
                multialignment="center", alpha=0.9)

    ax.set_ylim(0, 900)
    ax.set_ylabel("Number of Preference Pairs", color=WHITE)
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.set_yticklabels(["0", "200", "400", "600", "800"], color=GRAY)
    ax.tick_params(axis="x", colors=WHITE, labelsize=10)
    ax.annotate("🏆 v4 = Production Model", xy=(3, 730), ha="center", color=GREEN,
                fontsize=10, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUT, "dpo_dataset_evolution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ dpo_dataset_evolution.png")

# ─────────────────────────────────────────────────────────────
# 4. RADAR CHART — DPO MODEL QUALITY
# ─────────────────────────────────────────────────────────────
def plot_radar():
    cats   = ["Risk Salience", "Actionability", "Citation Present",
              "Format Compliance", "Readability", "Overall Quality"]
    base   = [0.00, 0.00, 0.00, 0.00, 0.808, 0.121]
    lora   = [1.00, 1.00, 1.00, 1.00, 0.878, 0.982]
    dpo    = [1.00, 1.00, 1.00, 1.00, 0.878, 0.982]

    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    for lst in [base, lora, dpo]:
        lst += lst[:1]

    fig = dark_fig(8, 7)
    ax = fig.add_subplot(111, polar=True, facecolor=CARD)
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor("#0d1117")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color=WHITE, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=GRAY, fontsize=7)
    ax.tick_params(colors=GRAY)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(color=BORDER, alpha=0.5)

    ax.plot(angles, base, color=RED,    linewidth=2, linestyle="solid")
    ax.fill(angles, base, color=RED,    alpha=0.15)
    ax.plot(angles, lora, color=BLUE,   linewidth=2, linestyle="solid")
    ax.fill(angles, lora, color=BLUE,   alpha=0.15)
    ax.plot(angles, dpo,  color=GREEN,  linewidth=2.5, linestyle="solid")
    ax.fill(angles, dpo,  color=GREEN,  alpha=0.25)

    legend = [mpatches.Patch(color=RED, label="Baseline"),
              mpatches.Patch(color=BLUE, label="LoRA SFT (Gen.)"),
              mpatches.Patch(color=GREEN, label="DPO Aligned")]
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.35, 1.1),
              facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE, fontsize=9)
    ax.set_title("DPO Quality Radar: Baseline vs LoRA vs DPO", color=WHITE,
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dpo_quality_radar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ dpo_quality_radar.png")

# ─────────────────────────────────────────────────────────────
# 5. V4 EVAL RESULTS — HORIZONTAL BAR
# ─────────────────────────────────────────────────────────────
def plot_v4_eval():
    metrics = ["Decision\nAccuracy", "Hallucination\nCatch Rate", "Refusal\nAccuracy",
               "Grounding\nAccuracy", "Adversarial\nRobustness", "Multi-hop\nCompleteness"]
    scores  = [0.8125, 0.40, 0.9286, 1.00, 0.00, 1.00]
    colors  = [GREEN if s >= 0.8 else ORANGE if s >= 0.4 else RED for s in scores]

    fig = dark_fig(10, 6)
    ax = fig.add_subplot(111, facecolor=CARD)
    style_ax(ax, "DPO v4 — Evaluation Results (48-sample Holdout)")

    bars = ax.barh(metrics, scores, color=colors, alpha=0.88, zorder=3, height=0.55,
                   edgecolor=DARK, linewidth=1)
    for bar, s in zip(bars, scores):
        ax.text(min(s + 0.02, 0.97), bar.get_y() + bar.get_height()/2,
                f"{s:.1%}", va="center", color=WHITE, fontsize=10, fontweight="bold")

    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xticklabels([f"{v:.0%}" for v in np.arange(0, 1.1, 0.2)], color=GRAY)
    ax.set_yticklabels(metrics, color=WHITE, fontsize=9)
    ax.axvline(x=0.8, color=YELLOW, linestyle="--", linewidth=1.2, alpha=0.6, label="80% threshold")
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE, fontsize=9)

    legend_patches = [mpatches.Patch(color=GREEN,  label="≥ 80% (Strong)"),
                      mpatches.Patch(color=ORANGE, label="40–80% (Moderate)"),
                      mpatches.Patch(color=RED,    label="< 40% (Weak)")]
    ax.legend(handles=legend_patches, facecolor=CARD, edgecolor=BORDER, labelcolor=WHITE,
              fontsize=9, loc="lower right")
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUT, "dpo_v4_eval_results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ dpo_v4_eval_results.png")

# ─────────────────────────────────────────────────────────────
# 6. TRAINING CONFIG COMPARISON HEATMAP
# ─────────────────────────────────────────────────────────────
def plot_config_heatmap():
    configs = {
        "LoRA r":         [16, 64, 64, 64],
        "LoRA alpha":     [32, 128, 128, 128],
        "Beta (×100)":    [10, 15, 10, 10],
        "Epochs":         [3, 4, 3, 3],
        "LR (×1e6)":      [50, 50, 30, 20],
        "Eff. Batch":     [8, 16, 16, 32],
        "Max Len (÷100)": [10, 10, 15, 10],
    }
    versions = ["v1", "v2", "v3", "v4"]
    labels   = list(configs.keys())
    data     = np.array(list(configs.values()), dtype=float)
    norm_data = data / data.max(axis=1, keepdims=True)

    fig = dark_fig(10, 6)
    ax = fig.add_subplot(111)
    ax.set_facecolor(CARD)
    fig.patch.set_facecolor(DARK)

    im = ax.imshow(norm_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(versions))); ax.set_xticklabels(versions, color=WHITE, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(labels)));   ax.set_yticklabels(labels,   color=WHITE, fontsize=10)
    ax.tick_params(colors=GRAY)

    for i in range(len(labels)):
        for j in range(len(versions)):
            ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center",
                    color="black" if norm_data[i,j] > 0.5 else WHITE, fontsize=10, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_title("Training Hyperparameter Progression Across DPO Versions", color=WHITE,
                 fontsize=11, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=GRAY)
    cbar.set_label("Relative Scale (per row)", color=GRAY, fontsize=8)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUT, "dpo_config_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ dpo_config_heatmap.png")

# ─────────────────────────────────────────────────────────────
# 7. DPO vs BASELINE IMPROVEMENT DELTA
# ─────────────────────────────────────────────────────────────
def plot_delta():
    metrics = ["Overall Quality", "Risk Salience", "Actionability",
               "Citation Present", "Format Compliance", "Readability"]
    delta   = [0.8605, 1.00, 1.00, 1.00, 1.00, 0.0703]

    fig = dark_fig(10, 5)
    ax = fig.add_subplot(111, facecolor=CARD)
    style_ax(ax, "DPO Aligned — Improvement Over Baseline (Δ Score)")

    colors = [GREEN if d >= 0.5 else BLUE for d in delta]
    bars = ax.bar(metrics, delta, color=colors, alpha=0.88, zorder=3, width=0.55,
                  edgecolor=DARK, linewidth=1.2)
    for bar, d in zip(bars, delta):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"+{d:.1%}", ha="center", va="bottom", color=WHITE, fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f"+{v:.0%}" for v in np.arange(0, 1.1, 0.2)], color=GRAY)
    ax.tick_params(axis="x", colors=WHITE, labelsize=9, rotation=10)
    ax.set_ylabel("Delta (DPO − Baseline)", color=WHITE)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(OUT, "dpo_improvement_delta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ dpo_improvement_delta.png")

# ─────────────────────────────────────────────────────────────
# 8. DPO PIPELINE ARCHITECTURE DIAGRAM
# ─────────────────────────────────────────────────────────────
def plot_pipeline():
    fig = dark_fig(14, 6)
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK)
    fig.patch.set_facecolor(DARK)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("ContractSense DPO Alignment Pipeline — Stage 7", color=WHITE,
                 fontsize=13, fontweight="bold", pad=8)

    def box(x, y, w, h, color, label, sublabel="", fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                       linewidth=1.5, edgecolor=color, facecolor=CARD)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0), label,
                ha="center", va="center", color=WHITE, fontsize=fontsize, fontweight="bold")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center", color=GRAY, fontsize=7.5)

    def arrow(x1, y, x2, color=GRAY):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))

    # Base Model
    box(0.3, 2.2, 2.5, 1.6, BLUE, "Base Model", "Mistral-7B-Instruct-v0.2\n4-bit NF4 Quantization")
    arrow(2.8, 3.0, 3.5, BLUE)

    # LoRA
    box(3.5, 2.2, 2.5, 1.6, PURPLE, "LoRA Adapters", "r=64, alpha=128\ndropout=0.05\n~1.2% trainable")
    arrow(6.0, 3.0, 6.7, PURPLE)

    # Dataset
    box(6.7, 2.2, 2.5, 1.6, ORANGE, "DPO Dataset", "v1→v4 (200–700+ pairs)\nprompt / chosen / rejected\n5 categories")
    arrow(9.2, 3.0, 9.9, ORANGE)

    # DPO Trainer
    box(9.9, 2.2, 2.5, 1.6, GREEN, "TRL DPOTrainer", "beta=0.10–0.15\nKL-divergence penalty\nref model = frozen LoRA")

    # Output arrow down
    ax.annotate("", xy=(11.15, 2.2), xytext=(11.15, 1.5),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    box(9.9, 0.3, 2.5, 1.2, GREEN, "DPO-Aligned Model", "merge_and_unload() → HF Hub\n22Jay/ContractSense-Grounded-DPO")

    # Dataset version labels above
    for i, (lbl, col) in enumerate([("v1: Basic\n~200 pairs", GRAY), ("v2: Research\n556 pairs", BLUE),
                                    ("v3: Reasoning\n602 pairs", ORANGE), ("v4: Diversity\n700+ pairs ✓", GREEN)]):
        bx = 6.7 + 0.0 + i * 0.62
        ax.text(7.95, 4.5 - i*0.01, lbl, ha="center", va="center", color=col,
                fontsize=7.5, fontweight="bold" if i == 3 else "normal")

    ax.text(1.55, 4.3, "Input", color=GRAY, fontsize=9, ha="center")
    ax.text(13.2, 0.9, "Output", color=GREEN, fontsize=9, ha="center")
    fig.tight_layout(pad=1)
    fig.savefig(os.path.join(OUT, "dpo_pipeline_architecture.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ dpo_pipeline_architecture.png")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Saving images to: {os.path.abspath(OUT)}\n")
    plot_three_way()
    plot_quality_metrics()
    plot_dataset_evolution()
    plot_radar()
    plot_v4_eval()
    plot_config_heatmap()
    plot_delta()
    plot_pipeline()
    print("\n✅ All DPO README images generated successfully!")
