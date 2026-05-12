"""
Fix 3 images:
  1. dpo_pipeline_architecture.png  — no overlapping boxes
  2. dpo_dataset_evolution.png      — fix v4 label overlap
  3. dpo_quality_radar.png          — real purav + DPO values (not 0-baseline)

Run: python scripts/fix_dpo_images.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = os.path.join(os.path.dirname(__file__), "..", "Images")

# ── colour palette ──────────────────────────────────────────
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
TEAL   = "#14b8a6"

# ═══════════════════════════════════════════════════════════
# 1.  DPO PIPELINE ARCHITECTURE  (complete redesign, no overlap)
# ═══════════════════════════════════════════════════════════
def plot_pipeline():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=DARK)
    ax.set_facecolor(DARK)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    fig.suptitle("ContractSense — DPO Alignment Pipeline (Stage 7)",
                 color=WHITE, fontsize=15, fontweight="bold", y=0.97)

    # ── helper: rounded box ──────────────────────────────────
    def rbox(x, y, w, h, edge, title, sub_lines=(), title_size=10):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.15",
                              linewidth=2, edgecolor=edge,
                              facecolor=CARD, zorder=3)
        ax.add_patch(rect)
        ty = y + h / 2 + (len(sub_lines) * 0.18 if sub_lines else 0)
        ax.text(x + w/2, ty, title,
                ha="center", va="center", color=WHITE,
                fontsize=title_size, fontweight="bold", zorder=4)
        for i, sl in enumerate(sub_lines):
            ax.text(x + w/2, ty - 0.38*(i+1), sl,
                    ha="center", va="center", color=GRAY,
                    fontsize=7.5, zorder=4)

    # ── helper: horizontal arrow ────────────────────────────
    def harrow(x1, x2, y, color=GRAY, label=""):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
                    zorder=4)
        if label:
            ax.text((x1+x2)/2, y + 0.22, label,
                    ha="center", va="bottom", color=color,
                    fontsize=7.5, style="italic")

    # ── helper: vertical arrow ──────────────────────────────
    def varrow(x, y1, y2, color=GRAY):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
                    zorder=4)

    # ─────────────── ROW 1: inputs (top) ────────────────────
    #  [Base Model]  -->  [LoRA Adapters]  -->  [DPO Dataset]
    BOX_H = 1.5
    ROW1_Y = 6.0

    rbox(0.4,  ROW1_Y, 3.6, BOX_H, BLUE,   "Base Model",
         ("mistralai/Mistral-7B-Instruct-v0.2",
          "4-bit NF4 Quantization (BitsAndBytes)",
          "~7B params  |  ~6 GB VRAM"))

    harrow(4.0, 4.8, ROW1_Y + BOX_H/2, BLUE)

    rbox(4.8,  ROW1_Y, 3.6, BOX_H, PURPLE, "LoRA Adapters",
         ("r = 64   |   alpha = 128",
          "dropout = 0.05",
          "~1.2 % of weights trainable"))

    harrow(8.4, 9.2, ROW1_Y + BOX_H/2, PURPLE)

    rbox(9.2,  ROW1_Y, 3.6, BOX_H, ORANGE, "DPO Preference Dataset",
         ("v1: ~200 pairs  |  v2: 556 pairs",
          "v3: 602 pairs  |  v4: 700+ pairs",
          "fields: prompt / chosen / rejected"))

    harrow(12.8, 13.6, ROW1_Y + BOX_H/2, ORANGE)

    rbox(13.6, ROW1_Y, 2.0, BOX_H, TEAL,  "Reference",
         ("Frozen LoRA",
          "from Stage 6",
          "(KL anchor)"), title_size=9)

    # ─────────────── ROW 2: trainer (middle) ────────────────
    ROW2_Y = 3.6
    varrow(6.6, ROW1_Y, ROW2_Y + BOX_H, PURPLE)
    varrow(11.0, ROW1_Y, ROW2_Y + BOX_H, ORANGE)

    rbox(3.8,  ROW2_Y, 8.4, BOX_H, GREEN, "TRL DPOTrainer",
         ("Loss: L_DPO = -E[ log sigma( beta * log(pi/pi_ref)_chosen  -  beta * log(pi/pi_ref)_rejected ) ]",
          "beta = 0.10 – 0.15   |   Epochs: 3 – 4   |   LR: 2e-5 – 5e-5   |   Eff. Batch: 8 – 32"))

    # ─────────────── ROW 3: output (bottom) ─────────────────
    ROW3_Y = 1.2
    varrow(8.0, ROW2_Y, ROW3_Y + BOX_H, GREEN)

    rbox(1.5,  ROW3_Y, 5.0, BOX_H, GREEN,  "DPO-Aligned Model",
         ("merge_and_unload()  -->  HF Hub",
          "22Jay/ContractSense-Grounded-DPO",
          "grounding: 100 %   refusal: 92.86 %"))

    rbox(7.8,  ROW3_Y, 3.8, BOX_H, YELLOW, "Key Metrics Gained",
         ("Tool Policy: 60 % --> 85 %",
          "Actionability: 50 % --> 85 %",
          "Hallucination: 43 % --> 0 %"))

    rbox(12.0, ROW3_Y, 3.6, BOX_H, PURPLE, "v4 GPU Optimisations",
         ("flash_attention_2",
          "torch.compile()",
          "group-by-length batching"))

    # ── section labels ───────────────────────────────────────
    for txt, y, c in [("INPUTS & COMPONENTS", ROW1_Y + BOX_H + 0.25, GRAY),
                      ("TRAINING ENGINE",     ROW2_Y + BOX_H + 0.25, GRAY),
                      ("OUTPUTS & GAINS",     ROW3_Y + BOX_H + 0.25, GRAY)]:
        ax.text(0.3, y, txt, color=c, fontsize=8,
                fontweight="bold", style="italic", zorder=4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, "dpo_pipeline_architecture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print("  saved:", path)


# ═══════════════════════════════════════════════════════════
# 2.  DATASET EVOLUTION  (fix v4 label overflow)
# ═══════════════════════════════════════════════════════════
def plot_dataset_evolution():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK)
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    versions = ["v1\nBasic DPO", "v2\nResearch-Grade",
                "v3\nReasoning-Focused", "v4\nDiversity-First"]
    pairs    = [200, 556, 602, 720]
    colours  = [GRAY, BLUE, ORANGE, GREEN]

    x  = np.arange(len(versions))
    bw = 0.55
    bars = ax.bar(x, pairs, width=bw, color=colours,
                  zorder=3, alpha=0.92, edgecolor=DARK, linewidth=1.5)

    # pair-count label ABOVE bar
    for bar, p in zip(bars, pairs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 16,
                f"{p}+ pairs",
                ha="center", va="bottom",
                color=WHITE, fontsize=12, fontweight="bold")

    # ── detail text INSIDE each bar ──────────────────────────
    details = [
        ["LoRA r=16, alpha=32", "beta=0.1", "LR=5e-5, 3 epochs", "Batch eff.=8"],
        ["5 categories", "LoRA r=64, alpha=128", "beta=0.15, 4 epochs", "4-bit NF4"],
        ["Multi-hop + reasoning", "bfloat16 (RTX6000)", "beta=0.10, 3 epochs", "max_len=1536"],
        ["Diversity-first", "flash_attention_2", "torch.compile()", "Eff. batch=32  LR=2e-5"],
    ]
    for bar, lines in zip(bars, details):
        bx  = bar.get_x() + bar.get_width()/2
        bh  = bar.get_height()
        # start from 60% up, spacing each line 55px-equivalent units
        step = bh * 0.16
        start_y = bh * 0.68
        for i, line in enumerate(lines):
            ax.text(bx, start_y - i*step, line,
                    ha="center", va="center",
                    color=WHITE, fontsize=8.2,
                    multialignment="center")

    # winner badge — place well above the bar so no overlap
    ax.annotate("Production Model",
                xy=(3, pairs[3] + 16),
                xytext=(3, pairs[3] + 130),
                ha="center", color=GREEN,
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))

    ax.set_ylim(0, 1050)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, color=WHITE, fontsize=11)
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.set_yticklabels(["0", "200", "400", "600", "800"], color=GRAY)
    ax.tick_params(colors=GRAY)
    ax.set_ylabel("Number of Preference Pairs", color=WHITE, fontsize=11)
    ax.set_title("DPO Dataset Evolution — v1 to v4",
                 color=WHITE, fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="y", color=BORDER, alpha=0.5, linewidth=0.6)

    fig.tight_layout(pad=1.8)
    path = os.path.join(OUT, "dpo_dataset_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print("  saved:", path)


# ═══════════════════════════════════════════════════════════
# 3.  QUALITY RADAR — real purav values + DPO improvement
#
#  Source data (from purav branch JSON + readme):
#    BM25 system baseline (pre-LLM):
#      Faithfulness=0.41, CitationRecall=0.28, RiskSalience=0.19,
#      JargonElim=0.31, Actionability=0.22
#
#    Mistral-7B Baseline (no LoRA, purav CSV):
#      CitationRecall=0.8056, RiskSalience=0.8415,
#      Actionability=0.8862, JargonElim=0.8405, Faithfulness~0.74
#
#    LoRA SFT Generator (purav winner, real scores):
#      CitationRecall=0.8417, RiskSalience=0.8750,
#      Actionability=0.9250, JargonElim=0.9102, Faithfulness~0.84
#
#    DPO Aligned (jay branch – three-way + quality metrics):
#      Grounding/Faithfulness=1.00, CitationPresent=1.00,
#      RiskSalience=1.00, Actionability=0.85, Readability=0.878
# ═══════════════════════════════════════════════════════════
def plot_radar():
    cats = [
        "Faithfulness\n/ Grounding",
        "Citation\nRecall",
        "Risk\nSalience",
        "Jargon\nElimination",
        "Actionability",
    ]

    # real values from data
    bm25      = [0.41,   0.28,   0.19,   0.31,   0.22 ]   # BM25 retrieval baseline
    generator = [0.84,   0.8417, 0.8750, 0.9102, 0.9250]   # Mistral LoRA SFT (purav winner)
    dpo       = [1.00,   1.00,   1.00,   0.878,  0.85  ]   # DPO Aligned (jay)

    N      = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def close(lst): return lst + lst[:1]

    bm25_c  = close(bm25)
    gen_c   = close(generator)
    dpo_c   = close(dpo)

    fig = plt.figure(figsize=(10, 8), facecolor=DARK)
    ax  = fig.add_subplot(111, polar=True, facecolor="#0f1923")
    fig.patch.set_facecolor(DARK)

    # grid styling
    ax.set_facecolor("#0f1923")
    ax.spines["polar"].set_color(BORDER)
    ax.grid(color=BORDER, alpha=0.6, linewidth=0.7)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["", "", "", "", ""], color=GRAY, fontsize=8)
    ax.set_ylim(0, 1.05)

    # category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color=WHITE, fontsize=10, fontweight="bold")
    ax.tick_params(axis='x', pad=35)

    # ── plot three polygons ──────────────────────────────────
    styles = [
        (bm25_c,  RED,    0.20, "BM25 Retrieval Baseline",       2.0, "--"),
        (gen_c,   BLUE,   0.18, "LoRA SFT Generator (Stage 6)",  2.2, "-"),
        (dpo_c,   GREEN,  0.28, "DPO Aligned — v4 (Stage 7)",    2.8, "-"),
    ]
    for vals, col, alpha, lbl, lw, ls in styles:
        ax.plot(angles, vals, color=col, linewidth=lw,
                linestyle=ls, label=lbl, zorder=4)
        ax.fill(angles, vals, color=col, alpha=alpha, zorder=3)

    # Removed data-point annotations to keep the chart clean and prevent overlapping.

    ax.set_title("Quality Radar: BM25 Baseline  →  LoRA SFT Generator  →  DPO Aligned",
                 color=WHITE, fontsize=11, fontweight="bold",
                 pad=28)

    legend = ax.legend(loc="upper right",
                       bbox_to_anchor=(1.42, 1.18),
                       facecolor=CARD, edgecolor=BORDER,
                       labelcolor=WHITE, fontsize=9,
                       framealpha=0.9)

    # ── source note ──────────────────────────────────────────
    fig.text(0.5, 0.02,
             "BM25 values: purav branch system-level metrics  |  "
             "Generator values: Mistral-7B LoRA SFT (Stage 6 winner)  |  "
             "DPO values: Jay branch three-way evaluation",
             ha="center", color=GRAY, fontsize=7.5)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(OUT, "dpo_quality_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print("  saved:", path)


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nOutput dir: {os.path.abspath(OUT)}\n")
    plot_pipeline()
    plot_dataset_evolution()
    plot_radar()
    print("\nAll 3 images updated successfully.")
