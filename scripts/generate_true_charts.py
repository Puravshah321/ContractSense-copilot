import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import os

# Create Images directory if it doesn't exist
IMG_DIR = Path(r"c:\Users\Jay\Desktop\DAU\SEM-2\DL\Project\ContractSense-copilot\Images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 1. METRICS COMPARISON (Stage 6 vs Stage 7)
# ==========================================
labels = ['Overall Quality', 'Format Compliance', 'Risk Salience', 'Actionability', 'Citation Recall']
sft_scores = [0.8778, 0.9583, 0.8750, 0.9250, 0.8417]
dpo_scores = [0.9817, 1.0000, 1.0000, 1.0000, 1.0000]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, sft_scores, width, label='Stage 6: SFT (LoRA)', color='#4c72b0')
rects2 = ax.bar(x + width/2, dpo_scores, width, label='Stage 7: DPO Aligned', color='#55a868')

ax.set_ylabel('Scores (out of 1.0)')
ax.set_title('ContractSense: True SFT vs DPO Alignment Benchmarks', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.legend(loc='lower right')
ax.set_ylim([0.7, 1.05]) # Focus on the upper register

# Add data labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
out_path_1 = IMG_DIR / "true_metrics_comparison.png"
plt.savefig(out_path_1, dpi=300)
print(f"Saved: {out_path_1}")


# ==========================================
# 2. EDGE-CASE ERROR ELIMINATION
# ==========================================
error_labels = ['Missing Citation', 'Missing Risk Label', 'Missing Action', 'Broken JSON Format']
sft_errors = [19, 15, 9, 5]  # Derived from the 120-sample evaluate gap (e.g., 0.84 recall = 15.8% error = ~19 cases)
dpo_errors = [0, 0, 0, 0]

x2 = np.arange(len(error_labels))

fig2, ax2 = plt.subplots(figsize=(9, 5))
rects3 = ax2.bar(x2 - width/2, sft_errors, width, label='Stage 6: SFT (LoRA) Errors', color='#c44e52')
rects4 = ax2.bar(x2 + width/2, dpo_errors, width, label='Stage 7: DPO Errors', color='#55a868')

ax2.set_ylabel('Number of Error Occurrences (per 120 Docs)')
ax2.set_title('Error / Outlier Elimination via DPO Alignment', fontweight='bold', pad=20)
ax2.set_xticks(x2)
ax2.set_xticklabels(error_labels)
ax2.legend()
ax2.set_ylim([0, 25])

def autolabel_errors(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel_errors(rects3)
autolabel_errors(rects4)

plt.tight_layout()
out_path_2 = IMG_DIR / "true_error_elimination.png"
plt.savefig(out_path_2, dpi=300)
print(f"Saved: {out_path_2}")
