"""
compare_outputs.py
Aligns baseline, LoRA, and DPO outputs for the same prompt.
Stores side-by-side comparison for analysis.
"""

import json
from pathlib import Path
from typing import Optional


def build_comparison_table(
    prompts: list,
    baseline_outputs: list,
    lora_outputs: list,
    dpo_outputs: list,
) -> list:
    comparisons = []
    for i, prompt in enumerate(prompts):
        comparisons.append({
            "index": i,
            "prompt": prompt[:300],
            "baseline": baseline_outputs[i] if i < len(baseline_outputs) else "",
            "lora": lora_outputs[i] if i < len(lora_outputs) else "",
            "dpo": dpo_outputs[i] if i < len(dpo_outputs) else "",
        })
    return comparisons


def compare_from_dataset(
    dataset_path: str,
    output_path: str,
    sample_size: int = 50,
) -> list:
    with open(dataset_path) as f:
        data = json.load(f)

    import random
    random.seed(42)
    if len(data) > sample_size:
        data = random.sample(data, sample_size)

    comparisons = []
    for i, entry in enumerate(data):
        comparisons.append({
            "index": i,
            "prompt": entry["prompt"][:300],
            "baseline_rejected": entry["rejected"],
            "chosen_dpo": entry["chosen"],
            "metadata": entry.get("metadata", {}),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparisons, f, indent=2)

    print(f"📋 Saved {len(comparisons)} comparison samples to {output_path}")
    return comparisons


def save_comparison(comparisons: list, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"💾 Saved {len(comparisons)} comparisons to {output_path}")


def print_comparison(comparison: dict):
    print("=" * 70)
    print(f"PROMPT: {comparison['prompt'][:200]}")
    print("-" * 70)
    print(f"BASELINE:\n{comparison.get('baseline', comparison.get('baseline_rejected', ''))[:300]}")
    print("-" * 70)
    print(f"LoRA:\n{comparison.get('lora', 'N/A')[:300]}")
    print("-" * 70)
    print(f"DPO:\n{comparison.get('dpo', comparison.get('chosen_dpo', ''))[:300]}")
    print("=" * 70)
