"""
dataset_cleaner.py
Validates and filters the raw DPO dataset:
- Removes mismatched prompt pairs
- Ensures chosen has RISK / ACTION / CITATION
- Ensures rejected is weaker (no risk label, no action)
- Removes duplicates
- Trims sequences exceeding max length
- Balances risk level distribution (HIGH / MEDIUM / LOW)
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter
from typing import Optional


def _has_risk_label(text: str) -> bool:
    return bool(re.search(r"RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)", text))


def _has_action(text: str) -> bool:
    return bool(re.search(r"ACTION:", text))


def _has_citation(text: str) -> bool:
    return bool(re.search(r"CITATION:\s*\[.+?\]", text))


def _extract_risk_level(text: str) -> Optional[str]:
    m = re.search(r"RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)", text)
    return m.group(1) if m else None


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _token_count_approx(text: str) -> int:
    return len(text.split())


def validate_pair(entry: dict, max_seq_tokens: int = 512) -> tuple:
    prompt = entry.get("prompt", "")
    chosen = entry.get("chosen", "")
    rejected = entry.get("rejected", "")

    if not prompt or not chosen or not rejected:
        return False, "missing_field"

    if not _has_risk_label(chosen):
        return False, "chosen_no_risk"

    if not _has_action(chosen):
        return False, "chosen_no_action"

    if not _has_citation(chosen):
        return False, "chosen_no_citation"

    if _has_risk_label(rejected) and _has_action(rejected) and _has_citation(rejected):
        return False, "rejected_too_good"

    total_tokens = _token_count_approx(prompt) + _token_count_approx(chosen) + _token_count_approx(rejected)
    if total_tokens > max_seq_tokens * 3:
        return False, "too_long"

    if len(chosen.strip()) < 30:
        return False, "chosen_too_short"

    return True, "ok"


def clean_dataset(
    input_path: str,
    output_path: str,
    max_seq_tokens: int = 512,
    balance_risk: bool = True,
    target_per_level: Optional[int] = None,
) -> list:
    with open(input_path) as f:
        raw_data = json.load(f)

    print(f"📥 Loaded {len(raw_data)} raw pairs from {input_path}")

    valid = []
    rejection_stats = Counter()
    seen_hashes = set()

    for entry in raw_data:
        ok, reason = validate_pair(entry, max_seq_tokens)
        if not ok:
            rejection_stats[reason] += 1
            continue

        h = _text_hash(entry["prompt"] + entry["chosen"])
        if h in seen_hashes:
            rejection_stats["duplicate"] += 1
            continue
        seen_hashes.add(h)

        valid.append(entry)

    print(f"✅ {len(valid)} valid pairs after filtering")
    print(f"❌ Rejection breakdown: {dict(rejection_stats)}")

    if balance_risk:
        risk_buckets = {}
        for entry in valid:
            risk = _extract_risk_level(entry["chosen"]) or "UNKNOWN"
            risk_buckets.setdefault(risk, []).append(entry)

        print(f"📊 Risk distribution before balancing:")
        for level, items in sorted(risk_buckets.items()):
            print(f"   {level}: {len(items)}")

        if target_per_level is None:
            min_count = min(len(v) for v in risk_buckets.values() if v)
            target_per_level = min_count

        balanced = []
        for level, items in risk_buckets.items():
            if len(items) >= target_per_level:
                balanced.extend(items[:target_per_level])
            else:
                multiplier = (target_per_level // len(items)) + 1
                expanded = (items * multiplier)[:target_per_level]
                balanced.extend(expanded)

        valid = balanced
        print(f"⚖️  Balanced to {len(valid)} pairs ({target_per_level} per risk level)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(valid, f, indent=2)

    print(f"💾 Saved {len(valid)} cleaned pairs to {output_path}")
    return valid


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from alignment.config import RAW_DPO_DATASET_PATH, VALIDATED_DATASET_PATH, MAX_SEQ_LENGTH

    clean_dataset(
        RAW_DPO_DATASET_PATH,
        VALIDATED_DATASET_PATH,
        max_seq_tokens=MAX_SEQ_LENGTH,
        balance_risk=True,
    )
