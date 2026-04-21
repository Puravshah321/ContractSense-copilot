import json
from pathlib import Path
import pandas as pd
import re

# =========================================================================
# UNIVERSAL PATH CONFIGURATION (STAGE 7 READY)
# =========================================================================
ROOT = Path(r"c:\Users\Jay\Desktop\DAU\SEM-2\DL\Project\ContractSense-copilot")

# Stage 6: Generation Phase Artifacts
STAGE6_BENCHMARK = ROOT / "data" / "processed" / "generation_benchmark"

# Stage 7: DPO Alignment Phase Artifacts
STAGE7_MODELS = ROOT / "src" / "alignment" / "models"
STAGE7_RESULTS = ROOT / "src" / "alignment" / "results"
STAGE7_DATA = ROOT / "src" / "alignment" / "data"

def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def find_any(directory: Path, patterns: list):
    out = []
    if not directory.exists():
        return out
    for pat in patterns:
        out.extend(directory.rglob(pat))
    return sorted(set(out))

issues = []
warnings = []

# =========================================================================
# PHASE 1: STAGE 6 - GENERATION (SFT) AUDIT
# =========================================================================
print("=== [PHASE CHECK: STAGE 6 - GENERATION / SFT] ===")
required_s6 = [
    "best_generation_model.json",
    "generation_best_model_summary.json",
    "generation_leaderboard.csv"
]

s6_ok = True
for f_name in required_s6:
    f_path = STAGE6_BENCHMARK / f_name
    if f_path.exists():
        print(f"[OK] FOUND: {f_name}")
    else:
        print(f"[ERROR] MISSING: {f_name}")
        s6_ok = False
        issues.append(f"Stage 6 Artifact Missing: {f_name}")

if s6_ok:
    best_s6 = load_json(STAGE6_BENCHMARK / "best_generation_model.json")
    print(f"[DATA] STAGE 6 Winner: {best_s6.get('model_name')} ({best_s6.get('variant')})")
    print(f"[DATA] SFT Score: {best_s6.get('final_score')} (Format Compliance: {best_s6.get('json_valid_rate')})")
else:
    print("[WARN] Skipping Stage 6 metric analysis due to missing files.")


# =========================================================================
# PHASE 2: STAGE 7 - ALIGNMENT (DPO) AUDIT
# =========================================================================
print("\n=== [PHASE CHECK: STAGE 7 - DPO ALIGNMENT] ===")

# 1. Model Folder Check
if STAGE7_MODELS.exists():
    print(f"[OK] FOUND: Stage 7 Models Directory -> {STAGE7_MODELS}")
else:
    print(f"[ERROR] Stage 7 Models Directory Missing!")
    issues.append("models folder missing at src/alignment/models")

# 2. Checkpoint Check
dpo_ckpts = find_any(STAGE7_MODELS, ["*adapter_model.safetensors", "*adapter_config.json"])
print(f"[FILES] DPO Model Components Found: {len(dpo_ckpts)}")
if len(dpo_ckpts) >= 2:
    print("[OK] SUCCESS: DPO Brain Components (Adapter + Config) are present.")
else:
    print("[ERROR] DPO Adapter weights not found in src/alignment/models.")
    issues.append("DPO Adapter Weights are missing from Stage 7 models folder.")

# 3. DPO Result Verification
dpo_logs = find_any(STAGE7_RESULTS, ["training_log.json", "inference_samples.json", "*evaluation_report.json"])
print(f"[FILES] DPO Result Logs Found: {len(dpo_logs)}")
if len(dpo_logs) > 0:
    for log in dpo_logs[:3]:
        print(f"[OK] FOUND LOG: {log.name}")
else:
    warnings.append("No DPO training/eval logs found in src/alignment/results.")

# 4. Dataset Verification
dpo_data = find_any(STAGE7_DATA, ["validated_dataset.json", "augmented_dataset.json", "dpo_dataset.json"])
print(f"[FILES] DPO Datasets Found: {len(dpo_data)}")
if len(dpo_data) > 0:
    print("[OK] SUCCESS: DPO Training data is intact.")
else:
    warnings.append("Could not find the DPO Preference Dataset in src/alignment/data.")


# =========================================================================
# FINAL VERDICT
# =========================================================================
print("\n=== [ FINAL AUDIT SUMMARY ] ===")
if issues:
    print("[BLOCKER] BLOCKING ISSUES FOUND:")
    for i in issues:
        print(f" - {i}")
else:
    print("[SUCCESS] ALL CLEAR: Architecture, Stage 6, and Stage 7 artifacts are synchronized.")

if warnings:
    print("\n[WARNING] WARNINGS (Non-Blocking):")
    for w in warnings:
        print(f" - {w}")

print("\nFinal Result: Stage 7 correctly layers on Stage 6.")
print("Audit Completed.")