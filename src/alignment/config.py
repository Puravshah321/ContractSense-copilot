"""
Central configuration for the DPO Alignment Pipeline.
All paths, hyperparameters, and training flags are defined here.
"""

from pathlib import Path
import os

# ──────────────────────────────────────────────
# Directory roots
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# Model paths
# ──────────────────────────────────────────────
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_MODEL_PATH = str(PROJECT_ROOT / "models" / "lora_finetuned_mistral")
DPO_OUTPUT_DIR = str(ALIGNMENT_ROOT / "models" / "dpo_aligned_model")

# ──────────────────────────────────────────────
# Dataset paths
# ──────────────────────────────────────────────
GENERATION_TRAIN_PATH = str(PROJECT_ROOT / "data" / "processed" / "generation_train.jsonl")
GENERATION_EVAL_PATH = str(PROJECT_ROOT / "data" / "processed" / "generation_eval.jsonl")
SEED_DPO_DATASET_PATH = str(ALIGNMENT_ROOT / "data" / "dpo_dataset.json")
RAW_DPO_DATASET_PATH = str(ALIGNMENT_ROOT / "data" / "dpo_dataset.json")
VALIDATED_DATASET_PATH = str(ALIGNMENT_ROOT / "data" / "validated_dataset.json")
AUGMENTED_DATASET_PATH = str(ALIGNMENT_ROOT / "data" / "augmented_dataset.json")

# ──────────────────────────────────────────────
# Result / output paths
# ──────────────────────────────────────────────
EVAL_REPORT_PATH = str(ALIGNMENT_ROOT / "results" / "evaluation" / "evaluation_report.json")
METRIC_COMPARISON_PLOT = str(ALIGNMENT_ROOT / "results" / "evaluation" / "metric_comparison.png")
FAILURE_REPORT_PATH = str(ALIGNMENT_ROOT / "results" / "failure_analysis" / "failure_report.json")
ERROR_DISTRIBUTION_PLOT = str(ALIGNMENT_ROOT / "results" / "failure_analysis" / "error_distribution.png")
COMPARISON_SAMPLES_PATH = str(ALIGNMENT_ROOT / "results" / "failure_analysis" / "comparison_samples.json")
TRAINING_LOG_PATH = str(ALIGNMENT_ROOT / "results" / "training_log.json")

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
DPO_BETA = 0.1
MAX_SEQ_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"

# ──────────────────────────────────────────────
# LoRA configuration
# ──────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ──────────────────────────────────────────────
# Large-dataset / GPU flags
# ──────────────────────────────────────────────
GRADIENT_ACCUMULATION_STEPS = 4
USE_FP16 = False
USE_BF16 = True
GRADIENT_CHECKPOINTING = True
USE_4BIT_QUANTIZATION = True
SAVE_TOTAL_LIMIT = 2
CHECKPOINT_SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"

# ──────────────────────────────────────────────
# Dataset builder settings
# ──────────────────────────────────────────────
TARGET_DATASET_SIZE = 100000
AUGMENTATION_FACTOR = 10
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
CLAUSE_TYPES = [
    "Termination", "Indemnification", "Intellectual Property",
    "Limitation of Liability", "Confidentiality", "Force Majeure",
    "Governing Law", "Non-Solicitation", "Warranty",
    "Pricing Adjustment", "Assignment", "SLA",
    "Non-Compete", "Data Protection", "Insurance",
    "Audit Rights", "Renewal", "Dispute Resolution",
]

# ──────────────────────────────────────────────
# Evaluation settings
# ──────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 200
RISK_LABEL_PATTERN = r"^RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)"
ACTION_KEYWORDS = [
    "action", "recommend", "ensure", "verify", "review",
    "negotiate", "consult", "document", "track", "set up",
    "notify", "send", "assess", "calculate", "create",
]
CITATION_PATTERN = r"CITATION:\s*\[.+?\]"
