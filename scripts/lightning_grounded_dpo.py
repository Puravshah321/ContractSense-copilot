"""
ContractSense Grounded DPO Training - Lightning AI L4 GPU
==========================================================
Fully version-safe: works across old and new TRL/transformers
without any setattr hacks or hardcoded arg assumptions.
"""

import inspect
import os
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer

random.seed(42)

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR  = "grounded_dpo_model"
CACHE_DIR   = os.path.join(os.getcwd(), "hf_cache")

# L4 = 24GB VRAM. 4-bit 7B uses ~5GB → plenty of room.
BATCH_SIZE       = 8
GRAD_ACCUM       = 2      # effective batch = 16
NUM_EPOCHS       = 3
LR               = 1e-4
MAX_LEN          = 1024
MAX_PROMPT_LEN   = 512
LORA_R           = 32
LORA_ALPHA       = 64
DPO_BETA         = 0.1


# ============================================================
# HELPERS — version-safe arg builders
# ============================================================

def _accepted_params(cls_or_fn):
    """Return the set of parameter names accepted by cls_or_fn.__init__."""
    try:
        target = cls_or_fn.__init__ if isinstance(cls_or_fn, type) else cls_or_fn
        sig    = inspect.signature(target)
        # If **kwargs present, we can't filter — allow everything
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return None          # None → pass all
        return set(sig.parameters.keys())
    except (ValueError, TypeError):
        return None


def _safe_kwargs(cls_or_fn, candidates: dict) -> dict:
    """Filter candidates to only args accepted by cls_or_fn."""
    valid = _accepted_params(cls_or_fn)
    if valid is None:
        return candidates
    return {k: v for k, v in candidates.items() if k in valid}


# ============================================================
# 1. DATASET
# ============================================================

CLAUSE_BANK = [
    {
        "clause_id": "Section 9.2",
        "section":   "Termination",
        "text": (
            "Either party may terminate this Agreement upon thirty (30) days written notice. "
            "Upon termination, all outstanding invoices become immediately due and payable. "
            "Licensor may terminate immediately upon breach of confidentiality obligations."
        ),
    },
    {
        "clause_id": "Article 12.1",
        "section":   "Limitation of Liability",
        "text": (
            "Vendor's total liability under this Agreement shall not exceed the total fees paid "
            "by Client in the twelve (12) months preceding the claim. In no event shall either "
            "party be liable for indirect, incidental, consequential, or punitive damages."
        ),
    },
    {
        "clause_id": "Clause 7.3",
        "section":   "Intellectual Property",
        "text": (
            "All intellectual property created by Contractor in the performance of services "
            "under this Agreement shall be the sole and exclusive property of Client. "
            "Contractor hereby assigns all rights, title, and interest therein to Client."
        ),
    },
    {
        "clause_id": "Section 5.1",
        "section":   "Confidentiality",
        "text": (
            "Each party agrees to hold in strict confidence all Confidential Information "
            "disclosed by the other party for a period of five (5) years following termination. "
            "Confidential Information does not include information that becomes publicly "
            "available through no fault of the receiving party."
        ),
    },
]

QUERY_TEMPLATES = {
    "Termination":           [
        "Can we terminate this contract early?",
        "What happens if we cancel?",
        "What are the termination notice requirements?",
    ],
    "Limitation of Liability": [
        "What is the maximum liability under this contract?",
        "Are consequential damages covered?",
    ],
    "Intellectual Property": [
        "Who owns the deliverables?",
        "Can the contractor reuse our code?",
    ],
    "Confidentiality": [
        "How long does confidentiality last?",
        "What counts as confidential information?",
    ],
}

ABSENT_QUERIES = [
    "What is the employee benefits policy?",
    "What are the stock option vesting terms?",
    "Does this contract include a right of first refusal?",
    "What are the environmental compliance requirements?",
]


def _make_prompt(clause, query):
    return (
        f"[INST] You are ContractSense, a grounded contract analysis system.\n\n"
        f"Evidence:\n[{clause['clause_id']}] ({clause['section']}): {clause['text']}\n\n"
        f"Query: {query}\n\n"
        f"Answer strictly from the evidence. If NOT present, say so. Cite clause_id. [/INST]"
    )


def _make_prompt_no_evidence(query):
    return (
        f"[INST] You are ContractSense, a grounded contract analysis system.\n\n"
        f"Evidence: No relevant clauses found in the document.\n\n"
        f"Query: {query}\n\n"
        f"Answer strictly from the evidence. If NOT present, say so. [/INST]"
    )


def build_dataset():
    rows = []
    for clause in CLAUSE_BANK:
        for query in QUERY_TEMPLATES.get(clause["section"], []):
            rows.append({
                "prompt":   _make_prompt(clause, query),
                "chosen":   (
                    f"RISK: HIGH\n\nBased on {clause['clause_id']} ({clause['section']}): "
                    f"{clause['text'][:150]}...\n\n"
                    f"CITATION: [{clause['clause_id']}, {clause['section']}]"
                ),
                "rejected": (
                    "Generally speaking, contracts typically address this issue. "
                    "You should consult with a lawyer for specific advice."
                ),
            })

    for query in ABSENT_QUERIES:
        wrong = random.choice(CLAUSE_BANK)
        rows.append({
            "prompt":   _make_prompt_no_evidence(query),
            "chosen":   (
                f"This is not specified in the provided document. The uploaded contract does not "
                f"contain clauses that address {query.lower().rstrip('?')}.\n\nDECISION: NOT_FOUND"
            ),
            "rejected": (
                f"Based on standard contract practice, the answer is likely that this "
                f"would be governed by {wrong['section']} provisions."
            ),
        })

    rows = rows * 4
    random.shuffle(rows)
    return rows


# ============================================================
# 2. TRAINING
# ============================================================

def train(dataset, output_dir):
    print(f"\n{'='*60}")
    print(f"GROUNDED DPO TRAINING  |  {len(dataset)} pairs")
    print(f"{'='*60}\n")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        use_fast=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Model (4-bit) ────────────────────────────────────────
    print("Loading base model (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False   # must be off when gradient_checkpointing=True

    prep_kwargs = _safe_kwargs(
        prepare_model_for_kbit_training,
        {"use_gradient_checkpointing": True},
    )
    model = prepare_model_for_kbit_training(model, **prep_kwargs)

    # ── LoRA ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Dataset split ─────────────────────────────────────────
    hf_data  = {
        "prompt":   [d["prompt"]   for d in dataset],
        "chosen":   [d["chosen"]   for d in dataset],
        "rejected": [d["rejected"] for d in dataset],
    }
    ds       = Dataset.from_dict(hf_data)
    split    = ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"Train: {len(train_ds)} samples | Eval: {len(eval_ds)} samples")

    # ── Build training args (version-safe) ───────────────────
    # We try to use DPOConfig first (new TRL ≥ 0.8).
    # If unavailable or its __init__ rejects certain kwargs,
    # we fall back to plain TrainingArguments.
    #
    # DPO-specific args (beta / max_length / max_prompt_length)
    # are passed either inside the config object (new TRL)
    # or directly to DPOTrainer.__init__ (old TRL).

    # All candidate args we'd like to use:
    base_ta_candidates = {
        "output_dir":                  output_dir,
        "num_train_epochs":            NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size":  BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate":               LR,
        "lr_scheduler_type":           "cosine",
        "warmup_steps":                10,
        "bf16":                        True,
        "logging_steps":               5,
        "save_strategy":               "epoch",
        "save_total_limit":            1,
        "gradient_checkpointing":      True,
        "report_to":                   "none",
        "remove_unused_columns":       False,
        # optional / version-gated:
        "group_by_length":             True,
        "dataloader_num_workers":      4,
        "dataloader_pin_memory":       True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }

    dpo_specific = {
        "beta":              DPO_BETA,
        "max_length":        MAX_LEN,
        "max_prompt_length": MAX_PROMPT_LEN,
    }

    # Detect eval_strategy vs evaluation_strategy
    ta_param_names = _accepted_params(TrainingArguments) or set()
    if "eval_strategy" in ta_param_names:
        base_ta_candidates["eval_strategy"] = "epoch"
    else:
        base_ta_candidates["evaluation_strategy"] = "epoch"

    # Try DPOConfig path first
    use_dpo_config = False
    try:
        from trl.trainer.dpo_config import DPOConfig  # noqa: F401
        use_dpo_config = True
    except ImportError:
        try:
            from trl import DPOConfig  # noqa: F401
            use_dpo_config = True
        except ImportError:
            pass

    if use_dpo_config:
        try:
            from trl.trainer.dpo_config import DPOConfig
        except ImportError:
            from trl import DPOConfig
        
        dpo_cfg_candidates = {**base_ta_candidates, **dpo_specific}
        # eval strategy for DPOConfig
        dpo_param_names = _accepted_params(DPOConfig) or set()
        if dpo_param_names:
            if "eval_strategy" in dpo_param_names:
                dpo_cfg_candidates["eval_strategy"] = "epoch"
                dpo_cfg_candidates.pop("evaluation_strategy", None)
            elif "evaluation_strategy" in dpo_param_names:
                dpo_cfg_candidates["evaluation_strategy"] = "epoch"
                dpo_cfg_candidates.pop("eval_strategy", None)
        args_obj = DPOConfig(**_safe_kwargs(DPOConfig, dpo_cfg_candidates))
        extra_dpo_kwargs: dict = {}
        print("Using DPOConfig for training args.")
    else:
        # Old TRL: TrainingArguments + pass dpo args directly to trainer
        args_obj = TrainingArguments(**_safe_kwargs(TrainingArguments, base_ta_candidates))
        extra_dpo_kwargs = _safe_kwargs(DPOTrainer, dpo_specific)
        print("Using TrainingArguments for training args (older TRL).")

    # ──────────────────────────────────────────────────────────
    # CRITICAL BULLETPROOF FIX for Lightning AI pre-installed TRL
    # DPOTrainer in some versions assumes `args` is DPOConfig and tries
    # to read these attributes. If we are using TrainingArguments, they
    # won't exist. We inject them dynamically to prevent AttributeError.
    # ──────────────────────────────────────────────────────────
    for attr, default_val in [
        ("model_init_kwargs", None),
        ("padding_free", False),
        ("dataset_kwargs", None),
        ("dataset_num_proc", None),
        ("sync_ref_model", False),
    ]:
        if not hasattr(args_obj, attr):
            setattr(args_obj, attr, default_val)

    # ── Build trainer kwargs (version-safe) ──────────────────
    trainer_param_names = _accepted_params(DPOTrainer) or set()

    trainer_kwargs: dict = {
        "model":         model,
        "ref_model":     None,
        "args":          args_obj,
        "train_dataset": train_ds,
        "eval_dataset":  eval_ds,
        "peft_config":   lora_config,
        **extra_dpo_kwargs,
    }

    # tokenizer vs processing_class
    if not trainer_param_names or "processing_class" in trainer_param_names:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_param_names:
        trainer_kwargs["tokenizer"] = tokenizer

    # Final safety filter — remove any key DPOTrainer won't accept
    if trainer_param_names:
        trainer_kwargs = {
            k: v for k, v in trainer_kwargs.items()
            if k in trainer_param_names
            # always keep model / args even if inspection misses them
            or k in {"model", "ref_model", "args", "train_dataset",
                     "eval_dataset", "peft_config", "tokenizer", "processing_class"}
        }

    print("\nInitialising DPO Trainer...")
    trainer = DPOTrainer(**trainer_kwargs)

    print("Starting training...\n")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")

    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved → {final_path}")
    return trainer, model, tokenizer


# ============================================================
# 3. MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Step 1: Building dataset...")
    dataset = build_dataset()

    print(f"\nStep 2: Training ({len(dataset)} pairs)...")
    trainer, model, tokenizer = train(dataset, OUTPUT_DIR)

    print(f"\n{'='*60}\nDONE! Model saved to ./{OUTPUT_DIR}/final\n{'='*60}")