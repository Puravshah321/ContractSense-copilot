"""
ContractSense V2 — Research-Grade DPO Training + Eval + Push
=============================================================
Single file for Lightning AI GPU (L4 / RTX / A-series friendly).
Run: python lightning_train_v2.py

Does everything:
  1. Builds 500+ pair dataset (5 categories)
  2. Trains DPO-aligned Mistral-7B
  3. Evaluates: grounding accuracy, hallucination rate, refusal accuracy, decision accuracy
  4. Pushes merged model to Hugging Face
"""

import inspect, json, os, random, time, warnings
import re
warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from trl import DPOTrainer

random.seed(42)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _bool_env(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    try:
        return torch.cuda.get_device_capability(0)
    except Exception:
        return 0, 0


def _bf16_supported():
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            pass
    major, _ = _gpu_capability()
    return major >= 8


def _train_dtype_flags():
    use_bf16 = _bf16_supported()
    # Explicit override knobs for special setups.
    if _bool_env("FORCE_FP16", False):
        use_bf16 = False
    if _bool_env("FORCE_BF16", False):
        use_bf16 = True
    return use_bf16, (not use_bf16)


def _quant_compute_dtype():
    return torch.bfloat16 if _bf16_supported() else torch.float16


def _configure_lightning_runtime_profile():
    """Set runtime defaults tuned for Lightning GPUs, including RTX P6000 profiles."""
    cpu_total = os.cpu_count() or 8
    cpu_workers = min(40, max(8, cpu_total - 8))
    os.environ.setdefault("DATALOADER_WORKERS", str(cpu_workers))
    os.environ.setdefault("OMP_NUM_THREADS", str(min(32, cpu_total)))
    os.environ.setdefault("MKL_NUM_THREADS", str(min(32, cpu_total)))

    if not torch.cuda.is_available():
        print("CUDA not available; running in CPU profile.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    os.environ.setdefault("LIGHTNING_GPU_NAME", gpu_name)
    gpu_name_upper = gpu_name.upper()
    bf16_ok = _bf16_supported()

    if "L4" in gpu_name_upper:
        os.environ.setdefault("PER_DEVICE_BATCH_SIZE", "4")
        os.environ.setdefault("GRAD_ACCUM", "4")
        os.environ.setdefault("SAVE_STEPS", "50")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print("Runtime profile: NVIDIA L4 (24GB) tuned defaults applied.")
    elif "P6000" in gpu_name_upper:
        os.environ.setdefault("PER_DEVICE_BATCH_SIZE", "3")
        os.environ.setdefault("GRAD_ACCUM", "8")
        os.environ.setdefault("SAVE_STEPS", "50")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print("Runtime profile: RTX P6000 tuned defaults applied (high-throughput, fp16-safe).")
    else:
        os.environ.setdefault("PER_DEVICE_BATCH_SIZE", "4")
        os.environ.setdefault("GRAD_ACCUM", "6")
        os.environ.setdefault("SAVE_STEPS", "75")
        print(f"Runtime profile: generic CUDA ({gpu_name}) defaults applied.")

    print(f"Precision profile: {'bf16' if bf16_ok else 'fp16'} training path.")


_configure_lightning_runtime_profile()

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR   = "grounded_dpo_model_v2"
CACHE_DIR    = os.path.join(os.getcwd(), "hf_cache")
HF_USERNAME  = "22Jay"
HF_REPO      = f"{HF_USERNAME}/ContractSense-Grounded-DPO"

BATCH_SIZE   = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "4"))  # GPU profile default, env-overridable
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))             # effective batch = 16
NUM_EPOCHS   = 4
LR           = 5e-5
MAX_LEN      = 1024
MAX_PROMPT   = 512
LORA_R       = 64
LORA_ALPHA   = 128
DPO_BETA     = 0.15
DATALOADER_WORKERS = int(os.environ.get("DATALOADER_WORKERS", str(min(4, max(2, (os.cpu_count() or 8) // 2)))))
SAVE_STEPS   = int(os.environ.get("SAVE_STEPS", "50"))
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "0") == "1"
USE_BF16, USE_FP16 = _train_dtype_flags()


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _accepted_params(cls_or_fn):
    try:
        target = cls_or_fn.__init__ if isinstance(cls_or_fn, type) else cls_or_fn
        sig = inspect.signature(target)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return None
        return set(sig.parameters.keys())
    except (ValueError, TypeError):
        return None

def _safe_kwargs(cls_or_fn, candidates):
    valid = _accepted_params(cls_or_fn)
    if valid is None:
        return candidates
    return {k: v for k, v in candidates.items() if k in valid}


def _load_4bit_base_model():
    """
    L4-safe 4-bit loader.

    Avoids `device_map="auto"` first because auto-placement can trigger low-level
    memory mapping/offload issues on some managed environments.
    """
    print("Loading base model (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_quant_compute_dtype(),
        bnb_4bit_use_double_quant=True,
    )

    load_attempts = [
        {
            "quantization_config": bnb,
            "device_map": {"": 0},
            "torch_dtype": _quant_compute_dtype(),
            "low_cpu_mem_usage": True,
            "cache_dir": CACHE_DIR,
            "attn_implementation": "sdpa",
        },
        {
            "quantization_config": bnb,
            "device_map": {"": 0},
            "torch_dtype": _quant_compute_dtype(),
            "low_cpu_mem_usage": True,
            "cache_dir": CACHE_DIR,
        },
        {
            "quantization_config": bnb,
            "device_map": "auto",
            "torch_dtype": _quant_compute_dtype(),
            "low_cpu_mem_usage": True,
            "cache_dir": CACHE_DIR,
        },
    ]

    last_error = None
    for idx, attempt in enumerate(load_attempts, 1):
        try:
            print(f"  -> Load attempt {idx} with device_map={attempt.get('device_map')}")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                **_safe_kwargs(AutoModelForCausalLM.from_pretrained, attempt),
            )
            model.config.use_cache = False
            return model
        except Exception as e:
            last_error = e
            print(f"  -> Attempt {idx} failed: {type(e).__name__}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)

    raise RuntimeError(
        "Unable to load the 4-bit base model on this environment. "
        "Clear hf_cache and retry, or verify bitsandbytes/transformers compatibility."
    ) from last_error


def _latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-")[-1])
        except ValueError:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            checkpoints.append((step, path))
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


# ══════════════════════════════════════════════════════════════
# 1. DATASET (imported from dpo_dataset_v2)
# ══════════════════════════════════════════════════════════════

def build_dataset():
    """Import and build the v2 dataset."""
    # Try importing from same directory
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from dpo_dataset_v2 import build_dataset_v2, print_stats
        rows = build_dataset_v2()
        rows = _augment_intent_dataset(rows)
        print_stats(rows)
        print(f"Intent-augmented rows: {len(rows)}")
        return rows
    except ImportError:
        print("ERROR: dpo_dataset_v2.py not found in same directory!")
        print("Please copy it next to this script on Lightning AI.")
        raise


def _extract_query_from_prompt(prompt):
    match = re.search(r"Query:\s*(.+?)\n\nRules:", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _replace_query_in_prompt(prompt, new_query):
    if "Query:" in prompt and "\n\nRules:" in prompt:
        return re.sub(r"Query:\s*(.+?)\n\nRules:", f"Query: {new_query}\n\nRules:", prompt, flags=re.DOTALL)
    return prompt


def _intent_variants(query):
    q = query.strip().rstrip("?")
    variants = [
        ("factual", f"What clause directly addresses: {q}?"),
        ("yes_no", f"Is this explicitly stated in the contract: {q}?"),
        ("analytical", f"Analyze {q} by combining all relevant clauses and summarize key obligations."),
        ("extraction", f"List all clauses relevant to: {q}."),
    ]
    # Keep unique while preserving order.
    seen = set()
    out = []
    for label, v in variants:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((label, v))
    return out


def _augment_intent_dataset(rows):
    """Expand dataset with intent-diverse prompts and stronger contrastive negatives."""
    augmented = list(rows)
    # Use a stable subset to avoid exploding dataset size.
    base_rows = [r for r in rows if r.get("category", "").startswith("A_")][:220]

    for row in base_rows:
        q = _extract_query_from_prompt(row.get("prompt", ""))
        if not q:
            continue
        for intent_label, qv in _intent_variants(q):
            prompt = _replace_query_in_prompt(row["prompt"], qv)
            chosen = row["chosen"]
            rejected = row["rejected"]

            if intent_label == "analytical":
                if "DECISION:" not in chosen:
                    chosen = chosen + "\n\nDECISION: ANSWER"
                chosen = (
                    "Structured findings from contract evidence:\n"
                    f"- Finding: {chosen}\n"
                    "- Impact: Contractual obligations identified from cited clauses.\n"
                    "- Risk level: MEDIUM"
                )
                rejected = (
                    "General legal commentary without concept grouping or clause-grounded synthesis."
                )
            elif intent_label == "extraction":
                chosen = (
                    "Extracted relevant clauses from evidence:\n"
                    + re.sub(r"\s+", " ", chosen)[:260]
                )
                rejected = "Narrative summary without listing concrete clauses."

            augmented.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "category": f"F_intent_{intent_label}",
                }
            )

        # Add explicit concept-confusion negative for alignment training.
        augmented.append(
            {
                "prompt": _replace_query_in_prompt(row["prompt"], f"What are the financial commitments related to: {q}?"),
                "chosen": row["chosen"],
                "rejected": (
                    "This answer focuses on audit/process clauses and ignores direct financial obligations, fees, and payment terms."
                ),
                "category": "G_concept_alignment_negative",
            }
        )

    return augmented


# ══════════════════════════════════════════════════════════════
# 2. TRAINING
# ══════════════════════════════════════════════════════════════

def train_dpo(dataset, output_dir):
    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"GROUNDED DPO V2 TRAINING | {total} pairs | {NUM_EPOCHS} epochs")
    print(f"LoRA r={LORA_R} alpha={LORA_ALPHA} | beta={DPO_BETA} | lr={LR}")
    print(f"{'='*60}\n")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, cache_dir=CACHE_DIR, use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Model (4-bit NF4 for L4 24GB)
    model = _load_4bit_base_model()

    prep_kw = _safe_kwargs(prepare_model_for_kbit_training, {"use_gradient_checkpointing": True})
    model = prepare_model_for_kbit_training(model, **prep_kw)

    # ── LoRA (wider rank for better truthfulness learning)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    # ── Dataset split
    hf = Dataset.from_dict({
        "prompt":   [d["prompt"]   for d in dataset],
        "chosen":   [d["chosen"]   for d in dataset],
        "rejected": [d["rejected"] for d in dataset],
    })
    split = hf.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Training args (version-safe)
    base_args = {
        "output_dir": output_dir, "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE, "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM, "learning_rate": LR,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.05,
        "bf16": USE_BF16, "fp16": USE_FP16,
        "logging_steps": 5, "save_strategy": "steps", "save_steps": SAVE_STEPS, "save_total_limit": 2,
        "gradient_checkpointing": True, "report_to": "none",
        "remove_unused_columns": False, "group_by_length": True,
        "optim": "paged_adamw_8bit",
        "dataloader_num_workers": DATALOADER_WORKERS, "dataloader_pin_memory": True,
        "dataloader_persistent_workers": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    dpo_args = {"beta": DPO_BETA, "max_length": MAX_LEN, "max_prompt_length": MAX_PROMPT}

    ta_params = _accepted_params(TrainingArguments) or set()
    base_args["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "epoch"

    # Try DPOConfig first
    use_dpo_config = False
    DPOConfigClass = None
    try:
        from trl.trainer.dpo_config import DPOConfig as _DC
        DPOConfigClass = _DC; use_dpo_config = True
    except ImportError:
        try:
            from trl import DPOConfig as _DC
            DPOConfigClass = _DC; use_dpo_config = True
        except ImportError:
            pass

    if use_dpo_config:
        merged = {**base_args, **dpo_args}
        dp = _accepted_params(DPOConfigClass) or set()
        if dp:
            if "eval_strategy" in dp:
                merged["eval_strategy"] = "epoch"; merged.pop("evaluation_strategy", None)
            elif "evaluation_strategy" in dp:
                merged["evaluation_strategy"] = "epoch"; merged.pop("eval_strategy", None)
        args_obj = DPOConfigClass(**_safe_kwargs(DPOConfigClass, merged))
        extra_dpo = {}
        print("Using DPOConfig.")
    else:
        args_obj = TrainingArguments(**_safe_kwargs(TrainingArguments, base_args))
        extra_dpo = _safe_kwargs(DPOTrainer, dpo_args)
        print("Using TrainingArguments (older TRL).")

    # Bulletproof attribute injection
    for attr, val in [
        ("model_init_kwargs", None), ("padding_free", False),
        ("dataset_kwargs", None), ("dataset_num_proc", None), ("sync_ref_model", False),
    ]:
        if not hasattr(args_obj, attr):
            setattr(args_obj, attr, val)

    # ── Trainer kwargs
    tp = _accepted_params(DPOTrainer) or set()
    tkw = {
        "model": model, "ref_model": None, "args": args_obj,
        "train_dataset": train_ds, "eval_dataset": eval_ds,
        "peft_config": lora_config, **extra_dpo,
    }
    if not tp or "processing_class" in tp:
        tkw["processing_class"] = tokenizer
    elif "tokenizer" in tp:
        tkw["tokenizer"] = tokenizer

    if tp:
        keep = {"model","ref_model","args","train_dataset","eval_dataset","peft_config","tokenizer","processing_class"}
        tkw = {k: v for k, v in tkw.items() if k in tp or k in keep}

    print("\nInitialising DPO Trainer...")
    trainer = DPOTrainer(**tkw)

    print("Starting training...\n")
    t0 = time.time()
    resume_checkpoint = _latest_checkpoint(output_dir) if RESUME_FROM_CHECKPOINT else None
    existing_checkpoint = _latest_checkpoint(output_dir)
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    elif existing_checkpoint:
        print(
            f"Found existing checkpoint {existing_checkpoint}, but RESUME_FROM_CHECKPOINT=0. "
            "Starting a fresh training run."
        )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    final = os.path.join(output_dir, "final")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"Model saved → {final}")
    return trainer, model, tokenizer


# ══════════════════════════════════════════════════════════════
# 3. EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, tokenizer, dataset, max_samples=60):
    """Run evaluation metrics on the trained model."""
    import re

    print(f"\n{'='*60}")
    print("EVALUATION — Grounding & Decision Accuracy")
    print(f"{'='*60}\n")

    # Sample from each category
    by_cat = {}
    for d in dataset:
        cat = d.get("category", "unknown")
        by_cat.setdefault(cat, []).append(d)

    samples = []
    per_cat = max(5, max_samples // max(len(by_cat), 1))
    for cat, items in by_cat.items():
        samples.extend(random.sample(items, min(per_cat, len(items))))
    random.shuffle(samples)
    samples = samples[:max_samples]

    metrics = {
        "total": 0, "correct_decision": 0,
        "hallucination_caught": 0, "hallucination_total": 0,
        "not_found_correct": 0, "not_found_total": 0,
        "grounded_correct": 0, "grounded_total": 0,
    }

    model.eval()
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        cat = sample.get("category", "unknown")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, temperature=0.1,
                do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        metrics["total"] += 1

        # Decision accuracy
        expected_decision = "NOT_FOUND" if "NOT_FOUND" in chosen else ("ESCALATE" if "ESCALATE" in chosen else "ANSWER")
        got_decision = "NOT_FOUND" if "NOT_FOUND" in response else ("ESCALATE" if "ESCALATE" in response else "ANSWER")

        if expected_decision == got_decision:
            metrics["correct_decision"] += 1

        # Hallucination detection
        if cat.startswith("B_"):
            metrics["hallucination_total"] += 1
            if "NOT_FOUND" in response or "not present" in response.lower() or "does not" in response.lower():
                metrics["hallucination_caught"] += 1

        # Refusal accuracy
        if cat.startswith("C_"):
            metrics["not_found_total"] += 1
            if "NOT_FOUND" in response or "not specified" in response.lower():
                metrics["not_found_correct"] += 1

        # Grounding (does it cite the right clause?)
        if cat.startswith("A_"):
            metrics["grounded_total"] += 1
            clause_match = re.search(r"(Section|Article|Clause)\s+[\d\.]+", chosen)
            if clause_match and clause_match.group() in response:
                metrics["grounded_correct"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(samples)}...")

    # Compute rates
    results = {
        "total_samples": metrics["total"],
        "decision_accuracy": metrics["correct_decision"] / max(metrics["total"], 1),
        "hallucination_rate": 1.0 - (metrics["hallucination_caught"] / max(metrics["hallucination_total"], 1)),
        "refusal_accuracy": metrics["not_found_correct"] / max(metrics["not_found_total"], 1),
        "not_found_accuracy": metrics["not_found_correct"] / max(metrics["not_found_total"], 1),
        "grounding_accuracy": metrics["grounded_correct"] / max(metrics["grounded_total"], 1),
    }

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Samples evaluated:    {results['total_samples']}")
    print(f"  Decision Accuracy:    {results['decision_accuracy']:.1%}")
    print(f"  Hallucination Rate:   {results['hallucination_rate']:.1%} (lower is better)")
    print(f"  Refusal Accuracy:     {results['refusal_accuracy']:.1%}")
    print(f"  Grounding Accuracy:   {results['grounding_accuracy']:.1%}")
    print(f"{'='*50}\n")

    return results


# ══════════════════════════════════════════════════════════════
# 4. PUSH TO HUGGING FACE
# ══════════════════════════════════════════════════════════════

def generate_eval_charts(eval_results, output_dir):
    """Save PNG charts after training/evaluation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart generation.")
        return []

    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    labels = ["decision", "not_found", "grounding"]
    values = [
        eval_results.get("decision_accuracy", 0.0),
        eval_results.get("refusal_accuracy", 0.0),
        eval_results.get("grounding_accuracy", 0.0),
    ]
    colors = ["#2563EB", "#DC2626", "#059669"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("ContractSense Grounded DPO Evaluation")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.03, f"{value:.0%}", ha="center", fontweight="bold")
    fig.tight_layout()
    path1 = os.path.join(image_dir, "dpo_eval_metrics.png")
    fig.savefig(path1, dpi=200)
    plt.close(fig)

    hallucination = eval_results.get("hallucination_rate", 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["hallucination_rate"], [hallucination], color="#EA580C")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("DPO Hallucination Rate")
    ax.text(0, hallucination + 0.03, f"{hallucination:.0%}", ha="center", fontweight="bold")
    fig.tight_layout()
    path2 = os.path.join(image_dir, "dpo_hallucination_rate.png")
    fig.savefig(path2, dpi=200)
    plt.close(fig)

    print(f"Saved charts -> {path1}, {path2}")
    return [path1, path2]


def push_to_hf(adapter_dir):
    print(f"\n{'='*60}")
    print(f"PUSHING TO HUGGING FACE → {HF_REPO}")
    print(f"{'='*60}\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading base model for merge...")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.float16, device_map="cuda:0",
            low_cpu_mem_usage=True, cache_dir=CACHE_DIR,
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="cuda:0",
            low_cpu_mem_usage=True, cache_dir=CACHE_DIR,
        )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)

    print("Merging LoRA adapters...")
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft_model.merge_and_unload()

    print("Uploading to Hugging Face Hub...")
    merged.push_to_hub(HF_REPO)
    tok.push_to_hub(HF_REPO)

    print(f"\n✅ PUSHED! → https://huggingface.co/{HF_REPO}")


# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════

def _write_run_summary(output_dir, summary):
    summary_path = os.path.join(output_dir, "run_summary.md")
    lines = [
        "# ContractSense Lightning Run Summary",
        "",
        f"- Dataset pairs: {summary.get('dataset_size')}",
        f"- Output dir: `{summary.get('output_dir')}`",
        f"- HF repo: `{summary.get('hf_repo')}`",
        f"- Eval results: `{summary.get('eval_path')}`",
        f"- Precision metrics: `{summary.get('precision_path')}`",
        "",
        "## DPO Metrics",
    ]
    eval_results = summary.get("eval_results", {})
    for key in [
        "decision_accuracy", "hallucination_rate", "not_found_accuracy", "grounding_accuracy",
        "intent_alignment_accuracy", "structure_match_accuracy", "concept_purity_score",
    ]:
        if key in eval_results:
            lines.append(f"- {key}: {eval_results[key]:.4f}")

    comparison = summary.get("comparison", {})
    if comparison:
        lines.extend(["", "## Model Comparison"])
        for model_name in ["baseline", "generator", "dpo"]:
            metrics = comparison.get(model_name, {})
            if not metrics:
                continue
            pretty = ", ".join(
                f"{metric}={value:.4f}" for metric, value in metrics.items() if isinstance(value, (int, float))
            )
            lines.append(f"- {model_name}: {pretty}")

    lines.extend(["", "## Artifacts"])
    for path in summary.get("artifact_paths", []):
        lines.append(f"- `{path}`")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Run summary saved -> {summary_path}")
    return summary_path


def run_full_pipeline(output_dir=OUTPUT_DIR, push=True):
    os.makedirs(output_dir, exist_ok=True)
    artifact_paths = []

    print("Step 1: Building research-grade DPO dataset...")
    dataset = build_dataset()

    print(f"\nStep 2: Training DPO ({len(dataset)} pairs)...")
    trainer, model, tokenizer = train_dpo(dataset, output_dir)

    print("\nStep 3: Running evaluation...")
    eval_samples = random.sample(dataset, min(60, len(dataset)))
    eval_results = evaluate_model(model, tokenizer, eval_samples)

    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    artifact_paths.append(eval_path)
    print(f"Eval results saved -> {eval_path}")

    print("\nStep 3B: Generating evaluation charts...")
    artifact_paths.extend(generate_eval_charts(eval_results, output_dir))

    precision_metrics = None
    precision_rows = None
    precision_path = None
    try:
        from evaluate_precision_pipeline import evaluate as evaluate_precision_pipeline
        from evaluate_precision_pipeline import write_outputs as write_precision_outputs
        precision_metrics, precision_rows = evaluate_precision_pipeline()
        precision_output_dir = os.path.join(output_dir, "images")
        write_precision_outputs(precision_metrics, precision_rows, precision_output_dir)
        precision_path = os.path.join(output_dir, "precision_pipeline_metrics.json")
        with open(precision_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": precision_metrics, "cases": precision_rows}, f, indent=2)
        artifact_paths.append(precision_path)
        print(f"Precision pipeline metrics saved -> {precision_path}")
    except Exception as e:
        print(f"Precision pipeline evaluation skipped: {e}")

    comparison = {}
    comparison_cases = {}
    try:
        from evaluate_model_comparison import compare_models, write_comparison_outputs
        comparison, comparison_cases = compare_models(eval_results)
        comparison_paths = write_comparison_outputs(comparison, comparison_cases, os.path.join(output_dir, "images"))
        artifact_paths.extend(comparison_paths)
        print("Baseline/Generator/DPO comparison saved:")
        for path in comparison_paths:
            print(f"  -> {path}")
    except Exception as e:
        print(f"Model comparison evaluation skipped: {e}")

    final_path = os.path.join(output_dir, "final")
    if push:
        print("\nStep 4: Pushing to Hugging Face...")
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        time.sleep(2)
        push_to_hf(final_path)

    summary = {
        "dataset_size": len(dataset),
        "output_dir": output_dir,
        "hf_repo": HF_REPO,
        "eval_path": eval_path,
        "eval_results": eval_results,
        "precision_path": precision_path,
        "precision_metrics": precision_metrics,
        "comparison": comparison,
        "comparison_cases": comparison_cases,
        "artifact_paths": artifact_paths,
        "pushed_to_hub": bool(push),
    }
    summary["summary_path"] = _write_run_summary(output_dir, summary)
    return summary


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Dataset
    print("Step 1: Building research-grade DPO dataset...")
    dataset = build_dataset()

    # Step 2: Train
    print(f"\nStep 2: Training DPO ({len(dataset)} pairs)...")
    trainer, model, tokenizer = train_dpo(dataset, OUTPUT_DIR)

    # Step 3: Evaluate
    print("\nStep 3: Running evaluation...")
    eval_samples = random.sample(dataset, min(60, len(dataset)))
    eval_results = evaluate_model(model, tokenizer, eval_samples)

    # Save eval results
    eval_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Eval results saved → {eval_path}")

    print("\nStep 3B: Generating evaluation charts...")
    generate_eval_charts(eval_results, OUTPUT_DIR)

    try:
        from evaluate_precision_pipeline import evaluate as evaluate_precision_pipeline
        from evaluate_precision_pipeline import write_outputs as write_precision_outputs
        precision_metrics, precision_rows = evaluate_precision_pipeline()
        write_precision_outputs(precision_metrics, precision_rows, os.path.join(OUTPUT_DIR, "images"))
        precision_path = os.path.join(OUTPUT_DIR, "precision_pipeline_metrics.json")
        with open(precision_path, "w") as f:
            json.dump({"metrics": precision_metrics, "cases": precision_rows}, f, indent=2)
        print(f"Precision pipeline metrics saved -> {precision_path}")
    except Exception as e:
        print(f"Precision pipeline evaluation skipped: {e}")

    try:
        from evaluate_model_comparison import compare_models, write_comparison_outputs
        comparison, comparison_cases = compare_models(eval_results)
        comparison_paths = write_comparison_outputs(comparison, comparison_cases, os.path.join(OUTPUT_DIR, "images"))
        print("Baseline/Generator/DPO comparison saved:")
        for path in comparison_paths:
            print(f"  -> {path}")
    except Exception as e:
        print(f"Model comparison evaluation skipped: {e}")

    # Step 4: Push
    print("\nStep 4: Pushing to Hugging Face...")
    final_path = os.path.join(OUTPUT_DIR, "final")

    # Free GPU memory before push
    del trainer, model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(2)

    push_to_hf(final_path)

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"  Model: https://huggingface.co/{HF_REPO}")
    print(f"  Eval:  {eval_path}")
    print(f"{'='*60}")
