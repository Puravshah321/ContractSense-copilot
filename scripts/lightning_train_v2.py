"""
ContractSense V2 — Research-Grade DPO Training + Eval + Push
=============================================================
Single file for Lightning AI L4 GPU.
Run: python lightning_train_v2.py

Does everything:
  1. Builds 500+ pair dataset (5 categories)
  2. Trains DPO-aligned Mistral-7B
  3. Evaluates: grounding accuracy, hallucination rate, refusal accuracy, decision accuracy
  4. Pushes merged model to Hugging Face
"""

import inspect, json, os, random, time, warnings
warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,
)
from trl import DPOTrainer

random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR   = "grounded_dpo_model_v2"
CACHE_DIR    = os.path.join(os.getcwd(), "hf_cache")
HF_USERNAME  = "22Jay"
HF_REPO      = f"{HF_USERNAME}/ContractSense-Grounded-DPO"

BATCH_SIZE   = int(os.environ.get("PER_DEVICE_BATCH_SIZE", "4"))  # L4 24GB friendly
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM", "4"))             # effective batch = 16
NUM_EPOCHS   = 4
LR           = 5e-5
MAX_LEN      = 1024
MAX_PROMPT   = 512
LORA_R       = 64
LORA_ALPHA   = 128
DPO_BETA     = 0.15
DATALOADER_WORKERS = min(4, max(2, (os.cpu_count() or 8) // 2))   # Lightning 8 CPU / 24GB RAM friendly
SAVE_STEPS   = int(os.environ.get("SAVE_STEPS", "50"))
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "0") == "1"


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
        print_stats(rows)
        return rows
    except ImportError:
        print("ERROR: dpo_dataset_v2.py not found in same directory!")
        print("Please copy it next to this script on Lightning AI.")
        raise


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
    print("Loading base model (4-bit NF4)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, cache_dir=CACHE_DIR, attn_implementation="sdpa",
    )
    model.config.use_cache = False

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
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.05, "bf16": True,
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
