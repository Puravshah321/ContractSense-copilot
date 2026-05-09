"""
ContractSense V3 — Reasoning-Focused DPO Training + Eval + Push
================================================================
Upload to Lightning AI RTX 6000 and run:
    python lightning_train_v3.py

Key differences from v2:
  - Imports from dpo_dataset_v3.py (reasoning-focused pairs)
  - Improved evaluation: adds reasoning accuracy & analytical completeness
  - Uses RTX 6000 (48GB) — loads in bfloat16 instead of 4-bit for better quality
"""
import inspect, json, os, random, time, warnings
warnings.filterwarnings("ignore")
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

random.seed(42)

BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR  = "grounded_dpo_model_v3"
CACHE_DIR   = os.path.join(os.getcwd(), "hf_cache")
HF_REPO     = "22Jay/ContractSense-Grounded-DPO"

# RTX 6000 48GB → can use larger batch, bfloat16
BATCH_SIZE  = 4
GRAD_ACCUM  = 4
NUM_EPOCHS  = 3
LR          = 3e-5
MAX_LEN     = 1536
MAX_PROMPT  = 768
LORA_R      = 64
LORA_ALPHA  = 128
DPO_BETA    = 0.10   # Lower beta = more reasoning flexibility


def _accepted_params(cls_or_fn):
    try:
        target = cls_or_fn.__init__ if isinstance(cls_or_fn, type) else cls_or_fn
        sig = inspect.signature(target)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return None
        return set(sig.parameters.keys())
    except Exception:
        return None

def _safe_kwargs(cls_or_fn, candidates):
    valid = _accepted_params(cls_or_fn)
    return candidates if valid is None else {k: v for k, v in candidates.items() if k in valid}


def build_dataset():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dpo_dataset_v3 import build_dataset_v3, print_stats
    rows = build_dataset_v3()
    print_stats(rows)
    return rows


def train_dpo(dataset, output_dir):
    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"CONTRACTSENSE V3 DPO TRAINING | {total} pairs | {NUM_EPOCHS} epochs")
    print(f"LoRA r={LORA_R} alpha={LORA_ALPHA} | beta={DPO_BETA} | lr={LR}")
    print(f"{'='*60}\n")

    os.makedirs(CACHE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, cache_dir=CACHE_DIR, use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # RTX 6000 48GB: use 4-bit NF4 (same as before, proven stable)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, cache_dir=CACHE_DIR,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, **_safe_kwargs(prepare_model_for_kbit_training, {"use_gradient_checkpointing": True}))

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    hf = Dataset.from_dict({
        "prompt":   [d["prompt"]   for d in dataset],
        "chosen":   [d["chosen"]   for d in dataset],
        "rejected": [d["rejected"] for d in dataset],
    })
    split = hf.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    base_args = {
        "output_dir": output_dir, "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LR, "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05, "bf16": True,
        "logging_steps": 5, "save_strategy": "epoch", "save_total_limit": 1,
        "gradient_checkpointing": True, "report_to": "none",
        "remove_unused_columns": False, "group_by_length": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    dpo_args = {"beta": DPO_BETA, "max_length": MAX_LEN, "max_prompt_length": MAX_PROMPT}

    ta_params = _accepted_params(TrainingArguments) or set()
    base_args["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "epoch"

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
        args_obj = DPOConfigClass(**_safe_kwargs(DPOConfigClass, merged))
        extra_dpo = {}
    else:
        args_obj = TrainingArguments(**_safe_kwargs(TrainingArguments, base_args))
        extra_dpo = _safe_kwargs(DPOTrainer, dpo_args)

    for attr, val in [("model_init_kwargs", None), ("padding_free", False),
                      ("dataset_kwargs", None), ("sync_ref_model", False)]:
        if not hasattr(args_obj, attr):
            setattr(args_obj, attr, val)

    tp = _accepted_params(DPOTrainer) or set()
    tkw = {"model": model, "ref_model": None, "args": args_obj,
           "train_dataset": train_ds, "eval_dataset": eval_ds,
           "peft_config": lora_config, **extra_dpo}
    if not tp or "processing_class" in tp:
        tkw["processing_class"] = tokenizer
    elif "tokenizer" in tp:
        tkw["tokenizer"] = tokenizer

    trainer = DPOTrainer(**{k: v for k, v in tkw.items() if not tp or k in tp or k in {"model","ref_model","args","train_dataset","eval_dataset","peft_config","tokenizer","processing_class"}})

    print("Starting training...\n")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    final = os.path.join(output_dir, "final")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"Model saved → {final}")
    return trainer, model, tokenizer


def evaluate_model(model, tokenizer, dataset, max_samples=60):
    """
    Improvement #12: Extended evaluation.
    Metrics: decision accuracy, hallucination rate, refusal accuracy,
             grounding accuracy, reasoning depth, analytical completeness.
    """
    import re
    print(f"\n{'='*55}")
    print("EVALUATION v3 — Reasoning + Grounding Metrics")
    print(f"{'='*55}\n")

    by_cat = {}
    for d in dataset:
        by_cat.setdefault(d.get("category", "unknown"), []).append(d)

    samples = []
    per_cat = max(5, max_samples // max(len(by_cat), 1))
    for cat, items in by_cat.items():
        samples.extend(random.sample(items, min(per_cat, len(items))))
    random.shuffle(samples)
    samples = samples[:max_samples]

    m = {"total": 0, "correct_decision": 0,
         "hallucination_caught": 0, "hallucination_total": 0,
         "not_found_correct": 0, "not_found_total": 0,
         "grounded_correct": 0, "grounded_total": 0,
         "reasoning_sections": 0, "reasoning_total": 0,
         "analytical_complete": 0, "analytical_total": 0}

    model.eval()
    for i, sample in enumerate(samples):
        inputs = tokenizer(sample["prompt"], return_tensors="pt", truncation=True, max_length=MAX_PROMPT)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=300, temperature=0.1,
                do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cat = sample.get("category", "unknown")
        chosen = sample["chosen"]

        m["total"] += 1
        expected = "NOT_FOUND" if "NOT_FOUND" in chosen else ("ESCALATE" if "ESCALATE" in chosen else "ANSWER")
        got = "NOT_FOUND" if "NOT_FOUND" in response else ("ESCALATE" if "ESCALATE" in response else "ANSWER")
        if expected == got:
            m["correct_decision"] += 1

        if cat.startswith("D_"):
            m["hallucination_total"] += 1
            if "incorrect" in response.lower() or "NOT_FOUND" in response or "incorrect" in response.lower():
                m["hallucination_caught"] += 1

        if cat.startswith("C_"):
            m["not_found_total"] += 1
            if "NOT_FOUND" in response or "not specified" in response.lower():
                m["not_found_correct"] += 1

        if cat.startswith("A_"):
            m["grounded_total"] += 1
            clause_match = re.search(r"(Section|Article|Clause)\s+[\d\.]+", chosen)
            if clause_match and clause_match.group() in response:
                m["grounded_correct"] += 1

        # Reasoning sections check (Improvement #12)
        if cat.startswith("A_") or cat.startswith("B_"):
            m["reasoning_total"] += 1
            has_explicit = "directly supported" in response.lower() or "explicitly" in response.lower()
            has_implied = "implied" in response.lower() or "interpretation" in response.lower()
            if has_explicit or has_implied:
                m["reasoning_sections"] += 1

        # Analytical completeness check
        if cat.startswith("B_"):
            m["analytical_total"] += 1
            has_multi_cite = response.count("CITATION") >= 1 or (response.count("[Section") + response.count("[Article") + response.count("[Clause")) >= 2
            if has_multi_cite:
                m["analytical_complete"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(samples)}...")

    results = {
        "total_samples": m["total"],
        "decision_accuracy":      m["correct_decision"] / max(m["total"], 1),
        "hallucination_catch_rate": m["hallucination_caught"] / max(m["hallucination_total"], 1),
        "refusal_accuracy":       m["not_found_correct"] / max(m["not_found_total"], 1),
        "grounding_accuracy":     m["grounded_correct"] / max(m["grounded_total"], 1),
        "reasoning_depth_rate":   m["reasoning_sections"] / max(m["reasoning_total"], 1),
        "analytical_completeness": m["analytical_complete"] / max(m["analytical_total"], 1),
    }

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS v3")
    print(f"{'='*50}")
    for k, v in results.items():
        if k == "total_samples":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.1%}")
    print(f"{'='*50}\n")
    return results


def push_to_hf(adapter_dir):
    print(f"\n{'='*55}")
    print(f"PUSHING TO HUGGING FACE → {HF_REPO}")
    print(f"{'='*55}\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.push_to_hub(HF_REPO)
    tok.push_to_hub(HF_REPO)
    print(f"\n✅ PUSHED → https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Step 1: Building reasoning-focused DPO v3 dataset...")
    dataset = build_dataset()

    print(f"\nStep 2: Training ({len(dataset)} pairs)...")
    trainer, model, tokenizer = train_dpo(dataset, OUTPUT_DIR)

    print("\nStep 3: Running extended evaluation (Improvement #12)...")
    eval_samples = random.sample(dataset, min(60, len(dataset)))
    results = evaluate_model(model, tokenizer, eval_samples)

    eval_path = os.path.join(OUTPUT_DIR, "eval_results_v3.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {eval_path}")

    print("\nStep 4: Pushing to HuggingFace...")
    del trainer, model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(3)
    push_to_hf(os.path.join(OUTPUT_DIR, "final"))

    print(f"\n{'='*55}")
    print("ALL DONE!")
    print(f"  Model: https://huggingface.co/{HF_REPO}")
    print(f"  Eval:  {eval_path}")
    print(f"{'='*55}")
