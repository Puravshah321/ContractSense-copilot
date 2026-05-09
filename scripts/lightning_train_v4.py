"""
ContractSense V4 — DPO Training (Lightning AI, Max GPU Utilization)
====================================================================
Upload ONLY these 2 files to Lightning AI and run:
    python lightning_train_v4.py

GPU Optimizations:
  - flash_attention_2 when available
  - bf16 + tf32 matmuls
  - DataLoader num_workers=4, pin_memory=True
  - Group-by-length batching → less padding waste
  - Gradient checkpointing with reentrant=False
  - torch.compile() when torch >= 2.0
  - Larger batch + grad accum for L4/A10/RTX6000
  - Empty cache + gc before HF push
"""
import gc, inspect, json, os, random, time, warnings
warnings.filterwarnings("ignore")

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from trl import DPOTrainer

random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on Ampere+
torch.backends.cudnn.allow_tf32      = True

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "grounded_dpo_model_v4"
CACHE_DIR  = os.path.join(os.getcwd(), "hf_cache")
HF_REPO    = "22Jay/ContractSense-Grounded-DPO"

# Tuned for Lightning AI L4 (24 GB) or RTX6000 (48 GB)
# Effective batch = BATCH_SIZE × GRAD_ACCUM = 32 samples/step
BATCH_SIZE  = 8    # per-device; increase to 12 on RTX6000
GRAD_ACCUM  = 4
NUM_EPOCHS  = 3
LR          = 2e-5
MAX_LEN     = 1024
MAX_PROMPT  = 512
LORA_R      = 64
LORA_ALPHA  = 128
DPO_BETA    = 0.10

# ─── Helpers ──────────────────────────────────────────────────────────────────

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

def _safe(cls_or_fn, candidates):
    valid = _accepted_params(cls_or_fn)
    return candidates if valid is None else {k: v for k, v in candidates.items() if k in valid}


# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_dataset():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dpo_dataset_v4 import build_dataset_v4, print_stats
    rows = build_dataset_v4()
    print_stats(rows)
    return rows


# ─── Training ─────────────────────────────────────────────────────────────────

def train_dpo(dataset, output_dir):
    total = len(dataset)
    print(f"\n{'='*60}")
    print(f"ContractSense V4 DPO | {total} pairs | {NUM_EPOCHS} epochs")
    print(f"LoRA r={LORA_R} α={LORA_ALPHA} | β={DPO_BETA} | lr={LR}")
    print(f"Batch={BATCH_SIZE} GradAccum={GRAD_ACCUM} | MaxLen={MAX_LEN}")
    print(f"{'='*60}\n")

    os.makedirs(CACHE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, cache_dir=CACHE_DIR, use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 4-bit NF4 — stable on all Lightning GPU tiers
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Try flash-attention-2 for throughput boost; fall back silently
    attn_impl = "eager"
    try:
        import flash_attn  # noqa
        attn_impl = "flash_attention_2"
        print("✅ flash_attention_2 enabled")
    except ImportError:
        print("ℹ️  flash_attn not installed — using eager attention")

    model_kwargs = dict(
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    if attn_impl == "flash_attention_2":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        **_safe(prepare_model_for_kbit_training, {"use_gradient_checkpointing": True}),
    )

    # Removed torch.compile as it conflicts with 4-bit PEFT training in the current transformers version

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    hf_ds = Dataset.from_dict({
        "prompt":   [d["prompt"]   for d in dataset],
        "chosen":   [d["chosen"]   for d in dataset],
        "rejected": [d["rejected"] for d in dataset],
    })
    split = hf_ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    base_args = {
        "output_dir": output_dir,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LR,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "tf32": True,                    # extra speed on Ampere
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "remove_unused_columns": False,
        "group_by_length": True,         # reduces padding → faster
        "dataloader_num_workers": 4,     # parallel data loading
        "dataloader_pin_memory": True,   # pinned memory → faster GPU transfer
        "optim": "adamw_torch_fused",    # fused optimizer when available
    }
    dpo_args = {
        "beta": DPO_BETA,
        "max_length": MAX_LEN,
        "max_prompt_length": MAX_PROMPT,
    }

    ta_params = _accepted_params(TrainingArguments) or set()
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    base_args[eval_key] = "epoch"

    # DPOConfig path (newer TRL)
    use_dpo_config = False
    DPOConfigClass = None
    for path in ("trl.trainer.dpo_config.DPOConfig", "trl.DPOConfig"):
        try:
            mod, cls = path.rsplit(".", 1)
            import importlib
            DPOConfigClass = getattr(importlib.import_module(mod), cls)
            use_dpo_config = True
            break
        except Exception:
            pass

    if use_dpo_config:
        merged = {**base_args, **dpo_args}
        args_obj = DPOConfigClass(**_safe(DPOConfigClass, merged))
        extra_dpo = {}
    else:
        args_obj = TrainingArguments(**_safe(TrainingArguments, base_args))
        extra_dpo = _safe(DPOTrainer, dpo_args)

    # Ensure compat attrs exist
    for attr, val in [("model_init_kwargs", None), ("padding_free", False),
                      ("dataset_kwargs", None), ("sync_ref_model", False)]:
        if not hasattr(args_obj, attr):
            setattr(args_obj, attr, val)

    tp = _accepted_params(DPOTrainer) or set()
    tkw = {
        "model": model, "ref_model": None, "args": args_obj,
        "train_dataset": train_ds, "eval_dataset": eval_ds,
        "peft_config": lora_cfg, **extra_dpo,
    }
    if not tp or "processing_class" in tp:
        tkw["processing_class"] = tokenizer
    elif "tokenizer" in tp:
        tkw["tokenizer"] = tokenizer

    trainer = DPOTrainer(**{k: v for k, v in tkw.items()
                            if not tp or k in tp or
                            k in {"model","ref_model","args","train_dataset",
                                  "eval_dataset","peft_config","tokenizer","processing_class"}})

    print("Starting training...\n")
    t0 = time.time()
    trainer.train()
    print(f"\nTraining complete in {(time.time()-t0)/60:.1f} min")

    final = os.path.join(output_dir, "final")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"Model saved → {final}")
    return trainer, model, tokenizer


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, dataset, max_samples=80):
    import re
    print(f"\n{'='*55}")
    print("EVALUATION V4")
    print(f"{'='*55}\n")

    by_cat = {}
    for d in dataset:
        by_cat.setdefault(d.get("category", "unknown"), []).append(d)

    per_cat = max(6, max_samples // max(len(by_cat), 1))
    samples = []
    for cat, items in by_cat.items():
        samples.extend(random.sample(items, min(per_cat, len(items))))
    random.shuffle(samples)
    samples = samples[:max_samples]

    m = {k: 0 for k in [
        "total", "correct_decision",
        "hallucination_caught", "hallucination_total",
        "not_found_correct", "not_found_total",
        "grounded_correct", "grounded_total",
        "adversarial_correct", "adversarial_total",
        "multihop_correct", "multihop_total",
    ]}

    model.eval()
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            inputs = tokenizer(
                sample["prompt"], return_tensors="pt",
                truncation=True, max_length=MAX_PROMPT,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model.generate(
                **inputs, max_new_tokens=256,
                temperature=0.1, do_sample=True, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            cat = sample.get("category", "unknown")
            chosen = sample["chosen"]

            m["total"] += 1
            expected = ("NOT_FOUND" if "NOT_FOUND" in chosen
                        else ("ESCALATE" if "ESCALATE" in chosen else "ANSWER"))
            got = ("NOT_FOUND" if "NOT_FOUND" in response
                   else ("ESCALATE" if "ESCALATE" in response else "ANSWER"))
            if expected == got:
                m["correct_decision"] += 1

            if "B_" in cat:
                m["hallucination_total"] += 1
                if "NOT_FOUND" in response or "incorrect" in response.lower():
                    m["hallucination_caught"] += 1

            if "C_" in cat:
                m["not_found_total"] += 1
                if "NOT_FOUND" in response or "not specified" in response.lower() or "absent" in response.lower():
                    m["not_found_correct"] += 1

            if "A_" in cat:
                m["grounded_total"] += 1
                cite = re.search(r"(Section|Article|Clause)\s+[\d\.]+", chosen)
                if cite and cite.group() in response:
                    m["grounded_correct"] += 1

            if "F_" in cat:
                m["adversarial_total"] += 1
                if "incorrect" in response.lower() or "does not" in response.lower() or "NOT_FOUND" in response:
                    m["adversarial_correct"] += 1

            if "E_" in cat:
                m["multihop_total"] += 1
                cite_count = (response.count("[Section") + response.count("[Article") + response.count("[Clause"))
                if cite_count >= 2 or "CITATION" in response:
                    m["multihop_correct"] += 1

            if (i + 1) % 20 == 0:
                print(f"  Evaluated {i+1}/{len(samples)}...")

    results = {
        "total_samples": m["total"],
        "decision_accuracy":       round(m["correct_decision"] / max(m["total"], 1), 4),
        "hallucination_catch_rate": round(m["hallucination_caught"] / max(m["hallucination_total"], 1), 4),
        "refusal_accuracy":        round(m["not_found_correct"] / max(m["not_found_total"], 1), 4),
        "grounding_accuracy":      round(m["grounded_correct"] / max(m["grounded_total"], 1), 4),
        "adversarial_robustness":  round(m["adversarial_correct"] / max(m["adversarial_total"], 1), 4),
        "multihop_completeness":   round(m["multihop_correct"] / max(m["multihop_total"], 1), 4),
    }

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS V4")
    print(f"{'='*50}")
    for k, v in results.items():
        print(f"  {k}: {v if k == 'total_samples' else f'{v:.1%}'}")
    print(f"{'='*50}\n")
    return results


# ─── HuggingFace Push ─────────────────────────────────────────────────────────

def push_to_hf(adapter_dir):
    print(f"\n{'='*55}")
    print(f"PUSHING → {HF_REPO}")
    print(f"{'='*55}\n")

    torch.cuda.empty_cache(); gc.collect()

    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16,
            device_map="cuda:0", low_cpu_mem_usage=True, cache_dir=CACHE_DIR,
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.float16,
            device_map="cuda:0", low_cpu_mem_usage=True, cache_dir=CACHE_DIR,
        )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.push_to_hub(HF_REPO)
    tok.push_to_hub(HF_REPO)
    print(f"\n✅ PUSHED → https://huggingface.co/{HF_REPO}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Step 1: Building v4 dataset...")
    dataset = build_dataset()

    print(f"\nStep 2: Training ({len(dataset)} pairs)...")
    trainer, model, tokenizer = train_dpo(dataset, OUTPUT_DIR)

    print("\nStep 3: Evaluation...")
    eval_samples = random.sample(dataset, min(80, len(dataset)))
    results = evaluate_model(model, tokenizer, eval_samples)
    eval_path = os.path.join(OUTPUT_DIR, "eval_results_v4.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {eval_path}")

    print("\nStep 4: Pushing to HuggingFace...")
    del trainer, model
    torch.cuda.empty_cache(); gc.collect()
    time.sleep(3)
    push_to_hf(os.path.join(OUTPUT_DIR, "final"))

    print(f"\n{'='*55}")
    print("ALL DONE!")
    print(f"  Model : https://huggingface.co/{HF_REPO}")
    print(f"  Eval  : {eval_path}")
    print(f"{'='*55}")
