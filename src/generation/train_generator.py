"""LoRA SFT trainer for Stage 6 generator model candidates."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from src.generation.prompt_templates import SYSTEM_PROMPT

DEFAULT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
]


@dataclass(frozen=True)
class TrainConfig:
    base_model: str
    output_dir: Path
    train_file: Path
    eval_file: Path
    epochs: int = 2
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    seed: int = 42


def _infer_risk(clause_text: str) -> str:
    text = clause_text.lower()
    if any(k in text for k in ("unlimited", "indemnity", "liability", "penalty", "termination")):
        return "HIGH"
    if any(k in text for k in ("notice", "renewal", "pricing", "payment")):
        return "MEDIUM"
    return "LOW"


def _build_response(clause: dict[str, Any]) -> dict[str, Any]:
    risk = _infer_risk(clause.get("clause_text", ""))
    clause_id = clause.get("clause_id", "unknown")
    char_count = int(clause.get("char_count", 0))
    page_guess = max(1, char_count // 3000 + 1)
    span_end = min(char_count if char_count > 0 else 240, 480)

    return {
        "risk_level": risk,
        "plain_explanation": (
            f"{risk} risk: this clause can materially affect commercial or legal exposure. "
            "It should be reviewed with focus on obligations, exceptions, and negotiation leverage."
        ),
        "key_obligation": "Track obligations and deadlines defined by this clause before contract execution.",
        "recommended_action": "Negotiate explicit limits, clearer notice windows, and written exceptions for critical scenarios.",
        "citation": {
            "clause_id": clause_id,
            "page_number": page_guess,
            "char_span": [0, span_end],
        },
    }


def build_generator_dataset_from_clauses(
    clauses_path: str | Path,
    output_train: str | Path,
    output_eval: str | Path,
    eval_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[int, int]:
    path = Path(clauses_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(rows)

    samples: list[dict[str, str]] = []
    for clause in rows:
        clause_text = str(clause.get("clause_text", "")).strip()
        if len(clause_text) < 80:
            continue

        instruction = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Query: Explain risk in this clause for business stakeholders.\n"
            f"Clause: {clause_text[:1800]}\n"
            "Return only JSON."
        )
        response = json.dumps(_build_response(clause), ensure_ascii=True)
        samples.append({"text": f"<s>[INST] {instruction} [/INST] {response}</s>"})

    split_idx = int(len(samples) * (1 - eval_ratio))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    Path(output_train).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_train).open("w", encoding="utf-8") as f:
        for row in train_samples:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    with Path(output_eval).open("w", encoding="utf-8") as f:
        for row in eval_samples:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return len(train_samples), len(eval_samples)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_single_model(config: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    train_dataset = Dataset.from_list(_load_jsonl(config.train_file))
    eval_dataset = Dataset.from_list(_load_jsonl(config.eval_file))

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
    )

    trainer.train()
    trainer.save_model(str(config.output_dir))

    history = trainer.state.log_history
    train_losses = [h["loss"] for h in history if "loss" in h]
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    final_train_loss = float(train_losses[-1]) if train_losses else None
    final_eval_loss = float(eval_losses[-1]) if eval_losses else None
    generalization_gap = None
    if final_train_loss is not None and final_eval_loss is not None:
        generalization_gap = round(final_eval_loss - final_train_loss, 4)

    metrics = {
        "base_model": config.base_model,
        "model_label": config.base_model.split("/")[-1],
        "adapter_path": str(config.output_dir),
        "finetuned_model": True,
        "train_loss": final_train_loss,
        "eval_loss": final_eval_loss,
        "generalization_gap": generalization_gap,
        "overfit_flag": bool(generalization_gap is not None and generalization_gap > 0.35),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
    }

    metrics_path = config.output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_model_candidates(
    train_file: str | Path,
    eval_file: str | Path,
    output_root: str | Path,
    model_names: list[str] | None = None,
    epochs: int = 2,
    learning_rate: float = 2e-4,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    model_names = model_names or DEFAULT_MODELS
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        safe = model_name.replace("/", "__")
        out_dir = root / safe
        cfg = TrainConfig(
            base_model=model_name,
            output_dir=out_dir,
            train_file=Path(train_file),
            eval_file=Path(eval_file),
            epochs=epochs,
            learning_rate=learning_rate,
        )
        results.append(train_single_model(cfg))

    leaderboard = {
        "tested_models": [r["base_model"] for r in results],
        "finetuned_models": [r["base_model"] for r in results],
        "results": results,
    }
    (root / "model_training_comparison.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
    return results
