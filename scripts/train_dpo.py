"""
Step 2: DPO Training
Loads the LoRA fine-tuned Mistral model (from Member 2) and applies
Direct Preference Optimization (DPO) using the preference dataset.

Requirements:
    pip install trl transformers peft accelerate bitsandbytes datasets torch

Usage:
    python 2_train_dpo.py \
        --base_model ./models/lora_finetuned_mistral \
        --dataset data/dpo_dataset.json \
        --output_dir ./models/dpo_aligned_model \
        --beta 0.1 \
        --epochs 3
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training for ContractSense")
    parser.add_argument("--base_model", default="./models/lora_finetuned_mistral",
                        help="Path to LoRA fine-tuned Mistral model (from Member 2)")
    parser.add_argument("--dataset", default="data/dpo_dataset.json",
                        help="Path to DPO dataset JSON file")
    parser.add_argument("--output_dir", default="./models/dpo_aligned_model")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (higher = closer to reference model)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Load model in 4-bit quantization")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate setup without actually training")
    return parser.parse_args()


def load_dataset_from_json(path: str):
    """Load DPO dataset and convert to HuggingFace Dataset format."""
    from datasets import Dataset

    with open(path) as f:
        raw = json.load(f)

    # DPOTrainer expects: prompt, chosen, rejected (all strings)
    records = [
        {
            "prompt": entry["prompt"],
            "chosen": entry["chosen"],
            "rejected": entry["rejected"],
        }
        for entry in raw
    ]

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"✅ Dataset: {len(split['train'])} train, {len(split['test'])} eval pairs")
    return split["train"], split["test"]


def load_model_and_tokenizer(model_path: str, use_4bit: bool):
    """Load base model with optional 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    print(f"🔄 Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print("✅ Model loaded")
    return model, tokenizer


def build_lora_config(r: int, alpha: int, dropout: float):
    """Build LoRA configuration for DPO fine-tuning."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )


def build_training_args(output_dir: str, epochs: int, batch_size: int,
                        grad_accum: int, lr: float, max_length: int):
    """Build DPO training arguments."""
    from trl import DPOConfig

    return DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_length=max_length,
        max_prompt_length=512,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none",       # Set to "wandb" if you want experiment tracking
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )


def train(args):
    from trl import DPOTrainer
    from peft import get_peft_model, prepare_model_for_kbit_training

    # Load data
    train_dataset, eval_dataset = load_dataset_from_json(args.dataset)

    if args.dry_run:
        print("\n🔍 DRY RUN: Dataset loaded successfully.")
        print(f"   Sample prompt: {train_dataset[0]['prompt'][:100]}...")
        print("   Training would proceed with the above configuration.")
        print("   Run without --dry_run to actually train.")
        return

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.use_4bit)

    # Prepare model for LoRA
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    training_args = build_training_args(
        args.output_dir, args.epochs, args.batch_size,
        args.grad_accum, args.lr, args.max_length
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # None = use frozen copy of model as reference (memory-efficient)
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=args.beta,
    )

    print("\n🚀 Starting DPO training...")
    trainer.train()

    # Save final model
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config for reproducibility
    config_log = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "beta": args.beta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    with open(Path(args.output_dir) / "training_config.json", "w") as f:
        json.dump(config_log, f, indent=2)

    print(f"\n✅ DPO-aligned model saved to: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)