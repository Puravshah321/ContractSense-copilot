"""
dpo_trainer.py
Core DPO training logic using TRL DPOTrainer.
- Loads LoRA-finetuned Mistral model
- Applies PEFT config for DPO
- Loads and tokenizes dataset (prompt/chosen/rejected)
- Configures training arguments with large-dataset support
- Runs training loop
- Saves final model
"""

import json
from pathlib import Path

from datasets import Dataset


def load_dpo_dataset(dataset_path: str, test_size: float = 0.1, seed: int = 42):
    with open(dataset_path) as f:
        raw = json.load(f)

    records = [
        {
            "prompt": entry["prompt"],
            "chosen": entry["chosen"],
            "rejected": entry["rejected"],
        }
        for entry in raw
        if all(k in entry for k in ("prompt", "chosen", "rejected"))
    ]

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    print(f"✅ Dataset: {len(split['train'])} train, {len(split['test'])} eval pairs")
    return split["train"], split["test"]


def load_model_and_tokenizer(
    model_path: str,
    use_4bit: bool = True,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def build_peft_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
):
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def build_training_args(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 5e-5,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    warmup_ratio: float = 0.1,
    use_bf16: bool = True,
    use_fp16: bool = False,
    gradient_checkpointing: bool = True,
    save_total_limit: int = 2,
    save_strategy: str = "epoch",
    eval_strategy: str = "epoch",
    lr_scheduler: str = "cosine",
):
    from trl import DPOConfig

    return DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=True,
        save_total_limit=save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=gradient_checkpointing,
    )


def train_dpo(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    beta: float = 0.1,
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 5e-5,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    dry_run: bool = False,
):
    from trl import DPOTrainer
    from peft import get_peft_model, prepare_model_for_kbit_training

    train_dataset, eval_dataset = load_dpo_dataset(dataset_path)

    if dry_run:
        print("\n🔍 DRY RUN: Dataset loaded successfully.")
        print(f"   Sample prompt: {train_dataset[0]['prompt'][:100]}...")
        print("   Training would proceed with the above configuration.")
        return None

    model, tokenizer = load_model_and_tokenizer(model_path, use_4bit)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = build_peft_config(lora_r, lora_alpha, lora_dropout)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = build_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=beta,
    )

    print("\n🚀 Starting DPO training...")
    train_result = trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    config_log = {
        "base_model": model_path,
        "dataset": dataset_path,
        "beta": beta,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "total_train_samples": len(train_dataset),
        "total_eval_samples": len(eval_dataset),
    }
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(config_log, f, indent=2)

    print(f"\n✅ DPO-aligned model saved to: {output_dir}")
    return train_result
