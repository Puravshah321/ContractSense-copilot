# generator/lora_train.py
"""
MAHAK'S LoRA FINE-TUNING - Simple Working Version
Uses standard Trainer instead of SFTTrainer
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# Training data
TRAINING_DATA = [
    {
        "instruction": "Analyze this contract clause",
        "input": "Vendor shall indemnify, defend, and hold harmless Customer from any claims arising from Vendor's breach.",
        "output": "Decision: REVIEW\nRisk: HIGH\nExplanation: This indemnification clause transfers liability to your company. [Clause: indemnity_001]\nCitation: indemnity_001"
    },
    {
        "instruction": "Analyze this contract clause",
        "input": "IN NO EVENT SHALL VENDOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES.",
        "output": "Decision: REVIEW\nRisk: HIGH\nExplanation: This clause removes vendor responsibility for many types of losses. [Clause: liability_001]\nCitation: liability_001"
    },
    {
        "instruction": "Analyze this contract clause",
        "input": "Either party may terminate this Agreement with 30 days written notice.",
        "output": "Decision: ACCEPT\nRisk: LOW\nExplanation: Standard termination clause. [Clause: term_001]\nCitation: term_001"
    },
    {
        "instruction": "Analyze this contract clause",
        "input": "This Agreement shall automatically renew for successive one-year terms unless either party provides written notice of non-renewal at least 60 days prior.",
        "output": "Decision: RENEGOTIATE\nRisk: HIGH\nExplanation: Auto-renewal clause could lock you in. [Clause: autorenew_001]\nCitation: autorenew_001"
    },
]

class SimpleLoRATrainer:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        # LoRA config
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Model ready! Trainable: {trainable:,} ({100*trainable/total:.2f}% of {total:,})")
        return True
    
    def prepare_dataset(self):
        """Prepare dataset"""
        def format_func(ex):
            text = f"""[INST] {ex['instruction']}

Clause: {ex['input']} [/INST]

{ex['output']}"""
            return {"text": text}
        
        dataset = Dataset.from_list(TRAINING_DATA)
        dataset = dataset.map(format_func)
        
        def tokenize(ex):
            return self.tokenizer(
                ex["text"],
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors=None
            )
        
        tokenized = dataset.map(tokenize, batched=False)
        print(f"Dataset ready: {len(tokenized)} examples")
        return tokenized
    
    def train(self):
        """Run training"""
        if not self.load_model():
            return False
        
        dataset = self.prepare_dataset()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Training args
        training_args = TrainingArguments(
            output_dir="./lora_output",
            num_train_epochs=10,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            learning_rate=1e-4,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print("\nStarting training...")
        trainer.train()
        
        # Save
        os.makedirs("./lora_adapters/final", exist_ok=True)
        self.model.save_pretrained("./lora_adapters/final")
        self.tokenizer.save_pretrained("./lora_adapters/final")
        print("\nTraining complete! Saved to ./lora_adapters/final")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("MAHAK'S LoRA FINE-TUNING")
    print("=" * 60)
    
    choice = input("Run LoRA training? (yes/no): ").lower()
    
    if choice == "yes":
        trainer = SimpleLoRATrainer(model_name="microsoft/phi-2")
        trainer.train()
    else:
        print("Skipping LoRA training.")
        