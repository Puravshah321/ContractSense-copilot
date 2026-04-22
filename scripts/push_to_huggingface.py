"""
Hugging Face Upload Script
Run this on Lightning AI after training to push your grounded model to HF.
This allows you to use the Hugging Face Free Serverless API!

Usage:
  pip install huggingface_hub
  hf auth login
  python scripts/push_to_huggingface.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "grounded_dpo_model/final" 

# IMPORTANT: Change this to your Hugging Face username!
HF_USERNAME = "22Jay" 
REPO_NAME = f"{HF_USERNAME}/ContractSense-Grounded-DPO"

def main():
    print("Clearing GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading base model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            cache_dir="hf_cache",
        )
    except TypeError:
        # Fallback for older transformers versions
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            cache_dir="hf_cache",
        )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir="hf_cache")
    
    print("Merging specialized DPO adapters into the base model...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    
    # Merge weights so it can be hosted natively on HF API
    merged_model = peft_model.merge_and_unload()
    
    print(f"Pushing merged model to Hugging Face Hub at: {REPO_NAME}")
    
    # In newer transformers, 'use_auth_token' is removed, and it automatically reads from 'hf auth login'
    merged_model.push_to_hub(REPO_NAME)
    tokenizer.push_to_hub(REPO_NAME)
    
    print(f"\n✅ SUCCESSFULLY PUSHED!")
    print(f"Your model is now hosted for free at: https://huggingface.co/{REPO_NAME}")
    print(f"You can now use the HF Serverless API in your local UI.")

if __name__ == "__main__":
    main()
