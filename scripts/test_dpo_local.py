import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
import os

# Disable warning loops
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = r"c:\Users\Jay\Desktop\DAU\SEM-2\DL\Project\ContractSense-copilot\src\alignment\models\dpo_aligned_model"

if not os.path.exists(ADAPTER_PATH):
    print(f"ERROR: Could not find DPO model at {ADAPTER_PATH}")
    exit(1)

print("1. Loading Mistral Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("2. Loading Base Mistral Model (~14GB)...")
# We load in 4-bit so it fits on your local GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

print("3. Fusing your DPO 'Brain Adapters' onto the Base Model...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("\n" + "="*60)
print("🧠 MODEL READY! PREPARING TEST DATA...")
print("="*60)

test_clause = """
9. TERMINATION FOR CONVENIENCE
Either party may terminate this Agreement at any time, for any reason or no reason, 
by providing the other party with at least ninety (90) days prior written notice. 
Upon such termination, Customer shall immediately pay all outstanding invoices for 
services rendered up to the date of termination.
"""
test_query = "Can the vendor just cancel this contract suddenly? What do I owe them?"

# Format the exact way the model was trained during DPO
prompt = f"Clause: {test_clause.strip()}\n\nQuery: {test_query}\n\n[/INST]"

print("\n🚀 Injecting clause and query into the Model...")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

start_time = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.1,  # Low temp for deterministic formatting
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
elapsed = time.time() - start_time

print(f"Generation finished in {elapsed:.2f} seconds.")

# Extract just the model's response, ignoring the prompt
raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
final_answer = raw_response.split("[/INST]")[-1].strip()

print("\n" + "="*60)
print("             🤖 CONTRACTSENSE DPO OUTPUT")
print("="*60)
print(final_answer)
print("="*60)
