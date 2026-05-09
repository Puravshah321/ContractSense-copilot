"""
Lightning AI API Server - Grounded DPO Model
Run this on your Lightning AI GPU instance AFTER training!

Usage:
  pip install fastapi uvicorn pyngrok torch transformers peft
  python scripts/serve_dpo_api.py
"""

import torch
import json
import re
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI()

# ── Configuration ──
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "grounded_dpo_model/final"  # Path where your trained model was saved
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE" # Optional if you need it, but free tier usually works without it for simple tests.

# Load model globally so it only spins up once
model = None
tokenizer = None

class QueryRequest(BaseModel):
    query: str
    evidence_context: str

@app.on_event("startup")
def load_model():
    global model, tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model in 4-bit...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading DPO Adapters from {ADAPTER_DIR}...")
    try:
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        print("✅ GPU Model loaded and ready to serve!")
    except Exception as e:
        print(f"Warning: Could not load adapters (Did you train it first?): {e}")
        model = base # Fallback to base model if untrained


@app.post("/generate")
def generate_answer(req: QueryRequest):
    """Takes evidence from the local UI, runs it through the GPU LLM, and returns the grounded answer."""
    global model, tokenizer

    prompt = (
        f"[INST] You are ContractSense, a grounded contract analysis system.\n\n"
        f"EVIDENCE:\n{req.evidence_context}\n\n"
        f"QUERY: {req.query}\n\n"
        f"STRICT RULES:\n"
        f"1. Answer ONLY using the Evidence above.\n"
        f"2. If the answer is not in the evidence, reply with DECISION: NOT_FOUND.\n"
        f"3. Otherwise, quote the evidence, cite the clause_id, and provide a RISK label.\n"
        f"[/INST]\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Simple extraction to return to UI
    decision = "ANSWER"
    if "NOT_FOUND" in response or "not specified" in response.lower() or "no relevant" in response.lower():
        decision = "NOT_FOUND"

    risk_match = re.search(r"RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)", response)
    risk_level = risk_match.group(1) if risk_match else "MEDIUM"

    return {
        "answer": response,
        "decision": decision,
        "risk_level": risk_level
    }

if __name__ == "__main__":
    # Open a local tunnel using ngrok so your laptop can talk to Lightning AI!
    public_url = ngrok.connect(8000).public_url
    print(f"\n{'='*60}")
    print(f"🚀 API TUNNEL OPEN!")
    print(f"📋 Copy this URL into your Streamlit UI: {public_url}")
    print(f"{'='*60}\n")
    
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
