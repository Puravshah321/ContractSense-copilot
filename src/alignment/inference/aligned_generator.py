"""
aligned_generator.py
Loads the DPO-aligned model and generates structured output.
Enforces output schema: risk_level, explanation, action, citation.
Parses raw model output into the structured format used by the pipeline.
"""

import re
import json
from typing import Optional


RISK_PATTERN = re.compile(r"RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)")
ACTION_PATTERN = re.compile(r"ACTION:\s*(.+?)(?:\n\n|$)", re.DOTALL)
CITATION_PATTERN = re.compile(r"CITATION:\s*\[(.+?)\]")


def load_aligned_model(model_path: str, use_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"🔄 Loading DPO-aligned model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model.eval()
    print("✅ DPO-aligned model loaded")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    clause: str,
    query: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    prompt = f"Clause: {clause}\n\nQuery: {query}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def parse_structured_output(raw_output: str) -> dict:
    result = {
        "risk_level": None,
        "explanation": "",
        "action": "",
        "citation": "",
        "raw_output": raw_output,
        "parse_success": False,
    }

    risk_match = RISK_PATTERN.search(raw_output)
    if risk_match:
        result["risk_level"] = risk_match.group(1)

    action_match = ACTION_PATTERN.search(raw_output)
    if action_match:
        result["action"] = action_match.group(1).strip()

    citation_match = CITATION_PATTERN.search(raw_output)
    if citation_match:
        result["citation"] = citation_match.group(1).strip()

    parts = raw_output.split("ACTION:")
    if len(parts) > 1:
        explanation_part = parts[0]
        explanation_part = RISK_PATTERN.sub("", explanation_part).strip()
        result["explanation"] = explanation_part.strip("\n ")
    else:
        explanation_part = raw_output
        explanation_part = RISK_PATTERN.sub("", explanation_part)
        explanation_part = CITATION_PATTERN.sub("", explanation_part)
        result["explanation"] = explanation_part.strip("\n ")

    result["parse_success"] = all([
        result["risk_level"],
        len(result["explanation"]) > 10,
        len(result["action"]) > 5,
        len(result["citation"]) > 0,
    ])

    return result


def generate_structured(
    model,
    tokenizer,
    clause: str,
    query: str,
    max_new_tokens: int = 512,
) -> dict:
    raw = generate_response(model, tokenizer, clause, query, max_new_tokens)
    structured = parse_structured_output(raw)
    structured["prompt"] = f"Clause: {clause}\n\nQuery: {query}"
    return structured


def batch_generate(
    model,
    tokenizer,
    prompts: list,
    max_new_tokens: int = 512,
) -> list:
    results = []
    for i, item in enumerate(prompts):
        clause = item.get("clause", "")
        query = item.get("query", "")

        result = generate_structured(model, tokenizer, clause, query, max_new_tokens)
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(prompts)}")

    return results
