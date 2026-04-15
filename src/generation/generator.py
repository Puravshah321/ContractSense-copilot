"""Inference wrapper for base and LoRA-adapted generator models."""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)


class ContractGenerator:
    """Loads a base model and optional LoRA adapter for Stage 6 generation."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        use_4bit: bool = True,
        max_new_tokens: int = 320,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if adapter_path:
            log.info("Loading LoRA adapter: %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)

        self.model = model.eval()

    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_json(self, prompt: str) -> dict[str, Any]:
        text = self.generate_text(prompt)
        return self._safe_parse_json(text)

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "", 1).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
            return {
                "risk_level": "MEDIUM",
                "plain_explanation": text[:500],
                "key_obligation": "Unable to parse structured output.",
                "recommended_action": "Review this clause with legal team.",
                "citation": {
                    "clause_id": "unknown",
                    "page_number": -1,
                    "char_span": [0, 0],
                },
            }
