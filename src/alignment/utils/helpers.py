"""
helpers.py
Text formatting, parsing, and validation helpers
used across the alignment pipeline.
"""

import re
import json
from typing import Optional


RISK_PATTERN = re.compile(r"RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)")
ACTION_PATTERN = re.compile(r"ACTION:\s*(.+?)(?:\n\n|$)", re.DOTALL)
CITATION_PATTERN = re.compile(r"CITATION:\s*\[(.+?)\]")


def extract_risk_level(text: str) -> Optional[str]:
    m = RISK_PATTERN.search(text)
    return m.group(1) if m else None


def extract_action(text: str) -> Optional[str]:
    m = ACTION_PATTERN.search(text)
    return m.group(1).strip() if m else None


def extract_citation(text: str) -> Optional[str]:
    m = CITATION_PATTERN.search(text)
    return m.group(1).strip() if m else None


def format_dpo_prompt(clause: str, query: str) -> str:
    return f"Clause: {clause}\n\nQuery: {query}"


def parse_prompt(prompt: str) -> dict:
    result = {"clause": "", "query": ""}
    clause_match = re.search(r"Clause:\s*(.+?)\n\nQuery:\s*(.+)", prompt, re.DOTALL)
    if clause_match:
        result["clause"] = clause_match.group(1).strip()
        result["query"] = clause_match.group(2).strip()
    return result


def truncate_text(text: str, max_tokens: int = 512) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + "..."


def validate_structured_output(output: str) -> dict:
    checks = {
        "has_risk": bool(RISK_PATTERN.search(output)),
        "has_action": bool(ACTION_PATTERN.search(output)),
        "has_citation": bool(CITATION_PATTERN.search(output)),
        "token_count": len(output.split()),
        "is_valid": False,
    }
    checks["is_valid"] = all([checks["has_risk"], checks["has_action"], checks["has_citation"]])
    return checks


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def save_json(data, path: str):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def clean_model_output(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'</s>$', '', raw)
    raw = re.sub(r'\[/INST\]', '', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()
