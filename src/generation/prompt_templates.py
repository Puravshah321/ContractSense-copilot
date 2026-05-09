"""Prompt templates for citation-first contract generation."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are ContractSense, an enterprise contract risk intelligence assistant.
Always answer in strict JSON and prioritize citation-first explanations.
Rules:
1) Ground every claim in provided clauses only.
2) Start plain_explanation with risk level in the first sentence.
3) Use plain business language, avoid legal jargon where possible.
4) Include one concrete recommended action.
5) Keep citation fields complete and consistent with provided clause metadata.

Return JSON with exactly these keys:
- risk_level: LOW|MEDIUM|HIGH|CRITICAL
- plain_explanation: string
- key_obligation: string
- recommended_action: string
- citation: {clause_id: string, page_number: int, char_span: [int, int]}
"""


def build_user_prompt(
    query: str,
    clauses: list[dict[str, Any]],
    tool_results: dict[str, Any] | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Build deterministic user prompt payload consumed by Stage 6 generator."""
    payload = {
        "query": query,
        "clauses": clauses[:3],
        "tool_results": tool_results or {},
        "chat_history": chat_history or [],
        "output_contract": {
            "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
            "plain_explanation": "2-4 concise sentences",
            "key_obligation": "single sentence",
            "recommended_action": "single sentence",
            "citation": {
                "clause_id": "string",
                "page_number": "int",
                "char_span": [0, 0],
            },
        },
    }
    return (
        "Generate the final answer for the following retrieval package. "
        "Use citation-first style and return only valid JSON.\n\n"
        f"{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )
