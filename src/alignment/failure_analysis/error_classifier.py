"""
error_classifier.py
Classifies failures into categories:
- missing_risk_label
- no_action
- wrong_citation
- hallucination
- verbosity
- unclear_explanation
- format_violation
"""

import re
from typing import Optional


RISK_PATTERN = re.compile(r"^RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)", re.MULTILINE)
ACTION_PATTERN = re.compile(r"ACTION:", re.MULTILINE)
CITATION_PATTERN = re.compile(r"CITATION:\s*\[.+?\]")

HALLUCINATION_MARKERS = [
    "as an ai", "i cannot", "i'm sorry", "i don't have",
    "this is not", "i am not sure", "hypothetically",
    "in general terms", "typically speaking",
]

JARGON_MARKERS = [
    "notwithstanding", "hereinafter", "aforementioned",
    "pursuant to", "herein", "therein", "whereas",
    "shall be deemed", "ipso facto",
]


def classify_error(output: str, prompt: str = "") -> list:
    errors = []

    if not RISK_PATTERN.search(output):
        errors.append("missing_risk_label")

    if not ACTION_PATTERN.search(output):
        errors.append("no_action")

    if not CITATION_PATTERN.search(output):
        errors.append("missing_citation")

    lower_output = output.lower()
    for marker in HALLUCINATION_MARKERS:
        if marker in lower_output:
            errors.append("hallucination")
            break

    token_count = len(output.split())
    if token_count > 400:
        errors.append("verbosity")
    elif token_count < 15:
        errors.append("too_brief")

    jargon_count = sum(1 for j in JARGON_MARKERS if j in lower_output)
    if jargon_count >= 3:
        errors.append("excessive_jargon")

    risk_match = RISK_PATTERN.search(output)
    if risk_match:
        explanation_start = output[risk_match.end():]
        explanation = explanation_start.split("ACTION:")[0] if "ACTION:" in explanation_start else explanation_start
        if len(explanation.strip().split()) < 5:
            errors.append("unclear_explanation")

    if "RISK:" in output and "ACTION:" in output and "CITATION:" in output:
        risk_pos = output.index("RISK:")
        action_pos = output.index("ACTION:")
        citation_pos = output.index("CITATION:")
        if not (risk_pos < action_pos < citation_pos):
            errors.append("format_violation")

    return errors if errors else ["none"]


def classify_batch(outputs: list, prompts: list = None) -> list:
    if prompts is None:
        prompts = [""] * len(outputs)

    return [
        {
            "index": i,
            "output_preview": output[:200],
            "errors": classify_error(output, prompt),
        }
        for i, (output, prompt) in enumerate(zip(outputs, prompts))
    ]


def get_error_statistics(classifications: list) -> dict:
    from collections import Counter

    all_errors = []
    for entry in classifications:
        all_errors.extend(entry["errors"])

    counts = dict(Counter(all_errors))

    total = len(classifications)
    failed = sum(1 for c in classifications if c["errors"] != ["none"])

    return {
        "total_samples": total,
        "failed_samples": failed,
        "success_rate": round((total - failed) / max(total, 1), 4),
        "error_type_counts": counts,
        "most_common_error": max(counts, key=counts.get) if counts else "none",
    }
