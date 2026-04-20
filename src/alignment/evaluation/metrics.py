"""
metrics.py
Defines all evaluation metrics for the DPO alignment pipeline:
- risk_salience: checks if output starts with a valid RISK label
- actionability: checks for actionable keywords / ACTION section
- citation_present: checks for proper CITATION format
- readability: simple token-length / Flesch-like score
- confidence_score: optional composite quality score
"""

import re
from typing import Optional


RISK_PATTERN = re.compile(r"^RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)", re.MULTILINE)
ACTION_PATTERN = re.compile(r"ACTION:", re.MULTILINE)
CITATION_PATTERN = re.compile(r"CITATION:\s*\[.+?\]")

ACTION_KEYWORDS = [
    "review", "negotiate", "ensure", "verify", "consult",
    "document", "track", "set up", "notify", "send",
    "assess", "calculate", "create", "monitor", "brief",
    "compare", "check", "recommend", "action", "prepare",
]


def risk_salience(output: str) -> float:
    return 1.0 if RISK_PATTERN.search(output) else 0.0


def extract_risk_level(output: str) -> Optional[str]:
    m = RISK_PATTERN.search(output)
    return m.group(1) if m else None


def actionability(output: str) -> float:
    if ACTION_PATTERN.search(output):
        return 1.0

    lower_output = output.lower()
    keyword_hits = sum(1 for kw in ACTION_KEYWORDS if kw in lower_output)
    return min(1.0, keyword_hits / 3.0)


def citation_present(output: str) -> float:
    return 1.0 if CITATION_PATTERN.search(output) else 0.0


def readability(output: str, max_tokens: int = 300) -> float:
    tokens = output.split()
    token_count = len(tokens)

    if token_count == 0:
        return 0.0

    if token_count > max_tokens:
        return max(0.0, 1.0 - (token_count - max_tokens) / max_tokens)

    sentences = re.split(r'[.!?]+', output)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    avg_sentence_length = token_count / sentence_count

    if 8 <= avg_sentence_length <= 25:
        structure_score = 1.0
    elif avg_sentence_length < 5:
        structure_score = 0.5
    else:
        structure_score = max(0.3, 1.0 - (avg_sentence_length - 25) / 50)

    length_score = min(1.0, token_count / 50)

    return (structure_score * 0.6 + length_score * 0.4)


def format_compliance(output: str) -> float:
    score = 0.0
    total = 3.0

    if RISK_PATTERN.search(output):
        score += 1.0
    if ACTION_PATTERN.search(output):
        score += 1.0
    if CITATION_PATTERN.search(output):
        score += 1.0

    return score / total


def composite_quality_score(output: str) -> dict:
    r_sal = risk_salience(output)
    act = actionability(output)
    cit = citation_present(output)
    read = readability(output)
    fmt = format_compliance(output)

    overall = (
        r_sal * 0.25 +
        act * 0.25 +
        cit * 0.20 +
        read * 0.15 +
        fmt * 0.15
    )

    return {
        "risk_salience": round(r_sal, 4),
        "actionability": round(act, 4),
        "citation_present": round(cit, 4),
        "readability": round(read, 4),
        "format_compliance": round(fmt, 4),
        "overall_quality": round(overall, 4),
    }


def compute_metrics_batch(outputs: list) -> dict:
    if not outputs:
        return {}

    metrics = [composite_quality_score(o) for o in outputs]

    avg = {}
    for key in metrics[0]:
        values = [m[key] for m in metrics]
        avg[key] = round(sum(values) / len(values), 4)

    return {
        "average": avg,
        "count": len(outputs),
        "per_sample": metrics,
    }
