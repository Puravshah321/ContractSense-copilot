"""
Legal taxonomy and clause classifier.

This is intentionally lightweight and deterministic for the app path, but the
labels mirror the DPO data categories so a learned classifier can replace it.
"""
import re


_CATEGORY_RULES = [
    ("Warranty/Representation", [r"\bwarrant(?:y|ies|s)?\b", r"\brepresent(?:s|ations?)?\b", r"\bguarantee\b"]),
    ("Financial/Payment", [r"\bpay(?:ment|able)?\b", r"\bfees?\b", r"\bcommission\b", r"\binvoice\b", r"\bcharges?\b", r"\bcosts?\b", r"\bexpenses?\b"]),
    ("Liability/Remedies", [r"\bliabilit", r"\bdamages?\b", r"\bliquidated\s+damages\b", r"\bremed(?:y|ies)\b", r"\bcontract\s+value\b", r"\bcompensate\b"]),
    ("Indemnification", [r"\bindemnif", r"\bhold\s+harmless\b", r"\bthird-party\s+claims?\b"]),
    ("Term/Duration", [r"\bterm\b", r"\bvalid\s+up\s+to\b", r"\bvalid\s+for\b", r"\bexpires?\b", r"\bexpiration\b", r"\beffective\s+date\b"]),
    ("Survival", [r"\bsurviv(?:e|al)\b", r"\bafter\s+(?:expiration|termination)\b", r"\bperpetuit"]),
    ("Confidentiality", [r"\bconfidential", r"\bnon-disclosure\b", r"\bdisclos(?:e|ure)\b", r"\bneed\s+to\s+know\b"]),
    ("Permitted Disclosure", [r"\bpermitted\s+disclos", r"\bmay\s+share\b", r"\bstqc\b", r"\bgovernment\s+entities\b"]),
    ("Data Protection", [r"\bpersonal\s+data\b", r"\baudit\s+(?:information|data)\b", r"\boutside\s+india\b", r"\boutside\s+the\s+country\b", r"\bstorage\b", r"\bprocessing\b"]),
    ("Use Restriction", [r"\buse\b.*\bscope\b", r"\bnot\s+to\s+(?:make|retain|disclose|send|engage)\b", r"\bshall\s+not\b", r"\bwithout\s+(?:the\s+)?(?:express\s+)?(?:prior\s+)?written\s+(?:consent|approval)\b"]),
    ("Non-Solicitation", [r"\bnon\s*-?\s*solicitation\b", r"\bsolicit\b"]),
    ("Dispute Resolution", [r"\barbitration\b", r"\bgoverning\s+law\b", r"\bjurisdiction\b", r"\bdispute\b"]),
]

_CATEGORY_WEIGHTS = {
    "Warranty/Representation": {"warranty": 2.0},
    "Financial/Payment": {"financial_commitment": 1.5},
    "Liability/Remedies": {"liability": 1.3, "financial_commitment": 0.8},
    "Indemnification": {"liability": 1.1},
    "Term/Duration": {"term": 1.8},
    "Survival": {"survival": 1.6},
    "Confidentiality": {"confidentiality": 1.3, "sharing": 1.0, "data_use": 0.8},
    "Permitted Disclosure": {"sharing": 1.4},
    "Data Protection": {"data_use": 1.4, "sharing": 0.6},
    "Use Restriction": {"restriction": 1.3, "data_use": 1.2, "sharing": 1.0},
    "Non-Solicitation": {"restriction": 0.8},
    "Dispute Resolution": {},
}

_OBLIGATION_PATTERNS = [
    r"\bshall\b", r"\bmust\b", r"\brequired\b", r"\bobligation\b",
    r"\bnot\s+to\b", r"\bshall\s+not\b", r"\bmay\s+not\b",
    r"\bwithout\s+(?:the\s+)?(?:express\s+)?(?:prior\s+)?written\s+(?:consent|approval)\b",
    r"\bpay\b", r"\bcompensate\b", r"\bliable\b",
]


def classify_clause(chunk):
    text = f"{chunk.section} {chunk.text}".lower()
    labels = []
    for category, patterns in _CATEGORY_RULES:
        if any(re.search(p, text) for p in patterns):
            labels.append(category)
    if not labels:
        labels.append("General")
    return labels


def obligation_score(text):
    lower = text.lower()
    hits = sum(1 for p in _OBLIGATION_PATTERNS if re.search(p, lower))
    return min(hits / 4.0, 1.0)


def semantic_relevance_score(chunk, query_profile):
    labels = classify_clause(chunk)
    score = 0.0
    for label in labels:
        if label in query_profile.expected_categories:
            score += 1.0
        for concept in query_profile.concepts:
            score += _CATEGORY_WEIGHTS.get(label, {}).get(concept, 0.0)

    text = chunk.text.lower()
    if "financial_commitment" in query_profile.concepts:
        if re.search(r"\baudit\b|\brecord\b|\bprocessing\b", text) and not re.search(r"\bpay\b|\bfee\b|\bcommission\b|\bliquidated\s+damages\b|\bcontract\s+value\b|\bliable\b", text):
            score -= 0.8
        score += obligation_score(text) * 0.6
    if query_profile.answer_type in {"list", "risk_table"}:
        score += obligation_score(text) * 0.4
    if "warranty" in query_profile.concepts and "Warranty/Representation" not in labels:
        score -= 2.0
    if "term" in query_profile.concepts and "Survival" in labels and "Term/Duration" not in labels:
        score -= 0.9
    return score, labels


def extract_obligation_summary(chunk):
    sentences = re.split(r"(?<=[.!?])\s+", chunk.text.strip())
    selected = []
    for sent in sentences:
        if obligation_score(sent) > 0 or re.search(r"\bliquidated\s+damages\b|\bcontract\s+value\b|\bvalid\s+up\s+to\b", sent.lower()):
            selected.append(sent.strip())
    return " ".join(selected[:2]) if selected else " ".join(sentences[:1]).strip()
