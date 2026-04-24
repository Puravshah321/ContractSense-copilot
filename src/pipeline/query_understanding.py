"""
Query understanding layer.

Maps a user question to:
  - query kind: factual / analytical / comparative / risk
  - answer type: yes_no / fact / list / risk_table / comparison
  - target legal concepts for retrieval and filtering
"""
import re
from dataclasses import dataclass, asdict


@dataclass
class QueryProfile:
    query_kind: str
    answer_type: str
    concepts: list
    expected_categories: list
    allow_multi_clause: bool
    retrieval_depth: int
    confidence: float

    def to_dict(self):
        return asdict(self)


_CONCEPT_RULES = [
    ("financial_commitment", ["financial commitment", "commercial obligation", "fees", "fee", "commission", "payment", "pay", "paid", "invoice", "cost", "charge", "expense", "liquidated damages", "penalty", "contract value"]),
    ("warranty", ["warranty", "warranties", "warrant", "guarantee"]),
    ("liability", ["liability", "liable", "damages", "indemnify", "indemnification", "hold harmless", "cap", "limitation"]),
    ("confidentiality", ["confidential", "confidentiality", "non-disclosure", "disclose", "disclosure"]),
    ("data_use", ["data", "audit data", "training", "ai", "model", "processing", "storage", "analysis", "outside india", "outside the country"]),
    ("sharing", ["share", "sharing", "third party", "third-party", "external", "affiliate", "subsidiary", "associate"]),
    ("term", ["duration", "term", "valid", "validity", "expiration", "expiry", "effective date", "signing"]),
    ("survival", ["survive", "survival", "continue", "after termination", "after expiration"]),
    ("restriction", ["restrict", "restriction", "prohibit", "not to", "shall not", "must not", "without consent"]),
    ("risk", ["risk", "burden", "exposure", "concern", "highest", "critical", "danger"]),
]

_CATEGORY_MAP = {
    "financial_commitment": ["Financial/Payment", "Liability/Remedies", "Indemnification"],
    "warranty": ["Warranty/Representation"],
    "liability": ["Liability/Remedies", "Indemnification"],
    "confidentiality": ["Confidentiality"],
    "data_use": ["Data Protection", "Confidentiality", "Use Restriction"],
    "sharing": ["Confidentiality", "Permitted Disclosure", "Use Restriction"],
    "term": ["Term/Duration"],
    "survival": ["Survival", "Confidentiality"],
    "restriction": ["Use Restriction", "Confidentiality", "Non-Solicitation"],
    "risk": ["Liability/Remedies", "Data Protection", "Confidentiality", "Use Restriction", "Financial/Payment"],
}


def _contains_any(text, needles):
    return any(n in text for n in needles)


def _is_yes_no(query):
    return bool(re.match(r"^\s*(is|are|can|could|may|does|do|did|has|have|must|should|will|would)\b", query.lower()))


def classify_query(query):
    q = query.lower()
    concepts = []
    for concept, needles in _CONCEPT_RULES:
        if _contains_any(q, needles):
            concepts.append(concept)

    if not concepts:
        concepts = ["general"]

    if re.search(r"\b(compare|difference|versus|vs\.?|better|between)\b", q):
        query_kind = "comparative"
        answer_type = "comparison"
    elif re.search(r"\b(list|summarize|summary|all|what are|which obligations|financial commitments|commitments|risks|burdens)\b", q):
        query_kind = "analytical"
        answer_type = "risk_table" if "risk" in concepts else "list"
    elif "risk" in concepts:
        query_kind = "risk"
        answer_type = "risk_table"
    elif _is_yes_no(query):
        query_kind = "factual"
        answer_type = "yes_no"
    else:
        query_kind = "factual"
        answer_type = "fact"

    expected_categories = []
    for concept in concepts:
        expected_categories.extend(_CATEGORY_MAP.get(concept, []))
    expected_categories = sorted(set(expected_categories))

    allow_multi_clause = query_kind in {"analytical", "comparative", "risk"} or answer_type in {"list", "risk_table", "comparison"}
    retrieval_depth = 12 if allow_multi_clause else 5
    confidence = min(1.0, 0.35 + 0.15 * len([c for c in concepts if c != "general"]))

    return QueryProfile(
        query_kind=query_kind,
        answer_type=answer_type,
        concepts=concepts,
        expected_categories=expected_categories,
        allow_multi_clause=allow_multi_clause,
        retrieval_depth=retrieval_depth,
        confidence=round(confidence, 3),
    )
