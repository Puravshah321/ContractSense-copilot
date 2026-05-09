"""
Query Understanding Layer — upgraded.

Improvements implemented:
  #1  Explicit query intent classification (factual / yes-no / analytical /
      financial / risk / compliance / extraction)
  #9  Confidence calibration signal in QueryProfile
"""
import re
from dataclasses import dataclass, asdict


@dataclass
class QueryProfile:
    query_kind: str          # factual | analytical | comparative | risk | extraction | compliance | financial
    answer_type: str         # yes_no | fact | list | risk_table | comparison | extraction | financial
    concepts: list           # list of legal concept labels
    expected_categories: list
    allow_multi_clause: bool
    retrieval_depth: int
    confidence: float

    def to_dict(self):
        return asdict(self)


# ── Concept detection rules ─────────────────────────────────────────
_CONCEPT_RULES = [
    ("financial_commitment",
     ["financial commitment", "commercial obligation", "fees", "fee", "commission",
      "payment", "pay", "paid", "invoice", "cost", "charge", "expense",
      "liquidated damages", "penalty", "contract value", "service credit", "sla credit",
      "compensation", "entitled", "how much"]),
    ("warranty",
     ["warranty", "warranties", "warrant", "guarantee", "representation", "fitness for purpose"]),
    ("liability",
     ["liability", "liable", "damages", "indemnify", "indemnification", "hold harmless",
      "cap", "limitation", "maximum exposure", "shall not exceed"]),
    ("confidentiality",
     ["confidential", "confidentiality", "non-disclosure", "disclose", "disclosure", "trade secret"]),
    ("data_use",
     ["data", "audit data", "training", "ai", "model", "processing", "storage",
      "analysis", "outside india", "outside the country", "personal data", "data breach"]),
    ("sharing",
     ["share", "sharing", "third party", "third-party", "external", "affiliate",
      "subsidiary", "associate", "transfer", "assign"]),
    ("sla",
     ["sla", "service level", "uptime", "availability", "response time", "downtime",
      "service credit", "performance metric", "down", "server failure", "outage"]),
    ("termination",
     ["terminate", "termination", "stop work", "suspend services", "cease performance",
      "material breach", "cure period", "notice of breach"]),
    ("ip",
     ["intellectual property", "copyright", "patent", "trademark", "work product",
      "proprietary", "ownership", "assignment of ip"]),
    ("term",
     ["duration", "term", "valid", "validity", "expiration", "expiry", "effective date", "signing"]),
    ("survival",
     ["survive", "survival", "continue", "after termination", "after expiration"]),
    ("restriction",
     ["restrict", "restriction", "prohibit", "not to", "shall not", "must not",
      "without consent", "non-compete", "non-solicitation"]),
    ("compliance",
     ["comply", "compliance", "regulation", "regulatory", "statutory", "law",
      "gdpr", "ccpa", "audit rights"]),
    ("risk",
     ["risk", "burden", "exposure", "concern", "highest", "critical", "danger",
      "potential issue", "vulnerability"]),
    ("force_majeure",
     ["force majeure", "act of god", "beyond control", "extraordinary circumstances"]),
    ("dispute",
     ["dispute", "arbitration", "mediation", "jurisdiction", "governing law", "litigation"]),
]

_CATEGORY_MAP = {
    "financial_commitment": ["Financial/Payment", "Liability/Remedies", "Indemnification"],
    "warranty":             ["Warranty/Representation"],
    "liability":            ["Liability/Remedies", "Indemnification"],
    "confidentiality":      ["Confidentiality"],
    "data_use":             ["Data Protection", "Confidentiality", "Use Restriction"],
    "sharing":              ["Confidentiality", "Permitted Disclosure", "Use Restriction"],
    "sla":                  ["SLA/Service Level", "Liability/Remedies", "Financial/Payment"],
    "termination":          ["Termination"],
    "ip":                   ["Intellectual Property"],
    "term":                 ["Term/Duration"],
    "survival":             ["Survival", "Confidentiality"],
    "restriction":          ["Use Restriction", "Confidentiality", "Non-Compete/Non-Solicit"],
    "compliance":           ["Compliance", "Data Protection"],
    "risk":                 ["Liability/Remedies", "Data Protection", "Confidentiality",
                             "Use Restriction", "Financial/Payment"],
    "force_majeure":        ["Force Majeure"],
    "dispute":              ["Dispute Resolution"],
}


def _contains_any(text, needles):
    return any(n in text for n in needles)


def _is_yes_no(query):
    return bool(re.match(r"^\s*(is|are|can|could|may|does|do|did|has|have|must|should|will|would)\b", query.lower()))


def classify_query(query):
    """
    Classify the user query into a structured QueryProfile.

    Query kinds:
      - factual:      Single fact / yes-no lookup
      - analytical:   Multi-clause synthesis, reasoning required
      - comparative:  Compare clauses or positions
      - risk:         Risk-scan across the document
      - extraction:   Extract all clauses of a type
      - financial:    Financial analysis / cost computation
      - compliance:   Compliance / regulatory check
    """
    q = query.lower()
    concepts = []
    for concept, needles in _CONCEPT_RULES:
        if _contains_any(q, needles):
            concepts.append(concept)
    if not concepts:
        concepts = ["general"]

    # ── Kind classification (order matters — most specific first) ────
    if re.search(r"\b(extract|show\s+all\s+clauses|list\s+all\s+clauses|which\s+clauses|give\s+all\s+clauses|clause\s+numbers?)\b", q):
        query_kind = "extraction"
        answer_type = "extraction"
    elif re.search(r"\b(compare|difference|versus|vs\.?|better|between)\b", q):
        query_kind = "comparative"
        answer_type = "comparison"
    elif re.search(r"\b(comply|complian(?:ce|t)|regulat|statutory|gdpr|ccpa|audit\s+right)\b", q):
        query_kind = "compliance"
        answer_type = "list"
    elif re.search(r"\b(financial\s+(?:impact|exposure|commitment|cost|risk)|how\s+much|total\s+amount|monetary|dollar|rupee|budget)\b", q):
        query_kind = "financial"
        answer_type = "financial"
    elif re.search(r"\b(list|summarize|summary|all|what\s+are|which\s+obligations|financial\s+commitments|commitments|risks|burdens|impact|analy[sz]e|implications?|consequences?)\b", q):
        query_kind = "analytical"
        answer_type = "risk_table" if "risk" in concepts else "list"
    elif "risk" in concepts and not _is_yes_no(query):
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

    allow_multi_clause = query_kind in {"analytical", "comparative", "risk", "extraction", "compliance", "financial"} \
        or answer_type in {"list", "risk_table", "comparison", "extraction", "financial"}

    retrieval_depth = (
        16 if query_kind in {"analytical", "extraction", "risk"} else
        12 if query_kind in {"compliance", "financial", "comparative"} else
        5
    )
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
