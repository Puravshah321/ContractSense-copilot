"""
Legal Ontology and Clause Taxonomy — upgraded.

Improvements implemented:
  #3  Legal ontology / clause taxonomy with metadata labels
  #4  Legal synonym expansion (stop-work → suspend services etc.)
  #9  Confidence calibration helpers (ambiguity, obligation density)
"""
import re


# ── Category rules (patterns → legal label) ─────────────────────────
_CATEGORY_RULES = [
    ("Warranty/Representation",   [r"\bwarrant(?:y|ies|s)?\b", r"\brepresent(?:s|ations?)?\b", r"\bguarantee\b"]),
    ("Financial/Payment",         [r"\bpay(?:ment|able)?\b", r"\bfees?\b", r"\bcommission\b", r"\binvoice\b", r"\bcharges?\b", r"\bcosts?\b", r"\bexpenses?\b"]),
    ("Liability/Remedies",        [r"\bliabilit", r"\bdamages?\b", r"\bliquidated\s+damages\b", r"\bremed(?:y|ies)\b", r"\bcontract\s+value\b", r"\bcompensate\b", r"\bgross\s+negligence\b", r"\bwillful\s+misconduct\b", r"\bwilful\s+misconduct\b"]),
    ("Indemnification",           [r"\bindemnif", r"\bhold\s+harmless\b", r"\bthird-party\s+claims?\b"]),
    ("SLA/Service Level",         [r"\bservice\s+level\b", r"\buptime\b", r"\bavailability\b", r"\bsla\b", r"\bresponse\s+time\b", r"\bdowntime\b", r"\bservice\s+credit\b"]),
    ("Termination",               [r"\bterminate\b", r"\btermination\b", r"\bcease\s+performance\b", r"\bstop\s+work\b", r"\bsuspend\s+services?\b", r"\bmaterial\s+breach\b", r"\bcure\b", r"\bnotice\s+of\s+breach\b"]),
    ("Term/Duration",             [r"\bterm\b", r"\bvalid\s+up\s+to\b", r"\bvalid\s+for\b", r"\bexpires?\b", r"\bexpiration\b", r"\beffective\s+date\b"]),
    ("Survival",                  [r"\bsurviv(?:e|al)\b", r"\bafter\s+(?:expiration|termination)\b", r"\bperpetuit"]),
    ("Confidentiality",           [r"\bconfidential", r"\bnon-disclosure\b", r"\bdisclos(?:e|ure)\b", r"\bneed\s+to\s+know\b"]),
    ("Permitted Disclosure",      [r"\bpermitted\s+disclos", r"\bmay\s+share\b", r"\bstqc\b", r"\bgovernment\s+entities\b"]),
    ("Data Protection",           [r"\bpersonal\s+data\b", r"\baudit\s+(?:information|data)\b", r"\boutside\s+india\b", r"\boutside\s+the\s+country\b", r"\bstorage\b", r"\bprocessing\b", r"\bsecurity\s+(?:obligations?|requirements?|incident)\b", r"\bransomware\b", r"\bprivacy\b"]),
    ("Use Restriction",           [r"\buse\b.*\bscope\b", r"\bnot\s+to\s+(?:make|retain|disclose|send|engage)\b", r"\bshall\s+not\b", r"\bwithout\s+(?:the\s+)?(?:express\s+)?(?:prior\s+)?written\s+(?:consent|approval)\b"]),
    ("Intellectual Property",     [r"\bintellectual\s+property\b", r"\bcopyright\b", r"\bpatent\b", r"\btrademark\b", r"\bwork\s+product\b", r"\bproprietary\b"]),
    ("Non-Compete/Non-Solicit",   [r"\bnon\s*-?\s*compet", r"\bnon\s*-?\s*solicitation\b", r"\bsolicit\b", r"\brestrictive\s+covenant\b"]),
    ("Force Majeure",             [r"\bforce\s+majeure\b", r"\bact\s+of\s+god\b", r"\bcircumstances\s+beyond\b"]),
    ("Dispute Resolution",        [r"\barbitration\b", r"\bgoverning\s+law\b", r"\bjurisdiction\b", r"\bdispute\b", r"\bmediation\b"]),
    ("Compliance",                [r"\bcomplian(?:ce|t)\b", r"\bregulat(?:ion|ory)\b", r"\bstatutory\b", r"\blaw\s+and\s+regulation\b"]),
    ("Assignment",                [r"\bassign(?:ment)?\b", r"\btransfer\s+of\s+(?:rights|obligations)\b", r"\bsuccessor\b"]),
]

# ── Legal synonym expansion (Improvement #4) ─────────────────────────
LEGAL_SYNONYMS = {
    "stop work":            ["suspend services", "cease performance", "termination rights", "material breach", "stop work"],
    "suspend":              ["suspend services", "cease performance", "stop work", "halt", "discontinue"],
    "compensation":         ["damages", "service credits", "refunds", "liability", "reimbursement", "indemnity", "remedies"],
    "damages":              ["compensation", "service credits", "refunds", "liability", "reimbursement", "liquidated damages"],
    "service credit":       ["compensation", "damages", "refund", "reimbursement", "sla credit", "service level"],
    "liability cap":        ["limitation of liability", "cap on damages", "maximum exposure", "liability limit", "shall not exceed"],
    "shall not exceed":     ["liability cap", "limitation of liability", "cap on damages", "maximum exposure"],
    "termination":          ["cease performance", "stop work", "suspend services", "end of agreement", "material breach"],
    "confidentiality":      ["non-disclosure", "trade secret", "proprietary information", "classified", "sensitive"],
    "non-disclosure":       ["confidentiality", "trade secret", "proprietary information", "sensitive"],
    "intellectual property": ["ip", "copyright", "patent", "trademark", "work product", "proprietary"],
    "warranty":             ["representation", "guarantee", "warrants", "fitness for purpose"],
    "indemnification":      ["indemnify", "hold harmless", "third-party claims", "indemnity"],
    "force majeure":        ["act of god", "beyond reasonable control", "extraordinary circumstances"],
    "dispute":              ["arbitration", "mediation", "jurisdiction", "governing law", "litigation"],
    "data breach":          ["security incident", "unauthorized access", "data loss", "privacy breach"],
    "assignment":           ["transfer of rights", "novation", "successor", "delegate"],
}

# ── Category weights for scoring (concept × label → weight) ─────────
_CATEGORY_WEIGHTS = {
    "Warranty/Representation":  {"warranty": 2.0},
    "Financial/Payment":        {"financial_commitment": 1.5},
    "Liability/Remedies":       {"liability": 1.3, "financial_commitment": 0.8},
    "Indemnification":          {"liability": 1.1},
    "SLA/Service Level":        {"liability": 1.0, "financial_commitment": 0.9},
    "Termination":              {"termination": 1.8},
    "Term/Duration":            {"term": 1.8},
    "Survival":                 {"survival": 1.6},
    "Confidentiality":          {"confidentiality": 1.3, "sharing": 1.0, "data_use": 0.8},
    "Permitted Disclosure":     {"sharing": 1.4},
    "Data Protection":          {"data_use": 1.4, "sharing": 0.6},
    "Use Restriction":          {"restriction": 1.3, "data_use": 1.2, "sharing": 1.0},
    "Intellectual Property":    {"ip": 1.5},
    "Non-Compete/Non-Solicit":  {"restriction": 0.8},
    "Force Majeure":            {},
    "Dispute Resolution":       {},
    "Compliance":               {"restriction": 0.5},
    "Assignment":               {"restriction": 0.6},
}

# ── Obligation patterns ─────────────────────────────────────────────
_OBLIGATION_PATTERNS = [
    r"\bshall\b", r"\bmust\b", r"\brequired\b", r"\bobligation\b",
    r"\bnot\s+to\b", r"\bshall\s+not\b", r"\bmay\s+not\b",
    r"\bwithout\s+(?:the\s+)?(?:express\s+)?(?:prior\s+)?written\s+(?:consent|approval)\b",
    r"\bpay\b", r"\bcompensate\b", r"\bliable\b",
    r"\bwarrants?\b", r"\brepresents?\b",
]

# ── Ambiguity signals (Improvement #9 – confidence calibration) ─────
_AMBIGUITY_PATTERNS = [
    r"\breasonable\b", r"\bcommercially\s+reasonable\b", r"\bbest\s+efforts?\b",
    r"\bto\s+the\s+extent\b", r"\bas\s+applicable\b", r"\bmay\s+include\b",
    r"\bincluding\s+but\s+not\s+limited\s+to\b", r"\binter\s+alia\b",
    r"\bat\s+(?:its|the\s+(?:party's|company's))\s+(?:sole\s+)?discretion\b",
    r"\bnotwithstanding\b",
]


# ── Public API ──────────────────────────────────────────────────────

def classify_clause(chunk):
    """Return list of legal category labels for a chunk."""
    text = f"{chunk.section} {chunk.text}".lower()
    labels = []
    for category, patterns in _CATEGORY_RULES:
        if any(re.search(p, text) for p in patterns):
            labels.append(category)
    return labels or ["General"]


def obligation_score(text):
    """0..1 density of obligation language in text."""
    lower = text.lower()
    hits = sum(1 for p in _OBLIGATION_PATTERNS if re.search(p, lower))
    return min(hits / 4.0, 1.0)


def ambiguity_score(text):
    """0..1 — higher means more vague/discretionary language."""
    lower = text.lower()
    hits = sum(1 for p in _AMBIGUITY_PATTERNS if re.search(p, lower))
    return min(hits / 3.0, 1.0)


def expand_legal_synonyms(query):
    """Return set of legal synonym terms to add to query expansion."""
    q = query.lower()
    extras = set()
    for trigger, synonyms in LEGAL_SYNONYMS.items():
        if trigger in q:
            extras.update(synonyms)
    return extras


def semantic_relevance_score(chunk, query_profile):
    """Score a chunk against query_profile based on taxonomy alignment."""
    labels = classify_clause(chunk)
    score = 0.0
    for label in labels:
        if label in query_profile.expected_categories:
            score += 1.0
        for concept in query_profile.concepts:
            score += _CATEGORY_WEIGHTS.get(label, {}).get(concept, 0.0)

    text = chunk.text.lower()
    if "financial_commitment" in query_profile.concepts:
        if re.search(r"\baudit\b|\brecord\b|\bprocessing\b", text) and \
           not re.search(r"\bpay\b|\bfee\b|\bcommission\b|\bliquidated\s+damages\b|\bcontract\s+value\b|\bliable\b", text):
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
    """Extract the most obligation-dense sentences from a clause."""
    sentences = re.split(r"(?<=[.!?])\s+", chunk.text.strip())
    selected = []
    for sent in sentences:
        if obligation_score(sent) > 0 or re.search(
            r"\bliquidated\s+damages\b|\bcontract\s+value\b|\bvalid\s+up\s+to\b", sent.lower()
        ):
            selected.append(sent.strip())
    return " ".join(selected[:2]) if selected else " ".join(sentences[:1]).strip()


def calibrate_confidence(base_score, retrieved_chunks, n_assumptions=0):
    """
    Improvement #9: Confidence calibration.
    Adjusts base_score downward based on:
      - average ambiguity of retrieved clauses
      - number of assumptions made in reasoning
      - clause agreement (single strong clause = higher conf)
    Returns float 0..1.
    """
    if not retrieved_chunks:
        return 0.0
    avg_ambiguity = sum(ambiguity_score(r["chunk"].text) for r in retrieved_chunks) / len(retrieved_chunks)
    assumption_penalty = min(n_assumptions * 0.08, 0.3)
    calibrated = base_score * (1.0 - 0.4 * avg_ambiguity) - assumption_penalty
    return round(max(0.0, min(calibrated, 1.0)), 3)
