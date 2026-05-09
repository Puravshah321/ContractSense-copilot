"""
Evidence Extractor — NEW module.

Improvement #6: Evidence extraction and normalization.
Extracts structured legal facts from raw clause text before
passing to the reasoning layer. Produces machine-readable
fact summaries instead of raw text dumps.

Extracted fact types:
  - explicit_right
  - obligation
  - prohibition
  - deadline
  - penalty
  - liability_cap
  - condition
  - exception
  - notice_period
  - monetary_value
"""
import re
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LegalFact:
    fact_type: str          # obligation | right | prohibition | deadline | penalty | liability_cap | condition | exception | notice_period | monetary
    text: str               # the exact clause sentence
    clause_id: str
    section: str
    page: int
    value: Optional[str] = None  # extracted number/date/amount if present

    def to_dict(self):
        return asdict(self)


# ── Extraction patterns ────────────────────────────────────────────
_OBLIGATION_RE   = re.compile(r"\b(shall|must|is\s+required\s+to|agrees?\s+to|undertakes?\s+to)\b", re.I)
_RIGHT_RE        = re.compile(r"\b(may|is\s+entitled\s+to|has\s+the\s+right\s+to|is\s+permitted\s+to|can\s+at\s+its\s+option)\b", re.I)
_PROHIBITION_RE  = re.compile(r"\b(shall\s+not|must\s+not|may\s+not|is\s+prohibited\s+from|not\s+to|is\s+not\s+permitted\s+to)\b", re.I)
_DEADLINE_RE     = re.compile(r"\bwithin\s+(\d+)\s*(business\s+)?days?\b|\bno\s+later\s+than\b|\bby\s+(?:the\s+)?\d{1,4}\b", re.I)
_NOTICE_RE       = re.compile(r"\b(\d+)\s*(?:calendar\s+|business\s+)?days?\s+(?:written\s+)?notice\b", re.I)
_PENALTY_RE      = re.compile(r"\bliquidated\s+damages?\b|\bpenalt(?:y|ies)\b|\bfine\b|\bforfeiture\b", re.I)
_LIABILITY_CAP_RE = re.compile(r"\bshall\s+not\s+exceed\b|\bmaximum\s+(?:aggregate\s+)?liability\b|\bcap(?:ped)?\s+(?:at|to)\b", re.I)
_CONDITION_RE    = re.compile(r"\bif\s+and\s+only\s+if\b|\bprovided\s+that\b|\bsubject\s+to\b|\bconditioned\s+(?:on|upon)\b|\bupon\s+(?:the\s+)?occurrence\b", re.I)
_EXCEPTION_RE    = re.compile(r"\bnotwithstanding\b|\bexcept\s+(as|for|where)\b|\bother\s+than\b|\bunless\b|\bdoes\s+not\s+include\b", re.I)
_MONETARY_RE     = re.compile(r"(?:\$|USD|INR|EUR|GBP|Rs\.?)\s*[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s*(?:percent|%|lakh|crore|million|billion)\b", re.I)
_TIME_RE         = re.compile(r"\b(\d+)\s+(?:calendar\s+|business\s+)?(?:days?|months?|years?)\b", re.I)


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if len(s.strip()) > 10]


def _extract_value(sentence):
    m = _MONETARY_RE.search(sentence) or _TIME_RE.search(sentence) or _NOTICE_RE.search(sentence)
    return m.group(0).strip() if m else None


def extract_legal_facts(chunk):
    """
    Extract structured legal facts from a single clause chunk.
    Returns list of LegalFact objects.
    """
    facts = []
    sentences = _split_sentences(chunk.text)
    for sent in sentences:
        fact_type = None
        if _LIABILITY_CAP_RE.search(sent):
            fact_type = "liability_cap"
        elif _PENALTY_RE.search(sent):
            fact_type = "penalty"
        elif _DEADLINE_RE.search(sent) or _NOTICE_RE.search(sent):
            fact_type = "notice_period" if _NOTICE_RE.search(sent) else "deadline"
        elif _EXCEPTION_RE.search(sent):
            fact_type = "exception"
        elif _CONDITION_RE.search(sent):
            fact_type = "condition"
        elif _PROHIBITION_RE.search(sent):
            fact_type = "prohibition"
        elif _OBLIGATION_RE.search(sent):
            fact_type = "obligation"
        elif _RIGHT_RE.search(sent):
            fact_type = "explicit_right"

        if fact_type:
            facts.append(LegalFact(
                fact_type=fact_type,
                text=sent,
                clause_id=chunk.clause_id,
                section=chunk.section,
                page=chunk.page,
                value=_extract_value(sent),
            ))

    # If no structured facts found, add the whole text as a general obligation
    if not facts and chunk.text.strip():
        facts.append(LegalFact(
            fact_type="obligation",
            text=chunk.text.strip()[:400],
            clause_id=chunk.clause_id,
            section=chunk.section,
            page=chunk.page,
            value=None,
        ))
    return facts


def extract_facts_from_evidence(retrieved_chunks):
    """
    Extract structured facts from all retrieved chunks.
    Returns a flat list of LegalFact objects sorted by fact_type priority.
    """
    _PRIORITY = {
        "liability_cap": 0, "penalty": 1, "notice_period": 2,
        "deadline": 3, "prohibition": 4, "obligation": 5,
        "explicit_right": 6, "condition": 7, "exception": 8,
    }
    all_facts = []
    for item in retrieved_chunks:
        all_facts.extend(extract_legal_facts(item["chunk"]))
    return sorted(all_facts, key=lambda f: _PRIORITY.get(f.fact_type, 9))


def normalize_facts_to_summary(facts):
    """
    Produce a normalized, machine-readable text summary from facts.
    This is what gets fed to the legal reasoning / synthesis layer.
    """
    if not facts:
        return "No structured legal facts could be extracted."
    lines = []
    by_type = {}
    for f in facts:
        by_type.setdefault(f.fact_type, []).append(f)

    priority = ["liability_cap", "penalty", "notice_period", "deadline",
                "prohibition", "obligation", "explicit_right", "condition", "exception"]

    for ft in priority:
        items = by_type.get(ft, [])
        if not items:
            continue
        label = ft.replace("_", " ").title()
        lines.append(f"\n[{label}]")
        for item in items[:3]:
            val_str = f" (Value: {item.value})" if item.value else ""
            lines.append(f"  • {item.text}{val_str} [{item.clause_id}, {item.section}]")

    return "\n".join(lines).strip()
