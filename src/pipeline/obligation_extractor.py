"""
Obligation extractor.

Extracts actionable obligations/rights/constraints from selected clauses so
analytical answers are synthesized from normalized facts instead of raw text.
"""
import re


_TRIGGER_PATTERNS = [
    r"\bshall\b",
    r"\bmust\b",
    r"\brequired\s+to\b",
    r"\bmay\s+not\b",
    r"\bshall\s+not\b",
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?written\s+(?:consent|approval)\b",
    r"\bpay(?:ment|able)?\b",
    r"\bfees?\b",
    r"\bcommission\b",
    r"\binvoice\b",
    r"\bliabilit",
    r"\bindemnif",
]


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _is_actionable(sentence):
    lower = sentence.lower()
    return any(re.search(p, lower) for p in _TRIGGER_PATTERNS)


def _obligation_type(sentence):
    s = sentence.lower()
    if re.search(r"\bshall\s+not\b|\bmust\s+not\b|\bmay\s+not\b|\bwithout\s+.*written\s+(?:consent|approval)\b", s):
        return "restriction"
    if re.search(r"\bpay(?:ment|able)?\b|\bfees?\b|\bcommission\b|\bcosts?\b|\bcharges?\b", s):
        return "financial_obligation"
    if re.search(r"\bindemnif|\bliabilit|\bdamages?\b|\bremed(?:y|ies)\b", s):
        return "liability_obligation"
    if re.search(r"\bmay\b|\bentitled\b|\bright\b", s):
        return "right"
    return "obligation"


def _concept_match(concepts, sentence, labels):
    s = sentence.lower()
    label_set = set(labels or [])
    if "financial_commitment" in concepts:
        return bool(
            re.search(r"\bpay(?:ment|able)?\b|\bfees?\b|\bcommission\b|\binvoice\b|\bcosts?\b|\bcharges?\b", s)
            or "Financial/Payment" in label_set
        )
    if "liability" in concepts:
        return bool(re.search(r"\bliabilit|\bindemnif|\bdamages?\b", s) or {"Liability/Remedies", "Indemnification"} & label_set)
    if "confidentiality" in concepts:
        return bool(re.search(r"\bconfidential|\bdisclos(?:e|ure)\b", s) or "Confidentiality" in label_set)
    if "term" in concepts:
        return bool(re.search(r"\bterm\b|\bvalid\b|\bexpir", s) or "Term/Duration" in label_set)
    if "risk" in concepts:
        return True
    return True


def extract_obligations_from_chunk(chunk, taxonomy_labels, query_profile):
    obligations = []
    for sentence in _split_sentences(chunk.text):
        if not _is_actionable(sentence):
            continue
        if not _concept_match(query_profile.concepts, sentence, taxonomy_labels):
            continue
        obligations.append(
            {
                "text": sentence,
                "type": _obligation_type(sentence),
                "clause_id": chunk.clause_id,
                "section": chunk.section,
                "page": chunk.page,
                "taxonomy": list(taxonomy_labels or []),
            }
        )
    return obligations


def extract_obligations(evidence_chunks, query_profile, max_items=18):
    all_items = []
    for item in evidence_chunks:
        chunk = item["chunk"]
        labels = item.get("taxonomy", [])
        all_items.extend(extract_obligations_from_chunk(chunk, labels, query_profile))

    # De-duplicate by normalized sentence + clause.
    seen = set()
    unique = []
    for item in all_items:
        key = (item["clause_id"], re.sub(r"\s+", " ", item["text"].strip().lower()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:max_items]
