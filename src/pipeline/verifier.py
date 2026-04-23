"""
Post-generation grounding verifier.

This is claim-level grounding, not a cosmetic percentage:
  1. split the answer into factual claims
  2. compare each claim against retrieved evidence
  3. report supported_claims / total_claims
"""
import re
from difflib import SequenceMatcher


_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was",
    "were", "has", "have", "had", "been", "will", "would", "shall",
    "should", "can", "could", "may", "might", "not", "but", "also",
    "into", "your", "our", "their", "any", "all", "some", "each",
    "its", "according", "clause", "section", "article", "page", "answer",
    "yes", "no", "risk", "citation", "decision",
}


def _extract_claims(answer_text):
    """Extract individual factual claims from answer text."""
    cleaned = re.sub(r"\*\*[^*]+:\*\*", " ", answer_text)
    cleaned = re.sub(r"^Answer:\s*(YES|NO|NOT_FOUND)\s*", " ", cleaned, flags=re.IGNORECASE)
    parts = []
    for line in cleaned.splitlines():
        line = line.strip(" -\t")
        if not line:
            continue
        parts.extend(re.split(r"(?<=[.!?])\s+", line))

    claims = []
    for part in parts:
        claim = part.strip()
        if len(claim) < 18:
            continue
        if claim.lower().startswith(("grounding", "evidence:", "citation:", "decision:")):
            continue
        if claim.lower().startswith("according to") and not re.search(r"\b(shall|must|may|will|is|are|does|continues|lasts|requires|states)\b", claim.lower()):
            continue
        claims.append(claim)
    return claims


def _tokens(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())) - _STOPWORDS


def _numbers(text):
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text.lower()))


def _evidence_sentences(evidence_texts):
    out = []
    for text in evidence_texts:
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 12)
    return out


def _claim_supported(claim, evidence_sentences, evidence_text, threshold=0.5):
    """Return (supported, score) for a single claim."""
    claim_tokens = _tokens(claim)
    claim_numbers = _numbers(claim)
    if not claim_tokens and not claim_numbers:
        return True, 1.0

    evidence_tokens = _tokens(evidence_text)
    evidence_numbers = _numbers(evidence_text)
    if claim_numbers and not claim_numbers <= evidence_numbers:
        return False, 0.0

    token_overlap = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
    best_sentence_similarity = 0.0
    for sentence in evidence_sentences:
        ratio = SequenceMatcher(None, claim.lower(), sentence.lower()).ratio()
        sent_tokens = _tokens(sentence)
        sent_overlap = len(claim_tokens & sent_tokens) / max(len(claim_tokens), 1)
        best_sentence_similarity = max(best_sentence_similarity, ratio, sent_overlap)

    score = max(token_overlap, best_sentence_similarity)
    if claim_numbers:
        score = max(score, token_overlap + 0.15)
    return score >= threshold, min(score, 1.0)


def verify_grounding(answer_data, evidence_chunks):
    """
    Verify that each generated claim is grounded in retrieved evidence.

    Returns:
        dict with verdict, supported_ratio, supported_claims, total_claims,
        unsupported_claims, and compact details.
    """
    answer_text = answer_data.get("answer", "")
    decision = answer_data.get("decision", "ANSWER")

    if decision == "NOT_FOUND":
        return {
            "verdict": "VERIFIED",
            "supported_ratio": 1.0,
            "supported_claims": 0,
            "total_claims": 0,
            "unsupported_claims": [],
            "details": "NOT_FOUND response; no factual answer claims to verify.",
        }

    if not answer_text or not evidence_chunks:
        return {
            "verdict": "REJECTED",
            "supported_ratio": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
            "unsupported_claims": [],
            "details": "Empty answer or no evidence provided.",
        }

    evidence_texts = [r["chunk"].text for r in evidence_chunks]
    evidence_text = "\n".join(evidence_texts)
    sentences = _evidence_sentences(evidence_texts)
    claims = _extract_claims(answer_text)

    if not claims:
        return {
            "verdict": "VERIFIED",
            "supported_ratio": 1.0,
            "supported_claims": 0,
            "total_claims": 0,
            "unsupported_claims": [],
            "details": "No verifiable factual claims extracted.",
        }

    supported = 0
    unsupported = []
    details = []
    for claim in claims:
        is_supported, score = _claim_supported(claim, sentences, evidence_text)
        if is_supported:
            supported += 1
            details.append(f"SUPPORTED ({score:.2f}): {claim[:80]}...")
        else:
            unsupported.append(claim)
            details.append(f"UNSUPPORTED ({score:.2f}): {claim[:80]}...")

    ratio = supported / len(claims)
    if ratio >= 0.9:
        verdict = "VERIFIED"
    elif ratio >= 0.67:
        verdict = "PARTIALLY_VERIFIED"
    else:
        verdict = "REJECTED"

    return {
        "verdict": verdict,
        "supported_ratio": round(ratio, 3),
        "supported_claims": supported,
        "total_claims": len(claims),
        "unsupported_claims": unsupported,
        "details": "; ".join(details[:5]),
    }
