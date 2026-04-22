"""
Post-Generation Grounding Verifier.
Checks that every claim in the generated answer is supported by the retrieved evidence.
Returns: VERIFIED / PARTIALLY_VERIFIED / REJECTED
"""
import re
from difflib import SequenceMatcher


def _extract_claims(answer_text):
    """Extract individual claims/sentences from the answer."""
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and not s.startswith("**Note"):
            claims.append(s)
    return claims


def _claim_supported(claim, evidence_texts, threshold=0.25):
    """Check if a claim has sufficient support in the evidence."""
    claim_tokens = set(re.findall(r"\w{3,}", claim.lower()))
    skip = {"the", "and", "for", "with", "this", "that", "from",
            "are", "was", "were", "has", "have", "had", "been",
            "will", "would", "shall", "should", "can", "could",
            "may", "might", "not", "but", "also", "into", "your",
            "our", "their", "any", "all", "some", "each", "its"}
    claim_tokens -= skip

    if not claim_tokens:
        return True, 1.0

    best_overlap = 0.0
    for ev_text in evidence_texts:
        ev_tokens = set(re.findall(r"\w{3,}", ev_text.lower())) - skip
        if not ev_tokens:
            continue
        overlap = len(claim_tokens & ev_tokens) / len(claim_tokens)
        best_overlap = max(best_overlap, overlap)

        # Also check substring similarity
        ratio = SequenceMatcher(None, claim.lower(), ev_text.lower()).ratio()
        best_overlap = max(best_overlap, ratio)

    return best_overlap >= threshold, best_overlap


def verify_grounding(answer_data, evidence_chunks):
    """
    Verify that the generated answer is grounded in the retrieved evidence.

    Args:
        answer_data: dict from the generator (must have 'answer' key)
        evidence_chunks: list of retrieved chunk dicts

    Returns:
        dict with: verdict, supported_ratio, unsupported_claims, details
    """
    answer_text = answer_data.get("answer", "")
    decision = answer_data.get("decision", "ANSWER")

    if decision == "NOT_FOUND":
        return {
            "verdict": "VERIFIED",
            "supported_ratio": 1.0,
            "unsupported_claims": [],
            "details": "Refusal response — no grounding check needed.",
        }

    if not answer_text or not evidence_chunks:
        return {
            "verdict": "REJECTED",
            "supported_ratio": 0.0,
            "unsupported_claims": [],
            "details": "Empty answer or no evidence provided.",
        }

    evidence_texts = [r["chunk"].text for r in evidence_chunks]
    claims = _extract_claims(answer_text)

    if not claims:
        return {
            "verdict": "VERIFIED",
            "supported_ratio": 1.0,
            "unsupported_claims": [],
            "details": "No verifiable claims extracted.",
        }

    supported = 0
    unsupported = []
    details = []

    for claim in claims:
        is_supported, score = _claim_supported(claim, evidence_texts)
        if is_supported:
            supported += 1
            details.append(f"SUPPORTED ({score:.2f}): {claim[:80]}...")
        else:
            unsupported.append(claim)
            details.append(f"UNSUPPORTED ({score:.2f}): {claim[:80]}...")

    ratio = supported / len(claims) if claims else 0.0

    if ratio >= 0.8:
        verdict = "VERIFIED"
    elif ratio >= 0.5:
        verdict = "PARTIALLY_VERIFIED"
    else:
        verdict = "REJECTED"

    return {
        "verdict": verdict,
        "supported_ratio": round(ratio, 3),
        "unsupported_claims": unsupported,
        "details": "; ".join(details[:5]),
    }
