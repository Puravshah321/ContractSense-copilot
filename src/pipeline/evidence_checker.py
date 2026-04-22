"""
Evidence Sufficiency Checker.
Determines if retrieved passages contain enough information to answer the query.
Returns: SUFFICIENT / PARTIAL / INSUFFICIENT with a confidence score.
"""
import re


_QUERY_TYPE_KEYWORDS = {
    "risk":        ["risk", "danger", "concern", "problem", "issue", "flaw", "weakness"],
    "obligation":  ["must", "shall", "required", "obligation", "responsible", "duty"],
    "right":       ["can", "may", "allowed", "entitled", "right", "permission"],
    "definition":  ["what is", "define", "means", "definition", "constitute"],
    "timeline":    ["when", "how long", "duration", "period", "days", "months", "years", "deadline"],
    "consequence": ["happen", "result", "penalty", "breach", "violation", "failure"],
    "comparison":  ["difference", "compare", "versus", "vs", "between"],
}


def _classify_query_intent(query):
    q = query.lower()
    scores = {}
    for intent, keywords in _QUERY_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[intent] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)


def _compute_lexical_overlap(query, evidence_text):
    """Fraction of query terms found in evidence."""
    q_tokens = set(re.findall(r"\w+", query.lower()))
    e_tokens = set(re.findall(r"\w+", evidence_text.lower()))
    stop = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "shall", "should", "may", "might", "can", "could", "of", "in",
            "to", "for", "with", "on", "at", "by", "from", "this", "that",
            "it", "its", "or", "and", "not", "no", "if", "but", "what",
            "how", "when", "where", "who", "which", "we", "our", "i", "my"}
    q_tokens -= stop
    if not q_tokens:
        return 0.0
    overlap = q_tokens & e_tokens
    return len(overlap) / len(q_tokens)


def _check_legal_content_signals(evidence_text):
    """Check for presence of legal language indicating substantive content."""
    signals = [
        r"shall\b", r"must\b", r"may\b", r"herein", r"pursuant",
        r"notwithstanding", r"whereas", r"agrees?\b", r"warrants?\b",
        r"indemnif", r"terminat", r"liabilit", r"confidential",
        r"\d+\s*(?:days?|months?|years?)", r"\d+%", r"\$[\d,]+",
    ]
    text_lower = evidence_text.lower()
    hits = sum(1 for s in signals if re.search(s, text_lower))
    return min(hits / 5.0, 1.0)


def check_evidence_sufficiency(query, retrieved_chunks, min_score=0.3):
    """
    Determine if the retrieved evidence is sufficient to answer the query.

    Args:
        query: the user's question
        retrieved_chunks: list of dicts with 'chunk' key containing Chunk objects
        min_score: threshold for SUFFICIENT classification

    Returns:
        dict with keys: decision, confidence, reasoning, intent
    """
    if not retrieved_chunks:
        return {
            "decision": "INSUFFICIENT",
            "confidence": 0.0,
            "reasoning": "No relevant passages were retrieved from the document.",
            "intent": _classify_query_intent(query),
        }

    combined_text = " ".join(r["chunk"].text for r in retrieved_chunks)

    lexical_overlap = _compute_lexical_overlap(query, combined_text)
    legal_signal = _check_legal_content_signals(combined_text)
    retrieval_score = max(r["score"] for r in retrieved_chunks)

    combined_score = (
        0.35 * lexical_overlap
        + 0.35 * legal_signal
        + 0.30 * min(retrieval_score * 2, 1.0)
    )

    if combined_score >= min_score * 2:
        decision = "SUFFICIENT"
    elif combined_score >= min_score:
        decision = "PARTIAL"
    else:
        decision = "INSUFFICIENT"

    intent = _classify_query_intent(query)

    reasons = []
    if lexical_overlap > 0.5:
        reasons.append("Strong query-evidence term overlap")
    elif lexical_overlap < 0.2:
        reasons.append("Low query-evidence term overlap")
    if legal_signal > 0.6:
        reasons.append("Rich legal content detected in evidence")
    if retrieval_score > 0.3:
        reasons.append(f"High retrieval confidence ({retrieval_score:.2f})")

    return {
        "decision": decision,
        "confidence": round(combined_score, 3),
        "reasoning": "; ".join(reasons) if reasons else "Marginal evidence quality",
        "intent": intent,
    }
