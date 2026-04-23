"""
Evidence Sufficiency Checker.
Determines if retrieved passages contain enough information to answer the query.
Returns: SUFFICIENT / CONFLICTING / INSUFFICIENT with a confidence score.
"""
import re

from src.pipeline.retriever import expand_query_keywords, clause_keyword_overlap


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


def _is_yes_no_query(query):
    q = query.strip().lower()
    return bool(re.match(r"^(is|are|can|could|may|does|do|did|has|have|must|should|will|would)\b", q))


def _strict_topic_match(query, chunk):
    """Intent-specific clause match so secondary boost terms do not create false positives."""
    q = query.lower()
    text = f"{chunk.section} {chunk.text[:700]}".lower()
    topic_groups = [
        (("warranty", "warranties", "guarantee"), ("warranty", "warranties", "warrants", "guarantee", "representations")),
        (("duration", "how long", "term"), ("duration", "term", "period", "effective", "commencement", "expiration", "expires", "year", "month", "day")),
        (("outside india", "india"), ("india", "outside india", "cross-border", "transfer")),
        (("data", "personal data"), ("data", "personal data", "processor", "controller", "privacy")),
        (("liability", "damages", "cap"), ("liability", "liable", "damages", "cap", "limit", "limitation")),
        (("termination", "terminate"), ("termination", "terminate", "notice", "breach")),
    ]
    for triggers, required_terms in topic_groups:
        if any(t in q for t in triggers):
            return any(term in text for term in required_terms)
    return True


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


def _has_conflicting_evidence(retrieved_chunks):
    """Conservative contradiction detector. Only escalates when both poles appear."""
    text = "\n".join(r["chunk"].text.lower() for r in retrieved_chunks)
    for action in ["assign", "transfer", "disclose", "share", "terminate", "indemnify"]:
        positive = (
            re.search(rf"\bmay\s+{action}\b", text)
            or re.search(rf"\bshall\s+{action}\b", text)
            or re.search(rf"\b{action}\w*\s+is\s+(?:permitted|allowed)\b", text)
        )
        negative = (
            re.search(rf"\bmay\s+not\s+{action}\b", text)
            or re.search(rf"\bshall\s+not\s+{action}\b", text)
            or re.search(rf"\bmust\s+not\s+{action}\b", text)
            or re.search(rf"\b{action}\w*\s+is\s+prohibited\b", text)
        )
        if positive and negative:
            return True
    return False


def check_evidence_sufficiency(query, retrieved_chunks, min_score=0.28):
    """
    Determine if the retrieved evidence is sufficient to answer the query.

    Args:
        query: the user's question
        retrieved_chunks: list of dicts with 'chunk' key containing Chunk objects
        min_score: threshold for SUFFICIENT classification

    Returns:
        dict with keys: decision, confidence, reasoning, intent
    """
    intent = _classify_query_intent(query)
    query_keywords = sorted(expand_query_keywords(query))

    if not retrieved_chunks:
        return {
            "decision": "INSUFFICIENT",
            "confidence": 0.0,
            "reasoning": "No relevant passages were retrieved from the document.",
            "intent": intent,
            "has_relevant_clause": False,
            "relevant_clause_count": 0,
            "query_keywords": query_keywords,
            "top_score": 0.0,
            "is_yes_no": _is_yes_no_query(query),
        }

    combined_text = " ".join(r["chunk"].text for r in retrieved_chunks)

    lexical_overlap = _compute_lexical_overlap(query, combined_text)
    legal_signal = _check_legal_content_signals(combined_text)
    retrieval_score = max(r["score"] for r in retrieved_chunks)
    clause_matches = [
        max(float(r.get("keyword_overlap", 0.0)), clause_keyword_overlap(query, r["chunk"]))
        for r in retrieved_chunks
    ]
    best_clause_match = max(clause_matches) if clause_matches else 0.0
    relevant_clause_count = sum(
        1
        for s, r in zip(clause_matches, retrieved_chunks)
        if s >= 0.18 and _strict_topic_match(query, r["chunk"])
    )
    has_relevant_clause = relevant_clause_count > 0

    combined_score = (
        0.40 * best_clause_match
        + 0.25 * lexical_overlap
        + 0.15 * legal_signal
        + 0.20 * min(max(retrieval_score, 0.0), 1.0)
    )

    top_score_threshold = 0.18
    conflicting_evidence = _has_conflicting_evidence(retrieved_chunks)

    # NO EVIDENCE != ESCALATE. If the retrieved text does not actually
    # contain the user's topic keywords, the correct behavior is NOT_FOUND.
    if not has_relevant_clause:
        decision = "INSUFFICIENT"
    elif retrieval_score < top_score_threshold and combined_score < min_score:
        decision = "INSUFFICIENT"
    elif conflicting_evidence:
        decision = "CONFLICTING"
    else:
        decision = "SUFFICIENT"

    reasons = []
    if has_relevant_clause:
        reasons.append(f"Clause match check passed ({relevant_clause_count} relevant)")
    else:
        reasons.append("Clause match check failed")
    if lexical_overlap > 0.5:
        reasons.append("Strong query-evidence term overlap")
    elif lexical_overlap < 0.2:
        reasons.append("Low query-evidence term overlap")
    if legal_signal > 0.6:
        reasons.append("Rich legal content detected in evidence")
    if retrieval_score > 0.3:
        reasons.append(f"High retrieval confidence ({retrieval_score:.2f})")
    if conflicting_evidence:
        reasons.append("Potentially conflicting evidence detected")

    return {
        "decision": decision,
        "confidence": round(combined_score, 3),
        "reasoning": "; ".join(reasons) if reasons else "Marginal evidence quality",
        "intent": intent,
        "has_relevant_clause": has_relevant_clause,
        "relevant_clause_count": relevant_clause_count,
        "query_keywords": query_keywords,
        "top_score": round(float(retrieval_score), 3),
        "clause_match_score": round(float(best_clause_match), 3),
        "conflicting_evidence": conflicting_evidence,
        "is_yes_no": _is_yes_no_query(query),
    }
