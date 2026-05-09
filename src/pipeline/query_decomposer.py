"""
Query decomposition for analytical prompts.

Breaks high-level analytical questions into narrower intents so retrieval can
target one concept at a time.
"""
import re


_CONCEPT_TO_SUBQUERY = {
    "financial_commitment": "What payment obligations, fees, commissions, penalties, or compensation duties are defined?",
    "liability": "What liabilities, remedies, caps, or indemnification obligations are stated?",
    "warranty": "What warranties or representations are provided or disclaimed?",
    "confidentiality": "What confidentiality obligations, disclosure limits, and exceptions are specified?",
    "data_use": "What data processing, storage, transfer, and usage restrictions are defined?",
    "sharing": "What sharing/disclosure permissions or restrictions apply to third parties?",
    "term": "What is the contract term, validity period, and expiry behavior?",
    "survival": "Which obligations survive termination or expiry?",
    "restriction": "What restrictions, prohibitions, or consent requirements are imposed?",
    "risk": "Which clauses create high legal, commercial, or operational risk?",
}


def _split_enumerated_items(query):
    parts = re.split(r"[;\n]|\b(?:and|also|plus)\b", query, flags=re.IGNORECASE)
    out = []
    for p in parts:
        cleaned = p.strip(" .,:-\t")
        if len(cleaned) >= 16:
            out.append(cleaned)
    return out


def decompose_query(query, query_profile):
    """
    Return a list of focused sub-queries.

    Factual queries keep single-shot behavior.
    Analytical/comparative/risk queries expand into concept-aligned tasks.
    """
    if query_profile.query_kind == "factual":
        return [query]

    sub_queries = []
    for concept in query_profile.concepts:
        prompt = _CONCEPT_TO_SUBQUERY.get(concept)
        if prompt:
            sub_queries.append(prompt)

    # Preserve explicit multi-part asks from the user wording.
    for item in _split_enumerated_items(query):
        if item.lower() != query.lower():
            sub_queries.append(item)

    # Always include the original query as an aggregation target.
    sub_queries.append(query)

    deduped = []
    seen = set()
    for sq in sub_queries:
        key = sq.lower().strip()
        if key and key not in seen:
            deduped.append(sq)
            seen.add(key)
    return deduped
