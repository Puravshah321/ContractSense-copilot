"""
Post-retrieval semantic filtering and reranking.

This layer uses the query profile and legal ontology to remove clauses that are
lexically related but legally irrelevant.
"""
from src.pipeline.legal_ontology import semantic_relevance_score
import re


def _passes_concept_constraints(item, query_profile):
    labels = set(item.get("taxonomy", []))
    text = item["chunk"].text.lower()

    if "financial_commitment" in query_profile.concepts:
        allowed = {"Financial/Payment", "Liability/Remedies", "Indemnification"}
        if not labels & allowed:
            return False
        return bool(re.search(
            r"\bpay\b|\bfees?\b|\bcommission\b|\binvoice\b|\bcharges?\b|\bcosts?\b|"
            r"\bliquidated\s+damages\b|\bcontract\s+value\b|\bliabilit|\bcompensate\b|\bindemnif",
            text,
        ))

    if "warranty" in query_profile.concepts:
        return "Warranty/Representation" in labels

    return True


def filter_and_rerank(query, retrieved, query_profile, keep_k=None):
    if not retrieved:
        return []

    scored = []
    for item in retrieved:
        semantic_score, labels = semantic_relevance_score(item["chunk"], query_profile)
        base_score = float(item.get("score", 0.0))
        keyword_overlap = float(item.get("keyword_overlap", 0.0))
        final_score = base_score + semantic_score + (0.25 * keyword_overlap)
        enriched = dict(item)
        enriched["semantic_score"] = round(semantic_score, 3)
        enriched["final_score"] = round(final_score, 3)
        enriched["taxonomy"] = labels
        scored.append(enriched)

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    if query_profile.expected_categories:
        aligned = [
            r for r in scored
            if set(r.get("taxonomy", [])) & set(query_profile.expected_categories)
            or r.get("semantic_score", 0.0) > 0.5
        ]
    else:
        aligned = scored

    if query_profile.allow_multi_clause:
        keep_k = keep_k or 6
        min_score = -0.25
    else:
        keep_k = keep_k or 3
        min_score = 0.05

    filtered = [
        r for r in aligned
        if r.get("semantic_score", 0.0) >= min_score and _passes_concept_constraints(r, query_profile)
    ]
    if not filtered:
        filtered = scored[:keep_k]
    return filtered[:keep_k]
