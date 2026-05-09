"""
Post-retrieval semantic filtering and reranking.

This layer uses the query profile and legal ontology to remove clauses that are
lexically related but legally irrelevant.
"""
from src.pipeline.legal_ontology import semantic_relevance_score
import re
import os


_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    if os.environ.get("USE_CROSS_ENCODER", "0") != "1":
        return None
    try:
        from sentence_transformers import CrossEncoder
        model_name = os.environ.get("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")
        _CROSS_ENCODER = CrossEncoder(model_name)
        return _CROSS_ENCODER
    except Exception:
        return None


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

    cross_encoder = _get_cross_encoder()

    scored = []
    for item in retrieved:
        semantic_score, labels = semantic_relevance_score(item["chunk"], query_profile)
        base_score = float(item.get("score", 0.0))
        keyword_overlap = float(item.get("keyword_overlap", 0.0))
        intent_bonus = 0.0
        if query_profile.expected_categories and set(labels) & set(query_profile.expected_categories):
            intent_bonus += 0.35
        if query_profile.query_kind in {"analytical", "risk", "extraction"} and labels and labels[0] != "General":
            intent_bonus += 0.1

        rerank_score = 0.0
        if cross_encoder is not None:
            pair = [[query, item["chunk"].text[:1200]]]
            try:
                pred = cross_encoder.predict(pair)
                rerank_score = float(pred[0])
                # Normalize rough CE output range into 0-1-like contribution.
                rerank_score = max(min((rerank_score + 5.0) / 10.0, 1.0), -1.0)
            except Exception:
                rerank_score = 0.0

        final_score = base_score + semantic_score + (0.25 * keyword_overlap) + intent_bonus + (0.45 * rerank_score)
        enriched = dict(item)
        enriched["semantic_score"] = round(semantic_score, 3)
        enriched["intent_bonus"] = round(intent_bonus, 3)
        enriched["rerank_score"] = round(rerank_score, 3)
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
