"""
Post-retrieval semantic filtering and reranking.

This layer uses the query profile and legal ontology to remove clauses that are
lexically related but legally irrelevant.
"""
from src.pipeline.legal_ontology import semantic_relevance_score
import re
import os


_CROSS_ENCODER = None

_CONCEPT_EVIDENCE_PATTERNS = {
    "financial_commitment": r"\bpay\b|\bfees?\b|\binvoice\b|\bpayable\b|\bdue\b|\bliquidated\s+damages\b",
    "termination": r"\bterminat\w*\b|\bcure\b|\bnotice\b|\bbreach\b",
    "subcontractor": r"\bsubcontract\w*\b|\bcontractor\w*\b|\bdeemed\b|\battribut\w+\b|\bresponsib\w+\s+for\s+the\s+acts?\b",
    "gross_negligence": r"\bgross\s+negligence\b|\bwillful\s+misconduct\b|\bwilful\s+misconduct\b|\bfraud\b",
    "survival": r"\bsurviv\w*\b|\bfollowing\s+termination\b|\bafter\s+termination\b",
    "liability": r"\bliabil\w*\b|\bcap\b|\blimit\w*\b|\bindemn\w*\b|\bdamages?\b",
    "confidentiality": r"\bconfidential\w*\b|\bdisclos\w*\b|\bneed\s+to\s+know\b",
    "data_use": r"\bdata\b|\bprivacy\b|\bsecurity\b|\bpersonal\s+data\b|\bprocessing\b",
}


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
    concepts = set(query_profile.concepts)
    if len(concepts) <= 1:
        multi_concept = False
    else:
        multi_concept = True

    checks = []

    if "financial_commitment" in concepts:
        allowed = {"Financial/Payment", "Liability/Remedies", "Indemnification"}
        checks.append(bool(labels & allowed) and bool(re.search(
            r"\bpay\b|\bfees?\b|\bcommission\b|\binvoice\b|\bcharges?\b|\bcosts?\b|"
            r"\bliquidated\s+damages\b|\bcontract\s+value\b|\bliabilit|\bcompensate\b|\bindemnif",
            text,
        )))

    if "termination" in concepts:
        allowed = {"Termination", "Survival", "Term/Duration"}
        checks.append(bool(labels & allowed) and bool(re.search(r"terminat|cure|breach|survive|notice|expiry|expiration", text)))

    if "subcontractor" in concepts:
        allowed = {"Liability/Remedies", "Indemnification", "Data Protection", "Force Majeure"}
        checks.append(bool(labels & allowed) and bool(re.search(r"subcontract|contractor|deemed|attribut|acts?\s+undertaken|responsib\w+\s+for\s+the\s+acts", text)))

    if "gross_negligence" in concepts:
        allowed = {"Liability/Remedies", "Indemnification", "Data Protection", "Confidentiality"}
        checks.append(bool(labels & allowed) and bool(re.search(r"gross\s+negligence|willful\s+misconduct|wilful\s+misconduct|fraud|notwithstanding|except", text)))

    if "survival" in concepts:
        allowed = {"Survival", "Termination", "Confidentiality", "Data Protection"}
        checks.append(bool(labels & allowed) and bool(re.search(r"surviv|after\s+termination|following\s+termination|continue\s+after", text)))

    if "liability" in concepts:
        allowed = {"Liability/Remedies", "Indemnification", "Warranty/Representation"}
        checks.append(bool(labels & allowed) and bool(re.search(r"liabil|damage|indemn|cap|limit|maximum|negligence|negligent", text)))

    if "confidentiality" in concepts:
        allowed = {"Confidentiality", "Data Protection", "Use Restriction"}
        checks.append(bool(labels & allowed) and bool(re.search(r"confident|disclos|secret|data|private|personally identifiable", text)))

    if "data_use" in concepts:
        allowed = {"Data Protection", "Confidentiality", "Use Restriction", "Intellectual Property"}
        checks.append(bool(labels & allowed) and bool(re.search(r"data|processing|storage|privacy|security|breach|personal", text)))

    if "warranty" in concepts:
        checks.append("Warranty/Representation" in labels or "Liability/Remedies" in labels)

    if not checks:
        return True
    if multi_concept:
        return any(checks)
    return checks[0]


def _item_supports_concept(item, concept):
    text = f"{item['chunk'].section} {item['chunk'].text}".lower()
    pattern = _CONCEPT_EVIDENCE_PATTERNS.get(concept)
    if not pattern:
        return False
    return re.search(pattern, text) is not None


def filter_and_rerank(query, retrieved, query_profile, keep_k=None):
    if not retrieved:
        return []

    cross_encoder = _get_cross_encoder()
    
    # Create a dense rerank query that focuses on legal concepts rather than story words
    rerank_query = " ".join(query_profile.concepts) + " " + " ".join(query_profile.expected_categories)
    if not rerank_query.strip():
        rerank_query = query[:200]
    else:
        # Add core parts of the original query to maintain some context
        rerank_query += " " + query[:150]

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
            pair = [[rerank_query, item["chunk"].text[:1200]]]
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

    # Limit same clause text to max 2 occurrences to improve diversity
    diversity_filtered = []
    text_counts = {}
    for s in scored:
        txt = s["chunk"].text.strip()
        text_counts[txt] = text_counts.get(txt, 0) + 1
        if text_counts[txt] <= 2:
            diversity_filtered.append(s)
    scored = diversity_filtered

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

    if query_profile.allow_multi_clause and len(query_profile.concepts) > 1:
        chosen = list(filtered[:keep_k])
        chosen_ids = {id(item["chunk"]) for item in chosen}
        for concept in query_profile.concepts:
            if concept == "general":
                continue
            if any(_item_supports_concept(item, concept) for item in chosen):
                continue
            for candidate in scored:
                if id(candidate["chunk"]) in chosen_ids:
                    continue
                if _item_supports_concept(candidate, concept):
                    chosen.append(candidate)
                    chosen_ids.add(id(candidate["chunk"]))
                    break
        chosen.sort(key=lambda x: x["final_score"], reverse=True)
        return chosen[: max(keep_k, min(len(chosen), keep_k + 3))]

    return filtered[:keep_k]
