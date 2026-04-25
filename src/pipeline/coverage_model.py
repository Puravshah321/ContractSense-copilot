"""
Coverage-based sufficiency model for analytical queries.

Instead of requiring one clause to fully answer a complex question, this model
checks whether the combined evidence covers key aspects of the query.
"""
import re


_ASPECT_PATTERNS = {
    "financial_commitment": [r"\bpay(?:ment|able)?\b", r"\bfees?\b", r"\bcommission\b", r"\binvoice\b", r"\bliquidated\s+damages\b", r"\bcontract\s+value\b"],
    "liability": [r"\bliabilit", r"\bdamages?\b", r"\bcap\b", r"\blimit(?:ation)?\b", r"\bindemnif"],
    "warranty": [r"\bwarrant(?:y|ies|s)?\b", r"\brepresent(?:ation|ations|s)?\b", r"\bguarantee\b"],
    "confidentiality": [r"\bconfidential\b", r"\bdisclos(?:e|ure)\b", r"\bneed\s+to\s+know\b"],
    "data_use": [r"\bdata\b", r"\bprocessing\b", r"\bstorage\b", r"\boutside\s+india\b", r"\boutside\s+the\s+country\b"],
    "sharing": [r"\bshare\b", r"\bthird\s*-?party\b", r"\bdisclos(?:e|ure)\b", r"\bwithout\s+.*written\s+(?:consent|approval)\b"],
    "term": [r"\bterm\b", r"\bvalid\b", r"\bexpir(?:y|ation|es)\b", r"\beffective\s+date\b"],
    "survival": [r"\bsurviv(?:e|al)\b", r"\bafter\s+(?:termination|expiration)\b"],
    "restriction": [r"\bshall\s+not\b", r"\bmust\s+not\b", r"\bwithout\s+.*written\s+(?:consent|approval)\b", r"\bnot\s+to\b"],
    "risk": [r"\brisk\b", r"\bliabilit", r"\btermination\b", r"\bindemnif", r"\bpenalt(?:y|ies)\b"],
}


def _aspect_present(aspect, text, taxonomy_labels):
    patterns = _ASPECT_PATTERNS.get(aspect, [])
    if any(re.search(p, text) for p in patterns):
        return True
    if aspect == "financial_commitment" and ("Financial/Payment" in taxonomy_labels or "Liability/Remedies" in taxonomy_labels):
        return True
    if aspect == "liability" and ("Liability/Remedies" in taxonomy_labels or "Indemnification" in taxonomy_labels):
        return True
    if aspect == "confidentiality" and "Confidentiality" in taxonomy_labels:
        return True
    if aspect == "data_use" and "Data Protection" in taxonomy_labels:
        return True
    if aspect == "term" and "Term/Duration" in taxonomy_labels:
        return True
    if aspect == "survival" and "Survival" in taxonomy_labels:
        return True
    return False


def assess_coverage(query_profile, retrieved_chunks, sub_queries=None):
    if not retrieved_chunks:
        return {
            "coverage_ratio": 0.0,
            "covered_aspects": [],
            "missing_aspects": [],
            "decision": "INSUFFICIENT",
            "sub_queries": sub_queries or [],
        }

    aspects = [c for c in query_profile.concepts if c != "general"]
    if not aspects:
        aspects = ["general"]

    combined_text = "\n".join(r["chunk"].text.lower() for r in retrieved_chunks)
    all_labels = set()
    for r in retrieved_chunks:
        all_labels.update(r.get("taxonomy", []))

    covered = []
    missing = []
    if aspects == ["general"]:
        covered = ["general"]
    else:
        for aspect in aspects:
            if _aspect_present(aspect, combined_text, all_labels):
                covered.append(aspect)
            else:
                missing.append(aspect)

    coverage_ratio = len(covered) / max(len(aspects), 1)

    if coverage_ratio >= 0.6 and len(retrieved_chunks) >= 2:
        decision = "COVERED"
    elif coverage_ratio >= 0.25 and len(retrieved_chunks) >= 1:
        decision = "PARTIAL"
    else:
        decision = "INSUFFICIENT"

    return {
        "coverage_ratio": round(coverage_ratio, 3),
        "covered_aspects": covered,
        "missing_aspects": missing,
        "decision": decision,
        "sub_queries": sub_queries or [],
    }
