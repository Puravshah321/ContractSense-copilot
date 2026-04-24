"""
Answer type controller and structured synthesis.

Turns semantically aligned clauses into the expected output form instead of
dumping retrieved passages.
"""
from src.pipeline.legal_ontology import extract_obligation_summary


def should_use_structured_controller(query_profile):
    return query_profile.answer_type in {"list", "risk_table", "comparison"}


def _evidence_list(evidence_chunks, limit=6):
    evidence = []
    for r in evidence_chunks[:limit]:
        c = r["chunk"]
        evidence.append({
            "clause_id": c.clause_id,
            "section": c.section,
            "page": c.page,
            "taxonomy": r.get("taxonomy", []),
            "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
        })
    return evidence


def _risk_level_for_clause(labels, text):
    lower = text.lower()
    if "Liability/Remedies" in labels or "Indemnification" in labels:
        return "HIGH"
    if "Data Protection" in labels or "Use Restriction" in labels:
        return "HIGH"
    if "Financial/Payment" in labels:
        return "MEDIUM"
    if "Term/Duration" in labels:
        return "LOW"
    return "MEDIUM"


def generate_structured_answer(query, evidence_chunks, evidence_check, query_profile):
    evidence = _evidence_list(evidence_chunks)
    if not evidence_chunks:
        prefix = "Answer: NOT_FOUND\n\n" if query_profile.answer_type == "yes_no" else ""
        return {
            "answer": prefix + "This is not specified in the provided document.",
            "risk_level": "N/A",
            "evidence": [],
            "confidence": "HIGH",
            "decision": "NOT_FOUND",
            "action": "",
        }

    if query_profile.answer_type == "list":
        lines = []
        for idx, item in enumerate(evidence_chunks, 1):
            c = item["chunk"]
            labels = ", ".join(item.get("taxonomy", ["General"]))
            summary = extract_obligation_summary(c)
            lines.append(f"{idx}. {labels}: {summary} [{c.clause_id}, {c.section}]")
        answer = "Relevant obligations/commitments found:\n" + "\n".join(lines)
        return {
            "answer": answer,
            "risk_level": "MEDIUM",
            "evidence": evidence,
            "confidence": "HIGH" if evidence_check.get("confidence", 0) > 0.5 else "MEDIUM",
            "decision": "ANSWER",
            "action": "Review the listed obligations against business requirements and negotiation thresholds.",
        }

    if query_profile.answer_type == "risk_table":
        rows = []
        for idx, item in enumerate(evidence_chunks, 1):
            c = item["chunk"]
            labels = item.get("taxonomy", ["General"])
            risk = _risk_level_for_clause(labels, c.text)
            summary = extract_obligation_summary(c)
            rows.append(f"{idx}. {risk}: {summary} [{c.clause_id}, {c.section}]")
        return {
            "answer": "Ranked risks based on retrieved clauses:\n" + "\n".join(rows),
            "risk_level": "HIGH" if any(row.startswith(tuple([f"{i}. HIGH" for i in range(1, 10)])) for row in rows) else "MEDIUM",
            "evidence": evidence,
            "confidence": "HIGH" if evidence_check.get("confidence", 0) > 0.5 else "MEDIUM",
            "decision": "ANSWER",
            "action": "Prioritize HIGH items and verify caps, consent requirements, and survival obligations.",
        }

    if query_profile.answer_type == "comparison":
        lines = []
        for item in evidence_chunks:
            c = item["chunk"]
            labels = ", ".join(item.get("taxonomy", ["General"]))
            lines.append(f"- {labels}: {extract_obligation_summary(c)} [{c.clause_id}]")
        return {
            "answer": "Comparison points from the contract:\n" + "\n".join(lines),
            "risk_level": "MEDIUM",
            "evidence": evidence,
            "confidence": "MEDIUM",
            "decision": "ANSWER",
            "action": "Compare the cited clauses side by side before making a decision.",
        }

    return None
