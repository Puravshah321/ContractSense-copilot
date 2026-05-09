"""
Legal Reasoning Layer — NEW module.

Improvement #7: True legal reasoning intelligence layer.
Sits between evidence extraction and final answer synthesis.

Reasons about:
  - What is explicitly stated
  - What is strongly implied (bounded inference)
  - What is ambiguous
  - Conflicting provisions
  - Absence of evidence vs evidence of absence
  - Cross-clause synthesis
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReasoningOutput:
    explicit_findings: List[str]       # Directly stated in evidence
    implied_interpretations: List[str]  # Strongly implied but not explicit
    ambiguities: List[str]             # Genuinely unclear / discretionary
    conflicts: List[str]               # Contradictory provisions
    missing_information: List[str]     # Not found in evidence
    risk_implications: List[str]       # Legal risk summary
    reasoning_depth: str               # shallow | moderate | deep
    confidence: float                  # 0..1
    n_assumptions: int                 # How many inferences were made

    def to_sections(self):
        """Return as structured dict for final answer builder."""
        return {
            "explicit_findings":      self.explicit_findings,
            "implied_interpretations": self.implied_interpretations,
            "ambiguities":            self.ambiguities,
            "conflicts":              self.conflicts,
            "missing_information":    self.missing_information,
            "risk_implications":      self.risk_implications,
            "reasoning_depth":        self.reasoning_depth,
            "confidence":             self.confidence,
            "n_assumptions":          self.n_assumptions,
        }


# ── Inference rules (implied interpretation patterns) ──────────────
_IMPLICATION_RULES = [
    # SLA + liability_cap → recovery likely restricted
    (
        lambda facts, qp: any(f.fact_type in {"penalty", "notice_period"} for f in facts)
            and _has_category(qp, "SLA/Service Level")
            and not any(f.fact_type == "liability_cap" for f in facts),
        "The contract specifies SLA performance requirements. Without an explicit cap on recovery, "
        "the liability exposure for SLA breaches may be unconstrained."
    ),
    (
        lambda facts, qp: any(f.fact_type == "liability_cap" for f in facts)
            and _has_category(qp, "SLA/Service Level"),
        "Even if SLA remedies exist, the liability cap clause likely limits the total recoverable amount "
        "for SLA failures. The maximum financial recovery is bounded by the liability cap."
    ),
    # Confidentiality + no exceptions → strict confidentiality
    (
        lambda facts, qp: any("confidential" in f.text.lower() for f in facts)
            and not any(f.fact_type == "exception" for f in facts)
            and _has_category(qp, "Confidentiality"),
        "No explicit exceptions to confidentiality were found. This suggests a strict confidentiality "
        "obligation with no carve-outs, which is unusually restrictive."
    ),
    # Termination without cure period → immediate risk
    (
        lambda facts, qp: any("terminat" in f.text.lower() for f in facts)
            and not any("cure" in f.text.lower() or "remedy" in f.text.lower() for f in facts)
            and _has_category(qp, "Termination"),
        "No cure period was found in the termination provisions. This implies the non-breaching party "
        "may terminate immediately upon breach without opportunity for the defaulting party to rectify."
    ),
    # No explicit SLA compensation → recovery likely restricted
    (
        lambda facts, qp: _has_category(qp, "SLA/Service Level")
            and not any("credit" in f.text.lower() or "compensat" in f.text.lower() for f in facts)
            and any(f.fact_type == "liability_cap" for f in facts),
        "No explicit SLA compensation mechanism was found. Given the liability cap, the likely "
        "implication is that recovery is restricted to the cap amount, not a per-incident credit."
    ),
    # IP assignment without license-back → contractor loses all rights
    (
        lambda facts, qp: _has_category(qp, "Intellectual Property")
            and any("assign" in f.text.lower() for f in facts)
            and not any("licen" in f.text.lower() for f in facts),
        "IP appears to be assigned to the client with no explicit license-back to the contractor. "
        "This implies the contractor may not reuse any IP created under this agreement."
    ),
]


def _has_category(query_profile, category):
    return category in getattr(query_profile, "expected_categories", [])


def _detect_conflicts(retrieved_chunks):
    """Detect conflicting provisions across clauses."""
    conflicts = []
    all_text = [(r["chunk"].clause_id, r["chunk"].text.lower()) for r in retrieved_chunks]

    conflict_pairs = [
        ("shall not assign", "may assign"),
        ("shall not disclose", "may disclose"),
        ("shall not transfer", "may transfer"),
        ("may not terminate", "may terminate"),
        ("unlimited liability", "shall not exceed"),
    ]

    for cid1, text1 in all_text:
        for cid2, text2 in all_text:
            if cid1 >= cid2:
                continue
            for neg, pos in conflict_pairs:
                if neg in text1 and pos in text2:
                    conflicts.append(
                        f"{cid1} states '{neg}' while {cid2} states '{pos}' — these may conflict."
                    )
                elif neg in text2 and pos in text1:
                    conflicts.append(
                        f"{cid2} states '{neg}' while {cid1} states '{pos}' — these may conflict."
                    )
    return list(set(conflicts))


def _detect_ambiguities(facts, retrieved_chunks):
    """Identify genuinely ambiguous provisions."""
    ambiguities = []
    ambig_patterns = [
        (r"\breasonable\s+(?:efforts?|care|time|notice)\b", "vague standard — 'reasonable' is subjective"),
        (r"\bmutually\s+agreed\b", "requires future mutual agreement — not yet defined"),
        (r"\bcommercially\s+reasonable\b", "commercially reasonable standard is jurisdiction-dependent"),
        (r"\bsole\s+discretion\b", "unilateral discretion — creates power imbalance"),
        (r"\bas\s+determined\s+by\b", "determination mechanism not specified"),
        (r"\bfrom\s+time\s+to\s+time\b", "may change without a defined process"),
    ]
    for item in retrieved_chunks:
        text = item["chunk"].text
        cid = item["chunk"].clause_id
        for pattern, description in ambig_patterns:
            if re.search(pattern, text, re.I):
                ambiguities.append(f"{cid}: {description}.")
    return list(set(ambiguities))


def _assess_missing(query_profile, retrieved_chunks):
    """Identify what the query asked for that is not in evidence."""
    missing = []
    categories_found = set()
    for item in retrieved_chunks:
        from src.pipeline.legal_ontology import classify_clause
        labels = classify_clause(item["chunk"])
        categories_found.update(labels)

    expected = set(getattr(query_profile, "expected_categories", []))
    for cat in expected:
        if cat not in categories_found:
            missing.append(f"No {cat} clause was found in the retrieved evidence.")
    return missing


def _compute_reasoning_depth(facts, conflicts, implications):
    n = len(facts) + len(conflicts) * 2 + len(implications) * 2
    if n >= 10:
        return "deep"
    if n >= 5:
        return "moderate"
    return "shallow"


def reason_about_evidence(query, retrieved_chunks, facts, query_profile, evidence_check):
    """
    Core legal reasoning function.

    Args:
        query: the user query string
        retrieved_chunks: list of retrieval dicts
        facts: list of LegalFact from evidence_extractor
        query_profile: QueryProfile from query_understanding
        evidence_check: output of check_evidence_sufficiency

    Returns:
        ReasoningOutput
    """
    explicit_findings = []
    implied = []
    n_assumptions = 0

    # ── Explicit findings from extracted facts ──────────────────────
    from src.pipeline.evidence_extractor import normalize_facts_to_summary
    if facts:
        for f in facts[:6]:
            val_str = f" ({f.value})" if f.value else ""
            explicit_findings.append(
                f"{f.fact_type.replace('_', ' ').title()}{val_str}: {f.text} [{f.clause_id}, {f.section}]"
            )

    # ── Implied interpretations (bounded inference) ─────────────────
    for condition_fn, implication_text in _IMPLICATION_RULES:
        try:
            if condition_fn(facts, query_profile):
                implied.append(implication_text)
                n_assumptions += 1
        except Exception:
            continue

    # ── Conflicts, ambiguities, missing ────────────────────────────
    conflicts = _detect_conflicts(retrieved_chunks)
    ambiguities = _detect_ambiguities(facts, retrieved_chunks)
    missing = _assess_missing(query_profile, retrieved_chunks)

    # ── Risk implications ───────────────────────────────────────────
    risk_implications = []
    for f in facts:
        if f.fact_type == "liability_cap":
            risk_implications.append(f"Liability is capped: {f.value or 'amount specified in clause'}.")
        elif f.fact_type == "penalty":
            risk_implications.append(f"Financial penalty applies: {f.text[:120]}")
        elif f.fact_type == "prohibition":
            risk_implications.append(f"Prohibited action creates breach risk: {f.text[:100]}")

    if conflicts:
        risk_implications.append("Conflicting provisions create legal uncertainty — legal review required.")
    if ambiguities:
        risk_implications.append("Ambiguous terms may require negotiation or legal interpretation.")
    if missing and not explicit_findings:
        risk_implications.append("Absence of explicit provisions may default to statutory law or be unfavorable.")

    depth = _compute_reasoning_depth(facts, conflicts, implied)

    # ── Confidence calibration ──────────────────────────────────────
    base_conf = evidence_check.get("confidence", 0.5)
    from src.pipeline.legal_ontology import calibrate_confidence
    calibrated_conf = calibrate_confidence(base_conf, retrieved_chunks, n_assumptions)

    return ReasoningOutput(
        explicit_findings=explicit_findings,
        implied_interpretations=implied,
        ambiguities=ambiguities,
        conflicts=conflicts,
        missing_information=missing,
        risk_implications=risk_implications,
        reasoning_depth=depth,
        confidence=calibrated_conf,
        n_assumptions=n_assumptions,
    )
