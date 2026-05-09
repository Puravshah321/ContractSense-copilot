"""
Evidence-Aware Synthesis — upgraded.

Improvement #8: Structured output with explicitly separated sections:
  1. Directly Supported Findings
  2. Implied Interpretation
  3. Unsupported Assumptions
  4. Missing Information
  5. Risk Implications

Also feeds multi-clause and cross-clause analysis (Improvement #11).
"""
from src.pipeline.legal_reasoner import ReasoningOutput


def _risk_from_group(group_name, items):
    joined = " ".join(i["text"].lower() for i in items)
    if "liability" in group_name.lower() or "indemn" in joined:
        return "HIGH"
    if "financial" in group_name.lower() or any(i.get("type") == "financial_obligation" for i in items):
        return "MEDIUM"
    if "restriction" in group_name.lower() and "shall not" in joined:
        return "HIGH"
    return "MEDIUM"


def _impact_for_group(group_name):
    g = group_name.lower()
    if "financial" in g:
        return "Commercial exposure through direct payment/fee obligations."
    if "liability" in g or "indemn" in g:
        return "Potential legal exposure due to damages/indemnity obligations."
    if "confidential" in g or "data" in g:
        return "Compliance and data governance risk if obligations are breached."
    if "term" in g or "survival" in g:
        return "Ongoing obligations can remain enforceable over time."
    if "sla" in g or "service level" in g:
        return "Operational risk — SLA breaches may trigger financial penalties."
    return "Operational and contractual obligations require active compliance."


def _group_key(item):
    labels = item.get("taxonomy") or []
    if labels:
        return labels[0]
    t = item.get("type", "obligation")
    if t == "financial_obligation":
        return "Financial/Payment"
    if t == "liability_obligation":
        return "Liability/Remedies"
    if t == "restriction":
        return "Use Restriction"
    return "General"


def synthesize_with_reasoning(reasoning: ReasoningOutput, coverage=None):
    """
    Produce a fully structured evidence-aware synthesis from ReasoningOutput.

    Returns a string formatted in clearly labelled sections.
    """
    lines = []

    # ── Section 1: Directly Supported Findings ──────────────────────
    if reasoning.explicit_findings:
        lines.append("**Directly Supported Findings**")
        for finding in reasoning.explicit_findings[:5]:
            lines.append(f"  • {finding}")
        lines.append("")

    # ── Section 2: Implied Interpretation ──────────────────────────
    if reasoning.implied_interpretations:
        lines.append("**Implied Interpretation** _(bounded inference from evidence)_")
        for implication in reasoning.implied_interpretations[:3]:
            lines.append(f"  • {implication}")
        lines.append("")

    # ── Section 3: Conflicting Provisions ──────────────────────────
    if reasoning.conflicts:
        lines.append("**⚠ Conflicting Provisions Detected**")
        for conflict in reasoning.conflicts[:2]:
            lines.append(f"  • {conflict}")
        lines.append("")

    # ── Section 4: Missing Information ─────────────────────────────
    if reasoning.missing_information:
        lines.append("**Missing Information** _(not found in document)_")
        for missing in reasoning.missing_information[:4]:
            lines.append(f"  • {missing}")
        lines.append("")

    # ── Section 5: Ambiguities ─────────────────────────────────────
    if reasoning.ambiguities:
        lines.append("**Ambiguities Requiring Clarification**")
        for ambig in reasoning.ambiguities[:3]:
            lines.append(f"  • {ambig}")
        lines.append("")

    # ── Section 6: Risk Implications ───────────────────────────────
    if reasoning.risk_implications:
        lines.append("**Risk Implications**")
        for risk in reasoning.risk_implications[:4]:
            lines.append(f"  ⚠ {risk}")
        lines.append("")

    # ── Coverage gaps ───────────────────────────────────────────────
    if coverage and coverage.get("missing_aspects"):
        missing = ", ".join(coverage["missing_aspects"])
        lines.append(f"**Coverage Gaps:** {missing}.")
        lines.append("")

    # ── Reasoning quality indicator ─────────────────────────────────
    conf_pct = int(reasoning.confidence * 100)
    lines.append(
        f"_Reasoning depth: {reasoning.reasoning_depth} | "
        f"Confidence: {conf_pct}% | "
        f"Inferences made: {reasoning.n_assumptions}_"
    )

    result = "\n".join(lines).strip()
    if not result:
        return "No actionable analysis could be generated from the available evidence."
    return result


def synthesize_obligations(obligations, coverage=None, max_groups=5, max_points_per_group=3):
    """Legacy synthesis path (kept for backward compatibility with answer_controller)."""
    if not obligations:
        return "No actionable obligations were extracted from the selected clauses."

    grouped = {}
    for item in obligations:
        grouped.setdefault(_group_key(item), []).append(item)

    ordered = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_groups]

    lines = ["Structured findings from contract evidence:", ""]
    for idx, (group, items) in enumerate(ordered, 1):
        risk = _risk_from_group(group, items)
        impact = _impact_for_group(group)
        lines.append(f"{idx}. Finding ({group})")
        for point in items[:max_points_per_group]:
            lines.append(f"- Evidence: {point['text']} [{point['clause_id']}, {point['section']}, p.{point['page']}]")
        lines.append(f"- Impact: {impact}")
        lines.append(f"- Risk level: {risk}")
        lines.append("")

    if coverage and coverage.get("missing_aspects"):
        missing = ", ".join(coverage["missing_aspects"])
        lines.append(f"Coverage gaps: {missing}.")

    return "\n".join(lines).strip()
