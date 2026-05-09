"""
Multi-clause synthesis.

Combines extracted obligations into a structured response grouped by legal
concept and includes evidence, impact, and risk level.
"""


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


def synthesize_obligations(obligations, coverage=None, max_groups=5, max_points_per_group=3):
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
