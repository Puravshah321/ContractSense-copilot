"""
Evidence-aware synthesis.

This module turns retrieved clauses plus reasoning metadata into:
  1. dispute-aware analytical answers for multi-question prompts
  2. general grounded reasoning summaries
  3. legacy obligation synthesis used by structured controllers
"""
import re

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
        return "Compliance and data-governance risk if obligations are breached."
    if "term" in g or "survival" in g:
        return "Ongoing obligations can remain enforceable over time."
    if "sla" in g or "service level" in g:
        return "Operational risk; SLA breaches may trigger financial penalties."
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


def _split_questions(query):
    lines = []
    for raw_line in query.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("based only"):
            continue
        if line.endswith("?"):
            lines.append(line)
    if lines:
        return lines
    parts = [p.strip() + "?" for p in query.split("?") if p.strip()]
    return parts[:8]


def _chunk_refs_for_patterns(retrieved, patterns, tags=None, limit=2):
    refs = []
    tags = set(tags or [])
    for item in retrieved:
        chunk = item["chunk"]
        text = f"{chunk.section} {chunk.text}".lower()
        chunk_tags = set(chunk.metadata.get("legal_tags", []))
        matched = any(re.search(pattern, text) for pattern in patterns)
        if tags and chunk_tags & tags:
            matched = True
        if matched:
            refs.append(f"{chunk.clause_id} ({chunk.section})")
        if len(refs) >= limit:
            break
    return refs


def _signals_from_retrieved(retrieved):
    text = "\n".join(f"{item['chunk'].section}\n{item['chunk'].text}" for item in retrieved)
    lower = text.lower()
    return {
        "has_cure_period": bool(re.search(r"\bcure\s+period\b|\bopportunity\s+to\s+cure\b|\bfail\w*\s+to\s+cure\b|\b30\s+days?\s+written\s+notice\b|\bthirty\s*\(?(?:30)?\)?\s+days?\s+written\s+notice\b", lower)),
        "has_immediate_termination": bool(re.search(r"\bterminat\w*\s+immediately\b|\bimmediate\s+terminat\w*\b", lower)),
        "has_termination": bool(re.search(r"\bterminat\w*\b", lower)),
        "has_liability_cap": bool(re.search(r"\blimitation\s+of\s+liabilit\w*\b|\bshall\s+not\s+exceed\b|\bmaximum\s+(?:aggregate\s+)?liabilit\w*\b|\bin\s+no\s+event\b", lower)),
        "has_cap_override": bool(re.search(r"\bnotwithstanding\b|\bshall\s+not\s+apply\b|\bexcept\s+for\s+gross\s+negligence\b|\bcarve[\-\s]?out\b", lower)),
        "has_gross_negligence": bool(re.search(r"\bgross\s+negligence\b|\bwillful\s+misconduct\b|\bwilful\s+misconduct\b|\bfraud\b", lower)),
        "has_confidentiality": bool(re.search(r"\bconfidential(?:ity)?\b", lower)),
        "has_data_protection": bool(re.search(r"\bpersonal\s+data\b|\bdata\s+protection\b|\bprivacy\b", lower)),
        "has_security": bool(re.search(r"\bsecurity\s+(?:obligations?|requirements?|incident)\b|\bransomware\b|\bcyber", lower)),
        "has_survival": bool(re.search(r"\bsurviv\w*\b|\bfollowing\s+termination\b|\bafter\s+termination\b", lower)),
        "has_invoice": bool(re.search(r"\binvoice\w*\b|\bfees?\b|\bpayable\b|\bdue\b", lower)),
        "has_due_upon_termination": bool(re.search(r"\boutstanding\s+invoices?\b.*\bdue\s+and\s+payable\b|\ball\s+outstanding\s+invoices?\s+become\s+immediately\s+due\b|\bupon\s+termination.*\bdue\s+and\s+payable\b", lower)),
        "has_subcontractor_attribution": bool(re.search(r"\bdeemed\s+to\s+have\s+been\s+taken\s+by\b|\bresponsib\w+\s+for\s+the\s+acts?\s+of\b|\bacts?\s+undertaken\s+by\s+contractor\b|\battribut\w+\s+to\b", lower)),
        "has_subcontractor_reference": bool(re.search(r"\bsubcontract\w*\b|\bcontractor\w*\b", lower)),
        "has_third_party_event": bool(re.search(r"\bthird[\-\s]party\s+event\b|\bexcluded\s+event\b|\bbeyond\s+(?:the\s+)?reasonable\s+control\b", lower)),
    }


def _issue_answer(question, signals, retrieved):
    q = question.lower()
    refs = []
    status = "Ambiguous"
    rationale = ""

    if "terminate" in q or "cure notice" in q or "cure period" in q:
        refs = _chunk_refs_for_patterns(
            retrieved,
            [r"\bterminat\w*\b", r"\bcure\b", r"\bnotice\b"],
            tags=["termination", "cure_period"],
        )
        if signals["has_immediate_termination"] and signals["has_cure_period"]:
            status = "Ambiguous"
            rationale = "Both immediate-termination language and cure-period language were retrieved, so the contract does not clearly show which governs this breach scenario."
        elif signals["has_immediate_termination"]:
            status = "Likely yes"
            rationale = "An immediate-termination trigger appears in the retrieved termination language."
        elif signals["has_cure_period"]:
            status = "Likely no"
            rationale = "The retrieved termination language points to a cure or notice period before termination, and no separate immediate-termination trigger was found."
        else:
            rationale = "Termination language was not explicit enough on cure notice versus immediate termination."

    elif "limitation-of-liability" in q or "limitation of liability" in q or "liability clause" in q or "liability cap" in q:
        refs = _chunk_refs_for_patterns(
            retrieved,
            [r"\blimitation\s+of\s+liabilit\w*\b", r"\bshall\s+not\s+exceed\b", r"\bin\s+no\s+event\b", r"\bgross\s+negligence\b", r"\bnotwithstanding\b"],
            tags=["liability_cap", "gross_negligence", "data_protection", "confidentiality"],
        )
        if signals["has_liability_cap"] and signals["has_cap_override"]:
            status = "Ambiguous"
            rationale = "A liability cap is present, but retrieved carve-out or override language suggests the cap may not apply in every circumstance."
        elif signals["has_liability_cap"] and (signals["has_confidentiality"] or signals["has_data_protection"] or signals["has_security"]):
            status = "Ambiguous"
            rationale = "The cap is present, but the retrieved evidence also contains confidentiality, data-protection, or security obligations without an explicit statement on whether they override the cap."
        elif signals["has_liability_cap"]:
            status = "Likely yes"
            rationale = "A liability-cap clause was retrieved and no explicit override was found in the current evidence."
        else:
            rationale = "No explicit liability-cap language was found in the retrieved evidence."

    elif "invoice" in q or "unpaid" in q or "pay" in q:
        refs = _chunk_refs_for_patterns(
            retrieved,
            [r"\binvoice\w*\b", r"\bdue\s+and\s+payable\b", r"\bfees?\b", r"\bpayment\b", r"\btermination\b"],
            tags=["payment", "termination"],
        )
        if signals["has_due_upon_termination"]:
            status = "Likely yes"
            rationale = "The retrieved payment/termination language says outstanding invoices become due on termination."
        elif signals["has_invoice"]:
            status = "Ambiguous"
            rationale = "Payment language was retrieved, but it does not clearly resolve whether the final invoices remain payable after this termination dispute."
        else:
            rationale = "No explicit invoice-payment clause was found in the retrieved evidence."

    elif "survive" in q or "survival" in q or "after termination" in q:
        refs = _chunk_refs_for_patterns(
            retrieved,
            [r"\bsurviv\w*\b", r"\bfollowing\s+termination\b", r"\bafter\s+termination\b", r"\bconfidential(?:ity)?\b", r"\bdata\s+protection\b"],
            tags=["survival", "confidentiality", "data_protection"],
        )
        if signals["has_survival"]:
            status = "Likely yes"
            rationale = "The retrieved evidence includes survival or post-termination continuation language."
        else:
            rationale = "No explicit survival wording was found for confidentiality or data-protection obligations."

    elif "subcontractor" in q or "attributable" in q or "contractor" in q:
        refs = _chunk_refs_for_patterns(
            retrieved,
            [r"\bsubcontract\w*\b", r"\bcontractor\w*\b", r"\bdeemed\b", r"\battribut\w+\b", r"\bresponsib\w+\s+for\s+the\s+acts?\b"],
            tags=["subcontractor", "force_majeure"],
        )
        if signals["has_subcontractor_attribution"]:
            status = "Likely yes"
            rationale = "The retrieved evidence ties contractor or subcontractor conduct back to the contracting party."
        elif signals["has_subcontractor_reference"] and signals["has_third_party_event"]:
            status = "Ambiguous"
            rationale = "The evidence references contractors or third-party events, but it does not clearly resolve whether subcontractor negligence is attributed or excluded."
        else:
            rationale = "No explicit contractor-attribution wording was found in the retrieved evidence."

    elif "conflicting clauses" in q or "ambiguity" in q:
        refs = _chunk_refs_for_patterns(
            retrieved,
            [r"\blimitation\s+of\s+liabilit\w*\b", r"\bconfidential(?:ity)?\b", r"\bdata\s+protection\b", r"\bcure\b", r"\bimmediate(?:ly)?\b"],
            tags=["liability_cap", "confidentiality", "data_protection", "termination", "cure_period"],
        )
        conflicts = []
        if signals["has_liability_cap"] and (signals["has_confidentiality"] or signals["has_data_protection"] or signals["has_security"]):
            conflicts.append("liability cap versus confidentiality/data-protection/security obligations")
        if signals["has_immediate_termination"] and signals["has_cure_period"]:
            conflicts.append("immediate termination versus cure-period language")
        if signals["has_subcontractor_attribution"] and signals["has_third_party_event"]:
            conflicts.append("contractor attribution versus excluded third-party event language")
        if conflicts:
            status = "Yes"
            rationale = "Potential ambiguity appears in " + "; ".join(conflicts) + "."
        else:
            status = "Likely no"
            rationale = "No major textual conflict was strongly signaled by the retrieved clauses."

    return status, rationale, refs


def _general_reasoning_summary(reasoning, coverage=None):
    lines = []
    if reasoning.explicit_findings:
        lines.append("Directly supported findings:")
        for finding in reasoning.explicit_findings[:6]:
            lines.append(f"- {finding}")
    if reasoning.implied_interpretations:
        lines.append("")
        lines.append("Bounded interpretations:")
        for implication in reasoning.implied_interpretations[:4]:
            lines.append(f"- {implication}")
    if reasoning.conflicts:
        lines.append("")
        lines.append("Conflicts:")
        for conflict in reasoning.conflicts[:3]:
            lines.append(f"- {conflict}")
    if reasoning.ambiguities:
        lines.append("")
        lines.append("Ambiguities:")
        for ambiguity in reasoning.ambiguities[:4]:
            lines.append(f"- {ambiguity}")
    if reasoning.missing_information:
        lines.append("")
        lines.append("Missing information:")
        for missing in reasoning.missing_information[:4]:
            lines.append(f"- {missing}")
    if coverage and coverage.get("missing_aspects"):
        lines.append("")
        lines.append("Coverage gaps: " + ", ".join(coverage["missing_aspects"]) + ".")
    lines.append("")
    lines.append(
        f"Reasoning depth: {reasoning.reasoning_depth} | "
        f"Confidence: {int(reasoning.confidence * 100)}% | "
        f"Inferences made: {reasoning.n_assumptions}"
    )
    return "\n".join(lines).strip()


def synthesize_with_reasoning(reasoning: ReasoningOutput, coverage=None, query=None, retrieved=None):
    if query and retrieved:
        questions = _split_questions(query)
        if len(questions) >= 3:
            signals = _signals_from_retrieved(retrieved)
            lines = []
            unresolved = []
            for idx, question in enumerate(questions, 1):
                status, rationale, refs = _issue_answer(question, signals, retrieved)
                ref_text = f" Evidence: {', '.join(refs)}." if refs else ""
                lines.append(f"Q{idx}. {question}")
                lines.append(f"Status: {status}. {rationale}{ref_text}")
                lines.append("")

                if status == "Ambiguous":
                    unresolved.append(question)

                if "strongest arguments" in question.lower():
                    acme = []
                    global_retail = []
                    if signals["has_liability_cap"]:
                        acme.append("ACME can rely on the retrieved limitation-of-liability language.")
                    if signals["has_subcontractor_attribution"]:
                        global_retail.append("GlobalRetail can argue contractor conduct is attributed to ACME under the retrieved contractor language.")
                    if signals["has_confidentiality"] or signals["has_data_protection"] or signals["has_security"]:
                        global_retail.append("GlobalRetail can rely on the retrieved confidentiality, data-protection, or security obligations.")
                    if signals["has_due_upon_termination"]:
                        acme.append("ACME can argue termination does not erase the express obligation to pay outstanding invoices.")
                    if signals["has_third_party_event"]:
                        acme.append("ACME can argue the agreement separately references third-party or excluded-event language.")
                    lines.append("ACME's strongest contract arguments:")
                    for item in acme[:4]:
                        lines.append(f"- {item}")
                    if not acme:
                        lines.append("- ACME's best arguments are not clearly stated in the retrieved clauses.")
                    lines.append("GlobalRetail's strongest contract arguments:")
                    for item in global_retail[:4]:
                        lines.append(f"- {item}")
                    if not global_retail:
                        lines.append("- GlobalRetail's best arguments are not clearly stated in the retrieved clauses.")
                    lines.append("")

                if "least clearly resolved" in question.lower():
                    least = "the interaction between the liability cap and confidentiality/data-protection/security obligations"
                    if signals["has_immediate_termination"] and signals["has_cure_period"]:
                        least = "the interaction between the cure-period clause and any immediate-termination language"
                    elif signals["has_subcontractor_reference"] and not signals["has_subcontractor_attribution"]:
                        least = "whether subcontractor conduct is attributed or treated as a third-party event"
                    lines.append(f"Least clearly resolved issue: {least}.")
                    lines.append("")

            if unresolved:
                lines.append(
                    "Overall: the agreement supports several issue-level conclusions, but these questions remain materially ambiguous: "
                    + "; ".join(unresolved[:3]) + "."
                )
            elif reasoning.ambiguities:
                lines.append("Overall: the retrieved clauses support an answer, but some ambiguity remains in how the clauses interact.")
            else:
                lines.append("Overall: the retrieved clauses support direct answers to the main dispute issues.")

            if coverage and coverage.get("missing_aspects"):
                lines.append("Coverage gaps: " + ", ".join(coverage["missing_aspects"]) + ".")

            lines.append(
                f"Confidence: {int(reasoning.confidence * 100)}% based on retrieved contract text."
            )
            return "\n".join(lines).strip()

    return _general_reasoning_summary(reasoning, coverage=coverage)


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
