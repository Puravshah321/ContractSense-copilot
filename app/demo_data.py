"""
ContractSense - Clause Detection & Risk Analysis Engine
Extracts clauses from PDF text and generates structured risk reports.
"""
import re

# ============================================================
# CLAUSE TYPE DETECTION PATTERNS
# ============================================================
CLAUSE_PATTERNS = {
    "Termination": [
        r"terminat(?:ion|e|ed|ing)",
        r"cancel(?:lation)?",
        r"expir(?:ation|e|y)",
        r"end\s+of\s+(?:the\s+)?agreement",
        r"right\s+to\s+terminate",
    ],
    "Limitation of Liability": [
        r"limit(?:ation)?\s+of\s+liabilit(?:y|ies)",
        r"aggregate\s+liabilit(?:y|ies)",
        r"shall\s+not\s+exceed",
        r"consequential\s+damages",
        r"indirect\s+damages",
        r"punitive\s+damages",
        r"cap\s+on\s+(?:damages|liability)",
    ],
    "Indemnification": [
        r"indemnif(?:y|ication|ied)",
        r"hold\s+harmless",
        r"defend\s+and\s+indemnify",
        r"indemnit(?:y|ies)",
    ],
    "Confidentiality": [
        r"confidential(?:ity)?",
        r"non[\-\s]?disclosure",
        r"proprietary\s+information",
        r"trade\s+secret",
        r"nda",
    ],
    "Intellectual Property": [
        r"intellectual\s+property",
        r"ip\s+(?:rights|assignment|ownership)",
        r"work\s+(?:made\s+)?for\s+hire",
        r"copyright",
        r"patent",
        r"trademark",
        r"assign(?:s|ment)?\s+all\s+rights",
    ],
    "Force Majeure": [
        r"force\s+majeure",
        r"act(?:s)?\s+of\s+god",
        r"beyond\s+(?:the\s+)?(?:reasonable\s+)?control",
        r"unforeseeable\s+(?:event|circumstance)",
    ],
    "Payment Terms": [
        r"payment\s+terms?",
        r"invoic(?:e|ing)",
        r"net\s+\d+\s+days?",
        r"late\s+(?:payment|fee|charge)",
        r"interest\s+(?:rate|charge)",
    ],
    "Warranty": [
        r"warrant(?:y|ies|s)",
        r"represent(?:ation)?s?\s+and\s+warrant",
        r"as[\-\s]?is",
        r"merchantabilit(?:y|ies)",
        r"fitness\s+for\s+a\s+particular\s+purpose",
    ],
    "Non-Compete": [
        r"non[\-\s]?compet(?:e|ition)",
        r"restrictive\s+covenant",
        r"competing\s+(?:business|product|service)",
    ],
    "Governing Law": [
        r"governing\s+law",
        r"jurisdiction",
        r"venue",
        r"dispute\s+resolution",
        r"arbitrat(?:ion|e)",
        r"mediat(?:ion|e)",
    ],
    "Data Protection": [
        r"data\s+protect(?:ion)?",
        r"gdpr",
        r"personal\s+data",
        r"data\s+process(?:ing|or)",
        r"privacy",
        r"data\s+breach",
    ],
    "Insurance": [
        r"insurance",
        r"coverage",
        r"policy\s+limit",
        r"certificate\s+of\s+insurance",
    ],
}


def detect_clauses(text):
    """Detect clause types present in the given text."""
    text_lower = text.lower()
    found = {}

    paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.|\(?[a-z]\)|\(?[ivx]+\))', text)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 40]

    for clause_type, patterns in CLAUSE_PATTERNS.items():
        for pattern in patterns:
            for i, para in enumerate(paragraphs):
                if re.search(pattern, para.lower()):
                    if clause_type not in found:
                        found[clause_type] = para[:600]
                    break
            if clause_type in found:
                break

    return found


# ============================================================
# RISK ANALYSIS TEMPLATES
# ============================================================
RISK_ANALYSIS = {
    "Termination": {
        "risk_level": "HIGH",
        "risks": [
            "Unilateral termination rights allow either party to exit without cause, creating business continuity risk.",
            "Short notice periods may not allow sufficient time to find replacement vendors or transition services.",
            "Immediate payment obligations upon termination can create unexpected cash flow pressure.",
            "Absence of transition assistance requirements leaves you without support during handover.",
        ],
        "action": "Negotiate a minimum contract term before convenience termination applies. Add transition assistance obligations and extend notice periods to at least 120 days for enterprise contracts.",
    },
    "Limitation of Liability": {
        "risk_level": "CRITICAL",
        "risks": [
            "Liability caps based on fees paid may be grossly insufficient if a major incident occurs (e.g., data breach costs could be 100x the contract value).",
            "Blanket exclusion of consequential damages eliminates recovery for the most impactful loss categories (lost profits, business interruption).",
            "Mutual limitation clauses protect the vendor disproportionately since their exposure is typically greater than yours.",
            "No carve-outs for gross negligence or willful misconduct means caps apply even when the vendor acts recklessly.",
        ],
        "action": "Push for higher liability caps (2-3x annual fees), carve out data breaches and IP infringement from the cap, and exclude gross negligence/willful misconduct from limitation provisions.",
    },
    "Indemnification": {
        "risk_level": "MEDIUM",
        "risks": [
            "One-sided indemnification may only protect one party while leaving the other exposed.",
            "Broad indemnification scope without insurance backing creates an unfunded obligation.",
            "Lack of notice requirements for claims could result in delayed response and increased exposure.",
            "No cap on indemnification obligations could create unlimited financial exposure for the indemnifying party.",
        ],
        "action": "Verify that the indemnifying party carries adequate insurance. Add mutual indemnification where appropriate. Include claim notification procedures and cooperaetion requirements.",
    },
    "Confidentiality": {
        "risk_level": "MEDIUM",
        "risks": [
            "Time-limited confidentiality obligations (typically 2-5 years) may expire before the information loses its value.",
            "Overly broad definitions of 'Confidential Information' can create compliance burdens.",
            "Lack of injunctive relief provisions makes enforcement difficult since monetary damages are hard to prove.",
            "Exceptions for publicly available information can be exploited if the receiving party contributes to public disclosure.",
        ],
        "action": "Extend survival period for trade secrets to indefinite. Add injunctive relief provisions. Include specific remedies for breach and require return/destruction of confidential materials upon termination.",
    },
    "Intellectual Property": {
        "risk_level": "HIGH",
        "risks": [
            "Broad IP assignment clauses may capture pre-existing IP that the contractor brings to the engagement.",
            "Work-for-hire provisions may not be valid for all types of works under copyright law.",
            "Lack of license-back provisions means the contractor cannot reuse general tools and methodologies.",
            "No escrow arrangements for source code creates dependency risk if the contractor becomes unavailable.",
        ],
        "action": "Clearly define and exclude pre-existing IP. Add a perpetual license for contractor's background IP. Implement source code escrow for critical deliverables.",
    },
    "Force Majeure": {
        "risk_level": "HIGH",
        "risks": [
            "Overly broad force majeure definitions can be misused to excuse foreseeable supply chain disruptions.",
            "No maximum duration means obligations could be suspended indefinitely without termination rights.",
            "Lack of mitigation requirements allows the affected party to passively wait rather than seek alternatives.",
            "Fee handling during force majeure is often unaddressed, creating payment disputes.",
        ],
        "action": "Add a sunset clause allowing termination after 90-180 days of continuous force majeure. Require mitigation efforts and regular status updates. Specify fee suspension during the affected period.",
    },
    "Payment Terms": {
        "risk_level": "MEDIUM",
        "risks": [
            "Short payment windows (Net 15 or immediate) can strain cash flow.",
            "High late payment interest rates may exceed statutory limits in some jurisdictions.",
            "Automatic fee escalation clauses can result in unexpected cost increases.",
            "Lack of dispute resolution for contested invoices may force payment before resolution.",
        ],
        "action": "Negotiate Net 30 or Net 45 payment terms. Cap late fees at statutory limits. Add a right to withhold payment for disputed invoices during resolution.",
    },
    "Warranty": {
        "risk_level": "MEDIUM",
        "risks": [
            "'As-is' disclaimers eliminate all implied warranties, leaving you with no quality guarantees.",
            "Short warranty periods may expire before defects are discovered in complex deliverables.",
            "Exclusive remedy clauses that limit you to repair/replacement may be inadequate for critical failures.",
            "Disclaimer of fitness for particular purpose means the vendor has no obligation to meet your specific requirements.",
        ],
        "action": "Negotiate explicit performance warranties tied to acceptance criteria. Extend warranty periods to at least 12 months. Resist 'as-is' clauses for custom deliverables.",
    },
    "Non-Compete": {
        "risk_level": "HIGH",
        "risks": [
            "Overly broad geographic or industry restrictions may be unenforceable and create legal costs.",
            "Extended non-compete durations (2+ years) may not survive judicial scrutiny in many jurisdictions.",
            "Lack of compensation during the non-compete period may render the clause unenforceable in some regions.",
            "Vague definitions of 'competing business' create uncertainty about permitted activities.",
        ],
        "action": "Narrow the scope to specific products/services, limit duration to 12 months, define geographic boundaries clearly, and consider garden leave compensation.",
    },
    "Governing Law": {
        "risk_level": "LOW",
        "risks": [
            "Foreign jurisdiction clauses may increase litigation costs and create strategic disadvantages.",
            "Mandatory arbitration may limit discovery rights and appeal options.",
            "Lack of venue specification can lead to forum shopping by the opposing party.",
        ],
        "action": "Negotiate for your home jurisdiction. If arbitration is required, specify a neutral institution (ICC, AAA) and location. Include provisions for emergency injunctive relief in courts.",
    },
    "Data Protection": {
        "risk_level": "CRITICAL",
        "risks": [
            "Non-compliant data processing terms can result in regulatory fines (up to 4% of global revenue under GDPR).",
            "Lack of data breach notification requirements delays incident response.",
            "Unclear data ownership provisions may give the vendor rights to use your data.",
            "Cross-border data transfer without adequate safeguards violates privacy regulations.",
        ],
        "action": "Ensure GDPR/CCPA compliance. Add 72-hour breach notification requirements. Clarify data ownership and restrict vendor use. Include data processing agreements as annexes.",
    },
    "Insurance": {
        "risk_level": "MEDIUM",
        "risks": [
            "Inadequate coverage limits may not cover actual losses in a major incident.",
            "Lack of certificate of insurance requirements means you cannot verify coverage.",
            "No requirement to maintain insurance post-termination leaves you exposed for latent claims.",
        ],
        "action": "Require certificates of insurance before contract execution. Set minimum coverage limits appropriate to the engagement. Require tail coverage for at least 2 years post-termination.",
    },
}

# Default risk analysis for unrecognized clauses
DEFAULT_RISK = {
    "risk_level": "MEDIUM",
    "risks": [
        "This clause should be reviewed by qualified legal counsel to assess specific implications.",
        "Standard commercial terms may not adequately protect your organization's interests.",
        "Ambiguous language could be interpreted differently by each party in case of a dispute.",
    ],
    "action": "Consult with your legal team and compare against industry-standard provisions before signing.",
}


def generate_initial_analysis(clauses_found):
    """Generate a comprehensive risk report from detected clauses."""
    if not clauses_found:
        return None

    report_parts = []
    report_parts.append("I've analyzed your contract and identified the following clauses and their associated risks:\n")

    risk_priority = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_clauses = sorted(
        clauses_found.keys(),
        key=lambda c: risk_priority.get(
            RISK_ANALYSIS.get(c, DEFAULT_RISK)["risk_level"], 99
        ),
    )

    critical_count = 0
    high_count = 0

    for clause_type in sorted_clauses:
        analysis = RISK_ANALYSIS.get(clause_type, DEFAULT_RISK)
        risk = analysis["risk_level"]

        if risk == "CRITICAL":
            critical_count += 1
            risk_emoji = "CRITICAL"
        elif risk == "HIGH":
            high_count += 1
            risk_emoji = "HIGH"
        elif risk == "MEDIUM":
            risk_emoji = "MEDIUM"
        else:
            risk_emoji = "LOW"

        report_parts.append(f"---\n**{clause_type}** | Risk: **{risk_emoji}**\n")
        for r in analysis["risks"]:
            report_parts.append(f"- {r}")
        report_parts.append(f"\n**Recommended Action:** {analysis['action']}\n")

    # Summary
    total = len(sorted_clauses)
    summary = f"\n---\n**Summary:** Analyzed **{total} clauses**"
    if critical_count > 0:
        summary += f" | **{critical_count} CRITICAL** risks found"
    if high_count > 0:
        summary += f" | **{high_count} HIGH** risks found"
    summary += ".\n\nFeel free to ask me anything about specific clauses, risks, or negotiation strategies!"
    report_parts.append(summary)

    return "\n".join(report_parts)


def answer_followup(question, clauses_found, pdf_text):
    """Generate a contextual answer to a follow-up question."""
    q_lower = question.lower()

    # Try to match question to a specific clause type
    matched_clause = None
    for clause_type in clauses_found:
        if clause_type.lower() in q_lower:
            matched_clause = clause_type
            break

    # Keyword matching for common questions
    if not matched_clause:
        keyword_map = {
            "terminat": "Termination",
            "cancel": "Termination",
            "liabil": "Limitation of Liability",
            "damage": "Limitation of Liability",
            "cap": "Limitation of Liability",
            "indemnif": "Indemnification",
            "hold harmless": "Indemnification",
            "confidential": "Confidentiality",
            "nda": "Confidentiality",
            "secret": "Confidentiality",
            "ip": "Intellectual Property",
            "intellectual prop": "Intellectual Property",
            "patent": "Intellectual Property",
            "copyright": "Intellectual Property",
            "code": "Intellectual Property",
            "force majeure": "Force Majeure",
            "pandemic": "Force Majeure",
            "disaster": "Force Majeure",
            "payment": "Payment Terms",
            "invoice": "Payment Terms",
            "fee": "Payment Terms",
            "warrant": "Warranty",
            "guarant": "Warranty",
            "non-compet": "Non-Compete",
            "compete": "Non-Compete",
            "restrict": "Non-Compete",
            "governing law": "Governing Law",
            "jurisdiction": "Governing Law",
            "arbitrat": "Governing Law",
            "data protect": "Data Protection",
            "privacy": "Data Protection",
            "gdpr": "Data Protection",
            "breach": "Data Protection",
            "insurance": "Insurance",
            "coverage": "Insurance",
        }
        for keyword, clause in keyword_map.items():
            if keyword in q_lower:
                matched_clause = clause
                break

    if matched_clause and matched_clause in clauses_found:
        analysis = RISK_ANALYSIS.get(matched_clause, DEFAULT_RISK)
        clause_text = clauses_found[matched_clause]

        response = f"Regarding the **{matched_clause}** clause in your contract:\n\n"
        response += f"**Risk Level:** {analysis['risk_level']}\n\n"

        # Provide the actual clause text
        response += f"**Relevant clause text:**\n> {clause_text[:400]}{'...' if len(clause_text) > 400 else ''}\n\n"

        # Add relevant risks
        response += "**Key concerns:**\n"
        for r in analysis["risks"]:
            response += f"- {r}\n"

        response += f"\n**What you should do:** {analysis['action']}"

        return response

    # Generic response for unmatched questions
    if any(w in q_lower for w in ["risk", "danger", "concern", "worry", "problem"]):
        clause_list = ", ".join(clauses_found.keys())
        return (
            f"Based on my analysis, the key risk areas in your contract are across these clauses: **{clause_list}**.\n\n"
            f"The most critical areas to focus on are those marked as CRITICAL or HIGH risk in my initial analysis above. "
            f"Would you like me to dive deeper into any specific clause?"
        )

    if any(w in q_lower for w in ["negotiate", "change", "modify", "improve", "fix"]):
        response = "Here are the top negotiation priorities for your contract:\n\n"
        risk_priority = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_clauses = sorted(
            clauses_found.keys(),
            key=lambda c: risk_priority.get(RISK_ANALYSIS.get(c, DEFAULT_RISK)["risk_level"], 99),
        )
        for i, clause_type in enumerate(sorted_clauses[:5], 1):
            analysis = RISK_ANALYSIS.get(clause_type, DEFAULT_RISK)
            response += f"**{i}. {clause_type}:** {analysis['action']}\n\n"
        return response

    if any(w in q_lower for w in ["summary", "summarize", "overview", "tldr"]):
        critical = []
        high = []
        for ct in clauses_found:
            a = RISK_ANALYSIS.get(ct, DEFAULT_RISK)
            if a["risk_level"] == "CRITICAL":
                critical.append(ct)
            elif a["risk_level"] == "HIGH":
                high.append(ct)

        response = f"**Quick Summary:** Your contract contains **{len(clauses_found)} identified clauses**.\n\n"
        if critical:
            response += f"**CRITICAL risks** in: {', '.join(critical)}\n"
        if high:
            response += f"**HIGH risks** in: {', '.join(high)}\n"
        response += "\nI recommend focusing negotiations on the CRITICAL and HIGH risk areas before signing."
        return response

    # Fallback
    clause_list = ", ".join(clauses_found.keys())
    return (
        f"I've identified these clause types in your contract: **{clause_list}**.\n\n"
        f"Could you be more specific about which clause or risk you'd like to explore? "
        f"For example, you can ask me about termination rights, liability caps, IP ownership, "
        f"confidentiality obligations, or any other specific concern."
    )
