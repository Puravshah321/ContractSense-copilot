"""
ContractSense Copilot - Demo Response Engine
Pre-computed structured responses that showcase the DPO-aligned model's capabilities.
Used for Streamlit Cloud deployment where GPU is not available.
"""

DEMO_CLAUSES = {
    "termination": {
        "title": "Termination for Convenience",
        "text": (
            "Section 9.2 -- Termination for Convenience. Either party may terminate this Agreement "
            "at any time, for any reason or no reason, by providing the other party with at least "
            "ninety (90) days prior written notice. Upon such termination, Customer shall immediately "
            "pay all outstanding invoices for services rendered up to the date of termination. Any "
            "prepaid fees for services not yet rendered shall be refunded within thirty (30) business days."
        ),
    },
    "liability": {
        "title": "Limitation of Liability",
        "text": (
            "Article 12.1 -- Limitation of Liability. IN NO EVENT SHALL EITHER PARTY'S AGGREGATE "
            "LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED THE TOTAL AMOUNT PAID BY "
            "CUSTOMER DURING THE TWELVE (12) MONTH PERIOD IMMEDIATELY PRECEDING THE EVENT GIVING "
            "RISE TO THE CLAIM. NEITHER PARTY SHALL BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, "
            "CONSEQUENTIAL, OR PUNITIVE DAMAGES, REGARDLESS OF THE CAUSE OF ACTION."
        ),
    },
    "ip_assignment": {
        "title": "Intellectual Property Assignment",
        "text": (
            "Clause 7.3 -- Intellectual Property. All intellectual property, including but not limited "
            "to inventions, designs, software code, documentation, and trade secrets, created by "
            "Contractor in the performance of this Agreement shall be considered 'work made for hire' "
            "and shall be the sole and exclusive property of Client. Contractor hereby irrevocably "
            "assigns all rights, title, and interest in such intellectual property to Client."
        ),
    },
    "confidentiality": {
        "title": "Confidentiality & Non-Disclosure",
        "text": (
            "Section 5.1 -- Confidentiality. Each party agrees to hold in strict confidence all "
            "Confidential Information disclosed by the other party. 'Confidential Information' means "
            "any non-public information, including technical data, trade secrets, business plans, "
            "financial information, and customer data. This obligation shall survive for a period of "
            "five (5) years following the termination of this Agreement. Exceptions apply where "
            "information becomes publicly available through no fault of the receiving party."
        ),
    },
    "indemnification": {
        "title": "Indemnification",
        "text": (
            "Article 11.2 -- Indemnification. Vendor shall indemnify, defend, and hold harmless "
            "Customer and its officers, directors, employees, and agents from and against any and all "
            "claims, damages, losses, liabilities, costs, and expenses (including reasonable attorneys' "
            "fees) arising out of or relating to: (a) any breach of Vendor's representations or "
            "warranties; (b) Vendor's negligence or willful misconduct; or (c) any claim that the "
            "deliverables infringe upon the intellectual property rights of any third party."
        ),
    },
    "force_majeure": {
        "title": "Force Majeure",
        "text": (
            "Section 14.1 -- Force Majeure. Neither party shall be liable for any failure or delay "
            "in performing its obligations under this Agreement if such failure or delay results from "
            "circumstances beyond the reasonable control of that party, including but not limited to "
            "acts of God, natural disasters, war, terrorism, riots, embargoes, acts of civil or "
            "military authorities, fire, floods, pandemics, strikes, or shortages of transportation, "
            "facilities, fuel, energy, labor, or materials. The affected party must provide written "
            "notice within fourteen (14) days of the occurrence."
        ),
    },
}


DEMO_QUERIES = {
    "termination": [
        "Can the vendor just cancel this contract suddenly?",
        "What do I owe them if I cancel?",
        "What are the risks in this termination clause?",
    ],
    "liability": [
        "What's our maximum recovery if the vendor causes a major loss?",
        "Are we protected against consequential damages?",
        "What are the key risks in this liability clause?",
    ],
    "ip_assignment": [
        "Who owns the code our contractor writes?",
        "Can the contractor reuse our work for other clients?",
        "What happens to IP if the contract ends?",
    ],
    "confidentiality": [
        "How long are we bound by confidentiality?",
        "What counts as confidential information?",
        "What are the risks in this confidentiality clause?",
    ],
    "indemnification": [
        "What does the vendor have to cover if something goes wrong?",
        "Are we protected against IP infringement claims?",
        "What are the key obligations for the vendor?",
    ],
    "force_majeure": [
        "What happens if a pandemic disrupts the contract?",
        "How quickly must we notify the other party?",
        "What are the risks if force majeure is invoked?",
    ],
}


DEMO_RESPONSES = {
    "termination": {
        "Can the vendor just cancel this contract suddenly?": {
            "risk_level": "MEDIUM",
            "explanation": (
                "No, neither party can cancel suddenly. The clause requires ninety (90) days of prior "
                "written notice before termination takes effect. However, the 'for any reason or no reason' "
                "language means either party can walk away without needing to justify the decision, which "
                "creates uncertainty for long-term planning and resource allocation."
            ),
            "action": (
                "Negotiate a 'termination for cause only' provision or request a minimum contract term "
                "(e.g., 12 months) before the convenience termination right kicks in. This protects your "
                "business continuity and investment in the partnership."
            ),
            "citation": "Section 9.2, chars 0-156",
        },
        "What do I owe them if I cancel?": {
            "risk_level": "HIGH",
            "explanation": (
                "Upon termination, you must immediately pay all outstanding invoices for services already "
                "rendered. The word 'immediately' is aggressive -- it means payment is due on the date of "
                "termination, not within a standard payment cycle (e.g., Net 30). Any prepaid fees for "
                "unrendered services will be refunded within thirty (30) business days."
            ),
            "action": (
                "Request that 'immediately' be changed to 'within thirty (30) days of the termination date' "
                "to align with standard payment terms. Also confirm in writing that any work-in-progress "
                "deliverables will be transferred to you upon termination."
            ),
            "citation": "Section 9.2, chars 157-350",
        },
        "What are the risks in this termination clause?": {
            "risk_level": "HIGH",
            "explanation": (
                "The primary risk is the unilateral 'for convenience' termination right, which allows the "
                "vendor to terminate for any reason with just 90 days notice. This could disrupt your "
                "operations mid-project. Additionally, the immediate payment obligation upon termination "
                "creates cash flow pressure, and the clause does not address transition assistance or "
                "data migration obligations."
            ),
            "action": (
                "1) Add a transition assistance clause requiring the vendor to support migration for at "
                "least 60 days post-termination. 2) Include data return/destruction obligations. "
                "3) Negotiate a longer notice period (120-180 days) for enterprise contracts."
            ),
            "citation": "Section 9.2, chars 0-350",
        },
    },
    "liability": {
        "What's our maximum recovery if the vendor causes a major loss?": {
            "risk_level": "CRITICAL",
            "explanation": (
                "Your maximum recovery is capped at the total fees paid to the vendor in the twelve (12) "
                "months preceding the claim. If you paid $100,000, that is your ceiling -- even if damages "
                "are $10 million. Furthermore, the exclusion of indirect and consequential damages means "
                "lost profits, business interruption costs, and reputational damage are not recoverable."
            ),
            "action": (
                "Negotiate a higher liability cap (2x-3x annual fees minimum) and carve out exceptions "
                "for data breaches, IP infringement, and confidentiality violations. These high-impact "
                "scenarios should not be subject to the general liability cap."
            ),
            "citation": "Article 12.1, chars 0-280",
        },
        "Are we protected against consequential damages?": {
            "risk_level": "HIGH",
            "explanation": (
                "No. The clause explicitly excludes liability for indirect, incidental, special, "
                "consequential, and punitive damages for both parties. This means if the vendor's failure "
                "causes your business to lose customers or revenue, you cannot claim those losses. The "
                "mutual nature of this exclusion means it applies equally to both sides."
            ),
            "action": (
                "Push for a 'carve-out' that excludes data breaches, willful misconduct, and IP "
                "infringement from the consequential damages waiver. At minimum, ensure your cyber "
                "insurance policy covers gaps created by this limitation."
            ),
            "citation": "Article 12.1, chars 200-420",
        },
        "What are the key risks in this liability clause?": {
            "risk_level": "CRITICAL",
            "explanation": (
                "Three critical risks: (1) The 12-month fee cap creates a massive exposure gap if annual "
                "contract value is low relative to potential damages. (2) The blanket exclusion of "
                "consequential damages eliminates recovery for the most impactful loss categories. "
                "(3) The phrase 'REGARDLESS OF THE CAUSE OF ACTION' means these limits apply even if "
                "the vendor acted negligently or in bad faith."
            ),
            "action": (
                "1) Set a minimum liability floor (e.g., $1M) regardless of fees paid. "
                "2) Carve out gross negligence and willful misconduct from the cap. "
                "3) Add a separate, higher cap for data security incidents. "
                "4) Ensure your insurance portfolio covers the exposure gap."
            ),
            "citation": "Article 12.1, chars 0-420",
        },
    },
    "ip_assignment": {
        "Who owns the code our contractor writes?": {
            "risk_level": "LOW",
            "explanation": (
                "You (the Client) own everything. The clause uses both 'work made for hire' doctrine and "
                "a direct assignment of rights, which is a belt-and-suspenders approach that provides "
                "maximum protection. All inventions, designs, software code, documentation, and trade "
                "secrets created during the engagement belong exclusively to you."
            ),
            "action": (
                "Ensure the contractor also signs a separate IP assignment agreement and that pre-existing "
                "IP (tools, frameworks, libraries the contractor brings) is explicitly excluded and "
                "licensed to you under a perpetual, royalty-free license."
            ),
            "citation": "Clause 7.3, chars 0-380",
        },
        "Can the contractor reuse our work for other clients?": {
            "risk_level": "LOW",
            "explanation": (
                "No. The 'sole and exclusive property' language combined with the irrevocable assignment "
                "means the contractor has no retained rights whatsoever. They cannot reuse, reproduce, "
                "or distribute any work product created under this agreement for any other client or "
                "purpose without your explicit written consent."
            ),
            "action": (
                "Add a non-compete clause specific to the deliverables if the contractor works in the "
                "same industry. Also include audit rights to verify the contractor is not reusing your "
                "proprietary work elsewhere."
            ),
            "citation": "Clause 7.3, chars 200-380",
        },
        "What happens to IP if the contract ends?": {
            "risk_level": "MEDIUM",
            "explanation": (
                "The IP assignment is irrevocable, meaning it survives termination permanently. However, "
                "the clause does not address: (1) return of source code and documentation upon termination, "
                "(2) escrow arrangements for critical code, or (3) the contractor's obligation to provide "
                "transition assistance for ongoing maintenance of the codebase."
            ),
            "action": (
                "Add explicit provisions for: source code escrow, documentation handover within 30 days "
                "of termination, and a 90-day transition support period where the contractor assists "
                "your team in understanding the codebase."
            ),
            "citation": "Clause 7.3, chars 0-380",
        },
    },
    "confidentiality": {
        "How long are we bound by confidentiality?": {
            "risk_level": "MEDIUM",
            "explanation": (
                "The confidentiality obligation survives for five (5) years after the contract ends. "
                "This is a standard duration for commercial agreements but may be insufficient for "
                "trade secrets, which ideally should be protected indefinitely. After 5 years, either "
                "party could potentially use or disclose the other's confidential information."
            ),
            "action": (
                "For trade secrets specifically, negotiate an indefinite survival period or 'for as long "
                "as the information remains a trade secret.' Keep the 5-year term for general business "
                "information but carve out trade secrets for permanent protection."
            ),
            "citation": "Section 5.1, chars 280-420",
        },
        "What counts as confidential information?": {
            "risk_level": "LOW",
            "explanation": (
                "The definition is comprehensive and covers: technical data, trade secrets, business plans, "
                "financial information, and customer data. The broad 'any non-public information' language "
                "ensures wide coverage. The exception for publicly available information (through no fault "
                "of the receiving party) is standard and reasonable."
            ),
            "action": (
                "Consider marking all confidential documents with a 'CONFIDENTIAL' watermark to avoid "
                "disputes about what qualifies. Also add 'personnel information' and 'pricing strategies' "
                "to the explicit list if they are not already covered."
            ),
            "citation": "Section 5.1, chars 100-280",
        },
        "What are the risks in this confidentiality clause?": {
            "risk_level": "MEDIUM",
            "explanation": (
                "Two main risks: (1) The 5-year survival period may be too short to protect trade secrets "
                "that have indefinite value. (2) The clause does not specify remedies for breach -- "
                "without an explicit injunctive relief provision, you may only be able to claim monetary "
                "damages, which are difficult to prove for confidentiality breaches."
            ),
            "action": (
                "1) Add an injunctive relief clause stating that a breach would cause irreparable harm "
                "and entitles the non-breaching party to seek immediate injunctive relief. "
                "2) Include a liquidated damages provision for quantifiable breaches. "
                "3) Extend the survival period for trade secrets to 'indefinite.'"
            ),
            "citation": "Section 5.1, chars 0-420",
        },
    },
    "indemnification": {
        "What does the vendor have to cover if something goes wrong?": {
            "risk_level": "LOW",
            "explanation": (
                "The vendor must cover all claims, damages, losses, liabilities, costs, and expenses "
                "(including legal fees) arising from: (a) breach of their representations or warranties, "
                "(b) negligence or willful misconduct, and (c) IP infringement claims. This is a "
                "comprehensive indemnification scope that provides strong protection for your organization."
            ),
            "action": (
                "Verify that the vendor carries adequate insurance (general liability, E&O, and cyber "
                "liability) to actually fund these indemnification obligations. An indemnification "
                "clause is only as strong as the indemnitor's ability to pay."
            ),
            "citation": "Article 11.2, chars 0-400",
        },
        "Are we protected against IP infringement claims?": {
            "risk_level": "LOW",
            "explanation": (
                "Yes. Subsection (c) explicitly requires the vendor to indemnify you against any claim "
                "that the deliverables infringe upon third-party intellectual property rights. This means "
                "if someone sues you for using the vendor's software because it copies their patented "
                "technology, the vendor must cover all legal costs and damages."
            ),
            "action": (
                "Ensure the indemnification also includes a 'cure' provision: if infringement is found, "
                "the vendor must either (1) obtain a license for continued use, (2) modify the deliverable "
                "to be non-infringing, or (3) replace it with a functional equivalent."
            ),
            "citation": "Article 11.2, subsection (c), chars 300-420",
        },
        "What are the key obligations for the vendor?": {
            "risk_level": "MEDIUM",
            "explanation": (
                "The vendor has three key obligations: indemnify (pay for losses), defend (actively fight "
                "legal claims on your behalf using their own counsel), and hold harmless (ensure you suffer "
                "no net loss). The 'defend' obligation is particularly valuable because it shifts the "
                "burden of litigation costs to the vendor immediately, not just after a judgment."
            ),
            "action": (
                "Add a requirement that you have the right to approve the vendor's choice of legal "
                "counsel and to participate in settlement negotiations. Without this, the vendor could "
                "settle a claim in a way that harms your reputation or business interests."
            ),
            "citation": "Article 11.2, chars 0-200",
        },
    },
    "force_majeure": {
        "What happens if a pandemic disrupts the contract?": {
            "risk_level": "HIGH",
            "explanation": (
                "Pandemics are explicitly listed as a force majeure event, meaning neither party would be "
                "liable for failure to perform during the disruption. However, the clause does not specify: "
                "(1) a maximum duration for force majeure before either party can terminate, (2) whether "
                "partial performance is still required, or (3) how fees are handled during the period."
            ),
            "action": (
                "Add a 'sunset clause' stating that if force majeure continues for more than 90-180 days, "
                "either party may terminate without penalty. Also specify that fees are suspended during "
                "the force majeure period and that the affected party must use reasonable efforts to "
                "resume performance."
            ),
            "citation": "Section 14.1, chars 0-420",
        },
        "How quickly must we notify the other party?": {
            "risk_level": "MEDIUM",
            "explanation": (
                "Written notice must be provided within fourteen (14) days of the occurrence. This is a "
                "relatively tight deadline, and failure to notify within this window could arguably waive "
                "your right to invoke force majeure protections. The clause does not specify what the "
                "notice must contain or how it should be delivered."
            ),
            "action": (
                "Extend the notice period to thirty (30) days to allow time for proper assessment. "
                "Also specify that notice should include: (1) a description of the event, (2) estimated "
                "duration, (3) affected obligations, and (4) mitigation steps being taken."
            ),
            "citation": "Section 14.1, chars 380-480",
        },
        "What are the risks if force majeure is invoked?": {
            "risk_level": "HIGH",
            "explanation": (
                "The biggest risk is indefinite suspension of obligations with no termination right. "
                "A vendor could invoke force majeure and effectively freeze the contract for months or "
                "years while you remain locked in. The broad definition (including 'shortages of materials') "
                "could also be abused to excuse routine supply chain issues that should be managed."
            ),
            "action": (
                "1) Add a termination right after a defined force majeure period (90-180 days). "
                "2) Narrow the definition to exclude events that are reasonably foreseeable or insurable. "
                "3) Require the affected party to demonstrate that reasonable mitigation efforts were made."
            ),
            "citation": "Section 14.1, chars 0-480",
        },
    },
}


RISK_COLORS = {
    "LOW": "#22C55E",
    "MEDIUM": "#F59E0B",
    "HIGH": "#EF4444",
    "CRITICAL": "#DC2626",
}

RISK_ICONS = {
    "LOW": "checkmark",
    "MEDIUM": "warning",
    "HIGH": "error",
    "CRITICAL": "error",
}
