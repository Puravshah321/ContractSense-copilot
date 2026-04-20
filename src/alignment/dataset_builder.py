"""
Step 1: DPO Dataset Creation
Creates preference pairs (chosen/rejected) for DPO training.
Each pair has: prompt (clause + query), chosen (good response), rejected (bad response).
"""

import json
import random
from pathlib import Path

# ─────────────────────────────────────────────
# Raw seed data: (clause_text, user_query)
# ─────────────────────────────────────────────
SEED_DATA = [
    (
        "Either party may terminate this Agreement upon thirty (30) days written notice. "
        "Upon termination, all outstanding invoices become immediately due and payable. "
        "Licensor may terminate immediately upon breach of confidentiality obligations.",
        "Can we terminate this contract early?"
    ),
    (
        "Vendor shall indemnify, defend, and hold harmless Client from any third-party claims "
        "arising from Vendor's gross negligence or willful misconduct. Indemnification shall not "
        "apply to claims arising from Client's own actions or omissions.",
        "Who is responsible if someone sues us over the vendor's mistake?"
    ),
    (
        "All intellectual property created by Contractor in the performance of services under "
        "this Agreement shall be the sole and exclusive property of Client. Contractor hereby "
        "assigns all rights, title, and interest therein to Client.",
        "Who owns the code our contractor writes for us?"
    ),
    (
        "Payment is due Net-30 from date of invoice. Late payments shall accrue interest at "
        "1.5% per month (18% per annum). Client shall reimburse Vendor for reasonable collection "
        "costs including attorney's fees if payment is more than 60 days overdue.",
        "What happens if we pay late?"
    ),
    (
        "This Agreement shall be governed by the laws of the State of Delaware without regard to "
        "its conflict of law provisions. Any dispute shall be resolved by binding arbitration in "
        "Wilmington, Delaware under JAMS rules. Each party waives its right to jury trial.",
        "Where do we resolve disputes and what law applies?"
    ),
    (
        "Contractor shall not disclose, use, or reproduce any Confidential Information for a "
        "period of five (5) years following termination of this Agreement. Confidential "
        "Information does not include information that becomes publicly available through no "
        "fault of Contractor.",
        "How long must we keep information confidential after the contract ends?"
    ),
    (
        "Vendor's total liability under this Agreement shall not exceed the total fees paid by "
        "Client in the twelve (12) months preceding the claim. In no event shall either party "
        "be liable for indirect, incidental, consequential, or punitive damages.",
        "What's the maximum we can recover if the vendor causes a major loss?"
    ),
    (
        "Client may not assign this Agreement or any rights hereunder without Vendor's prior "
        "written consent, which shall not be unreasonably withheld. This Agreement shall be "
        "binding upon permitted successors and assigns.",
        "Can we transfer this contract to our subsidiary?"
    ),
    (
        "Vendor guarantees 99.5% monthly uptime for all production services. Downtime exceeding "
        "this threshold entitles Client to a service credit equal to 10% of monthly fees per "
        "each additional 0.5% downtime. Credits are Client's sole remedy for SLA breach.",
        "What do we get if the vendor's service goes down frequently?"
    ),
    (
        "Either party may seek injunctive or other equitable relief in any court of competent "
        "jurisdiction without first submitting to arbitration where breach of confidentiality "
        "or IP ownership is alleged. This exception does not waive the arbitration obligation "
        "for other disputes.",
        "Can we go to court directly for anything, or must we always arbitrate?"
    ),
]


# ─────────────────────────────────────────────
# Response templates
# ─────────────────────────────────────────────

def make_chosen(clause: str, query: str, doc_id: str, span_start: int, span_end: int) -> str:
    """
    HIGH-QUALITY response:
    - Starts with explicit RISK label
    - Plain business language (no dense legalese)
    - Concrete recommended action
    - Citation
    """
    chosen_map = {
        "Can we terminate this contract early?": (
            "RISK: HIGH\n\n"
            "You can exit with 30 days' written notice, but all unpaid bills become due the same day you terminate. "
            "If you've broken any confidentiality obligation, the other party can cut ties immediately—no notice required.\n\n"
            "ACTION: Before sending a termination notice, clear all outstanding invoices and verify no confidentiality "
            "clauses have been violated. Have legal counsel review the notice letter.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "Who is responsible if someone sues us over the vendor's mistake?": (
            "RISK: MEDIUM\n\n"
            "The vendor must defend and pay for lawsuits caused by their own serious negligence or intentional wrongdoing. "
            "However, if the claim partly stems from something your company did, the vendor is off the hook for that portion.\n\n"
            "ACTION: Document all vendor actions carefully. If a lawsuit arises, segregate which actions were theirs vs. yours "
            "before invoking indemnification. Notify vendor immediately upon any claim.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "Who owns the code our contractor writes for us?": (
            "RISK: LOW\n\n"
            "Your company owns all code and intellectual property the contractor creates under this agreement. "
            "The contractor has already transferred all rights to you by signing.\n\n"
            "ACTION: Ensure this IP assignment clause is present before work begins. Register key deliverables if needed. "
            "Keep signed copies of the agreement as proof of ownership.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "What happens if we pay late?": (
            "RISK: HIGH\n\n"
            "Late payments cost you 18% per year in interest. If you're more than 60 days overdue, you also owe the "
            "vendor's lawyer fees and collection costs on top of the original bill.\n\n"
            "ACTION: Set up invoice reminders 5 days before the 30-day deadline. If cash flow is tight, proactively "
            "negotiate a payment extension before the due date—after 60 days the costs escalate sharply.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "Where do we resolve disputes and what law applies?": (
            "RISK: MEDIUM\n\n"
            "All disputes go to private arbitration in Delaware under JAMS rules—no jury, no public court. "
            "Delaware law governs. This limits your ability to litigate publicly and may be costly for small claims.\n\n"
            "ACTION: Assess whether JAMS arbitration fits your risk tolerance. Consider negotiating a small-claims "
            "carve-out or a mediation step before arbitration. Ensure your legal team is familiar with JAMS procedures.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "How long must we keep information confidential after the contract ends?": (
            "RISK: MEDIUM\n\n"
            "Confidentiality obligations last 5 years after the contract ends. Information that becomes publicly known "
            "through no fault of yours is excluded.\n\n"
            "ACTION: Create a data-retention policy that flags and restricts access to this vendor's confidential data "
            "for the full 5-year window. Train relevant staff before contract termination.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "What's the maximum we can recover if the vendor causes a major loss?": (
            "RISK: HIGH\n\n"
            "You can only recover up to what you paid the vendor in the last 12 months, regardless of the actual damage. "
            "You cannot claim lost profits, consequential losses, or punitive damages.\n\n"
            "ACTION: Calculate your 12-month vendor spend and compare it to potential exposure. If the gap is large, "
            "negotiate a higher liability cap or require vendor to carry adequate professional liability insurance.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "Can we transfer this contract to our subsidiary?": (
            "RISK: MEDIUM\n\n"
            "You need the vendor's written permission before assigning the contract. The good news: they can't refuse "
            "without a reasonable business reason.\n\n"
            "ACTION: Send a written assignment request to the vendor before the subsidiary transaction closes. "
            "Build in 2–3 weeks lead time for their approval. Document their response.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "What do we get if the vendor's service goes down frequently?": (
            "RISK: MEDIUM\n\n"
            "For every 0.5% of extra downtime beyond 99.5%, you receive a 10% monthly fee credit. "
            "However, credits are your only remedy—you cannot sue for business losses caused by outages.\n\n"
            "ACTION: Track uptime independently using a monitoring tool. Calculate whether credits adequately cover "
            "your downtime losses; if not, negotiate a higher SLA penalty or the right to terminate for repeated breaches.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
        "Can we go to court directly for anything, or must we always arbitrate?": (
            "RISK: LOW\n\n"
            "You can go directly to court—without arbitration first—only for confidentiality breaches or IP ownership "
            "disputes. Everything else must go through arbitration.\n\n"
            "ACTION: Understand this exception before a dispute arises. For IP or confidentiality emergencies, "
            "you may seek an immediate court injunction. Keep this clause visible to your legal team.\n\n"
            f"CITATION: [{doc_id}, spans {span_start}–{span_end}]"
        ),
    }
    return chosen_map.get(query, f"RISK: MEDIUM\n\nThe clause addresses this question.\n\nACTION: Review with counsel.\n\nCITATION: [{doc_id}, spans {span_start}–{span_end}]")


def make_rejected(clause: str, query: str) -> str:
    """
    LOW-QUALITY response:
    - No risk label
    - Dense legal language
    - No concrete action
    - No citation
    """
    rejected_map = {
        "Can we terminate this contract early?":
            "The agreement provides for termination upon thirty (30) days written notice by either party. "
            "Upon termination, outstanding invoices become immediately due and payable. Immediate termination "
            "is permissible in the event of a breach of confidentiality obligations.",

        "Who is responsible if someone sues us over the vendor's mistake?":
            "The indemnification provision obligates the Vendor to indemnify, defend, and hold harmless the Client "
            "from third-party claims arising from Vendor's gross negligence or willful misconduct, excluding claims "
            "attributable to Client's own actions or omissions.",

        "Who owns the code our contractor writes for us?":
            "Pursuant to the intellectual property assignment clause, all intellectual property created by the "
            "Contractor in performance of services shall be the sole and exclusive property of Client, "
            "with all rights, title, and interest assigned thereto.",

        "What happens if we pay late?":
            "Late payments accrue interest at 1.5% per month. If payment is more than 60 days overdue, "
            "Client shall be responsible for collection costs including attorney's fees.",

        "Where do we resolve disputes and what law applies?":
            "The Agreement is governed by Delaware law. Disputes are subject to binding arbitration in "
            "Wilmington, Delaware under JAMS rules, with jury trial rights waived.",

        "How long must we keep information confidential after the contract ends?":
            "The confidentiality obligation extends for five years following termination. Publicly available "
            "information is excluded from the definition of Confidential Information.",

        "What's the maximum we can recover if the vendor causes a major loss?":
            "The Agreement caps Vendor's total liability at fees paid in the preceding twelve months. "
            "Consequential, indirect, and punitive damages are excluded.",

        "Can we transfer this contract to our subsidiary?":
            "Assignment requires Vendor's prior written consent, which shall not be unreasonably withheld. "
            "Permitted assigns are bound by the Agreement.",

        "What do we get if the vendor's service goes down frequently?":
            "The SLA guarantees 99.5% monthly uptime. Breach entitles Client to service credits of 10% "
            "per additional 0.5% downtime increment. Credits constitute Client's sole remedy.",

        "Can we go to court directly for anything, or must we always arbitrate?":
            "Parties may seek injunctive relief in court for confidentiality or IP disputes without first "
            "arbitrating. All other disputes remain subject to the arbitration clause.",
    }
    return rejected_map.get(query, "The clause addresses this matter as described in the agreement.")


# ─────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────

def build_dataset(output_path: str = "dpo_dataset.json") -> list:
    dataset = []

    for idx, (clause, query) in enumerate(SEED_DATA):
        doc_id = f"CUAD_DOC_{idx+1:03d}"
        span_start = random.randint(0, 50)
        span_end = span_start + len(clause.split())

        prompt = f"Clause: {clause}\n\nQuery: {query}"
        chosen = make_chosen(clause, query, doc_id, span_start, span_end)
        rejected = make_rejected(clause, query)

        dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "doc_id": doc_id,
                "span_start": span_start,
                "span_end": span_end,
                "query_type": "contract_risk"
            }
        })

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Saved {len(dataset)} DPO pairs to {output_path}")
    return dataset


if __name__ == "__main__":
    data = build_dataset("data/dpo_dataset.json")
    print(f"\nSample entry:\n{json.dumps(data[0], indent=2)}")