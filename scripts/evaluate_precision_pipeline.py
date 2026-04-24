"""
Evaluate the precision-engineered ContractSense pipeline.

Tracks mandatory metrics:
  - retrieval accuracy
  - decision accuracy
  - hallucination rate
  - NOT_FOUND accuracy

Also writes PNG charts to Images/ so Lightning AI runs produce visual output.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.orchestrator import ContractSensePipeline


SAMPLE_CONTRACT = """
1. Definitions.
The term "Confidential Information" shall include all information and materials furnished by either Party, including Results of any information security audits, tests, analysis, extracts or usages carried out by the Auditor in connection with the Auditee's products and/or services, IT infrastructure, etc.

2. Protection of Confidential Information.
Auditor shall use the Confidential Information as necessary only in connection with scope of audit and in accordance with the terms and conditions contained herein. Auditor shall not make or retain copy of any details of results of any information security audits, tests, analysis, extracts or usages carried out by the Auditor without the express written consent of Auditee. Auditor shall not disclose or in any way assist or permit the disclosure of any Confidential Information to any other person or entity without the express written consent of the auditee. Auditor shall not send Auditee's audit information or data and/or any such Confidential Information at any time outside India for the purpose of storage, processing, analysis or handling without the express written consent of the Auditee.

4. Permitted disclosure of audit related information.
The auditor may share audit information with STQC or similar Government entities mandated under the law as and when called upon to do so by such agencies with prior written information to the auditee.

5. Financial Commitments.
Auditee shall pay the Auditor a fixed audit fee of INR 50,000 after completion of audit fieldwork. No commission is payable under this Agreement.

6. Remedies.
Auditor acknowledges that any actual or threatened disclosure or use of the Confidential Information by Auditor would be a breach of this agreement and may cause immediate and irreparable harm to Auditee or to its clients. In addition, Auditor shall compensate the Auditee for the loss or damages caused to the auditee, actual and liquidated damages, which may be demanded by Auditee with liquidated damages not to exceed the Contract value.

7. Need to Know.
Auditor shall restrict disclosure of such Confidential Information to its employees and/or consultants with a need to know, shall use the Confidential Information only for the purposes set forth in the Agreement, and shall not disclose such Confidential Information to any affiliates, subsidiaries, associates and/or third party without prior written approval of the Auditee. No Information relating to auditee shall be hosted or taken outside the country in any circumstances.

8. Audit Records.
Auditor shall maintain audit records and processing logs for inspection. These records do not create any fee, commission, or payment obligation.

9. No Conflict.
The parties represent and warrant that the performance of its obligations hereunder do not and shall not conflict with any other agreement or obligation of the respective parties.

12. Entire Agreement.
This Agreement constitutes the entire understanding and agreement between the parties, and supersedes all previous or contemporaneous agreement or communications.

17. Survival.
Both parties agree that all of their obligations undertaken herein with respect to Confidential Information received pursuant to this Agreement shall survive till perpetuity even after expiration or termination of this Agreement.

18. Non-solicitation.
During the term of this Agreement and thereafter for a further period of two (2) years Auditor shall not solicit or attempt to solicit Auditee's employees and/or consultants.

20. Term.
This Agreement shall come into force on the date of its signing by both the parties and shall be valid up to one year.
"""


EVAL_CASES = [
    {
        "query": "Is there warranty clause?",
        "expected_decision": "NOT_FOUND",
        "expected_section": None,
        "expected_answer_contains": "NOT_FOUND",
    },
    {
        "query": "What is the duration (term) of this agreement?",
        "expected_decision": "ANSWER",
        "expected_section": "Term",
        "expected_answer_contains": "one year",
    },
    {
        "query": "Can the auditor share confidential data with external teams or third parties?",
        "expected_decision": "ANSWER",
        "expected_section": "Need to Know",
        "expected_answer_contains": "Answer: NO",
    },
    {
        "query": "Does this agreement allow using audit data for training AI models?",
        "expected_decision": "ANSWER",
        "expected_section": "Protection of Confidential Information",
        "expected_answer_contains": "Answer: NO",
    },
    {
        "query": "What penalty amount must be paid for breach of contract?",
        "expected_decision": "ANSWER",
        "expected_section": "Remedies",
        "expected_answer_contains": "no fixed penalty amount",
    },
    {
        "query": "What are the financial commitments in this agreement?",
        "expected_decision": "ANSWER",
        "expected_section": "Financial Commitments",
        "expected_answer_contains": "INR 50,000",
    },
    {
        "query": "Which clause creates the highest compliance burden?",
        "expected_decision": "NOT_FOUND",
        "expected_section": None,
        "expected_answer_contains": "not specified",
    },
]


def _top_section(result):
    if not result.evidence:
        return None
    return result.evidence[0].get("section")


def evaluate():
    pipeline = ContractSensePipeline()
    pipeline.load_document(SAMPLE_CONTRACT, "precision_eval_contract.txt")

    rows = []
    for case in EVAL_CASES:
        result = pipeline.query(case["query"], top_k=3)
        top_section = _top_section(result)
        retrieval_ok = bool((
            case["expected_section"] is None
            and result.decision == "NOT_FOUND"
        ) or (
            case["expected_section"] is not None
            and top_section
            and case["expected_section"].lower() in top_section.lower()
        ))
        decision_ok = result.decision == case["expected_decision"]
        answer_ok = case["expected_answer_contains"].lower() in result.answer.lower()
        hallucinated = result.decision == "ANSWER" and not answer_ok

        rows.append({
            "query": case["query"],
            "expected_decision": case["expected_decision"],
            "actual_decision": result.decision,
            "expected_section": case["expected_section"],
            "top_section": top_section,
            "answer": result.answer,
            "retrieval_ok": retrieval_ok,
            "decision_ok": decision_ok,
            "answer_ok": answer_ok,
            "hallucinated": hallucinated,
            "grounding_ratio": result.verification.get("supported_ratio", 0.0),
        })

    total = len(rows)
    not_found_rows = [r for r in rows if r["expected_decision"] == "NOT_FOUND"]
    metrics = {
        "total": total,
        "retrieval_accuracy": sum(r["retrieval_ok"] for r in rows) / total,
        "decision_accuracy": sum(r["decision_ok"] for r in rows) / total,
        "hallucination_rate": sum(r["hallucinated"] for r in rows) / total,
        "not_found_accuracy": sum(r["actual_decision"] == "NOT_FOUND" for r in not_found_rows) / max(len(not_found_rows), 1),
        "average_grounding_ratio": sum(r["grounding_ratio"] for r in rows) / total,
    }
    return metrics, rows


def write_outputs(metrics, rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "precision_pipeline_metrics.json").write_text(
        json.dumps({"metrics": metrics, "cases": rows}, indent=2),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    image_paths = []
    labels = ["retrieval", "decision", "not_found", "grounding"]
    values = [
        metrics["retrieval_accuracy"],
        metrics["decision_accuracy"],
        metrics["not_found_accuracy"],
        metrics["average_grounding_ratio"],
    ]
    colors = ["#2563EB", "#059669", "#DC2626", "#7C3AED"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("ContractSense Precision Pipeline Metrics")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.03, f"{value:.0%}", ha="center", fontweight="bold")
    fig.tight_layout()
    path = output_dir / "precision_pipeline_metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    image_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["hallucination_rate"], [metrics["hallucination_rate"]], color="#EA580C")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Hallucination Rate After Guardrails")
    ax.text(0, metrics["hallucination_rate"] + 0.03, f"{metrics['hallucination_rate']:.0%}", ha="center", fontweight="bold")
    fig.tight_layout()
    path = output_dir / "precision_hallucination_rate.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    image_paths.append(str(path))
    return image_paths


def main():
    metrics, rows = evaluate()
    image_paths = write_outputs(metrics, rows, ROOT / "Images")
    print(json.dumps(metrics, indent=2))
    for path in image_paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
