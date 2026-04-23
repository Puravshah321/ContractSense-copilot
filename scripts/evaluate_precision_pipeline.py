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
Section 2.1 Term.
This Agreement begins on the Effective Date and continues for one (1) year unless earlier terminated in accordance with this Agreement. The parties acknowledge that the one-year term controls the duration of the Agreement and any extension must be signed in writing by both parties.

Section 3.1 Confidentiality.
Each party shall protect Confidential Information using reasonable care and shall not disclose Confidential Information to third parties except to employees and advisors who need to know it for the permitted purpose. These confidentiality duties survive termination for three (3) years.

Section 4.2 Data Protection.
Personal Data shall be processed only within India and shall not be transferred outside India without Controller's prior written consent. Processor shall implement appropriate technical and organizational measures to prevent unauthorized access, alteration, disclosure, or destruction of Personal Data.

Section 9.1 Termination.
Either party may terminate this Agreement for material breach if the breach remains uncured for thirty (30) days after written notice. Upon termination, each party shall return or destroy Confidential Information as requested by the disclosing party.

Section 18.1 Entire Agreement.
This Agreement constitutes the entire agreement between the parties and supersedes all prior discussions, understandings, purchase orders, and proposals. No amendment is effective unless it is in writing and signed by both parties.
"""


EVAL_CASES = [
    {
        "query": "Is there warranty clause?",
        "expected_decision": "NOT_FOUND",
        "expected_section": None,
        "expected_answer_contains": "NOT_FOUND",
    },
    {
        "query": "What is the duration of this agreement?",
        "expected_decision": "ANSWER",
        "expected_section": "Term",
        "expected_answer_contains": "one (1) year",
    },
    {
        "query": "Can data be shared outside India?",
        "expected_decision": "ANSWER",
        "expected_section": "Data Protection",
        "expected_answer_contains": "Answer: NO",
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
