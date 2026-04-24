"""
Baseline vs Generator vs DPO comparison.

Baseline: shallow keyword retrieval with naive answer/NOT_FOUND decision.
Generator: full local ContractSense semantic pipeline.
DPO: trained model metrics from lightning_train_v2.py evaluation.
"""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_precision_pipeline import EVAL_CASES, SAMPLE_CONTRACT, evaluate as evaluate_generator
from src.pipeline.chunker import chunk_document
from src.pipeline.retriever import HybridRetriever


def _baseline_evaluate():
    chunks = chunk_document(SAMPLE_CONTRACT, "baseline_eval_contract.txt")
    retriever = HybridRetriever(chunks)
    rows = []
    for case in EVAL_CASES:
        retrieved = retriever.retrieve(case["query"], top_k=3, candidate_k=10)
        top = retrieved[0]["chunk"] if retrieved else None
        text = " ".join(r["chunk"].text for r in retrieved).lower()
        q_terms = {t for t in case["query"].lower().replace("?", "").split() if len(t) > 3}
        overlap = sum(1 for t in q_terms if t in text)
        decision = "ANSWER" if retrieved and overlap > 0 else "NOT_FOUND"
        top_section = top.section if top else None
        retrieval_ok = bool((
            case["expected_section"] is None and decision == "NOT_FOUND"
        ) or (
            case["expected_section"] is not None
            and top_section
            and case["expected_section"].lower() in top_section.lower()
        ))
        answer_ok = case["expected_answer_contains"].lower() in text
        hallucinated = decision == "ANSWER" and not answer_ok
        rows.append({
            "query": case["query"],
            "actual_decision": decision,
            "expected_decision": case["expected_decision"],
            "top_section": top_section,
            "retrieval_ok": retrieval_ok,
            "decision_ok": decision == case["expected_decision"],
            "answer_ok": answer_ok,
            "hallucinated": hallucinated,
            "grounding_ratio": 1.0 if decision == "ANSWER" and retrieved else 0.0,
        })

    total = len(rows)
    not_found_rows = [r for r in rows if r["expected_decision"] == "NOT_FOUND"]
    return {
        "retrieval_accuracy": sum(r["retrieval_ok"] for r in rows) / total,
        "decision_accuracy": sum(r["decision_ok"] for r in rows) / total,
        "hallucination_rate": sum(r["hallucinated"] for r in rows) / total,
        "not_found_accuracy": sum(r["actual_decision"] == "NOT_FOUND" for r in not_found_rows) / max(len(not_found_rows), 1),
        "grounding_accuracy": sum(r["grounding_ratio"] for r in rows) / total,
    }, rows


def _normalize_dpo_metrics(dpo_eval_results):
    if not dpo_eval_results:
        return {
            "retrieval_accuracy": None,
            "decision_accuracy": None,
            "hallucination_rate": None,
            "not_found_accuracy": None,
            "grounding_accuracy": None,
        }
    return {
        "retrieval_accuracy": None,
        "decision_accuracy": dpo_eval_results.get("decision_accuracy"),
        "hallucination_rate": dpo_eval_results.get("hallucination_rate"),
        "not_found_accuracy": dpo_eval_results.get("not_found_accuracy", dpo_eval_results.get("refusal_accuracy")),
        "grounding_accuracy": dpo_eval_results.get("grounding_accuracy"),
    }


def compare_models(dpo_eval_results=None):
    baseline_metrics, baseline_rows = _baseline_evaluate()
    generator_metrics, generator_rows = evaluate_generator()
    generator_metrics = {
        "retrieval_accuracy": generator_metrics.get("retrieval_accuracy"),
        "decision_accuracy": generator_metrics.get("decision_accuracy"),
        "hallucination_rate": generator_metrics.get("hallucination_rate"),
        "not_found_accuracy": generator_metrics.get("not_found_accuracy"),
        "grounding_accuracy": generator_metrics.get("average_grounding_ratio"),
    }
    return {
        "baseline": baseline_metrics,
        "generator": generator_metrics,
        "dpo": _normalize_dpo_metrics(dpo_eval_results),
    }, {
        "baseline_cases": baseline_rows,
        "generator_cases": generator_rows,
    }


def write_comparison_outputs(comparison, cases, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "model_comparison_metrics.json"
    json_path.write_text(json.dumps({"metrics": comparison, "cases": cases}, indent=2), encoding="utf-8")

    metrics = ["retrieval_accuracy", "decision_accuracy", "hallucination_rate", "not_found_accuracy", "grounding_accuracy"]
    csv_path = output_dir / "model_comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + metrics)
        for model_name, vals in comparison.items():
            writer.writerow([model_name] + [vals.get(m) for m in metrics])

    image_paths = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return [str(json_path), str(csv_path)]

    plot_metrics = ["decision_accuracy", "not_found_accuracy", "grounding_accuracy"]
    models = list(comparison.keys())
    x = range(len(plot_metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"baseline": "#DC2626", "generator": "#2563EB", "dpo": "#059669"}
    for offset, model_name in enumerate(models):
        values = [comparison[model_name].get(m) or 0 for m in plot_metrics]
        positions = [i + (offset - 1) * width for i in x]
        ax.bar(positions, values, width=width, label=model_name, color=colors.get(model_name))
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_metrics, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Baseline vs Generator vs DPO")
    ax.legend()
    fig.tight_layout()
    chart_path = output_dir / "baseline_generator_dpo_comparison.png"
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    image_paths.append(str(chart_path))

    fig, ax = plt.subplots(figsize=(7, 4))
    values = [comparison[m].get("hallucination_rate") or 0 for m in models]
    ax.bar(models, values, color=[colors.get(m) for m in models])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Hallucination Rate Comparison")
    fig.tight_layout()
    hallucination_path = output_dir / "hallucination_rate_comparison.png"
    fig.savefig(hallucination_path, dpi=200)
    plt.close(fig)
    image_paths.append(str(hallucination_path))

    return [str(json_path), str(csv_path)] + image_paths


def main():
    comparison, cases = compare_models()
    paths = write_comparison_outputs(comparison, cases, ROOT / "Images")
    print(json.dumps(comparison, indent=2))
    for path in paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
