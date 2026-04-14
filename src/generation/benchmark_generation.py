"""Evaluation and plotting utilities for Stage 6 generation model comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.generation.generator import ContractGenerator
from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_prompt


def _safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _json_valid(obj: dict[str, Any]) -> float:
    required = {"risk_level", "plain_explanation", "key_obligation", "recommended_action", "citation"}
    return 1.0 if required.issubset(set(obj.keys())) else 0.0


def _risk_salience(obj: dict[str, Any]) -> float:
    expl = str(obj.get("plain_explanation", "")).strip().lower()
    risk = str(obj.get("risk_level", "")).strip().lower()
    if not expl or not risk:
        return 0.0
    first_sentence = expl.split(".")[0]
    return 1.0 if risk in first_sentence else 0.0


def _citation_recall(obj: dict[str, Any], gold: dict[str, Any]) -> float:
    citation = obj.get("citation", {}) if isinstance(obj, dict) else {}
    if not isinstance(citation, dict):
        return 0.0
    clause_ok = str(citation.get("clause_id", "")) == str(gold.get("clause_id", ""))
    page_ok = int(citation.get("page_number", -1)) == int(gold.get("page_number", -2))
    return 1.0 if clause_ok and page_ok else 0.0


def _actionability(obj: dict[str, Any]) -> float:
    action = str(obj.get("recommended_action", "")).strip()
    return 1.0 if len(action.split()) >= 5 else 0.0


def _jargon_elimination(obj: dict[str, Any]) -> float:
    jargon_terms = {
        "notwithstanding",
        "heretofore",
        "thereunder",
        "pursuant",
        "hereinafter",
        "aforementioned",
    }
    text = str(obj.get("plain_explanation", "")).lower()
    tokens = re.findall(r"[a-zA-Z]+", text)
    if not tokens:
        return 0.0
    jargon_hits = sum(1 for t in tokens if t in jargon_terms)
    return max(0.0, 1.0 - (jargon_hits / max(1, len(tokens))))


def evaluate_model_on_holdout(
    base_model: str,
    adapter_path: str | None,
    holdout_path: str | Path,
    max_cases: int = 120,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(holdout_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[:max_cases]

    generator = ContractGenerator(base_model=base_model, adapter_path=adapter_path)

    json_valid_sum = 0.0
    citation_sum = 0.0
    salience_sum = 0.0
    action_sum = 0.0
    jargon_sum = 0.0

    for row in rows:
        clauses = [
            {
                "clause_id": row.get("clause_id", "unknown"),
                "page_number": row.get("page_number", 1),
                "char_span": row.get("char_span", [0, 200]),
                "clause_text": row.get("clause_text", ""),
            }
        ]

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            + build_user_prompt(
                query=row.get("query", "Explain this clause"),
                clauses=clauses,
                tool_results={},
                chat_history=[],
            )
        )
        pred = generator.generate_json(prompt)

        gold_citation = {
            "clause_id": row.get("clause_id", "unknown"),
            "page_number": row.get("page_number", 1),
        }

        json_valid_sum += _json_valid(pred)
        citation_sum += _citation_recall(pred, gold_citation)
        salience_sum += _risk_salience(pred)
        action_sum += _actionability(pred)
        jargon_sum += _jargon_elimination(pred)

    n = max(1, len(rows))
    return {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "json_valid_rate": round(json_valid_sum / n, 4),
        "citation_recall": round(citation_sum / n, 4),
        "risk_salience_score": round(salience_sum / n, 4),
        "actionability_score": round(action_sum / n, 4),
        "jargon_elimination_rate": round(jargon_sum / n, 4),
        "eval_cases": n,
    }


def compare_baseline_vs_lora(
    model_training_records: list[dict[str, Any]],
    holdout_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    for rec in model_training_records:
        base = rec["base_model"]
        adapter = rec["adapter_path"]
        model_name = rec.get("model_label") or base.split("/")[-1]

        baseline = evaluate_model_on_holdout(base_model=base, adapter_path=None, holdout_path=holdout_path)
        baseline["variant"] = "baseline"
        baseline["model_name"] = model_name
        baseline["train_loss"] = None
        baseline["eval_loss"] = None
        baseline["generalization_gap"] = None
        baseline["overfit_flag"] = False
        baseline["finetuned_model"] = False

        tuned = evaluate_model_on_holdout(base_model=base, adapter_path=adapter, holdout_path=holdout_path)
        tuned["variant"] = "lora_finetuned"
        tuned["model_name"] = model_name
        tuned["train_loss"] = rec.get("train_loss")
        tuned["eval_loss"] = rec.get("eval_loss")
        tuned["generalization_gap"] = rec.get("generalization_gap")
        tuned["overfit_flag"] = rec.get("overfit_flag", False)
        tuned["finetuned_model"] = True

        metrics.extend([baseline, tuned])

    df = pd.DataFrame(metrics)
    csv_path = output / "generation_model_comparison.csv"
    json_path = output / "generation_model_comparison.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    _plot_metrics(df, output)
    return df


def _plot_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    metric_cols = [
        "citation_recall",
        "risk_salience_score",
        "actionability_score",
        "jargon_elimination_rate",
        "json_valid_rate",
    ]

    baseline_df = df[df["variant"] == "baseline"].copy()
    tuned_df = df[df["variant"] == "lora_finetuned"].copy()

    merged = baseline_df.merge(
        tuned_df,
        on=["base_model", "model_name"],
        suffixes=("_baseline", "_tuned"),
    )

    plt.figure(figsize=(14, 6))
    x = range(len(merged))
    width = 0.35
    baseline_vals = merged["citation_recall_baseline"]
    tuned_vals = merged["citation_recall_tuned"]
    plt.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], tuned_vals, width=width, label="LoRA")
    plt.xticks(list(x), merged["model_name"], rotation=20, ha="right")
    plt.ylabel("Citation Recall")
    plt.title("Stage 6 Baseline vs LoRA Citation Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "generation_citation_recall_comparison.png", dpi=200)
    plt.close()

    agg_rows = []
    for _, row in merged.iterrows():
        for metric in metric_cols:
            agg_rows.append({
                "base_model": row["base_model"],
                "model_name": row["model_name"],
                "metric": metric,
                "baseline": row[f"{metric}_baseline"],
                "lora": row[f"{metric}_tuned"],
                "delta": row[f"{metric}_tuned"] - row[f"{metric}_baseline"],
            })

    agg = pd.DataFrame(agg_rows)
    pivot = agg.pivot(index="metric", columns="base_model", values="delta")
    plt.figure(figsize=(14, 6))
    pivot.plot(kind="bar", figsize=(14, 6))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("LoRA - Baseline")
    plt.title("Generation Metric Delta by Base Model")
    plt.tight_layout()
    plt.savefig(output_dir / "generation_metric_delta_by_model.png", dpi=200)
    plt.close()

    of = tuned_df[["base_model", "train_loss", "eval_loss", "generalization_gap", "overfit_flag"]].copy()
    of.to_csv(output_dir / "generation_overfit_check.csv", index=False)


def build_overall_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Rank all 4 relevant outputs: three baseline transformers plus their LoRA outputs."""

    ranking = df.copy()
    ranking["quality_score"] = (
        0.40 * ranking["citation_recall"]
        + 0.20 * ranking["risk_salience_score"]
        + 0.20 * ranking["actionability_score"]
        + 0.10 * ranking["json_valid_rate"]
        + 0.10 * ranking["jargon_elimination_rate"]
    )

    ranking["generalization_penalty"] = ranking["generalization_gap"].fillna(0.0).clip(lower=0.0)
    ranking["final_score"] = ranking["quality_score"] - (0.15 * ranking["generalization_penalty"])

    leaderboard = ranking.sort_values(by=["final_score", "citation_recall", "risk_salience_score", "actionability_score"], ascending=False).copy()
    return leaderboard
