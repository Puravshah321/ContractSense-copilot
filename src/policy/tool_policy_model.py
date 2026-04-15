"""Tool-policy model for ContractSense.

This module provides a small DistilBERT classifier that predicts which tool
should run next from a query and retrieved clause context.

The tool set matches the report pipeline:
- SearchContract
- GetClauseRiskProfile
- CompareClause
- CreateTicket

The training data can be synthesized from the processed CUAD clause corpus.
That keeps Stage 4 self-contained and avoids introducing a new external dataset.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.retrieval.embedder import load_clauses

log = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "distilbert-base-uncased"
MAX_CONTEXT_CHARS = 360
DEFAULT_RANDOM_SEED = 42
DEFAULT_SPLIT_STRATEGY = "group_contract"

TOOL_LABELS = {
    "SearchContract": 0,
    "GetClauseRiskProfile": 1,
    "CompareClause": 2,
    "CreateTicket": 3,
}

LABEL_TO_TOOL = {label: tool for tool, label in TOOL_LABELS.items()}


@dataclass(frozen=True)
class ToolPolicyExample:
    """One supervised example for the tool policy classifier."""

    query: str
    context: str
    tool: str
    clause_id: str | None = None
    source: str = "synthetic"


SEARCH_TEMPLATES = [
    "Find the clause about {topic}.",
    "Search the contract for {topic}.",
    "Show me the section on {topic}.",
    "Where does the agreement discuss {topic}?",
]

RISK_TEMPLATES = [
    "What is the risk profile of this clause?",
    "Explain the obligations and risk in this clause.",
    "Is this clause risky or restrictive?",
    "Summarize the legal risk in plain English.",
]

COMPARE_TEMPLATES = [
    "Compare this clause to a standard market clause.",
    "How does this clause differ from a normal clause?",
    "Compare this clause against common market terms.",
    "Is this clause unusually strict compared with the norm?",
]

ESCALATE_TEMPLATES = [
    "This looks critical. Escalate it to legal.",
    "Create a ticket for this high-risk clause.",
    "This clause may need escalation because of risk.",
    "I am not confident about this clause. Please open a ticket.",
]


HIGH_RISK_PATTERNS = (
    "terminate",
    "termination",
    "indemnify",
    "indemnity",
    "liability",
    "liable",
    "breach",
    "penalty",
    "damages",
    "confidential",
    "non-compete",
    "exclusivity",
    "assignment",
    "governing law",
    "arbitration",
    "uncapped",
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _derive_topic(clause_text: str) -> str:
    first_line = clause_text.splitlines()[0].strip() if clause_text else "contract clause"
    cleaned = first_line.rstrip(".")
    if len(cleaned) > 90:
        cleaned = cleaned[:87].rstrip() + "..."
    return cleaned or "contract clause"


def _context_snippet(clause_text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    text = _normalize_whitespace(clause_text)
    return text[:max_chars]


def _looks_high_risk(clause_text: str) -> bool:
    lowered = clause_text.lower()
    return any(pattern in lowered for pattern in HIGH_RISK_PATTERNS)


def _pick_templates(tool: str) -> list[str]:
    if tool == "SearchContract":
        return SEARCH_TEMPLATES
    if tool == "GetClauseRiskProfile":
        return RISK_TEMPLATES
    if tool == "CompareClause":
        return COMPARE_TEMPLATES
    if tool == "CreateTicket":
        return ESCALATE_TEMPLATES
    raise ValueError(f"Unknown tool label: {tool}")


def build_tool_policy_records(
    clauses: list[dict[str, Any]],
    max_clauses: int = 300,
    examples_per_tool: int = 1,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Build a balanced synthetic tool-policy dataset from clause records."""

    if not clauses:
        return []

    rng = random.Random(seed)
    selected_clauses = clauses[:]
    rng.shuffle(selected_clauses)
    selected_clauses = selected_clauses[: min(max_clauses, len(selected_clauses))]

    records: list[dict[str, Any]] = []
    for clause in selected_clauses:
        clause_text = str(clause.get("clause_text") or clause.get("text") or "")
        if not clause_text.strip():
            continue

        topic = _derive_topic(clause_text)
        context = _context_snippet(clause_text)
        is_high_risk = _looks_high_risk(clause_text)

        tool_specs = [
            ("SearchContract", SEARCH_TEMPLATES),
            ("GetClauseRiskProfile", RISK_TEMPLATES),
            ("CompareClause", COMPARE_TEMPLATES),
            ("CreateTicket", ESCALATE_TEMPLATES),
        ]

        for tool, templates in tool_specs:
            if tool == "CreateTicket" and not is_high_risk:
                query_templates = templates[:2]
            else:
                query_templates = templates[:examples_per_tool]

            for template in query_templates:
                query = template.format(topic=topic)
                records.append(
                    {
                        "query": query,
                        "context": context,
                        "tool": tool,
                        "label": TOOL_LABELS[tool],
                        "clause_id": clause.get("clause_id"),
                        "source": "synthetic_cuad",
                        "high_risk_clause": is_high_risk,
                    }
                )

    return records


def save_tool_policy_records(records: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_tool_policy_records(data_path: str | Path) -> list[dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Tool-policy data not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _format_input(example: dict[str, Any]) -> str:
    return f"QUERY: {example['query']} CONTEXT: {example['context']}"


def _tokenize_records(tokenizer: Any, records: list[dict[str, Any]], max_length: int = 256) -> Dataset:
    dataset = Dataset.from_list(records)

    def preprocess(batch: dict[str, list[Any]]) -> dict[str, Any]:
        texts = [f"QUERY: {query} CONTEXT: {context}" for query, context in zip(batch["query"], batch["context"], strict=False)]
        encoded = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        encoded["labels"] = batch["label"]
        return encoded

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def _compute_metrics(prediction_output: Any) -> dict[str, float]:
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    labels = prediction_output.label_ids
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
    }


def _slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _split_records(
    records: list[dict[str, Any]],
    test_size: float,
    seed: int,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_strategy == "random":
        train_records, eval_records = train_test_split(
            records,
            test_size=test_size,
            random_state=seed,
            stratify=[record["label"] for record in records],
        )
        return train_records, eval_records

    if split_strategy != "group_contract":
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    groups: list[str] = []
    for idx, record in enumerate(records):
        clause_id = str(record.get("clause_id") or "")
        if "_clause_" in clause_id:
            groups.append(clause_id.split("_clause_", 1)[0])
        elif clause_id:
            groups.append(clause_id)
        else:
            groups.append(f"record_{idx}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, eval_idx = next(splitter.split(records, groups=groups))
    train_records = [records[int(i)] for i in train_idx]
    eval_records = [records[int(i)] for i in eval_idx]

    # If grouped split collapses class diversity in eval, fallback to stratified random split.
    eval_labels = {record["label"] for record in eval_records}
    if len(eval_labels) < 2:
        log.warning(
            "Grouped split produced low label diversity in eval set; falling back to random stratified split."
        )
        return _split_records(records, test_size=test_size, seed=seed, split_strategy="random")

    return train_records, eval_records


def _set_if_exists(obj: Any, field: str, value: Any) -> None:
    if hasattr(obj, field):
        setattr(obj, field, value)


def _build_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.02,
        warmup_ratio=0.1,
        label_smoothing_factor=0.05,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=25,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
    )


def _configure_model_regularization(model: Any) -> None:
    _set_if_exists(model.config, "hidden_dropout_prob", 0.2)
    _set_if_exists(model.config, "attention_probs_dropout_prob", 0.2)
    _set_if_exists(model.config, "seq_classif_dropout", 0.2)


def _build_eval_summary(
    model_name: str,
    output_path: Path,
    metrics: dict[str, Any],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> dict[str, float | str]:
    return {
        "model_name": model_name,
        "model_path": str(output_path),
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
        "f1_macro": float(metrics.get("eval_f1_macro", 0.0)),
        "samples_train": float(len(train_records)),
        "samples_eval": float(len(eval_records)),
    }


def _save_model_artifacts(trainer: Trainer, tokenizer: Any, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def _run_trainer_training_and_eval(trainer: Trainer) -> dict[str, Any]:
    trainer.train()
    return trainer.evaluate()


def _create_trainer(
    model: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: Any,
) -> Trainer:
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )


def _build_tokenizer_and_model(model_name: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TOOL_LABELS),
        ignore_mismatched_sizes=True,
    )
    _configure_model_regularization(model)
    return tokenizer, model


def _prepare_tokenized_datasets(
    tokenizer: Any,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    max_length: int,
) -> tuple[Dataset, Dataset]:
    train_dataset = _tokenize_records(tokenizer, train_records, max_length=max_length)
    eval_dataset = _tokenize_records(tokenizer, eval_records, max_length=max_length)
    return train_dataset, eval_dataset


def _build_training_components(
    model_name: str,
    model_save_path: str | Path,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> tuple[Any, Any, Trainer, Path]:
    tokenizer, model = _build_tokenizer_and_model(model_name)
    train_dataset, eval_dataset = _prepare_tokenized_datasets(
        tokenizer,
        train_records,
        eval_records,
        max_length=max_length,
    )
    training_args = _build_training_args(
        output_dir=str(model_save_path),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    trainer = _create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    return tokenizer, model, trainer, Path(model_save_path)


def _train_and_export(
    tokenizer: Any,
    trainer: Trainer,
    model_name: str,
    output_path: Path,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> dict[str, float | str]:
    metrics = _run_trainer_training_and_eval(trainer)
    _save_model_artifacts(trainer, tokenizer, output_path)
    return _build_eval_summary(model_name, output_path, metrics, train_records, eval_records)


def _pick_best_model_row(results: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_results = sorted(
        results,
        key=lambda item: (float(item["f1_macro"]), float(item["accuracy"])),
        reverse=True,
    )
    return sorted_results[0]


def _save_benchmark_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _ordered_model_list(base_model: str, candidate_models: list[str]) -> list[str]:
    ordered_models: list[str] = []
    for model_name in [base_model, *candidate_models]:
        if model_name not in ordered_models:
            ordered_models.append(model_name)
    return ordered_models


def _benchmark_models(
    ordered_models: list[str],
    benchmark_path: Path,
    base_model: str,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_name in ordered_models:
        model_dir = benchmark_path / _slugify_model_name(model_name)
        log.info("Training tool-policy model: %s", model_name)
        metrics = _train_one_tool_policy_model(
            train_records=train_records,
            eval_records=eval_records,
            model_name=model_name,
            model_save_path=model_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            seed=seed,
        )
        metrics["is_base_model"] = model_name == base_model
        results.append(metrics)
    return results


def _create_benchmark_summary(
    base_model: str,
    candidate_models: list[str],
    split_strategy: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    sorted_results = sorted(
        results,
        key=lambda item: (float(item["f1_macro"]), float(item["accuracy"])),
        reverse=True,
    )
    best = sorted_results[0]
    return {
        "base_model": base_model,
        "candidate_models": candidate_models,
        "split_strategy": split_strategy,
        "selection_metric": "f1_macro",
        "best_model": best["model_name"],
        "best_model_path": best["model_path"],
        "best_accuracy": float(best["accuracy"]),
        "best_f1_macro": float(best["f1_macro"]),
        "results": sorted_results,
    }


def _train_single_model_for_policy(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    model_name: str,
    model_save_path: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> dict[str, float | str]:
    tokenizer, _model, trainer, output_path = _build_training_components(
        model_name=model_name,
        model_save_path=model_save_path,
        train_records=train_records,
        eval_records=eval_records,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
    )
    return _train_and_export(
        tokenizer=tokenizer,
        trainer=trainer,
        model_name=model_name,
        output_path=output_path,
        train_records=train_records,
        eval_records=eval_records,
    )


def _build_train_eval_records(
    records: list[dict[str, Any]],
    test_size: float,
    seed: int,
    split_strategy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_records, eval_records = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=[record["label"] for record in records],
    )
    if split_strategy == "random":
        return train_records, eval_records
    return _split_records(records, test_size=test_size, seed=seed, split_strategy=split_strategy)


def _train_one_tool_policy_model(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    model_name: str,
    model_save_path: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> dict[str, float | str]:
    return _train_single_model_for_policy(
        train_records=train_records,
        eval_records=eval_records,
        model_name=model_name,
        model_save_path=model_save_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
    )


def train_tool_policy_model(
    data_path: str | Path,
    model_save_path: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    max_length: int = 256,
    test_size: float = 0.15,
    seed: int = DEFAULT_RANDOM_SEED,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> dict[str, float]:
    """Train the DistilBERT tool-policy classifier and save it to disk."""

    records = load_tool_policy_records(data_path)
    if not records:
        raise ValueError(f"No tool-policy records found in {data_path}")

    train_records, eval_records = _build_train_eval_records(
        records,
        test_size=test_size,
        seed=seed,
        split_strategy=split_strategy,
    )
    metrics = _train_one_tool_policy_model(
        train_records=train_records,
        eval_records=eval_records,
        model_name=base_model,
        model_save_path=model_save_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
    )
    return {
        "accuracy": float(metrics["accuracy"]),
        "f1_macro": float(metrics["f1_macro"]),
        "samples_train": float(metrics["samples_train"]),
        "samples_eval": float(metrics["samples_eval"]),
    }


def benchmark_tool_policy_models(
    data_path: str | Path,
    benchmark_dir: str | Path,
    base_model: str,
    candidate_models: list[str],
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    max_length: int = 256,
    test_size: float = 0.15,
    seed: int = DEFAULT_RANDOM_SEED,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> dict[str, Any]:
    """Train and compare multiple tool-policy backbones, then choose the best."""

    records = load_tool_policy_records(data_path)
    if not records:
        raise ValueError(f"No tool-policy records found in {data_path}")

    train_records, eval_records = _build_train_eval_records(
        records,
        test_size=test_size,
        seed=seed,
        split_strategy=split_strategy,
    )

    ordered_models = _ordered_model_list(base_model, candidate_models)

    benchmark_path = Path(benchmark_dir)
    benchmark_path.mkdir(parents=True, exist_ok=True)

    results = _benchmark_models(
        ordered_models=ordered_models,
        benchmark_path=benchmark_path,
        base_model=base_model,
        train_records=train_records,
        eval_records=eval_records,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        seed=seed,
    )

    summary = _create_benchmark_summary(
        base_model=base_model,
        candidate_models=candidate_models,
        split_strategy=split_strategy,
        results=results,
    )

    _save_benchmark_summary(benchmark_path / "model_comparison.json", summary)
    return summary


class ToolPolicyModel:
    """Lightweight inference wrapper for the trained tool-policy model."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        confidence_threshold: float = 0.45,
        margin_threshold: float = 0.08,
    ) -> None:
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, query: str, context: str, max_length: int = 256) -> dict[str, float]:
        inputs = self.tokenizer(
            [f"QUERY: {query} CONTEXT: {context}"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        return {LABEL_TO_TOOL[idx]: float(probabilities[idx]) for idx in range(len(probabilities))}

    def predict(self, query: str, context: str, max_length: int = 256) -> dict[str, Any]:
        probabilities = self.predict_proba(query, context, max_length=max_length)
        sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        tool, top_confidence = sorted_probs[0]
        second_confidence = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0

        low_confidence = top_confidence < self.confidence_threshold
        low_margin = (top_confidence - second_confidence) < self.margin_threshold
        abstained = low_confidence or low_margin

        selected_tool = tool
        fallback_tool: str | None = None
        if abstained:
            fallback_tool = rule_based_tool_policy(query, context)
            selected_tool = fallback_tool

        return {
            "tool": selected_tool,
            "confidence": top_confidence,
            "model_tool": tool,
            "abstained": abstained,
            "fallback_tool": fallback_tool,
            "probabilities": probabilities,
        }


def rule_based_tool_policy(query: str, context: str = "") -> str:
    """Deterministic fallback when the classifier is unavailable."""

    text = f"{query} {context}".lower()
    if any(token in text for token in ("compare", "difference", "standard", "market", "normal")):
        return "CompareClause"
    if any(token in text for token in ("risk", "risky", "risk profile", "meaning", "interpretation", "explain")):
        return "GetClauseRiskProfile"
    if any(token in text for token in ("escalate", "ticket", "critical", "urgent", "low confidence", "review")):
        return "CreateTicket"
    return "SearchContract"


def build_dataset_from_clauses(
    clauses_path: str | Path,
    output_path: str | Path,
    max_clauses: int = 300,
    examples_per_tool: int = 1,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, Any]]:
    clauses = load_clauses(clauses_path)
    records = build_tool_policy_records(
        clauses,
        max_clauses=max_clauses,
        examples_per_tool=examples_per_tool,
        seed=seed,
    )
    save_tool_policy_records(records, output_path)
    log.info("Saved %d tool-policy records to %s", len(records), output_path)
    return records