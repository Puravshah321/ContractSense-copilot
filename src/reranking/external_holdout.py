"""Build strict external holdout splits for reranker training/evaluation.

This module creates reranker pairs with a contract-level (or family-level)
holdout so evaluation data remains fully unseen during training.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.retrieval.embedder import load_clauses


@dataclass(frozen=True)
class ClauseRow:
    clause_id: str
    contract_id: str
    family_id: str
    split: str
    clause_text: str


def _derive_contract_id(record: dict[str, Any]) -> str:
    contract_id = str(record.get("contract_id") or "").strip()
    if contract_id:
        return contract_id

    clause_id = str(record.get("clause_id") or "").strip()
    if "_clause_" in clause_id:
        return clause_id.split("_clause_", 1)[0]
    if clause_id:
        return clause_id
    return "unknown_contract"


def _derive_family_id(contract_id: str, family_regex: str | None = None) -> str:
    if not family_regex:
        return contract_id
    return re.sub(family_regex, "", contract_id)


def _normalize_clause_rows(
    records: list[dict[str, Any]],
    family_regex: str | None = None,
) -> list[ClauseRow]:
    rows: list[ClauseRow] = []
    for record in records:
        clause_text = str(record.get("clause_text") or record.get("text") or "").strip()
        if not clause_text:
            continue
        clause_id = str(record.get("clause_id") or "").strip() or f"generated_{len(rows):06d}"
        contract_id = _derive_contract_id(record)
        family_id = _derive_family_id(contract_id, family_regex=family_regex)
        split = str(record.get("split") or "unknown").strip().lower()
        rows.append(
            ClauseRow(
                clause_id=clause_id,
                contract_id=contract_id,
                family_id=family_id,
                split=split,
                clause_text=clause_text,
            )
        )
    return rows


def _first_sentence(text: str, max_chars: int = 200) -> str:
    sentence = text.strip().split("\n", 1)[0].strip()
    if len(sentence) > max_chars:
        sentence = sentence[: max_chars - 3].rstrip() + "..."
    if len(sentence) < 24:
        return f"Find clause: {sentence}" if sentence else "Find relevant contract clause."
    return sentence


def _choose_holdout_families(
    rows: list[ClauseRow],
    holdout_ratio: float,
    seed: int,
    prefer_explicit_test_split: bool,
) -> tuple[set[str], str]:
    rng = random.Random(seed)

    explicit_test_rows = [row for row in rows if row.split == "test"]
    if prefer_explicit_test_split and explicit_test_rows:
        return {row.family_id for row in explicit_test_rows}, "explicit_test_split"

    families = sorted({row.family_id for row in rows})
    if not families:
        raise ValueError("No families found in clause records.")

    num_holdout = max(1, int(len(families) * holdout_ratio))
    holdout = set(rng.sample(families, k=min(num_holdout, len(families))))
    return holdout, "random_family_group_split"


def split_external_holdout(
    rows: list[ClauseRow],
    holdout_ratio: float = 0.2,
    seed: int = 42,
    prefer_explicit_test_split: bool = True,
) -> tuple[list[ClauseRow], list[ClauseRow], dict[str, Any]]:
    holdout_families, split_mode = _choose_holdout_families(
        rows,
        holdout_ratio=holdout_ratio,
        seed=seed,
        prefer_explicit_test_split=prefer_explicit_test_split,
    )

    train_rows = [row for row in rows if row.family_id not in holdout_families]
    holdout_rows = [row for row in rows if row.family_id in holdout_families]

    if not train_rows or not holdout_rows:
        raise ValueError(
            "External holdout split failed: one partition is empty. "
            "Adjust holdout_ratio or disable prefer_explicit_test_split."
        )

    train_families = {row.family_id for row in train_rows}
    holdout_family_set = {row.family_id for row in holdout_rows}
    overlap = train_families.intersection(holdout_family_set)
    if overlap:
        raise ValueError(f"Family leakage detected across split: {sorted(overlap)[:5]}")

    metadata = {
        "split_mode": split_mode,
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "train_clauses": len(train_rows),
        "holdout_clauses": len(holdout_rows),
        "train_contracts": len({row.contract_id for row in train_rows}),
        "holdout_contracts": len({row.contract_id for row in holdout_rows}),
        "train_families": len(train_families),
        "holdout_families": len(holdout_family_set),
        "family_overlap": 0,
    }
    return train_rows, holdout_rows, metadata


def _sample_query_rows(
    rows: list[ClauseRow],
    max_queries_per_contract: int,
    seed: int,
) -> list[ClauseRow]:
    rng = random.Random(seed)
    by_contract: dict[str, list[ClauseRow]] = {}
    for row in rows:
        by_contract.setdefault(row.contract_id, []).append(row)

    selected: list[ClauseRow] = []
    for contract_id in sorted(by_contract):
        contract_rows = by_contract[contract_id][:]
        rng.shuffle(contract_rows)
        selected.extend(contract_rows[: max_queries_per_contract])
    return selected


def build_reranker_pairs(
    rows: list[ClauseRow],
    negatives_per_positive: int,
    max_queries_per_contract: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = rows[:]

    query_rows = _sample_query_rows(
        rows,
        max_queries_per_contract=max_queries_per_contract,
        seed=seed,
    )

    pairs: list[dict[str, Any]] = []
    for query_row in query_rows:
        query_text = _first_sentence(query_row.clause_text)

        pairs.append(
            {
                "query": query_text,
                "clause_text": query_row.clause_text,
                "label": 1.0,
                "query_clause_id": query_row.clause_id,
                "candidate_clause_id": query_row.clause_id,
                "query_contract_id": query_row.contract_id,
                "candidate_contract_id": query_row.contract_id,
                "query_family_id": query_row.family_id,
                "candidate_family_id": query_row.family_id,
            }
        )

        negatives = [
            candidate
            for candidate in pool
            if candidate.contract_id != query_row.contract_id
        ]
        if not negatives:
            continue

        if len(negatives) > negatives_per_positive:
            sampled_negatives = rng.sample(negatives, k=negatives_per_positive)
        else:
            sampled_negatives = negatives

        for neg in sampled_negatives:
            pairs.append(
                {
                    "query": query_text,
                    "clause_text": neg.clause_text,
                    "label": 0.0,
                    "query_clause_id": query_row.clause_id,
                    "candidate_clause_id": neg.clause_id,
                    "query_contract_id": query_row.contract_id,
                    "candidate_contract_id": neg.contract_id,
                    "query_family_id": query_row.family_id,
                    "candidate_family_id": neg.family_id,
                }
            )

    return pairs


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_external_holdout_pair_files(
    clauses_path: str | Path,
    train_output_path: str | Path,
    holdout_output_path: str | Path,
    metadata_output_path: str | Path,
    holdout_ratio: float = 0.2,
    negatives_per_positive: int = 3,
    max_queries_per_contract: int = 20,
    seed: int = 42,
    family_regex: str | None = None,
    prefer_explicit_test_split: bool = True,
) -> dict[str, Any]:
    raw_records = load_clauses(clauses_path)
    rows = _normalize_clause_rows(raw_records, family_regex=family_regex)

    train_rows, holdout_rows, metadata = split_external_holdout(
        rows,
        holdout_ratio=holdout_ratio,
        seed=seed,
        prefer_explicit_test_split=prefer_explicit_test_split,
    )

    train_pairs = build_reranker_pairs(
        train_rows,
        negatives_per_positive=negatives_per_positive,
        max_queries_per_contract=max_queries_per_contract,
        seed=seed,
    )
    holdout_pairs = build_reranker_pairs(
        holdout_rows,
        negatives_per_positive=negatives_per_positive,
        max_queries_per_contract=max_queries_per_contract,
        seed=seed + 1,
    )

    metadata.update(
        {
            "clauses_path": str(clauses_path),
            "train_output_path": str(train_output_path),
            "holdout_output_path": str(holdout_output_path),
            "negatives_per_positive": negatives_per_positive,
            "max_queries_per_contract": max_queries_per_contract,
            "train_pairs": len(train_pairs),
            "holdout_pairs": len(holdout_pairs),
            "family_regex": family_regex,
            "prefer_explicit_test_split": prefer_explicit_test_split,
        }
    )

    _write_jsonl(train_output_path, train_pairs)
    _write_jsonl(holdout_output_path, holdout_pairs)

    meta_path = Path(metadata_output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata