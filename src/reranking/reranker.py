"""Cross-encoder reranking utilities for ContractSense.

The reranker takes a short query and a list of candidate clauses, scores each
query-clause pair with a cross-encoder, and returns the best clauses ordered by
relevance. An optional risk-aware blend can boost high-risk clauses when the
payload contains risk metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class RerankCandidate:
    """A single clause candidate presented to the reranker."""

    clause_id: str
    clause_text: str
    score: float = 0.0
    risk_score: float | None = None
    payload: dict[str, Any] | None = None


def _candidate_text(candidate: dict[str, Any]) -> str:
    text = candidate.get("clause_text") or candidate.get("text") or ""
    return str(text).strip()


def _candidate_risk_score(candidate: dict[str, Any]) -> float | None:
    if "risk_score" in candidate and candidate["risk_score"] is not None:
        try:
            return float(candidate["risk_score"])
        except (TypeError, ValueError):
            return None

    risk_label = str(candidate.get("risk_label", "")).strip().lower()
    if not risk_label:
        return None
    if risk_label in {"high", "high_risk", "critical"}:
        return 1.0
    if risk_label in {"medium", "moderate"}:
        return 0.5
    if risk_label in {"low", "low_risk", "none"}:
        return 0.0
    return None


class CrossEncoderReranker:
    """Score and rerank clause candidates with a cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        risk_weight: float = 0.15,
    ) -> None:
        self.model_name = model_name
        self.risk_weight = risk_weight
        log.info("Loading reranker model: %s", model_name)
        self.model = CrossEncoder(model_name, device=device)

    def score_pairs(self, query: str, clauses: Iterable[str]) -> np.ndarray:
        pairs = [(query, clause) for clause in clauses]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        return np.asarray(scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        apply_risk_boost: bool = True,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        texts = [_candidate_text(candidate) for candidate in candidates]
        scores = self.score_pairs(query, texts)

        reranked: list[dict[str, Any]] = []
        for candidate, base_score in zip(candidates, scores, strict=False):
            clause = dict(candidate)
            clause["reranker_score"] = float(base_score)

            risk_score = _candidate_risk_score(candidate)
            clause["risk_score"] = risk_score

            final_score = float(base_score)
            if apply_risk_boost and risk_score is not None:
                final_score = float(base_score + (self.risk_weight * risk_score))

            clause["final_rerank_score"] = final_score
            reranked.append(clause)

        reranked.sort(key=lambda item: item["final_rerank_score"], reverse=True)
        return reranked[:top_k]

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_path))
        log.info("Saved reranker checkpoint to %s", output_path)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        device: str | None = None,
        risk_weight: float = 0.15,
    ) -> "CrossEncoderReranker":
        instance = cls.__new__(cls)
        instance.model_name = str(model_path)
        instance.risk_weight = risk_weight
        instance.model = CrossEncoder(str(model_path), device=device)
        return instance