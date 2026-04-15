"""DistilBERT-based reranker baseline for ContractSense.

This baseline is intentionally simple: it scores query-clause pairs with a
sequence classification head and reranks candidates by the predicted relevance
logit. It is useful for the notebook comparison matrix and as a compact
baseline against larger cross-encoder backbones.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)

DEFAULT_DISTILBERT_MODEL = "distilbert-base-uncased"


class DistilBERTReranker:
    """Score query-clause pairs with a DistilBERT sequence classifier."""

    def __init__(self, model_name: str = DEFAULT_DISTILBERT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, query: str, clauses: list[str], max_length: int = 256) -> np.ndarray:
        inputs = self.tokenizer(
            [query] * len(clauses),
            clauses,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)
        return logits.detach().cpu().numpy().astype(np.float32)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidates:
            return []

        texts = [str(candidate.get("clause_text") or candidate.get("text") or "") for candidate in candidates]
        scores = self.score_pairs(query, texts)

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores, strict=False):
            row = dict(candidate)
            row["distilbert_score"] = float(score)
            reranked.append(row)

        reranked.sort(key=lambda item: item["distilbert_score"], reverse=True)
        return reranked[:top_k]

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        log.info("Saved DistilBERT reranker to %s", output_path)
