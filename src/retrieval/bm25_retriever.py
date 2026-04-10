"""BM25 Baseline Retriever for ContractSense.

Implements a classic sparse retrieval baseline using BM25 (Okapi BM25).
This serves as the comparison baseline against the dense retriever
(Legal-BERT / all-MiniLM) stored in Qdrant.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import string
from pathlib import Path

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    """a about an and are as at be been by for from has had
    he in is it its of on or that the to was were will
    with shall this which such any""".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t and t not in _STOP_WORDS]
    return tokens


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:

    def __init__(
        self,
        clauses: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.clauses = clauses
        log.info("Tokenizing %d clauses for BM25 index…", len(clauses))
        self._corpus: list[list[str]] = [
            _tokenize(c["clause_text"]) for c in clauses
        ]
        self._bm25 = BM25Okapi(self._corpus, k1=k1, b=b)
        log.info("BM25 index built successfully.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            log.warning("Query tokenized to empty list — returning no results.")
            return []

        scores = self._bm25.get_scores(q_tokens)

        # Get top-k indices sorted by descending score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            clause = dict(self.clauses[idx])   # copy
            clause["bm25_score"] = float(scores[idx])
            results.append(clause)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the retriever (index + clause list) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("BM25 retriever saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        """Load a previously pickled BM25Retriever from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found at {path}")
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        log.info("BM25 retriever loaded from %s (%d clauses)", path, len(obj.clauses))
        return obj


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_bm25_from_jsonl(
    clauses_path: str | Path,
    save_index_path: str | Path | None = None,
) -> BM25Retriever:
    clauses_path = Path(clauses_path)
    clauses: list[dict] = []
    with clauses_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                clauses.append(json.loads(line))

    log.info("Loaded %d clauses from %s", len(clauses), clauses_path)
    retriever = BM25Retriever(clauses)

    if save_index_path:
        retriever.save(save_index_path)

    return retriever
