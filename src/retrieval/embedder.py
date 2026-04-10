"""Clause Embedder for ContractSense knowledge base.

Reads the processed clauses JSONL file produced by clause_segmenter.py and
encodes every clause text into a dense vector using a sentence-transformers
compatible model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_clauses(clauses_path: str | Path) -> list[dict]:
    """Load all clause records from a JSONL file."""
    path = Path(clauses_path)
    if not path.exists():
        raise FileNotFoundError(f"Clauses file not found: {path}")

    clauses: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                clauses.append(json.loads(line))

    log.info("Loaded %d clause records from %s", len(clauses), path)
    return clauses


def _batch(items: list, size: int) -> Iterator[list]:
    """Yield successive chunks of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# Core embedder
# ---------------------------------------------------------------------------

class ClauseEmbedder:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
    ) -> None:
        log.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        log.info("Embedding dimension: %d", self.embedding_dim)

    def embed(self, clauses: list[dict]) -> np.ndarray:
        texts = [c["clause_text"] for c in clauses]
        log.info(
            "Embedding %d clauses (batch_size=%d)…", len(texts), self.batch_size
        )
        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity = dot product
        )
        log.info("Done. Embedding matrix shape: %s", embeddings.shape)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string → 1-D float32 vector."""
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0].astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed CUAD clauses into dense vectors.")
    p.add_argument(
        "--clauses-path",
        type=Path,
        default=Path("data/processed/clauses.jsonl"),
        help="Path to the processed clauses JSONL file.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="sentence-transformers model to use.",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/clause_embeddings.npy"),
        help="Where to save the (N, D) numpy array of embeddings.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Encoding batch size.",
    )
    return p

def main() -> None:
    args = build_parser().parse_args()
    clauses = load_clauses(args.clauses_path)
    embedder = ClauseEmbedder(
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    embeddings = embedder.embed(clauses)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), embeddings)
    log.info("Saved embeddings → %s  (shape=%s)", out, embeddings.shape)


if __name__ == "__main__":
    main()
