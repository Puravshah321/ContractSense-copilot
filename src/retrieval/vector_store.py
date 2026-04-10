"""Qdrant Vector Store for ContractSense.

Handles all interaction with the Qdrant vector database:
  - Creating / resetting a collection
  - Upserting clause embeddings (in batches)
  - Searching by dense query vector
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "contractsense_clauses"
DEFAULT_TOP_K = 5
UPSERT_BATCH_SIZE = 256  # Qdrant handles large batches fine


class QdrantVectorStore:

    def __init__(
        self,
        qdrant_url: str | None = None,
        persist_path: str | Path | None = None,
        collection_name: str = COLLECTION_NAME,
        embedding_dim: int = 384,         # all-MiniLM-L6-v2 default
    ) -> None:
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        if qdrant_url:
            log.info("Connecting to Qdrant server at %s", qdrant_url)
            self.client = QdrantClient(url=qdrant_url)
        elif persist_path:
            path = str(Path(persist_path))
            log.info("Using on-disk Qdrant at %s", path)
            self.client = QdrantClient(path=path)
        else:
            log.info("Using in-memory Qdrant (ephemeral)")
            self.client = QdrantClient(":memory:")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, reset: bool = False) -> None:
        """Create the Qdrant collection (optionally resetting if it exists)."""
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in existing:
            if reset:
                log.warning("Resetting existing collection '%s'", self.collection_name)
                self.client.delete_collection(self.collection_name)
            else:
                log.info("Collection '%s' already exists — skipping creation.", self.collection_name)
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        log.info(
            "Created collection '%s'  (dim=%d, metric=cosine)",
            self.collection_name,
            self.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert(
        self,
        clauses: list[dict],
        embeddings: np.ndarray,
        batch_size: int = UPSERT_BATCH_SIZE,
    ) -> None:
        assert len(clauses) == len(embeddings), (
            f"Mismatch: {len(clauses)} clauses vs {len(embeddings)} embeddings"
        )

        total = len(clauses)
        uploaded = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_clauses = clauses[start:end]
            batch_vecs = embeddings[start:end]

            points = []
            for i, (clause, vec) in enumerate(zip(batch_clauses, batch_vecs)):
                point_id = start + i   # sequential integer ID

                # Store all metadata as Qdrant payload (searchable / filterable)
                payload: dict[str, Any] = {
                    "clause_id":    clause.get("clause_id", ""),
                    "contract_id":  clause.get("contract_id", ""),
                    "split":        clause.get("split", ""),
                    "clause_index": clause.get("clause_index", -1),
                    "num_clauses":  clause.get("num_clauses", -1),
                    "char_count":   clause.get("char_count", 0),
                    "clause_text":  clause.get("clause_text", ""),
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vec.tolist(),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            uploaded += len(points)

            if (start // batch_size) % 10 == 0:
                log.info("Upserted %d / %d points", uploaded, total)

        log.info("✅ Upsert complete. Total points in collection: %d", total)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float | None = None,
        contract_id_filter: str | None = None,
    ) -> list[dict]:
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        query_filter = None
        if contract_id_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="contract_id",
                        match=MatchValue(value=contract_id_filter),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            result = {"score": hit.score, "point_id": hit.id}
            result.update(hit.payload or {})
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of vectors stored in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count  # type: ignore[return-value]

    def collection_info(self) -> dict:
        """Return a summary dict of the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name":            self.collection_name,
            "points_count":    info.points_count,
            "vector_size":     info.config.params.vectors.size,
            "distance_metric": info.config.params.vectors.distance.value,
            "status":          info.status,
        }
