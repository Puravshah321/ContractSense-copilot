"""
Hybrid Retriever: TF-IDF (lightweight) + optional Dense Embeddings.
Works on CPU for Streamlit Cloud; upgrades to sentence-transformers when available.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRetriever:
    """
    Two-stage retriever:
      1. TF-IDF sparse retrieval (always available, CPU-only)
      2. Dense embedding retrieval (optional, when sentence-transformers is installed)
    Combines via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, chunks, use_dense=False, dense_model_name=None):
        """
        Args:
            chunks: list of Chunk objects (must have .text attribute)
            use_dense: whether to load sentence-transformers for dense retrieval
            dense_model_name: model name for dense embeddings
        """
        self.chunks = chunks
        self.texts = [c.text for c in chunks]

        self._tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf.fit_transform(self.texts)

        self._dense_model = None
        self._dense_embeddings = None
        if use_dense:
            self._load_dense(dense_model_name or "BAAI/bge-small-en-v1.5")

    def _load_dense(self, model_name):
        try:
            from sentence_transformers import SentenceTransformer
            self._dense_model = SentenceTransformer(model_name)
            self._dense_embeddings = self._dense_model.encode(
                self.texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except ImportError:
            pass

    def _tfidf_search(self, query, top_k=10):
        query_vec = self._tfidf.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _dense_search(self, query, top_k=10):
        if self._dense_model is None or self._dense_embeddings is None:
            return []
        q_emb = self._dense_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        )
        scores = np.dot(self._dense_embeddings, q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _rrf_fuse(self, sparse_results, dense_results, k=60):
        """Reciprocal Rank Fusion to combine sparse + dense rankings."""
        scores = {}
        for rank, (idx, _) in enumerate(sparse_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k chunks for the given query.

        Returns list of dicts:
          {chunk: Chunk, score: float, retrieval_method: str}
        """
        sparse = self._tfidf_search(query, top_k=top_k * 2)

        if self._dense_model is not None:
            dense = self._dense_search(query, top_k=top_k * 2)
            fused = self._rrf_fuse(sparse, dense)
            method = "hybrid_rrf"
        else:
            fused = [(idx, score) for idx, score in sparse]
            method = "tfidf_only"

        results = []
        for idx, score in fused[:top_k]:
            results.append({
                "chunk": self.chunks[idx],
                "score": score,
                "retrieval_method": method,
            })
        return results
