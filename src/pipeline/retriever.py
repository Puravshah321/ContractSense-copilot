"""
Hybrid Retriever: TF-IDF (lightweight) + optional Dense Embeddings.
Works on CPU for Streamlit Cloud; upgrades to sentence-transformers when available.

Precision upgrade:
  - expand legal query keywords
  - retrieve 10+ candidates, rerank, keep the best evidence-grade clauses
  - boost section/topic matches
  - penalize generic clauses such as Entire Agreement / General boilerplate
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "this", "that", "it", "its", "or",
    "and", "not", "no", "if", "but", "what", "how", "when", "where",
    "who", "which", "there", "this", "agreement", "contract", "clause",
    "section", "include", "includes", "including", "contain", "contains",
}

_QUERY_EXPANSIONS = {
    "warranty": ["warranty", "warranties", "warrants", "guarantee", "guarantees", "representation", "representations"],
    "warranties": ["warranty", "warranties", "warrants", "guarantee", "representation", "representations"],
    "guarantee": ["warranty", "guarantee", "warrants", "representation"],
    "duration": ["duration", "term", "valid", "validity", "period", "effective", "commencement", "expiration", "expires", "year", "month", "day", "signing"],
    "term": ["duration", "term", "valid", "validity", "period", "effective", "commencement", "expiration", "expires", "signing"],
    "long": ["duration", "term", "period", "year", "month", "day"],
    "liability": ["liability", "liable", "cap", "limit", "limitation", "damages", "indemnity", "indemnification"],
    "data": ["data", "information", "confidential", "audit", "personal", "processor", "controller", "breach", "transfer", "share", "disclose", "outside", "cross-border", "use"],
    "india": ["india", "outside", "cross-border", "transfer", "data", "personal"],
    "shared": ["share", "shared", "disclose", "disclosure", "transfer", "third-party", "third party"],
    "share": ["share", "shared", "disclose", "disclosure", "transfer", "third-party", "third party"],
    "confidential": ["confidential", "confidentiality", "disclose", "disclosure", "receiving", "information"],
    "third": ["third-party", "third party", "other person", "entity", "disclose", "disclosure", "approval", "consent"],
    "party": ["third-party", "third party", "other person", "entity", "disclose", "disclosure", "approval", "consent"],
    "external": ["third-party", "third party", "other person", "entity", "disclose", "disclosure", "approval", "consent"],
    "training": ["training", "train", "ai", "model", "use", "audit", "data", "confidential", "scope", "purpose", "copy", "retain"],
    "ai": ["ai", "training", "model", "use", "audit", "data", "confidential", "scope", "purpose", "copy", "retain"],
    "models": ["model", "models", "training", "ai", "use", "audit", "data", "confidential", "scope", "purpose"],
    "penalty": ["penalty", "penalties", "damages", "liquidated", "loss", "compensate", "contract value", "breach", "remedies"],
    "termination": ["termination", "terminate", "expires", "notice", "breach", "term"],
    "payment": ["payment", "pay", "invoice", "fees", "late", "interest"],
    "indemnification": ["indemnification", "indemnify", "indemnity", "hold harmless", "third-party", "claim"],
}

_SECTION_PRIORITIES = {
    "warranty": ["warranty", "representations", "representation", "liability"],
    "guarantee": ["warranty", "representations", "representation"],
    "duration": ["term", "duration", "commencement", "expiration", "valid"],
    "term": ["term", "duration", "commencement", "expiration", "valid"],
    "liability": ["limitation of liability", "liability", "indemnification"],
    "data": ["data protection", "privacy", "security", "confidentiality", "protection of confidential information", "permitted disclosure", "need to know"],
    "india": ["data protection", "privacy", "security", "confidentiality"],
    "termination": ["termination", "term"],
    "share": ["permitted disclosure", "need to know", "confidentiality", "protection of confidential information"],
    "shared": ["permitted disclosure", "need to know", "confidentiality", "protection of confidential information"],
    "training": ["protection of confidential information", "confidentiality", "definitions"],
    "ai": ["protection of confidential information", "confidentiality", "definitions"],
    "penalty": ["remedies", "liability"],
}

_GENERIC_SECTION_TERMS = {
    "entire agreement", "miscellaneous", "general", "severability", "waiver",
    "amendment", "notices", "counterparts", "interpretation",
}


def _tokenize(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())) - _STOPWORDS


def _trigger_present(trigger, query_lower):
    if " " in trigger or "-" in trigger:
        return trigger in query_lower
    return bool(re.search(rf"\b{re.escape(trigger)}\b", query_lower))


def expand_query_keywords(query):
    """Return strict, legally useful query keywords used by retrieval and gates."""
    q_lower = query.lower()
    terms = _tokenize(query)
    for trigger, additions in _QUERY_EXPANSIONS.items():
        if _trigger_present(trigger, q_lower):
            terms.update(additions)
    return {t.lower() for t in terms if len(t) >= 3}


def clause_keyword_overlap(query, chunk):
    """Score whether a chunk actually talks about the user's requested topic."""
    query_terms = expand_query_keywords(query)
    if not query_terms:
        return 0.0

    section_text = f"{chunk.section} {chunk.text[:500]}".lower()
    hits = 0
    for term in query_terms:
        if term in section_text:
            hits += 1
    return hits / max(len(query_terms), 1)


def _section_priority_bonus(query, chunk):
    q_lower = query.lower()
    section = (chunk.section or "").lower()
    head = chunk.text[:250].lower()
    bonus = 0.0
    for trigger, preferred_sections in _SECTION_PRIORITIES.items():
        if not _trigger_present(trigger, q_lower):
            continue
        for preferred in preferred_sections:
            if preferred in section:
                bonus = max(bonus, 0.35)
            elif preferred in head:
                bonus = max(bonus, 0.18)
    return bonus


def _generic_penalty(chunk):
    section = (chunk.section or "").lower()
    head = chunk.text[:180].lower()
    if "entire agreement" in section or "entire agreement" in head:
        return 0.45
    if "survival" in section and any(t in head for t in ["term", "duration", "valid up to"]):
        return 0.0
    if any(term == section.strip() for term in _GENERIC_SECTION_TERMS):
        return 0.20
    if any(term in head[:80] for term in _GENERIC_SECTION_TERMS):
        return 0.15
    return 0.0


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

    def _expanded_query_text(self, query):
        keywords = sorted(expand_query_keywords(query))
        if not keywords:
            return query
        return f"{query} " + " ".join(keywords)

    def _tfidf_search(self, query, top_k=10):
        query_vec = self._tfidf.transform([self._expanded_query_text(query)])
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

    def _rerank(self, query, fused_results):
        reranked = []
        for idx, base_score in fused_results:
            chunk = self.chunks[idx]
            keyword_score = clause_keyword_overlap(query, chunk)
            section_bonus = _section_priority_bonus(query, chunk)
            penalty = _generic_penalty(chunk)
            q_lower = query.lower()
            chunk_text = f"{chunk.section} {chunk.text[:700]}".lower()
            if ("warranty" in q_lower or "warrant" in q_lower) and not re.search(r"\bwarrant(?:y|ies|s)?\b|\bguarantee\b", chunk_text):
                penalty += 0.75
            if ("duration" in q_lower or "term" in q_lower) and "survival" in chunk_text and "valid up to" not in chunk_text:
                penalty += 0.35
            if ("duration" in q_lower or "term" in q_lower) and re.search(r"\bterm\b.*\bvalid\s+up\s+to\b|\bvalid\s+up\s+to\b.*\bone\s+year\b", chunk_text):
                section_bonus += 0.75
            if any(t in q_lower for t in ["share", "third", "external"]) and re.search(r"\bnot\s+to\s+disclose\b|\bthird\s+party\b|\bother\s+person\s+or\s+entity\b|\bneed\s+to\s+know\b", chunk_text):
                section_bonus += 0.45
            if any(_trigger_present(t, q_lower) for t in ["ai", "training", "model"]) and re.search(r"\buse\b.*\bscope\s+of\s+audit\b|\bnot\s+to\s+make\s+or\s+retain\s+copy\b|\baudit\s+information\b", chunk_text):
                section_bonus += 0.45
            if "penalty" in q_lower and re.search(r"\bliquidated\s+damages\b|\bcontract\s+value\b|\bloss\s+or\s+damages\b", chunk_text):
                section_bonus += 0.55
            final_score = float(base_score) + (0.55 * keyword_score) + section_bonus - penalty
            reranked.append((idx, final_score, keyword_score, section_bonus, penalty))
        return sorted(reranked, key=lambda x: x[1], reverse=True)

    def retrieve(self, query, top_k=3, candidate_k=10):
        """
        Retrieve candidates, rerank them, and keep the top evidence clauses.

        Returns list of dicts:
          {chunk: Chunk, score: float, retrieval_method: str}
        """
        candidate_k = max(candidate_k, top_k * 3, 10)
        sparse = self._tfidf_search(query, top_k=candidate_k)

        if self._dense_model is not None:
            dense = self._dense_search(query, top_k=candidate_k)
            fused = self._rrf_fuse(sparse, dense)
            method = "hybrid_rrf"
        else:
            fused = [(idx, score) for idx, score in sparse]
            method = "tfidf_only"

        reranked = self._rerank(query, fused)
        results = []
        for idx, score, keyword_score, section_bonus, penalty in reranked[:top_k]:
            results.append({
                "chunk": self.chunks[idx],
                "score": score,
                "retrieval_method": method,
                "keyword_overlap": round(keyword_score, 3),
                "section_bonus": round(section_bonus, 3),
                "generic_penalty": round(penalty, 3),
            })
        return results
