"""
ContractSense Grounded Pipeline Orchestrator.
Wires together: Chunker → Retriever → Evidence Checker → Generator → Verifier.
Single entry point for the entire pipeline.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.pipeline.chunker import chunk_document
from src.pipeline.retriever import HybridRetriever
from src.pipeline.evidence_checker import check_evidence_sufficiency
from src.pipeline.generator import generate_grounded_answer
from src.pipeline.verifier import verify_grounding
from src.pipeline.query_understanding import classify_query
from src.pipeline.query_decomposer import decompose_query
from src.pipeline.semantic_filter import filter_and_rerank
from src.pipeline.coverage_model import assess_coverage
from src.pipeline.answer_controller import (
    generate_structured_answer,
    should_use_structured_controller,
)


@dataclass
class PipelineResult:
    query: str
    answer: str
    risk_level: str
    decision: str           # ANSWER | NOT_FOUND | ESCALATE
    confidence: str
    action: str
    evidence: list
    verification: dict
    evidence_check: dict
    query_profile: dict
    coverage: dict
    sub_queries: list
    retrieved_count: int
    chunk_count: int
    latency_ms: float
    pipeline_trace: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class ContractSensePipeline:
    """
    Full grounded pipeline for contract analysis.

    Usage:
        pipeline = ContractSensePipeline()
        pipeline.load_document(pdf_text, "contract.pdf")
        result = pipeline.query("What are the termination rights?")
    """

    def __init__(self, use_dense=False, use_llm=False, dense_model=None):
        self.use_dense = use_dense
        self.use_llm = use_llm
        self.dense_model = dense_model
        self.retriever: Optional[HybridRetriever] = None
        self.chunks = []
        self.document_loaded = False
        self.document_name = ""

    def load_document(self, text, source_name="document"):
        """Chunk a document and build retrieval index."""
        self.document_name = source_name
        self.chunks = chunk_document(text, source_name)

        if not self.chunks:
            self.document_loaded = False
            return 0

        self.retriever = HybridRetriever(
            self.chunks,
            use_dense=self.use_dense,
            dense_model_name=self.dense_model,
        )
        self.document_loaded = True
        return len(self.chunks)

    def query(self, user_query, top_k=3):
        """
        Run the full grounded pipeline for a user query.

        Returns a PipelineResult with full trace of every pipeline stage.
        """
        start = time.time()
        trace = []

        # Gate: document must be loaded
        if not self.document_loaded or not self.retriever:
            return PipelineResult(
                query=user_query,
                answer="Please upload a contract document first.",
                risk_level="N/A",
                decision="NOT_FOUND",
                confidence="LOW",
                action="Upload a PDF contract to begin analysis.",
                evidence=[],
                verification={"verdict": "N/A"},
                evidence_check={"decision": "INSUFFICIENT"},
                query_profile={},
                coverage={},
                sub_queries=[user_query],
                retrieved_count=0,
                chunk_count=0,
                latency_ms=0,
                pipeline_trace=["ERROR: No document loaded"],
            )

        # Stage 1: Understand query + retrieve
        trace.append("Stage 1: Classifying query intent...")
        query_profile = classify_query(user_query)
        trace.append(
            f"  -> Kind: {query_profile.query_kind}, answer_type: {query_profile.answer_type}, "
            f"concepts: {', '.join(query_profile.concepts)}"
        )

        is_factual_path = query_profile.query_kind == "factual" and query_profile.answer_type in {"yes_no", "fact"}
        is_analytical_path = not is_factual_path

        sub_queries = [user_query]
        if is_analytical_path:
            trace.append("Stage 1B: Decomposing analytical query...")
            sub_queries = decompose_query(user_query, query_profile)
            trace.append(f"  -> Generated {len(sub_queries)} sub-queries")

        trace.append("Stage 2: Retrieving relevant clauses...")
        if is_factual_path:
            retrieval_top_k = max(3, min(5, top_k))
            raw_candidates = self.retriever.retrieve(
                user_query,
                top_k=retrieval_top_k,
                candidate_k=max(12, retrieval_top_k * 2),
            )
        else:
            retrieval_top_k = max(query_profile.retrieval_depth, top_k, 8)
            raw_candidates = []
            for sq in sub_queries:
                part = self.retriever.retrieve(
                    sq,
                    top_k=retrieval_top_k,
                    candidate_k=max(20, retrieval_top_k * 3),
                )
                for item in part:
                    enriched = dict(item)
                    enriched["sub_query"] = sq
                    raw_candidates.append(enriched)

        # Deduplicate by clause identity while preserving the best score.
        by_clause = {}
        for item in raw_candidates:
            chunk = item["chunk"]
            key = (chunk.clause_id, chunk.section, chunk.page)
            if key not in by_clause or float(item.get("score", 0.0)) > float(by_clause[key].get("score", 0.0)):
                by_clause[key] = item
        retrieved_raw = sorted(by_clause.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        trace.append(f"  -> Retrieved {len(retrieved_raw)} unique raw chunks")

        trace.append("Stage 3: Semantic filtering and taxonomy reranking...")
        keep_k = 3 if is_factual_path else 8
        retrieved = filter_and_rerank(user_query, retrieved_raw, query_profile, keep_k=keep_k)
        trace.append(f"  -> Kept {len(retrieved)} semantically aligned chunks")

        # Stage 2: Evidence sufficiency check
        trace.append("Stage 4: Checking evidence sufficiency...")
        evidence_check = check_evidence_sufficiency(user_query, retrieved)
        evidence_check["query_profile"] = query_profile.to_dict()
        trace.append(f"  -> Decision: {evidence_check['decision']} (conf: {evidence_check['confidence']})")

        trace.append("Stage 4B: Coverage-based sufficiency model...")
        coverage = assess_coverage(query_profile, retrieved, sub_queries=sub_queries)
        trace.append(
            f"  -> Coverage: {coverage['coverage_ratio']:.0%} ({coverage['decision']}); "
            f"missing: {', '.join(coverage['missing_aspects']) if coverage['missing_aspects'] else 'none'}"
        )

        # Stage 3: Decision gate
        trace.append("Stage 5: Decision gate...")
        analytical_partial = False
        if evidence_check["decision"] == "CONFLICTING":
            trace.append("  -> GATE: ESCALATE because evidence appears conflicting")
        elif is_factual_path and evidence_check["decision"] == "INSUFFICIENT":
            trace.append("  -> GATE: NOT_FOUND (strict factual mode)")
        elif is_analytical_path and evidence_check["decision"] == "INSUFFICIENT" and coverage["decision"] == "PARTIAL":
            analytical_partial = True
            trace.append("  -> GATE: PARTIAL analytical coverage accepted; will answer with explicit gaps")
        elif evidence_check["decision"] == "INSUFFICIENT":
            trace.append("  -> GATE: NOT_FOUND because no relevant clause passed the match check")
        else:
            trace.append("  -> GATE: Sufficient evidence; generating grounded answer")

        # Stage 4: Generate grounded answer
        trace.append("Stage 6: Generating answer with answer-type controller...")
        
        import os
        mode = "rule"
        if os.environ.get("HF_API_KEY"):
            mode = "hf_api"
            trace.append("  -> Routing query to Hugging Face Serverless API...")
        elif os.environ.get("LIGHTNING_API_URL") and os.environ.get("LIGHTNING_API_URL") != "http://REPLACE_WITH_YOUR_NGROK_URL/generate":
            mode = "api"
            trace.append("  -> Routing query to Lightning AI GPU API...")
        elif self.use_llm:
            mode = "llm" # Local GPU Fallback

        effective_evidence_check = dict(evidence_check)
        if analytical_partial:
            effective_evidence_check["decision"] = "SUFFICIENT"
            effective_evidence_check["confidence"] = max(float(effective_evidence_check.get("confidence", 0.0)), 0.35)

        if effective_evidence_check["decision"] == "SUFFICIENT" and should_use_structured_controller(query_profile):
            answer_data = generate_structured_answer(user_query, retrieved, effective_evidence_check, query_profile)
        else:
            answer_data = generate_grounded_answer(
                user_query, retrieved, effective_evidence_check, mode=mode,
            )

        if is_analytical_path and analytical_partial and answer_data["decision"] == "ANSWER":
            if coverage["missing_aspects"]:
                missing = ", ".join(coverage["missing_aspects"])
                answer_data["answer"] += (
                    f"\n\nCoverage gaps: the current evidence does not fully cover these aspects: {missing}."
                )
            answer_data["confidence"] = "MEDIUM"
        trace.append(f"  -> Decision: {answer_data['decision']}, Risk: {answer_data['risk_level']}")

        # Stage 5: Verify grounding
        trace.append("Stage 7: Verifying grounding...")
        verification = verify_grounding(answer_data, retrieved)
        trace.append(f"  -> Verdict: {verification['verdict']} ({verification['supported_ratio']:.0%} supported)")

        # Stage 6: Override if verification fails
        if verification["verdict"] == "REJECTED" and answer_data["decision"] == "ANSWER":
            trace.append("Stage 8: OVERRIDE - unsupported answer changed to NOT_FOUND")
            answer_data["decision"] = "NOT_FOUND"
            answer_data["confidence"] = "HIGH"
            answer_data["risk_level"] = "N/A"
            answer_data["answer"] = (
                "This is not specified in the provided document. The generated answer was "
                "not sufficiently supported by the retrieved evidence."
            )
            answer_data["action"] = ""
            answer_data["evidence"] = []

        latency = (time.time() - start) * 1000
        trace.append(f"Pipeline complete in {latency:.0f}ms")

        return PipelineResult(
            query=user_query,
            answer=answer_data["answer"],
            risk_level=answer_data["risk_level"],
            decision=answer_data["decision"],
            confidence=answer_data["confidence"],
            action=answer_data["action"],
            evidence=answer_data["evidence"],
            verification=verification,
            evidence_check=evidence_check,
            query_profile=query_profile.to_dict(),
            coverage=coverage,
            sub_queries=sub_queries,
            retrieved_count=len(retrieved),
            chunk_count=len(self.chunks),
            latency_ms=round(latency, 1),
            pipeline_trace=trace,
        )

    def get_all_risks(self):
        """
        Scan the entire document for risk clauses.
        Called once after document load for the initial risk report.
        """
        if not self.document_loaded:
            return None

        risk_queries = [
            "What are the termination rights and risks?",
            "What are the liability limitations and caps?",
            "What are the indemnification obligations?",
            "What are the confidentiality requirements?",
            "What are the intellectual property ownership terms?",
            "What are the payment terms and penalties?",
            "What is the dispute resolution mechanism?",
            "Are there any non-compete or restrictive covenants?",
            "What are the warranty representations?",
            "What are the data protection obligations?",
        ]

        results = []
        for q in risk_queries:
            result = self.query(q, top_k=3)
            if result.decision != "NOT_FOUND":
                results.append(result)

        return results
