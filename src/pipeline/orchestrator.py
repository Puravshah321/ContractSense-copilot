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
from src.pipeline.semantic_filter import filter_and_rerank
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

        trace.append("Stage 2: Retrieving relevant clauses...")
        retrieval_top_k = max(top_k, query_profile.retrieval_depth if query_profile.allow_multi_clause else 5)
        retrieved_raw = self.retriever.retrieve(
            user_query,
            top_k=retrieval_top_k,
            candidate_k=max(15, retrieval_top_k * 2),
        )
        trace.append(f"  -> Retrieved {len(retrieved_raw)} raw chunks")

        trace.append("Stage 3: Semantic filtering and taxonomy reranking...")
        retrieved = filter_and_rerank(user_query, retrieved_raw, query_profile)
        trace.append(f"  -> Kept {len(retrieved)} semantically aligned chunks")

        # Stage 2: Evidence sufficiency check
        trace.append("Stage 4: Checking evidence sufficiency...")
        evidence_check = check_evidence_sufficiency(user_query, retrieved)
        evidence_check["query_profile"] = query_profile.to_dict()
        trace.append(f"  -> Decision: {evidence_check['decision']} (conf: {evidence_check['confidence']})")

        # Stage 3: Decision gate
        trace.append("Stage 5: Decision gate...")
        if evidence_check["decision"] == "INSUFFICIENT":
            trace.append("  -> GATE: NOT_FOUND because no relevant clause passed the match check")
        elif evidence_check["decision"] == "CONFLICTING":
            trace.append("  -> GATE: ESCALATE because evidence appears conflicting")
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
        
        if evidence_check["decision"] == "SUFFICIENT" and should_use_structured_controller(query_profile):
            answer_data = generate_structured_answer(user_query, retrieved, evidence_check, query_profile)
        else:
            answer_data = generate_grounded_answer(
                user_query, retrieved, evidence_check, mode=mode,
            )
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
