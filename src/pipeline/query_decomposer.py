"""
Query Decomposer — upgraded.

Improvement #2: Concept-aware query decomposition.
Breaks a complex legal question into targeted sub-queries,
each covering a distinct legal intent (obligation, right,
liability, time reference, monetary implication, remedy).
"""
import re
from src.pipeline.query_understanding import QueryProfile


# ── Sub-query templates per legal concept ──────────────────────────
_CONCEPT_TEMPLATES = {
    "sla": [
        "What are the SLA performance metrics and uptime requirements?",
        "What are the service credit or compensation terms for SLA breach?",
        "What is the liability cap or remedy for SLA failures?",
    ],
    "liability": [
        "What is the limitation of liability or cap on damages?",
        "What types of damages are excluded from liability?",
        "Are there indemnification obligations related to this?",
    ],
    "termination": [
        "What are the grounds and notice requirements for termination?",
        "What obligations survive after termination?",
        "Are there cure periods before termination takes effect?",
        "What payments or obligations arise upon termination?",
    ],
    "financial_commitment": [
        "What are the direct payment obligations and amounts?",
        "What are the penalties or liquidated damages for breach?",
        "What are the liability caps on financial exposure?",
    ],
    "confidentiality": [
        "What information is classified as confidential?",
        "What are the obligations of the receiving party regarding confidential information?",
        "What are the permitted disclosures of confidential information?",
        "How long does the confidentiality obligation last?",
    ],
    "data_use": [
        "What are the permitted uses of the data under this agreement?",
        "Are there restrictions on storing or transferring data outside of jurisdiction?",
        "What security or data protection obligations apply?",
    ],
    "ip": [
        "Who owns the intellectual property created under this agreement?",
        "Are there any IP assignment or license provisions?",
        "What IP rights does each party retain?",
    ],
    "compliance": [
        "What regulatory or legal compliance obligations apply?",
        "Are there audit rights or compliance verification provisions?",
        "What happens in case of a compliance failure?",
    ],
    "dispute": [
        "What is the dispute resolution mechanism?",
        "Which jurisdiction and governing law applies?",
        "Is arbitration or mediation required before litigation?",
    ],
    "warranty": [
        "What warranties does each party provide?",
        "Are there any implied warranty disclaimers?",
        "What is the remedy for breach of warranty?",
    ],
    "force_majeure": [
        "What events qualify as force majeure?",
        "What obligations are excused under force majeure?",
        "What notice is required to invoke force majeure?",
    ],
}

# ── Cross-clause multi-hop templates (Improvement #11) ─────────────
_MULTI_HOP_TEMPLATES = [
    ("payment", "termination",
     ["What payments become due upon termination?",
      "Are there any penalties that apply when the contract terminates?"]),
    ("sla", "liability",
     ["If the SLA is breached, what is the maximum financial exposure?",
      "Do liability caps apply to SLA breach compensation?"]),
    ("confidentiality", "survival",
     ["Does the confidentiality obligation survive termination?",
      "How long does confidentiality last after the contract ends?"]),
    ("termination", "liability",
     ["What is the liability of each party after termination?",
      "Does the limitation of liability clause apply post-termination?"]),
    ("ip", "termination",
     ["What happens to IP rights if the contract is terminated?",
      "Can the licensee continue using IP after termination?"]),
]


def _has_concept(concepts, name):
    return name in concepts


def decompose_query(query, query_profile: QueryProfile):
    """
    Decompose a complex query into targeted sub-queries.

    Returns a list of strings (original query + sub-queries).
    Always includes the original query as the first element.
    """
    sub_queries = [query]
    concepts = set(query_profile.concepts)

    # Add concept-specific sub-queries
    for concept, templates in _CONCEPT_TEMPLATES.items():
        if _has_concept(concepts, concept):
            sub_queries.extend(templates)

    # Add multi-hop cross-clause queries (Improvement #11)
    for c1, c2, templates in _MULTI_HOP_TEMPLATES:
        if _has_concept(concepts, c1) and _has_concept(concepts, c2):
            sub_queries.extend(templates)

    # Analytical fallback: if no concept-specific decomposition, use generic
    if len(sub_queries) == 1 and query_profile.query_kind in {"analytical", "risk"}:
        sub_queries += [
            "What obligations does this create?",
            "What rights does this grant?",
            "What are the financial or liability implications?",
            "What are the time limits or notice requirements?",
            "What remedies are available in case of breach?",
        ]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for sq in sub_queries:
        if sq not in seen:
            seen.add(sq)
            unique.append(sq)

    return unique
