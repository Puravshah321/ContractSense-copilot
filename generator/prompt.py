# generator/prompt.py
"""
MAHAK'S PROMPT DESIGN
This file provides prompt templates that work WITH your existing model.py
Your model.py already has built-in prompts in phi2_analysis() and saulm_analysis()
This file is for reference and future LLM integration
"""

def build_prompt(query: str, context: str) -> str:
    """
    Build prompt for contract analysis
    This matches the prompt style used in your model.py
    """
    return f"""Analyze these contract clauses.

Clauses:
{context}

Question: {query}

Answer with exactly:
Decision: [ACCEPT/REVIEW/RENEGOTIATE/ESCALATE]
Risk: [LOW/MEDIUM/HIGH/CRITICAL]
Explanation: [your analysis with Clause ID]
Citation: [clause_id]

Answer:"""


def build_saulm_prompt(query: str, context: str) -> str:
    """
    Build prompt for SaulLM-7B (matches your model.py)
    """
    return f"""<s>[INST] You are ContractSense, a legal contract analyst.

CONTRACT CLAUSES:
{context}

USER QUESTION: {query}

Respond with exactly:
Decision: [ACCEPT/REVIEW/RENEGOTIATE/ESCALATE]
Risk: [LOW/MEDIUM/HIGH/CRITICAL]
Explanation: [Your analysis with [Clause: id]]
Citation: [clause_id]

[/INST]

Decision: """


def build_phi2_prompt(query: str, context: str) -> str:
    """
    Build prompt for Phi-2 (matches your model.py)
    """
    return f"""Analyze these contract clauses.

Clauses:
{context}

Question: {query}

Answer with exactly:
Decision: [ACCEPT/REVIEW/RENEGOTIATE/ESCALATE]
Risk: [LOW/MEDIUM/HIGH/CRITICAL]
Explanation: [your analysis with Clause ID]
Citation: [clause_id]

Answer:"""