# generator/prompt.py - Improved version

def build_system_prompt() -> str:
    """System prompt that defines the assistant's role"""
    return """You are ContractSense, an expert contract analyst.

Rules:
1. Always cite the clause: [Clause: clause_id]
2. Start with risk level (LOW/MEDIUM/HIGH/CRITICAL)
3. Give actionable decision (ACCEPT/REVIEW/RENEGOTIATE/ESCALATE)
4. Explain in plain business English

Risk definitions:
- LOW: Standard, no concerns
- MEDIUM: Review recommended
- HIGH: Legal review required
- CRITICAL: Do not sign, escalate"""


def build_prompt(query: str, context: str) -> str:
    """Build prompt for contract analysis"""
    return f"""Analyze these contract clauses.

Clauses:
{context}

Question: {query}

Answer with exactly:
Decision: [ACCEPT/REVIEW/RENEGOTIATE/ESCALATE]
Risk: [LOW/MEDIUM/HIGH/CRITICAL]
Explanation: [your analysis with [Clause: id]]
Citation: [clause_id]

Answer:"""


def build_saulm_prompt(query: str, context: str) -> str:
    """Build prompt for SaulLM-7B"""
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
    """Build prompt for Phi-2"""
    return f"""Analyze these contract clauses.

Clauses:
{context}

Question: {query}

Answer with exactly:
Decision: [ACCEPT/REVIEW/RENEGOTIATE/ESCALATE]
Risk: [LOW/MEDIUM/HIGH/CRITICAL]
Explanation: [your analysis with [Clause: id]]
Citation: [clause_id]

Answer:"""