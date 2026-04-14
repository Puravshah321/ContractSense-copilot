# generator/output_format.py
"""
MAHAK'S OUTPUT FORMAT - Matches the JSON output from your model.py
Your model.py already produces this exact format in rule_based_analysis()
"""

from typing import Literal
from pydantic import BaseModel, Field

class ContractOutput(BaseModel):
    """
    Standard output format for contract analysis
    This EXACTLY matches what your model.py returns
    """
    
    decision: Literal["ACCEPT", "REVIEW", "RENEGOTIATE", "ESCALATE"] = Field(
        description="Recommended action for this clause",
        example="REVIEW"
    )
    
    risk: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Risk level assessment",
        example="HIGH"
    )
    
    explanation: str = Field(
        description="Plain English explanation with citations. Your model.py includes [Clause: id] format.",
        example="HIGH RISK: This indemnification clause transfers significant liability... [Clause: train_00348_clause_211]",
        max_length=500
    )
    
    citation: str = Field(
        description="Clause ID from the knowledge base",
        example="train_00348_clause_211"
    )


# Example of what YOUR model.py produces
YOUR_MODEL_OUTPUT_EXAMPLE = {
    "decision": "REVIEW",
    "risk": "HIGH",
    "explanation": "HIGH RISK: This indemnification clause transfers significant legal and financial liability to your company. [Clause: train_00348_clause_211]",
    "citation": "train_00348_clause_211"
}


def validate_output(output: dict) -> bool:
    """
    Validate that output matches required format
    Use this to test your model.py outputs
    """
    required_keys = ["decision", "risk", "explanation", "citation"]
    
    # Check all keys exist
    for key in required_keys:
        if key not in output:
            print(f" Missing key: {key}")
            return False
    
    # Check decision values
    valid_decisions = ["ACCEPT", "REVIEW", "RENEGOTIATE", "ESCALATE"]
    if output["decision"].upper() not in valid_decisions:
        print(f" Invalid decision: {output['decision']}")
        return False
    
    # Check risk values
    valid_risks = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if output["risk"].upper() not in valid_risks:
        print(f" Invalid risk: {output['risk']}")
        return False
    
    # Check explanation has citation
    if "[Clause:" not in output["explanation"] and "Clause:" not in output["explanation"]:
        print(f" Warning: Explanation missing citation format")
    
    return True


def format_output(decision: str, risk: str, explanation: str, citation: str) -> dict:
    """Helper function to format output consistently with your model.py"""
    return {
        "decision": decision.upper(),
        "risk": risk.upper(),
        "explanation": explanation.strip(),
        "citation": citation.strip()
    }