"""
Grounded Answer Generator.
Two modes:
  1. Rule-based (CPU, works everywhere) - for Streamlit Cloud demos
  2. LLM-based (GPU, uses DPO-aligned Mistral) - for Lightning AI / local GPU
Both modes enforce strict grounding: answer ONLY from evidence, cite sources.
"""
import re
import json


# ============================================================
# STRICT GROUNDING PROMPT (used in LLM mode)
# ============================================================
SYSTEM_PROMPT = """You are ContractSense, a contract analysis system.

STRICT RULES:
1. Use ONLY the provided clause evidence to answer. Do NOT add information not present in the evidence.
2. If the answer is not contained in the evidence, respond with decision: "NOT_FOUND" and say: "This is not specified in the provided document."
3. ALWAYS cite the clause_id and quote the exact text you are referencing.
4. Do NOT generate generic legal advice. Every claim must be traceable to a specific clause.
5. Classify risk as LOW, MEDIUM, HIGH, or CRITICAL based on the actual clause language.

OUTPUT FORMAT (strict JSON):
{
  "answer": "Plain English explanation grounded in evidence",
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "evidence": [{"clause_id": "...", "text": "exact quote from clause"}],
  "confidence": "HIGH | MEDIUM | LOW",
  "decision": "ANSWER | NOT_FOUND | ESCALATE",
  "action": "Concrete recommended action based on the clause"
}"""


def _build_llm_prompt(query, evidence_chunks):
    """Build the full prompt for the LLM with evidence context."""
    context_parts = []
    for i, chunk_data in enumerate(evidence_chunks):
        c = chunk_data["chunk"]
        context_parts.append(
            f"[Evidence {i+1}] clause_id: {c.clause_id} | section: {c.section} | page: {c.page}\n"
            f"{c.text}"
        )
    context = "\n\n".join(context_parts)
    return f"{SYSTEM_PROMPT}\n\n---\nEVIDENCE:\n{context}\n\n---\nQUERY: {query}\n\nRespond in strict JSON:"


# ============================================================
# RISK DETECTION PATTERNS (used in rule-based mode)
# ============================================================
_RISK_PATTERNS = {
    "CRITICAL": [
        r"unlimited\s+liabilit", r"no\s+(?:cap|limit|ceiling)",
        r"sole\s+discretion", r"waive\s+all\s+(?:rights|claims)",
        r"irrevocab(?:le|ly)", r"perpetual\s+(?:license|right)",
    ],
    "HIGH": [
        r"terminat(?:e|ion)\s+(?:immediately|without\s+(?:cause|notice))",
        r"shall\s+not\s+exceed", r"consequential\s+damages?\s+(?:are\s+)?excluded",
        r"(?:all|any)\s+intellectual\s+property.*(?:assign|transfer)",
        r"immediate(?:ly)?\s+(?:due|payable)", r"non[\-\s]?compet(?:e|ition)",
        r"indemnif(?:y|ication)", r"hold\s+harmless",
    ],
    "MEDIUM": [
        r"(?:prior\s+)?written\s+(?:notice|consent)", r"reasonable\s+efforts?",
        r"confidential(?:ity)?", r"(?:\d+)\s+(?:days?|months?|years?)",
        r"governing\s+law", r"arbitrat(?:ion|e)",
        r"warranty", r"represent(?:ation)?s?\s+and\s+warrant",
    ],
    "LOW": [
        r"standard\s+terms?", r"mutual(?:ly)?\s+agree",
        r"best\s+efforts?", r"commercially\s+reasonable",
    ],
}


def _detect_risk_level(text):
    text_lower = text.lower()
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        for pattern in _RISK_PATTERNS[level]:
            if re.search(pattern, text_lower):
                return level
    return "MEDIUM"


_ACTION_TEMPLATES = {
    "CRITICAL": (
        "This clause requires immediate legal review. Engage your legal team to negotiate "
        "amendments before signing. Consider rejecting or significantly modifying this provision."
    ),
    "HIGH": (
        "Review this clause with legal counsel. Negotiate specific carve-outs, caps, or "
        "modifications to reduce your exposure. Document your negotiation positions."
    ),
    "MEDIUM": (
        "Standard commercial language but verify the specific terms align with your business "
        "requirements. Consider whether the timeframes, thresholds, or obligations are appropriate."
    ),
    "LOW": (
        "This clause contains standard, balanced terms. No immediate action required, "
        "but include in your overall contract review checklist."
    ),
}


def generate_grounded_answer(query, evidence_chunks, evidence_check, mode="rule"):
    """
    Generate a strictly grounded answer.

    Args:
        query: user question
        evidence_chunks: list of retrieved chunk dicts
        evidence_check: output from evidence_checker
        mode: "rule" for CPU-only, "llm" for GPU model

    Returns:
        dict with: answer, risk_level, evidence, confidence, decision, action
    """
    # DECISION GATE: refuse if evidence is insufficient
    if evidence_check["decision"] == "INSUFFICIENT" or not evidence_chunks:
        return {
            "answer": "This is not specified in the provided document. The uploaded contract does not contain clauses that address this question.",
            "risk_level": "N/A",
            "evidence": [],
            "confidence": "LOW",
            "decision": "NOT_FOUND",
            "action": "Upload a contract that contains relevant clauses, or rephrase your question.",
        }

    if mode == "hf_api":
        return _generate_hf_api_answer(query, evidence_chunks, evidence_check)

    if mode == "api":
        return _generate_api_answer(query, evidence_chunks, evidence_check)

    if mode == "llm":
        return _generate_llm_answer(query, evidence_chunks, evidence_check)

    return _generate_rule_answer(query, evidence_chunks, evidence_check)


def _generate_hf_api_answer(query, evidence_chunks, evidence_check):
    """
    Calls the Hugging Face Free Serverless Inference API.
    """
    import os
    import requests

    # The user should set these environment variables
    hf_api_key = os.environ.get("HF_API_KEY")
    repo_id = os.environ.get("HF_REPO_ID", "22Jay/ContractSense-Grounded-DPO") # Point to the new DPO aligned model!
    
    if not hf_api_key:
        print("HF_API_KEY not found. Falling back to rule-based generation.")
        return _generate_rule_answer(query, evidence_chunks, evidence_check)

    api_url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    context_parts = []
    for i, r in enumerate(evidence_chunks):
        c = r["chunk"]
        context_parts.append(f"[{c.clause_id}] {c.section}: {c.text}")
    
    prompt = (
        f"[INST] You are ContractSense, a grounded contract analysis system.\n\n"
        f"EVIDENCE:\n" + "\n".join(context_parts) + "\n\n"
        f"QUERY: {query}\n\n"
        f"STRICT RULES:\n"
        f"1. Answer ONLY using the Evidence above.\n"
        f"2. If the answer is not in the evidence, reply with DECISION: NOT_FOUND.\n"
        f"3. Otherwise, quote the evidence, cite the clause_id, and provide a RISK building.\n"
        f"Output in JSON format matching {{'answer': '...', 'decision': 'ANSWER', 'risk_level': 'MEDIUM'}} if possible."
        f"[/INST]"
    )

    try:
        resp = requests.post(
            api_url, 
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 400, "temperature": 0.2, "return_full_text": False}
            },
            timeout=40
        )
        if resp.status_code == 200:
            result = resp.json()
            response_text = result[0]["generated_text"]
            
            # Simple extraction 
            decision = "ANSWER"
            if "NOT_FOUND" in response_text or "not specified" in response_text.lower():
                decision = "NOT_FOUND"

            import re
            risk_match = re.search(r"RISK:\s*(LOW|MEDIUM|HIGH|CRITICAL)", response_text)
            risk_level = risk_match.group(1) if risk_match else "MEDIUM"

            # Format UI Evidence
            evidence_list = []
            for r in evidence_chunks[:3]:
                c = r["chunk"]
                evidence_list.append({
                    "clause_id": c.clause_id,
                    "section": c.section,
                    "page": c.page,
                    "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
                })
                
            return {
                "answer": response_text.replace("```json", "").replace("```", "").strip(),
                "risk_level": risk_level,
                "evidence": evidence_list,
                "confidence": "HIGH",
                "decision": decision,
                "action": _ACTION_TEMPLATES.get(risk_level, _ACTION_TEMPLATES["MEDIUM"]),
            }
        elif str(resp.status_code).startswith("5"):
            print(f"HF API Model is loading or down (Status {resp.status_code}).")
        else:
            print(f"HF API Error: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"HF Request Error: {e}")
        
    # Fallback to local CPU logic
    return _generate_rule_answer(query, evidence_chunks, evidence_check)


def _generate_api_answer(query, evidence_chunks, evidence_check):
    """
    Calls the external API running on Lightning AI via the Ngrok Tunnel.
    Add your ngrok URL to environment variables or hardcode here.
    """
    import os
    import requests

    api_url = os.environ.get("LIGHTNING_API_URL", "http://REPLACE_WITH_YOUR_NGROK_URL/generate")
    if "REPLACE_WITH" in api_url:
        return _generate_rule_answer(query, evidence_chunks, evidence_check)

    context_parts = []
    for i, r in enumerate(evidence_chunks):
        c = r["chunk"]
        context_parts.append(f"[{c.clause_id}] {c.section}: {c.text}")
    
    try:
        resp = requests.post(
            api_url, 
            json={"query": query, "evidence_context": "\n".join(context_parts)},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            
            # Format UI
            evidence_list = []
            for r in evidence_chunks[:3]:
                c = r["chunk"]
                evidence_list.append({
                    "clause_id": c.clause_id,
                    "section": c.section,
                    "page": c.page,
                    "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
                })
                
            return {
                "answer": data["answer"],
                "risk_level": data["risk_level"],
                "evidence": evidence_list,
                "confidence": "HIGH",
                "decision": data["decision"],
                "action": _ACTION_TEMPLATES.get(data["risk_level"], _ACTION_TEMPLATES["MEDIUM"]),
            }
    except Exception as e:
        print(f"API Error: {e}")
        
    # If the API fails or isn't running, gracefully fallback to CPU rule logic
    return _generate_rule_answer(query, evidence_chunks, evidence_check)


def _generate_rule_answer(query, evidence_chunks, evidence_check):
    """Rule-based grounded generation — works on CPU, no model needed."""
    combined_text = " ".join(r["chunk"].text for r in evidence_chunks)
    risk_level = _detect_risk_level(combined_text)

    # Build evidence citations
    evidence_list = []
    for r in evidence_chunks[:3]:
        c = r["chunk"]
        evidence_list.append({
            "clause_id": c.clause_id,
            "section": c.section,
            "page": c.page,
            "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
        })

    # Build answer grounded in the actual text
    answer_parts = []
    q_lower = query.lower()

    for r in evidence_chunks[:2]:
        c = r["chunk"]
        text = c.text.strip()

        # Extract key sentences that are most relevant
        sentences = re.split(r"(?<=[.!?])\s+", text)
        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            # Check if sentence contains query-relevant terms
            query_terms = set(re.findall(r"\w{4,}", q_lower))
            sent_terms = set(re.findall(r"\w{4,}", sent_lower))
            if query_terms & sent_terms or any(kw in sent_lower for kw in [
                "shall", "must", "may", "will", "right", "obligation",
                "terminat", "liab", "confiden", "indemn", "warrant",
            ]):
                relevant.append(sent)

        if relevant:
            answer_parts.append(
                f"According to {c.clause_id} ({c.section}, page {c.page}): "
                + " ".join(relevant[:3])
            )
        else:
            answer_parts.append(
                f"The clause at {c.clause_id} ({c.section}, page {c.page}) states: "
                + " ".join(sentences[:2])
            )

    answer = "\n\n".join(answer_parts) if answer_parts else "Evidence found but could not extract a specific answer."

    confidence = "HIGH" if evidence_check["confidence"] > 0.6 else \
                 "MEDIUM" if evidence_check["confidence"] > 0.3 else "LOW"

    decision = "ANSWER" if evidence_check["decision"] == "SUFFICIENT" else "ESCALATE"

    if decision == "ESCALATE":
        answer += "\n\n**Note:** The evidence is partial. The above is based on available clauses, but additional contract sections may contain relevant provisions. Consider reviewing the full document."

    return {
        "answer": answer,
        "risk_level": risk_level,
        "evidence": evidence_list,
        "confidence": confidence,
        "decision": decision,
        "action": _ACTION_TEMPLATES[risk_level],
    }


def _generate_llm_answer(query, evidence_chunks, evidence_check):
    """
    LLM-based generation using DPO-aligned Mistral.
    This is called when running on GPU (Lightning AI / local).
    Falls back to rule-based if model isn't loaded.
    """
    try:
        prompt = _build_llm_prompt(query, evidence_chunks)

        from src.pipeline._model_cache import get_model_and_tokenizer
        model, tokenizer = get_model_and_tokenizer()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return _generate_rule_answer(query, evidence_chunks, evidence_check)

    except Exception:
        return _generate_rule_answer(query, evidence_chunks, evidence_check)
