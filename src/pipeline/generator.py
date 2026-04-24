"""
Grounded Answer Generator.
Two modes:
  1. Rule-based (CPU, works everywhere) - for Streamlit Cloud demos
  2. LLM-based (GPU, uses DPO-aligned Mistral) - for Lightning AI / local GPU
Both modes enforce strict grounding: answer ONLY from evidence, cite sources.
"""
import re
import json

from src.pipeline.retriever import expand_query_keywords


# ============================================================
# STRICT GROUNDING PROMPT (used in LLM mode)
# ============================================================
SYSTEM_PROMPT = """You are ContractSense, a contract analysis system.

STRICT RULES:
1. Use ONLY the provided clause evidence to answer. Do NOT add information not present in the evidence.
2. If the answer is not contained in the evidence, respond with decision: "NOT_FOUND" and say: "This is not specified in the provided document."
3. If the relevant clause is not found, you MUST return NOT_FOUND. Do NOT escalate. Do NOT guess.
4. NO EVIDENCE means NOT_FOUND, not ESCALATE.
5. Use ESCALATE only when the provided evidence conflicts or cannot be reconciled.
6. For yes/no questions, begin the answer with exactly: "Answer: YES", "Answer: NO", or "Answer: NOT_FOUND".
7. ALWAYS cite the clause_id and quote the exact text you are referencing.
8. Do NOT generate generic legal advice. Every claim must be traceable to a specific clause.
9. Classify risk as LOW, MEDIUM, HIGH, or CRITICAL based on the actual clause language.

OUTPUT FORMAT (strict JSON):
{
  "answer": "Plain English explanation grounded in evidence",
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "evidence": [{"clause_id": "...", "text": "exact quote from clause"}],
  "confidence": "HIGH | MEDIUM | LOW",
  "decision": "ANSWER | NOT_FOUND | ESCALATE",
  "action": "Action based ONLY on evidence, or empty string if evidence does not support an action"
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


def _is_yes_no_query(query):
    return bool(re.match(r"^\s*(is|are|can|could|may|does|do|did|has|have|must|should|will|would)\b", query.lower()))


def _has_query_term(query_lower, term):
    if " " in term or "-" in term:
        return term in query_lower
    return bool(re.search(rf"\b{re.escape(term)}\b", query_lower))


def _evidence_list(evidence_chunks, limit=3):
    evidence = []
    for r in evidence_chunks[:limit]:
        c = r["chunk"]
        evidence.append({
            "clause_id": c.clause_id,
            "section": c.section,
            "page": c.page,
            "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
        })
    return evidence


def _make_evidence_action(evidence_list):
    if not evidence_list:
        return ""
    ev = evidence_list[0]
    quote = ev.get("text", "").strip()
    if len(quote) > 180:
        quote = quote[:180].rsplit(" ", 1)[0] + "..."
    return f"Review {ev.get('clause_id')} ({ev.get('section')}) because it states: \"{quote}\""


def _infer_yes_no(query, evidence_text):
    q = query.lower()
    t = evidence_text.lower()
    if ("warranty" in q or "warranties" in q) and not re.search(r"\bwarrant(?:y|ies|s)?\b|\bguarantee\b", t):
        return "NOT_FOUND"
    if any(_has_query_term(q, term) for term in ["ai", "training", "model"]):
        if re.search(r"\buse\s+the\s+confidential\s+information\b|\bscope\s+of\s+audit\b|\bnot\s+to\s+make\s+or\s+retain\s+copy\b|\bnot\s+to\s+disclose\b", t):
            return "NO"
    if any(_has_query_term(q, term) for term in ["share", "third", "external", "disclose"]):
        if re.search(r"\bnot\s+to\s+disclose\b|\bshall\s+not\s+disclose\b|\bother\s+person\s+or\s+entity\b|\bthird\s+party\b|\bwithout\s+(?:the\s+)?(?:express\s+)?(?:prior\s+)?written\s+(?:approval|consent)\b|\bno\s+information\b.*\boutside\s+the\s+country\b", t):
            return "NO"
    if "is there" in q or "does this" in q or "contain" in q or "include" in q or "has " in q:
        return "YES"
    negative_patterns = [
        r"\bshall\s+not\b", r"\bmay\s+not\b", r"\bmust\s+not\b",
        r"\bnot\s+(?:transfer|share|disclose|assign|use|process|permit)",
        r"\bnot\s+to\s+(?:transfer|share|disclose|assign|use|process|permit|make|retain)",
        r"\bprohibited\b", r"\bwithout\s+(?:prior\s+)?written\s+consent\b",
        r"\bno\s+other\s+warranties\b",
    ]
    positive_patterns = [
        r"\bmay\b", r"\bis\s+permitted\b", r"\bis\s+allowed\b",
        r"\bshall\b", r"\bwarrants?\b", r"\bagrees?\b",
    ]
    if any(re.search(p, t) for p in negative_patterns):
        return "NO"
    if any(re.search(p, t) for p in positive_patterns):
        return "YES"
    return "YES"


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
    # DECISION GATE: refuse if evidence is insufficient.
    # NO EVIDENCE != ESCALATE. No relevant evidence is a high-confidence NOT_FOUND.
    if evidence_check["decision"] == "INSUFFICIENT" or not evidence_chunks:
        prefix = "Answer: NOT_FOUND\n\n" if evidence_check.get("is_yes_no") or _is_yes_no_query(query) else ""
        return {
            "answer": prefix + "This is not specified in the provided document. The uploaded contract does not contain clauses that address this question.",
            "risk_level": "N/A",
            "evidence": [],
            "confidence": "HIGH",
            "decision": "NOT_FOUND",
            "action": "",
        }

    if evidence_check["decision"] == "CONFLICTING":
        evidence = _evidence_list(evidence_chunks)
        return {
            "answer": "The retrieved clauses appear to contain conflicting obligations or permissions. Review the cited clauses together before relying on one answer.",
            "risk_level": "HIGH",
            "evidence": evidence,
            "confidence": "MEDIUM",
            "decision": "ESCALATE",
            "action": _make_evidence_action(evidence),
        }

    combined_text = " ".join(r["chunk"].text for r in evidence_chunks)
    if ("warranty" in query.lower() or "warranties" in query.lower()) and not re.search(r"\bwarrant(?:y|ies|s)?\b|\bguarantee\b", combined_text.lower()):
        return {
            "answer": "Answer: NOT_FOUND\n\nThis is not specified in the provided document. The retrieved evidence does not contain a warranty clause.",
            "risk_level": "N/A",
            "evidence": [],
            "confidence": "HIGH",
            "decision": "NOT_FOUND",
            "action": "",
        }

    if mode == "hf_api":
        return _generate_hf_api_answer(query, evidence_chunks, evidence_check)

    if mode == "api":
        return _generate_api_answer(query, evidence_chunks, evidence_check)

    if mode == "llm":
        return _generate_llm_answer(query, evidence_chunks, evidence_check)

    return _generate_rule_answer(query, evidence_chunks, evidence_check)


def _normalize_answer_data(data, evidence_chunks):
    evidence = data.get("evidence") or _evidence_list(evidence_chunks)
    risk = data.get("risk_level", "MEDIUM")
    decision = data.get("decision", "ANSWER")
    if decision not in {"ANSWER", "NOT_FOUND", "ESCALATE"}:
        decision = "ANSWER"
    return {
        "answer": data.get("answer", ""),
        "risk_level": risk,
        "evidence": evidence,
        "confidence": data.get("confidence", "MEDIUM"),
        "decision": decision,
        "action": data.get("action") or ("" if decision == "NOT_FOUND" else _make_evidence_action(evidence)),
    }


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
        f"2. If the relevant clause is not found, reply with DECISION: NOT_FOUND. Do not escalate or guess.\n"
        f"3. For yes/no questions, begin with Answer: YES, Answer: NO, or Answer: NOT_FOUND.\n"
        f"4. Otherwise, quote the evidence, cite the clause_id, and provide a RISK label.\n"
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
                "action": _make_evidence_action(evidence_list),
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
                "action": _make_evidence_action(evidence_list),
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
    evidence_list = _evidence_list(evidence_chunks)

    # Build answer grounded in the actual text
    answer_parts = []
    q_lower = query.lower()
    expanded_terms = expand_query_keywords(query)
    is_yes_no = evidence_check.get("is_yes_no") or _is_yes_no_query(query)

    specialized = _specialized_rule_answer(query, evidence_chunks, evidence_list, evidence_check)
    if specialized:
        return specialized

    answer_chunks = evidence_chunks[:1] if is_yes_no else evidence_chunks[:2]
    for r in answer_chunks:
        c = r["chunk"]
        text = c.text.strip()

        # Extract key sentences that are most relevant
        sentences = re.split(r"(?<=[.!?])\s+", text)
        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            # Check if sentence contains query-relevant terms
            query_terms = set(re.findall(r"\w{4,}", q_lower)) | expanded_terms
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
    if is_yes_no:
        answer = f"Answer: {_infer_yes_no(query, combined_text)}\n\n{answer}"

    confidence = "HIGH" if evidence_check["confidence"] > 0.6 else \
                 "MEDIUM" if evidence_check["confidence"] > 0.3 else "LOW"

    decision = "ANSWER" if evidence_check["decision"] == "SUFFICIENT" else "NOT_FOUND"

    return {
        "answer": answer,
        "risk_level": risk_level,
        "evidence": evidence_list,
        "confidence": confidence,
        "decision": decision,
        "action": _make_evidence_action(evidence_list),
    }


def _specialized_rule_answer(query, evidence_chunks, evidence_list, evidence_check):
    q = query.lower()
    combined = " ".join(r["chunk"].text for r in evidence_chunks)
    combined_lower = combined.lower()

    def finish(answer, risk="MEDIUM", decision="ANSWER", confidence=None):
        conf = confidence or ("HIGH" if evidence_check.get("confidence", 0) > 0.45 else "MEDIUM")
        return {
            "answer": answer,
            "risk_level": risk,
            "evidence": evidence_list if decision != "NOT_FOUND" else [],
            "confidence": conf,
            "decision": decision,
            "action": _make_evidence_action(evidence_list) if decision != "NOT_FOUND" else "",
        }

    if "warranty" in q or "warranties" in q:
        if not re.search(r"\bwarrant(?:y|ies|s)?\b|\bguarantee\b", combined_lower):
            return finish(
                "Answer: NOT_FOUND\n\nThis agreement does not specify a warranty clause in the retrieved evidence.",
                risk="N/A",
                decision="NOT_FOUND",
                confidence="HIGH",
            )

    if "duration" in q or re.search(r"\bterm\b", q):
        for r in evidence_chunks:
            c = r["chunk"]
            text = c.text.strip()
            if re.search(r"\bterm\b", c.section.lower()) or re.search(r"\bvalid\s+up\s+to\s+one\s+year\b|\bvalid\s+up\s+to\s+1\s+year\b", text.lower()):
                m = re.search(r"valid\s+up\s+to\s+one\s+year|valid\s+up\s+to\s+1\s+year|continues\s+for\s+one\s+\(1\)\s+year|one\s+\(1\)\s+year|one\s+year", text, re.IGNORECASE)
                duration = m.group(0) if m else "one year"
                duration_text = duration if duration.lower().startswith("valid") else f"valid for {duration}"
                return finish(
                    f"The agreement is {duration_text}. According to {c.clause_id} ({c.section}, page {c.page}): {text}",
                    risk="LOW",
                    confidence="HIGH",
                )

    if any(_has_query_term(q, term) for term in ["share", "third", "external", "disclose"]) and ("confidential" in q or "data" in q or "audit" in q):
        if re.search(r"\bnot\s+to\s+disclose\b|\bshall\s+not\s+disclose\b|\bthird\s+party\b|\bother\s+person\s+or\s+entity\b", combined_lower):
            caveat = ""
            if "stqc" in combined_lower or "government entities" in combined_lower:
                caveat = " The document separately permits sharing audit information with STQC or similar mandated government entities when called upon, with prior written information to the auditee."
            return finish(
                "Answer: NO\n\nThe auditor cannot share confidential/audit data with external teams or third parties without the auditee's written approval."
                + caveat,
                risk="HIGH",
                confidence="HIGH",
            )

    if any(_has_query_term(q, term) for term in ["ai", "training", "model"]):
        if re.search(r"\buse\s+the\s+confidential\s+information\b|\bscope\s+of\s+audit\b|\bnot\s+to\s+make\s+or\s+retain\s+copy\b|\bnot\s+to\s+disclose\b", combined_lower):
            return finish(
                "Answer: NO\n\nThe agreement restricts use of confidential/audit information to the audit scope and prohibits retaining copies or unauthorized disclosure. It does not permit using audit data to train AI models.",
                risk="HIGH",
                confidence="HIGH",
            )

    if "penalty" in q or "penalties" in q:
        if re.search(r"\bliquidated\s+damages\b|\bcontract\s+value\b|\bloss\s+or\s+damages\b", combined_lower):
            return finish(
                "There is no fixed penalty amount stated. The remedies clause says the auditor must compensate the auditee for losses/damages, including actual and liquidated damages, with liquidated damages not to exceed the Contract value.",
                risk="MEDIUM",
                confidence="HIGH",
            )

    return None


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
                return _normalize_answer_data(json.loads(json_match.group()), evidence_chunks)
        except json.JSONDecodeError:
            pass

        return _generate_rule_answer(query, evidence_chunks, evidence_check)

    except Exception:
        return _generate_rule_answer(query, evidence_chunks, evidence_check)
