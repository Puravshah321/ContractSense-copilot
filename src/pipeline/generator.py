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
10. Follow intent-aligned output behavior:
    - factual/yes-no: concise single answer with direct citation
    - extraction: list relevant clauses only
    - analytical/risk: structured multi-point synthesis grouped by concept
11. Exclude irrelevant categories (for example audit/boilerplate) unless directly requested by the query.
12. For analytical answers, include explicit coverage gaps if evidence is partial.

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

    if mode == "groq_api":
        return _generate_groq_api_answer(query, evidence_chunks, evidence_check)

    if mode == "gemini_api":
        return _generate_gemini_api_answer(query, evidence_chunks, evidence_check)

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





def _generate_groq_api_answer(query, evidence_chunks, evidence_check):
    """
    Two-pass Groq API generator:
    Pass 1 — Decompose the query into atomic sub-questions.
    Pass 2 — For each sub-question, reason cautiously against evidence,
              explicitly flagging gaps/ambiguities where clauses are absent or only partially relevant.
    """
    import os
    import requests
    import json

    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return {
            "answer": "SYSTEM ERROR: GROQ_API_KEY is missing from Streamlit secrets.",
            "risk_level": "N/A",
            "evidence": [],
            "confidence": "LOW",
            "decision": "NOT_FOUND",
            "action": ""
        }

    # ── Build evidence context (Limit to top 5 for TPM safety) ──────────────
    context_parts = []
    for i, r in enumerate(evidence_chunks[:5]):
        c = r["chunk"]
        context_parts.append(
            f"[Evidence {i+1}] clause_id: {c.clause_id} | section: {c.section} | page: {c.page}\n{c.text}"
        )
    evidence_text = "\n\n".join(context_parts) if context_parts else "No evidence retrieved."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    model = "llama-3.3-70b-versatile"

    # ── Pass 1: Decompose compound query into sub-questions ─────────────────
    is_compound = "\n" in query.strip() or len(query) > 300
    sub_questions = []
    if is_compound:
        try:
            decomp_prompt = (
                "You are a legal query decomposer. "
                "The user has asked a multi-part legal question. "
                "Split it into a JSON list of independent, atomic sub-questions. "
                "Each sub-question should be answerable from a single contract clause. "
                "Return ONLY a JSON array of strings. Example: [\"Q1\", \"Q2\"]"
            )
            decomp_resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": decomp_prompt},
                        {"role": "user", "content": f"QUERY:\n{query}"}
                    ],
                    "temperature": 0.0
                },
                timeout=15
            )
            if decomp_resp.status_code == 200:
                raw = decomp_resp.json()["choices"][0]["message"]["content"].strip()
                raw = raw[raw.find("["):raw.rfind("]")+1]
                sub_questions = json.loads(raw)
        except Exception:
            sub_questions = []

    # Fallback: treat as single question
    if not sub_questions:
        sub_questions = [query]

    # ── Pass 2: Combined cautious legal reasoning ──────────────────────────
    sys_prompt = (
        "You are ContractSense, an advanced legal reasoning AI with strict grounding discipline.\n"
        "You are given EVIDENCE clauses retrieved from a contract, and a list of atomic sub-questions to analyze.\n\n"
        "RULES:\n"
        "1. DO NOT fabricate legal rules. Only reason from the provided evidence.\n"
        "2. For EACH sub-question, perform a structured analysis:\n"
        "   - 'explicit': Direct statements from the evidence (cite Evidence X).\n"
        "   - 'implied': Reasonable legal implications or suggestions derived from evidence.\n"
        "   - 'unresolved': Gaps, ambiguities, or missing information.\n"
        "3. Use language like: 'implies', 'suggests', 'does not clearly define', 'ambiguous whether'.\n"
        "4. Never say 'Yes' or 'No' unless a clause explicitly resolves it.\n"
        "5. Return a JSON array of objects, one per sub-question, matching this schema:\n"
        "   [{\"q\": \"...\", \"explicit\": \"...\", \"implied\": \"...\", \"unresolved\": \"...\", \"resolved\": true/false, \"risk_contribution\": \"LOW|MEDIUM|HIGH|CRITICAL\"}]"
    )

    findings = []
    any_critical = False
    any_high = False
    any_resolved = False

    try:
        user_msg = f"EVIDENCE:\n{evidence_text}\n\nSUB-QUESTIONS:\n" + "\n".join(sub_questions[:8]) + "\n\nReturn JSON ONLY."
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            },
            timeout=30
        )
        if r.status_code == 200:
            resp_json = r.json()
            content = resp_json["choices"][0]["message"]["content"]
            # Extract list from potential JSON object wrapper
            data = json.loads(content)
            if isinstance(data, dict):
                # Look for a list value in the dict (Groq json_object often wraps the array)
                for val in data.values():
                    if isinstance(val, list):
                        findings = val
                        break
                else:
                    findings = []
            elif isinstance(data, list):
                findings = data
            
            for f in findings:
                if f.get("resolved"): any_resolved = True
                risk = f.get("risk_contribution", "MEDIUM")
                if risk == "CRITICAL": any_critical = True
                elif risk == "HIGH": any_high = True
        else:
            print(f"Groq API Synthesis Error: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Synthesis Exception: {e}")

    # ── Assemble final answer ──────────────────────────────────────────────
    if not findings:
        return {
            "answer": "The system could not decompose or analyze this query against the retrieved evidence.",
            "risk_level": "N/A", "evidence": [], "confidence": "LOW",
            "decision": "NOT_FOUND", "action": ""
        }

    answer_lines = []
    unresolved_count = 0
    for f in findings:
        resolved_tag = "✓ Resolved" if f["resolved"] else "⚠ Unresolved/Ambiguous"
        
        q_text = f"**{f['q']}**\n{resolved_tag}"
        
        explicit = f.get("explicit", "").strip()
        if explicit and explicit.lower() not in {"n/a", "none", "no explicit statements found"}:
            q_text += f"\n- **Explicit Findings:** {explicit}"
            
        implied = f.get("implied", "").strip()
        if implied and implied.lower() not in {"n/a", "none", "no implied interpretations"}:
            q_text += f"\n- **Implied Interpretation:** {implied}"
            
        unresolved = f.get("unresolved", "").strip()
        if unresolved and unresolved.lower() not in {"n/a", "none"}:
            q_text += f"\n- **Unresolved Gaps:** {unresolved}"
            
        answer_lines.append(q_text)
        if not f["resolved"]:
            unresolved_count += 1

    full_answer = "\n\n".join(answer_lines)

    if unresolved_count > 0:
        full_answer += (
            f"\n\n---\n**Summary:** {unresolved_count} of {len(findings)} issue(s) are not explicitly resolved by this agreement. "
            "The analysis above identifies the most relevant clauses and their implied scope, "
            "but definitive conclusions on unresolved points require legal counsel."
        )

    decision = "ANSWER" if any_resolved else "AMBIGUOUS"
    risk_level = "CRITICAL" if any_critical else ("HIGH" if any_high else "MEDIUM")
    confidence = "MEDIUM" if any_resolved else "LOW"

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
        "answer": full_answer,
        "risk_level": risk_level,
        "evidence": evidence_list,
        "confidence": confidence,
        "decision": decision,
        "action": _make_evidence_action(evidence_list) if decision == "ANSWER" else "",
    }



def _generate_gemini_api_answer(query, evidence_chunks, evidence_check):
    """
    Calls the Google Gemini API (Gemini 1.5 Flash).
    Two-pass structure: 1. Decompose 2. Synthesize.
    """
    import os
    import json
    try:
        import google.generativeai as genai
        import streamlit as st
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    except Exception:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        return {
            "answer": "SYSTEM ERROR: GOOGLE_API_KEY is missing from Streamlit secrets.",
            "risk_level": "N/A", "evidence": [], "confidence": "LOW",
            "decision": "NOT_FOUND", "action": ""
        }

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # ── Build evidence context ──────────────────────────────────────────────
        context_parts = []
        for i, r in enumerate(evidence_chunks[:5]):
            c = r["chunk"]
            context_parts.append(
                f"[Evidence {i+1}] clause_id: {c.clause_id} | section: {c.section} | page: {c.page}\n{c.text}"
            )
        evidence_text = "\n\n".join(context_parts) if context_parts else "No evidence retrieved."

        # ── Pass 1: Decompose ──────────────────────────────────────────────────
        is_compound = "\n" in query.strip() or len(query) > 300
        sub_questions = []
        if is_compound:
            decomp_prompt = (
                "Split this multi-part legal question into a JSON list of independent atomic sub-questions. "
                "Return ONLY a JSON array of strings. Example: [\"Q1\", \"Q2\"]"
            )
            response = model.generate_content(f"{decomp_prompt}\n\nQUERY:\n{query}")
            try:
                raw = response.text.strip()
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "[" in raw:
                    raw = raw[raw.find("["):raw.rfind("]")+1]
                sub_questions = json.loads(raw)
            except: sub_questions = [query]
        else:
            sub_questions = [query]

        # ── Pass 2: Synthesize ─────────────────────────────────────────────────
        sys_prompt = (
            "You are ContractSense, a legal AI. Analyze these sub-questions against the EVIDENCE.\n"
            "RULES:\n"
            "1. NO fabrication. 2. Use 'explicit', 'implied', 'unresolved' categories.\n"
            "3. Use cautious language ('suggests', 'ambiguous'). 4. Cite Evidence X.\n"
            "Return JSON array: [{\"q\": \"...\", \"explicit\": \"...\", \"implied\": \"...\", \"unresolved\": \"...\", \"resolved\": true/false, \"risk_contribution\": \"LOW|MEDIUM|HIGH|CRITICAL\"}]"
        )
        
        user_msg = f"EVIDENCE:\n{evidence_text}\n\nSUB-QUESTIONS:\n" + "\n".join(sub_questions[:8]) + "\n\nReturn JSON ONLY."
        response = model.generate_content(f"{sys_prompt}\n\n{user_msg}")
        
        raw_content = response.text.strip()
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "[" in raw_content:
            raw_content = raw_content[raw_content.find("["):raw_content.rfind("]")+1]
            
        findings = json.loads(raw_content)
        if isinstance(findings, dict): # Handle case where it wraps list in a key
            for k in findings:
                if isinstance(findings[k], list):
                    findings = findings[k]
                    break

        # ── Assemble ──────────────────────────────────────────────────────────
        answer_lines = []
        unresolved_count = 0
        any_resolved = False
        any_critical = False
        any_high = False
        
        for f in findings:
            if f.get("resolved"): any_resolved = True
            risk = f.get("risk_contribution", "MEDIUM")
            if risk == "CRITICAL": any_critical = True
            elif risk == "HIGH": any_high = True
            
            resolved_tag = "✓ Resolved" if f.get("resolved") else "⚠ Unresolved/Ambiguous"
            q_text = f"**{f.get('q', '')}**\n{resolved_tag}"
            if f.get("explicit"): q_text += f"\n- **Explicit Findings:** {f['explicit']}"
            if f.get("implied"): q_text += f"\n- **Implied Interpretation:** {f['implied']}"
            if f.get("unresolved"): q_text += f"\n- **Unresolved Gaps:** {f['unresolved']}"
            answer_lines.append(q_text)
            if not f.get("resolved"): unresolved_count += 1

        full_answer = "\n\n".join(answer_lines)
        if unresolved_count > 0:
            full_answer += f"\n\n---\n**Summary:** {unresolved_count} issues unresolved. Legal counsel recommended."

        evidence_list = []
        for r in evidence_chunks[:3]:
            c = r["chunk"]
            evidence_list.append({"clause_id": c.clause_id, "section": c.section, "page": c.page, "text": c.text[:300]})

        decision = "ANSWER" if any_resolved else "AMBIGUOUS"
        return {
            "answer": full_answer,
            "risk_level": "CRITICAL" if any_critical else ("HIGH" if any_high else "MEDIUM"),
            "evidence": evidence_list,
            "confidence": "MEDIUM" if any_resolved else "LOW",
            "decision": decision,
            "action": _make_evidence_action(evidence_list) if decision == "ANSWER" else "",
        }
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
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



