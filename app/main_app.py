"""
ContractSense Legal Copilot — Professional UI
Upload → Auto risk scan → Structured Dispute Analysis Report
"""
import streamlit as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.orchestrator import ContractSensePipeline


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ContractSense — Legal Copilot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (no raw HTML injected into chat messages) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif !important; }

.stApp { background: linear-gradient(135deg, #080B14 0%, #0D1117 60%, #080B14 100%); }

[data-testid="stSidebar"] { background: #0A0E1A; border-right: 1px solid #1C2333; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Force Sidebar Toggle to always be visible and white */
[data-testid="collapsedControl"] { 
    visibility: visible !important; 
    color: #E2E8F0 !important; 
    background: rgba(13, 17, 23, 0.7) !important;
    border-radius: 8px !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Logo */
.cs-logo {
    font-size: 1.5rem; font-weight: 800;
    background: linear-gradient(135deg, #4F8EF7, #A78BFA 60%, #F472B6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}
.cs-tagline {
    font-size: 0.72rem; color: #374151; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 16px;
}

/* Doc card */
.doc-card {
    background: #131929; border: 1px solid #1C2333;
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
}
.doc-card .doc-name { font-size: 0.82rem; font-weight: 600; color: #E2E8F0; }
.doc-card .doc-meta { font-size: 0.72rem; color: #4B5563; margin-top: 3px; }

/* Welcome */
.welcome-card {
    background: #0D1117; border: 1px solid #1C2333; border-radius: 20px;
    padding: 52px 60px; text-align: center; max-width: 620px;
    margin: 80px auto;
}
.welcome-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #4F8EF7, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.welcome-sub { color: #4B5563; font-size: 0.9rem; line-height: 1.7; margin-bottom: 28px; }
.welcome-step {
    background: #131929; border: 1px solid #1C2333; border-radius: 10px;
    padding: 10px 16px; margin: 8px 0; text-align: left; color: #94A3B8; font-size: 0.85rem;
}

/* Chat input */
[data-testid="stChatInput"] { padding-bottom: 20px; background: transparent !important; }
[data-testid="stChatInput"] > div { background: transparent !important; border: none !important; }
[data-testid="stChatInput"] textarea {
    background: #131929!important; 
    border: 1px solid #2D3748!important;
    color: #E2E8F0!important; 
    border-radius: 16px!important;
    padding: 14px 20px!important;
    font-size: 1rem!important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2)!important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease!important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #4F8EF7!important;
    box-shadow: 0 0 0 2px rgba(79, 142, 247, 0.2)!important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #1C2333; border-radius: 4px; }

/* Sidebar buttons */
[data-testid="stSidebar"] button {
    background: #131929!important; border: 1px solid #1C2333!important;
    color: #94A3B8!important; border-radius: 8px!important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("pipeline", None), ("messages", []), ("pdf_name", None),
    ("doc_loaded", False), ("chunk_count", 0),
]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.pipeline is None:
    st.session_state.pipeline = ContractSensePipeline()


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file):
    try:
        import PyPDF2
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n\n".join(p.extract_text() or "" for p in reader.pages).strip() or None
    except Exception as e:
        return f"ERROR: {e}"


def _risk_icon(risk):
    return {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(
        (risk or "").upper(), "⚫"
    )


def _conf_color(conf):
    return {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "⚫"}.get((conf or "").upper(), "⚫")


import re

def _render_styled_clause(text, query=""):
    """Render full clause text in a scrollable, styled panel with basic highlighting."""
    # Basic highlighting for common terms if they appear
    terms_to_highlight = ["malware", "harmful code", "malicious code", "liability", "indemnify", "survive", "terminate", "confidential"]
    html_text = text.replace("<", "&lt;").replace(">", "&gt;")
    
    # Simple naive highlighting for visual demo purposes
    for term in terms_to_highlight:
        # Case insensitive replace using regex
        pattern = re.compile(f"({term})", re.IGNORECASE)
        html_text = pattern.sub(r"<span style='background-color:#4F8EF733; color:#8DB3F9; padding:0 2px; border-radius:3px; font-weight:600;'>\1</span>", html_text)
        
    return f"<div style='background:#131929; padding:14px; border-radius:8px; border:1px solid #1C2333; font-family:\"JetBrains Mono\", monospace; font-size:0.82rem; color:#A0AEC0; white-space:pre-wrap; max-height:400px; overflow-y:auto; line-height:1.5;'>{html_text}</div>"


def render_assistant_message(msg: dict):
    """
    Render a stored assistant message using native Streamlit components.
    """
    if msg.get("is_risk_scan"):
        risk_groups = msg.get("risk_groups", {})
        total = sum(len(v) for v in (risk_groups or {}).values())
        if not risk_groups or total == 0:
            st.markdown(f"I've analyzed **{st.session_state.pdf_name}** ({st.session_state.chunk_count} sections). No specific risk clauses identified. Ask me anything about the contract.")
            return

        st.markdown(f"I've analyzed **{st.session_state.pdf_name}** ({st.session_state.chunk_count} sections). Found **{total} distinct risk clause(s)**:")
        RISK_ICONS = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            items = risk_groups.get(level, [])
            if not items: continue
            st.markdown(f"### {RISK_ICONS[level]} {level} Risk — {len(items)} clause{'s' if len(items) > 1 else ''}")
            for item in items:
                sec  = item.get("section", "General")
                cat  = item.get("category", "")
                with st.expander(f"📎 {cat} · {sec}"):
                    if item.get("legal_tags"):
                        st.markdown(" ".join(f"`{t}`" for t in item.get("legal_tags")[:4]))
                    st.markdown(_render_styled_clause(item.get("text", "")), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("Ask me anything about specific clauses, disputes, risks, or obligations.")
        return


    risk   = msg.get("risk", "N/A")
    dec    = msg.get("decision", "")
    conf   = msg.get("confidence", "LOW")
    lat    = msg.get("latency", 0)

    # ── Meta row ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
    with col1:
        st.markdown(f"{_risk_icon(risk)} **Risk:** `{risk}`")
    with col2:
        st.markdown(f"**Decision:** `{dec}`")
    with col3:
        st.markdown(f"{_conf_color(conf)} **Confidence:** `{conf}`")
    with col4:
        st.markdown(f"<span style='color:#374151;font-size:0.75rem;font-family:monospace'>{lat:.0f}ms</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Answer body (pure markdown — safe) ────────────────────────────────
    st.markdown(msg.get("answer_md", ""), unsafe_allow_html=False)

    # ── Evidence expander ─────────────────────────────────────────────────
    evidence = msg.get("evidence", [])
    if evidence:
        st.markdown("<br/>", unsafe_allow_html=True)
        for ev in evidence:
            cid  = ev.get("clause_id", "?")
            sec  = ev.get("section", "General")
            pg   = ev.get("page", "?")
            txt  = ev.get("text", "")
            tags = ev.get("legal_tags", [])

            with st.expander(f"📎 {cid} · {sec} · p.{pg}", expanded=False):
                if tags:
                    st.markdown(" ".join(f"`{t}`" for t in tags[:5]))
                st.markdown(_render_styled_clause(txt), unsafe_allow_html=True)

    # ── Grounding bar ─────────────────────────────────────────────────────
    v = msg.get("verification") or {}
    verdict = v.get("verdict", "")
    ratio   = v.get("supported_ratio", 0.0)
    if verdict:
        pct   = int(ratio * 100)
        label = {"VERIFIED": "✅ Verified", "PARTIALLY_VERIFIED": "🟡 Partial", "REJECTED": "❌ Rejected"}.get(verdict, verdict)
        st.progress(ratio, text=f"Grounding: {label} ({pct}% supported)")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="cs-logo">⚖️ ContractSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-tagline">Grounded Legal Copilot</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Upload Contract PDF**")
    uploaded = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded and uploaded.name != st.session_state.pdf_name:
        st.session_state.pdf_name  = uploaded.name
        st.session_state.messages  = []
        text = extract_pdf_text(uploaded)

        if text and not str(text).startswith("ERROR:"):
            pipeline = ContractSensePipeline()
            count    = pipeline.load_document(text, uploaded.name)
            st.session_state.pipeline    = pipeline
            st.session_state.doc_loaded  = True
            st.session_state.chunk_count = count

            with st.spinner("Scanning for risks…"):
                risk_groups = pipeline.get_all_risks()

            st.session_state.messages.append({
                "role":     "assistant",
                "answer_md": "",
                "is_risk_scan": True,
                "risk_groups": risk_groups,
                "risk": "N/A", "decision": "SCAN", "confidence": "N/A", "latency": 0,
                "evidence": [], "verification": None,
            })
            st.rerun()
        else:
            err = text if str(text).startswith("ERROR:") else "Could not extract text (scanned/image PDF)."
            st.session_state.messages.append({
                "role": "assistant", "answer_md": err,
                "risk": "N/A", "decision": "ERROR", "confidence": "LOW", "latency": 0,
                "evidence": [], "verification": None,
            })
            st.rerun()

    # Doc status card
    if st.session_state.pdf_name:
        st.markdown(
            f'<div class="doc-card">'
            f'<div class="doc-name">📄 {st.session_state.pdf_name}</div>'
            f'<div class="doc-meta">{st.session_state.chunk_count} chunks · indexed ✓</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div style="color:#374151;font-size:0.8rem;">No document uploaded</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    show_trace = st.checkbox("Show pipeline trace", value=False)

    if st.button("🔄 New Analysis", use_container_width=True):
        st.session_state.pipeline    = ContractSensePipeline()
        st.session_state.messages    = []
        st.session_state.pdf_name    = None
        st.session_state.doc_loaded  = False
        st.session_state.chunk_count = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem;color:#1F2937;line-height:1.6;">'
        '"This system prioritizes grounded legal reasoning and uncertainty awareness over generative fluency."'
        '</div>',
        unsafe_allow_html=True,
    )


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
      <div class="welcome-title">⚖️ ContractSense</div>
      <div class="welcome-sub">
        Grounded explainable legal reasoning under uncertainty.<br/>
        Every answer is traceable to specific contract clauses.
      </div>
      <div class="welcome-step">① Upload a contract PDF in the sidebar</div>
      <div class="welcome-step">② Get an instant risk audit with evidence citations</div>
      <div class="welcome-step">③ Ask dispute, clause, or negotiation questions</div>
    </div>
    """, unsafe_allow_html=True)

else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg)
                if show_trace and msg.get("trace"):
                    with st.expander("🔍 Pipeline Trace", expanded=False):
                        st.code("\n".join(msg["trace"]), language=None)
            else:
                st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your contract — disputes, clauses, risks, obligations…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.doc_loaded:
            with st.spinner("Running grounded analysis…"):
                result = st.session_state.pipeline.query(prompt)

            msg_data = {
                "role":        "assistant",
                "answer_md":   result.answer if isinstance(result.answer, str) else "\n".join(str(x) for x in result.answer),
                "risk":        result.risk_level or "N/A",
                "decision":    result.decision or "N/A",
                "confidence":  result.confidence or "LOW",
                "latency":     result.latency_ms,
                "evidence":    result.evidence or [],
                "verification": result.verification or {},
                "trace":       result.pipeline_trace,
            }
            st.session_state.messages.append(msg_data)
            render_assistant_message(msg_data)

            if show_trace and result.pipeline_trace:
                with st.expander("🔍 Pipeline Trace", expanded=False):
                    st.code("\n".join(result.pipeline_trace), language=None)
        else:
            msg = "Please upload a contract PDF first using the sidebar."
            st.markdown(msg)
            st.session_state.messages.append({
                "role": "assistant", "answer_md": msg,
                "risk": "N/A", "decision": "N/A", "confidence": "LOW", "latency": 0,
                "evidence": [], "verification": None,
            })
