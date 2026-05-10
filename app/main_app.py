"""
ContractSense Legal Copilot - Professional Legal Analysis UI
Upload PDF → Auto risk scan → Dispute analysis with legal memo formatting.
"""
import streamlit as st
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.orchestrator import ContractSensePipeline


def _build_pipeline():
    return ContractSensePipeline()


st.set_page_config(
    page_title="ContractSense — Legal Copilot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #080B14 0%, #0D1117 60%, #080B14 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #1C2333;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar Branding ── */
.cs-logo {
    font-size: 1.5rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #4F8EF7, #A78BFA 60%, #F472B6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}
.cs-tagline {
    font-size: 0.72rem; font-weight: 400; color: #4B5563;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 16px;
}

/* ── Document Status Card ── */
.doc-card {
    background: #131929; border: 1px solid #1C2333;
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
}
.doc-card .doc-name {
    font-size: 0.82rem; font-weight: 600; color: #E2E8F0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.doc-card .doc-meta { font-size: 0.72rem; color: #4B5563; margin-top: 3px; }

/* ── Grounding Meter ── */
.grounding-wrap { margin: 8px 0; }
.grounding-label { font-size: 0.72rem; color: #6B7280; margin-bottom: 4px; }
.grounding-bar-bg {
    background: #1C2333; border-radius: 6px; height: 6px; overflow: hidden;
}
.grounding-bar-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #EF4444, #F59E0B, #22C55E);
    transition: width 0.5s ease;
}

/* ── Risk Badges ── */
.risk-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase;
}
.risk-critical { background: rgba(220,38,38,0.15); color: #FCA5A5; border: 1px solid rgba(220,38,38,0.3); }
.risk-high     { background: rgba(239,68,68,0.12);  color: #FDA4AF; border: 1px solid rgba(239,68,68,0.25); }
.risk-medium   { background: rgba(245,158,11,0.12); color: #FCD34D; border: 1px solid rgba(245,158,11,0.25); }
.risk-low      { background: rgba(34,197,94,0.12);  color: #86EFAC; border: 1px solid rgba(34,197,94,0.25); }
.risk-na       { background: rgba(107,114,128,0.12); color: #9CA3AF; border: 1px solid rgba(107,114,128,0.2); }

/* ── Decision Tags ── */
.decision-tag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 9px; border-radius: 6px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.5px;
}
.tag-answer    { background: rgba(34,197,94,0.12);   color: #86EFAC; }
.tag-ambiguous { background: rgba(99,102,241,0.12);  color: #A5B4FC; }
.tag-notfound  { background: rgba(107,114,128,0.12); color: #9CA3AF; }
.tag-escalate  { background: rgba(245,158,11,0.12);  color: #FCD34D; }

/* ── Confidence Chips ── */
.conf-high   { color: #86EFAC; font-size: 0.72rem; font-weight: 600; }
.conf-medium { color: #FCD34D; font-size: 0.72rem; font-weight: 600; }
.conf-low    { color: #9CA3AF; font-size: 0.72rem; font-weight: 600; }

/* ── Evidence Cards ── */
.ev-card {
    background: #0D1117; border: 1px solid #1C2333;
    border-left: 3px solid #4F8EF7; border-radius: 0 8px 8px 0;
    padding: 10px 14px; margin: 6px 0; font-size: 0.8rem;
}
.ev-card .ev-meta { color: #4F8EF7; font-size: 0.72rem; font-weight: 600; margin-bottom: 5px; }
.ev-card .ev-tags { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 6px; }
.ev-tag {
    background: #1C2333; color: #6B7280; border-radius: 4px;
    padding: 1px 7px; font-size: 0.65rem; font-weight: 500;
}
.ev-text { color: #94A3B8; line-height: 1.5; }

/* ── Trace Box ── */
.trace-box {
    background: #0A0E1A; border: 1px solid #1C2333; border-radius: 8px;
    padding: 10px 14px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: #4B5563; line-height: 1.8;
}

/* ── Welcome Screen ── */
.welcome-wrap {
    display: flex; align-items: center; justify-content: center;
    min-height: 70vh;
}
.welcome-card {
    background: #0D1117; border: 1px solid #1C2333; border-radius: 20px;
    padding: 52px 60px; text-align: center; max-width: 620px;
}
.welcome-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, #4F8EF7, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.welcome-sub {
    color: #4B5563; font-size: 0.9rem; line-height: 1.7; margin-bottom: 28px;
}
.welcome-step {
    background: #131929; border: 1px solid #1C2333; border-radius: 10px;
    padding: 10px 16px; margin: 8px 0; text-align: left;
    color: #94A3B8; font-size: 0.85rem; display: flex; gap: 10px; align-items: center;
}
.step-num {
    background: linear-gradient(135deg, #4F8EF7, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; font-size: 1rem; min-width: 20px;
}

/* ── Chat Message Styling ── */
.meta-row {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 12px;
}
.latency-chip {
    color: #374151; font-size: 0.68rem; font-family: 'JetBrains Mono', monospace;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1C2333; border-radius: 4px; }

/* Sidebar button */
[data-testid="stSidebar"] button {
    background: #131929 !important; border: 1px solid #1C2333 !important;
    color: #94A3B8 !important; border-radius: 8px !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] button:hover {
    border-color: #4F8EF7 !important; color: #E2E8F0 !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #0D1117 !important; border: 1px solid #1C2333 !important;
    color: #E2E8F0 !important; border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #4F8EF7 !important; box-shadow: 0 0 0 2px rgba(79,142,247,0.1) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
for k, v in [
    ("pipeline", None), ("messages", []), ("pdf_name", None),
    ("doc_loaded", False), ("chunk_count", 0)
]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.pipeline is None:
    st.session_state.pipeline = _build_pipeline()


def extract_pdf_text(uploaded_file):
    try:
        import PyPDF2
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        return text.strip() or None
    except Exception as e:
        return f"ERROR: {str(e)}"


def _risk_class(risk):
    r = (risk or "").lower().strip()
    return r if r in ("critical", "high", "medium", "low") else "na"


def _grounding_bar(ratio):
    pct = int(ratio * 100)
    color = "#EF4444" if pct < 40 else ("#F59E0B" if pct < 70 else "#22C55E")
    label = "Rejected" if pct < 40 else ("Partial" if pct < 70 else "Verified")
    return f"""
    <div class="grounding-wrap">
      <div class="grounding-label">Grounding: {label} ({pct}% supported)</div>
      <div class="grounding-bar-bg">
        <div class="grounding-bar-fill" style="width:{pct}%; background:{color};"></div>
      </div>
    </div>"""


def format_result_as_message(result):
    """Format a PipelineResult into a professional legal analysis card."""
    risk = result.risk_level or "N/A"
    rc = _risk_class(risk)
    decision = result.decision or "NOT_FOUND"
    conf = (result.confidence or "LOW").upper()

    decision_classes = {
        "ANSWER": "tag-answer", "NOT_FOUND": "tag-notfound",
        "ESCALATE": "tag-escalate", "AMBIGUOUS": "tag-ambiguous"
    }
    dc = decision_classes.get(decision, "tag-notfound")
    conf_class = {"HIGH": "conf-high", "MEDIUM": "conf-medium", "LOW": "conf-low"}.get(conf, "conf-low")

    parts = []

    # ── Meta row ──
    parts.append(
        f'<div class="meta-row">'
        f'<span class="risk-badge risk-{rc}">⚠ Risk: {risk}</span>'
        f'<span class="decision-tag {dc}">{decision}</span>'
        f'<span class="{conf_class}">● {conf} Confidence</span>'
        f'<span class="latency-chip">{result.latency_ms:.0f}ms</span>'
        f'</div>'
    )

    # ── Answer body ──
    ans = result.answer
    if isinstance(ans, list):
        ans = "\n".join(str(x) for x in ans)
    parts.append(str(ans or ""))

    # ── Evidence panel ──
    if result.evidence:
        parts.append("\n---\n**📎 Retrieved Evidence**")
        for ev in result.evidence[:5]:
            cid = ev.get("clause_id", "?")
            sec = ev.get("section", "General")
            pg = ev.get("page", "?")
            txt = (ev.get("text", ""))[:250]
            tags = ev.get("legal_tags", [])
            tag_html = "".join(f'<span class="ev-tag">{t}</span>' for t in tags[:4])
            parts.append(
                f'<div class="ev-card">'
                f'<div class="ev-meta">{cid} · {sec} · p.{pg}</div>'
                f'<div class="ev-text">{txt}{"..." if len(ev.get("text","")) > 250 else ""}</div>'
                + (f'<div class="ev-tags">{tag_html}</div>' if tag_html else "") +
                f'</div>'
            )

    # ── Grounding meter ──
    v = result.verification or {}
    if v.get("verdict"):
        ratio = v.get("supported_ratio", 0)
        parts.append(f'\n{_grounding_bar(ratio)}')

    return "\n\n".join(parts)


def run_initial_scan():
    pipeline = st.session_state.pipeline
    results = pipeline.get_all_risks()
    if not results:
        return "I've analyzed your document but couldn't identify specific risk clauses. Ask me anything about the contract."

    parts = [
        f"I've analyzed **{st.session_state.pdf_name}** "
        f"({st.session_state.chunk_count} sections identified). "
        f"Here are the risks I found:\n"
    ]
    for r in results:
        rc = _risk_class(r.risk_level)
        sec = r.evidence[0].get("section", "General") if r.evidence else "General"
        answer_preview = "\n".join((r.answer or "").split("\n")[:3])
        parts.append(
            f'<span class="risk-badge risk-{rc}">Risk: {r.risk_level}</span> '
            f'**{sec}**\n\n{answer_preview}\n'
        )
    parts.append("\n---\nAsk me anything about specific clauses, risks, or negotiation strategies.")
    return "\n\n".join(parts)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="cs-logo">⚖️ ContractSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="cs-tagline">Grounded Legal Copilot</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Upload Contract**")
    uploaded = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if uploaded.name != st.session_state.pdf_name:
            st.session_state.pdf_name = uploaded.name
            st.session_state.messages = []
            text = extract_pdf_text(uploaded)
            if text and not str(text).startswith("ERROR:"):
                pipeline = _build_pipeline()
                count = pipeline.load_document(text, uploaded.name)
                st.session_state.pipeline = pipeline
                st.session_state.doc_loaded = True
                st.session_state.chunk_count = count
                with st.spinner("Scanning for risks…"):
                    scan_msg = run_initial_scan()
                st.session_state.messages.append({"role": "assistant", "content": scan_msg})
                st.rerun()
            else:
                err = text if str(text).startswith("ERROR:") else "Could not extract text from PDF (may be scanned/image-based)."
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.rerun()

    if st.session_state.pdf_name:
        st.markdown(
            f'<div class="doc-card">'
            f'<div class="doc-name">📄 {st.session_state.pdf_name}</div>'
            f'<div class="doc-meta">{st.session_state.chunk_count} chunks · indexed ✓</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div style="color:#374151; font-size:0.8rem;">No document uploaded</div>', unsafe_allow_html=True)

    st.markdown("---")
    show_trace = st.checkbox("Show pipeline trace", value=False)

    if st.button("🔄 New Analysis", use_container_width=True):
        st.session_state.pipeline = _build_pipeline()
        st.session_state.messages = []
        st.session_state.pdf_name = None
        st.session_state.doc_loaded = False
        st.session_state.chunk_count = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem; color:#1F2937; line-height:1.6;">'
        '"This system prioritizes grounded legal reasoning and uncertainty awareness over generative fluency."'
        '</div>',
        unsafe_allow_html=True
    )


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-card">
        <div class="welcome-title">ContractSense</div>
        <div class="welcome-sub">
          Grounded explainable legal reasoning under uncertainty.<br/>
          Every answer is traceable to specific contract clauses.
        </div>
        <div class="welcome-step"><span class="step-num">1</span>Upload a contract PDF in the sidebar</div>
        <div class="welcome-step"><span class="step-num">2</span>Get an instant risk audit with evidence citations</div>
        <div class="welcome-step"><span class="step-num">3</span>Ask dispute, clause, or negotiation questions — grounded answers only</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if "trace" in msg and show_trace:
                with st.expander("🔍 Pipeline Trace", expanded=False):
                    st.markdown(
                        '<div class="trace-box">' + "<br/>".join(msg["trace"]) + "</div>",
                        unsafe_allow_html=True,
                    )

if prompt := st.chat_input("Ask about your contract — disputes, clauses, risks, obligations…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.doc_loaded:
            with st.spinner("Running grounded analysis…"):
                result = st.session_state.pipeline.query(prompt)
            response = format_result_as_message(result)
            st.markdown(response, unsafe_allow_html=True)
            if show_trace:
                with st.expander("🔍 Pipeline Trace", expanded=False):
                    st.markdown(
                        '<div class="trace-box">' + "<br/>".join(result.pipeline_trace) + "</div>",
                        unsafe_allow_html=True,
                    )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "trace": result.pipeline_trace,
            })
        else:
            msg = "Please upload a contract PDF first using the sidebar."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
