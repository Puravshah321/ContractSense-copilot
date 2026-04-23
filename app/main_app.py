"""
ContractSense Legal Copilot - Grounded Pipeline UI
Upload PDF -> Auto risk scan -> Chat with evidence + citations.
"""
import streamlit as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.orchestrator import ContractSensePipeline

st.set_page_config(
    page_title="ContractSense",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──
st.markdown("""
<style>
.stApp { background-color: #0D0D0D; }
[data-testid="stSidebar"] {
    background: #171717;
    border-right: 1px solid #2A2A2A;
}
.sidebar-logo {
    font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.risk-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
}
.risk-critical { background: #DC262633; color: #FCA5A5; }
.risk-high { background: #EF444433; color: #FCA5A5; }
.risk-medium { background: #F59E0B33; color: #FCD34D; }
.risk-low { background: #22C55E33; color: #86EFAC; }
.risk-na { background: #6B728033; color: #9CA3AF; }
.evidence-box {
    background: #1A1A2E; border: 1px solid #2D2D44; border-radius: 8px;
    padding: 10px 14px; margin: 6px 0; font-size: 0.82rem; color: #9CA3AF;
}
.trace-box {
    background: #111; border: 1px solid #222; border-radius: 6px;
    padding: 8px 12px; font-size: 0.72rem; color: #6B7280;
    font-family: monospace;
}
.decision-tag {
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 0.68rem; font-weight: 600;
}
.tag-answer { background: #22C55E22; color: #86EFAC; }
.tag-notfound { background: #EF444422; color: #FCA5A5; }
.tag-escalate { background: #F59E0B22; color: #FCD34D; }
.welcome-card {
    background: #121218; border: 1px solid #2A2A2A; border-radius: 16px;
    padding: 40px; text-align: center; margin: 60px auto;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session State ──
if "pipeline" not in st.session_state:
    st.session_state.pipeline = ContractSensePipeline()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


def extract_pdf_text(uploaded_file):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return None


def format_result_as_message(result):
    """Format a PipelineResult into a rich chat message."""
    risk = result.risk_level
    risk_class = risk.lower().replace("/", "").strip() if risk else "na"
    if risk_class not in ("critical", "high", "medium", "low"):
        risk_class = "na"

    decision_class = {
        "ANSWER": "tag-answer", "NOT_FOUND": "tag-notfound", "ESCALATE": "tag-escalate",
    }.get(result.decision, "tag-notfound")

    # Build message
    parts = []

    # Risk + Decision badges
    parts.append(
        f'<span class="risk-badge risk-{risk_class}">Risk: {risk}</span> &nbsp; '
        f'<span class="decision-tag {decision_class}">{result.decision}</span> &nbsp; '
        f'<span style="color:#6B7280; font-size:0.7rem;">Confidence: {result.confidence} | {result.latency_ms:.0f}ms</span>'
    )

    parts.append("")  # spacing
    parts.append(result.answer)

    # Evidence-only action
    if result.action and result.decision != "NOT_FOUND":
        parts.append(f"\n**Action based ONLY on evidence:** {result.action}")

    # Evidence
    if result.evidence:
        parts.append("\n**Evidence:**")
        for ev in result.evidence[:3]:
            cid = ev.get("clause_id", "?")
            sec = ev.get("section", "")
            pg = ev.get("page", "?")
            txt = ev.get("text", "")[:200]
            parts.append(
                f'<div class="evidence-box">'
                f'<b>{cid}</b> ({sec}, p.{pg})<br/>{txt}...</div>'
            )

    # Verification
    v = result.verification
    if v.get("verdict"):
        emoji = {"VERIFIED": "Verified", "PARTIALLY_VERIFIED": "Partial", "REJECTED": "Rejected"}.get(v["verdict"], v["verdict"])
        ratio = v.get("supported_ratio", 0)
        parts.append(
            f'\n<span style="color:#6B7280; font-size:0.75rem;">'
            f'Grounding: {emoji} ({ratio:.0%} claims supported)</span>'
        )

    return "\n".join(parts)


def run_initial_scan():
    """Run the initial full-document risk scan."""
    pipeline = st.session_state.pipeline
    results = pipeline.get_all_risks()

    if not results:
        return "I analyzed your document but couldn't identify specific risk clauses. Try asking me specific questions about the contract."

    parts = [f"I've analyzed **{st.session_state.pdf_name}** ({st.session_state.chunk_count} sections identified). Here are the risks I found:\n"]

    for r in results:
        risk_class = r.risk_level.lower() if r.risk_level not in ("N/A",) else "na"
        if risk_class not in ("critical", "high", "medium", "low"):
            risk_class = "na"

        parts.append(f'<span class="risk-badge risk-{risk_class}">Risk: {r.risk_level}</span>')

        # Extract section name from evidence
        section = ""
        if r.evidence:
            section = r.evidence[0].get("section", "")

        parts.append(f"**{section or 'General'}**\n")
        # Truncate answer for the overview
        answer_lines = r.answer.split("\n")
        short = "\n".join(answer_lines[:4])
        parts.append(f"{short}\n")

    parts.append("\n---\nAsk me anything about specific clauses, risks, or negotiation strategies.")
    return "\n".join(parts)


# ── SIDEBAR ──
with st.sidebar:
    st.markdown('<div class="sidebar-logo">ContractSense</div>', unsafe_allow_html=True)
    st.caption("Grounded Legal Copilot")
    st.markdown("---")

    st.markdown("#### Upload Contract")
    uploaded = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if uploaded.name != st.session_state.pdf_name:
            st.session_state.pdf_name = uploaded.name
            st.session_state.messages = []

            text = extract_pdf_text(uploaded)
            if text:
                pipeline = ContractSensePipeline()
                count = pipeline.load_document(text, uploaded.name)
                st.session_state.pipeline = pipeline
                st.session_state.doc_loaded = True
                st.session_state.chunk_count = count

                # Auto-scan
                with st.spinner("Scanning for risks..."):
                    scan_msg = run_initial_scan()
                st.session_state.messages.append({"role": "assistant", "content": scan_msg})
                st.rerun()
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Could not extract text from this PDF. It may be scanned or image-based.",
                })
                st.rerun()

    if st.session_state.pdf_name:
        st.markdown(f"**{st.session_state.pdf_name}**")
        st.caption(f"{st.session_state.chunk_count} chunks indexed")
    else:
        st.caption("No document uploaded")

    st.markdown("---")

    # Pipeline trace toggle
    show_trace = st.checkbox("Show pipeline trace", value=False)

    if st.button("New Analysis", use_container_width=True):
        st.session_state.pipeline = ContractSensePipeline()
        st.session_state.messages = []
        st.session_state.pdf_name = None
        st.session_state.doc_loaded = False
        st.session_state.chunk_count = 0
        st.rerun()


# ── MAIN CHAT ──
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:2rem; font-weight:800; color:white; margin-bottom:12px;">
            ContractSense Copilot
        </div>
        <div style="color:#9CA3AF; font-size:0.95rem; line-height:1.6; margin-bottom:24px;">
            Document-grounded contract intelligence.<br/>
            Every answer is backed by evidence from your actual contract.
        </div>
        <div style="text-align:left; display:inline-block; color:#D1D5DB; font-size:0.9rem;">
            <p>1. Upload a contract PDF in the sidebar</p>
            <p>2. Get an instant risk audit with evidence citations</p>
            <p>3. Ask follow-up questions &mdash; grounded answers only</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if "trace" in msg and show_trace:
                with st.expander("Pipeline Trace", expanded=False):
                    st.markdown(
                        '<div class="trace-box">' + "<br/>".join(msg["trace"]) + "</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask about your contract..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.doc_loaded:
            with st.spinner("Running grounded pipeline..."):
                result = st.session_state.pipeline.query(prompt)
            response = format_result_as_message(result)
            st.markdown(response, unsafe_allow_html=True)
            if show_trace:
                with st.expander("Pipeline Trace", expanded=False):
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
