"""
ContractSense Legal Copilot - Corrected Dynamic UI
Upload PDF -> Risk Scan -> Chat with contract evidence.
"""

import streamlit as st
import sys
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.orchestrator import ContractSensePipeline


def _build_pipeline():
    use_llm = os.environ.get("FORCE_LOCAL_LLM", "0") == "1"
    return ContractSensePipeline(use_llm=use_llm)


st.set_page_config(
    page_title="ContractSense",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(37,99,235,0.20), transparent 30%),
        radial-gradient(circle at bottom right, rgba(20,184,166,0.12), transparent 30%),
        linear-gradient(135deg, #050816 0%, #0B1020 50%, #111827 100%);
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background: #07111F !important;
    border-right: 1px solid rgba(148,163,184,0.18);
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    max-width: 1350px;
}

.brand-card {
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(37,99,235,0.24), rgba(20,184,166,0.10));
    border: 1px solid rgba(148,163,184,0.18);
    margin-bottom: 22px;
}

.brand-title {
    font-size: 1.5rem;
    font-weight: 900;
    color: white;
}

.brand-subtitle {
    color: #93C5FD;
    font-size: 0.82rem;
    margin-top: 4px;
}

.kicker {
    color: #60A5FA;
    font-weight: 800;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.app-title {
    font-size: 2.7rem;
    font-weight: 900;
    letter-spacing: -2px;
    color: white;
    margin-top: 0.4rem;
}

.app-subtitle {
    color: #CBD5E1;
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 900px;
}

.metric-card {
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 22px;
    padding: 22px;
    min-height: 120px;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 900;
    color: white;
}

.metric-label {
    color: #94A3B8;
    font-size: 0.86rem;
    margin-top: 8px;
}

.glass-card {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.24);
}

.section-title {
    color: white;
    font-size: 1.25rem;
    font-weight: 850;
    margin-bottom: 8px;
}

.section-text {
    color: #CBD5E1;
    font-size: 0.95rem;
    line-height: 1.65;
}

.risk-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 900;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.risk-critical {
    background: rgba(239,68,68,0.15);
    color: #FCA5A5;
    border: 1px solid rgba(239,68,68,0.35);
}

.risk-high {
    background: rgba(249,115,22,0.15);
    color: #FDBA74;
    border: 1px solid rgba(249,115,22,0.35);
}

.risk-medium {
    background: rgba(245,158,11,0.15);
    color: #FCD34D;
    border: 1px solid rgba(245,158,11,0.35);
}

.risk-low {
    background: rgba(16,185,129,0.15);
    color: #6EE7B7;
    border: 1px solid rgba(16,185,129,0.35);
}

.risk-na {
    background: rgba(148,163,184,0.12);
    color: #CBD5E1;
    border: 1px solid rgba(148,163,184,0.28);
}

.decision-tag {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 900;
    border: 1px solid rgba(148,163,184,0.22);
    margin-bottom: 8px;
}

.tag-answer {
    color: #6EE7B7;
}

.tag-notfound {
    color: #FCA5A5;
}

.tag-escalate {
    color: #FCD34D;
}

.evidence-box {
    background: rgba(15,23,42,0.85);
    border-left: 4px solid #3B82F6;
    border-radius: 16px;
    padding: 16px;
    margin: 12px 0;
    border-top: 1px solid rgba(148,163,184,0.12);
    border-right: 1px solid rgba(148,163,184,0.12);
    border-bottom: 1px solid rgba(148,163,184,0.12);
    color: #CBD5E1;
    line-height: 1.65;
}

.evidence-box b {
    color: #93C5FD;
}

.small-muted {
    color: #94A3B8;
    font-size: 0.85rem;
}

[data-testid="stChatMessage"] {
    background: rgba(15,23,42,0.58);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 22px;
    padding: 14px;
    margin-bottom: 14px;
}

.stButton > button {
    border-radius: 16px !important;
    background: linear-gradient(135deg, #2563EB, #10B981) !important;
    color: white !important;
    border: none !important;
    height: 3rem;
    font-weight: 800 !important;
}

.stFileUploader {
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    padding: 10px;
    border: 1px dashed rgba(147,197,253,0.35);
}
</style>
""", unsafe_allow_html=True)


# ───────────────── SESSION STATE ─────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = _build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "document_history" not in st.session_state:
    st.session_state.document_history = {}


# ───────────────── HELPERS ─────────────────
def extract_pdf_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    try:
        import fitz
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""

        for page in pdf:
            text += page.get_text("text") + "\n"

        if text.strip():
            return text

    except Exception:
        pass

    try:
        import io
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return text

    except Exception:
        return ""


def get_risk_class(risk):
    risk_class = risk.lower().replace("/", "").strip() if risk else "na"

    if risk_class not in ("critical", "high", "medium", "low"):
        risk_class = "na"

    return risk_class


def save_current_document_history():
    if st.session_state.pdf_name:
        st.session_state.document_history[st.session_state.pdf_name] = {
            "messages": st.session_state.messages.copy(),
            "chunk_count": st.session_state.chunk_count,
            "doc_loaded": st.session_state.doc_loaded,
            "pipeline": st.session_state.pipeline,
        }


def load_document_history(pdf_name):
    saved = st.session_state.document_history.get(pdf_name)

    if saved:
        st.session_state.messages = saved.get("messages", [])
        st.session_state.chunk_count = saved.get("chunk_count", 0)
        st.session_state.doc_loaded = saved.get("doc_loaded", False)
        st.session_state.pipeline = saved.get("pipeline", _build_pipeline())


def format_result_as_message(result):
    risk = getattr(result, "risk_level", "N/A")
    risk_class = get_risk_class(risk)

    decision = getattr(result, "decision", "ANSWER")

    decision_class = {
        "ANSWER": "tag-answer",
        "NOT_FOUND": "tag-notfound",
        "ESCALATE": "tag-escalate",
    }.get(decision, "tag-answer")

    parts = []

    parts.append(
        f'<span class="risk-badge risk-{risk_class}">Risk: {risk}</span> '
        f'<span class="decision-tag {decision_class}">{decision}</span>'
    )

    confidence = getattr(result, "confidence", None)
    latency = getattr(result, "latency_ms", None)

    if confidence is not None and latency is not None:
        parts.append(
            f'<div class="small-muted">Confidence: {confidence} · Latency: {latency:.0f}ms</div>'
        )

    answer = getattr(result, "answer", "No answer generated.")
    parts.append(answer)

    evidence = getattr(result, "evidence", [])

    if evidence:
        parts.append("\n**Evidence from contract:**")

        for ev in evidence[:3]:
            cid = ev.get("clause_id", "?")
            sec = ev.get("section", "")
            pg = ev.get("page", "?")
            txt = ev.get("text", "")[:240]

            parts.append(
                f'<div class="evidence-box">'
                f'<b>{cid}</b> · {sec} · Page {pg}<br>'
                f'{txt}...'
                f'</div>'
            )

    verification = getattr(result, "verification", {})

    if verification and verification.get("verdict"):
        ratio = verification.get("supported_ratio", 0)

        parts.append(
            f'<div class="small-muted">Grounding: {verification["verdict"]} · '
            f'{ratio:.0%} claims supported</div>'
        )

    return "\n".join(parts)


def run_initial_scan():
    try:
        results = st.session_state.pipeline.get_all_risks()
    except Exception as e:
        return f"Risk scan could not be completed. Error: {e}"

    if not results:
        return (
            "### Executive Risk Snapshot\n"
            "The document was uploaded and indexed, but no specific risk clauses were automatically detected. "
            "You can still ask scenario-based questions about the agreement."
        )

    parts = [
        "### Executive Risk Snapshot\n",
        f"Analyzed **{st.session_state.pdf_name}** with "
        f"**{st.session_state.chunk_count} sections indexed**.\n"
    ]

    for r in results:
        risk_class = get_risk_class(getattr(r, "risk_level", "N/A"))

        section = ""
        if getattr(r, "evidence", []):
            section = r.evidence[0].get("section", "")

        answer = getattr(r, "answer", "")
        short = "\n".join(answer.split("\n")[:3])

        parts.append(
            f'<span class="risk-badge risk-{risk_class}">Risk: {getattr(r, "risk_level", "N/A")}</span>'
        )
        parts.append(f"**{section or 'General Contract Risk'}**")
        parts.append(short)
        parts.append("")

    return "\n".join(parts)


def process_uploaded_pdf(uploaded):
    save_current_document_history()

    st.session_state.pdf_name = uploaded.name
    st.session_state.messages = []
    st.session_state.doc_loaded = False
    st.session_state.chunk_count = 0
    st.session_state.pipeline = _build_pipeline()

    text = extract_pdf_text(uploaded)

    if not text or len(text.strip()) < 50:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "I could not extract enough readable text from this PDF. "
                "It may be scanned/image-based. Try using a text-based PDF or OCR first."
            )
        })
        save_current_document_history()
        return

    try:
        count = st.session_state.pipeline.load_document(text, uploaded.name)
        count = int(count or 0)
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Document upload worked, but indexing failed. Error: {e}"
        })
        save_current_document_history()
        return

    st.session_state.chunk_count = count
    st.session_state.doc_loaded = count > 0

    if count == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "The PDF text was extracted, but the pipeline created 0 searchable sections. "
                "Please check your `load_document()` chunking logic."
            )
        })
        save_current_document_history()
        return

    with st.spinner("Running executive risk analysis..."):
        scan_msg = run_initial_scan()

    st.session_state.messages.append({
        "role": "assistant",
        "content": scan_msg
    })

    save_current_document_history()


# ───────────────── SIDEBAR ─────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-card">
        <div class="brand-title">ContractSense</div>
        <div class="brand-subtitle">Enterprise Legal Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Contract Intake")

    uploaded = st.file_uploader(
        "Upload Contract PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded and uploaded.name != st.session_state.pdf_name:
        process_uploaded_pdf(uploaded)
        st.rerun()

    st.markdown("---")

    if st.session_state.pdf_name:
        st.markdown("#### Active Document")
        st.markdown(f"**{st.session_state.pdf_name}**")
        st.caption(f"{st.session_state.chunk_count} sections indexed")

        if st.session_state.doc_loaded:
            st.success("Document ready for chat")
        else:
            st.error("Document not indexed")
    else:
        st.caption("No contract uploaded yet.")

    if st.session_state.document_history:
        st.markdown("---")
        st.markdown("#### Temporary PDF History")

        selected_doc = st.selectbox(
            "Switch document session",
            list(st.session_state.document_history.keys()),
            label_visibility="collapsed"
        )

        if st.button("Open Selected Session", use_container_width=True):
            save_current_document_history()
            st.session_state.pdf_name = selected_doc
            load_document_history(selected_doc)
            st.rerun()

    st.markdown("---")

    show_trace = st.checkbox("Show pipeline trace", value=False)

    if st.button("Start New Analysis", use_container_width=True):
        save_current_document_history()
        st.session_state.pipeline = _build_pipeline()
        st.session_state.messages = []
        st.session_state.pdf_name = None
        st.session_state.doc_loaded = False
        st.session_state.chunk_count = 0
        st.rerun()

    if st.button("Clear Temporary History", use_container_width=True):
        st.session_state.document_history = {}
        st.session_state.pipeline = _build_pipeline()
        st.session_state.messages = []
        st.session_state.pdf_name = None
        st.session_state.doc_loaded = False
        st.session_state.chunk_count = 0
        st.rerun()


# ───────────────── MAIN HEADER ─────────────────
st.markdown('<div class="kicker">AI-Powered Contract Risk Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-title">Contract review workspace for business decisions.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="app-subtitle">'
    'Upload contracts, keep separate temporary PDF sessions, review risk snapshots, '
    'and ask grounded scenario-based questions.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{'Ready' if st.session_state.doc_loaded else 'Waiting'}</div>
            <div class="metric-label">Document Status</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.chunk_count}</div>
            <div class="metric-label">Indexed Sections</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.messages)}</div>
            <div class="metric-label">Messages</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(st.session_state.document_history)}</div>
            <div class="metric-label">Temporary Sessions</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# ───────────────── TABS ─────────────────
tab_chat, tab_risks, tab_playbook, tab_history = st.tabs([
    "Chat Workspace",
    "Risk Dashboard",
    "Scenario Playbook",
    "PDF History"
])


with tab_chat:
    if not st.session_state.messages:
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">Start with a contract PDF</div>
                <div class="section-text">
                    Upload a PDF from the sidebar. ContractSense will index it,
                    generate a risk snapshot, and then you can ask contract questions.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("""
            <div class="glass-card">
                <div class="section-title">Example questions</div>
                <div class="section-text">
                    • Is virtualization allowed?<br>
                    • Is liability capped?<br>
                    • Can services be suspended for late payment?<br>
                    • Are SLA credits the only remedy?
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

                if "trace" in msg and show_trace:
                    with st.expander("Pipeline Trace", expanded=False):
                        st.write("\n".join(msg["trace"]))


with tab_risks:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">Risk Dashboard</div>
        <div class="section-text">
            Refresh the risk scan for the currently active contract.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.session_state.doc_loaded:
        if st.button("Refresh Risk Scan", use_container_width=True):
            scan_msg = run_initial_scan()
            st.session_state.messages.append({
                "role": "assistant",
                "content": scan_msg
            })
            save_current_document_history()
            st.rerun()
    else:
        st.warning("Upload and index a contract PDF first.")


with tab_playbook:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">Scenario Playbook</div>
        <div class="section-text">
            Try these scenario-based questions in the chat.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    q1, q2 = st.columns(2)

    with q1:
        st.markdown("#### Platform / SDK Use")
        st.code("A developer installs Apple SDKs on a Windows PC using virtualization tools. Is this allowed?")
        st.code("Can the software be used outside Apple-branded hardware?")

        st.markdown("#### Liability")
        st.code("Is liability always capped under this agreement?")
        st.code("Are there exceptions for confidentiality or data breach?")

    with q2:
        st.markdown("#### Termination")
        st.code("Can Apple terminate the agreement immediately?")
        st.code("What happens to developer rights after termination?")

        st.markdown("#### Data & Privacy")
        st.code("Can personal data be collected under this agreement?")
        st.code("What privacy obligations does the developer have?")


with tab_history:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">Temporary PDF History</div>
        <div class="section-text">
            Stored only during this Streamlit session. Restarting the app clears it.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.session_state.document_history:
        for doc_name, data in st.session_state.document_history.items():
            with st.expander(doc_name):
                st.write(f"Messages: {len(data.get('messages', []))}")
                st.write(f"Indexed sections: {data.get('chunk_count', 0)}")
                st.write(f"Ready: {data.get('doc_loaded', False)}")
    else:
        st.info("No temporary document history yet.")


# ───────────────── CHAT INPUT ─────────────────
if prompt := st.chat_input("Ask a contract scenario question..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.pdf_name:
            msg = "Please upload a contract PDF first using the sidebar."
            st.markdown(msg)

            st.session_state.messages.append({
                "role": "assistant",
                "content": msg
            })

        elif not st.session_state.doc_loaded:
            msg = (
                "The PDF is uploaded, but it was not indexed into searchable sections. "
                "Check whether text extraction worked and whether `load_document()` returned more than 0 chunks."
            )
            st.markdown(msg)

            st.session_state.messages.append({
                "role": "assistant",
                "content": msg
            })

        else:
            with st.spinner("Analyzing legal semantics..."):
                time.sleep(0.6)

            with st.spinner("Retrieving contract evidence..."):
                time.sleep(0.6)

            with st.spinner("Generating grounded answer..."):
                result = st.session_state.pipeline.query(prompt)

            response = format_result_as_message(result)

            placeholder = st.empty()
            full_text = ""

            for i in range(0, len(response), 5):
                full_text += response[i:i + 5]
                placeholder.markdown(full_text + "▌", unsafe_allow_html=True)
                time.sleep(0.004)

            placeholder.markdown(full_text, unsafe_allow_html=True)

            msg_payload = {
                "role": "assistant",
                "content": response,
            }

            if hasattr(result, "pipeline_trace"):
                msg_payload["trace"] = result.pipeline_trace

            st.session_state.messages.append(msg_payload)

        save_current_document_history()