"""
ContractSense Legal Copilot - Chat-Based Contract Analysis
Upload a PDF. Get instant risk analysis. Ask follow-up questions.
"""
import streamlit as st
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from demo_data import detect_clauses, generate_initial_analysis, answer_followup

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ContractSense",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS - ChatGPT-like dark theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* Main background */
.stApp {
    background-color: #0D0D0D;
}
.stMainBlockContainer {
    max-width: 860px;
    margin: 0 auto;
    padding-top: 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #171717;
    border-right: 1px solid #2A2A2A;
}

/* Header */
.brand-header {
    text-align: center;
    padding: 20px 0 10px 0;
}
.brand-name {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #A78BFA, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.brand-tag {
    font-size: 0.82rem;
    color: #6B7280;
    margin-top: 2px;
}

/* Chat messages */
.stChatMessage {
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stChatMessage"]:nth-child(odd) {
    background: #1A1A1A !important;
}
[data-testid="stChatMessage"]:nth-child(even) {
    background: #0D0D0D !important;
}

/* Chat input */
.stChatInput > div {
    background: #1A1A1A !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
}
.stChatInput textarea {
    color: #E5E7EB !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 1px dashed #333 !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* Status chips */
.status-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.chip-ready {
    background: rgba(34,197,94,0.15);
    color: #22C55E;
    border: 1px solid #22C55E33;
}
.chip-waiting {
    background: rgba(107,114,128,0.15);
    color: #6B7280;
    border: 1px solid #6B728033;
}

/* Sidebar file info */
.file-info {
    background: #1F1F1F;
    border: 1px solid #2A2A2A;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    font-size: 0.8rem;
    color: #9CA3AF;
    line-height: 1.6;
}

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, #1A1530, #1E1A2E);
    border: 1px solid #6C63FF22;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    margin: 60px auto;
    max-width: 600px;
}
.welcome-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}
.welcome-sub {
    font-size: 0.95rem;
    color: #9CA3AF;
    line-height: 1.6;
}
.welcome-steps {
    text-align: left;
    margin-top: 24px;
    padding: 0 20px;
}
.welcome-step {
    display: flex;
    align-items: flex-start;
    margin: 12px 0;
    color: #D1D5DB;
    font-size: 0.88rem;
}
.step-num {
    background: #6C63FF;
    color: white;
    min-width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 12px;
    margin-top: 2px;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "clauses_found" not in st.session_state:
    st.session_state.clauses_found = {}
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================
def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(uploaded_file)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        try:
            import fitz  # PyMuPDF
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            uploaded_file.seek(0)
            return "\n\n".join(text_parts)
        except ImportError:
            return None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0;">
        <p style="font-size:1.2rem; font-weight:700; color:#A78BFA; margin:0;">ContractSense</p>
        <p style="font-size:0.75rem; color:#6B7280; margin:0;">Legal Copilot</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # PDF Upload
    st.markdown("**Upload Contract**")
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        # Show file info
        file_size = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div class="file-info">
            <b>{uploaded_file.name}</b><br/>
            Size: {file_size:.1f} KB
        </div>
        """, unsafe_allow_html=True)

        # Status
        if st.session_state.analyzed:
            st.markdown(
                '<span class="status-chip chip-ready">Analyzed</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-chip chip-waiting">Ready to Analyze</span>',
                unsafe_allow_html=True,
            )

        # Process PDF if new file
        if uploaded_file.name != st.session_state.pdf_name:
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            st.session_state.clauses_found = {}
            st.session_state.messages = []
            st.session_state.analyzed = False

            if st.session_state.pdf_text:
                st.session_state.clauses_found = detect_clauses(st.session_state.pdf_text)

                # Auto-generate initial analysis
                analysis = generate_initial_analysis(st.session_state.clauses_found)
                if analysis:
                    st.session_state.messages.append({"role": "assistant", "content": analysis})
                    st.session_state.analyzed = True
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            "I've processed your PDF but couldn't detect any standard contract clauses. "
                            "This might not be a legal contract, or the format may be unusual. "
                            "Feel free to paste specific clauses and ask me about them!"
                        ),
                    })
                    st.session_state.analyzed = True
                st.rerun()
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't extract text from this PDF. It may be scanned or image-based. Please try a text-based PDF.",
                })
                st.session_state.analyzed = True
                st.rerun()

        if st.session_state.clauses_found:
            st.markdown("---")
            st.markdown("**Detected Clauses**")
            for clause_type in st.session_state.clauses_found:
                st.markdown(f"- {clause_type}")

    else:
        st.markdown(
            '<span class="status-chip chip-waiting">No document uploaded</span>',
            unsafe_allow_html=True,
        )

    # New chat button
    st.markdown("---")
    if st.button("New Analysis", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pdf_text = None
        st.session_state.clauses_found = {}
        st.session_state.pdf_name = None
        st.session_state.analyzed = False
        st.rerun()


# ============================================================
# MAIN CHAT AREA
# ============================================================

# Welcome screen (no PDF uploaded yet)
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">ContractSense</div>
        <div class="welcome-sub">
            Your AI-powered legal copilot for contract risk analysis.
            Upload a contract and I'll identify every risk, explain it in plain English,
            and tell you exactly what to do about it.
        </div>
        <div class="welcome-steps">
            <div class="welcome-step">
                <div class="step-num">1</div>
                <div>Upload a contract PDF using the sidebar</div>
            </div>
            <div class="welcome-step">
                <div class="step-num">2</div>
                <div>I'll automatically scan for risks and flag critical issues</div>
            </div>
            <div class="welcome-step">
                <div class="step-num">3</div>
                <div>Ask follow-up questions about any clause or concern</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about your contract..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.pdf_text and st.session_state.clauses_found:
            response = answer_followup(
                prompt,
                st.session_state.clauses_found,
                st.session_state.pdf_text,
            )
        elif st.session_state.pdf_text:
            response = (
                "I've processed your document but didn't find standard contract clauses. "
                "Could you point me to a specific section or paste the clause you'd like me to analyze?"
            )
        else:
            response = (
                "Please upload a contract PDF first using the sidebar, "
                "and I'll analyze it for risks and potential issues."
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
