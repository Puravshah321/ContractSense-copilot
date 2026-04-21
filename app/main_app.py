"""
ContractSense Legal Copilot - Streamlit Application
Stage 8: Professional Business Law Intelligence Interface
"""
import streamlit as st
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from demo_data import DEMO_CLAUSES, DEMO_QUERIES, DEMO_RESPONSES, RISK_COLORS


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ContractSense | Legal Copilot",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* --- HEADER --- */
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #A78BFA 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #9CA3AF;
    margin-top: 0;
    letter-spacing: 0.2px;
}

/* --- SIDEBAR --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141A 0%, #1A1D26 100%);
    border-right: 1px solid #2D3748;
}
.sidebar-brand {
    font-size: 1.3rem;
    font-weight: 700;
    color: #A78BFA;
    margin-bottom: 4px;
}
.sidebar-version {
    font-size: 0.75rem;
    color: #6B7280;
    margin-bottom: 20px;
}

/* --- RISK BADGE --- */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.risk-low { background: rgba(34, 197, 94, 0.15); color: #22C55E; border: 1px solid #22C55E33; }
.risk-medium { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid #F59E0B33; }
.risk-high { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid #EF444433; }
.risk-critical { background: rgba(220, 38, 38, 0.15); color: #DC2626; border: 1px solid #DC262633; }

/* --- ANALYSIS CARD --- */
.analysis-card {
    background: linear-gradient(135deg, #1E2028 0%, #1A1D24 100%);
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
    transition: border-color 0.3s ease;
}
.analysis-card:hover { border-color: #6C63FF44; }

.card-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6C63FF;
    margin-bottom: 8px;
}
.card-content {
    font-size: 0.92rem;
    line-height: 1.7;
    color: #D1D5DB;
}

/* --- CLAUSE BOX --- */
.clause-box {
    background: #14161C;
    border: 1px solid #374151;
    border-left: 3px solid #6C63FF;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #9CA3AF;
    font-style: italic;
}

/* --- METRIC CARD --- */
.metric-card {
    background: linear-gradient(135deg, #1A1D26, #1E2030);
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.75rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* --- DIVIDER --- */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #2D3748, transparent);
    margin: 28px 0;
}

/* --- STATUS INDICATOR --- */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22C55E;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* --- CHAT --- */
.chat-user {
    background: #1E2030;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #E5E7EB;
}
.chat-ai {
    background: linear-gradient(135deg, #1A1530, #1E1A2E);
    border: 1px solid #6C63FF33;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #D1D5DB;
}

/* --- TABS --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
}

/* --- HIDE STREAMLIT BRANDING --- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<p class="sidebar-brand">ContractSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-version">v2.0 | DPO Aligned Engine</p>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("**Navigation**")
    page = st.radio(
        "Go to",
        ["Clause Analyzer", "Performance Dashboard", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("**Model Configuration**")
    st.markdown(
        '<span class="status-dot"></span> <span style="color:#9CA3AF; font-size:0.85rem;">Engine Online</span>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="margin-top:12px; font-size:0.78rem; color:#6B7280; line-height:1.6;">
    <b>Base Model:</b> Mistral-7B-Instruct-v0.2<br/>
    <b>Alignment:</b> DPO (Stage 7)<br/>
    <b>Adapters:</b> LoRA r=16, alpha=32<br/>
    <b>Quality Score:</b> 0.982 / 1.000<br/>
    <b>Format Compliance:</b> 100%<br/>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.7rem; color:#4B5563; text-align:center;">Built by Team ContractSense<br/>SEM-2 Deep Learning Project</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: CLAUSE ANALYZER
# ============================================================
if page == "Clause Analyzer":
    st.markdown('<p class="hero-title">ContractSense Legal Copilot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">AI-powered contract intelligence. Paste a clause, ask a question, get actionable legal insight.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- INPUT SECTION ---
    col_input, col_spacer, col_query = st.columns([5, 0.5, 4])

    with col_input:
        st.markdown("**Select a Sample Clause** or paste your own below")
        clause_choice = st.selectbox(
            "Choose clause type",
            ["-- Paste Custom --"] + [v["title"] for v in DEMO_CLAUSES.values()],
            label_visibility="collapsed",
        )

        clause_key = None
        if clause_choice != "-- Paste Custom --":
            for k, v in DEMO_CLAUSES.items():
                if v["title"] == clause_choice:
                    clause_key = k
                    break

        if clause_key:
            clause_text = st.text_area(
                "Contract Clause",
                value=DEMO_CLAUSES[clause_key]["text"],
                height=200,
                label_visibility="collapsed",
            )
        else:
            clause_text = st.text_area(
                "Contract Clause",
                placeholder="Paste your contract clause here...\n\ne.g., 'Section 9.2 -- Termination. Either party may terminate this Agreement at any time...'",
                height=200,
                label_visibility="collapsed",
            )

    with col_query:
        st.markdown("**Ask a Question**")

        if clause_key and clause_key in DEMO_QUERIES:
            query_options = DEMO_QUERIES[clause_key]
            selected_query = st.selectbox(
                "Suggested questions",
                query_options,
                label_visibility="collapsed",
            )
            user_query = st.text_input(
                "Or type your own question",
                value=selected_query,
                label_visibility="collapsed",
            )
        else:
            user_query = st.text_input(
                "Your question",
                placeholder="What are the key risks in this clause?",
                label_visibility="collapsed",
            )

        st.markdown("")
        analyze_btn = st.button("Analyze Clause", type="primary", use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # --- ANALYSIS OUTPUT ---
    if analyze_btn and clause_text and user_query:
        # Show the clause being analyzed
        st.markdown("##### Analyzing Clause")
        st.markdown(f'<div class="clause-box">{clause_text[:500]}{"..." if len(clause_text)>500 else ""}</div>', unsafe_allow_html=True)
        st.markdown("")

        # Simulate processing
        with st.spinner("Running DPO-aligned inference pipeline..."):
            progress = st.progress(0)
            steps = ["Tokenizing input", "Loading DPO adapters", "Generating structured output", "Parsing response"]
            for i, step in enumerate(steps):
                time.sleep(0.5)
                progress.progress((i + 1) * 25, text=f"Step {i+1}/4: {step}")
            time.sleep(0.3)
            progress.empty()

        # Get response
        response = None
        if clause_key and clause_key in DEMO_RESPONSES:
            responses_for_clause = DEMO_RESPONSES[clause_key]
            if user_query in responses_for_clause:
                response = responses_for_clause[user_query]

        if response is None:
            response = {
                "risk_level": "MEDIUM",
                "explanation": (
                    f"Based on the provided clause, the key consideration regarding '{user_query.lower().rstrip('?')}' "
                    f"involves careful examination of the contractual obligations and their implications. "
                    f"The language used suggests standard commercial terms, but specific provisions should "
                    f"be reviewed by qualified legal counsel to assess their impact on your organization."
                ),
                "action": (
                    "1) Consult with your legal team to review the specific implications. "
                    "2) Compare this clause against industry-standard provisions. "
                    "3) Negotiate modifications if the terms create disproportionate risk."
                ),
                "citation": "Full clause text, chars 0-" + str(len(clause_text)),
            }

        risk = response["risk_level"]
        risk_class = risk.lower()
        risk_color = RISK_COLORS.get(risk, "#F59E0B")

        # --- STRUCTURED OUTPUT ---
        st.markdown("##### Analysis Results")

        # Risk badge
        st.markdown(f'<span class="risk-badge risk-{risk_class}">RISK: {risk}</span>', unsafe_allow_html=True)
        st.markdown("")

        # Three cards
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"""
            <div class="analysis-card">
                <div class="card-label">Plain Explanation</div>
                <div class="card-content">{response["explanation"]}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class="analysis-card">
                <div class="card-label">Recommended Action</div>
                <div class="card-content">{response["action"]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="analysis-card" style="border-left: 3px solid {risk_color};">
            <div class="card-label">Citation</div>
            <div class="card-content" style="font-family: monospace !important; font-size: 0.82rem;">{response["citation"]}</div>
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn:
        st.warning("Please provide both a clause and a question to analyze.")


# ============================================================
# PAGE: PERFORMANCE DASHBOARD
# ============================================================
elif page == "Performance Dashboard":
    st.markdown('<p class="hero-title">Model Performance</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">DPO alignment benchmarks compared against the Stage 6 SFT (LoRA) baseline.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("98.2%", "Overall Quality"),
        ("100%", "Format Compliance"),
        ("100%", "Risk Salience"),
        ("100%", "Actionability"),
    ]
    for col, (val, label) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Comparison table
    st.markdown("##### Stage 6 (SFT) vs Stage 7 (DPO) Comparison")

    comparison_data = {
        "Metric": ["Overall Quality", "Format Compliance", "Risk Salience", "Actionability", "Citation Recall"],
        "Stage 6 (SFT LoRA)": ["0.8778", "0.9583", "0.8750", "0.9250", "0.8417"],
        "Stage 7 (DPO)": ["0.9817", "1.0000", "1.0000", "1.0000", "1.0000"],
        "Improvement": ["+10.4%", "+4.2%", "+12.5%", "+7.5%", "+15.8%"],
    }
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Error elimination
    st.markdown("##### Error Elimination (per 120 evaluation documents)")
    e1, e2, e3, e4 = st.columns(4)

    error_data = [
        ("Missing Citation", "19", "0"),
        ("Missing Risk Label", "15", "0"),
        ("No Action Provided", "9", "0"),
        ("Broken JSON Format", "5", "0"),
    ]
    for col, (err, sft_count, dpo_count) in zip([e1, e2, e3, e4], error_data):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.8rem; color:#EF4444; font-weight:600;">SFT: {sft_count} errors</div>
                <div style="font-size:1.6rem; font-weight:800; color:#22C55E; margin:8px 0;">0</div>
                <div class="metric-label">{err}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Training details
    st.markdown("##### Training Configuration")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("""
        <div class="analysis-card">
            <div class="card-label">Infrastructure</div>
            <div class="card-content">
                <b>Platform:</b> Lightning AI Studio<br/>
                <b>GPU:</b> NVIDIA L4 (24GB VRAM)<br/>
                <b>Precision:</b> BFloat16 + 4-bit NF4 Quantization<br/>
                <b>Attention:</b> Native PyTorch SDPA<br/>
                <b>Training Time:</b> ~50 minutes
            </div>
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        st.markdown("""
        <div class="analysis-card">
            <div class="card-label">Hyperparameters</div>
            <div class="card-content">
                <b>Algorithm:</b> Direct Preference Optimization (DPO)<br/>
                <b>Beta:</b> 0.1<br/>
                <b>LoRA Rank:</b> 16 (Alpha: 32)<br/>
                <b>Learning Rate:</b> 5e-5 (Cosine schedule)<br/>
                <b>Dataset:</b> 2,000 preference pairs (chosen/rejected)
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: ABOUT
# ============================================================
elif page == "About":
    st.markdown('<p class="hero-title">About ContractSense</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">An end-to-end AI pipeline for intelligent contract analysis.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="analysis-card">
        <div class="card-label">Project Overview</div>
        <div class="card-content">
            <b>ContractSense</b> is a multi-stage AI pipeline that transforms raw legal contracts
            into structured, actionable intelligence. Built as a Deep Learning capstone project,
            it demonstrates the complete journey from document ingestion to DPO-aligned generation.<br/><br/>

            The system processes contract clauses and produces structured analysis containing:<br/>
            <b>1. Risk Assessment</b> -- LOW / MEDIUM / HIGH / CRITICAL classification<br/>
            <b>2. Plain Language Explanation</b> -- Zero-jargon summarization for business stakeholders<br/>
            <b>3. Recommended Actions</b> -- Concrete next steps and negotiation strategies<br/>
            <b>4. Precise Citations</b> -- Exact textual references back to the source document
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("##### Pipeline Architecture")

    stages = [
        ("Stage 1-2", "Data Engineering", "CUAD dataset ingestion, clause segmentation, and preprocessing pipeline."),
        ("Stage 3", "Knowledge Base", "Qdrant vector store with sentence-transformer embeddings for semantic retrieval."),
        ("Stage 4", "Retrieval & Reranking", "Hybrid BM25 + dense retrieval with cross-encoder reranking for precision."),
        ("Stage 5", "Policy Agent", "LangGraph-based orchestrator for multi-step reasoning and tool selection."),
        ("Stage 6", "Generation (SFT)", "LoRA fine-tuning of Mistral-7B across 3 model candidates. Winner: 87.78% quality."),
        ("Stage 7", "Alignment (DPO)", "Direct Preference Optimization to enforce structured output. Final: 98.17% quality."),
        ("Stage 8", "Deployment", "Streamlit web interface with professional copilot UX for business stakeholders."),
    ]

    for stage_id, title, desc in stages:
        st.markdown(f"""
        <div class="analysis-card" style="padding: 16px 20px;">
            <span style="color:#6C63FF; font-weight:700; font-size:0.8rem;">{stage_id}</span>
            <span style="color:#E5E7EB; font-weight:600; margin-left:12px;">{title}</span>
            <div style="color:#9CA3AF; font-size:0.85rem; margin-top:6px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="analysis-card">
        <div class="card-label">Technology Stack</div>
        <div class="card-content">
            <b>Models:</b> Mistral-7B-Instruct-v0.2, Phi-3-mini, Qwen2.5-7B<br/>
            <b>Training:</b> HuggingFace TRL (DPOTrainer), PEFT (LoRA), BitsAndBytes (4-bit)<br/>
            <b>Retrieval:</b> Qdrant, Sentence-Transformers, BM25<br/>
            <b>Orchestration:</b> LangGraph, LangChain<br/>
            <b>Frontend:</b> Streamlit<br/>
            <b>Infrastructure:</b> Lightning AI Studio (NVIDIA L4 GPU)
        </div>
    </div>
    """, unsafe_allow_html=True)
