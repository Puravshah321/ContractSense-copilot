# ⚖️ ContractSense Copilot  
### A Trustworthy AI System for Grounded Contract Understanding and Legal Decision Support

<p align="center">

<img src="https://img.shields.io/badge/Domain-Legal%20AI-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Architecture-Hybrid%20RAG-success?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Alignment-DPO-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Training-Lightning%20AI-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LLM-Mistral%207B-important?style=for-the-badge"/>

</p>

---

# Project Overview

ContractSense Copilot is an AI-powered legal assistant designed to analyze, retrieve, and reason over legal contracts using evidence-grounded language modeling.

Traditional legal AI systems primarily focus on generating fluent responses. However, in legal environments, fluent responses alone are not sufficient. A legally incorrect or hallucinated answer can create serious risks.

ContractSense addresses this problem through a retrieval and verification-driven architecture that prioritizes:

- grounded legal reasoning,
- evidence-supported responses,
- hallucination reduction,
- refusal handling,
- and decision-aware AI behavior.

The system combines:

- Hybrid Retrieval-Augmented Generation (Hybrid RAG)
- Semantic Legal Retrieval
- Legal-Aware Reranking
- Grounding Verification
- DPO-Based Alignment
- Lightning AI Training Infrastructure

to create a more reliable legal intelligence system.

---

# Problem Statement

Legal contracts are:

- lengthy,
- highly structured,
- domain-specific,
- and difficult to interpret manually.

Although Large Language Models can summarize and answer questions about contracts, they frequently suffer from:

- hallucinated legal interpretations,
- unsupported claims,
- incorrect clause references,
- overconfident responses,
- and lack of transparency.

Most legal chatbots attempt to answer every query regardless of whether reliable evidence exists.

ContractSense was developed to solve this issue by creating an AI system that can:

- retrieve legal evidence  
- verify grounding  
- detect insufficient evidence  
- refuse unsupported claims  
- escalate uncertain cases safely  

---

# Core Novelty

### Decision-Aware Grounded Legal Reasoning

The primary innovation of ContractSense is the introduction of a decision-aware legal AI pipeline.

Unlike traditional legal RAG systems that always attempt to generate responses, ContractSense evaluates whether sufficient contractual evidence exists before answering.

The system intelligently decides between:

```text
ANSWER
NOT_FOUND
ESCALATE
```

This transforms the model from a simple legal chatbot into a trustworthy legal decision-support system.

---

# Key Contributions

## 1. Evidence-Grounded Hybrid Retrieval

The project combines:

- TF-IDF Sparse Retrieval
- Dense Semantic Retrieval
- Reciprocal Rank Fusion (RRF)
- Legal-aware reranking

to improve retrieval precision and reduce incorrect evidence selection.

---

## 2. DPO-Based Refusal & Escalation Learning

The model is trained using Direct Preference Optimization (DPO) to learn:

- grounded answering,
- refusal behavior,
- hallucination prevention,
- contradiction detection,
- and escalation handling.

Instead of optimizing only for fluent responses, the model is aligned toward safer legal reasoning.

---

## 3. Grounding Verification Layer

A dedicated verification stage checks whether generated responses are actually supported by retrieved evidence.

If sufficient evidence is not available, the model safely returns:

```text
NOT_FOUND
```

or

```text
ESCALATE
```

instead of hallucinating legal information.

---

## 4. Semantic + Structural Legal Chunking

The system performs:

- heading-aware chunking,
- clause-preserving segmentation,
- section-aware splitting,
- sentence-level chunking.

This preserves legal meaning and improves retrieval quality.

---

## 5. Lightning AI-Based Distributed Training

The DPO alignment pipeline was trained using **Lightning AI**, enabling:

- scalable experimentation,
- efficient GPU utilization,
- modular training workflows,
- reproducible model training,
- and streamlined evaluation pipelines.

Lightning AI was used to orchestrate:

- LoRA fine-tuning,
- DPO preference optimization,
- evaluation tracking,
- and model checkpointing.

---

# Complete System Workflow

```text
                         ┌────────────────────┐
                         │  Contract Document │
                         │    PDF / TXT File  │
                         └─────────┬──────────┘
                                   │
                                   ▼
                 ┌────────────────────────────────┐
                 │ Document Parsing & Cleaning    │
                 └────────────────┬───────────────┘
                                  │
                                  ▼
                 ┌────────────────────────────────┐
                 │ Semantic Legal Chunking        │
                 │ Section + Clause Preservation  │
                 └────────────────┬───────────────┘
                                  │
                                  ▼
                 ┌────────────────────────────────┐
                 │ Hybrid Retrieval Engine        │
                 │ TF-IDF + Dense Embeddings      │
                 └────────────────┬───────────────┘
                                  │
                                  ▼
                 ┌────────────────────────────────┐
                 │ Reciprocal Rank Fusion (RRF)   │
                 │ + Legal-Aware Reranking        │
                 └────────────────┬───────────────┘
                                  │
                                  ▼
                 ┌────────────────────────────────┐
                 │ Grounded LLM Response          │
                 │ Mistral-7B-Instruct-v0.2       │
                 └────────────────┬───────────────┘
                                  │
                                  ▼
                 ┌────────────────────────────────┐
                 │ Grounding Verification Layer   │
                 │ Evidence Sufficiency Check     │
                 └────────────────┬───────────────┘
                                  │
               ┌──────────────────┴──────────────────┐
               ▼                                     ▼
        ┌───────────────┐                   ┌────────────────┐
        │   ANSWER      │                   │ ESCALATE /     │
        │ Evidence Safe │                   │ NOT_FOUND      │
        └───────────────┘                   └────────────────┘
```

---

# Detailed Working Pipeline

## Step 1 — Contract Ingestion

The uploaded contract is parsed and converted into structured text while preserving:

- headings,
- legal clauses,
- section hierarchy,
- and page boundaries.

This ensures legal context remains intact throughout processing.

---

## Step 2 — Semantic Legal Chunking

The contract is split into evidence-grade chunks using:

- semantic segmentation,
- clause boundaries,
- heading-aware splitting,
- and sentence-aware chunking.

Each chunk stores metadata including:

| Metadata | Purpose |
|---|---|
| Clause ID | Legal clause reference |
| Section Name | Section identification |
| Page Number | Original document mapping |
| Token Count | Chunk sizing |
| Character Offsets | Traceability |

This creates retrieval-ready legal evidence units.

---

## Step 3 — Hybrid Retrieval

When a user asks a legal query, the system retrieves relevant evidence using two retrieval mechanisms.

### Sparse Retrieval

Uses:

- TF-IDF Vectorization
- Keyword overlap matching

to capture exact legal terminology.

---

### Dense Retrieval

Uses:

- sentence embeddings,
- semantic similarity search,

to capture conceptual meaning beyond keywords.

---

### Reciprocal Rank Fusion (RRF)

The sparse and dense rankings are combined using Reciprocal Rank Fusion to improve retrieval robustness and ranking quality.

---

## Step 4 — Legal-Aware Reranking

Retrieved chunks are reranked using legal heuristics such as:

- clause importance,
- legal keyword priorities,
- section relevance,
- and boilerplate penalties.

This prioritizes highly relevant legal evidence.

---

## Step 5 — Grounded Response Generation

The top-ranked evidence chunks are passed to the LLM along with the user query.

Base model used:

```python
Mistral-7B-Instruct-v0.2
```

The model generates:

- grounded legal answers,
- clause-aware summaries,
- evidence-supported explanations,
- and citation-oriented responses.

---

## Step 6 — Grounding Verification

Before returning the final output, the system verifies:

- evidence sufficiency,
- citation support,
- answer grounding,
- and contradiction presence.

If evidence is weak or missing, the system safely returns:

```text
NOT_FOUND
```

or

```text
ESCALATE
```

instead of hallucinating information.

---

# DPO Alignment Workflow

## What is DPO?

Direct Preference Optimization (DPO) is a preference alignment technique used to teach the model safer and more reliable legal reasoning behavior.

Instead of learning from only correct answers, the model learns from comparisons between:

| Preferred Output | Rejected Output |
|---|---|
| Grounded Answer | Hallucinated Answer |
| Safe Refusal | Unsupported Claim |
| Evidence-backed Response | Fabricated Interpretation |
| Escalation Behavior | Overconfident Generation |

---

## DPO Dataset Categories

The project includes multiple DPO dataset variants:

| Dataset | Purpose |
|---|---|
| DPO v2 | Grounding & hallucination prevention |
| DPO v3 | Multi-hop reasoning |
| DPO v4 | Diversity & anti-overfitting |

---

### Training Categories

#### A — Correct Grounding
Evidence-supported legal answers.

#### B — Hallucination Negatives
Fabricated or unsupported legal claims.

#### C — Absence Detection
Cases where evidence does not exist.

#### D — Partial Evidence
Queries requiring escalation due to insufficient support.

#### E — Contradiction & Multi-hop
Cross-clause legal reasoning and conflict detection.

---

# Lightning AI Training Pipeline

The complete DPO training workflow was implemented using **Lightning AI**.

Lightning AI enabled:

- distributed training,
- scalable experimentation,
- modular pipeline orchestration,
- automatic checkpointing,
- efficient GPU utilization,
- and reproducible workflows.

---

## Training Workflow

```text
DPO Dataset Creation
          │
          ▼
Preference Pair Generation
          │
          ▼
LoRA Fine-Tuning Setup
          │
          ▼
Lightning AI Trainer
          │
          ▼
DPO Optimization
          │
          ▼
Evaluation & Metrics
          │
          ▼
Model Checkpointing
          │
          ▼
Inference Deployment
```

---

# Training Configuration

| Component | Configuration |
|---|---|
| Base Model | Mistral-7B-Instruct-v0.2 |
| Alignment Method | DPO |
| Fine-Tuning | LoRA |
| Framework | Lightning AI |
| Quantization | 4-bit NF4 |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| DPO Beta | 0.15 |

---

# Evaluation Metrics

The system evaluates both retrieval quality and legal reasoning reliability.

| Metric | Purpose |
|---|---|
| Retrieval Accuracy | Correct evidence retrieval |
| Grounding Accuracy | Evidence-supported outputs |
| Hallucination Rate | Unsupported generated claims |
| Decision Accuracy | Correct ANSWER / ESCALATE behavior |
| Refusal Accuracy | Safe refusal capability |
| Intent Alignment Accuracy | Query understanding quality |
| Structure Match Accuracy | Output consistency |
| Concept Purity Score | Legal semantic relevance |

---

# Improvements Achieved

Compared to baseline legal retrieval systems, ContractSense demonstrates:

- Lower hallucination rates  
- Better evidence grounding  
- Safer refusal behavior  
- Improved retrieval precision  
- More reliable legal reasoning  
- Better escalation handling  
- Stronger decision alignment  

---

# Tech Stack

## AI & Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- Sentence Transformers
- Scikit-learn

---

## Training Infrastructure

- Lightning AI
- LoRA Fine-Tuning
- DPO Alignment
- Quantized Training

---

## Retrieval Pipeline

- TF-IDF Retrieval
- Dense Embeddings
- Hybrid Search
- Reciprocal Rank Fusion

---

## Application Layer

- Streamlit
- Interactive Legal Copilot UI

---

# Project Structure

```bash
ContractSense-copilot/
│
├── app/
│   ├── main_app.py
│   └── demo_data.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── grounded_dpo_model/
│
├── scripts/
│   ├── dpo_dataset_v2.py
│   ├── dpo_dataset_v3.py
│   ├── dpo_dataset_v4.py
│   ├── lightning_train_v2.py
│   ├── evaluate_model_comparison.py
│   └── evaluate_precision_pipeline.py
│
├── src/
│   ├── pipeline/
│   ├── retrieval/
│   ├── generation/
│   └── alignment/
│
└── README.md
```

---

# Real-World Impact

ContractSense addresses one of the most important challenges in legal AI:

> preventing hallucinated legal reasoning.

The system shifts the objective of legal language models from:

> generating the most fluent answer

to:

> generating the safest evidence-supported legal response.

This makes the project more aligned with the future direction of trustworthy AI systems.

---

# Future Scope

Future improvements may include:

- Multi-document legal comparison
- Human-in-the-loop legal review
- Real-time compliance analysis
- Enterprise legal workflows
- Advanced citation tracing
- Cross-contract contradiction analysis
- RLHF-based legal refinement

---

# Contributors

**Group 10**
- Sanjana Nathani
- Purav Shah
- Jay Salot
- Mahak Khurdia

M.Sc. Data Science
DAU, Gandhinagar

---

# 📖 Research Areas Explored

- Retrieval-Augmented Generation (RAG)
- Legal NLP
- Trustworthy AI
- Grounded Language Modeling
- Direct Preference Optimization (DPO)
- Hybrid Information Retrieval
- Hallucination Reduction

---

# 🏁 Conclusion

ContractSense Copilot is not designed to be just another legal chatbot.

It is designed as a trustworthy legal reasoning system that prioritizes:

- grounding,
- transparency,
- evidence verification,
- hallucination prevention,
- and responsible AI decision-making.

By combining hybrid retrieval, grounding verification, legal-aware reranking, DPO alignment, and Lightning AI-based scalable training, the project demonstrates how modern LLM systems can evolve beyond fluent generation toward safer and more reliable legal intelligence systems.

The central contribution of the project lies in teaching AI not only:

> how to answer,

but also:

> when not to answer.

---

<p align="center">

## ⚖️ ContractSense Copilot  
### Building Trustworthy and Safer Legal AI Systems

</p>