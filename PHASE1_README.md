# ContractSense — Phase 1: Knowledge Base & Baseline Retrieval

This document outlines **Phase 1** of the **ContractSense** project (Enterprise Contract Risk Intelligence Copilot). 

The goal of this phase was to process raw enterprise contracts, chunk them into semantically meaningful clauses, and build a robust hybrid retrieval foundation (Sparse + Dense) to ground our future AI Copilot.

---

## 🎯 Phase 1 Objectives Complete
1. **Load CUAD Dataset:** Import 510 expert-labeled enterprise contracts.
2. **Clause-Aware Chunking:** Segment full contract text into individual clauses (23,564 clauses generated).
3. **Dense Vector Database:** Embed clauses using a Transformer model and store them with metadata in **Qdrant**.
4. **Sparse Baseline:** Implement an Okapi **BM25** baseline for exact-match keyword search and evaluation.

---

## 🏗️ Architecture & Modules

### 1. Data Ingestion & Chunking (`src/ingestion/clause_segmenter.py`)
- **Source Data:** The Contract Understanding Atticus Dataset (CUAD).
- **Chunking Logic:** Uses regex-based parsing to identify logical breaks in legal documents (e.g., `ARTICLE IV`, `SECTION 5.`).
- **Output:** Transforms raw PDFs/text into a cleanly structured `clauses.jsonl` file where each clause has identifying metadata (`contract_id`, `clause_id`).

### 2. Dense Semantic Retriever (`src/retrieval/embedder.py` & `vector_store.py`)
- **Embedder (`embedder.py`):** Uses Hugging Face's `sentence-transformers` (specifically `all-MiniLM-L6-v2`) to encode 23,564 text clauses into 384-dimensional dense vectors. 
- **Database (`vector_store.py`):** Utilizes **Qdrant** (running locally on-disk) as our vector database. 
- **Why Qdrant?** Qdrant allows us to store the vectors alongside rich JSON payloads. This enables complex query filtering, such as `"Search for 'liability caps' but ONLY within contract_id = 'train_00042'"`, which is a critical requirement for our agentic tools.

### 3. Sparse Lexical Baseline (`src/retrieval/bm25_retriever.py`)
- **BM25 Retriever:** Parses all 23,564 clauses (removing legal stop words and punctuation) to build a frequency-based Okapi BM25 Index.
- **Why BM25?** It serves as our **academic baseline**. In our final evaluation, we will demonstrate how the semantic understanding of our dense Retriever/Agent architecture outperforms this traditional keyword-based approach, specifically in resolving legal synonyms (e.g., "Evergreen Provision" vs. "Auto-Renewal").

---

## 📁 Repository Structure (Phase 1)

```text
ContractSense-copilot/
├── data/
│   ├── raw/
│   │   └── cuad/                      # Raw CUAD dataset
│   └── processed/
│       ├── clauses.jsonl              # 23,564 segmented clauses (Metadata + Text)
│       ├── clause_embeddings.npy      # Cached 384-d numpy arrays
│       ├── qdrant_local/              # Qdrant Database storage
│       └── bm25_index.pkl             # Persisted BM25 sparse index
├── notebooks/
│   └── 02_knowledge_base_build.ipynb  # End-to-end execution and testing notebook
├── src/
│   ├── ingestion/
│   │   └── clause_segmenter.py        # Logic for parsing and chunking contracts
│   └── retrieval/
│       ├── __init__.py
│       ├── embedder.py                # Uses all-MiniLM-L6-v2 to encode text
│       ├── vector_store.py            # Qdrant DB wrapper for upsert & search
│       └── bm25_retriever.py          # Baseline BM25 implementation
└── requirements.txt                   # Includes qdrant-client, rank-bm25, sentence-transformers
```

*(Note: Large index files and `.npy` arrays are excluded from Git via `.gitignore` to preserve repository hygiene).*

---

## 🚀 How to Run Phase 1

All Phase 1 steps have been consolidated into a single Jupyter Notebook. 

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the Build Step**
Open and run all cells in: `notebooks/02_knowledge_base_build.ipynb`

This notebook will automatically:
1. Load the chunked `clauses.jsonl` file.
2. Embed all 23,000+ clauses using `sentence-transformers` (cached for speed on subsequent runs).
3. Initialize the local Qdrant collection and upsert the vectors with their payloads.
4. Build and save the BM25 Index.
5. Run a **Diagnostic Side-by-Side Test** pitting the Dense Retriever against the BM25 Retriever to log baseline comparisons.

---

## ⏭️ Next Steps (Phase 2)
With the Knowledge Base foundation complete, the team will proceed to Phase 2:
- Implementing the Tool Schema (e.g., `SearchContract`, `GetClauseRiskProfile`).
- Interleaving Reasoning and Action (ReAct loop).
- Fine-tuning the Retriever (`legal-bert`) and Generator (LoRA on `Mistral-7B`).
