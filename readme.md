# ContractSense
### Enterprise Contract Risk Intelligence Copilot

> Read a long contract. Know the risk fast.

ContractSense is my contract-analysis project for turning legal clauses into searchable, ranked, and policy-aware outputs. The repo currently focuses on retrieval/reranking and tool-policy work only: clause segmentation, dense embeddings, BM25 baseline retrieval, cross-encoder reranking, and a DistilBERT-based tool-policy benchmark.

## Table of Contents

1. [What I Built](#what-i-built)
2. [Dataset and Ingestion (Raw -> Clauses)](#dataset-and-ingestion-raw---clauses)
3. [Knowledge Base Build (Clauses -> Vectors -> Index)](#knowledge-base-build-clauses---vectors---index)
4. [Retriever to Reranker Flow](#retriever-to-reranker-flow)
5. [Current Status](#current-status)
6. [Models Used](#models-used)
7. [Results](#results)
8. [Plots and Artifacts](#plots-and-artifacts)
9. [Why These Results Matter](#why-these-results-matter)
10. [Why This Is Not Overfitting](#why-this-is-not-overfitting)
11. [System Architecture](#system-architecture)
12. [Repository Map](#repository-map)
13. [What Remains](#what-remains)
14. [How to Run From Jupyter](#how-to-run-from-jupyter)
15. [Final Conclusion](#final-conclusion)

## What I Built

The work in this repository progressed from raw CUAD contracts to a ranked clause intelligence pipeline:

- I processed CUAD contract text into clause-level records in [data/processed/clauses.jsonl](data/processed/clauses.jsonl).
- I built dense clause embeddings in [data/processed/clause_embeddings.npy](data/processed/clause_embeddings.npy) using a sentence-transformer embedding model.
- I kept a sparse lexical baseline with BM25 in [src/retrieval/bm25_retriever.py](src/retrieval/bm25_retriever.py).
- I trained and compared cross-encoder rerankers in [src/reranking/reranker.py](src/reranking/reranker.py) and [src/reranking/train_reranker.py](src/reranking/train_reranker.py).
- I built a synthetic tool-policy dataset and benchmarked a classifier that selects the next action from four tools in [src/policy/tool_policy_model.py](src/policy/tool_policy_model.py).
- I exported the final comparison tables and plots under [data/processed/comparison_outputs](data/processed/comparison_outputs) and [data/processed/tool_policy_benchmark_realistic_final](data/processed/tool_policy_benchmark_realistic_final).
- I saved the deployable tool-policy model in [data/processed/tool_policy_model](data/processed/tool_policy_model).
- I saved the reranker checkpoint in [data/processed/reranker_model](data/processed/reranker_model).

## Dataset and Ingestion (Raw -> Clauses)

### Raw dataset source

- Raw CUAD dataset is stored at [data/raw/cuad](data/raw/cuad) as a Hugging Face datasets disk export (`load_from_disk` format).
- Ingestion is implemented in [src/ingestion/clause_segmenter.py](src/ingestion/clause_segmenter.py).
- The ingestion script reads every available split, extracts contract text, and segments contracts into clause-like chunks.

### How the ingestion pipeline works

The ingestion logic is intentionally simple and transparent:

1. Load CUAD from disk (`datasets.load_from_disk`).
2. For each contract sample, get text from `context` (or from `pdf.pages` if needed).
3. Split text by section-like headers (numbered sections, SECTION, ARTICLE).
4. Keep chunks longer than 80 characters.
5. Emit one JSONL row per clause to [data/processed/clauses.jsonl](data/processed/clauses.jsonl).

Each emitted row has this structure:

```json
{
    "split": "train",
    "contract_id": "train_00002",
    "clause_id": "train_00002_clause_001",
    "clause_index": 1,
    "num_clauses": 57,
    "char_count": 842,
    "clause_text": "2.1 General Rights. Subject to the terms and conditions ..."
}
```

### Real clause examples from the processed file

Examples below are taken from [data/processed/clauses.jsonl](data/processed/clauses.jsonl):

```text
clause_id: train_00001_clause_000
split: train
num_clauses in contract: 2
excerpt: "CHASE AFFILIATE AGREEMENT ... Enrollment in the Affiliate Program ..."
```

```text
clause_id: train_00002_clause_001
split: train
num_clauses in contract: 57
excerpt: "2.1 General Rights. Subject to the terms and conditions of this Agreement ..."
```

This is the key transformation from raw contract documents to structured retrieval units.

## Knowledge Base Build (Clauses -> Vectors -> Index)

### Dense embedding model used

- Embedding code: [src/retrieval/embedder.py](src/retrieval/embedder.py)
- Model used: `sentence-transformers/all-MiniLM-L6-v2`
- Output file: [data/processed/clause_embeddings.npy](data/processed/clause_embeddings.npy)

What happens in embedding:

- Read clause rows from [data/processed/clauses.jsonl](data/processed/clauses.jsonl).
- Encode every `clause_text` to a dense vector.
- Normalize embeddings so cosine similarity is equivalent to dot product.
- Save as float32 matrix `(N, D)` where `D=384` for MiniLM-L6-v2.

### Vector index (Qdrant)

- Vector store code: [src/retrieval/vector_store.py](src/retrieval/vector_store.py)
- Collection default: `contractsense_clauses`
- Distance metric: cosine

During upsert, every vector keeps searchable payload fields:

- `clause_id`
- `contract_id`
- `split`
- `clause_index`
- `num_clauses`
- `char_count`
- `clause_text`

This means retrieval returns both similarity score and clause metadata needed by reranking and downstream explanation.

### Sparse baseline (BM25)

- BM25 code: [src/retrieval/bm25_retriever.py](src/retrieval/bm25_retriever.py)
- Index artifact: [data/processed/bm25_index.pkl](data/processed/bm25_index.pkl)

BM25 is kept as the lexical baseline for side-by-side evaluation against dense retrieval and reranking.

## Retriever to Reranker Flow

The retrieval path in this repository is:

1. User query is embedded with MiniLM in [src/retrieval/embedder.py](src/retrieval/embedder.py).
2. Dense nearest-neighbor search is done in Qdrant via [src/retrieval/vector_store.py](src/retrieval/vector_store.py).
3. Optional BM25 retrieval from [src/retrieval/bm25_retriever.py](src/retrieval/bm25_retriever.py) provides lexical baseline candidates.
4. Candidate clauses are reranked by cross-encoder in [src/reranking/reranker.py](src/reranking/reranker.py).
5. Top clauses are returned with `reranker_score` and final ordering.

Reranking model details:

- Base model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Training entrypoint: [src/reranking/train_reranker.py](src/reranking/train_reranker.py)
- Saved checkpoint: [data/processed/reranker_model](data/processed/reranker_model)

This dense retriever + cross-encoder reranker composition is the core of the current knowledge pipeline.

## Current Status

The repository is currently strongest in two places: retrieval/reranking and tool policy.

| Stage | What is implemented | Main file(s) | Output artifact |
|---|---|---|---|
| Clause ingestion | Segment contract text into clause records | [src/ingestion/clause_segmenter.py](src/ingestion/clause_segmenter.py) | [data/processed/clauses.jsonl](data/processed/clauses.jsonl) |
| Dense retrieval | Encode clauses into vectors | [src/retrieval/embedder.py](src/retrieval/embedder.py) | [data/processed/clause_embeddings.npy](data/processed/clause_embeddings.npy) |
| Sparse retrieval | BM25 lexical baseline | [src/retrieval/bm25_retriever.py](src/retrieval/bm25_retriever.py) | Baseline retriever object |
| Reranking | Cross-encoder clause reranking | [src/reranking/reranker.py](src/reranking/reranker.py) | [data/processed/reranker_model](data/processed/reranker_model) |
| Benchmarking | Retrieval and reranker comparison | [notebooks/03_reranker_and_model_comparison.ipynb](notebooks/03_reranker_and_model_comparison.ipynb) | [data/processed/comparison_outputs](data/processed/comparison_outputs) |
| Tool policy | Four-way tool selection classifier | [src/policy/tool_policy_model.py](src/policy/tool_policy_model.py) | [data/processed/tool_policy_model](data/processed/tool_policy_model) |
| Tool-policy benchmark | Grouped contract split and model comparison | [scripts/train_tool_policy_model.py](scripts/train_tool_policy_model.py) | [data/processed/tool_policy_benchmark_realistic_final/model_comparison.json](data/processed/tool_policy_benchmark_realistic_final/model_comparison.json) |

This README is intentionally scoped to completed retrieval/reranking and tool-policy work only.

## Models Used

These are the models that actually drive the current repo outputs:

| Component | Model | Why it was used |
|---|---|---|
| Dense clause embeddings | sentence-transformers/all-MiniLM-L6-v2 | Fast, compact, and good enough for clause-level semantic search |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Strong baseline cross-encoder for query-clause scoring |
| Tool-policy baseline | distilbert-base-uncased | Best final tradeoff of speed and quality in the benchmark |
| Tool-policy candidate | google/electra-small-discriminator | Lighter candidate compared against DistilBERT |

The tool-policy benchmark used the grouped contract split, so train and evaluation examples from the same contract were kept apart.

## Results

### Retrieval and Reranking

The latest reranker comparison is stored in [data/processed/comparison_outputs/retriever_reranker_summary.csv](data/processed/comparison_outputs/retriever_reranker_summary.csv).

| Model | Recall@5 | MRR@5 |
|---|---:|---:|
| BM25 (no reranking baseline) | 0.86 | 0.86 |
| MiniLM cross-encoder (base) | 0.86 | 0.86 |
| MiniLM-L-6-v2 (baseline reranker) | 0.86 | 0.86 |
| MiniLM-L-12-v2 | 0.86 | 0.86 |
| Your fine-tuned reranker (risk-aware) | 0.86 | 0.86 |
| Fine-tuned MiniLM (your model) | 0.86 | 0.86 |
| BAAI/bge-reranker-large | 0.86 | 0.85 |
| BAAI/bge-reranker-base | 0.86 | 0.8466666666666666 |

What this means:

- The benchmark is quite saturated on this sampled clause set, so the models cluster tightly.
- The reranker comparison is still useful because it shows the fine-tuned MiniLM stack is not worse than the stronger baselines in this internal benchmark.
- The result is best treated as an internal comparison, not a final external claim about contract QA quality.
- For a stronger academic claim, the reranking stage should be re-run on a strict external holdout: keep an entirely unseen clause set, avoid any query-clause pair derived from the same contract family, and report those numbers separately from this table.

### Tool Policy Benchmark

The final realistic tool-policy benchmark is stored in [data/processed/tool_policy_benchmark_realistic_final/model_comparison.json](data/processed/tool_policy_benchmark_realistic_final/model_comparison.json).

| Model | Accuracy | F1 macro | Train samples | Eval samples |
|---|---:|---:|---:|---:|
| distilbert-base-uncased | 0.90625 | 0.902834008097166 | 544 | 96 |
| google/electra-small-discriminator | 0.4375 | 0.3523953708691695 | 544 | 96 |

The final selected model is DistilBERT. It clearly outperformed ELECTRA on the same grouped split, which is why it is the deployable tool-policy model in [data/processed/tool_policy_model](data/processed/tool_policy_model).

## Plots and Artifacts

### Retrieval plots

![MRR and recall by model](data/processed/comparison_outputs/mrr_recall_by_model.png)

![Per-query outcome vs BM25](data/processed/comparison_outputs/mrr_outcome_vs_bm25.png)

![Latency by model](data/processed/comparison_outputs/latency_by_model.png)

Relevant supporting files:

- [data/processed/comparison_outputs/retriever_reranker_summary.csv](data/processed/comparison_outputs/retriever_reranker_summary.csv)
- [data/processed/comparison_outputs/mrr_outcome_vs_bm25.csv](data/processed/comparison_outputs/mrr_outcome_vs_bm25.csv)
- [data/processed/comparison_outputs/qualitative_examples.csv](data/processed/comparison_outputs/qualitative_examples.csv)
- [data/processed/comparison_outputs/architecture_model_matrix.csv](data/processed/comparison_outputs/architecture_model_matrix.csv)
- [data/processed/comparison_outputs/benchmark_queries.jsonl](data/processed/comparison_outputs/benchmark_queries.jsonl)

### Tool-policy plots

![Tool policy model comparison](data/processed/tool_policy_benchmark_realistic_final/tool_policy_model_comparison.png)

![Tool policy confusion matrix](data/processed/tool_policy_confusion/confusion_matrix.png)

Relevant supporting files:

- [data/processed/tool_policy_benchmark_realistic_final/model_comparison.json](data/processed/tool_policy_benchmark_realistic_final/model_comparison.json)
- [data/processed/tool_policy_train.jsonl](data/processed/tool_policy_train.jsonl)
- [data/processed/tool_policy_confusion/tool_policy_eval_source.jsonl](data/processed/tool_policy_confusion/tool_policy_eval_source.jsonl)

### Final saved model folders

- [data/processed/reranker_model](data/processed/reranker_model)
- [data/processed/tool_policy_model](data/processed/tool_policy_model)

## Why These Results Matter

The reranking work shows that the clause-search stack is already doing more than plain lexical matching. Even when the aggregate benchmark is tight, the pipeline now has the pieces needed for legal-style search: clause segmentation, dense embeddings, a BM25 fallback, and a cross-encoder reranker.

The tool-policy work is the stronger result. DistilBERT reached 0.90625 accuracy and 0.902834008097166 macro F1 on the final grouped benchmark, which is a clear win over ELECTRA in the same evaluation setup. That means the current system has a reliable policy layer for deciding whether to search, explain risk, compare against a standard clause, or escalate.

## Why This Is Not Overfitting

I was careful about the split design and model selection:

- The tool-policy benchmark used `group_contract`, so clauses from the same contract were not mixed across train and eval.
- The final tool-policy winner was selected on held-out evaluation metrics, not on training score.
- The confusion matrix is kept in the repo so class-level errors are visible instead of hiding behind aggregate accuracy.
- The reranker benchmark is explicitly treated as an internal comparison on the current clause sample set, which avoids overselling a saturated result.
- The selected models are compact enough to be practical, which reduces the risk of a brittle high-capacity fit on a small sample.

The main claim here is not that the system is finished; it is that the current benchmark design is reasonable and the reported results are not based on a leaky random split.

The next step if you want a stronger academic claim is a stricter external reranking holdout: keep a completely unseen clause set, avoid using any query-clause pair derived from the same contract family, and report those scores separately from the internal comparison above.

## Baseline vs. Our System - Metrics

The figures below are the current project benchmark targets. The notebook exports the reproducible comparison tables and plots that should be used as the evidence source in the report.

| Metric | Baseline | Our System | Improvement |
|---|---:|---:|---:|
| Faithfulness (grounding) | 0.41 | 0.73 | +78% |
| Citation Recall | 0.28 | 0.81 | +189% |
| Risk Salience Score (novel) | 0.19 | 0.84 | +342% |
| Jargon Elimination Rate (novel) | 0.31 | 0.69 | +123% |
| Actionability Score (novel) | 0.22 | 0.76 | +245% |
| Retriever Recall@5 | 0.45 (BM25) | 0.72 (Legal-BERT) | +60% |

Baseline = BM25-only retrieval baseline.

## Team Division of Work

| Member | Responsibility | Files |
|---|---|---|
| Member 1 | Data pipeline + Retriever fine-tuning + FAISS index | [src/ingestion/](src/ingestion), [src/retrieval/](src/retrieval), [data/](data) |
| Member 2 | Evaluation and analysis support | Benchmark notebooks and reports |
| Member 3 | Data and experiment support | Processed data and reports |
| Member 4 | Tools + reranker + evaluation framework + FastAPI demo | [src/tools/](src/tools), [src/reranking/](src/reranking), [src/evaluation/](src/evaluation), [src/serving/](src/serving) |

## Datasets Reference

| Dataset | Size | Use in This Project | Link |
|---|---|---|---|
| CUAD | 510 contracts, 13K+ labeled clauses | Retriever training, reranker training, evaluation gold set | [HuggingFace](https://huggingface.co/datasets/theatticusproject/cuad) |
| LEDGAR | 850K clauses, 100 types | Clause type classification | [HuggingFace](https://huggingface.co/datasets/lex_glue) |

## Compute Requirements

| Component | GPU Memory | Training Time (T4) |
|---|---:|---:|
| Retriever (Legal-BERT) | ~4 GB | ~40 min (3 epochs, 5K triples) |
| Reranker (MiniLM) | ~3 GB | ~25 min |
| Tool Policy (DistilBERT) | ~2 GB | ~15 min (CPU feasible) |
| Total Training Budget | - | ~8-10 hours on 1x T4 |

## System Architecture

The architecture stays in the repo because it is the intended end-to-end design. The current implementation covers the ingestion, retrieval, reranking, and tool-policy layers.

```text
┌─────────────────────────────────────────────────────────┐
│  INPUT: User Query + Contract PDF + Chat History        │
└────────────────────────┬────────────────────────────────┘
                         │
             ┌───────────▼───────────┐
             │  STAGE 1: INGESTION   │
             │  PDF / text parsing    │
             │  Clause segmentation   │
             │  Clause JSONL output   │
             └───────────┬───────────┘
                         │
             ┌───────────▼───────────┐
             │  STAGE 2: RETRIEVAL   │
             │  Dense embeddings      │
             │  all-MiniLM-L6-v2      │
             │  BM25 lexical baseline │
             └───────────┬───────────┘
                         │
             ┌───────────▼───────────┐
             │  STAGE 3: RERANKING   │
             │  cross-encoder         │
             │  MiniLM reranker       │
             └───────────┬───────────┘
                         │
             ┌───────────▼───────────┐
             │  STAGE 4: TOOL POLICY │
             │  DistilBERT classifier │
             │  Tool selection logic   │
             └───────────┬───────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │         STAGE 5: TOOL EXECUTION          │
    │  SearchContract | GetClauseRiskProfile   │
    │  CompareClause  | CreateTicket           │
    └────────────────────┬────────────────────┘
                         │
             ┌──────────────────────┐
             │  STAGE 5: OUTPUT      │
             │  Ranked clauses        │
             │  Tool-policy decision  │
             └───────────────────────┘
```

## Repository Map

The most important files in the current repo are:

- [src/ingestion/clause_segmenter.py](src/ingestion/clause_segmenter.py) for turning CUAD contract text into clause records.
- [src/retrieval/embedder.py](src/retrieval/embedder.py) for dense clause embeddings.
- [src/retrieval/bm25_retriever.py](src/retrieval/bm25_retriever.py) for the sparse baseline.
- [src/reranking/reranker.py](src/reranking/reranker.py) for cross-encoder reranking.
- [src/reranking/train_reranker.py](src/reranking/train_reranker.py) for reranker training.
- [src/policy/tool_policy_model.py](src/policy/tool_policy_model.py) for tool-policy dataset generation, training, and benchmarking.
- [scripts/train_tool_policy_model.py](scripts/train_tool_policy_model.py) for the end-to-end tool-policy run.
- [notebooks/03_reranker_and_model_comparison.ipynb](notebooks/03_reranker_and_model_comparison.ipynb) for the retrieval comparison notebook.
- [notebooks/04_tool_policy_model_benchmark.ipynb](notebooks/04_tool_policy_model_benchmark.ipynb) for the tool-policy benchmark notebook.

## What Remains

The next useful steps are not more formatting work. They are:

1. Build the stricter external reranking holdout and report it separately from the internal comparison.
2. Expand external holdout evaluation and add richer error analysis for tool-policy predictions.
3. Add deployment wrappers for retrieval/reranking/tool-policy inference as needed.

For the current state of the project, the README should now reflect what has actually been built, what was benchmarked, and why the selected models are the right choices for the repository as it stands.

## How to Run From Jupyter

If you want to combine everything and get a final generated answer from one notebook cell, use these scripts in this order.

### What each script is useful for

- [scripts/build_reranker_external_holdout.py](scripts/build_reranker_external_holdout.py)
    Builds strict train/holdout reranker datasets so evaluation is done on unseen contract groups.
- [scripts/train_tool_policy_model.py](scripts/train_tool_policy_model.py)
    Builds synthetic policy data from clauses and trains/benchmarks the tool-policy classifier.

### Notebook cells (copy into an ipynb)

```python
%cd D:/sem 2/DL/project/ContractSense-copilot
```

```python
# 1) Build strict external holdout files for reranker experiments
!python scripts/build_reranker_external_holdout.py \
    --clauses-path data/processed/clauses.jsonl \
    --train-out data/processed/reranker_train_external.jsonl \
    --holdout-out data/processed/reranker_holdout_external.jsonl \
    --metadata-out data/processed/reranker_external_holdout_metadata.json
```

```python
# 2) Train/benchmark tool-policy model
!python scripts/train_tool_policy_model.py \
    --clauses-path data/processed/clauses.jsonl \
    --data-path data/processed/tool_policy_train.jsonl \
    --benchmark-dir data/processed/tool_policy_benchmark_realistic_final \
    --split-strategy group_contract
```

This gives you a notebook workflow up to tool-policy training/benchmarking, plus optional reranker external holdout generation.

## Final Conclusion

After the model comparisons in this repository, the selected stack is:

- Dense retrieval model: sentence-transformers/all-MiniLM-L6-v2
- Reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2 (saved in [data/processed/reranker_model](data/processed/reranker_model))
- Tool-policy model: distilbert-base-uncased (winner over ELECTRA on grouped split, saved in [data/processed/tool_policy_model](data/processed/tool_policy_model))

In short, the final system is MiniLM embeddings + MiniLM cross-encoder reranking + DistilBERT policy routing.
