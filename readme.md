---
title: ContractSense-copilot
---

# ContractSense

Consolidated project README combining dataset description, usage, models, training, and references.

## Table of contents
- Project overview
- Dataset
- Installation
- Quick start / Running the demo
- Models (model cards)
- Training & evaluation
- Files of interest
- Contributing & license

## Project overview

This repository contains code, data, notebooks and models for ContractSense — a contract understanding and generation project with retrieval, reranking, and generation pipelines, plus DPO-aligned models and training scripts.

Key folders:
- `app/` — demo application and example data.
- `data/` — raw and processed datasets used across experiments.
- `src/` — project source (alignment, generation, ingestion, pipeline, etc.).
- `grounded_dpo_model/` — DPO datasets and related artifacts.
- `notebooks/` — exploratory and training notebooks.

## Dataset

Datasets used and produced are under `data/` and `grounded_dpo_model/`.

- Processed examples (examples):
  - `data/processed/clauses.jsonl` — clause-level JSONL lines used to build embeddings and retrieval indices.
  - `data/processed/clause_embeddings.npy` — precomputed clause embeddings.
  - `data/processed/generation_train.jsonl` and `generation_eval.jsonl` — generation training/eval data.
  - `data/processed/sample_contract_texts.jsonl` — sample documents.

- Raw sources:
  - `data/raw/` — original datasets (CUAD or others) used for ingestion.

- DPO datasets:
  - `grounded_dpo_model/dpo_dataset_v2.json` (and v3/v4) — preference / DPO datasets used for Direct Preference Optimization training and evaluation.

If you need a Dataset Card or more metadata, check `data/processed/` and the `scripts/` that build dataset artifacts (e.g., `scripts/dpo_dataset_*.py`).

## Installation

Recommended: use a Python virtual environment and install requirements.

Windows PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or with the bundled virtualenv `contractsense_env312` use its `activate` in `contractsense_env312/Scripts/`.

## Quick start / Running the demo

- Demo app entry: `app/main_app.py` — runs the demo interface.
- Demo data helper: `app/demo_data.py`.

Example to run the demo (after venv and deps installed):

```powershell
python app/main_app.py
```

Scripts of interest (examples):
- `scripts/train_generation_models.py` — training generation models
- `scripts/train_dpo.py` / `lightning_train_*.py` — training DPO / Lightning jobs
- `scripts/serve_dpo_api.py` — serve a local API for model inference

## Models (included model cards)

This repo contains model card README files under `src/alignment/models/dpo_aligned_model/` and its checkpoints. Key consolidated details:

Model: `dpo_aligned_model` (fine-tuned from `mistralai/Mistral-7B-Instruct-v0.2`)

Quick-start example (text-generation pipeline):

```python
from transformers import pipeline
question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="<path-or-hf-id>", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

Training notes (from model card):
- Trained with Direct Preference Optimization (DPO) using the TRL framework.
- Key frameworks / versions referenced:
  - PEFT 0.19.1
  - TRL 1.2.0
  - Transformers 5.5.4
  - PyTorch 2.8.0+cu128
  - Datasets 4.8.4

Citations (from model card):

Direct Preference Optimization:

```bibtex
@inproceedings{rafailov2023direct,
    title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
    author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
    year         = 2023,
    booktitle    = {NeurIPS 2023},
}
```

TRL citation (software):

```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```

Note: there are checkpoint-specific README placeholders at `src/alignment/models/dpo_aligned_model/checkpoint-*/README.md` with additional (mostly placeholder) model metadata.

## Training & evaluation

- Training scripts live under `scripts/` (see `lightning_train_*.py`, `train_dpo.py`, `train_generation_models.py`).
- Evaluation and benchmarking utilities: `scripts/evaluate_model_comparison.py`, `scripts/evaluate_precision_pipeline.py`.
- Notebooks in `notebooks/` provide step-throughs for knowledge base builds, reranker/model comparison, DPO alignment and generation-phase experiments.

## Files of interest
- `requirements.txt`, `requirements_training.txt`, `requirements_pipeline.txt` — dependency lists.
- `test_dpo_model.py` — small tests for the DPO model.
- `Images/` — figures and metrics JSON/CSV used for reports.

## Contributing

If you add new README content in subfolders, please also update this top-level `README.md` to keep documentation centralized.

## License & credits

Check project root for license information. Also many components reference model licenses (e.g., base model license) — verify before redistribution.

---

If you'd like, I can:
- expand any section with more detail pulled from specific notebooks or scripts,
- include inlined screenshots/images referenced by the sub-READMEs (I found no image links in the READMEs read),
- commit this `README.md` to git.
