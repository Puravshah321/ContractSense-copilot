# ContractSense Semantic Reasoning Upgrade - Lightning AI L4

## What Was Added

- Query intent classifier: factual, analytical, comparative, risk.
- Answer type controller: yes/no, fact, list, risk table, comparison.
- Legal taxonomy classifier for retrieved clauses.
- Semantic post-retrieval filter and reranker.
- Structured synthesis for analytical queries such as financial commitments.
- DPO examples for semantic confusion: warranty vs remedies, audit vs financial obligation, term vs survival, sharing vs NOT_FOUND, AI-training restrictions.
- Baseline vs Generator vs DPO evaluation outputs.

## Lightning AI Hardware Profile

Target:

- GPU: NVIDIA L4, 24 GB VRAM
- CPU: 8 cores
- RAM: 24 GB

Training defaults in `scripts/lightning_train_v2.py`:

```bash
PER_DEVICE_BATCH_SIZE=4
GRAD_ACCUM=4
SAVE_STEPS=50
RESUME_FROM_CHECKPOINT=0
```

This gives effective batch size 16 with 4-bit NF4 quantization and paged 8-bit AdamW.

## Fresh Training Command

Use this when starting a clean run:

```bash
rm -rf grounded_dpo_model_v2
pip install -r requirements_pipeline.txt
huggingface-cli login
python scripts/lightning_train_v2.py
```

Use this only after an interrupted run:

```bash
RESUME_FROM_CHECKPOINT=1 python scripts/lightning_train_v2.py
```

## Files Required On Lightning AI

```text
scripts/lightning_train_v2.py
scripts/dpo_dataset_v2.py
scripts/evaluate_precision_pipeline.py
scripts/evaluate_model_comparison.py

src/pipeline/
  answer_controller.py
  chunker.py
  evidence_checker.py
  generator.py
  legal_ontology.py
  orchestrator.py
  query_understanding.py
  retriever.py
  semantic_filter.py
  verifier.py
  __init__.py

src/__init__.py
requirements_pipeline.txt
```

## Generated Outputs

After the Lightning run:

```text
grounded_dpo_model_v2/final/
grounded_dpo_model_v2/eval_results.json
grounded_dpo_model_v2/precision_pipeline_metrics.json
grounded_dpo_model_v2/images/dpo_eval_metrics.png
grounded_dpo_model_v2/images/dpo_hallucination_rate.png
grounded_dpo_model_v2/images/model_comparison_metrics.json
grounded_dpo_model_v2/images/model_comparison_metrics.csv
grounded_dpo_model_v2/images/baseline_generator_dpo_comparison.png
grounded_dpo_model_v2/images/hallucination_rate_comparison.png
```

The script pushes the merged model to:

```text
22Jay/ContractSense-Grounded-DPO
```

## Local Validation Snapshot

Current deterministic semantic regression suite:

```text
retrieval_accuracy: 100%
decision_accuracy: 100%
hallucination_rate: 0%
not_found_accuracy: 100%
average_grounding_ratio: 100%
```

Baseline vs Generator local comparison:

```text
baseline hallucination_rate: 42.86%
generator hallucination_rate: 0%
baseline grounding_accuracy: 71.43%
generator grounding_accuracy: 100%
```

The DPO column is filled automatically after `lightning_train_v2.py` finishes model evaluation.
