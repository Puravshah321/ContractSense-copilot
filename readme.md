# ContractSense: DPO Alignment (Direct Preference Optimization)

## Overview
This branch encapsulates the **Stage 7** Direct Preference Optimization (DPO) pipeline for ContractSense. Our primary objective was to align our generative LLM (`Mistral-7B-Instruct-v0.2 + LoRA`) to output highly structured, human-centric legal analysis. 

Rather than standard conversational responses, the aligned model is strictly trained to enforce a standardized schema:
1. **Risk Salience** (Identifying LOW/MEDIUM/HIGH/CRITICAL risk).
2. **Plain Explanation** (Zero-jargon summarization).
3. **Actionability** (Explicit next steps or recommendations).
4. **Citation Validation** (Extracting precise textual spans).

---

## 🚀 What We Did (The Pipeline)

1. **DPO Pair Construction**: 
   We converted the raw, unaligned generator text into explicit `chosen` (perfectly formatted) versus `rejected` (dense, unformatted language) pairings.
2. **Cloud Scalability**: 
   We migrated the execution suite to **Lightning AI Studio** running an **NVIDIA L4 GPU (24GB VRAM)**. To maximize efficiency, we engineered the environment for batch-density throughput and switched to pure `bfloat16` with native PyTorch Scaled Dot-Product Attention (SDPA).
3. **TRL Model Optimization**: 
   We leveraged the HuggingFace `trl` (`DPOTrainer`) library natively. The DPO pipeline optimized our custom adapter weights purely based on preference margins, achieving extremely stable convergence with a final loss of `0.0064` in just under an hour.

---

## 📊 Alignment Success & Stage 6 Comparison

We benchmarked the DPO-aligned model directly against our **Stage 6 Winner** (`Mistral-7B-Instruct-v0.2 + LoRA SFT`). While Stage 6 had already achieved excellent structured output, DPO provided the final algorithmic push needed to achieve near-perfect structural reliability and completely eliminate outlier generation failures.

| Feature Dimension | Stage 6 Winner (LoRA SFT) | Stage 7 Winner (DPO Aligned) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Quality** | 0.877 | 0.982 | **+10.5%** |
| **Actionability** | 0.925 | 1.000 | **+7.5%** |
| **Risk Salience** | 0.875 | 1.000 | **+12.5%** |
| **Format Compliance** | 0.958 | 1.000 | **+4.2%** |

### Failure Analysis / Outlier Elimination
While Stage 6 was highly accurate, it still occasionally dropped JSON keys or forgot to pull exact citations in edge-case documents. DPO strictly penalized these behaviors, completely eliminating unformatted edge cases in the holdout evaluations.

| Error Type | Stage 6 Edge Cases | DPO Occurrences | Verdict |
| :--- | :---: | :---: | :--- |
| `no_action` | Occasional | **0** | Eliminated |
| `missing_citation` | Occasional | **0** | Eliminated |
| `missing_risk_label` | Occasional | **0** | Eliminated |

---

## 🖼️ Dashboard & Artifacts

All training artifacts, multi-model comparisons, and failure heatmaps have been successfully pushed to the repository.

1. **Performance Jump (LoRA vs DPO)**: Shows the massive absolute improvement pushed by DPO using our verified SFT LoRA benchmark. (`Images/true_metrics_comparison.png`)
2. **Error Elimination**: Demonstration of how DPO takes the handful of formatting errors that still existed in Stage 6 and completely zeros them out (`Images/true_error_elimination.png`).
3. **Training Curves**: Reward margin verification confirming proper convergence (`Images/training_curves.png`).

---

## 🔮 What's Next (Deployment)

Now that the generative AI "brain" is completely constructed, optimized, and strictly aligned to our custom schema, the execution moves formally to Interface Integration.

1. **End-to-End Orchestrator**: Connect the Retrieval stack -> Reranker -> Policy Agent -> DPO Generator into a single unified Python workflow.
2. **User Interface Construction**: Build the frontend application (Streamlit or React + FastAPI) allowing users to directly upload external PDFs and chat with the ContractSense Legal Copilot.
