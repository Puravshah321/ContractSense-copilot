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

## 📊 Alignment Success & Baseline Comparison

We benchmarked the DPO-aligned model directly against the previous Stage 6 Base Generation model across a 120-sample holdout set. The results demonstrate massive improvements in structural reliability.

| Feature Dimension | Baseline (SFT) | DPO Aligned | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Quality** | 0.121 | 0.982 | **+86.1%** |
| **Actionability** | 0.000 | 1.000 | **+100%** |
| **Risk Salience** | 0.000 | 1.000 | **+100%** |
| **Format Compliance** | 0.000 | 1.000 | **+100%** |
| **Readability** | 0.8077 | 0.8780 | **+7.03%** |

### Failure Analysis
We conducted a strict audit of generation errors across the holdout dataset. Our DPO model successfully eliminated all critical formatting failures.

| Error Type | Baseline Occurrences | DPO Occurrences | Verdict |
| :--- | :---: | :---: | :--- |
| `no_action` | 200 | **0** | Eliminated |
| `missing_citation` | 200 | **0** | Eliminated |
| `missing_risk_label` | 200 | **0** | Eliminated |

---

## 🖼️ Dashboard & Artifacts

All training artifacts, multi-model comparisons, and failure heatmaps have been successfully pushed to the repository.

1. **Radar Fingerprints**: Visual comparison showing how the DPO model stretches out actionability and risk parameters compared to LoRA and Base Mistral (`Images/radar_fingerprint.png`).
2. **Error Distributions**: Confirmation of the complete elimination of formatting errors (`Images/error_distribution.png`).
3. **Training Curves**: Reward margin verification confirming proper convergence (`Images/training_curves.png`).

---

## 🔮 What's Next (Deployment)

Now that the generative AI "brain" is completely constructed, optimized, and strictly aligned to our custom schema, the execution moves formally to Interface Integration.

1. **End-to-End Orchestrator**: Connect the Retrieval stack -> Reranker -> Policy Agent -> DPO Generator into a single unified Python workflow.
2. **User Interface Construction**: Build the frontend application (Streamlit or React + FastAPI) allowing users to directly upload external PDFs and chat with the ContractSense Legal Copilot.
