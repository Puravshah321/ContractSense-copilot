# generator/model.py
"""
MAHAK'S GENERATOR - COMPLETE ADAPTIVE VERSION
- Auto-detects GPU (4GB vs 8GB+)
- SaulLM-7B for 8GB+ GPUs (Best quality)
- Phi-2 for 4GB GPUs (Good quality)
- Rule-based fallback (Instant, always works)
- FIXED: Keyword detection for indemnification and auto-renewal
"""

import sys
import json
import torch
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever
from generator.prompt import build_saulm_prompt, build_phi2_prompt


class AdaptiveContractGenerator:
    def __init__(self):
        self.bm25 = None
        self.model = None
        self.tokenizer = None
        self.clauses = []
        
        # Model flags
        self.use_saulm = False
        self.use_phi2 = False
        self.use_rule = True
        
    # =========================================================
    # AUTO-DETECT BEST MODEL BASED ON GPU
    # =========================================================
    def auto_detect_model(self):
        """Automatically choose best model based on available GPU"""
        
        print("\nAuto-detecting best model...")
        
        # First check if CUDA is available
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_mem:.1f} GB")
            
            if gpu_mem >= 8:
                self.use_saulm = True
                self.use_phi2 = False
                self.use_rule = False
                print("   Using SaulLM-7B (Best quality, 5-15 sec per query)")
                return "saulm"
            elif gpu_mem >= 4:
                self.use_saulm = False
                self.use_phi2 = True
                self.use_rule = False
                print("   Using Phi-2 (Good quality, 30-60 sec per query)")
                return "phi2"
            else:
                self.use_rule = True
                print("   Using Rule-based (Instant, lower quality)")
                return "rule"
        else:
            self.use_rule = True
            print("   No GPU detected. Using Rule-based (Instant)")
            return "rule"
    
    # =========================================================
    # LOAD BM25 RETRIEVER
    # =========================================================
    def load_retriever(self):
        """Load BM25 retriever with clauses"""
        print("\nLoading BM25 retriever...")
        
        clauses_path = Path("data/processed/clauses.jsonl")
        
        if not clauses_path.exists():
            print(f"File not found: {clauses_path}")
            return False
        
        self.clauses = []
        with open(clauses_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.clauses.append(json.loads(line))
        
        print(f"   Loaded {len(self.clauses)} clauses")
        self.bm25 = BM25Retriever(self.clauses)
        return True
    
    # =========================================================
    # LOAD SAULM-7B (For 8GB+ GPU - BEST QUALITY)
    # =========================================================
    def load_saulm(self):
        """Load SaulLM-7B-Instruct - Specialized for legal text"""
        print("\nLoading SaulLM-7B-Instruct...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # 4-bit quantization for 8GB GPU
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "JohnSnowLabs/SaulLM-7B-Instruct",
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "JohnSnowLabs/SaulLM-7B-Instruct",
                trust_remote_code=True,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("   SaulLM-7B loaded!")
            return True
            
        except Exception as e:
            print(f"   SaulLM failed: {e}")
            print("   Falling back to Rule-based")
            self.use_saulm = False
            self.use_rule = True
            return False
    
    # =========================================================
    # LOAD PHI-2 (For 4GB GPU)
    # =========================================================
    def load_phi2(self):
        """Load Phi-2 - Fits in 4GB GPU"""
        print("\nLoading Phi-2...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # 4-bit quantization for 4GB GPU
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/phi-2",
                trust_remote_code=True,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("   Phi-2 loaded!")
            return True
            
        except Exception as e:
            print(f"   Phi-2 failed: {e}")
            print("   Falling back to Rule-based")
            self.use_phi2 = False
            self.use_rule = True
            return False
    
    # =========================================================
    # RETRIEVE CLAUSES
    # =========================================================
    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve relevant clauses"""
        results = self.bm25.search(query, top_k=top_k)
        
        clauses = []
        for r in results:
            clauses.append({
                "clause_id": r.get("clause_id", "unknown"),
                "clause_text": r.get("clause_text", ""),
                "score": r.get("bm25_score", 0)
            })
        return clauses
    
    # =========================================================
    # RULE-BASED ANALYSIS (Instant - FIXED VERSION)
    # =========================================================
    def rule_based_analysis(self, query: str, clauses: list) -> dict:
        """Fast rule-based analysis - INSTANT with FIXED keyword detection"""
        
        if not clauses:
            return {
                "decision": "ESCALATE",
                "risk": "MEDIUM",
                "explanation": "No relevant clauses found. Please consult legal team.",
                "citation": "N/A"
            }
        
        text = clauses[0]["clause_text"].lower()
        clause_id = clauses[0]["clause_id"]
        
        # Debug: Uncomment to see what's being analyzed
        # print(f"   Debug - Clause text: {text[:150]}...")
        
        # ==============================================
        # HIGH RISK CLAUSES - FIXED KEYWORDS
        # ==============================================
        
        # Indemnification (HIGH RISK)
        indemnify_keywords = ["indemnif", "hold harmless", "indemnity", "defend", "reimburse"]
        if any(keyword in text for keyword in indemnify_keywords):
            return {
                "decision": "REVIEW",
                "risk": "HIGH",
                "explanation": f"HIGH RISK: This indemnification clause transfers significant legal and financial liability to your company. You may be required to defend and compensate the vendor for claims arising from your breach or use of their services. Legal review strongly recommended before signing. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Auto-renewal (HIGH RISK)
        autorenew_keywords = ["auto renew", "automatically renew", "automatic renewal", "evergreen"]
        if any(keyword in text for keyword in autorenew_keywords):
            return {
                "decision": "RENEGOTIATE",
                "risk": "HIGH",
                "explanation": f"HIGH RISK: This auto-renewal clause automatically extends the contract unless you provide advance notice. Without proper tracking, this can lock you into unfavorable terms for extended periods. Recommend negotiating a 60-90 day opt-out notice period or removing auto-renewal entirely. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # No warranty / As-is (HIGH RISK)
        warranty_keywords = ["no warranty", "as is", "as-is", "disclaim", "without warranty"]
        if any(keyword in text for keyword in warranty_keywords):
            return {
                "decision": "REVIEW",
                "risk": "HIGH",
                "explanation": f"HIGH RISK: This clause disclaims all warranties, meaning the vendor provides no guarantees about product/service quality, fitness for purpose, or non-infringement. This shifts significant risk to your company. Legal review strongly recommended. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # ==============================================
        # MEDIUM RISK CLAUSES
        # ==============================================
        
        # Limitation of Liability (MEDIUM RISK)
        if "limit" in text and "liability" in text:
            return {
                "decision": "REVIEW",
                "risk": "MEDIUM",
                "explanation": f"MEDIUM RISK: This clause limits the vendor's liability, typically capping damages to the contract value. Review the liability cap amount - if too low (e.g., less than 12 months fees), your company may not be adequately protected for significant losses. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Confidentiality (MEDIUM RISK)
        if "confidential" in text:
            return {
                "decision": "REVIEW",
                "risk": "MEDIUM",
                "explanation": f"MEDIUM RISK: This confidentiality clause defines how sensitive information must be protected. Check the term length (should not exceed 5 years), exceptions for publicly available information, and required security measures. Standard but requires compliance. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Exclusive remedy (MEDIUM RISK)
        if "exclusive remedy" in text or "sole remedy" in text:
            return {
                "decision": "REVIEW",
                "risk": "MEDIUM",
                "explanation": f"MEDIUM RISK: This clause provides an exclusive or sole remedy, meaning this is the only recourse available if the vendor breaches. This limits your legal options significantly. Review carefully. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # ==============================================
        # LOW RISK CLAUSES (Standard/Acceptable)
        # ==============================================
        
        # Termination (LOW RISK)
        if "terminat" in text:
            return {
                "decision": "ACCEPT",
                "risk": "LOW",
                "explanation": f"LOW RISK: Standard termination clause. Allows either party to end the agreement with reasonable notice (typically 30-90 days). No major concerns identified. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Governing Law (LOW RISK)
        if "governing law" in text or "choice of law" in text:
            return {
                "decision": "ACCEPT",
                "risk": "LOW",
                "explanation": f"LOW RISK: This governing law clause specifies which state's laws apply to the contract. This is standard and typically acceptable, though ensure the chosen jurisdiction is reasonable for your business. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Force Majeure (LOW RISK)
        if "force majeure" in text:
            return {
                "decision": "ACCEPT",
                "risk": "LOW",
                "explanation": f"LOW RISK: This force majeure clause addresses delays caused by unforeseen events beyond either party's control (natural disasters, war, pandemic, etc.). This is standard protection for both parties. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Notice (LOW RISK)
        if "notice" in text and len(text) < 500:
            return {
                "decision": "ACCEPT",
                "risk": "LOW",
                "explanation": f"LOW RISK: This notice clause specifies how formal communications must be delivered (email, certified mail, etc.). Standard administrative provision with no major concerns. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Entire Agreement (LOW RISK)
        if "entire agreement" in text:
            return {
                "decision": "ACCEPT",
                "risk": "LOW",
                "explanation": f"LOW RISK: This entire agreement clause states that this contract represents the complete understanding between parties, superseding prior discussions. This is standard and prevents disputes about verbal promises. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Severability (LOW RISK)
        if "severab" in text:
            return {
                "decision": "ACCEPT",
                "risk": "LOW",
                "explanation": f"LOW RISK: This severability clause ensures that if one part of the contract is found invalid, the rest remains enforceable. This is standard boilerplate language. [Clause: {clause_id}]",
                "citation": clause_id
            }
        
        # Default for unknown clauses
        return {
            "decision": "REVIEW",
            "risk": "MEDIUM",
            "explanation": f"MEDIUM RISK: This clause requires review: \"{text[:150]}...\" The purpose and implications are not immediately clear. Recommend legal review to understand full impact on your business. [Clause: {clause_id}]",
            "citation": clause_id
        }
    
    # =========================================================
    # PHI-2 ANALYSIS (For 4GB GPU)
    # =========================================================
    def phi2_analysis(self, query: str, clauses: list) -> dict:
        """Phi-2 based analysis - 30-60 seconds"""
        
        if self.model is None:
            return self.rule_based_analysis(query, clauses)
        
        try:
            context = ""
            for i, c in enumerate(clauses[:3]):
                context += f"\n[Clause {i+1}: {c['clause_id']}]\n{c['clause_text'][:400]}\n"
            
            # Use improved prompt
            prompt = build_phi2_prompt(query, context)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Parse response
            decision_match = re.search(r'Decision:\s*\[?(\w+)\]?', response, re.IGNORECASE)
            risk_match = re.search(r'Risk:\s*\[?(\w+)\]?', response, re.IGNORECASE)
            explanation_match = re.search(r'Explanation:\s*(.+?)(?=Citation:|$)', response, re.IGNORECASE | re.DOTALL)
            citation_match = re.search(r'Citation:\s*(\S+)', response, re.IGNORECASE)
            
            return {
                "decision": decision_match.group(1).upper() if decision_match else "REVIEW",
                "risk": risk_match.group(1).upper() if risk_match else "MEDIUM",
                "explanation": explanation_match.group(1).strip()[:400] if explanation_match else response[:200],
                "citation": citation_match.group(1) if citation_match else (clauses[0]["clause_id"] if clauses else "unknown")
            }
            
        except Exception as e:
            print(f"   Phi-2 error: {e}")
            return self.rule_based_analysis(query, clauses)
    
    # =========================================================
    # SAULM ANALYSIS (For 8GB GPU)
    # =========================================================
    def saulm_analysis(self, query: str, clauses: list) -> dict:
        """SaulLM-7B analysis - 5-15 seconds, best quality"""
        
        if self.model is None:
            return self.rule_based_analysis(query, clauses)
        
        try:
            context = ""
            for i, c in enumerate(clauses[:3]):
                context += f"\n[CLAUSE {i+1}: {c['clause_id']}]\n{c['clause_text'][:500]}\n"
            
            # Use improved prompt
            prompt = build_saulm_prompt(query, context)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Parse response
            decision_match = re.search(r'Decision:\s*\[?(\w+)\]?', response, re.IGNORECASE)
            risk_match = re.search(r'Risk:\s*\[?(\w+)\]?', response, re.IGNORECASE)
            explanation_match = re.search(r'Explanation:\s*(.+?)(?=Citation:|$)', response, re.IGNORECASE | re.DOTALL)
            citation_match = re.search(r'Citation:\s*(\S+)', response, re.IGNORECASE)
            
            return {
                "decision": decision_match.group(1).upper() if decision_match else "REVIEW",
                "risk": risk_match.group(1).upper() if risk_match else "MEDIUM",
                "explanation": explanation_match.group(1).strip()[:500] if explanation_match else response[:200],
                "citation": citation_match.group(1) if citation_match else (clauses[0]["clause_id"] if clauses else "unknown")
            }
            
        except Exception as e:
            print(f"   SaulLM error: {e}")
            return self.rule_based_analysis(query, clauses)
    
    # =========================================================
    # MAIN ANSWER FUNCTION - WITH AUTO MODEL SELECTION
    # =========================================================
    def answer(self, query: str) -> dict:
        """Complete pipeline - auto-selects best available model"""
        
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Step 1: Retrieve
        start = time.time()
        clauses = self.retrieve(query)
        elapsed = time.time() - start
        
        print(f"Retrieved {len(clauses)} clauses in {elapsed:.2f}s")
        
        if clauses:
            print(f"   Top clause: {clauses[0]['clause_id']}")
        
        # Step 2: Analyze with best available model
        if self.use_saulm:
            print("Using SaulLM-7B (Best quality)...")
            start = time.time()
            result = self.saulm_analysis(query, clauses)
            elapsed = time.time() - start
            print(f"   Completed in {elapsed:.1f} seconds")
            
        elif self.use_phi2:
            print("Using Phi-2 (Good quality)...")
            start = time.time()
            result = self.phi2_analysis(query, clauses)
            elapsed = time.time() - start
            print(f"   Completed in {elapsed:.1f} seconds")
            
        else:
            print("Using Rule-based (Instant)...")
            result = self.rule_based_analysis(query, clauses)
        
        return result


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MAHAK'S ADAPTIVE CONTRACT GENERATOR")
    print("   Auto-detects GPU: 4GB → Phi-2, 8GB+ → SaulLM-7B")
    print("=" * 70)
    
    gen = AdaptiveContractGenerator()
    
    # Load retriever
    if not gen.load_retriever():
        print("Failed to load retriever")
        exit(1)
    
    # Auto-detect and load best model
    model_type = gen.auto_detect_model()
    
    if model_type == "saulm":
        gen.load_saulm()
    elif model_type == "phi2":
        gen.load_phi2()
    else:
        print("   Using Rule-based (no LLM needed)")
    
    # Test queries
    test_queries = [
        "indemnification risks",
        "auto renewal clause",
        "limitation of liability",
        "termination clause",
        "confidentiality clause",
        "force majeure"
    ]
    
    results = []
    
    for q in test_queries:
        result = gen.answer(q)
        results.append(result)
        
        print("\n" + "=" * 50)
        print("OUTPUT:")
        print("=" * 50)
        print(f"   Decision: {result['decision']}")
        print(f"   Risk: {result['risk']}")
        print(f"   Explanation: {result['explanation'][:200]}...")
        print(f"   Citation: {result['citation']}")
    
    # Save results for report
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("GENERATOR WORKING!")
    print(f"   Model mode: {model_type.upper()}")
    print(f"   Queries tested: {len(test_queries)}")
    print("   Results saved to: evaluation_results.json")
    print("=" * 70)