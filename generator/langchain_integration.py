# generator/langchain_integration.py
"""
BM25 RETRIEVER WRAPPER FOR LANGCHAIN
Wraps the existing BM25 retriever for LangChain integration
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as LangChainBM25
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class DocumentProcessor:
    """Processes clause data into LangChain documents"""
    
    def __init__(self):
        self.clauses = []
        self.documents = []
        
    def load_clause_data(self, data_path: str = "data/processed/clauses.jsonl") -> bool:
        """Load clause data from JSONL file"""
        print(f"Loading clause data from {data_path}...")
        
        file_path = Path(data_path)
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False
        
        self.clauses = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.clauses.append(json.loads(line))
        
        print(f"   Loaded {len(self.clauses)} clauses")
        return True
    
    def convert_to_documents(self) -> list:
        """Convert clauses to LangChain Document objects"""
        
        self.documents = []
        for clause in self.clauses:
            doc = Document(
                page_content=clause.get("clause_text", ""),
                metadata={
                    "clause_id": clause.get("clause_id", "unknown"),
                    "contract_id": clause.get("contract_id", "unknown"),
                    "source": "bm25_index"
                }
            )
            self.documents.append(doc)
        
        print(f"   Converted to {len(self.documents)} LangChain documents")
        return self.documents


class LangChainRetriever:
    """LangChain-compatible retriever using BM25"""
    
    def __init__(self):
        self.retriever = None
        self.documents = []
        
    def initialize(self, documents: list) -> None:
        """Initialize BM25 retriever with documents"""
        self.documents = documents
        self.retriever = LangChainBM25.from_documents(documents)
        print("   LangChain BM25 retriever initialized")
        
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve relevant documents"""
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        results = self.retriever.get_relevant_documents(query)
        return results[:top_k]


class ContractQAAnalyzer:
    """Complete QA system using LangChain and BM25"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.retriever = LangChainRetriever()
        self.llm = None
        self.prompt = None
        self.chain = None
        
    def load_data(self) -> bool:
        """Load and prepare clause data"""
        if not self.doc_processor.load_clause_data():
            return False
        
        documents = self.doc_processor.convert_to_documents()
        self.retriever.initialize(documents)
        return True
    
    def setup_llm(self, model_name: str = "microsoft/phi-2") -> bool:
        """Setup language model for analysis"""
        print("\nSetting up language model...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.95,
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print(f"   LLM ready: {model_name}")
            return True
            
        except Exception as e:
            print(f"   LLM setup failed: {e}")
            return False
    
    def create_prompt_template(self) -> None:
        """Create the prompt template for contract analysis"""
        
        template = """You are a contract analyst. Use the clauses below to answer the question.

RELEVANT CLAUSES:
{context}

QUESTION: {question}

Respond in this exact format:
Decision: [ACCEPT/REVIEW/RENEGOTIATE/ESCALATE]
Risk: [LOW/MEDIUM/HIGH/CRITICAL]
Explanation: [Your analysis with [Clause: id]]
Citation: [clause_id]

Answer:"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        print("   Prompt template created")
    
    def create_chain(self) -> None:
        """Create the complete LangChain processing chain"""
        
        if not self.retriever.retriever:
            raise ValueError("Retriever not initialized")
        
        if not self.llm:
            raise ValueError("LLM not initialized")
        
        def format_documents(docs):
            return "\n\n".join([
                f"[{doc.metadata['clause_id']}] {doc.page_content[:500]}" 
                for doc in docs
            ])
        
        self.chain = (
            {
                "context": self.retriever.retriever | format_documents,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("   LangChain chain created")
    
    def answer(self, query: str) -> dict:
        """Answer a contract query"""
        
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        if not self.chain:
            raise ValueError("Chain not initialized. Run setup first.")
        
        # Get response
        response = self.chain.invoke(query)
        
        # Parse response
        import re
        
        decision_match = re.search(r'Decision:\s*\[?(\w+)\]?', response, re.IGNORECASE)
        risk_match = re.search(r'Risk:\s*\[?(\w+)\]?', response, re.IGNORECASE)
        explanation_match = re.search(r'Explanation:\s*(.+?)(?=Citation:|$)', response, re.IGNORECASE | re.DOTALL)
        citation_match = re.search(r'Citation:\s*(\S+)', response, re.IGNORECASE)
        
        return {
            "decision": decision_match.group(1).upper() if decision_match else "REVIEW",
            "risk": risk_match.group(1).upper() if risk_match else "MEDIUM",
            "explanation": explanation_match.group(1).strip()[:500] if explanation_match else response[:300],
            "citation": citation_match.group(1) if citation_match else "unknown"
        }
    
    def setup_complete(self) -> bool:
        """Complete setup of all components"""
        print("=" * 60)
        print("Setting up Contract QA System")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        self.setup_llm()
        self.create_prompt_template()
        self.create_chain()
        
        print("\nSystem ready!")
        return True


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CONTRACT QA SYSTEM WITH LANGCHAIN")
    print("   Using BM25 retriever from existing codebase")
    print("=" * 70)
    
    analyzer = ContractQAAnalyzer()
    
    if analyzer.setup_complete():
        result = analyzer.answer("indemnification risks")
        
        print("\n" + "=" * 50)
        print("OUTPUT:")
        print("=" * 50)
        print(f"   Decision: {result['decision']}")
        print(f"   Risk: {result['risk']}")
        print(f"   Explanation: {result['explanation'][:200]}...")
        print(f"   Citation: {result['citation']}")
