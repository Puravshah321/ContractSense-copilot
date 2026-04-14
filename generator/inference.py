# generator/inference.py
"""
MAHAK'S ENHANCED INFERENCE API
Combines adaptive model.py with LangChain for better RAG pipeline
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import generators
from generator.model import AdaptiveContractGenerator
from generator.langchain_integration import ContractQAAnalyzer

app = FastAPI(
    title="Mahak's Enhanced Contract Analyzer API",
    description="Hybrid approach: adaptive model + LangChain RAG",
    version="2.0"
)

# Initialize generators
adaptive_generator = None
langchain_analyzer = None
use_langchain = False

@app.on_event("startup")
async def startup_event():
    """Load generators on startup"""
    global adaptive_generator, langchain_analyzer, use_langchain
    print("Starting API - Loading generators...")
    
    # Load adaptive model (rule-based + LLM)
    adaptive_generator = AdaptiveContractGenerator()
    adaptive_generator.load_retriever()
    adaptive_generator.auto_detect_model()
    print("   Adaptive model loaded")
    
    # Try loading LangChain (optional)
    try:
        langchain_analyzer = ContractQAAnalyzer()
        if langchain_analyzer.setup_complete():
            use_langchain = True
            print("   LangChain RAG pipeline loaded")
        else:
            print("   LangChain setup incomplete, using adaptive model only")
    except Exception as e:
        print(f"   LangChain failed ({e}), using adaptive model only")
    
    print("API Ready!")

class QueryRequest(BaseModel):
    query: str
    use_langchain: Optional[bool] = None

class QueryResponse(BaseModel):
    decision: str
    risk: str
    explanation: str
    citation: str
    method: str  # "adaptive" or "langchain"

@app.post("/analyze", response_model=QueryResponse)
async def analyze(request: QueryRequest):
    """
    Analyze a contract query using hybrid approach
    """
    try:
        # Choose method
        use_lc = request.use_langchain if request.use_langchain is not None else use_langchain
        
        if use_lc and langchain_analyzer:
            # Use LangChain RAG
            result = langchain_analyzer.answer(request.query)
            return QueryResponse(
                decision=result["decision"],
                risk=result["risk"],
                explanation=result["explanation"],
                citation=result["citation"],
                method="langchain"
            )
        else:
            # Use adaptive model
            result = adaptive_generator.answer(request.query)
            return QueryResponse(
                decision=result["decision"],
                risk=result["risk"],
                explanation=result["explanation"],
                citation=result["citation"],
                method="adaptive"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "adaptive_loaded": adaptive_generator is not None,
        "langchain_loaded": use_langchain,
        "default_method": "langchain" if use_langchain else "adaptive",
        "adaptive_mode": "rule-based" if adaptive_generator and adaptive_generator.use_rule else "llm"
    }

@app.get("/")
async def root():
    return {
        "name": "Mahak's Enhanced Contract Analyzer API",
        "version": "2.0",
        "endpoints": {
            "POST /analyze": "Analyze a contract query",
            "GET /health": "Check API status"
        },
        "methods": {
            "adaptive": "Fast rule-based or LLM analysis",
            "langchain": "LangChain RAG with retrieval (if available)"
        },
        "example": {
            "adaptive": {"query": "indemnification risks", "use_langchain": False},
            "langchain": {"query": "indemnification risks", "use_langchain": True}
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("Starting Mahak's Enhanced Contract Analyzer API v2.0")
    print("=" * 70)
    print("API: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)