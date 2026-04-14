# generator/inference.py
"""
MAHAK'S INFERENCE API
This API uses YOUR WORKING model.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import YOUR working model.py
from generator.model import AdaptiveContractGenerator

app = FastAPI(
    title="Mahak's Contract Generator API",
    description="Uses the working model.py with rule-based analysis",
    version="1.0"
)

# Initialize your generator
generator = None

@app.on_event("startup")
async def startup_event():
    """Load your generator on startup"""
    global generator
    print("🚀 Starting API - Loading Mahak's Generator...")
    generator = AdaptiveContractGenerator()
    generator.load_retriever()
    # Auto-detect will run, but will use rule-based on your laptop
    generator.auto_detect_model()
    print("✅ API Ready!")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    decision: str
    risk: str
    explanation: str
    citation: str

@app.post("/analyze", response_model=QueryResponse)
async def analyze(request: QueryRequest):
    """
    Analyze a contract query using Mahak's generator
    """
    try:
        # Call YOUR model.py's answer method
        result = generator.answer(request.query)
        
        return QueryResponse(
            decision=result["decision"],
            risk=result["risk"],
            explanation=result["explanation"],
            citation=result["citation"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "generator_loaded": generator is not None,
        "mode": "rule-based" if generator and generator.use_rule else "llm"
    }

@app.get("/")
async def root():
    return {
        "name": "Mahak's Contract Generator API",
        "endpoints": {
            "POST /analyze": "Send {'query': 'your question here'}",
            "GET /health": "Check API status"
        },
        "example": {
            "curl": 'curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d "{\\"query\\": \\"indemnification risks\\"}"'
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Starting Mahak's Generator API")
    print("=" * 60)
    print("API will be available at: http://localhost:8000")
    print("Try: curl -X POST http://localhost:8000/analyze -H 'Content-Type: application/json' -d '{\"query\": \"indemnification risks\"}'")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)