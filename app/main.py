from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.hybrid_pipeline import HybridPipelineFinal
import uvicorn

app = FastAPI(
    title="Lumora Analytics Enterprise Hybrid RAG",
    description="Structured + Semantic + ML-Powered Query System",
    version="1.0"
)

pipeline = HybridPipelineFinal()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {
        "message": "Lumora Analytics Enterprise Hybrid RAG System",
        "status": "running",
        "ml_model": "DistilBERT (Fine-tuned)",
        "docs": "http://127.0.0.1:8000/docs"
    }

@app.post("/query")
async def ask_question(request: QueryRequest):
    try:
        result = pipeline.handle_query(request.query)
        return {
            "success": True,
            "answer": result.get("answer"),
            "query_type": "structured" if "structured" in str(result).lower() else "semantic/hybrid"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy", "chunks": 322, "classifier": "DistilBERT"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)