from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from context_manager import generate_with_context
from typing import Optional, Dict, List

app = FastAPI(title="Trace PMT Protocol API", version="1.0.0")

class GenerateRequest(BaseModel):
    prompt: str
    user_id: str
    max_tokens: Optional[int] = 512

class GenerateResponse(BaseModel):
    response: str
    user_id: str
    has_context: bool
    semantic_analytics: Optional[Dict] = None

class SemanticAnalyticsResponse(BaseModel):
    user_id: str
    total_conversations: int
    average_similarity: float
    semantic_diversity: float
    top_topics: List[str]

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """
    Generate a response with context from past traces.
    """
    try:
        response = generate_with_context(request.prompt, request.user_id)
        
        # Get semantic analytics
        try:
            from semantic_embeddings import semantic_embeddings
            analytics = semantic_embeddings.get_semantic_analytics(request.user_id)
        except Exception as e:
            analytics = None
            print(f"⚠️ Failed to get semantic analytics: {e}")
        
        return GenerateResponse(
            response=response,
            user_id=request.user_id,
            has_context=True,
            semantic_analytics=analytics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/analytics/{user_id}", response_model=SemanticAnalyticsResponse)
async def get_semantic_analytics(user_id: str):
    """
    Get semantic analytics for a user.
    """
    try:
        from semantic_embeddings import semantic_embeddings
        analytics = semantic_embeddings.get_semantic_analytics(user_id)
        
        return SemanticAnalyticsResponse(
            user_id=user_id,
            total_conversations=analytics["total_conversations"],
            average_similarity=analytics["average_similarity"],
            semantic_diversity=analytics["semantic_diversity"],
            top_topics=analytics["top_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "trace-pmt-protocol"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 