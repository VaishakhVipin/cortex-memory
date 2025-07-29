#!/usr/bin/env python3
"""
🧠 Cortex API - Enterprise-Grade Context-Aware AI System
RESTful API for semantic context management and self-evolving memory.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
from datetime import datetime

from semantic_embeddings import semantic_embeddings
from self_evolving_context import self_evolving_context
from semantic_drift_detection import detect_semantic_drift
from context_manager import generate_with_context, generate_with_evolving_context

app = FastAPI(title="Cortex API", version="2.0.0")

# Request/Response Models
class ConversationRequest(BaseModel):
    user_id: str
    prompt: str
    response: str
    metadata: Optional[Dict] = None

class ContextRequest(BaseModel):
    user_id: str
    prompt: str
    limit: int = 5
    similarity_threshold: float = 0.3

class GenerationRequest(BaseModel):
    user_id: str
    prompt: str
    context_method: str = "semantic"  # "semantic" or "evolving"

class PruningRequest(BaseModel):
    user_id: str
    threshold: Optional[float] = None

class AnalyticsResponse(BaseModel):
    user_id: str
    total_memories: int
    pruned_memories: int
    kept_memories: int
    avg_success_rate: float
    avg_quality: float
    impact_ratio: float

class DriftResponse(BaseModel):
    user_id: str
    drift_detected: bool
    drift_score: float
    drift_type: str
    recommendations: List[str]

# Core API Endpoints
@app.post("/conversations", response_model=Dict)
async def store_conversation(request: ConversationRequest):
    """Store a conversation with semantic embeddings."""
    try:
        memory_id = semantic_embeddings.store_conversation_embedding(
            user_id=request.user_id,
            prompt=request.prompt,
            response=request.response,
            metadata=request.metadata or {}
        )
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "message": "Conversation stored successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context/semantic", response_model=Dict)
async def get_semantic_context(request: ContextRequest):
    """Get semantically similar context."""
    try:
        similar_contexts = semantic_embeddings.find_semantically_similar_context(
            user_id=request.user_id,
            current_prompt=request.prompt,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        context_data = []
        for context, score in similar_contexts:
            context_data.append({
                "memory_id": context.get("embedding_id"),
                "prompt": context.get("prompt"),
                "response": context.get("response"),
                "similarity_score": score,
                "metadata": context.get("metadata", {})
            })
        
        return {
            "status": "success",
            "contexts": context_data,
            "total_found": len(context_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/context/evolving", response_model=Dict)
async def get_evolving_context(request: ContextRequest):
    """Get context using self-evolving algorithms."""
    try:
        evolving_contexts = self_evolving_context.find_evolving_context(
            user_id=request.user_id,
            current_prompt=request.prompt,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        context_data = []
        for context, score in evolving_contexts:
            context_data.append({
                "memory_id": context.get("embedding_id"),
                "prompt": context.get("prompt"),
                "response": context.get("response"),
                "similarity_score": score,
                "metadata": context.get("metadata", {})
            })
        
        return {
            "status": "success",
            "contexts": context_data,
            "total_found": len(context_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=Dict)
async def generate_with_context_api(request: GenerationRequest):
    """Generate response with context awareness."""
    try:
        if request.context_method == "semantic":
            response = generate_with_context(
                user_id=request.user_id,
                prompt=request.prompt
            )
        elif request.context_method == "evolving":
            response = generate_with_evolving_context(
                user_id=request.user_id,
                prompt=request.prompt
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid context method")
        
        return {
            "status": "success",
            "response": response,
            "context_method": request.context_method
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Self-Evolving Context Endpoints
@app.post("/evolving/track", response_model=Dict)
async def track_context_effectiveness(
    user_id: str,
    memory_ids: List[str],
    response_quality: float,
    user_feedback: Optional[str] = None
):
    """Track context effectiveness for learning."""
    try:
        self_evolving_context.track_context_effectiveness(
            user_id=user_id,
            memory_ids=memory_ids,
            response_quality=response_quality,
            user_feedback=user_feedback
        )
        
        return {
            "status": "success",
            "message": "Context effectiveness tracked successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evolving/update-weights", response_model=Dict)
async def update_adaptive_weights(user_id: str):
    """Update adaptive weights for all user memories."""
    try:
        updated_count = self_evolving_context.update_adaptive_weights(user_id)
        
        return {
            "status": "success",
            "updated_memories": updated_count,
            "message": "Adaptive weights updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Auto-Pruning Endpoints
@app.post("/pruning/auto", response_model=Dict)
async def auto_prune_memories(request: PruningRequest):
    """Automatically prune low-impact memories."""
    try:
        pruning_stats = self_evolving_context.auto_pruning.prune_low_impact_memories(
            user_id=request.user_id,
            threshold=request.threshold
        )
        
        return {
            "status": "success",
            "pruning_stats": pruning_stats,
            "message": "Auto-pruning completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pruning/manual", response_model=Dict)
async def manual_prune_memories(user_id: str, memory_ids: List[str]):
    """Manually prune specific memories."""
    try:
        pruning_stats = self_evolving_context.manual_prune_memories(user_id, memory_ids)
        
        return {
            "status": "success",
            "pruning_stats": pruning_stats,
            "message": "Manual pruning completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoints
@app.get("/analytics/{user_id}", response_model=AnalyticsResponse)
async def get_performance_analytics(user_id: str):
    """Get comprehensive performance analytics."""
    try:
        metrics = self_evolving_context.get_performance_metrics(user_id)
        
        return AnalyticsResponse(
            user_id=user_id,
            total_memories=metrics["total_memories"],
            pruned_memories=metrics.get("pruned_memories", 0),
            kept_memories=metrics.get("kept_memories", 0),
            avg_success_rate=metrics["avg_success_rate"],
            avg_quality=metrics["avg_quality"],
            impact_ratio=metrics["impact_ratio"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/{user_id}", response_model=DriftResponse)
async def get_drift_analysis(user_id: str, time_window_hours: int = 24):
    """Get semantic drift analysis."""
    try:
        drift_results = detect_semantic_drift(
            user_id=user_id,
            time_window_hours=time_window_hours
        )
        
        return DriftResponse(
            user_id=user_id,
            drift_detected=drift_results["drift_detected"],
            drift_score=drift_results["drift_score"],
            drift_type=drift_results["drift_type"],
            recommendations=drift_results.get("recommendations", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System Health Endpoints
@app.get("/health")
async def health_check():
    """System health check."""
    try:
        # Test Redis connection
        from redis_client import r
        r.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "redis": "connected",
                "semantic_embeddings": "ready",
                "self_evolving_context": "ready"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get system-wide statistics."""
    try:
        from redis_client import r
        
        # Get basic Redis stats
        redis_info = r.info()
        
        # Count total memories across all users
        all_memories = r.keys("embedding:*")
        
        return {
            "status": "success",
            "system_stats": {
                "total_memories": len(all_memories),
                "redis_memory_usage": redis_info.get("used_memory_human", "N/A"),
                "redis_connected_clients": redis_info.get("connected_clients", 0),
                "uptime": redis_info.get("uptime_in_seconds", 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 