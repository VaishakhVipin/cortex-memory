from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import time

from context_manager import generate_with_context, generate_with_evolving_context
from self_evolving_context import self_evolving_context
from semantic_embeddings import semantic_embeddings

app = FastAPI(title="PMT Protocol API", version="2.0.0")

class GenerateRequest(BaseModel):
    prompt: str
    user_id: str
    use_evolving_context: bool = False

class GenerateResponse(BaseModel):
    response: str
    context_used: List[Dict]
    semantic_analytics: Dict
    evolving_analytics: Optional[Dict] = None

class EvolvingAnalyticsResponse(BaseModel):
    system_status: Dict
    performance_summary: Dict
    pruning_statistics: Dict
    pattern_analysis: Dict
    pruning_recommendations: List[Dict]

class PruningRequest(BaseModel):
    user_id: str
    threshold: Optional[float] = None
    trace_ids: Optional[List[str]] = None

class PruningResponse(BaseModel):
    total_traces: int
    pruned_traces: int
    kept_traces: int
    memory_saved_mb: float
    pruning_reasons: Dict

class PatternAnalysisResponse(BaseModel):
    query_structures: Dict
    topic_clusters: Dict
    intent_patterns: Dict
    temporal_patterns: Dict
    semantic_patterns: Dict
    sentiment_analysis: Dict
    complexity_analysis: Dict
    domain_analysis: Dict
    behavioral_patterns: Dict
    advanced_metrics: Dict

class PredictionRequest(BaseModel):
    user_id: str
    current_query: str

class PredictionResponse(BaseModel):
    likely_next_topics: List[str]
    query_structure_prediction: str
    intent_prediction: str
    confidence: float
    sentiment_prediction: str
    complexity_prediction: float
    domain_prediction: List[str]
    behavioral_insights: Dict

class DriftDetectionRequest(BaseModel):
    user_id: str
    threshold: Optional[float] = None

class DriftDetectionResponse(BaseModel):
    overall_drift_score: float
    drift_detected: bool
    drift_severity: str
    performance_drift: Dict
    behavioral_drift: Dict
    context_relevance_drift: Dict
    accuracy_drift: Dict
    recommendations: List[str]
    trends: Dict
    alerts: List[str]

class DriftSummaryResponse(BaseModel):
    user_id: str
    drift_status: str
    severity: str
    overall_score: float
    components_affected: List[str]
    recommendations_count: int
    alerts_count: int
    last_updated: float

@app.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """Generate a response with semantic context."""
    try:
        response = await generate_with_context(request.prompt, request.user_id)
        
        # Get semantic analytics
        analytics = semantic_embeddings.get_semantic_analytics(request.user_id)
        
        return GenerateResponse(
            response=response,
            context_used=[],  # TODO: Extract context used
            semantic_analytics=analytics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/evolving", response_model=GenerateResponse)
async def generate_with_evolving_context_endpoint(request: GenerateRequest):
    """Generate a response using the self-evolving context system."""
    try:
        response = await generate_with_evolving_context(request.prompt, request.user_id)
        
        # Get evolving analytics
        evolving_analytics = self_evolving_context.get_evolving_analytics(request.user_id)
        
        return GenerateResponse(
            response=response,
            context_used=[],  # TODO: Extract context used
            evolving_analytics=evolving_analytics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/semantic/{user_id}")
async def get_semantic_analytics(user_id: str):
    """Get semantic analytics for a user."""
    try:
        analytics = semantic_embeddings.get_semantic_analytics(user_id)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/evolving/{user_id}", response_model=EvolvingAnalyticsResponse)
async def get_evolving_analytics(user_id: str):
    """Get evolving analytics for a user."""
    try:
        analytics = self_evolving_context.get_evolving_analytics(user_id)
        return EvolvingAnalyticsResponse(**analytics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pruning/auto", response_model=PruningResponse)
async def auto_prune_traces(request: PruningRequest):
    """Automatically prune low-impact traces."""
    try:
        pruning_stats = self_evolving_context.auto_pruning.prune_low_impact_traces(
            request.user_id, request.threshold
        )
        return PruningResponse(**pruning_stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pruning/recommendations/{user_id}")
async def get_pruning_recommendations(user_id: str):
    """Get pruning recommendations without actually pruning."""
    try:
        recommendations = self_evolving_context.get_pruning_recommendations(user_id)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pruning/manual")
async def manual_prune_traces(user_id: str, trace_ids: List[str]):
    """Manually prune specific traces."""
    try:
        pruning_stats = self_evolving_context.manual_prune_traces(user_id, trace_ids)
        return pruning_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patterns/analyze/{user_id}", response_model=PatternAnalysisResponse)
async def analyze_query_patterns(user_id: str):
    """Analyze query patterns for a user."""
    try:
        patterns = self_evolving_context.analyze_query_patterns(user_id)
        return PatternAnalysisResponse(**patterns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patterns/predict", response_model=PredictionResponse)
async def predict_next_query_pattern(request: PredictionRequest):
    """Predict the next likely query pattern for a user."""
    try:
        prediction = self_evolving_context.predict_next_query_pattern(
            request.user_id, request.current_query
        )
        return PredictionResponse(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/drift/detect", response_model=DriftDetectionResponse)
async def detect_semantic_drift(request: DriftDetectionRequest):
    """Detect semantic drift for a user."""
    try:
        # Set custom threshold if provided
        if request.threshold is not None:
            self_evolving_context.set_drift_threshold(request.threshold)
        
        drift_analysis = self_evolving_context.detect_semantic_drift(request.user_id)
        return DriftDetectionResponse(**drift_analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

@app.get("/drift/summary/{user_id}", response_model=DriftSummaryResponse)
async def get_drift_summary(user_id: str):
    """Get a summary of drift detection results for a user."""
    try:
        summary = self_evolving_context.get_drift_summary(user_id)
        return DriftSummaryResponse(**summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift summary failed: {str(e)}")

@app.post("/drift/configure")
async def configure_drift_detection(component: str, enabled: bool = True):
    """Enable or disable drift detection components."""
    try:
        self_evolving_context.enable_drift_component(component, enabled)
        return {"message": f"Drift component '{component}' {'enabled' if enabled else 'disabled'}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configuration failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "self_evolving_context": "active",
            "auto_pruning": "active", 
            "pattern_recognition": "active",
            "semantic_drift_detection": "active",
            "semantic_embeddings": "active"
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 