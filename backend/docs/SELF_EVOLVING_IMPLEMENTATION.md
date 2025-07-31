# 🧠 Self-Evolving Context Model - Implementation Guide

## Overview

This guide provides step-by-step implementation of the Self-Evolving Context Model that builds on PMT's existing semantic context provision base.

## 🏗️ Architecture Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-EVOLVING LAYER                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Context       │  │   Recall        │  │   Adaptive   │ │
│  │   Scoring       │  │   Tracking      │  │   Weighting  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    EXISTING SEMANTIC BASE                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   semantic_     │  │   context_      │  │   core.py    │ │
│  │   embeddings.py │  │   manager.py    │  │   api.py     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Implementation Steps

### Step 1: Create Self-Evolving Context Engine

Create `self_evolving_context.py`:

```python
#!/usr/bin/env python3
"""
🧠 Self-Evolving Context Model
Builds on semantic context provision to create adaptive, learning context system.
"""

import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from redis_client import r
from semantic_embeddings import semantic_embeddings

class ContextScoringEngine:
    """Calculate adaptive context scores based on historical performance."""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.decay_factor = 0.95
    
    def calculate_context_score(self, trace_id: str, query: str, 
                              response_quality: float) -> float:
        """
        Calculate adaptive context score based on historical performance.
        
        Args:
            trace_id: ID of the context trace
            query: Current query
            response_quality: Quality of response (0-1)
            
        Returns:
            Adaptive context score (0-1)
        """
        # Get historical performance
        historical_score = self._get_historical_score(trace_id)
        
        # Get query pattern match
        pattern_score = self._get_pattern_match_score(trace_id, query)
        
        # Get temporal relevance
        temporal_score = self._get_temporal_score(trace_id)
        
        # Calculate adaptive score
        adaptive_score = (
            historical_score * 0.4 +
            pattern_score * 0.3 +
            temporal_score * 0.2 +
            response_quality * 0.1
        )
        
        return float(adaptive_score)
    
    def _get_historical_score(self, trace_id: str) -> float:
        """Get historical performance score for a trace."""
        redis_key = f"context_score:{trace_id}"
        data = r.get(redis_key)
        
        if data:
            score_data = json.loads(data)
            return score_data.get("historical_score", 0.5)
        
        return 0.5  # Default score for new traces
    
    def _get_pattern_match_score(self, trace_id: str, query: str) -> float:
        """Calculate pattern match score based on query similarity."""
        # Get trace data
        trace_data = semantic_embeddings.get_conversation_embedding(trace_id)
        if not trace_data:
            return 0.5
        
        # Calculate similarity with original trace query
        query_embedding = semantic_embeddings.generate_embedding(query)
        trace_embedding = semantic_embeddings.generate_embedding(trace_data["prompt"])
        
        similarity = semantic_embeddings.calculate_semantic_similarity(
            query_embedding, trace_embedding
        )
        
        return float(similarity)
    
    def _get_temporal_score(self, trace_id: str) -> float:
        """Calculate temporal relevance score."""
        trace_data = semantic_embeddings.get_conversation_embedding(trace_id)
        if not trace_data:
            return 0.5
        
        # Get creation time
        created_at = trace_data.get("created_at", time.time())
        current_time = time.time()
        
        # Calculate temporal decay (7-day half-life)
        time_diff = current_time - created_at
        decay_factor = np.exp(-time_diff / (7 * 24 * 3600))
        
        return float(decay_factor)

class RecallTracker:
    """Track context recall success and effectiveness."""
    
    def __init__(self):
        self.success_threshold = 0.7
    
    def track_recall_success(self, trace_id: str, query: str, 
                           response_quality: float, user_feedback: Optional[bool] = None):
        """
        Track whether injected context was helpful.
        
        Args:
            trace_id: ID of the context trace used
            query: Original query
            response_quality: Measured response quality (0-1)
            user_feedback: Explicit user feedback (True=helpful, False=not helpful)
        """
        # Determine if context was helpful
        was_helpful = self._determine_helpfulness(response_quality, user_feedback)
        
        # Update recall statistics
        self._update_recall_stats(trace_id, was_helpful, response_quality)
        
        # Update context score
        self._update_context_score(trace_id, was_helpful, response_quality)
        
        print(f"📊 Tracked recall success for {trace_id}: {'✅' if was_helpful else '❌'}")
    
    def _determine_helpfulness(self, response_quality: float, 
                             user_feedback: Optional[bool]) -> bool:
        """Determine if context was helpful based on quality and feedback."""
        if user_feedback is not None:
            return user_feedback
        
        # Use response quality as proxy for helpfulness
        return response_quality >= self.success_threshold
    
    def _update_recall_stats(self, trace_id: str, was_helpful: bool, 
                           response_quality: float):
        """Update recall statistics in Redis."""
        redis_key = f"recall_stats:{trace_id}"
        
        # Get existing stats
        data = r.get(redis_key)
        if data:
            stats = json.loads(data)
        else:
            stats = {
                "total_uses": 0,
                "successful_uses": 0,
                "total_quality": 0.0,
                "success_rate": 0.0,
                "avg_quality": 0.0
            }
        
        # Update stats
        stats["total_uses"] += 1
        stats["total_quality"] += response_quality
        
        if was_helpful:
            stats["successful_uses"] += 1
        
        stats["success_rate"] = stats["successful_uses"] / stats["total_uses"]
        stats["avg_quality"] = stats["total_quality"] / stats["total_uses"]
        
        # Store updated stats
        r.set(redis_key, json.dumps(stats))
    
    def _update_context_score(self, trace_id: str, was_helpful: bool, 
                            response_quality: float):
        """Update context score based on recall success."""
        redis_key = f"context_score:{trace_id}"
        
        # Get current score
        data = r.get(redis_key)
        if data:
            score_data = json.loads(data)
            current_score = score_data.get("historical_score", 0.5)
        else:
            current_score = 0.5
            score_data = {"historical_score": current_score}
        
        # Calculate new score with learning
        if was_helpful:
            new_score = current_score + (1.0 - current_score) * 0.1
        else:
            new_score = current_score * 0.95  # Decay for unsuccessful uses
        
        # Update score
        score_data["historical_score"] = float(new_score)
        score_data["last_updated"] = time.time()
        
        r.set(redis_key, json.dumps(score_data))

class AdaptiveWeighting:
    """Dynamically adjust context weights based on performance."""
    
    def __init__(self):
        self.weight_update_interval = 3600  # 1 hour
        self.min_weight = 0.1
        self.max_weight = 2.0
    
    def update_weights(self, user_id: str):
        """Update adaptive weights for all user traces."""
        user_embeddings = semantic_embeddings.get_user_embeddings(user_id, limit=1000)
        
        for embedding_data in user_embeddings:
            trace_id = embedding_data["embedding_id"]
            new_weight = self._calculate_adaptive_weight(trace_id)
            
            # Update metadata with new weight
            metadata = embedding_data.get("metadata", {})
            metadata["adaptive_weight"] = new_weight
            
            # Update in Redis
            redis_key = f"embedding:{trace_id}"
            json_safe_data = embedding_data.copy()
            json_safe_data["embedding"] = semantic_embeddings.encode_embedding_for_redis(
                embedding_data["embedding"]
            )
            r.set(redis_key, json.dumps(json_safe_data))
    
    def _calculate_adaptive_weight(self, trace_id: str) -> float:
        """Calculate adaptive weight based on performance metrics."""
        # Get recall statistics
        recall_key = f"recall_stats:{trace_id}"
        recall_data = r.get(recall_key)
        
        if not recall_data:
            return 1.0  # Default weight
        
        stats = json.loads(recall_data)
        success_rate = stats.get("success_rate", 0.5)
        avg_quality = stats.get("avg_quality", 0.5)
        total_uses = stats.get("total_uses", 0)
        
        # Calculate weight based on performance
        if total_uses < 3:
            # Not enough data, use default
            weight = 1.0
        else:
            # Weight based on success rate and quality
            performance_score = (success_rate * 0.7 + avg_quality * 0.3)
            weight = self.min_weight + (self.max_weight - self.min_weight) * performance_score
        
        return float(weight)

class AutoPruning:
    """Automatically remove low-impact traces."""
    
    def __init__(self):
        self.pruning_threshold = 0.1
        self.min_age_hours = 24  # Don't prune traces younger than 24 hours
    
    def prune_low_impact_traces(self, user_id: str, threshold: float = None):
        """
        Remove traces that have low impact or are outdated.
        
        Args:
            user_id: User identifier
            threshold: Minimum impact threshold (default: self.pruning_threshold)
        """
        if threshold is None:
            threshold = self.pruning_threshold
        
        user_embeddings = semantic_embeddings.get_user_embeddings(user_id, limit=1000)
        current_time = time.time()
        
        pruned_count = 0
        
        for embedding_data in user_embeddings:
            trace_id = embedding_data["embedding_id"]
            created_at = embedding_data.get("created_at", current_time)
            
            # Check if trace is old enough to consider pruning
            age_hours = (current_time - created_at) / 3600
            if age_hours < self.min_age_hours:
                continue
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(trace_id)
            
            # Prune if impact is below threshold
            if impact_score < threshold:
                self._prune_trace(trace_id, user_id)
                pruned_count += 1
        
        print(f"🧹 Pruned {pruned_count} low-impact traces for user {user_id}")
        return pruned_count
    
    def _calculate_impact_score(self, trace_id: str) -> float:
        """Calculate impact score for a trace."""
        # Get recall statistics
        recall_key = f"recall_stats:{trace_id}"
        recall_data = r.get(recall_key)
        
        if not recall_data:
            return 0.0  # No usage data = low impact
        
        stats = json.loads(recall_data)
        success_rate = stats.get("success_rate", 0.0)
        total_uses = stats.get("total_uses", 0)
        avg_quality = stats.get("avg_quality", 0.0)
        
        # Calculate impact score
        usage_score = min(total_uses / 10.0, 1.0)  # Normalize usage
        performance_score = (success_rate * 0.7 + avg_quality * 0.3)
        
        impact_score = usage_score * performance_score
        return float(impact_score)
    
    def _prune_trace(self, trace_id: str, user_id: str):
        """Remove a trace from the system."""
        # Remove from user embeddings list
        user_embeddings_key = f"user_embeddings:{user_id}"
        r.lrem(user_embeddings_key, 0, trace_id)
        
        # Remove trace data
        redis_key = f"embedding:{trace_id}"
        r.delete(redis_key)
        
        # Remove associated data
        r.delete(f"recall_stats:{trace_id}")
        r.delete(f"context_score:{trace_id}")
        r.delete(f"usage:{trace_id}")

class SelfEvolvingContext:
    """Main class that orchestrates the self-evolving context system."""
    
    def __init__(self):
        self.scoring_engine = ContextScoringEngine()
        self.recall_tracker = RecallTracker()
        self.adaptive_weighting = AdaptiveWeighting()
        self.auto_pruning = AutoPruning()
    
    def find_evolving_context(self, user_id: str, current_prompt: str, 
                            limit: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        Find context using self-evolving algorithms.
        
        Args:
            user_id: User identifier
            current_prompt: Current prompt
            limit: Maximum number of contexts to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (embedding_data, enhanced_similarity)
        """
        # Get base semantic matches
        base_matches = semantic_embeddings.find_semantically_similar_context(
            user_id, current_prompt, limit * 2, similarity_threshold * 0.5
        )
        
        # Apply self-evolving enhancements
        evolving_matches = []
        for embedding_data, base_similarity in base_matches:
            trace_id = embedding_data["embedding_id"]
            
            # Calculate adaptive context score
            context_score = self.scoring_engine.calculate_context_score(
                trace_id, current_prompt, 0.5  # Default quality
            )
            
            # Get adaptive weight
            metadata = embedding_data.get("metadata", {})
            adaptive_weight = metadata.get("adaptive_weight", 1.0)
            
            # Calculate enhanced similarity
            enhanced_similarity = (
                base_similarity * 0.4 +
                context_score * 0.4 +
                adaptive_weight * 0.2
            )
            
            if enhanced_similarity >= similarity_threshold:
                evolving_matches.append((embedding_data, enhanced_similarity))
        
        # Sort by enhanced similarity and return top results
        evolving_matches.sort(key=lambda x: x[1], reverse=True)
        return evolving_matches[:limit]
    
    def track_context_effectiveness(self, user_id: str, trace_ids: List[str], 
                                  response_quality: float, user_feedback: Optional[bool] = None):
        """
        Track effectiveness of used context.
        
        Args:
            user_id: User identifier
            trace_ids: List of trace IDs that were used
            response_quality: Quality of the response (0-1)
            user_feedback: Explicit user feedback
        """
        for trace_id in trace_ids:
            self.recall_tracker.track_recall_success(
                trace_id, "", response_quality, user_feedback
            )
        
        # Periodically update weights and prune
        self._periodic_maintenance(user_id)
    
    def _periodic_maintenance(self, user_id: str):
        """Perform periodic maintenance tasks."""
        current_time = time.time()
        last_maintenance_key = f"last_maintenance:{user_id}"
        
        # Check if maintenance is needed
        last_maintenance = r.get(last_maintenance_key)
        if last_maintenance:
            last_time = float(last_maintenance)
            if current_time - last_time < 3600:  # 1 hour
                return
        
        # Update weights
        self.adaptive_weighting.update_weights(user_id)
        
        # Prune low-impact traces
        self.auto_pruning.prune_low_impact_traces(user_id)
        
        # Update maintenance timestamp
        r.set(last_maintenance_key, str(current_time))

# Global instance
self_evolving_context = SelfEvolvingContext()
```

### Step 2: Integrate with Existing Semantic System

Update `semantic_embeddings.py` to include self-evolving capabilities:

```python
# Add to existing semantic_embeddings.py

def find_evolving_semantic_context(self, user_id: str, current_prompt: str, 
                                 limit: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[Dict, float]]:
    """
    Find semantically similar context using self-evolving algorithms.
    
    Args:
        user_id: User identifier
        current_prompt: Current prompt to find context for
        limit: Maximum number of similar contexts to return
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of tuples (embedding_data, enhanced_similarity)
    """
    # Use self-evolving context system
    from self_evolving_context import self_evolving_context
    
    return self_evolving_context.find_evolving_context(
        user_id, current_prompt, limit, similarity_threshold
    )
```

### Step 3: Update Context Manager

Update `context_manager.py` to use self-evolving context:

```python
# Add to existing context_manager.py

def generate_with_evolving_context(prompt: str, user_id: str) -> str:
    """
    Generate response using self-evolving context system.
    
    Args:
        prompt: User prompt
        user_id: User identifier
        
    Returns:
        Generated response
    """
    # Get evolving context
    from semantic_embeddings import semantic_embeddings
    from self_evolving_context import self_evolving_context
    
    similar_contexts = semantic_embeddings.find_evolving_semantic_context(
        user_id, prompt, limit=3, similarity_threshold=0.25
    )
    
    # Extract trace IDs for tracking
    trace_ids = [context_data["embedding_id"] for context_data, _ in similar_contexts]
    
    # Format context
    context = ""
    if similar_contexts:
        context_parts = []
        for embedding_data, similarity in similar_contexts:
            context_part = f"Previous conversation (relevance: {similarity:.2f}):\n"
            context_part += f"User: {embedding_data['prompt']}\n"
            context_part += f"Assistant: {embedding_data['response']}\n"
            context_parts.append(context_part)
        context = "\n".join(context_parts)
    
    # Generate response
    response = generate_with_context(prompt, user_id, context)
    
    # Track context effectiveness (you can implement response quality measurement)
    response_quality = 0.7  # Placeholder - implement actual quality measurement
    self_evolving_context.track_context_effectiveness(
        user_id, trace_ids, response_quality
    )
    
    return response
```

### Step 4: Update API Layer

Update `api.py` to include self-evolving endpoints:

```python
# Add to existing api.py

@app.post("/generate/evolving")
async def generate_with_evolving_context(request: GenerateRequest):
    """
    Generate response using self-evolving context system.
    """
    try:
        response = generate_with_evolving_context(request.prompt, request.user_id)
        
        return {
            "response": response,
            "context_system": "self_evolving",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/evolving/{user_id}")
async def get_evolving_analytics(user_id: str):
    """
    Get self-evolving context analytics.
    """
    try:
        from self_evolving_context import self_evolving_context
        
        # Get user embeddings
        user_embeddings = semantic_embeddings.get_user_embeddings(user_id, limit=1000)
        
        # Calculate evolving metrics
        total_traces = len(user_embeddings)
        high_impact_traces = 0
        total_success_rate = 0.0
        
        for embedding_data in user_embeddings:
            trace_id = embedding_data["embedding_id"]
            
            # Get recall stats
            recall_key = f"recall_stats:{trace_id}"
            recall_data = r.get(recall_key)
            
            if recall_data:
                stats = json.loads(recall_data)
                success_rate = stats.get("success_rate", 0.0)
                total_success_rate += success_rate
                
                if success_rate > 0.7:  # High impact threshold
                    high_impact_traces += 1
        
        avg_success_rate = total_success_rate / total_traces if total_traces > 0 else 0.0
        
        return {
            "user_id": user_id,
            "total_traces": total_traces,
            "high_impact_traces": high_impact_traces,
            "average_success_rate": avg_success_rate,
            "impact_ratio": high_impact_traces / total_traces if total_traces > 0 else 0.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 🧪 Testing the Implementation

Create `test_self_evolving.py`:

```python
#!/usr/bin/env python3
"""
🧪 Test Self-Evolving Context Model
"""

import time
from self_evolving_context import self_evolving_context
from semantic_embeddings import semantic_embeddings
from redis_client import r

def test_self_evolving_context():
    """Test the self-evolving context system."""
    print("🧪 Testing Self-Evolving Context Model")
    print("=" * 50)
    
    # Clear Redis for clean test
    r.flushdb()
    
    user_id = "test_evolving_user"
    
    # Create test conversations
    conversations = [
        {
            "prompt": "How do I implement authentication?",
            "response": "Authentication can be implemented using JWT tokens, OAuth2, or session-based auth. JWT is stateless and scalable, while OAuth2 provides third-party integration."
        },
        {
            "prompt": "What's the best way to handle errors?",
            "response": "Error handling should include proper logging, user-friendly messages, and graceful degradation. Use try-catch blocks and implement circuit breakers for resilience."
        },
        {
            "prompt": "How do I optimize database queries?",
            "response": "Database optimization includes indexing, query optimization, connection pooling, and caching. Use EXPLAIN to analyze query performance and add appropriate indexes."
        }
    ]
    
    # Store conversations
    trace_ids = []
    for conv in conversations:
        trace_id = semantic_embeddings.store_conversation_embedding(
            user_id, conv["prompt"], conv["response"]
        )
        trace_ids.append(trace_id)
    
    print(f"✅ Stored {len(trace_ids)} test conversations")
    
    # Test evolving context search
    test_queries = [
        "How do I secure my API?",
        "What's the best error handling strategy?",
        "How can I improve database performance?"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n🔍 Query {i+1}: {query}")
        
        # Find evolving context
        evolving_matches = self_evolving_context.find_evolving_context(
            user_id, query, limit=2, similarity_threshold=0.2
        )
        
        if evolving_matches:
            print(f"  Found {len(evolving_matches)} evolving contexts:")
            for j, (context_data, enhanced_similarity) in enumerate(evolving_matches):
                print(f"    {j+1}. Enhanced Similarity: {enhanced_similarity:.3f}")
                print(f"       Q: {context_data['prompt'][:60]}...")
        else:
            print("  No evolving contexts found")
    
    # Simulate context usage and tracking
    print(f"\n📊 Simulating context effectiveness tracking...")
    
    for trace_id in trace_ids:
        # Simulate successful usage
        self_evolving_context.track_context_effectiveness(
            user_id, [trace_id], 0.8, user_feedback=True
        )
    
    # Test adaptive weighting
    print(f"\n⚖️ Testing adaptive weighting...")
    self_evolving_context.adaptive_weighting.update_weights(user_id)
    
    # Test auto-pruning (should not prune since traces are new)
    print(f"\n🧹 Testing auto-pruning...")
    pruned_count = self_evolving_context.auto_pruning.prune_low_impact_traces(user_id)
    print(f"  Pruned {pruned_count} traces")
    
    print(f"\n✅ Self-evolving context test completed!")

if __name__ == "__main__":
    test_self_evolving_context()
```

## 🚀 Deployment Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Redis
```bash
redis-server
```

### 3. Run Tests
```bash
python test_self_evolving.py
```

### 4. Start API Server
```bash
uvicorn api:app --reload
```

### 5. Test API Endpoints
```bash
# Test evolving context generation
curl -X POST "http://localhost:8000/generate/evolving" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "How do I implement authentication?", "user_id": "test_user"}'

# Get evolving analytics
curl "http://localhost:8000/analytics/evolving/test_user"
```

## 📊 Monitoring & Analytics

### Key Metrics to Track
- **Context Relevance Score**: Average relevance of injected context
- **Recall Success Rate**: % of times context improved response
- **Token Efficiency**: Tokens saved vs. tokens injected
- **Memory Bloat Index**: Ratio of useful vs. total traces
- **Adaptation Rate**: Speed of weight updates

### Dashboard Integration
```python
@app.get("/dashboard/evolving/{user_id}")
async def get_evolving_dashboard(user_id: str):
    """Get comprehensive self-evolving dashboard data."""
    # Implementation for dashboard metrics
    pass
```

## 🔮 Next Steps

1. **Implement response quality measurement**
2. **Add user feedback collection**
3. **Create pattern recognition algorithms**
4. **Build comprehensive analytics dashboard**
5. **Add A/B testing capabilities**
6. **Implement drift detection**

---

**This implementation transforms PMT from a static semantic context system into an intelligent, self-optimizing context layer that continuously improves itself based on actual usage patterns and effectiveness.**