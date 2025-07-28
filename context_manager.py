import json
from typing import List, Dict, Optional
from redis_client import r

def search_traces(user_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Search for traces in Redis, optionally filtered by user_id.
    
    Args:
        user_id: Optional user ID to filter traces
        limit: Maximum number of traces to return
    
    Returns:
        List of trace dictionaries
    """
    traces = []
    pattern = "trace:*"
    
    for key in r.scan_iter(match=pattern, count=100):
        try:
            data = r.get(key)
            if data:
                trace = json.loads(data)
                if user_id is None or trace.get("metadata", {}).get("user_id") == user_id:
                    traces.append(trace)
        except (json.JSONDecodeError, KeyError):
            continue
    
    return traces[:limit]

def fetch_context(user_id: str, current_prompt: str, limit: int = 5) -> str:
    """
    Fetch relevant context from past traces using semantic similarity.
    
    Args:
        user_id: User ID to fetch context for
        current_prompt: Current prompt to find relevant context
        limit: Maximum number of context snippets to include
    
    Returns:
        Formatted context string from relevant past traces
    """
    try:
        from semantic_embeddings import semantic_embeddings
        
        # Use semantic context search for better relevance
        context = semantic_embeddings.semantic_context_search(
            user_id, current_prompt, limit=limit, similarity_threshold=0.3
        )
        
        if context:
            print(f"🧠 Found semantically relevant contexts")
            return context
        
        # Fallback to recency-based search if no semantic matches
        print(f"⚠️ No semantic matches found, using recency-based search")
        traces = search_traces(user_id=user_id, limit=limit * 2)  # Get more to filter
        
        # Sort by timestamp (most recent first)
        recent = sorted(traces, key=lambda t: t.get("timestamp", ""), reverse=True)
        
        context_snippets = []
        for trace in recent[:limit]:
            prompt = trace.get("prompt", "")
            response = trace.get("response", "")
            
            if prompt and response:
                snippet = f"User: {prompt}\nAssistant: {response}\n"
                context_snippets.append(snippet)
        
        return "\n".join(context_snippets)
        
    except Exception as e:
        print(f"⚠️ Semantic context search failed: {e}, using fallback")
        # Fallback to original method
        traces = search_traces(user_id=user_id, limit=limit * 2)  # Get more to filter
        
        # Sort by timestamp (most recent first)
        recent = sorted(traces, key=lambda t: t.get("timestamp", ""), reverse=True)
        
        context_snippets = []
        for trace in recent[:limit]:
            prompt = trace.get("prompt", "")
            response = trace.get("response", "")
            
            if prompt and response:
                snippet = f"User: {prompt}\nAssistant: {response}\n"
                context_snippets.append(snippet)
        
        return "\n".join(context_snippets)

def generate_with_context(prompt: str, user_id: str) -> str:
    """
    Generate response with context from past traces.
    
    Args:
        prompt: User's current prompt
        user_id: User ID for context fetching
    
    Returns:
        Generated response from Gemini with context
    """
    from gemini_api import call_gemini_api
    from core import log_gemini
    
    # Fetch relevant context
    context = fetch_context(user_id, prompt)
    
    # Build full prompt with context
    if context:
        full_prompt = f"{context}\nUser: {prompt}\nAssistant:"
    else:
        full_prompt = f"User: {prompt}\nAssistant:"
    
    # Call Gemini API
    response = call_gemini_api(full_prompt)
    
    # Log the trace
    metadata = {
        "user_id": user_id,
        "has_context": bool(context),
        "context_length": len(context)
    }
    log_gemini(prompt, response, metadata)
    
    return response 