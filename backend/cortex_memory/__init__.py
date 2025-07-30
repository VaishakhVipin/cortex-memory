#!/usr/bin/env python3
"""
🧠 Cortex Memory - Enterprise-Grade Context-Aware AI System
Python SDK for intelligent memory management and semantic understanding.
"""

__version__ = "2.0.0"
__author__ = "Cortex Team"
__description__ = "Context that learns what matters. Memory for agents that adapt."

# Core functionality
from .core import store_conversation, get_conversation
from .semantic_embeddings import semantic_embeddings
from .self_evolving_context import self_evolving_context
from .semantic_drift_detection import detect_semantic_drift
from .context_manager import (
    generate_with_context, 
    generate_with_evolving_context,
    generate_with_hybrid_context,
    generate_with_adaptive_context,
    get_context_analytics
)

# LLM Providers
from .llm_providers import (
    llm_manager,
    call_llm_api,
    LLMProvider,
    call_gemini_api  # Backward compatibility
)

# Enhanced client for API key usage
from .client import (
    CortexClient,
    CortexError,
    AuthenticationError,
    UsageLimitError,
    RateLimitError,
    CircuitBreakerError
)

# Public API - what users will import
__all__ = [
    # Core functions
    "store_conversation",
    "get_conversation", 
    "semantic_embeddings",
    "self_evolving_context",
    "detect_semantic_drift",
    "generate_with_context",
    "generate_with_evolving_context",
    "generate_with_hybrid_context",
    "generate_with_adaptive_context",
    "get_context_analytics",
    
    # LLM Providers
    "llm_manager",
    "call_llm_api",
    "LLMProvider",
    "call_gemini_api",
    
    # Client classes
    "CortexClient",
    
    # Exceptions
    "CortexError",
    "AuthenticationError", 
    "UsageLimitError",
    "RateLimitError",
    "CircuitBreakerError",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
]

# Convenience function for quick setup
def create_client(api_key: str, base_url: str = "https://api.cortex-memory.com") -> CortexClient:
    """
    Create a Cortex client instance with the given API key.
    
    Args:
        api_key: Your Cortex API key
        base_url: API base URL (default: production)
        
    Returns:
        Configured CortexClient instance
        
    Example:
        >>> from cortex_memory import create_client
        >>> client = create_client("your-api-key-here")
        >>> response = client.generate_with_context("How do I implement auth?")
    """
    return CortexClient(api_key=api_key, base_url=base_url)

# Add convenience function to __all__
__all__.append("create_client")
