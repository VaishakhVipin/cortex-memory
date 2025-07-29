#!/usr/bin/env python3
"""
🧠 Cortex - Enterprise-Grade Context-Aware AI System
Python SDK for intelligent memory management and semantic understanding.
"""

__version__ = "2.0.0"
__author__ = "Cortex Team"
__description__ = "Context that learns what matters. Memory for agents that adapt."

from .core import store_conversation, get_conversation
from .semantic_embeddings import semantic_embeddings
from .self_evolving_context import self_evolving_context
from .semantic_drift_detection import detect_semantic_drift
from .context_manager import generate_with_context, generate_with_evolving_context

__all__ = [
    "store_conversation",
    "get_conversation", 
    "semantic_embeddings",
    "self_evolving_context",
    "detect_semantic_drift",
    "generate_with_context",
    "generate_with_evolving_context"
]
