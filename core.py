#!/usr/bin/env python3
"""
🧠 Cortex Core - Central memory management system
Handles conversation storage and retrieval with semantic understanding.
"""

import uuid
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

from redis_client import r
from semantic_embeddings import semantic_embeddings

def store_conversation(user_id: str, prompt: str, response: str, 
                      metadata: Optional[Dict] = None) -> str:
    """
    Store a conversation with semantic embeddings.
    
    Args:
        user_id: User identifier
        prompt: User's prompt/question
        response: AI's response
        metadata: Additional metadata
        
    Returns:
        memory_id: Unique identifier for the stored conversation
    """
    # Generate unique memory ID
    memory_id = str(uuid.uuid4())
    
    # Store in Redis with TTL
    key = f"memory:{memory_id}"
    data = {
        "user_id": user_id,
        "prompt": prompt,
        "response": response,
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat(),
        "memory_id": memory_id
    }
    
    # Store in Redis (30 day TTL)
    r.setex(key, 30 * 24 * 60 * 60, json.dumps(data))
    
    # Store with semantic embeddings
    semantic_embeddings.store_conversation_embedding(
        user_id=user_id,
        prompt=prompt,
        response=response,
        metadata=metadata or {}
    )
    
    print(f"📦 Cortex memory logged: {key}")
    return memory_id

def get_conversation(memory_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a conversation by memory ID.
    
    Args:
        memory_id: Memory identifier
        
    Returns:
        Conversation data or None if not found
    """
    key = f"memory:{memory_id}"
    data = r.get(key)
    
    if data:
        return json.loads(data)
    return None
