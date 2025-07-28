# core.py
import redis
import uuid
import json
from datetime import datetime
from config import REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD

# Redis connection
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    username=REDIS_USERNAME,
    password=REDIS_PASSWORD,
)

def log_gemini(prompt: str, gemini_response: str, metadata: dict = None):
    trace_id = str(uuid.uuid4())
    data = {
        "llm": "gemini-2.0-flash",
        "prompt": prompt,
        "response": gemini_response,
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat()
    }
    key = f"trace:{trace_id}"
    r.set(key, json.dumps(data))
    
    # Store semantic embedding if user_id is provided
    user_id = metadata.get("user_id") if metadata else None
    if user_id:
        try:
            from semantic_embeddings import semantic_embeddings
            embedding_id = semantic_embeddings.store_conversation_embedding(
                user_id, prompt, gemini_response, metadata
            )
            data["embedding_id"] = embedding_id
            print(f"🧠 Semantic embedding stored: {embedding_id}")
        except Exception as e:
            print(f"⚠️ Failed to store semantic embedding: {e}")
    
    print(f"📦 Gemini trace logged: {key}")
    return trace_id
