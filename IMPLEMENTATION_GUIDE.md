# 🛠 IMPLEMENTATION GUIDE
## Technical Details for Next Features

---

## 🚀 **PHASE 1: SEMANTIC CONTEXT MATCHING**

### **1.1 Add Dependencies**
```bash
pip install sentence-transformers numpy scikit-learn
```

### **1.2 Create Embedding Manager**
```python
# semantic_context.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

class SemanticContextManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
        self.embedding_cache = {}
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.model.encode(text)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def store_conversation_embedding(self, trace_id: str, prompt: str, response: str, user_id: str):
        """Store conversation with its embedding"""
        # Combine prompt and response for context
        full_text = f"User: {prompt}\nAssistant: {response}"
        embedding = self.generate_embedding(full_text)
        
        # Store in Redis with metadata
        embedding_data = {
            "embedding": embedding.tolist(),
            "prompt": prompt,
            "response": response,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        r.set(f"embedding:{trace_id}", json.dumps(embedding_data))
    
    def find_semantic_context(self, current_prompt: str, user_id: str, limit: int = 5) -> List[dict]:
        """Find semantically relevant context"""
        current_embedding = self.generate_embedding(current_prompt)
        
        # Get all embeddings for user
        embeddings = []
        for key in r.scan_iter(match=f"embedding:*"):
            data = json.loads(r.get(key))
            if data.get("user_id") == user_id:
                embeddings.append({
                    "trace_id": key.split(":")[1],
                    "embedding": np.array(data["embedding"]),
                    "prompt": data["prompt"],
                    "response": data["response"],
                    "timestamp": data["timestamp"]
                })
        
        # Calculate similarities and rank
        similarities = []
        for emb in embeddings:
            similarity = self.calculate_similarity(current_embedding, emb["embedding"])
            # Weight by recency (last 24 hours get boost)
            recency_boost = 1.0
            if (datetime.now() - datetime.fromisoformat(emb["timestamp"])).days < 1:
                recency_boost = 1.2
            
            weighted_score = similarity * recency_boost
            similarities.append((weighted_score, emb))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in similarities[:limit]]
```

### **1.3 Update Context Manager**
```python
# Update context_manager.py
from semantic_context import SemanticContextManager

class EnhancedContextManager:
    def __init__(self):
        self.semantic_manager = SemanticContextManager()
    
    def fetch_context(self, user_id: str, current_prompt: str, limit: int = 5) -> str:
        """Enhanced context fetching with semantic matching"""
        # Get semantically relevant context
        semantic_contexts = self.semantic_manager.find_semantic_context(
            current_prompt, user_id, limit
        )
        
        # Format context
        context_snippets = []
        for ctx in semantic_contexts:
            snippet = f"User: {ctx['prompt']}\nAssistant: {ctx['response']}\n"
            context_snippets.append(snippet)
        
        return "\n".join(context_snippets)
    
    def log_with_embedding(self, prompt: str, response: str, user_id: str) -> str:
        """Log conversation with semantic embedding"""
        trace_id = log_gemini(prompt, response, {"user_id": user_id})
        
        # Store embedding
        self.semantic_manager.store_conversation_embedding(
            trace_id, prompt, response, user_id
        )
        
        return trace_id
```

---

## 🎯 **PHASE 2: CONTEXT COMPRESSION**

### **2.1 Create Context Compressor**
```python
# context_compressor.py
from typing import List, Dict
import re

class ContextCompressor:
    def __init__(self, max_tokens: int = 1000):
        self.max_tokens = max_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)"""
        return len(text) // 4
    
    def compress_context(self, context_snippets: List[str]) -> str:
        """Compress context while preserving key information"""
        full_context = "\n".join(context_snippets)
        
        if self.estimate_tokens(full_context) <= self.max_tokens:
            return full_context
        
        # Use LLM to summarize
        summary_prompt = f"""
        Summarize this conversation context while preserving key information:
        
        {full_context}
        
        Requirements:
        1. Keep important details and relationships
        2. Maintain conversation flow
        3. Preserve user preferences and patterns
        4. Keep under {self.max_tokens} tokens
        
        Summary:
        """
        
        compressed = call_gemini_api(summary_prompt)
        return compressed
    
    def extract_key_entities(self, context: str) -> List[str]:
        """Extract key entities and concepts from context"""
        # Simple entity extraction (can be enhanced with NER)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context)
        return list(set(entities))
    
    def hierarchical_compress(self, contexts: List[str]) -> str:
        """Hierarchical compression for very long contexts"""
        if len(contexts) <= 3:
            return self.compress_context(contexts)
        
        # Group contexts by similarity
        groups = self.group_similar_contexts(contexts)
        
        # Compress each group
        compressed_groups = []
        for group in groups:
            compressed = self.compress_context(group)
            compressed_groups.append(compressed)
        
        # Final compression
        return self.compress_context(compressed_groups)
```

### **2.2 Update Requirements**
```txt
# requirements.txt additions
sentence-transformers==2.2.2
numpy==1.24.3
scikit-learn==1.3.0
```

---

## 🧠 **PHASE 3: MULTI-LEVEL MEMORY**

### **3.1 Create Memory Architecture**
```python
# memory_architecture.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

class MemoryLevel:
    def __init__(self, name: str, ttl_hours: Optional[int] = None):
        self.name = name
        self.ttl_hours = ttl_hours
        self.memories = []
    
    def add_memory(self, memory: dict):
        """Add memory with optional TTL"""
        memory["added_at"] = datetime.now().isoformat()
        self.memories.append(memory)
        
        # Clean expired memories
        if self.ttl_hours:
            self.clean_expired()
    
    def clean_expired(self):
        """Remove expired memories"""
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)
        self.memories = [
            m for m in self.memories 
            if datetime.fromisoformat(m["added_at"]) > cutoff
        ]

class MemoryArchitecture:
    def __init__(self):
        self.working_memory = MemoryLevel("working", ttl_hours=1)  # 1 hour
        self.short_term_memory = MemoryLevel("short_term", ttl_hours=24)  # 24 hours
        self.long_term_memory = MemoryLevel("long_term")  # No TTL
        self.semantic_memory = {}  # Knowledge graph
    
    def add_memory(self, memory: dict, level: str = "working"):
        """Add memory to appropriate level"""
        if level == "working":
            self.working_memory.add_memory(memory)
        elif level == "short_term":
            self.short_term_memory.add_memory(memory)
        elif level == "long_term":
            self.long_term_memory.add_memory(memory)
    
    def get_relevant_memories(self, query: str, limit: int = 10) -> List[dict]:
        """Get relevant memories from all levels"""
        all_memories = []
        
        # Get from working memory (highest priority)
        all_memories.extend(self.working_memory.memories)
        
        # Get from short-term memory
        all_memories.extend(self.short_term_memory.memories)
        
        # Get from long-term memory
        all_memories.extend(self.long_term_memory.memories)
        
        # Rank by relevance and recency
        ranked_memories = self.rank_memories(all_memories, query)
        
        return ranked_memories[:limit]
    
    def rank_memories(self, memories: List[dict], query: str) -> List[dict]:
        """Rank memories by relevance and recency"""
        # Simple ranking (can be enhanced with semantic similarity)
        for memory in memories:
            # Calculate recency score
            added_time = datetime.fromisoformat(memory["added_at"])
            hours_ago = (datetime.now() - added_time).total_seconds() / 3600
            recency_score = max(0, 1 - (hours_ago / 24))  # Decay over 24 hours
            
            # Calculate relevance score (simple keyword matching)
            query_words = set(query.lower().split())
            memory_text = f"{memory.get('prompt', '')} {memory.get('response', '')}"
            memory_words = set(memory_text.lower().split())
            
            relevance_score = len(query_words & memory_words) / len(query_words) if query_words else 0
            
            # Combined score
            memory["score"] = (relevance_score * 0.7) + (recency_score * 0.3)
        
        # Sort by score
        memories.sort(key=lambda x: x.get("score", 0), reverse=True)
        return memories
    
    def consolidate_memories(self, user_id: str):
        """Consolidate memories periodically"""
        # Find similar memories and merge them
        # Extract key learnings
        # Update semantic memory
        pass
```

---

## 🔧 **PHASE 4: SESSION MANAGEMENT**

### **4.1 Create Session Manager**
```python
# session_manager.py
from typing import Dict, List, Optional
import uuid
from datetime import datetime

class Session:
    def __init__(self, session_id: str, user_id: str, session_type: str = "general"):
        self.session_id = session_id
        self.user_id = user_id
        self.session_type = session_type
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.context_history = []
        self.metadata = {}
    
    def add_context(self, context: str):
        """Add context to session history"""
        self.context_history.append({
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now()
    
    def get_recent_context(self, limit: int = 5) -> List[str]:
        """Get recent context from session"""
        recent = self.context_history[-limit:] if self.context_history else []
        return [ctx["context"] for ctx in recent]

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
    
    def create_session(self, user_id: str, session_type: str = "general") -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        session = Session(session_id, user_id, session_type)
        self.sessions[session_id] = session
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for user"""
        return [s for s in self.sessions.values() if s.user_id == user_id]
    
    def switch_session(self, user_id: str, session_id: str) -> bool:
        """Switch to existing session"""
        session = self.get_session(session_id)
        if session and session.user_id == user_id:
            session.last_activity = datetime.now()
            return True
        return False
    
    def merge_sessions(self, user_id: str, session_ids: List[str]) -> str:
        """Merge multiple sessions"""
        # Create new session
        new_session_id = self.create_session(user_id, "merged")
        new_session = self.get_session(new_session_id)
        
        # Merge context from all sessions
        all_context = []
        for session_id in session_ids:
            session = self.get_session(session_id)
            if session and session.user_id == user_id:
                all_context.extend(session.context_history)
        
        # Sort by timestamp and add to new session
        all_context.sort(key=lambda x: x["timestamp"])
        new_session.context_history = all_context
        
        return new_session_id
    
    def cleanup_inactive_sessions(self, hours: int = 24):
        """Clean up inactive sessions"""
        cutoff = datetime.now() - timedelta(hours=hours)
        inactive_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.last_activity < cutoff
        ]
        
        for session_id in inactive_sessions:
            del self.sessions[session_id]
```

---

## 📊 **PHASE 5: CONTEXT ANALYTICS**

### **5.1 Create Analytics System**
```python
# context_analytics.py
from typing import Dict, List
from datetime import datetime, timedelta
import json

class ContextAnalytics:
    def __init__(self):
        self.metrics = {}
    
    def track_context_usage(self, user_id: str, context_length: int, response_quality: float):
        """Track context usage metrics"""
        if user_id not in self.metrics:
            self.metrics[user_id] = {
                "context_usage": [],
                "response_quality": [],
                "patterns": {}
            }
        
        self.metrics[user_id]["context_usage"].append({
            "length": context_length,
            "timestamp": datetime.now().isoformat()
        })
        
        self.metrics[user_id]["response_quality"].append({
            "quality": response_quality,
            "timestamp": datetime.now().isoformat()
        })
    
    def analyze_conversation_patterns(self, user_id: str) -> Dict:
        """Analyze user conversation patterns"""
        if user_id not in self.metrics:
            return {}
        
        user_metrics = self.metrics[user_id]
        
        # Analyze context usage patterns
        context_lengths = [m["length"] for m in user_metrics["context_usage"]]
        avg_context_length = sum(context_lengths) / len(context_lengths) if context_lengths else 0
        
        # Analyze response quality patterns
        qualities = [m["quality"] for m in user_metrics["response_quality"]]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0
        
        return {
            "avg_context_length": avg_context_length,
            "avg_response_quality": avg_quality,
            "total_conversations": len(user_metrics["context_usage"]),
            "context_usage_trend": self.calculate_trend(context_lengths),
            "quality_trend": self.calculate_trend(qualities)
        }
    
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        if avg_second > avg_first * 1.1:
            return "improving"
        elif avg_second < avg_first * 0.9:
            return "declining"
        else:
            return "stable"
    
    def identify_context_gaps(self, user_id: str) -> List[str]:
        """Identify where context could be improved"""
        gaps = []
        
        # Analyze recent conversations for missing context
        # This is a simplified version - can be enhanced with more sophisticated analysis
        
        return gaps
    
    def measure_context_effectiveness(self, user_id: str) -> Dict:
        """Measure how well context improves responses"""
        if user_id not in self.metrics:
            return {"effectiveness": 0, "confidence": 0}
        
        user_metrics = self.metrics[user_id]
        
        # Calculate correlation between context length and response quality
        if len(user_metrics["context_usage"]) > 1:
            context_lengths = [m["length"] for m in user_metrics["context_usage"]]
            qualities = [m["quality"] for m in user_metrics["response_quality"]]
            
            # Simple correlation calculation
            correlation = self.calculate_correlation(context_lengths, qualities)
            
            return {
                "effectiveness": correlation,
                "confidence": min(len(context_lengths) / 10, 1.0)  # Confidence based on sample size
            }
        
        return {"effectiveness": 0, "confidence": 0}
    
    def calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate simple correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0
```

---

## 🚀 **INTEGRATION GUIDE**

### **Updated Main Context Manager**
```python
# enhanced_context_manager.py
from semantic_context import SemanticContextManager
from context_compressor import ContextCompressor
from memory_architecture import MemoryArchitecture
from session_manager import SessionManager
from context_analytics import ContextAnalytics

class EnhancedContextManager:
    def __init__(self):
        self.semantic_manager = SemanticContextManager()
        self.compressor = ContextCompressor(max_tokens=1000)
        self.memory = MemoryArchitecture()
        self.sessions = SessionManager()
        self.analytics = ContextAnalytics()
    
    def generate_with_context(self, prompt: str, user_id: str, session_id: str = None) -> str:
        """Enhanced context-aware generation"""
        
        # Get or create session
        if not session_id:
            session_id = self.sessions.create_session(user_id)
        
        session = self.sessions.get_session(session_id)
        
        # Get relevant context from multiple sources
        semantic_context = self.semantic_manager.find_semantic_context(prompt, user_id, 3)
        memory_context = self.memory.get_relevant_memories(prompt, 3)
        session_context = session.get_recent_context(2) if session else []
        
        # Combine and compress context
        all_context = []
        all_context.extend([ctx["prompt"] + "\n" + ctx["response"] for ctx in semantic_context])
        all_context.extend([ctx["prompt"] + "\n" + ctx["response"] for ctx in memory_context])
        all_context.extend(session_context)
        
        compressed_context = self.compressor.compress_context(all_context)
        
        # Generate response
        if compressed_context:
            full_prompt = f"{compressed_context}\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        response = call_gemini_api(full_prompt)
        
        # Log and track
        trace_id = self.log_with_embedding(prompt, response, user_id)
        
        # Update session
        if session:
            session.add_context(compressed_context)
        
        # Track analytics
        self.analytics.track_context_usage(
            user_id, 
            len(compressed_context), 
            0.8  # Placeholder quality score
        )
        
        return response
```

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Week 1: Semantic Context Matching**
- [ ] Install sentence-transformers
- [ ] Create SemanticContextManager
- [ ] Update context fetching with semantic matching
- [ ] Test with existing conversations

### **Week 2: Context Compression**
- [ ] Create ContextCompressor
- [ ] Implement LLM-based summarization
- [ ] Add token limit management
- [ ] Test compression quality

### **Week 3: Multi-Level Memory**
- [ ] Create MemoryArchitecture
- [ ] Implement memory levels and TTL
- [ ] Add memory ranking and retrieval
- [ ] Test memory consolidation

### **Week 4: Session Management**
- [ ] Create SessionManager
- [ ] Implement session creation and switching
- [ ] Add session merging capabilities
- [ ] Test session continuity

### **Week 5: Analytics & Integration**
- [ ] Create ContextAnalytics
- [ ] Integrate all components
- [ ] Add comprehensive testing
- [ ] Performance optimization

---

## 🎯 **SUCCESS CRITERIA**

### **Technical Success:**
- Context relevance score > 0.8
- Response quality improvement > 20%
- Token efficiency improvement > 30%
- Memory retrieval < 100ms

### **User Experience Success:**
- Conversation coherence score > 0.9
- User satisfaction improvement > 25%
- Context understanding accuracy > 85%
- Conversation continuity rating > 4.5/5

This implementation guide provides the technical foundation for building the ultimate context layer for LLM applications! 🚀 