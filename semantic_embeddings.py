import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from redis_client import r
import pickle
import base64

class SemanticEmbeddings:
    """
    Semantic embeddings manager for context-aware AI.
    Uses sentence-transformers to generate embeddings and stores them in Redis.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic embeddings with sentence-transformers model.
        
        Args:
            model_name: Sentence transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"🧠 Loaded semantic model: {model_name}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            numpy array of the embedding
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_embedding_for_redis(self, embedding: np.ndarray) -> str:
        """
        Encode numpy array for Redis storage.
        
        Args:
            embedding: numpy array to encode
            
        Returns:
            base64 encoded string
        """
        return base64.b64encode(pickle.dumps(embedding)).decode('utf-8')
    
    def decode_embedding_from_redis(self, encoded_embedding: str) -> np.ndarray:
        """
        Decode numpy array from Redis storage.
        
        Args:
            encoded_embedding: base64 encoded string
            
        Returns:
            numpy array
        """
        return pickle.loads(base64.b64decode(encoded_embedding.encode('utf-8')))
    
    def store_conversation_embedding(self, user_id: str, prompt: str, response: str, 
                                   metadata: Dict = None) -> str:
        """
        Store conversation embedding in Redis.
        
        Args:
            user_id: User identifier
            prompt: User prompt
            response: AI response
            metadata: Additional metadata
            
        Returns:
            Embedding ID
        """
        # Generate embedding for the full conversation
        conversation_text = f"User: {prompt}\nAssistant: {response}"
        embedding = self.generate_embedding(conversation_text)
        
        # Create embedding ID
        import uuid
        embedding_id = str(uuid.uuid4())
        
        # Prepare data for Redis
        embedding_data = {
            "embedding_id": embedding_id,
            "user_id": user_id,
            "prompt": prompt,
            "response": response,
            "conversation_text": conversation_text,
            "embedding": self.encode_embedding_for_redis(embedding),
            "metadata": metadata or {},
            "model_name": self.model_name,
            "timestamp": str(np.datetime64('now'))
        }
        
        # Store in Redis with multiple keys for different access patterns
        redis_key = f"embedding:{embedding_id}"
        user_embeddings_key = f"user_embeddings:{user_id}"
        
        # Store full embedding data
        r.set(redis_key, json.dumps(embedding_data))
        
        # Add to user's embedding list
        r.lpush(user_embeddings_key, embedding_id)
        
        # Set TTL for user embeddings (30 days)
        r.expire(user_embeddings_key, 30 * 24 * 60 * 60)
        
        print(f"📦 Stored semantic embedding: {embedding_id}")
        return embedding_id
    
    def get_conversation_embedding(self, embedding_id: str) -> Optional[Dict]:
        """
        Retrieve conversation embedding from Redis.
        
        Args:
            embedding_id: Embedding identifier
            
        Returns:
            Embedding data or None if not found
        """
        redis_key = f"embedding:{embedding_id}"
        data = r.get(redis_key)
        
        if data:
            embedding_data = json.loads(data)
            # Decode embedding
            embedding_data["embedding"] = self.decode_embedding_from_redis(
                embedding_data["embedding"]
            )
            return embedding_data
        return None
    
    def get_user_embeddings(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Get all embeddings for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of embeddings to retrieve
            
        Returns:
            List of embedding data
        """
        user_embeddings_key = f"user_embeddings:{user_id}"
        embedding_ids = r.lrange(user_embeddings_key, 0, limit - 1)
        
        embeddings = []
        for embedding_id in embedding_ids:
            embedding_data = self.get_conversation_embedding(embedding_id)
            if embedding_data:
                embeddings.append(embedding_data)
        
        return embeddings
    
    def calculate_semantic_similarity(self, embedding1: np.ndarray, 
                                    embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Reshape for sklearn cosine_similarity
        emb1_reshaped = embedding1.reshape(1, -1)
        emb2_reshaped = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1_reshaped, emb2_reshaped)[0][0]
        return float(similarity)
    
    def find_semantically_similar_context(self, user_id: str, current_prompt: str, 
                                        limit: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        Find semantically similar context for a given prompt.
        
        Args:
            user_id: User identifier
            current_prompt: Current prompt to find context for
            limit: Maximum number of similar contexts to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of tuples (embedding_data, similarity_score)
        """
        # Generate embedding for current prompt
        current_embedding = self.generate_embedding(current_prompt)
        
        # Get user's conversation embeddings
        user_embeddings = self.get_user_embeddings(user_id, limit=100)
        
        # Calculate similarities
        similarities = []
        for embedding_data in user_embeddings:
            stored_embedding = embedding_data["embedding"]
            similarity = self.calculate_semantic_similarity(current_embedding, stored_embedding)
            
            if similarity >= similarity_threshold:
                similarities.append((embedding_data, similarity))
        
        # Sort by similarity (highest first) and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def semantic_context_search(self, user_id: str, current_prompt: str, 
                              limit: int = 5, similarity_threshold: float = 0.3) -> str:
        """
        Perform semantic context search and return formatted context.
        
        Args:
            user_id: User identifier
            current_prompt: Current prompt
            limit: Maximum number of contexts to include
            similarity_threshold: Minimum similarity score
            
        Returns:
            Formatted context string
        """
        similar_contexts = self.find_semantically_similar_context(
            user_id, current_prompt, limit, similarity_threshold
        )
        
        if not similar_contexts:
            return ""
        
        # Format context
        context_parts = []
        for embedding_data, similarity in similar_contexts:
            context_part = f"Previous conversation (relevance: {similarity:.2f}):\n"
            context_part += f"User: {embedding_data['prompt']}\n"
            context_part += f"Assistant: {embedding_data['response']}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def get_semantic_analytics(self, user_id: str) -> Dict:
        """
        Get semantic analytics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Analytics data
        """
        user_embeddings = self.get_user_embeddings(user_id, limit=1000)
        
        if not user_embeddings:
            return {
                "total_conversations": 0,
                "average_similarity": 0,
                "semantic_diversity": 0,
                "top_topics": []
            }
        
        # Calculate average similarity between all embeddings
        similarities = []
        for i in range(len(user_embeddings)):
            for j in range(i + 1, len(user_embeddings)):
                sim = self.calculate_semantic_similarity(
                    user_embeddings[i]["embedding"],
                    user_embeddings[j]["embedding"]
                )
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        semantic_diversity = 1 - avg_similarity  # Higher diversity = lower average similarity
        
        return {
            "total_conversations": len(user_embeddings),
            "average_similarity": float(avg_similarity),
            "semantic_diversity": float(semantic_diversity),
            "top_topics": self._extract_top_topics(user_embeddings)
        }
    
    def _extract_top_topics(self, embeddings: List[Dict]) -> List[str]:
        """
        Extract top topics from embeddings (simplified version).
        
        Args:
            embeddings: List of embedding data
            
        Returns:
            List of top topics
        """
        # Simple keyword extraction (could be enhanced with more sophisticated NLP)
        all_text = " ".join([emb["conversation_text"] for emb in embeddings])
        words = all_text.lower().split()
        
        # Filter common words and count
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        word_counts = {}
        for word in words:
            if word not in common_words and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top 5 words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, count in top_words]

# Global instance
semantic_embeddings = SemanticEmbeddings() 