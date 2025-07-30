#!/usr/bin/env python3
"""
🧠 Cortex Semantic Embeddings - Advanced semantic understanding system
Handles embedding generation, storage, and similarity search for context retrieval.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .redis_client import r
import pickle
import base64
import time
import hashlib

# Try to import sentence-transformers, fallback to mock if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not available, using mock embeddings")

class SemanticEmbeddings:
    """
    Semantic embeddings manager for context-aware AI.
    Uses sentence-transformers to generate embeddings and stores them in Redis.
    Optimized for performance with batch processing and caching.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic embeddings with sentence-transformers model.
        
        Args:
            model_name: Sentence transformers model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("🔄 Using mock semantic embeddings (sentence-transformers not available)")
            self.model = None
            self.model_name = "mock-embeddings"
            self.use_mock = True
        else:
            try:
                print(f"🧠 Loading semantic model: {model_name}...")
                start_time = time.time()
                self.model = SentenceTransformer(model_name)
                load_time = time.time() - start_time
                self.model_name = model_name
                self.use_mock = False
                print(f"✅ Loaded semantic model in {load_time:.2f}s")
                
                # Enable model optimization
                if hasattr(self.model, 'max_seq_length'):
                    self.model.max_seq_length = 256  # Reduce for faster processing
                
            except Exception as e:
                print(f"⚠️ Failed to load sentence-transformers model: {e}")
                print("🔄 Falling back to mock semantic embeddings")
                self.model = None
                self.model_name = "mock-embeddings"
                self.use_mock = True
        
        # Performance optimizations
        self.embedding_cache = {}  # Simple in-memory cache
        self.cache_size_limit = 1000
        self.batch_size = 32  # Process multiple texts at once
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a given text with caching.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            numpy array of the embedding
        """
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        if self.use_mock:
            # Generate mock embedding using hash
            embedding = np.zeros(384)
            for i, char in enumerate(text_hash):
                if i < 384:
                    embedding[i] = ord(char) / 255.0
        else:
            # Truncate text if too long for performance
            if len(text) > 500:
                text = text[:500] + "..."
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Cache the result
        if len(self.embedding_cache) < self.cache_size_limit:
            self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently using batch processing.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of numpy arrays
        """
        if not texts:
            return []
        
        if self.use_mock:
            # Generate mock embeddings for batch
            embeddings = []
            for text in texts:
                text_hash = hashlib.md5(text.encode()).hexdigest()
                embedding = np.zeros(384)
                for i, char in enumerate(text_hash):
                    if i < 384:
                        embedding[i] = ord(char) / 255.0
                embeddings.append(embedding)
            return embeddings
        
        # Truncate texts for performance
        truncated_texts = []
        for text in texts:
            if len(text) > 500:
                truncated_texts.append(text[:500] + "...")
            else:
                truncated_texts.append(text)
        
        # Generate embeddings in batch
        embeddings = self.model.encode(truncated_texts, convert_to_numpy=True)
        
        # Cache individual embeddings
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if len(self.embedding_cache) < self.cache_size_limit:
                self.embedding_cache[text_hash] = embeddings[i]
        
        return embeddings
    
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
                                   metadata: Dict = None, skip_background_processing: bool = True) -> str:
        """
        Store conversation embedding in Redis with enterprise-grade features.
        
        Args:
            user_id: User identifier
            prompt: User prompt
            response: AI response
            metadata: Additional metadata
            skip_background_processing: Skip expensive background operations for speed
            
        Returns:
            Embedding ID
        """
        # Generate embedding for the full conversation
        conversation_text = f"User: {prompt}\nAssistant: {response}"
        embedding = self.generate_embedding(conversation_text)
        
        # Create embedding ID
        import uuid
        embedding_id = str(uuid.uuid4())
        
        # Enhanced metadata with temporal and semantic features
        enhanced_metadata = {
            "conversation_length": len(conversation_text),
            "prompt_complexity": self._calculate_complexity(prompt),
            "response_quality": self._calculate_response_quality(response),
            "semantic_density": self._calculate_semantic_density(conversation_text),
            "temporal_weight": 1.0,  # Will decay over time
            "memory_consolidation_score": 0.0,  # Will be updated
            "precision_score": 0.0,  # Will be calculated
            "recall_score": 0.0,  # Will be calculated
            "hierarchical_cluster": None,  # Will be assigned
            **(metadata or {})
        }
        
        # Prepare data for Redis
        embedding_data = {
            "embedding_id": embedding_id,
            "user_id": user_id,
            "prompt": prompt,
            "response": response,
            "conversation_text": conversation_text,
            "embedding": self.encode_embedding_for_redis(embedding),
            "metadata": enhanced_metadata,
            "model_name": self.model_name,
            "timestamp": str(np.datetime64('now')),
            "created_at": time.time()
        }
        
        # Store in Redis with multiple keys for different access patterns
        redis_key = f"embedding:{embedding_id}"
        user_embeddings_key = f"user_embeddings:{user_id}"
        
        # Store the embedding data
        r.set(redis_key, json.dumps(embedding_data))
        
        # Add to user's embedding list
        r.lpush(user_embeddings_key, embedding_id)
        
        # Set TTL for automatic cleanup (30 days)
        r.expire(redis_key, 30 * 24 * 3600)
        r.expire(user_embeddings_key, 30 * 24 * 3600)
        
        # Update memory consolidation and clustering (async-like) - skip for speed
        if not skip_background_processing:
            try:
                self._update_memory_consolidation(user_id)
                self._update_hierarchical_clustering(user_id)
            except Exception as e:
                print(f"⚠️ Background processing failed: {e}")
        
        print(f"📦 Stored enterprise semantic embedding: {embedding_id}")
        return embedding_id
    
    def run_background_processing(self, user_id: str):
        """
        Run expensive background processing operations for a user.
        This can be called separately when needed, not during fast batch operations.
        
        Args:
            user_id: User identifier
        """
        print(f"🔄 Running background processing for user: {user_id}")
        try:
            self._update_memory_consolidation(user_id)
            self._update_hierarchical_clustering(user_id)
            print(f"✅ Background processing completed for user: {user_id}")
        except Exception as e:
            print(f"⚠️ Background processing failed for {user_id}: {e}")
    
    def store_conversations_batch(self, conversations: List[Dict], skip_background_processing: bool = True) -> List[str]:
        """
        Store multiple conversations efficiently using batch processing.
        
        Args:
            conversations: List of dicts with keys: user_id, prompt, response, metadata
            skip_background_processing: Skip expensive background operations for speed
            
        Returns:
            List of embedding IDs
        """
        if not conversations:
            return []
        
        print(f"🚀 Batch processing {len(conversations)} conversations...")
        start_time = time.time()
        
        # Prepare conversation texts for batch embedding generation
        conversation_texts = []
        for conv in conversations:
            conversation_text = f"User: {conv['prompt']}\nAssistant: {conv['response']}"
            conversation_texts.append(conversation_text)
        
        # Generate embeddings in batch
        embeddings = self.generate_embeddings_batch(conversation_texts)
        
        # Process each conversation
        embedding_ids = []
        import uuid
        
        for i, conv in enumerate(conversations):
            # Create embedding ID
            embedding_id = str(uuid.uuid4())
            
            # Enhanced metadata
            conversation_text = conversation_texts[i]
            enhanced_metadata = {
                "conversation_length": len(conversation_text),
                "prompt_complexity": self._calculate_complexity(conv['prompt']),
                "response_quality": self._calculate_response_quality(conv['response']),
                "semantic_density": self._calculate_semantic_density(conversation_text),
                "temporal_weight": 1.0,
                "memory_consolidation_score": 0.0,
                "precision_score": 0.0,
                "recall_score": 0.0,
                "hierarchical_cluster": None,
                **(conv.get('metadata') or {})
            }
            
            # Prepare data for Redis
            embedding_data = {
                "embedding_id": embedding_id,
                "user_id": conv['user_id'],
                "prompt": conv['prompt'],
                "response": conv['response'],
                "conversation_text": conversation_text,
                "embedding": self.encode_embedding_for_redis(embeddings[i]),
                "metadata": enhanced_metadata,
                "model_name": self.model_name,
                "timestamp": str(np.datetime64('now')),
                "created_at": time.time()
            }
            
            # Store in Redis
            redis_key = f"embedding:{embedding_id}"
            user_embeddings_key = f"user_embeddings:{conv['user_id']}"
            
            r.set(redis_key, json.dumps(embedding_data))
            r.lpush(user_embeddings_key, embedding_id)
            r.expire(redis_key, 30 * 24 * 3600)
            r.expire(user_embeddings_key, 30 * 24 * 3600)
            
            embedding_ids.append(embedding_id)
        
        # Skip expensive background processing for speed
        if not skip_background_processing:
            # Update memory consolidation and clustering for all users (expensive operations)
            user_ids = list(set(conv['user_id'] for conv in conversations))
            for user_id in user_ids:
                try:
                    self._update_memory_consolidation(user_id)
                    self._update_hierarchical_clustering(user_id)
                except Exception as e:
                    print(f"⚠️ Background processing failed for {user_id}: {e}")
        else:
            print("⚡ Skipping background processing for speed")
        
        processing_time = time.time() - start_time
        print(f"✅ Batch processed {len(conversations)} conversations in {processing_time:.2f}s ({processing_time/len(conversations):.3f}s per conversation)")
        
        return embedding_ids
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        complexity = (avg_word_length * 0.4 + len(words) * 0.3 + sentence_count * 0.3) / 100
        return min(complexity, 1.0)
    
    def _calculate_response_quality(self, response: str) -> float:
        """Calculate response quality score."""
        if "Error" in response or len(response) < 10:
            return 0.1
        
        # Simple quality heuristics
        has_structure = any(char in response for char in ['*', '-', '1.', '2.'])
        has_details = len(response) > 100
        has_examples = any(word in response.lower() for word in ['example', 'instance', 'such as'])
        
        quality = 0.3 + (0.2 if has_structure else 0) + (0.3 if has_details else 0) + (0.2 if has_examples else 0)
        return min(quality, 1.0)
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density (unique concepts per word)."""
        words = text.lower().split()
        unique_words = len(set(words))
        return unique_words / len(words) if words else 0
    
    def _update_memory_consolidation(self, user_id: str):
        """Update memory consolidation scores based on temporal decay and relevance."""
        user_embeddings = self.get_user_embeddings(user_id, limit=100)
        temporal_key = f"temporal:{user_id}"
        
        current_time = time.time()
        
        for embedding_data in user_embeddings:
            embedding_id = embedding_data["embedding_id"]
            created_at = embedding_data.get("created_at", current_time)
            
            # Calculate temporal decay (exponential decay)
            time_diff = current_time - created_at
            decay_factor = np.exp(-time_diff / (7 * 24 * 3600))  # 7 days half-life
            
            # Calculate consolidation score based on usage and relevance
            usage_count = r.get(f"usage:{embedding_id}") or 0
            usage_score = min(int(usage_count) / 10, 1.0)  # Normalize to 0-1
            
            # Combine temporal decay with usage
            consolidation_score = (decay_factor * 0.6 + usage_score * 0.4)
            
            # Update metadata
            embedding_data["metadata"]["temporal_weight"] = float(decay_factor)
            embedding_data["metadata"]["memory_consolidation_score"] = float(consolidation_score)
            
            # Update in Redis (remove numpy array before JSON serialization)
            redis_key = f"embedding:{embedding_id}"
            # Create a copy without the numpy array for JSON serialization
            json_safe_data = embedding_data.copy()
            json_safe_data["embedding"] = self.encode_embedding_for_redis(embedding_data["embedding"])
            r.set(redis_key, json.dumps(json_safe_data))
    
    def _update_hierarchical_clustering(self, user_id: str):
        """Update hierarchical semantic clustering."""
        user_embeddings = self.get_user_embeddings(user_id, limit=100)
        
        if len(user_embeddings) < 3:
            return
        
        # Extract embeddings for clustering
        embeddings = [emb["embedding"] for emb in user_embeddings]
        embeddings_array = np.array(embeddings)
        
        # Perform hierarchical clustering
        from sklearn.cluster import AgglomerativeClustering
        
        # Determine optimal number of clusters
        n_clusters = min(5, max(2, len(embeddings) // 3))
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        
        cluster_labels = clustering.fit_predict(embeddings_array)
        
        # Update cluster assignments
        for i, embedding_data in enumerate(user_embeddings):
            embedding_data["metadata"]["hierarchical_cluster"] = int(cluster_labels[i])
            
            # Update in Redis (remove numpy array before JSON serialization)
            redis_key = f"embedding:{embedding_data['embedding_id']}"
            # Create a copy without the numpy array for JSON serialization
            json_safe_data = embedding_data.copy()
            json_safe_data["embedding"] = self.encode_embedding_for_redis(embedding_data["embedding"])
            r.set(redis_key, json.dumps(json_safe_data))
    
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
        Get user embeddings using optimized Redis pipelining for speed.
        """
        try:
            # Get embedding IDs for user using pipelining
            pipe = r.pipeline()
            pipe.lrange(f"user_embeddings:{user_id}", 0, limit - 1)
            embedding_ids = pipe.execute()[0]
            
            if not embedding_ids:
                return []
            
            # Batch get all embeddings using pipelining
            pipe = r.pipeline()
            for embedding_id in embedding_ids:
                pipe.get(f"embedding:{embedding_id}")
            
            embedding_data_list = pipe.execute()
            
            # Process results
            embeddings = []
            for i, embedding_data in enumerate(embedding_data_list):
                if embedding_data:
                    try:
                        data = json.loads(embedding_data)
                        data['embedding_id'] = embedding_ids[i]
                        
                        # Decode embedding from string to numpy array
                        if isinstance(data.get('embedding'), str):
                            data['embedding'] = self.decode_embedding_from_redis(data['embedding'])
                        
                        embeddings.append(data)
                    except json.JSONDecodeError:
                        continue
            
            return embeddings
            
        except Exception as e:
            print(f"⚠️ Error getting user embeddings: {e}")
            return []
    
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
        Find semantically similar context using optimized search.
        Much faster than previous version.
        """
        try:
            # Get user embeddings (limit to recent 20 for speed)
            user_embeddings = self.get_user_embeddings(user_id, limit=20)
            
            if not user_embeddings:
                return []
            
            # Generate embedding for current prompt
            current_embedding = self.generate_embedding(current_prompt)
            
            # Fast similarity calculation
            similarities = []
            for embedding_data in user_embeddings:
                try:
                    # Embedding is already decoded by get_conversation_embedding
                    stored_embedding = embedding_data["embedding"]
                    
                    # Calculate base similarity
                    base_similarity = self.calculate_semantic_similarity(current_embedding, stored_embedding)
                    
                    # Quick enhancement (no expensive operations)
                    enhanced_similarity = self._calculate_quick_similarity(
                        embedding_data, base_similarity, current_prompt
                    )
                    
                    # Lower threshold for better context finding
                    if enhanced_similarity >= similarity_threshold * 0.5:  # More lenient threshold
                        similarities.append((embedding_data, enhanced_similarity))
                        
                except Exception as e:
                    continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            print(f"⚠️ Semantic context search failed: {e}")
            return []
    
    def _calculate_quick_similarity(self, embedding_data: Dict, base_similarity: float, current_prompt: str) -> float:
        """
        Calculate enhanced similarity quickly without expensive operations.
        """
        try:
            metadata = embedding_data.get("metadata", {})
            
            # Get basic enhancement factors
            temporal_weight = metadata.get("temporal_weight", 1.0)
            quality_boost = metadata.get("response_quality", 0.5)
            
            # Calculate enhanced similarity (simplified)
            enhanced_similarity = (
                base_similarity * 0.8 +
                temporal_weight * 0.1 +
                quality_boost * 0.1
            )
            
            return max(0.0, min(1.0, enhanced_similarity))
            
        except Exception:
            return base_similarity
    
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
        Get enterprise-grade semantic analytics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Enhanced analytics data
        """
        user_embeddings = self.get_user_embeddings(user_id, limit=1000)
        
        if not user_embeddings:
            return {
                "total_conversations": 0,
                "average_similarity": 0,
                "semantic_diversity": 0,
                "top_topics": [],
                "memory_consolidation_score": 0,
                "precision_recall_metrics": {"precision": 0, "recall": 0, "f1_score": 0},
                "hierarchical_clusters": 0,
                "temporal_decay_rate": 0,
                "response_quality_score": 0,
                "semantic_density_score": 0
            }
        
        # Calculate basic metrics
        similarities = []
        consolidation_scores = []
        quality_scores = []
        density_scores = []
        temporal_weights = []
        cluster_ids = set()
        
        for i in range(len(user_embeddings)):
            for j in range(i + 1, len(user_embeddings)):
                sim = self.calculate_semantic_similarity(
                    user_embeddings[i]["embedding"],
                    user_embeddings[j]["embedding"]
                )
                similarities.append(sim)
        
        # Extract metadata metrics
        for emb in user_embeddings:
            metadata = emb.get("metadata", {})
            consolidation_scores.append(metadata.get("memory_consolidation_score", 0.5))
            quality_scores.append(metadata.get("response_quality", 0.5))
            density_scores.append(metadata.get("semantic_density", 0.5))
            temporal_weights.append(metadata.get("temporal_weight", 1.0))
            
            cluster_id = metadata.get("hierarchical_cluster")
            if cluster_id is not None:
                cluster_ids.add(cluster_id)
        
        # Calculate enhanced metrics
        avg_similarity = np.mean(similarities) if similarities else 0
        semantic_diversity = 1 - avg_similarity
        avg_consolidation = np.mean(consolidation_scores) if consolidation_scores else 0
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        avg_density = np.mean(density_scores) if density_scores else 0
        avg_temporal = np.mean(temporal_weights) if temporal_weights else 1.0
        
        # Get precision-recall metrics
        precision_recall = self.get_precision_recall_metrics(user_id)
        
        return {
            "total_conversations": len(user_embeddings),
            "average_similarity": float(avg_similarity),
            "semantic_diversity": float(semantic_diversity),
            "top_topics": self._extract_top_topics(user_embeddings),
            "memory_consolidation_score": float(avg_consolidation),
            "precision_recall_metrics": precision_recall,
            "hierarchical_clusters": len(cluster_ids),
            "temporal_decay_rate": float(1.0 - avg_temporal),
            "response_quality_score": float(avg_quality),
            "semantic_density_score": float(avg_density),
            "enterprise_grade_score": float(
                (avg_similarity * 0.2 + 
                 semantic_diversity * 0.15 + 
                 avg_consolidation * 0.2 + 
                 precision_recall["f1_score"] * 0.25 + 
                 avg_quality * 0.1 + 
                 avg_density * 0.1)
            )
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

    def get_precision_recall_metrics(self, user_id: str) -> Dict:
        """Calculate precision and recall metrics for semantic search."""
        user_embeddings = self.get_user_embeddings(user_id, limit=100)
        
        if len(user_embeddings) < 2:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # Simulate search queries and calculate metrics
        test_queries = [
            "What is the main topic?",
            "How does this work?",
            "What are the benefits?",
            "What are the challenges?",
            "What are the best practices?"
        ]
        
        precision_scores = []
        recall_scores = []
        
        for query in test_queries:
            # Find similar contexts
            similar_contexts = self.find_semantically_similar_context(
                user_id, query, limit=3, similarity_threshold=0.3
            )
            
            if similar_contexts:
                # Calculate precision (relevance of retrieved items)
                avg_similarity = np.mean([sim for _, sim in similar_contexts])
                precision_scores.append(avg_similarity)
                
                # Calculate recall (coverage of relevant items)
                total_relevant = len([emb for emb in user_embeddings 
                                   if self.calculate_semantic_similarity(
                                       self.generate_embedding(query), 
                                       emb["embedding"]
                                   ) >= 0.3])
                
                if total_relevant > 0:
                    recall = len(similar_contexts) / total_relevant
                    recall_scores.append(recall)
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0.0
        avg_recall = np.mean(recall_scores) if recall_scores else 0.0
        
        # Calculate F1 score
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            "precision": float(avg_precision),
            "recall": float(avg_recall),
            "f1_score": float(f1_score)
        }

# Global instance
semantic_embeddings = SemanticEmbeddings()

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
    from backend.cortex.self_evolving_context import self_evolving_context
    
    return self_evolving_context.find_evolving_context(
        user_id, current_prompt, limit, similarity_threshold
    )

# Add the method to the class
SemanticEmbeddings.find_evolving_semantic_context = find_evolving_semantic_context 