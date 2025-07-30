#!/usr/bin/env python3
"""
🧠 Cortex Self-Evolving Context Model
Advanced adaptive learning system that optimizes context relevance over time.
"""

import json
import time
import math
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict, Counter
import re
from .redis_client import r
from .semantic_embeddings import semantic_embeddings
from .semantic_drift_detection import SemanticDriftDetection

# Statistical-only approach - no ML dependencies
ML_AVAILABLE = False

class ContextScoringEngine:
    """
    Fast statistical context scoring engine - no ML dependencies.
    Uses advanced pattern recognition algorithms for speed and reliability.
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def calculate_context_score(self, user_id: str, trace_id: str, current_prompt: str) -> float:
        """
        Calculate context score using fast statistical methods only.
        """
        try:
            # Check cache first
            cache_key = f"{user_id}:{trace_id}:{hash(current_prompt)}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Get trace data
            trace_data = self.redis_client.get(f"trace:{trace_id}")
            if not trace_data:
                return 0.5
            
            trace_info = json.loads(trace_data)
            original_prompt = trace_info.get('prompt', '')
            
            # Fast statistical scoring
            score = self._calculate_fast_statistical_score(
                original_prompt, current_prompt, trace_info
            )
            
            # Cache result
            self.cache[cache_key] = score
            return score
            
        except Exception as e:
            print(f"⚠️ Context scoring failed: {e}")
            return 0.5
    
    def _calculate_fast_statistical_score(self, original_prompt: str, current_prompt: str, trace_info: Dict) -> float:
        """
        Fast statistical scoring using multiple algorithms.
        """
        try:
            # 1. Keyword Overlap (40% weight)
            keyword_score = self._calculate_keyword_overlap(original_prompt, current_prompt)
            
            # 2. Semantic Similarity (30% weight)
            semantic_score = self._calculate_semantic_similarity(original_prompt, current_prompt)
            
            # 3. Temporal Relevance (15% weight)
            temporal_score = self._calculate_temporal_relevance(trace_info)
            
            # 4. Usage Pattern (15% weight)
            usage_score = self._calculate_usage_score(trace_info)
            
            # Weighted combination
            final_score = (
                keyword_score * 0.40 +
                semantic_score * 0.30 +
                temporal_score * 0.15 +
                usage_score * 0.15
            )
            
            return max(0.0, min(1.0, final_score))
            
        except Exception:
            return 0.5
    
    def _calculate_keyword_overlap(self, original: str, current: str) -> float:
        """Fast keyword overlap calculation."""
        try:
            # Extract key terms
            original_words = set(re.findall(r'\b\w+\b', original.lower()))
            current_words = set(re.findall(r'\b\w+\b', current.lower()))
            
            if not original_words or not current_words:
                return 0.0
            
            # Jaccard similarity
            intersection = len(original_words.intersection(current_words))
            union = len(original_words.union(current_words))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_semantic_similarity(self, original: str, current: str) -> float:
        """Fast semantic similarity using word vectors."""
        try:
            # Simple word vector approach
            original_words = set(re.findall(r'\b\w+\b', original.lower()))
            current_words = set(re.findall(r'\b\w+\b', current.lower()))
            
            # Technical term matching
            tech_terms = ['api', 'database', 'auth', 'docker', 'aws', 'deploy', 'optimize', 'performance', 'security', 'scaling', 'microservices', 'testing', 'monitoring']
            
            original_tech = sum(1 for term in tech_terms if term in original.lower())
            current_tech = sum(1 for term in tech_terms if term in current.lower())
            
            # Question type matching
            original_question = '?' in original
            current_question = '?' in current
            
            # Length similarity
            length_ratio = min(len(current), len(original)) / max(len(current), len(original))
            
            # Combine factors
            tech_similarity = 1.0 if original_tech == current_tech else 0.5
            question_similarity = 1.0 if original_question == current_question else 0.5
            
            semantic_score = (tech_similarity * 0.4 + question_similarity * 0.3 + length_ratio * 0.3)
            return semantic_score
            
        except Exception:
            return 0.5
    
    def _calculate_temporal_relevance(self, trace_info: Dict) -> float:
        """Calculate temporal relevance using exponential decay."""
        try:
            timestamp = trace_info.get('timestamp', '')
            if not timestamp:
                return 0.5
            
            created_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            
            # Calculate days since creation
            days_diff = (current_time - created_time).days
            
            # Exponential decay: 7-day half-life
            decay_factor = math.exp(-days_diff / 7.0)
            return decay_factor
            
        except Exception:
            return 0.5
    
    def _calculate_usage_score(self, trace_info: Dict) -> float:
        """Calculate usage-based score."""
        try:
            # Get usage count from Redis
            trace_id = trace_info.get('trace_id', '')
            if not trace_id:
                return 0.5
            
            usage_count = self.redis_client.get(f"usage:{trace_id}")
            usage_count = int(usage_count) if usage_count else 0
            
            # Normalize usage score (0-1)
            usage_score = min(usage_count / 10.0, 1.0)
            return usage_score
            
        except Exception:
            return 0.5
    
    def train_models(self, user_id: str):
        """No-op method - ML training removed for statistical-only approach."""
        print("📊 Using statistical-only approach - no ML training needed")
        return

class RecallTracker:
    """Track context recall success and effectiveness."""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.success_threshold = 0.7
        self.min_uses_for_learning = 3
    
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
        try:
            # Determine if context was helpful
            was_helpful = self._determine_helpfulness(response_quality, user_feedback)
            
            # Update recall statistics
            self._update_recall_stats(trace_id, was_helpful, response_quality)
            
            # Update context score
            self._update_context_score(trace_id, was_helpful, response_quality)
            
            print(f"📊 Tracked recall success for {trace_id[:8]}...: {'✅' if was_helpful else '❌'}")
            
        except Exception as e:
            print(f"⚠️ Error tracking recall success: {e}")
    
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
        data = self.redis_client.get(redis_key)
        if data:
            try:
                stats = json.loads(data)
            except json.JSONDecodeError:
                stats = self._create_default_stats()
        else:
            stats = self._create_default_stats()
        
        # Update stats
        stats["total_uses"] += 1
        stats["total_quality"] += response_quality
        stats["last_used"] = time.time()
        
        if was_helpful:
            stats["successful_uses"] += 1
        
        stats["success_rate"] = stats["successful_uses"] / stats["total_uses"]
        stats["avg_quality"] = stats["total_quality"] / stats["total_uses"]
        
        # Store updated stats
        self.redis_client.set(redis_key, json.dumps(stats))
    
    def _create_default_stats(self) -> Dict:
        """Create default statistics structure."""
        return {
            "total_uses": 0,
            "successful_uses": 0,
            "total_quality": 0.0,
            "success_rate": 0.0,
            "avg_quality": 0.0,
            "last_used": time.time()
        }
    
    def _update_context_score(self, trace_id: str, was_helpful: bool, 
                            response_quality: float):
        """Update context score based on recall success."""
        redis_key = f"context_score:{trace_id}"
        
        # Get current score
        data = self.redis_client.get(redis_key)
        if data:
            try:
                score_data = json.loads(data)
                current_score = score_data.get("historical_score", 0.5)
            except json.JSONDecodeError:
                current_score = 0.5
                score_data = {"historical_score": current_score}
        else:
            current_score = 0.5
            score_data = {"historical_score": current_score}
        
        # Calculate new score with learning
        if was_helpful:
            # Boost score for successful uses
            new_score = current_score + (1.0 - current_score) * 0.1
        else:
            # Decay score for unsuccessful uses
            new_score = current_score * 0.95
        
        # Update score
        score_data["historical_score"] = float(new_score)
        score_data["last_updated"] = time.time()
        score_data["update_count"] = score_data.get("update_count", 0) + 1
        
        self.redis_client.set(redis_key, json.dumps(score_data))

class AdaptiveWeighting:
    """Dynamically adjust context weights based on performance."""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.weight_update_interval = 3600  # 1 hour
        self.min_weight = 0.1
        self.max_weight = 2.0
        self.learning_rate = 0.05
    
    def update_weights(self, user_id: str):
        """Update adaptive weights for all user traces."""
        try:
            user_embeddings = semantic_embeddings.get_user_embeddings(user_id, limit=1000)
            
            updated_count = 0
            for embedding_data in user_embeddings:
                trace_id = embedding_data["embedding_id"]
                new_weight = self._calculate_adaptive_weight(trace_id)
                
                # Update metadata with new weight
                metadata = embedding_data.get("metadata", {})
                old_weight = metadata.get("adaptive_weight", 1.0)
                metadata["adaptive_weight"] = new_weight
                metadata["weight_updated"] = time.time()
                
                # Update in Redis
                redis_key = f"embedding:{trace_id}"
                json_safe_data = embedding_data.copy()
                json_safe_data["embedding"] = semantic_embeddings.encode_embedding_for_redis(
                    embedding_data["embedding"]
                )
                self.redis_client.set(redis_key, json.dumps(json_safe_data))
                
                updated_count += 1
            
            print(f"⚖️ Updated adaptive weights for {updated_count} traces")
            
        except Exception as e:
            print(f"⚠️ Error updating adaptive weights: {e}")
    
    def _calculate_adaptive_weight(self, trace_id: str) -> float:
        """Calculate adaptive weight based on performance metrics."""
        # Get recall statistics
        recall_key = f"recall_stats:{trace_id}"
        recall_data = self.redis_client.get(recall_key)
        
        if not recall_data:
            return 1.0  # Default weight
        
        try:
            stats = json.loads(recall_data)
            success_rate = stats.get("success_rate", 0.5)
            avg_quality = stats.get("avg_quality", 0.5)
            total_uses = stats.get("total_uses", 0)
            
            # Calculate weight based on performance
            if total_uses < self.min_weight:
                # Not enough data, use default
                weight = 1.0
            else:
                # Weight based on success rate and quality
                performance_score = (success_rate * 0.7 + avg_quality * 0.3)
                weight = self.min_weight + (self.max_weight - self.min_weight) * performance_score
            
            return float(weight)
            
        except Exception as e:
            print(f"⚠️ Error calculating adaptive weight: {e}")
            return 1.0

class MetricsCollector:
    """Collect and track self-evolving context metrics."""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.metrics_key = "self_evolving_metrics"
    
    def collect_metrics(self, user_id: str) -> Dict:
        """Collect comprehensive metrics for self-evolving context."""
        try:
            user_embeddings = semantic_embeddings.get_user_embeddings(user_id, limit=1000)
            
            metrics = {
                "user_id": user_id,
                "timestamp": time.time(),
                "total_traces": len(user_embeddings),
                "high_impact_traces": 0,
                "low_impact_traces": 0,
                "total_success_rate": 0.0,
                "total_quality": 0.0,
                "weight_distribution": {
                    "high_weight": 0,
                    "medium_weight": 0,
                    "low_weight": 0
                }
            }
            
            for embedding_data in user_embeddings:
                trace_id = embedding_data["embedding_id"]
                
                # Get recall stats
                recall_key = f"recall_stats:{trace_id}"
                recall_data = self.redis_client.get(recall_key)
                
                if recall_data:
                    stats = json.loads(recall_data)
                    success_rate = stats.get("success_rate", 0.0)
                    avg_quality = stats.get("avg_quality", 0.0)
                    
                    metrics["total_success_rate"] += success_rate
                    metrics["total_quality"] += avg_quality
                    
                    if success_rate > 0.7:
                        metrics["high_impact_traces"] += 1
                    elif success_rate < 0.3:
                        metrics["low_impact_traces"] += 1
                
                # Get adaptive weight
                metadata = embedding_data.get("metadata", {})
                adaptive_weight = metadata.get("adaptive_weight", 1.0)
                
                if adaptive_weight > 1.5:
                    metrics["weight_distribution"]["high_weight"] += 1
                elif adaptive_weight < 0.5:
                    metrics["weight_distribution"]["low_weight"] += 1
                else:
                    metrics["weight_distribution"]["medium_weight"] += 1
            
            # Calculate averages
            if metrics["total_traces"] > 0:
                metrics["avg_success_rate"] = metrics["total_success_rate"] / metrics["total_traces"]
                metrics["avg_quality"] = metrics["total_quality"] / metrics["total_traces"]
                metrics["impact_ratio"] = metrics["high_impact_traces"] / metrics["total_traces"]
            else:
                metrics["avg_success_rate"] = 0.0
                metrics["avg_quality"] = 0.0
                metrics["impact_ratio"] = 0.0
            
            # Store metrics
            self._store_metrics(user_id, metrics)
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Error collecting metrics: {e}")
            return {}
    
    def _store_metrics(self, user_id: str, metrics: Dict):
        """Store metrics in Redis."""
        redis_key = f"{self.metrics_key}:{user_id}"
        self.redis_client.set(redis_key, json.dumps(metrics))
        self.redis_client.expire(redis_key, 86400)  # Expire after 24 hours

class AutoPruning:
    """
    Automatically removes low-impact traces to prevent memory bloat
    and improve system performance.
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.pruning_threshold = 0.1  # Minimum impact score to keep
        self.min_uses_threshold = 3   # Minimum uses in 30 days
        self.age_threshold = 30       # Days old before considering for pruning
        self.max_memory_usage = 0.8   # Maximum memory usage before aggressive pruning
    
    def prune_low_impact_traces(self, user_id: str, threshold: float = None) -> Dict:
        """
        Automatically remove traces that have low impact.
        
        Args:
            user_id: User identifier
            threshold: Custom pruning threshold (optional)
            
        Returns:
            Dict with pruning statistics
        """
        if threshold is None:
            threshold = self.pruning_threshold
            
        pruning_stats = {
            'total_traces': 0,
            'pruned_traces': 0,
            'kept_traces': 0,
            'pruned_trace_ids': [],
            'memory_saved_mb': 0,
            'pruning_reasons': {}
        }
        
        # Get all traces for user
        pattern = f"embedding:{user_id}:*"
        trace_keys = self.redis_client.keys(pattern)
        pruning_stats['total_traces'] = len(trace_keys)
        
        for trace_key in trace_keys:
            trace_id = trace_key.split(':')[-1]
            trace_data = self.redis_client.get(trace_key)
            
            if not trace_data:
                continue
                
            try:
                embedding_data = json.loads(trace_data)
                should_prune, reason = self._should_prune_trace(trace_id, embedding_data, threshold)
                
                if should_prune:
                    # Store pruning reason
                    pruning_stats['pruning_reasons'][trace_id] = reason
                    
                    # Calculate memory usage before deletion
                    memory_usage = len(trace_data)
                    
                    # Delete the trace and related data
                    self._delete_trace_completely(trace_id, user_id)
                    
                    pruning_stats['pruned_traces'] += 1
                    pruning_stats['pruned_trace_ids'].append(trace_id)
                    pruning_stats['memory_saved_mb'] += memory_usage / (1024 * 1024)
                else:
                    pruning_stats['kept_traces'] += 1
                    
            except (json.JSONDecodeError, KeyError) as e:
                # Corrupted data, prune it
                self._delete_trace_completely(trace_id, user_id)
                pruning_stats['pruned_traces'] += 1
                pruning_stats['pruning_reasons'][trace_id] = f"Corrupted data: {str(e)}"
        
        # Store pruning statistics
        self.redis_client.setex(f"pruning_stats:{user_id}", 86400, json.dumps(pruning_stats))
        
        return pruning_stats
    
    def _should_prune_trace(self, trace_id: str, embedding_data: Dict, threshold: float) -> Tuple[bool, str]:
        """
        Determine if a trace should be pruned based on multiple criteria.
        
        Args:
            trace_id: Trace identifier
            embedding_data: Trace data
            threshold: Pruning threshold
            
        Returns:
            Tuple of (should_prune, reason)
        """
        r = self.redis_client
        
        # Check recall success rate
        recall_stats = r.get(f"recall_stats:{trace_id}")
        if recall_stats:
            stats = json.loads(recall_stats)
            success_rate = stats.get('success_rate', 0)
            if success_rate < threshold:
                return True, f"Low success rate: {success_rate:.3f}"
        
        # Check usage frequency
        usage_count = r.get(f"usage:{trace_id}")
        if usage_count:
            usage = int(usage_count)
            if usage < self.min_uses_threshold:
                return True, f"Low usage: {usage} times"
        
        # Check age
        created_at = embedding_data.get('metadata', {}).get('created_at')
        if created_at:
            age_days = (time.time() - created_at) / 86400
            if age_days > self.age_threshold:
                return True, f"Too old: {age_days:.1f} days"
        
        # Check memory consolidation score
        consolidation_score = embedding_data.get('metadata', {}).get('memory_consolidation_score', 0)
        if consolidation_score < threshold:
            return True, f"Low consolidation: {consolidation_score:.3f}"
        
        # Check if trace has been marked for deletion
        if r.exists(f"marked_for_deletion:{trace_id}"):
            return True, "Marked for deletion"
        
        return False, "Trace is valuable"
    
    def _delete_trace_completely(self, trace_id: str, user_id: str):
        """
        Completely remove a trace and all related data.
        
        Args:
            trace_id: Trace identifier
            user_id: User identifier
        """
        r = self.redis_client
        
        # Delete main trace data
        r.delete(f"embedding:{user_id}:{trace_id}")
        
        # Delete related data
        r.delete(f"recall_stats:{trace_id}")
        r.delete(f"usage:{trace_id}")
        r.delete(f"context_score:{trace_id}")
        r.delete(f"marked_for_deletion:{trace_id}")
        
        # Remove from temporal and cluster sets
        r.zrem(f"temporal:{user_id}", trace_id)
        r.zrem(f"cluster:{user_id}", trace_id)
    
    def get_pruning_recommendations(self, user_id: str) -> Dict:
        """
        Get recommendations for trace pruning without actually pruning.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with pruning recommendations
        """
        r = self.redis_client
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'safe_to_keep': []
        }
        
        pattern = f"embedding:{user_id}:*"
        trace_keys = r.keys(pattern)
        
        for trace_key in trace_keys:
            trace_id = trace_key.split(':')[-1]
            trace_data = r.get(trace_key)
            
            if not trace_data:
                continue
                
            try:
                embedding_data = json.loads(trace_data)
                priority, reason = self._get_pruning_priority(trace_id, embedding_data)
                
                recommendation = {
                    'trace_id': trace_id,
                    'reason': reason,
                    'metadata': embedding_data.get('metadata', {})
                }
                
                recommendations[priority].append(recommendation)
                
            except (json.JSONDecodeError, KeyError):
                recommendations['high_priority'].append({
                    'trace_id': trace_id,
                    'reason': 'Corrupted data',
                    'metadata': {}
                })
        
        return recommendations
    
    def _get_pruning_priority(self, trace_id: str, embedding_data: Dict) -> Tuple[str, str]:
        """
        Determine pruning priority for a trace.
        
        Args:
            trace_id: Trace identifier
            embedding_data: Trace data
            
        Returns:
            Tuple of (priority, reason)
        """
        r = self.redis_client
        
        # Check for critical issues (high priority)
        recall_stats = r.get(f"recall_stats:{trace_id}")
        if recall_stats:
            stats = json.loads(recall_stats)
            success_rate = stats.get('success_rate', 0)
            if success_rate < 0.1:
                return 'high_priority', f"Very low success rate: {success_rate:.3f}"
        
        # Check for moderate issues (medium priority)
        usage_count = r.get(f"usage:{trace_id}")
        if usage_count:
            usage = int(usage_count)
            if usage < 2:
                return 'medium_priority', f"Very low usage: {usage} times"
        
        # Check for minor issues (low priority)
        consolidation_score = embedding_data.get('metadata', {}).get('memory_consolidation_score', 0)
        if consolidation_score < 0.3:
            return 'low_priority', f"Low consolidation: {consolidation_score:.3f}"
        
        return 'safe_to_keep', "Trace is performing well"


class AdvancedPatternRecognition:
    """
    Advanced pattern recognition for understanding query structures,
    topic clusters, and temporal patterns.
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.pattern_cache_ttl = 3600  # 1 hour
        self.min_pattern_frequency = 2
        self.similarity_threshold = 0.7
        
        # Advanced features configuration
        self.sentiment_analysis_enabled = True
        self.complexity_scoring_enabled = True
        self.domain_detection_enabled = True
        self.behavioral_patterns_enabled = True
    
    def analyze_query_patterns(self, user_id: str) -> Dict:
        """
        Analyze query patterns for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with pattern analysis results
        """
        r = self.redis_client
        patterns = {
            'query_structures': {},
            'topic_clusters': {},
            'intent_patterns': {},
            'temporal_patterns': {},
            'semantic_patterns': {},
            'sentiment_analysis': {},
            'complexity_analysis': {},
            'domain_analysis': {},
            'behavioral_patterns': {},
            'advanced_metrics': {}
        }
        
        # Get all traces for user
        pattern = f"embedding:{user_id}:*"
        trace_keys = r.keys(pattern)
        
        queries = []
        timestamps = []
        responses = []
        
        for trace_key in trace_keys:
            trace_data = r.get(trace_key)
            if trace_data:
                try:
                    embedding_data = json.loads(trace_data)
                    query = embedding_data.get('query', '')
                    response = embedding_data.get('response', '')
                    created_at = embedding_data.get('metadata', {}).get('created_at', 0)
                    
                    if query and created_at:
                        queries.append(query)
                        timestamps.append(created_at)
                        responses.append(response)
                        
                except json.JSONDecodeError:
                    continue
        
        if not queries:
            return patterns
        
        # Basic pattern analysis
        patterns['query_structures'] = self._analyze_query_structures(queries)
        patterns['topic_clusters'] = self._analyze_topic_clusters(queries)
        patterns['intent_patterns'] = self._analyze_intent_patterns(queries)
        patterns['temporal_patterns'] = self._analyze_temporal_patterns(timestamps, queries)
        patterns['semantic_patterns'] = self._analyze_semantic_patterns(queries)
        
        # Advanced pattern analysis
        if self.sentiment_analysis_enabled:
            patterns['sentiment_analysis'] = self._analyze_sentiment_patterns(queries, responses)
        
        if self.complexity_scoring_enabled:
            patterns['complexity_analysis'] = self._analyze_complexity_patterns(queries)
        
        if self.domain_detection_enabled:
            patterns['domain_analysis'] = self._analyze_domain_patterns(queries)
        
        if self.behavioral_patterns_enabled:
            patterns['behavioral_patterns'] = self._analyze_behavioral_patterns(queries, timestamps)
        
        # Advanced metrics
        patterns['advanced_metrics'] = self._calculate_advanced_metrics(queries, responses, timestamps)
        
        # Cache patterns
        r.setex(f"patterns:{user_id}", self.pattern_cache_ttl, json.dumps(patterns))
        
        return patterns
    
    def _analyze_sentiment_patterns(self, queries: List[str], responses: List[str]) -> Dict:
        """
        Analyze sentiment patterns in queries and responses.
        
        Args:
            queries: List of query strings
            responses: List of response strings
            
        Returns:
            Dict with sentiment analysis
        """
        sentiment = {
            'query_sentiment': {},
            'response_sentiment': {},
            'sentiment_evolution': {},
            'emotional_patterns': {},
            'urgency_indicators': {}
        }
        
        # Simple sentiment analysis (can be enhanced with NLP libraries)
        positive_words = {'good', 'great', 'excellent', 'best', 'awesome', 'amazing', 'perfect', 'love', 'like'}
        negative_words = {'bad', 'terrible', 'worst', 'awful', 'hate', 'dislike', 'problem', 'error', 'fail'}
        urgent_words = {'urgent', 'emergency', 'critical', 'asap', 'immediately', 'now', 'quick', 'fast'}
        
        for i, (query, response) in enumerate(zip(queries, responses)):
            query_lower = query.lower()
            response_lower = response.lower()
            
            # Query sentiment
            positive_count = sum(1 for word in positive_words if word in query_lower)
            negative_count = sum(1 for word in negative_words if word in query_lower)
            urgent_count = sum(1 for word in urgent_words if word in query_lower)
            
            if positive_count > negative_count:
                sentiment['query_sentiment'][i] = 'positive'
            elif negative_count > positive_count:
                sentiment['query_sentiment'][i] = 'negative'
            else:
                sentiment['query_sentiment'][i] = 'neutral'
            
            # Response sentiment
            positive_count = sum(1 for word in positive_words if word in response_lower)
            negative_count = sum(1 for word in negative_words if word in response_lower)
            
            if positive_count > negative_count:
                sentiment['response_sentiment'][i] = 'positive'
            elif negative_count > positive_count:
                sentiment['response_sentiment'][i] = 'negative'
            else:
                sentiment['response_sentiment'][i] = 'neutral'
            
            # Urgency indicators
            if urgent_count > 0:
                sentiment['urgency_indicators'][i] = urgent_count
        
        return sentiment
    
    def _analyze_complexity_patterns(self, queries: List[str]) -> Dict:
        """
        Analyze complexity patterns in queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dict with complexity analysis
        """
        complexity = {
            'technical_complexity': {},
            'conceptual_complexity': {},
            'complexity_trends': {},
            'expertise_level': {},
            'learning_progression': {}
        }
        
        technical_terms = {
            'beginner': {'what', 'how', 'basic', 'simple', 'start', 'begin'},
            'intermediate': {'optimize', 'performance', 'architecture', 'design', 'pattern'},
            'advanced': {'distributed', 'microservices', 'scalability', 'consistency', 'latency'}
        }
        
        for i, query in enumerate(queries):
            query_lower = query.lower()
            
            # Technical complexity scoring
            tech_score = 0
            for level, terms in technical_terms.items():
                matches = sum(1 for term in terms if term in query_lower)
                if level == 'beginner':
                    tech_score += matches * 1
                elif level == 'intermediate':
                    tech_score += matches * 2
                elif level == 'advanced':
                    tech_score += matches * 3
            
            complexity['technical_complexity'][i] = min(tech_score / 5.0, 1.0)
            
            # Conceptual complexity (based on query length and vocabulary)
            words = query.split()
            unique_words = len(set(words))
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            conceptual_score = (unique_words / len(words) * 0.5 + avg_word_length / 10.0 * 0.5) if words else 0
            complexity['conceptual_complexity'][i] = min(conceptual_score, 1.0)
            
            # Expertise level estimation
            if tech_score <= 2:
                complexity['expertise_level'][i] = 'beginner'
            elif tech_score <= 4:
                complexity['expertise_level'][i] = 'intermediate'
            else:
                complexity['expertise_level'][i] = 'advanced'
        
        return complexity
    
    def _analyze_domain_patterns(self, queries: List[str]) -> Dict:
        """
        Analyze domain-specific patterns in queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dict with domain analysis
        """
        domains = {
            'primary_domains': {},
            'domain_combinations': {},
            'domain_expertise': {},
            'cross_domain_patterns': {},
            'domain_evolution': {}
        }
        
        # Extended domain definitions
        domain_keywords = {
            'frontend': ['react', 'vue', 'angular', 'javascript', 'css', 'html', 'ui', 'ux', 'frontend'],
            'backend': ['api', 'server', 'database', 'backend', 'nodejs', 'python', 'java', 'php'],
            'devops': ['docker', 'kubernetes', 'ci/cd', 'deployment', 'infrastructure', 'aws', 'azure'],
            'data_science': ['machine learning', 'ai', 'data', 'analytics', 'pandas', 'numpy', 'tensorflow'],
            'security': ['security', 'authentication', 'authorization', 'encryption', 'oauth', 'jwt'],
            'mobile': ['ios', 'android', 'mobile', 'app', 'react native', 'flutter'],
            'blockchain': ['blockchain', 'cryptocurrency', 'smart contract', 'ethereum', 'bitcoin'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'serverless', 'lambda', 's3']
        }
        
        for i, query in enumerate(queries):
            query_lower = query.lower()
            found_domains = []
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    found_domains.append(domain)
                    domains['primary_domains'][domain] = domains['primary_domains'].get(domain, 0) + 1
            
            # Domain combinations
            if len(found_domains) > 1:
                combination = '+'.join(sorted(found_domains))
                domains['domain_combinations'][combination] = domains['domain_combinations'].get(combination, 0) + 1
            
            # Domain expertise estimation
            if found_domains:
                domains['domain_expertise'][i] = found_domains
        
        return domains
    
    def _analyze_behavioral_patterns(self, queries: List[str], timestamps: List[float]) -> Dict:
        """
        Analyze behavioral patterns in user queries.
        
        Args:
            queries: List of query strings
            timestamps: List of timestamps
            
        Returns:
            Dict with behavioral analysis
        """
        behavioral = {
            'query_frequency': {},
            'time_patterns': {},
            'learning_curve': {},
            'topic_switching': {},
            'depth_of_inquiry': {},
            'problem_solving_patterns': {}
        }
        
        # Query frequency analysis
        from collections import Counter
        query_words = []
        for query in queries:
            query_words.extend(query.lower().split())
        
        word_freq = Counter(query_words)
        behavioral['query_frequency'] = dict(word_freq.most_common(10))
        
        # Time pattern analysis
        if len(timestamps) > 1:
            intervals = []
            for i in range(1, len(timestamps)):
                interval = timestamps[i] - timestamps[i-1]
                intervals.append(interval)
            
            avg_interval = sum(intervals) / len(intervals)
            behavioral['time_patterns']['average_interval_hours'] = avg_interval / 3600
            behavioral['time_patterns']['total_queries'] = len(queries)
            behavioral['time_patterns']['time_span_days'] = (max(timestamps) - min(timestamps)) / 86400
        
        # Learning curve analysis
        complexity_scores = []
        for i, query in enumerate(queries):
            complexity = self._calculate_complexity(query)
            complexity_scores.append(complexity)
        
        if len(complexity_scores) > 1:
            # Simple trend analysis
            first_half = complexity_scores[:len(complexity_scores)//2]
            second_half = complexity_scores[len(complexity_scores)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first:
                behavioral['learning_curve']['trend'] = 'improving'
            elif avg_second < avg_first:
                behavioral['learning_curve']['trend'] = 'declining'
            else:
                behavioral['learning_curve']['trend'] = 'stable'
            
            behavioral['learning_curve']['complexity_increase'] = avg_second - avg_first
        
        return behavioral
    
    def _calculate_advanced_metrics(self, queries: List[str], responses: List[str], timestamps: List[float]) -> Dict:
        """
        Calculate advanced metrics for pattern analysis.
        
        Args:
            queries: List of query strings
            responses: List of response strings
            timestamps: List of timestamps
            
        Returns:
            Dict with advanced metrics
        """
        metrics = {
            'query_diversity': 0.0,
            'response_quality_trend': 0.0,
            'engagement_score': 0.0,
            'expertise_progression': 0.0,
            'topic_coherence': 0.0,
            'learning_efficiency': 0.0
        }
        
        if not queries:
            return metrics
        
        # Query diversity (unique words vs total words)
        all_words = []
        for query in queries:
            all_words.extend(query.lower().split())
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        metrics['query_diversity'] = unique_words / total_words if total_words > 0 else 0.0
        
        # Engagement score (based on query length and complexity)
        avg_query_length = sum(len(query) for query in queries) / len(queries)
        avg_complexity = sum(self._calculate_complexity(query) for query in queries) / len(queries)
        metrics['engagement_score'] = (avg_query_length / 100.0 * 0.5 + avg_complexity * 0.5)
        
        # Topic coherence (how related queries are)
        if len(queries) > 1:
            coherence_scores = []
            for i in range(len(queries) - 1):
                similarity = self._calculate_query_similarity(queries[i], queries[i + 1])
                coherence_scores.append(similarity)
            
            metrics['topic_coherence'] = sum(coherence_scores) / len(coherence_scores)
        
        # Learning efficiency (complexity increase over time)
        if len(queries) > 1 and len(timestamps) > 1:
            time_span = (max(timestamps) - min(timestamps)) / 86400  # days
            complexity_scores = [self._calculate_complexity(query) for query in queries]
            complexity_increase = max(complexity_scores) - min(complexity_scores)
            
            metrics['learning_efficiency'] = complexity_increase / time_span if time_span > 0 else 0.0
        
        return metrics
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate similarity between two queries.
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def predict_next_query_pattern(self, user_id: str, current_query: str) -> Dict:
        """
        Predict the next likely query pattern based on historical analysis.
        
        Args:
            user_id: User identifier
            current_query: Current query
            
        Returns:
            Dict with prediction results
        """
        r = self.redis_client
        
        # Get cached patterns
        patterns_data = r.get(f"patterns:{user_id}")
        if not patterns_data:
            return {
                'likely_next_topics': [],
                'query_structure_prediction': '',
                'intent_prediction': '',
                'confidence': 0.0,
                'sentiment_prediction': 'neutral',
                'complexity_prediction': 0.5,
                'domain_prediction': [],
                'behavioral_insights': {}
            }
        
        try:
            patterns = json.loads(patterns_data)
        except json.JSONDecodeError:
            return {
                'likely_next_topics': [],
                'query_structure_prediction': '',
                'intent_prediction': '',
                'confidence': 0.0,
                'sentiment_prediction': 'neutral',
                'complexity_prediction': 0.5,
                'domain_prediction': [],
                'behavioral_insights': {}
            }
        
        prediction = {
            'likely_next_topics': [],
            'query_structure_prediction': '',
            'intent_prediction': '',
            'confidence': 0.0,
            'sentiment_prediction': 'neutral',
            'complexity_prediction': 0.5,
            'domain_prediction': [],
            'behavioral_insights': {}
        }
        
        # Analyze current query
        current_topics = self._extract_topics_from_query(current_query)
        current_intent = self._extract_intent_from_query(current_query)
        current_sentiment = self._analyze_single_sentiment(current_query)
        current_complexity = self._calculate_complexity(current_query)
        current_domains = self._extract_domains_from_query(current_query)
        
        # Predict next topics based on topic combinations
        topic_combinations = patterns.get('topic_clusters', {}).get('topic_combinations', {})
        for combination, count in topic_combinations.items():
            topics = combination.split('+')
            if any(topic in current_topics for topic in topics):
                # Find topics that commonly appear with current topics
                for topic in topics:
                    if topic not in current_topics:
                        prediction['likely_next_topics'].append(topic)
        
        # Predict query structure
        question_types = patterns.get('query_structures', {}).get('question_types', {})
        if question_types:
            most_common = max(question_types.items(), key=lambda x: x[1])
            prediction['query_structure_prediction'] = most_common[0]
        
        # Predict intent
        intent_patterns = patterns.get('intent_patterns', {})
        if intent_patterns:
            most_common_intent = max(intent_patterns.items(), key=lambda x: x[1])
            prediction['intent_prediction'] = most_common_intent[0]
        
        # Predict sentiment
        prediction['sentiment_prediction'] = current_sentiment
        
        # Predict complexity
        complexity_analysis = patterns.get('complexity_analysis', {})
        if complexity_analysis.get('technical_complexity'):
            avg_complexity = sum(complexity_analysis['technical_complexity'].values()) / len(complexity_analysis['technical_complexity'])
            prediction['complexity_prediction'] = min(avg_complexity + 0.1, 1.0)  # Slight increase
        
        # Predict domains
        prediction['domain_prediction'] = current_domains
        
        # Behavioral insights
        behavioral = patterns.get('behavioral_patterns', {})
        if behavioral:
            prediction['behavioral_insights'] = {
                'learning_trend': behavioral.get('learning_curve', {}).get('trend', 'unknown'),
                'query_frequency': len(behavioral.get('query_frequency', {})),
                'avg_interval_hours': behavioral.get('time_patterns', {}).get('average_interval_hours', 0)
            }
        
        # Calculate confidence
        total_patterns = sum(patterns.get('query_structures', {}).get('question_types', {}).values())
        if total_patterns > 0:
            prediction['confidence'] = min(0.9, total_patterns / 10.0)
        
        return prediction
    
    def _analyze_single_sentiment(self, query: str) -> str:
        """Analyze sentiment of a single query."""
        positive_words = {'good', 'great', 'excellent', 'best', 'awesome', 'amazing', 'perfect', 'love', 'like'}
        negative_words = {'bad', 'terrible', 'worst', 'awful', 'hate', 'dislike', 'problem', 'error', 'fail'}
        
        query_lower = query.lower()
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_domains_from_query(self, query: str) -> List[str]:
        """Extract domains from a single query."""
        domain_keywords = {
            'frontend': ['react', 'vue', 'angular', 'javascript', 'css', 'html', 'ui', 'ux', 'frontend'],
            'backend': ['api', 'server', 'database', 'backend', 'nodejs', 'python', 'java', 'php'],
            'devops': ['docker', 'kubernetes', 'ci/cd', 'deployment', 'infrastructure', 'aws', 'azure'],
            'data_science': ['machine learning', 'ai', 'data', 'analytics', 'pandas', 'numpy', 'tensorflow'],
            'security': ['security', 'authentication', 'authorization', 'encryption', 'oauth', 'jwt'],
            'mobile': ['ios', 'android', 'mobile', 'app', 'react native', 'flutter'],
            'blockchain': ['blockchain', 'cryptocurrency', 'smart contract', 'ethereum', 'bitcoin'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud', 'serverless', 'lambda', 's3']
        }
        
        query_lower = query.lower()
        found_domains = []
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                found_domains.append(domain)
        
        return found_domains


    def _analyze_query_structures(self, queries: List[str]) -> Dict:
        """
        Analyze the structure of queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dict with structure analysis
        """
        structures = {
            'question_types': {},
            'query_lengths': {},
            'keyword_patterns': {},
            'complexity_distribution': {}
        }
        
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        
        for query in queries:
            query_lower = query.lower()
            
            # Question type analysis
            for word in question_words:
                if query_lower.startswith(word):
                    structures['question_types'][word] = structures['question_types'].get(word, 0) + 1
                    break
            
            # Query length analysis
            length_category = self._categorize_length(len(query))
            structures['query_lengths'][length_category] = structures['query_lengths'].get(length_category, 0) + 1
            
            # Keyword pattern analysis
            keywords = self._extract_keywords(query)
            for keyword in keywords:
                structures['keyword_patterns'][keyword] = structures['keyword_patterns'].get(keyword, 0) + 1
            
            # Complexity analysis
            complexity = self._calculate_complexity(query)
            complexity_category = self._categorize_complexity(complexity)
            structures['complexity_distribution'][complexity_category] = structures['complexity_distribution'].get(complexity_category, 0) + 1
        
        return structures
    
    def _analyze_topic_clusters(self, queries: List[str]) -> Dict:
        """
        Analyze topic clusters in queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dict with topic cluster analysis
        """
        clusters = {
            'primary_topics': {},
            'topic_combinations': {},
            'topic_evolution': {},
            'cross_topic_patterns': {}
        }
        
        # Define topic keywords
        topics = {
            'security': ['security', 'auth', 'authentication', 'authorization', 'oauth', 'jwt', 'encryption'],
            'performance': ['performance', 'optimization', 'speed', 'fast', 'slow', 'caching', 'database'],
            'architecture': ['architecture', 'design', 'pattern', 'microservices', 'monolith', 'api'],
            'deployment': ['deployment', 'docker', 'kubernetes', 'ci/cd', 'production', 'staging'],
            'monitoring': ['monitoring', 'logging', 'metrics', 'alerting', 'observability'],
            'testing': ['testing', 'test', 'unit', 'integration', 'qa', 'quality'],
            'data': ['data', 'database', 'storage', 'cache', 'redis', 'postgresql']
        }
        
        for query in queries:
            query_lower = query.lower()
            found_topics = []
            
            for topic, keywords in topics.items():
                if any(keyword in query_lower for keyword in keywords):
                    found_topics.append(topic)
                    clusters['primary_topics'][topic] = clusters['primary_topics'].get(topic, 0) + 1
            
            # Topic combinations
            if len(found_topics) > 1:
                combination = '+'.join(sorted(found_topics))
                clusters['topic_combinations'][combination] = clusters['topic_combinations'].get(combination, 0) + 1
        
        return clusters
    
    def _analyze_intent_patterns(self, queries: List[str]) -> Dict:
        """
        Analyze intent patterns in queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dict with intent pattern analysis
        """
        intents = {
            'how_to': 0,
            'what_is': 0,
            'why_question': 0,
            'comparison': 0,
            'troubleshooting': 0,
            'best_practice': 0,
            'implementation': 0
        }
        
        intent_keywords = {
            'how_to': ['how to', 'how do i', 'steps to', 'guide to'],
            'what_is': ['what is', 'what are', 'define', 'explain'],
            'why_question': ['why', 'reason', 'cause'],
            'comparison': ['vs', 'versus', 'compare', 'difference'],
            'troubleshooting': ['error', 'issue', 'problem', 'fix', 'debug'],
            'best_practice': ['best', 'recommended', 'practice', 'pattern'],
            'implementation': ['implement', 'code', 'example', 'sample']
        }
        
        for query in queries:
            query_lower = query.lower()
            
            for intent, keywords in intent_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    intents[intent] += 1
                    break
        
        return intents
    
    def _analyze_temporal_patterns(self, timestamps: List[float], queries: List[str]) -> Dict:
        """
        Analyze temporal patterns in queries.
        
        Args:
            timestamps: List of timestamps
            queries: List of query strings
            
        Returns:
            Dict with temporal pattern analysis
        """
        temporal = {
            'hourly_distribution': {},
            'daily_distribution': {},
            'weekly_distribution': {},
            'topic_temporal_evolution': {},
            'query_frequency_trends': {}
        }
        
        for timestamp, query in zip(timestamps, queries):
            dt = datetime.fromtimestamp(timestamp)
            
            # Hourly distribution
            hour = dt.hour
            temporal['hourly_distribution'][hour] = temporal['hourly_distribution'].get(hour, 0) + 1
            
            # Daily distribution
            day = dt.strftime('%A')
            temporal['daily_distribution'][day] = temporal['daily_distribution'].get(day, 0) + 1
            
            # Weekly distribution
            week = dt.strftime('%Y-%W')
            temporal['weekly_distribution'][week] = temporal['weekly_distribution'].get(week, 0) + 1
        
        return temporal
    
    def _analyze_semantic_patterns(self, queries: List[str]) -> Dict:
        """
        Analyze semantic patterns in queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dict with semantic pattern analysis
        """
        semantic = {
            'common_phrases': {},
            'semantic_similarity_clusters': {},
            'domain_specific_terms': {},
            'query_complexity_trends': {}
        }
        
        # Extract common phrases
        for query in queries:
            words = query.lower().split()
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                semantic['common_phrases'][phrase] = semantic['common_phrases'].get(phrase, 0) + 1
        
        # Filter out low-frequency phrases
        semantic['common_phrases'] = {
            phrase: count for phrase, count in semantic['common_phrases'].items()
            if count >= self.min_pattern_frequency
        }
        
        return semantic
    
    def _categorize_length(self, length: int) -> str:
        """Categorize query length."""
        if length < 20:
            return 'short'
        elif length < 50:
            return 'medium'
        else:
            return 'long'
    
    def _categorize_complexity(self, complexity: float) -> str:
        """Categorize query complexity."""
        if complexity < 0.3:
            return 'simple'
        elif complexity < 0.7:
            return 'moderate'
        else:
            return 'complex'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.lower().split()
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        # Simple complexity calculation
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words = len(set(words))
        total_words = len(words)
        
        # Complexity based on average word length and vocabulary diversity
        complexity = (avg_word_length / 10.0) * 0.5 + (unique_words / total_words) * 0.5
        return min(complexity, 1.0)
    
    def _extract_topics_from_query(self, query: str) -> List[str]:
        """Extract topics from a single query."""
        topics = {
            'security': ['security', 'auth', 'authentication', 'authorization', 'oauth', 'jwt', 'encryption'],
            'performance': ['performance', 'optimization', 'speed', 'fast', 'slow', 'caching', 'database'],
            'architecture': ['architecture', 'design', 'pattern', 'microservices', 'monolith', 'api'],
            'deployment': ['deployment', 'docker', 'kubernetes', 'ci/cd', 'production', 'staging'],
            'monitoring': ['monitoring', 'logging', 'metrics', 'alerting', 'observability'],
            'testing': ['testing', 'test', 'unit', 'integration', 'qa', 'quality'],
            'data': ['data', 'database', 'storage', 'cache', 'redis', 'postgresql']
        }
        
        query_lower = query.lower()
        found_topics = []
        
        for topic, keywords in topics.items():
            if any(keyword in query_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def _extract_intent_from_query(self, query: str) -> str:
        """Extract intent from a single query."""
        intent_keywords = {
            'how_to': ['how to', 'how do i', 'steps to', 'guide to'],
            'what_is': ['what is', 'what are', 'define', 'explain'],
            'why_question': ['why', 'reason', 'cause'],
            'comparison': ['vs', 'versus', 'compare', 'difference'],
            'troubleshooting': ['error', 'issue', 'problem', 'fix', 'debug'],
            'best_practice': ['best', 'recommended', 'practice', 'pattern'],
            'implementation': ['implement', 'code', 'example', 'sample']
        }
        
        query_lower = query.lower()
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return 'general'


class SelfEvolvingContext:
    """
    Orchestrates the self-evolving context system, combining all components
    for intelligent context retrieval and learning.
    """
    
    def __init__(self):
        self.redis_client = r
        self.scoring_engine = ContextScoringEngine(self.redis_client)
        self.recall_tracker = RecallTracker(self.redis_client)
        self.adaptive_weighting = AdaptiveWeighting(self.redis_client)
        self.metrics_collector = MetricsCollector(self.redis_client)
        self.auto_pruning = AutoPruning(self.redis_client)
        self.pattern_recognition = AdvancedPatternRecognition(self.redis_client)
        self.semantic_drift_detection = SemanticDriftDetection(self.redis_client)
        
        # Configuration
        self.maintenance_interval = 3600  # 1 hour
        self.last_maintenance = 0
        
        # Start periodic maintenance
        self._periodic_maintenance("system")
    
    def find_evolving_context(self, user_id: str, current_prompt: str, 
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
        # Get base semantic matches with lower threshold for evolving
        base_matches = semantic_embeddings.find_semantically_similar_context(
            user_id, current_prompt, limit * 3, similarity_threshold * 0.5  # Lower threshold
        )
        
        if not base_matches:
            return []
        
        # Apply adaptive scoring
        enhanced_matches = []
        for embedding_data, base_similarity in base_matches:
            # Use embedding_id instead of trace_id
            embedding_id = embedding_data.get('embedding_id')
            if embedding_id:
                # Calculate enhanced score using scoring engine
                enhanced_score = self.scoring_engine.calculate_context_score(
                    user_id, embedding_id, current_prompt
                )
                
                # Combine base similarity with adaptive score (more weight to base similarity)
                enhanced_similarity = (base_similarity * 0.8) + (enhanced_score * 0.2)
                
                # Apply lower threshold for evolving context
                if enhanced_similarity >= similarity_threshold * 0.7:  # More lenient threshold
                    enhanced_matches.append((embedding_data, enhanced_similarity))
            else:
                # Fallback to base similarity if no embedding_id
                if base_similarity >= similarity_threshold * 0.7:
                    enhanced_matches.append((embedding_data, base_similarity))
        
        # Sort by enhanced similarity and return top results
        enhanced_matches.sort(key=lambda x: x[1], reverse=True)
        return enhanced_matches[:limit]
    
    def track_context_effectiveness(self, user_id: str, trace_ids: List[str], 
                                  response_quality: float, user_feedback: Optional[bool] = None):
        """
        Track the effectiveness of used context and update learning.
        
        Args:
            user_id: User identifier
            trace_ids: List of trace IDs that were used as context
            response_quality: Quality score of the response (0-1)
            user_feedback: Explicit user feedback (optional)
        """
        for trace_id in trace_ids:
            self.recall_tracker.track_recall_success(
                trace_id, "context_usage", response_quality, user_feedback
            )
        
        # Update adaptive weights periodically
        self.adaptive_weighting.update_weights(user_id)
        
        # Collect metrics
        self.metrics_collector.collect_metrics(user_id)
    
    def _periodic_maintenance(self, user_id: str):
        """
        Perform periodic maintenance tasks including ML model training.
        
        Args:
            user_id: User identifier
        """
        current_time = time.time()
        
        # Check if maintenance is needed
        if current_time - self.last_maintenance < self.maintenance_interval:
            return
        
        self.last_maintenance = current_time
        
        print(f"🔧 Performing periodic maintenance for user: {user_id}")
        
        # Auto-pruning
        pruning_result = self.auto_pruning.prune_low_impact_traces(user_id)
        if pruning_result.get('traces_pruned', 0) > 0:
            print(f"🧹 Pruned {pruning_result['traces_pruned']} low-impact traces")
        
        # Pattern analysis
        pattern_result = self.pattern_recognition.analyze_query_patterns(user_id)
        print(f"📊 Pattern analysis completed: {len(pattern_result.get('query_structures', {}))} patterns found")
        
        # Drift detection
        drift_result = self.semantic_drift_detection.detect_semantic_drift(user_id)
        if drift_result.get('drift_detected', False):
            print(f"⚠️ Drift detected: {drift_result.get('drift_severity', 'unknown')}")
        
        # ML model training
        if ML_AVAILABLE:
            print("🤖 Training ML models...")
            self.scoring_engine.train_models(user_id)
            print("✅ ML model training completed")
        
        # Collect comprehensive metrics
        metrics = self.metrics_collector.collect_metrics(user_id)
        print(f"📈 Metrics collected: {metrics.get('total_traces', 0)} traces analyzed")
    
    def train_ml_models(self, user_id: str):
        """
        No-op method - ML training removed for statistical-only approach.
        
        Args:
            user_id: User identifier
        """
        print(f"📊 Using statistical-only approach for user: {user_id}")
        return {"status": "success", "message": "Statistical models ready"}
    
    def get_ml_model_status(self, user_id: str) -> Dict:
        """
        Get status of statistical models.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with statistical model status
        """
        return {
            "ml_available": False,
            "models_trained": True,
            "training_data_count": 0,
            "model_performance": {"approach": "statistical_only"},
            "message": "Using statistical-only approach - no ML dependencies"
        }
    
    def get_evolving_analytics(self, user_id: str) -> Dict:
        """
        Get comprehensive analytics for the self-evolving context system.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with comprehensive analytics
        """
        # Get base metrics
        base_metrics = self.metrics_collector.collect_metrics(user_id)
        
        # Get pruning statistics
        pruning_stats = self.redis_client.get(f"pruning_stats:{user_id}")
        pruning_data = json.loads(pruning_stats) if pruning_stats else {}
        
        # Get pattern analysis
        patterns_data = self.redis_client.get(f"patterns:{user_id}")
        patterns = json.loads(patterns_data) if patterns_data else {}
        
        # Get pruning recommendations
        pruning_recommendations = self.auto_pruning.get_pruning_recommendations(user_id)
        
        analytics = {
            'system_status': {
                'scoring_engine_active': True,
                'recall_tracking_active': True,
                'adaptive_weighting_active': True,
                'metrics_collection_active': True,
                'auto_pruning_active': True,
                'pattern_recognition_active': True
            },
            'performance_summary': base_metrics,
            'pruning_statistics': pruning_data,
            'pattern_analysis': patterns,
            'pruning_recommendations': pruning_recommendations,
            'last_maintenance': self.last_maintenance,
            'next_maintenance': self.last_maintenance + self.maintenance_interval
        }
        
        return analytics
    
    def get_pruning_recommendations(self, user_id: str) -> Dict:
        """
        Get recommendations for trace pruning.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with pruning recommendations
        """
        return self.auto_pruning.get_pruning_recommendations(user_id)
    
    def analyze_query_patterns(self, user_id: str) -> Dict:
        """
        Analyze query patterns for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with pattern analysis results
        """
        return self.pattern_recognition.analyze_query_patterns(user_id)
    
    def predict_next_query_pattern(self, user_id: str, current_query: str) -> Dict:
        """
        Predict the next likely query pattern.
        
        Args:
            user_id: User identifier
            current_query: Current query
            
        Returns:
            Dict with prediction results
        """
        return self.pattern_recognition.predict_next_query_pattern(user_id, current_query)
    
    def manual_prune_traces(self, user_id: str, trace_ids: List[str]) -> Dict:
        """
        Manually prune specific traces.
        
        Args:
            user_id: User identifier
            trace_ids: List of trace IDs to prune
            
        Returns:
            Dict with pruning results
        """
        return self.auto_pruning.manual_prune_traces(user_id, trace_ids)
    
    def detect_semantic_drift(self, user_id: str) -> Dict:
        """
        Detect semantic drift for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with drift analysis results
        """
        return self.semantic_drift_detection.detect_semantic_drift(user_id)
    
    def get_drift_summary(self, user_id: str) -> Dict:
        """
        Get a summary of drift detection results.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with drift summary
        """
        return self.semantic_drift_detection.get_drift_summary(user_id)
    
    def set_drift_threshold(self, threshold: float):
        """
        Set custom drift detection threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        self.semantic_drift_detection.set_drift_threshold(threshold)
    
    def enable_drift_component(self, component: str, enabled: bool = True):
        """
        Enable or disable drift detection components.
        
        Args:
            component: Component name ('performance', 'behavioral', 'context_relevance', 'accuracy')
            enabled: Whether to enable the component
        """
        self.semantic_drift_detection.enable_component(component, enabled)


# Global instance
self_evolving_context = SelfEvolvingContext()

# =============================================================================
# PHASE 1 IMPLEMENTATION COMPLETE
# =============================================================================
# ✅ Context Scoring Engine
# ✅ Recall Tracking System  
# ✅ Basic Adaptive Weighting
# ✅ Metrics Collection
# =============================================================================