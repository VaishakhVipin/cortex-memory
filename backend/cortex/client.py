#!/usr/bin/env python3
"""
🧠 Cortex Client - Main SDK Client
Handles API key authentication, usage tracking, and rate limiting.
"""

import os
import json
import time
import hashlib
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import requests

from .redis_client import r
from .core import store_conversation, get_conversation
from .semantic_embeddings import semantic_embeddings
from .self_evolving_context import self_evolving_context
from .semantic_drift_detection import detect_semantic_drift
from .context_manager import generate_with_context, generate_with_evolving_context

class CortexClient:
    """
    Main Cortex client for API key-based usage with pay-per-use functionality.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.cortex-memory.com"):
        """
        Initialize Cortex client with API key.
        
        Args:
            api_key: User's API key for authentication
            base_url: API base URL (default: production)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Cortex-Python-SDK/2.0.0'
        })
        
        # Validate API key on initialization
        self._validate_api_key()
    
    def _validate_api_key(self):
        """Validate API key and get user info."""
        try:
            response = self.session.get(f"{self.base_url}/auth/validate")
            response.raise_for_status()
            
            user_data = response.json()
            self.user_id = user_data.get('user_id')
            self.plan = user_data.get('plan', 'free')
            self.usage_limits = user_data.get('usage_limits', {})
            
            print(f"✅ Authenticated as user: {self.user_id} (Plan: {self.plan})")
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Invalid API key: {e}")
    
    def _check_usage_limits(self, operation: str) -> bool:
        """
        Check if user has remaining usage for the operation.
        
        Args:
            operation: Operation type (e.g., 'context_search', 'generation')
            
        Returns:
            True if usage is allowed, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/usage/check", params={
                'operation': operation
            })
            response.raise_for_status()
            
            usage_data = response.json()
            return usage_data.get('allowed', False)
            
        except requests.exceptions.RequestException:
            # Fallback to local check if API is unavailable
            return self._local_usage_check(operation)
    
    def _local_usage_check(self, operation: str) -> bool:
        """Local usage check as fallback."""
        # This would be replaced with actual usage tracking
        return True
    
    def _track_usage(self, operation: str, metadata: Dict = None):
        """Track API usage for billing."""
        try:
            self.session.post(f"{self.base_url}/usage/track", json={
                'operation': operation,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })
        except requests.exceptions.RequestException:
            # Silently fail if tracking fails
            pass
    
    def store_conversation(self, prompt: str, response: str, 
                          metadata: Optional[Dict] = None) -> str:
        """
        Store a conversation with automatic usage tracking.
        
        Args:
            prompt: User's prompt
            response: AI's response
            metadata: Additional metadata
            
        Returns:
            Memory ID of stored conversation
        """
        if not self._check_usage_limits('store_conversation'):
            raise Exception("Usage limit exceeded for conversation storage")
        
        try:
            memory_id = store_conversation(
                user_id=self.user_id,
                prompt=prompt,
                response=response,
                metadata=metadata
            )
            
            self._track_usage('store_conversation', {
                'memory_id': memory_id,
                'prompt_length': len(prompt),
                'response_length': len(response)
            })
            
            return memory_id
            
        except Exception as e:
            raise Exception(f"Failed to store conversation: {e}")
    
    def get_conversation(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve a conversation by memory ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Conversation data or None if not found
        """
        if not self._check_usage_limits('retrieve_conversation'):
            raise Exception("Usage limit exceeded for conversation retrieval")
        
        try:
            conversation = get_conversation(memory_id)
            
            self._track_usage('retrieve_conversation', {
                'memory_id': memory_id
            })
            
            return conversation
            
        except Exception as e:
            raise Exception(f"Failed to retrieve conversation: {e}")
    
    def find_semantic_context(self, prompt: str, limit: int = 5, 
                             similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Find semantically similar context with usage tracking.
        
        Args:
            prompt: Search prompt
            limit: Maximum number of results
            similarity_threshold: Similarity threshold
            
        Returns:
            List of similar contexts with scores
        """
        if not self._check_usage_limits('semantic_search'):
            raise Exception("Usage limit exceeded for semantic search")
        
        try:
            similar_contexts = semantic_embeddings.find_semantically_similar_context(
                user_id=self.user_id,
                current_prompt=prompt,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            # Format results
            results = []
            for context, score in similar_contexts:
                results.append({
                    'memory_id': context.get('embedding_id'),
                    'prompt': context.get('prompt'),
                    'response': context.get('response'),
                    'similarity_score': score,
                    'metadata': context.get('metadata', {})
                })
            
            self._track_usage('semantic_search', {
                'query_length': len(prompt),
                'results_count': len(results),
                'limit': limit,
                'threshold': similarity_threshold
            })
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to find semantic context: {e}")
    
    def find_evolving_context(self, prompt: str, limit: int = 5,
                             similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Find context using self-evolving algorithms with usage tracking.
        
        Args:
            prompt: Search prompt
            limit: Maximum number of results
            similarity_threshold: Similarity threshold
            
        Returns:
            List of evolving contexts with scores
        """
        if not self._check_usage_limits('evolving_search'):
            raise Exception("Usage limit exceeded for evolving search")
        
        try:
            evolving_contexts = self_evolving_context.find_evolving_context(
                user_id=self.user_id,
                current_prompt=prompt,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            # Format results
            results = []
            for context, score in evolving_contexts:
                results.append({
                    'memory_id': context.get('embedding_id'),
                    'prompt': context.get('prompt'),
                    'response': context.get('response'),
                    'similarity_score': score,
                    'metadata': context.get('metadata', {})
                })
            
            self._track_usage('evolving_search', {
                'query_length': len(prompt),
                'results_count': len(results),
                'limit': limit,
                'threshold': similarity_threshold
            })
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to find evolving context: {e}")
    
    def generate_with_context(self, prompt: str, context_method: str = "semantic") -> str:
        """
        Generate response with automatic context injection.
        
        Args:
            prompt: User's prompt
            context_method: "semantic" or "evolving"
            
        Returns:
            Generated response with injected context
        """
        if not self._check_usage_limits('generation'):
            raise Exception("Usage limit exceeded for response generation")
        
        try:
            if context_method == "semantic":
                response = generate_with_context(
                    user_id=self.user_id,
                    prompt=prompt
                )
            elif context_method == "evolving":
                response = generate_with_evolving_context(
                    user_id=self.user_id,
                    prompt=prompt
                )
            else:
                raise ValueError("Invalid context_method. Use 'semantic' or 'evolving'")
            
            self._track_usage('generation', {
                'prompt_length': len(prompt),
                'response_length': len(response),
                'context_method': context_method
            })
            
            return response
            
        except Exception as e:
            raise Exception(f"Failed to generate response: {e}")
    
    def get_analytics(self) -> Dict:
        """
        Get user analytics and usage statistics.
        
        Returns:
            Analytics data
        """
        if not self._check_usage_limits('analytics'):
            raise Exception("Usage limit exceeded for analytics")
        
        try:
            metrics = self_evolving_context.get_performance_metrics(self.user_id)
            
            self._track_usage('analytics', {
                'metrics_requested': list(metrics.keys())
            })
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Failed to get analytics: {e}")
    
    def detect_drift(self, time_window_hours: int = 24) -> Dict:
        """
        Detect semantic drift with usage tracking.
        
        Args:
            time_window_hours: Time window for drift detection
            
        Returns:
            Drift analysis results
        """
        if not self._check_usage_limits('drift_detection'):
            raise Exception("Usage limit exceeded for drift detection")
        
        try:
            drift_results = detect_semantic_drift(
                user_id=self.user_id,
                time_window_hours=time_window_hours
            )
            
            self._track_usage('drift_detection', {
                'time_window_hours': time_window_hours,
                'drift_detected': drift_results.get('drift_detected', False)
            })
            
            return drift_results
            
        except Exception as e:
            raise Exception(f"Failed to detect drift: {e}")
    
    def prune_memories(self, threshold: float = 0.3) -> Dict:
        """
        Prune low-impact memories with usage tracking.
        
        Args:
            threshold: Pruning threshold
            
        Returns:
            Pruning statistics
        """
        if not self._check_usage_limits('pruning'):
            raise Exception("Usage limit exceeded for memory pruning")
        
        try:
            pruning_stats = self_evolving_context.auto_pruning.prune_low_impact_memories(
                user_id=self.user_id,
                threshold=threshold
            )
            
            self._track_usage('pruning', {
                'threshold': threshold,
                'pruned_count': pruning_stats.get('pruned_memories', 0)
            })
            
            return pruning_stats
            
        except Exception as e:
            raise Exception(f"Failed to prune memories: {e}")
    
    def get_usage_stats(self) -> Dict:
        """
        Get current usage statistics for the user.
        
        Returns:
            Usage statistics
        """
        try:
            response = self.session.get(f"{self.base_url}/usage/stats")
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get usage stats: {e}")
    
    def get_plan_info(self) -> Dict:
        """
        Get current plan information and limits.
        
        Returns:
            Plan information
        """
        try:
            response = self.session.get(f"{self.base_url}/auth/plan")
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get plan info: {e}")