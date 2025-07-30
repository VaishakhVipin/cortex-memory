#!/usr/bin/env python3
"""
🧠 Cortex Client - Main SDK Client
Handles API key authentication, usage tracking, and rate limiting.
"""

import os
import json
import time
import hashlib
import asyncio
import logging
from typing import Dict, Optional, Any, List, Union
from datetime import datetime, timedelta
from functools import wraps
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .redis_client import r
from .core import store_conversation, get_conversation
from .semantic_embeddings import semantic_embeddings
from .self_evolving_context import self_evolving_context
from .semantic_drift_detection import detect_semantic_drift
from .context_manager import (
    generate_with_context, 
    generate_with_evolving_context,
    generate_with_hybrid_context,
    generate_with_adaptive_context
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CortexError(Exception):
    """Base exception for Cortex client errors."""
    pass

class AuthenticationError(CortexError):
    """Raised when API key authentication fails."""
    pass

class UsageLimitError(CortexError):
    """Raised when usage limits are exceeded."""
    pass

class RateLimitError(CortexError):
    """Raised when rate limits are exceeded."""
    pass

class CircuitBreakerError(CortexError):
    """Raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

def retry_on_failure(max_retries: int = 3, backoff_factor: float = 0.3):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, CortexError) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Exponential backoff
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
            
            raise last_exception
        return wrapper
    return decorator

class CortexClient:
    """
    Main Cortex client for API key-based usage with pay-per-use functionality.
    Enhanced with error handling, logging, retry logic, and circuit breakers.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.cortex-memory.com", 
                 timeout: int = 30, max_retries: int = 3):
        """
        Initialize Cortex client with API key.
        
        Args:
            api_key: User's API key for authentication
            base_url: API base URL (default: production)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize session with retry strategy
        self.session = self._create_session()
        
        # Circuit breakers for different operations
        self.circuit_breakers = {
            'auth': CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            'api': CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'usage': CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        }
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0
        }
        
        # Validate API key on initialization
        self._validate_api_key()
        
        logger.info(f"✅ Cortex client initialized for user: {getattr(self, 'user_id', 'unknown')}")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=0.3
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Cortex-Memory-Python-SDK/2.0.0'
        })
        
        return session
    
    @retry_on_failure(max_retries=3)
    def _validate_api_key(self):
        """Validate API key and get user info with retry logic."""
        try:
            start_time = time.time()
            
            response = self.session.get(
                f"{self.base_url}/auth/validate",
                timeout=self.timeout
            )
            
            # Track response time
            response_time = time.time() - start_time
            self._update_metrics(response_time, response.status_code == 200)
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code >= 500:
                raise CortexError(f"Server error: {response.status_code}")
            
            response.raise_for_status()
            
            user_data = response.json()
            self.user_id = user_data.get('user_id')
            self.plan = user_data.get('plan', 'free')
            self.usage_limits = user_data.get('usage_limits', {})
            
            logger.info(f"✅ Authenticated as user: {self.user_id} (Plan: {self.plan})")
            
        except requests.exceptions.Timeout:
            raise CortexError("Request timeout during API key validation")
        except requests.exceptions.ConnectionError:
            raise CortexError("Connection error during API key validation")
        except Exception as e:
            if isinstance(e, CortexError):
                raise
            raise CortexError(f"Unexpected error during API key validation: {e}")
    
    def _update_metrics(self, response_time: float, success: bool):
        """Update performance metrics."""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update average response time
        current_avg = self.metrics['average_response_time']
        total_requests = self.metrics['total_requests']
        self.metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    @retry_on_failure(max_retries=2)
    def _check_usage_limits(self, operation: str) -> bool:
        """
        Check if user has remaining usage for the operation.
        
        Args:
            operation: Operation type (e.g., 'context_search', 'generation')
            
        Returns:
            True if usage is allowed, False otherwise
        """
        try:
            start_time = time.time()
            
            response = self.session.get(
                f"{self.base_url}/usage/check",
                params={'operation': operation},
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, response.status_code == 200)
            
            if response.status_code == 429:
                raise UsageLimitError("Usage limit exceeded")
            
            response.raise_for_status()
            usage_data = response.json()
            return usage_data.get('allowed', False)
            
        except requests.exceptions.RequestException:
            # Fallback to local check if API is unavailable
            logger.warning("API unavailable, using local usage check")
            return self._local_usage_check(operation)
    
    def _local_usage_check(self, operation: str) -> bool:
        """Local usage check as fallback."""
        # This would be replaced with actual usage tracking
        return True
    
    @retry_on_failure(max_retries=2)
    def _track_usage(self, operation: str, metadata: Dict = None):
        """Track API usage for billing with retry logic."""
        try:
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/usage/track",
                json={
                    'operation': operation,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': metadata or {}
                },
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, response.status_code == 200)
            
            if response.status_code >= 400:
                logger.warning(f"Usage tracking failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Usage tracking failed: {e}")
    
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
            
        Raises:
            UsageLimitError: If usage limits are exceeded
            CortexError: If storage fails
        """
        try:
            if not self._check_usage_limits('store_conversation'):
                raise UsageLimitError("Usage limit exceeded for conversation storage")
            
            start_time = time.time()
            
            memory_id = store_conversation(
                user_id=self.user_id,
                prompt=prompt,
                response=response,
                metadata=metadata
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
            self._track_usage('store_conversation', {
                'memory_id': memory_id,
                'prompt_length': len(prompt),
                'response_length': len(response)
            })
            
            logger.info(f"✅ Conversation stored with memory ID: {memory_id}")
            return memory_id
            
        except UsageLimitError:
            raise
        except Exception as e:
            self._update_metrics(0, False)
            raise CortexError(f"Failed to store conversation: {e}")
    
    def get_conversation(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve a conversation by memory ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Conversation data or None if not found
            
        Raises:
            UsageLimitError: If usage limits are exceeded
            CortexError: If retrieval fails
        """
        try:
            if not self._check_usage_limits('retrieve_conversation'):
                raise UsageLimitError("Usage limit exceeded for conversation retrieval")
            
            start_time = time.time()
            
            conversation = get_conversation(memory_id)
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, conversation is not None)
            
            self._track_usage('retrieve_conversation', {
                'memory_id': memory_id
            })
            
            return conversation
            
        except UsageLimitError:
            raise
        except Exception as e:
            self._update_metrics(0, False)
            raise CortexError(f"Failed to retrieve conversation: {e}")
    
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
            
        Raises:
            UsageLimitError: If usage limits are exceeded
            CortexError: If search fails
        """
        try:
            if not self._check_usage_limits('semantic_search'):
                raise UsageLimitError("Usage limit exceeded for semantic search")
            
            start_time = time.time()
            
            similar_contexts = semantic_embeddings.find_semantically_similar_context(
                user_id=self.user_id,
                current_prompt=prompt,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
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
            
            logger.info(f"🔍 Found {len(results)} semantic contexts")
            return results
            
        except UsageLimitError:
            raise
        except Exception as e:
            self._update_metrics(0, False)
            raise CortexError(f"Failed to find semantic context: {e}")
    
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
            
        Raises:
            UsageLimitError: If usage limits are exceeded
            CortexError: If search fails
        """
        try:
            if not self._check_usage_limits('evolving_search'):
                raise UsageLimitError("Usage limit exceeded for evolving search")
            
            start_time = time.time()
            
            evolving_contexts = self_evolving_context.find_evolving_context(
                user_id=self.user_id,
                current_prompt=prompt,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
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
            
            logger.info(f"🧠 Found {len(results)} evolving contexts")
            return results
            
        except UsageLimitError:
            raise
        except Exception as e:
            self._update_metrics(0, False)
            raise CortexError(f"Failed to find evolving context: {e}")
    
    def generate_with_context(self, prompt: str, context_method: str = "semantic", provider: str = "auto") -> str:
        """
        Generate response with automatic context injection.
        
        Args:
            prompt: User's prompt
            context_method: "semantic", "evolving", "hybrid", or "adaptive"
            provider: LLM provider to use ("auto", "gemini", "claude", "openai")
            
        Returns:
            Generated response with injected context
            
        Raises:
            UsageLimitError: If usage limits are exceeded
            CortexError: If generation fails
        """
        try:
            if not self._check_usage_limits('generation'):
                raise UsageLimitError("Usage limit exceeded for response generation")
            
            start_time = time.time()
            
            if context_method == "semantic":
                response = generate_with_context(
                    user_id=self.user_id,
                    prompt=prompt,
                    provider=provider
                )
            elif context_method == "evolving":
                response = generate_with_evolving_context(
                    user_id=self.user_id,
                    prompt=prompt,
                    provider=provider
                )
            elif context_method == "hybrid":
                response = generate_with_hybrid_context(
                    user_id=self.user_id,
                    prompt=prompt,
                    provider=provider
                )
            elif context_method == "adaptive":
                response = generate_with_adaptive_context(
                    user_id=self.user_id,
                    prompt=prompt,
                    provider=provider
                )
            else:
                raise ValueError("Invalid context_method. Use 'semantic', 'evolving', 'hybrid', or 'adaptive'")
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
            self._track_usage('generation', {
                'prompt_length': len(prompt),
                'response_length': len(response),
                'context_method': context_method,
                'provider': provider
            })
            
            logger.info(f"🤖 Generated response with {context_method} context using {provider}")
            return response
            
        except UsageLimitError:
            raise
        except Exception as e:
            self._update_metrics(0, False)
            raise CortexError(f"Failed to generate response: {e}")
    
    def get_analytics(self) -> Dict:
        """
        Get user analytics and usage statistics.
        
        Returns:
            Analytics data
            
        Raises:
            UsageLimitError: If usage limits are exceeded
            CortexError: If analytics retrieval fails
        """
        try:
            if not self._check_usage_limits('analytics'):
                raise UsageLimitError("Usage limit exceeded for analytics")
            
            start_time = time.time()
            
            metrics = self_evolving_context.get_performance_metrics(self.user_id)
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
            self._track_usage('analytics', {
                'metrics_requested': list(metrics.keys())
            })
            
            return metrics
            
        except UsageLimitError:
            raise
        except Exception as e:
            self._update_metrics(0, False)
            raise CortexError(f"Failed to get analytics: {e}")
    
    def detect_drift(self, time_window_hours: int = 24) -> Dict:
        """
        Detect semantic drift with usage tracking.
        
        Args:
            time_window_hours: Time window for drift detection
            
        Returns:
            Drift analysis results
            
        Raises:
            UsageLimitError: If usage limits are exceeded
            CortexError: If drift detection fails
        """
        try:
            if not self._check_usage_limits('drift_detection'):
                raise UsageLimitError("Usage limit exceeded for drift detection")
            
            start_time = time.time()
            
            drift_results = detect_semantic_drift(
                user_id=self.user_id,
                time_window_hours=time_window_hours
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
            self._track_usage('drift_detection', {
                'time_window_hours': time_window_hours,
                'drift_detected': drift_results.get('drift_detected', False)
            })
            
            return drift_results
            
        except UsageLimitError:
            raise
        except Exception as e:
            self._update_metrics(0, False)
            raise CortexError(f"Failed to detect drift: {e}")
    
    def prune_memories(self, threshold: float = 0.3) -> Dict:
        """
        Prune low-impact memories with usage tracking.
        
        Args:
            threshold: Pruning threshold
            
        Returns:
            Pruning statistics
            
        Raises:
            UsageLimitError: If usage limits are exceeded
            CortexError: If pruning fails
        """
        try:
            if not self._check_usage_limits('pruning'):
                raise UsageLimitError("Usage limit exceeded for memory pruning")
            
            start_time = time.time()
            
            pruning_stats = self_evolving_context.auto_pruning.prune_low_impact_memories(
                user_id=self.user_id,
                threshold=threshold
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
            self._track_usage('pruning', {
                'threshold': threshold,
                'pruned_count': pruning_stats.get('pruned_memories', 0)
            })
            
            return pruning_stats
            
        except UsageLimitError:
            raise
        except Exception as e:
            self._update_metrics(0, False)
            raise CortexError(f"Failed to prune memories: {e}")
    
    @retry_on_failure(max_retries=2)
    def get_usage_stats(self) -> Dict:
        """
        Get current usage statistics for the user.
        
        Returns:
            Usage statistics
            
        Raises:
            CortexError: If stats retrieval fails
        """
        try:
            start_time = time.time()
            
            response = self.session.get(
                f"{self.base_url}/usage/stats",
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, response.status_code == 200)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise CortexError(f"Failed to get usage stats: {e}")
    
    @retry_on_failure(max_retries=2)
    def get_plan_info(self) -> Dict:
        """
        Get current plan information and limits.
        
        Returns:
            Plan information
            
        Raises:
            CortexError: If plan info retrieval fails
        """
        try:
            start_time = time.time()
            
            response = self.session.get(
                f"{self.base_url}/auth/plan",
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, response.status_code == 200)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise CortexError(f"Failed to get plan info: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """
        Get client performance metrics.
        
        Returns:
            Performance metrics
        """
        return {
            **self.metrics,
            'success_rate': (
                self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1)
            ),
            'circuit_breaker_states': {
                name: breaker.state for name, breaker in self.circuit_breakers.items()
            }
        }
    
    def close(self):
        """Close the client session and cleanup resources."""
        self.session.close()
        logger.info("🔒 Cortex client session closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()