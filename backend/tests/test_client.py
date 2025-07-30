#!/usr/bin/env python3
"""
🧠 Test Suite for Enhanced CortexClient
Tests error handling, retry logic, circuit breakers, and performance metrics.
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from cortex.client import (
    CortexClient,
    CortexError,
    AuthenticationError,
    UsageLimitError,
    RateLimitError,
    CircuitBreakerError,
    CircuitBreaker
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_successful_call(self):
        """Test successful function call keeps circuit closed."""
        cb = CircuitBreaker()
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_failure_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=2)
        
        def failing_func():
            raise Exception("test error")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "CLOSED"
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "OPEN"
        assert cb.failure_count == 2
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        def failing_func():
            raise Exception("test error")
        
        # Cause failure to open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should be in HALF_OPEN state
        assert cb.state == "HALF_OPEN"
        
        # Successful call should close circuit
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0


class TestCortexClient:
    """Test CortexClient functionality."""
    
    @pytest.fixture
    def mock_api_key(self):
        return "test-api-key-123"
    
    @pytest.fixture
    def mock_base_url(self):
        return "https://api.test.com"
    
    @pytest.fixture
    def client(self, mock_api_key, mock_base_url):
        """Create a test client with mocked API calls."""
        with patch('cortex.client.requests.Session') as mock_session:
            # Mock successful API key validation
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'user_id': 'test-user-123',
                'plan': 'free',
                'usage_limits': {'generation': 100}
            }
            mock_session.return_value.get.return_value = mock_response
            
            client = CortexClient(api_key=mock_api_key, base_url=mock_base_url)
            return client
    
    def test_client_initialization(self, client, mock_api_key, mock_base_url):
        """Test client initialization and API key validation."""
        assert client.api_key == mock_api_key
        assert client.base_url == mock_base_url
        assert client.user_id == 'test-user-123'
        assert client.plan == 'free'
        assert client.usage_limits == {'generation': 100}
    
    def test_client_authentication_error(self, mock_api_key, mock_base_url):
        """Test authentication error handling."""
        with patch('cortex.client.requests.Session') as mock_session:
            # Mock failed API key validation
            mock_response = Mock()
            mock_response.status_code = 401
            mock_session.return_value.get.return_value = mock_response
            
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                CortexClient(api_key=mock_api_key, base_url=mock_base_url)
    
    def test_client_rate_limit_error(self, mock_api_key, mock_base_url):
        """Test rate limit error handling."""
        with patch('cortex.client.requests.Session') as mock_session:
            # Mock rate limit response
            mock_response = Mock()
            mock_response.status_code = 429
            mock_session.return_value.get.return_value = mock_response
            
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                CortexClient(api_key=mock_api_key, base_url=mock_base_url)
    
    def test_store_conversation_success(self, client):
        """Test successful conversation storage."""
        with patch('cortex.client.store_conversation') as mock_store:
            mock_store.return_value = "memory-123"
            
            memory_id = client.store_conversation(
                prompt="Test prompt",
                response="Test response"
            )
            
            assert memory_id == "memory-123"
            mock_store.assert_called_once_with(
                user_id='test-user-123',
                prompt="Test prompt",
                response="Test response",
                metadata=None
            )
    
    def test_store_conversation_usage_limit(self, client):
        """Test usage limit error during conversation storage."""
        with patch.object(client, '_check_usage_limits', return_value=False):
            with pytest.raises(UsageLimitError, match="Usage limit exceeded"):
                client.store_conversation("Test prompt", "Test response")
    
    def test_find_semantic_context_success(self, client):
        """Test successful semantic context search."""
        with patch('cortex.client.semantic_embeddings') as mock_embeddings:
            mock_embeddings.find_semantically_similar_context.return_value = [
                ({'prompt': 'test', 'response': 'test', 'embedding_id': '123'}, 0.8)
            ]
            
            results = client.find_semantic_context("test query")
            
            assert len(results) == 1
            assert results[0]['similarity_score'] == 0.8
            assert results[0]['memory_id'] == '123'
    
    def test_generate_with_context_semantic(self, client):
        """Test context generation with semantic method."""
        with patch('cortex.client.generate_with_context') as mock_generate:
            mock_generate.return_value = "Generated response with context"
            
            response = client.generate_with_context("test prompt", "semantic")
            
            assert response == "Generated response with context"
            mock_generate.assert_called_once_with(
                user_id='test-user-123',
                prompt="test prompt"
            )
    
    def test_generate_with_context_evolving(self, client):
        """Test context generation with evolving method."""
        with patch('cortex.client.generate_with_evolving_context') as mock_generate:
            mock_generate.return_value = "Generated response with evolving context"
            
            response = client.generate_with_context("test prompt", "evolving")
            
            assert response == "Generated response with evolving context"
            mock_generate.assert_called_once_with(
                user_id='test-user-123',
                prompt="test prompt"
            )
    
    def test_generate_with_context_invalid_method(self, client):
        """Test error handling for invalid context method."""
        with pytest.raises(ValueError, match="Invalid context_method"):
            client.generate_with_context("test prompt", "invalid_method")
    
    def test_get_analytics_success(self, client):
        """Test successful analytics retrieval."""
        with patch('cortex.client.self_evolving_context') as mock_evolving:
            mock_evolving.get_performance_metrics.return_value = {
                'total_memories': 100,
                'context_hit_rate': 0.85
            }
            
            analytics = client.get_analytics()
            
            assert analytics['total_memories'] == 100
            assert analytics['context_hit_rate'] == 0.85
            mock_evolving.get_performance_metrics.assert_called_once_with('test-user-123')
    
    def test_detect_drift_success(self, client):
        """Test successful drift detection."""
        with patch('cortex.client.detect_semantic_drift') as mock_drift:
            mock_drift.return_value = {
                'drift_detected': False,
                'confidence': 0.95
            }
            
            drift_results = client.detect_drift(time_window_hours=24)
            
            assert drift_results['drift_detected'] == False
            assert drift_results['confidence'] == 0.95
            mock_drift.assert_called_once_with(
                user_id='test-user-123',
                time_window_hours=24
            )
    
    def test_prune_memories_success(self, client):
        """Test successful memory pruning."""
        with patch('cortex.client.self_evolving_context') as mock_evolving:
            mock_evolving.auto_pruning.prune_low_impact_memories.return_value = {
                'pruned_memories': 5,
                'remaining_memories': 95
            }
            
            pruning_stats = client.prune_memories(threshold=0.3)
            
            assert pruning_stats['pruned_memories'] == 5
            assert pruning_stats['remaining_memories'] == 95
            mock_evolving.auto_pruning.prune_low_impact_memories.assert_called_once_with(
                user_id='test-user-123',
                threshold=0.3
            )
    
    def test_get_performance_metrics(self, client):
        """Test performance metrics collection."""
        # Simulate some API calls
        client.metrics['total_requests'] = 10
        client.metrics['successful_requests'] = 8
        client.metrics['failed_requests'] = 2
        client.metrics['average_response_time'] = 0.5
        
        metrics = client.get_performance_metrics()
        
        assert metrics['total_requests'] == 10
        assert metrics['successful_requests'] == 8
        assert metrics['failed_requests'] == 2
        assert metrics['success_rate'] == 0.8
        assert 'circuit_breaker_states' in metrics
    
    def test_context_manager(self, mock_api_key, mock_base_url):
        """Test client as context manager."""
        with patch('cortex.client.requests.Session') as mock_session:
            # Mock successful API key validation
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'user_id': 'test-user-123',
                'plan': 'free',
                'usage_limits': {}
            }
            mock_session.return_value.get.return_value = mock_response
            
            with CortexClient(api_key=mock_api_key, base_url=mock_base_url) as client:
                assert client.user_id == 'test-user-123'
                # Session should be closed after context exit
                mock_session.return_value.close.assert_called_once()


class TestRetryLogic:
    """Test retry logic and exponential backoff."""
    
    def test_retry_on_failure_success(self):
        """Test successful retry after initial failure."""
        from cortex.client import retry_on_failure
        
        call_count = 0
        
        @retry_on_failure(max_retries=2)
        def failing_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("temporary failure")
            return "success"
        
        result = failing_then_succeed()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_failure_max_retries_exceeded(self):
        """Test retry logic when max retries are exceeded."""
        from cortex.client import retry_on_failure
        
        @retry_on_failure(max_retries=2)
        def always_fail():
            raise Exception("permanent failure")
        
        with pytest.raises(Exception, match="permanent failure"):
            always_fail()


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_network_timeout_handling(self, mock_api_key, mock_base_url):
        """Test handling of network timeouts."""
        with patch('cortex.client.requests.Session') as mock_session:
            mock_session.return_value.get.side_effect = requests.exceptions.Timeout()
            
            with pytest.raises(CortexError, match="Request timeout"):
                CortexClient(api_key=mock_api_key, base_url=mock_base_url)
    
    def test_connection_error_handling(self, mock_api_key, mock_base_url):
        """Test handling of connection errors."""
        with patch('cortex.client.requests.Session') as mock_session:
            mock_session.return_value.get.side_effect = requests.exceptions.ConnectionError()
            
            with pytest.raises(CortexError, match="Connection error"):
                CortexClient(api_key=mock_api_key, base_url=mock_base_url)
    
    def test_server_error_handling(self, mock_api_key, mock_base_url):
        """Test handling of server errors."""
        with patch('cortex.client.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_session.return_value.get.return_value = mock_response
            
            with pytest.raises(CortexError, match="Server error"):
                CortexClient(api_key=mock_api_key, base_url=mock_base_url)


class TestIntegration:
    """Integration tests for the complete client workflow."""
    
    @pytest.fixture
    def integration_client(self, mock_api_key, mock_base_url):
        """Create a client for integration testing."""
        with patch('cortex.client.requests.Session') as mock_session:
            # Mock API key validation
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'user_id': 'test-user-123',
                'plan': 'pro',
                'usage_limits': {'generation': 1000, 'semantic_search': 500}
            }
            mock_session.return_value.get.return_value = mock_response
            
            return CortexClient(api_key=mock_api_key, base_url=mock_base_url)
    
    def test_complete_workflow(self, integration_client):
        """Test a complete workflow: store -> search -> generate."""
        # Mock all the necessary functions
        with patch('cortex.client.store_conversation') as mock_store, \
             patch('cortex.client.semantic_embeddings') as mock_embeddings, \
             patch('cortex.client.generate_with_context') as mock_generate:
            
            # Setup mocks
            mock_store.return_value = "memory-123"
            mock_embeddings.find_semantically_similar_context.return_value = [
                ({'prompt': 'auth question', 'response': 'use JWT', 'embedding_id': '123'}, 0.9)
            ]
            mock_generate.return_value = "Here's how to implement authentication using JWT tokens..."
            
            # Execute workflow
            memory_id = integration_client.store_conversation(
                "How do I implement authentication?",
                "Use JWT tokens for secure authentication."
            )
            
            contexts = integration_client.find_semantic_context(
                "What's the best way to secure my API?",
                limit=3
            )
            
            response = integration_client.generate_with_context(
                "How do I secure my API?",
                "semantic"
            )
            
            # Verify results
            assert memory_id == "memory-123"
            assert len(contexts) == 1
            assert contexts[0]['similarity_score'] == 0.9
            assert "JWT tokens" in response
            
            # Verify all functions were called correctly
            mock_store.assert_called_once()
            mock_embeddings.find_semantically_similar_context.assert_called_once()
            mock_generate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 