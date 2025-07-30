# 🧠 Cortex Memory SDK Documentation
## Enterprise-Grade Context-Aware AI System

---

## 📦 Installation

```bash
# Install from PyPI
pip install cortex-memory

# Install with optional dependencies
pip install cortex-memory[api,ml,monitoring]

# Install for development
pip install cortex-memory[dev]
```

---

## 🚀 Quick Start

### Basic Usage

```python
from cortex import create_client

# Create client with your API key
client = create_client("your-api-key-here")

# Generate response with automatic context injection
response = client.generate_with_context("How do I implement authentication?")
print(response)
```

### Advanced Usage

```python
from cortex import CortexClient

# Create client with custom configuration
client = CortexClient(
    api_key="your-api-key-here",
    base_url="https://api.cortex-memory.com",
    timeout=30,
    max_retries=3
)

# Store a conversation
memory_id = client.store_conversation(
    prompt="How do I implement JWT authentication?",
    response="Use JWT tokens with proper expiration and refresh mechanisms.",
    metadata={"topic": "authentication", "difficulty": "intermediate"}
)

# Find similar context
contexts = client.find_semantic_context(
    prompt="What's the best way to secure my API?",
    limit=5,
    similarity_threshold=0.3
)

# Generate with evolving context
response = client.generate_with_context(
    prompt="How do I secure my API?",
    context_method="evolving"
)
```

---

## 🔑 API Key Management

### Getting an API Key

1. **Register** at [cortex-memory.com](https://cortex-memory.com)
2. **Choose a plan** (Free, Starter, Pro, Enterprise)
3. **Generate API key** in your dashboard
4. **Start using** the SDK

### API Key Security

```python
# Store API key securely
import os
from cortex import create_client

# Use environment variable
api_key = os.getenv("CORTEX_API_KEY")
client = create_client(api_key)

# Or use a secure configuration file
# config.py
CORTEX_API_KEY = "your-secure-api-key"
```

---

## 🧠 Core Features

### 1. Context-Aware Response Generation

```python
# Semantic context injection
response = client.generate_with_context(
    prompt="How do I implement user authentication?",
    context_method="semantic"  # Uses semantic similarity
)

# Self-evolving context injection
response = client.generate_with_context(
    prompt="How do I implement user authentication?",
    context_method="evolving"  # Uses adaptive learning
)
```

### 2. Memory Management

```python
# Store conversations
memory_id = client.store_conversation(
    prompt="User question",
    response="AI response",
    metadata={"topic": "authentication", "tags": ["security", "api"]}
)

# Retrieve conversations
conversation = client.get_conversation(memory_id)
print(conversation["prompt"])
print(conversation["response"])
```

### 3. Semantic Search

```python
# Find similar contexts
contexts = client.find_semantic_context(
    prompt="How do I secure my application?",
    limit=5,
    similarity_threshold=0.3
)

for context, score in contexts:
    print(f"Similarity: {score:.3f}")
    print(f"Prompt: {context['prompt']}")
    print(f"Response: {context['response'][:100]}...")
```

### 4. Self-Evolving Context

```python
# Find context using adaptive algorithms
contexts = client.find_evolving_context(
    prompt="How do I implement OAuth?",
    limit=3,
    similarity_threshold=0.4
)

# The system learns from each interaction
# and improves context relevance over time
```

---

## 📊 Analytics & Monitoring

### Performance Metrics

```python
# Get client performance metrics
metrics = client.get_performance_metrics()
print(f"Success Rate: {metrics['success_rate']:.2%}")
print(f"Average Response Time: {metrics['average_response_time']:.3f}s")
print(f"Total Requests: {metrics['total_requests']}")
```

### Usage Analytics

```python
# Get user analytics
analytics = client.get_analytics()
print(f"Total Memories: {analytics['total_memories']}")
print(f"Context Hit Rate: {analytics['context_hit_rate']:.2%}")
print(f"Memory Retention: {analytics['memory_retention_days']} days")
```

### Semantic Drift Detection

```python
# Detect changes in user behavior or system performance
drift_results = client.detect_drift(time_window_hours=24)
if drift_results['drift_detected']:
    print(f"Drift detected with confidence: {drift_results['confidence']:.2%}")
    print(f"Recommendation: {drift_results['recommendation']}")
```

---

## 🧹 Memory Management

### Auto-Pruning

```python
# Automatically remove low-impact memories
pruning_stats = client.prune_memories(threshold=0.3)
print(f"Pruned {pruning_stats['pruned_memories']} memories")
print(f"Remaining: {pruning_stats['remaining_memories']} memories")
```

### Manual Memory Management

```python
# Get usage statistics
usage_stats = client.get_usage_stats()
print(f"API Calls This Month: {usage_stats['monthly_calls']}")
print(f"Plan: {usage_stats['plan']}")
print(f"Remaining Calls: {usage_stats['remaining_calls']}")
```

---

## ⚡ Error Handling

### Comprehensive Error Types

```python
from cortex import (
    CortexError,
    AuthenticationError,
    UsageLimitError,
    RateLimitError,
    CircuitBreakerError
)

try:
    response = client.generate_with_context("Test prompt")
except AuthenticationError:
    print("Invalid API key")
except UsageLimitError:
    print("Usage limit exceeded")
except RateLimitError:
    print("Rate limit exceeded")
except CircuitBreakerError:
    print("Service temporarily unavailable")
except CortexError as e:
    print(f"Unexpected error: {e}")
```

### Retry Logic

The SDK automatically handles retries with exponential backoff:

```python
# Automatic retry on network failures
client = CortexClient(
    api_key="your-key",
    max_retries=3,  # Retry up to 3 times
    timeout=30      # 30 second timeout
)
```

### Circuit Breaker

Built-in circuit breaker prevents cascading failures:

```python
# Circuit breaker automatically opens on repeated failures
# and recovers after a timeout period
client = CortexClient(
    api_key="your-key",
    timeout=30
)
```

---

## 🔧 Advanced Configuration

### Custom Session Configuration

```python
import requests
from cortex import CortexClient

# Create custom session
session = requests.Session()
session.headers.update({
    'Custom-Header': 'value',
    'User-Agent': 'MyApp/1.0'
})

# Use with client (advanced usage)
client = CortexClient(
    api_key="your-key",
    base_url="https://api.cortex-memory.com"
)
client.session = session
```

### Context Manager Usage

```python
# Automatic resource cleanup
with CortexClient(api_key="your-key") as client:
    response = client.generate_with_context("Test prompt")
    # Session automatically closed after context exit
```

---

## 📈 Performance Optimization

### Batch Operations

```python
# Store multiple conversations efficiently
conversations = [
    ("How do I implement auth?", "Use JWT tokens..."),
    ("What's the best database?", "PostgreSQL is recommended..."),
    ("How to deploy to production?", "Use Docker containers...")
]

for prompt, response in conversations:
    client.store_conversation(prompt, response)
```

### Caching Strategies

```python
# The SDK automatically caches embeddings and frequently accessed data
# No additional configuration needed for optimal performance
```

---

## 🧪 Testing

### Unit Testing

```python
import pytest
from unittest.mock import patch
from cortex import CortexClient

def test_client_initialization():
    with patch('cortex.client.requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'user_id': 'test-user',
            'plan': 'free',
            'usage_limits': {}
        }
        mock_session.return_value.get.return_value = mock_response
        
        client = CortexClient(api_key="test-key")
        assert client.user_id == 'test-user'
```

### Integration Testing

```python
def test_complete_workflow():
    client = CortexClient(api_key="test-key")
    
    # Test complete workflow
    memory_id = client.store_conversation("Test prompt", "Test response")
    contexts = client.find_semantic_context("Test query")
    response = client.generate_with_context("Test generation")
    
    assert memory_id is not None
    assert len(contexts) >= 0
    assert response is not None
```

---

## 🔒 Security Best Practices

### API Key Security

```python
# ✅ Good: Use environment variables
import os
api_key = os.getenv("CORTEX_API_KEY")

# ❌ Bad: Hardcode API keys
api_key = "your-actual-api-key-here"
```

### Error Handling

```python
# ✅ Good: Handle specific errors
try:
    response = client.generate_with_context(prompt)
except UsageLimitError:
    # Handle usage limits gracefully
    pass

# ❌ Bad: Catch all exceptions
try:
    response = client.generate_with_context(prompt)
except:
    # Too broad, may hide important errors
    pass
```

### Rate Limiting

```python
# ✅ Good: Respect rate limits
import time

for prompt in prompts:
    try:
        response = client.generate_with_context(prompt)
    except RateLimitError:
        time.sleep(1)  # Wait before retrying
        continue
```

---

## 📚 Examples

### Chatbot Integration

```python
from cortex import create_client

class Chatbot:
    def __init__(self, api_key):
        self.client = create_client(api_key)
    
    def respond(self, user_message):
        # Generate context-aware response
        response = self.client.generate_with_context(
            user_message,
            context_method="evolving"
        )
        
        # Store the conversation
        self.client.store_conversation(
            user_message,
            response,
            metadata={"session_id": "chat-123"}
        )
        
        return response

# Usage
bot = Chatbot("your-api-key")
response = bot.respond("How do I implement user authentication?")
```

### API Wrapper

```python
from cortex import CortexClient
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/chat")
async def chat(message: str, api_key: str):
    try:
        client = CortexClient(api_key=api_key)
        response = client.generate_with_context(message)
        return {"response": response}
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except UsageLimitError:
        raise HTTPException(status_code=429, detail="Usage limit exceeded")
```

### Memory Management System

```python
from cortex import create_client

class MemoryManager:
    def __init__(self, api_key):
        self.client = create_client(api_key)
    
    def add_knowledge(self, question, answer, topic=None):
        return self.client.store_conversation(
            question, answer,
            metadata={"topic": topic, "type": "knowledge"}
        )
    
    def search_knowledge(self, query, limit=5):
        return self.client.find_semantic_context(query, limit=limit)
    
    def get_insights(self):
        return self.client.get_analytics()
    
    def cleanup(self):
        return self.client.prune_memories(threshold=0.2)

# Usage
manager = MemoryManager("your-api-key")
manager.add_knowledge(
    "How do I implement JWT?",
    "Use JWT tokens with proper expiration...",
    topic="authentication"
)
```

---

## 🆘 Troubleshooting

### Common Issues

#### Authentication Errors
```python
# Problem: Invalid API key
# Solution: Check your API key and ensure it's valid
client = create_client("correct-api-key-here")
```

#### Usage Limit Errors
```python
# Problem: Exceeded usage limits
# Solution: Upgrade your plan or wait for reset
try:
    response = client.generate_with_context(prompt)
except UsageLimitError:
    print("Upgrade your plan for more API calls")
```

#### Network Errors
```python
# Problem: Network connectivity issues
# Solution: Check internet connection and API endpoint
client = CortexClient(
    api_key="your-key",
    base_url="https://api.cortex-memory.com",
    timeout=60  # Increase timeout
)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create client with debug info
client = create_client("your-key")
# All API calls will be logged
```

---

## 📞 Support

- **Documentation**: [docs.cortex-memory.com](https://docs.cortex-memory.com)
- **GitHub**: [github.com/cortex-memory/cortex-memory](https://github.com/cortex-memory/cortex-memory)
- **Email**: support@cortex-memory.com
- **Discord**: [discord.gg/cortex-memory](https://discord.gg/cortex-memory)

---

## 📄 License

MIT License - see [LICENSE](https://github.com/cortex-memory/cortex-memory/blob/main/LICENSE) for details. 