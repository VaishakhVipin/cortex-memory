# 🧠 Cortex Memory - Complete Documentation

## 📖 **Table of Contents**

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Configuration](#configuration)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

---

## 🎯 **Overview**

Cortex Memory is an enterprise-grade context-aware AI system that provides intelligent memory management and semantic understanding for LLM applications. It combines semantic embeddings, self-evolving context, and multi-LLM support to create a powerful context layer for prompt chains.

### **Key Features**
- 🧠 **Semantic Context**: Find relevant past conversations using embeddings
- 🔄 **Self-Evolving Context**: Adaptive learning that improves over time
- 🤖 **Multi-LLM Support**: Gemini, Claude, OpenAI with automatic fallback
- 🎯 **Hybrid Context**: Combine multiple context strategies
- 📊 **Analytics & Monitoring**: Comprehensive performance tracking
- 🗑️ **Auto-Pruning**: Intelligent memory management
- 🚀 **Enterprise Ready**: Error handling, retry logic, circuit breakers

---

## 📦 **Installation**

### **From PyPI**
```bash
pip install cortex-memory
```

### **With Optional Dependencies**
```bash
# For API development
pip install cortex-memory[api]

# For monitoring and analytics
pip install cortex-memory[monitoring]

# For development
pip install cortex-memory[dev]
```

### **From Source**
```bash
git clone https://github.com/cortex-memory/cortex-memory.git
cd cortex-memory/backend
pip install -e .
```

---

## 🚀 **Quick Start**

### **1. Basic Setup**
```python
from cortex_memory import create_client

# Create client with your API key
client = create_client("your-api-key")

# Generate response with context
response = client.generate_with_context("How do I implement authentication?")
print(response)
```

### **2. Direct Function Usage**
```python
from cortex_memory import generate_with_context

# Generate response with semantic context
response = generate_with_context(
    user_id="user123",
    prompt="What's the best database for my use case?",
    provider="claude"  # Use specific LLM provider
)
```

### **3. Multi-LLM Support**
```python
from cortex_memory import call_llm_api, llm_manager

# Use specific provider
response = call_llm_api("Hello world", provider="openai")

# Check available providers
status = llm_manager.get_provider_status()
print(status)
```

---

## 🧠 **Core Concepts**

### **1. Context Generation Methods**

#### **Semantic Context**
Uses embeddings to find semantically similar past conversations.

```python
from cortex_memory import generate_with_context

response = generate_with_context(
    user_id="user123",
    prompt="How do I implement JWT?",
    provider="auto"
)
```

#### **Self-Evolving Context**
Uses adaptive learning to improve context relevance over time.

```python
from cortex_memory import generate_with_evolving_context

response = generate_with_evolving_context(
    user_id="user123",
    prompt="Based on our previous discussions...",
    provider="claude"
)
```

#### **Hybrid Context**
Combines semantic and evolving context for optimal results.

```python
from cortex_memory import generate_with_hybrid_context

response = generate_with_hybrid_context(
    user_id="user123",
    prompt="Complex technical question",
    semantic_weight=0.6,  # 60% semantic
    evolving_weight=0.4,  # 40% evolving
    provider="openai"
)
```

#### **Adaptive Context**
Automatically chooses the best context method based on query characteristics.

```python
from cortex_memory import generate_with_adaptive_context

response = generate_with_adaptive_context(
    user_id="user123",
    prompt="Your question here",
    provider="auto"
)
```

### **2. LLM Providers**

#### **Supported Providers**
- **Gemini**: Google Gemini 2.0 Flash
- **Claude**: Anthropic Claude 3.5 Sonnet
- **OpenAI**: GPT-4o Mini

#### **Provider Selection**
```python
# Automatic selection (recommended)
response = generate_with_context(user_id, prompt, provider="auto")

# Specific provider
response = generate_with_context(user_id, prompt, provider="claude")

# Direct LLM call
response = call_llm_api(prompt, provider="openai")
```

### **3. Memory Management**

#### **Store Conversations**
```python
from cortex_memory import store_conversation

memory_id = store_conversation(
    user_id="user123",
    prompt="How do I implement auth?",
    response="Here's how to implement authentication...",
    metadata={"topic": "authentication", "difficulty": "intermediate"}
)
```

#### **Retrieve Conversations**
```python
from cortex_memory import get_conversation

conversation = get_conversation(memory_id)
print(conversation)
```

---

## 📚 **API Reference**

### **Core Functions**

#### **`generate_with_context(user_id, prompt, provider="auto")`**
Generate response with semantic context injection.

**Parameters:**
- `user_id` (str): User identifier
- `prompt` (str): User's prompt
- `provider` (str): LLM provider ("auto", "gemini", "claude", "openai")

**Returns:**
- `str`: Generated response with context

#### **`generate_with_evolving_context(user_id, prompt, provider="auto")`**
Generate response with self-evolving context injection.

#### **`generate_with_hybrid_context(user_id, prompt, semantic_weight=0.6, evolving_weight=0.4, provider="auto")`**
Generate response with hybrid context combining semantic and evolving methods.

#### **`generate_with_adaptive_context(user_id, prompt, provider="auto")`**
Generate response with adaptive context selection based on query characteristics.

### **LLM Functions**

#### **`call_llm_api(prompt, provider="auto", max_tokens=512, **kwargs)`**
Unified LLM API call function.

**Parameters:**
- `prompt` (str): Input prompt
- `provider` (str): Provider to use
- `max_tokens` (int): Maximum output tokens
- `**kwargs`: Provider-specific parameters (temperature, top_p, etc.)

#### **`llm_manager.get_provider_status()`**
Get status of all LLM providers.

**Returns:**
- `Dict`: Provider availability and configuration

### **Memory Functions**

#### **`store_conversation(user_id, prompt, response, metadata=None)`**
Store a conversation in memory.

#### **`get_conversation(memory_id)`**
Retrieve a conversation by ID.

### **Client Class**

#### **`CortexClient(api_key, base_url="https://api.cortex-memory.com", timeout=30, max_retries=3)`**
Enterprise-grade client with error handling and usage tracking.

**Methods:**
- `generate_with_context(prompt, context_method="semantic", provider="auto")`
- `store_conversation(prompt, response, metadata=None)`
- `get_conversation(memory_id)`
- `find_semantic_context(prompt, limit=5, similarity_threshold=0.3)`
- `get_analytics()`
- `detect_drift(time_window_hours=24)`
- `prune_memories(threshold=0.3)`

---

## 🔧 **Advanced Usage**

### **1. Custom Context Strategies**

#### **Semantic Search**
```python
from cortex_memory import semantic_embeddings

# Find similar contexts
similar_contexts = semantic_embeddings.find_semantically_similar_context(
    user_id="user123",
    current_prompt="How do I implement OAuth2?",
    limit=5,
    similarity_threshold=0.3
)

for context, score in similar_contexts:
    print(f"Score: {score:.3f}")
    print(f"Q: {context.get('prompt')}")
    print(f"A: {context.get('response')}")
```

#### **Evolving Context Analysis**
```python
from cortex_memory import self_evolving_context

# Get evolving analytics
analytics = self_evolving_context.get_evolving_analytics("user123")
print(analytics)
```

### **2. Batch Operations**

#### **Store Multiple Conversations**
```python
from cortex_memory import semantic_embeddings

conversations = [
    {
        "user_id": "user123",
        "prompt": "How do I implement auth?",
        "response": "Here's how...",
        "metadata": {"topic": "authentication"}
    },
    {
        "user_id": "user123", 
        "prompt": "What's the best database?",
        "response": "For your use case...",
        "metadata": {"topic": "databases"}
    }
]

memory_ids = semantic_embeddings.store_conversations_batch(conversations)
```

### **3. Analytics and Monitoring**

#### **Get Context Analytics**
```python
from cortex_memory import get_context_analytics

analytics = get_context_analytics("user123")
print(analytics)
```

#### **Detect Semantic Drift**
```python
from cortex_memory import detect_semantic_drift

drift_analysis = detect_semantic_drift("user123", time_window_hours=24)
print(drift_analysis)
```

### **4. Error Handling**

#### **Custom Exceptions**
```python
from cortex_memory import (
    CortexClient, 
    CortexError, 
    AuthenticationError, 
    UsageLimitError,
    RateLimitError,
    CircuitBreakerError
)

try:
    client = CortexClient("your-api-key")
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
    print(f"Other error: {e}")
```

---

## ⚙️ **Configuration**

### **Environment Variables**

#### **Required**
```bash
# Redis connection
REDIS_URL=redis://localhost:6379

# LLM API keys (at least one required)
GEMINI_API_KEY=your_gemini_key
CLAUDE_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
```

#### **Optional**
```bash
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password

# Logging
LOG_LEVEL=INFO
```

### **Redis Setup**

#### **Local Redis**
```bash
# Install Redis
# macOS
brew install redis

# Ubuntu
sudo apt-get install redis-server

# Start Redis
redis-server
```

#### **Docker Redis**
```bash
docker run -d -p 6379:6379 redis:alpine
```

### **LLM Provider Setup**

#### **Gemini**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create API key
3. Set environment variable: `GEMINI_API_KEY=your_key`

#### **Claude**
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create API key
3. Set environment variable: `CLAUDE_API_KEY=your_key`

#### **OpenAI**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create API key
3. Set environment variable: `OPENAI_API_KEY=your_key`

---

## 📝 **Examples**

### **1. Chatbot with Memory**

```python
from cortex_memory import create_client

class MemoryChatbot:
    def __init__(self, api_key: str):
        self.client = create_client(api_key)
        self.user_id = "chatbot_user"
    
    def chat(self, message: str) -> str:
        """Generate response with context from previous conversations."""
        try:
            response = self.client.generate_with_context(
                prompt=message,
                context_method="hybrid",  # Use hybrid context
                provider="auto"  # Auto-select best provider
            )
            return response
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"

# Usage
chatbot = MemoryChatbot("your-api-key")
response = chatbot.chat("How do I implement authentication?")
print(response)
```

### **2. API Wrapper**

```python
from cortex_memory import generate_with_adaptive_context
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: dict):
    try:
        user_id = request.get("user_id")
        message = request.get("message")
        provider = request.get("provider", "auto")
        
        if not user_id or not message:
            raise HTTPException(status_code=400, detail="Missing user_id or message")
        
        response = generate_with_adaptive_context(
            user_id=user_id,
            prompt=message,
            provider=provider
        )
        
        return {"response": response, "provider": provider}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### **3. Memory Management System**

```python
from cortex_memory import (
    store_conversation, 
    get_conversation, 
    detect_semantic_drift,
    semantic_embeddings
)

class MemoryManager:
    def __init__(self, user_id: str):
        self.user_id = user_id
    
    def add_memory(self, prompt: str, response: str, metadata: dict = None):
        """Add a new memory."""
        memory_id = store_conversation(
            user_id=self.user_id,
            prompt=prompt,
            response=response,
            metadata=metadata
        )
        return memory_id
    
    def get_memory(self, memory_id: str):
        """Retrieve a specific memory."""
        return get_conversation(memory_id)
    
    def find_similar_memories(self, query: str, limit: int = 5):
        """Find memories similar to query."""
        return semantic_embeddings.find_semantically_similar_context(
            user_id=self.user_id,
            current_prompt=query,
            limit=limit
        )
    
    def check_drift(self):
        """Check for semantic drift."""
        return detect_semantic_drift(self.user_id)

# Usage
manager = MemoryManager("user123")
memory_id = manager.add_memory(
    "How do I implement JWT?",
    "Here's how to implement JWT authentication...",
    {"topic": "authentication", "difficulty": "intermediate"}
)

similar = manager.find_similar_memories("JWT implementation")
drift = manager.check_drift()
```

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **1. Redis Connection Error**
```
Error: Redis connection failed
```
**Solution:**
- Ensure Redis is running: `redis-server`
- Check Redis URL: `REDIS_URL=redis://localhost:6379`
- Verify Redis credentials if using authentication

#### **2. LLM Provider Not Available**
```
Error: All LLM providers failed
```
**Solution:**
- Set at least one API key: `GEMINI_API_KEY`, `CLAUDE_API_KEY`, or `OPENAI_API_KEY`
- Check API key validity
- Verify internet connection

#### **3. Semantic Model Loading Slow**
```
Loading semantic model: all-MiniLM-L6-v2...
```
**Solution:**
- This is normal on first run (takes ~4-5 seconds)
- Model is cached for subsequent runs
- Consider using a faster model if needed

#### **4. Memory Not Found**
```
Error: No contexts found
```
**Solution:**
- Ensure conversations are stored first
- Check user_id consistency
- Lower similarity threshold for more results

### **Performance Optimization**

#### **1. Batch Operations**
```python
# Use batch storage for multiple conversations
semantic_embeddings.store_conversations_batch(conversations)
```

#### **2. Caching**
```python
# Enable in-memory caching (enabled by default)
# Cache size is configurable in semantic_embeddings.py
```

#### **3. Background Processing**
```python
# Skip expensive background operations for faster response
semantic_embeddings.store_conversation_embedding(
    user_id, prompt, response, skip_background_processing=True
)
```

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/cortex-memory/cortex-memory.git
cd cortex-memory/backend

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
black cortex_memory/
isort cortex_memory/

# Type checking
mypy cortex_memory/
```

### **Code Style**
- Follow PEP 8
- Use type hints
- Add docstrings for all functions
- Write tests for new features

### **Testing**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_client.py

# Run with coverage
pytest --cov=cortex_memory

# Run integration tests
pytest -m integration
```

---

## 📞 **Support**

### **Documentation**
- [API Reference](https://docs.cortex-memory.com)
- [Examples](https://github.com/cortex-memory/cortex-memory/tree/main/examples)
- [Changelog](https://github.com/cortex-memory/cortex-memory/blob/main/CHANGELOG.md)

### **Community**
- [GitHub Issues](https://github.com/cortex-memory/cortex-memory/issues)
- [Discussions](https://github.com/cortex-memory/cortex-memory/discussions)
- [Discord](https://discord.gg/cortex-memory)

### **Enterprise Support**
- Email: enterprise@cortex-memory.com
- [Enterprise Documentation](https://enterprise.cortex-memory.com)

---

## 📄 **License**

MIT License - see [LICENSE](https://github.com/cortex-memory/cortex-memory/blob/main/LICENSE) file for details. 