# 🎯 Provider Parameter Feature

## ✅ **New Feature: LLM Provider Selection**

Users can now specify which LLM provider to use for context generation by passing a `provider` parameter to any context generation function.

## 🚀 **Usage Examples**

### **1. Direct Function Calls**

#### **Semantic Context with Specific Provider**
```python
from cortex_memory import generate_with_context

# Use Claude for semantic context
response = generate_with_context(
    user_id="user123",
    prompt="How do I implement authentication?",
    provider="claude"  # 🆕 Specify provider
)
```

#### **Hybrid Context with OpenAI**
```python
from cortex_memory import generate_with_hybrid_context

# Use OpenAI for hybrid context
response = generate_with_hybrid_context(
    user_id="user123",
    prompt="What's the best way to secure my API?",
    semantic_weight=0.6,
    evolving_weight=0.4,
    provider="openai"  # 🆕 Specify provider
)
```

#### **Evolving Context with Gemini**
```python
from cortex_memory import generate_with_evolving_context

# Use Gemini for evolving context
response = generate_with_evolving_context(
    user_id="user123",
    prompt="Based on our previous discussions...",
    provider="gemini"  # 🆕 Specify provider
)
```

#### **Adaptive Context with Auto-Selection**
```python
from cortex_memory import generate_with_adaptive_context

# Let the system choose the best provider automatically
response = generate_with_adaptive_context(
    user_id="user123",
    prompt="Complex technical question...",
    provider="auto"  # 🆕 Default behavior
)
```

### **2. Client Usage**

#### **CortexClient with Provider Selection**
```python
from cortex_memory import create_client

client = create_client("your-api-key")

# Use Claude for semantic context
response = client.generate_with_context(
    prompt="How do I implement JWT?",
    context_method="semantic",
    provider="claude"  # 🆕 Specify provider
)

# Use OpenAI for hybrid context
response = client.generate_with_context(
    prompt="What's the best database for my use case?",
    context_method="hybrid",
    provider="openai"  # 🆕 Specify provider
)

# Use Gemini for evolving context
response = client.generate_with_context(
    prompt="Based on our previous discussions...",
    context_method="evolving",
    provider="gemini"  # 🆕 Specify provider
)
```

### **3. Advanced Usage**

#### **Provider-Specific Parameters**
```python
from cortex_memory import call_llm_api

# Use Claude with custom parameters
response = call_llm_api(
    "Generate a creative story",
    provider="claude",
    max_tokens=1000,
    temperature=0.9,
    top_p=0.95
)
```

#### **Provider Status Check**
```python
from cortex_memory import llm_manager

# Check which providers are available
status = llm_manager.get_provider_status()
print(status)
# Output:
# {
#   'gemini': {'available': True, 'model': 'gemini-2.0-flash', 'has_api_key': True},
#   'claude': {'available': False, 'model': 'claude-3-5-sonnet-20241022', 'has_api_key': False},
#   'openai': {'available': False, 'model': 'gpt-4o-mini', 'has_api_key': False}
# }
```

## 📋 **Available Providers**

### **Provider Options**
- **`"auto"`** (default): Automatically selects the best available provider
- **`"gemini"`**: Google Gemini 2.0 Flash
- **`"claude"`**: Anthropic Claude 3.5 Sonnet
- **`"openai"`**: OpenAI GPT-4o Mini

### **Provider Models**
```python
# Default models for each provider
GEMINI_MODEL = "gemini-2.0-flash"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
OPENAI_MODEL = "gpt-4o-mini"
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Set up API keys for desired providers
GEMINI_API_KEY=your_gemini_key
CLAUDE_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
```

### **Provider Availability**
- **Gemini**: Available if `GEMINI_API_KEY` is set
- **Claude**: Available if `CLAUDE_API_KEY` is set
- **OpenAI**: Available if `OPENAI_API_KEY` is set

## 🎯 **Function Signatures**

### **Updated Function Signatures**
```python
# All context generation functions now support provider parameter
def generate_with_context(user_id: str, prompt: str, provider: str = "auto") -> str
def generate_with_evolving_context(user_id: str, prompt: str, provider: str = "auto") -> str
def generate_with_hybrid_context(user_id: str, prompt: str, semantic_weight: float = 0.6, evolving_weight: float = 0.4, provider: str = "auto") -> str
def generate_with_adaptive_context(user_id: str, prompt: str, provider: str = "auto") -> str

# Client method also supports provider
def generate_with_context(self, prompt: str, context_method: str = "semantic", provider: str = "auto") -> str
```

## 🔄 **Fallback Behavior**

### **Automatic Fallback**
When using `provider="auto"` or when a specific provider fails:

1. **Primary**: Tries the specified provider (or default)
2. **Fallback**: If primary fails, tries other available providers
3. **Error**: If all providers fail, returns error message

### **Fallback Order**
```python
# Default fallback order
DEFAULT_PROVIDER = LLMProvider.GEMINI
FALLBACK_PROVIDERS = [LLMProvider.CLAUDE, LLMProvider.OPENAI]
```

## 📊 **Usage Tracking**

### **Provider Information in Metadata**
All generated responses now include provider information:

```python
# Metadata now includes provider
metadata = {
    "context_method": "semantic",
    "contexts_found": 3,
    "provider": "claude"  # 🆕 Provider used
}
```

### **Client Usage Tracking**
```python
# Client tracks provider usage
self._track_usage('generation', {
    'prompt_length': len(prompt),
    'response_length': len(response),
    'context_method': context_method,
    'provider': provider  # 🆕 Provider tracking
})
```

## 🧪 **Testing Examples**

### **Test Different Providers**
```python
from cortex_memory import generate_with_context

# Test with different providers
providers = ["auto", "gemini", "claude", "openai"]

for provider in providers:
    try:
        response = generate_with_context(
            user_id="test_user",
            prompt="Hello, how are you?",
            provider=provider
        )
        print(f"✅ {provider}: {response[:50]}...")
    except Exception as e:
        print(f"❌ {provider}: {e}")
```

### **Test Client with Providers**
```python
from cortex_memory import create_client

client = create_client("test-key")

# Test different context methods with different providers
methods = ["semantic", "evolving", "hybrid", "adaptive"]
providers = ["auto", "gemini", "claude", "openai"]

for method in methods:
    for provider in providers:
        try:
            response = client.generate_with_context(
                prompt="Test prompt",
                context_method=method,
                provider=provider
            )
            print(f"✅ {method} + {provider}: Success")
        except Exception as e:
            print(f"❌ {method} + {provider}: {e}")
```

## 🎉 **Benefits**

### **1. Flexibility**
- ✅ **Provider Choice**: Users can choose their preferred LLM
- ✅ **Cost Optimization**: Use different providers for different use cases
- ✅ **Performance**: Select fastest provider for time-sensitive tasks

### **2. Reliability**
- ✅ **Automatic Fallback**: If one provider fails, others are tried
- ✅ **Redundancy**: Multiple providers ensure service availability
- ✅ **Error Handling**: Graceful degradation when providers are unavailable

### **3. Monitoring**
- ✅ **Usage Tracking**: Track which providers are used most
- ✅ **Performance Metrics**: Monitor response times per provider
- ✅ **Cost Analysis**: Analyze costs by provider

## 🚀 **Next Steps**

1. **✅ Provider Parameter**: Implemented and tested
2. **🔄 Provider Analytics**: Track provider performance
3. **🔄 Cost Optimization**: Smart provider selection based on cost/performance
4. **🔄 Custom Models**: Support for custom model endpoints

**The provider parameter feature is now fully implemented and ready for production use!** 🎯 