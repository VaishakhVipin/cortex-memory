# 🧠 Cortex Memory SDK

**The Smart Context Layer for Prompt Chains in LLMs**

Built by [Vaishakh Vipin](https://github.com/VaishakhVipin)

## Overview

Cortex Memory SDK is an enterprise-grade context-aware AI system that provides intelligent memory management for Large Language Models (LLMs). It combines semantic understanding with self-evolving patterns to deliver the most relevant context for your AI applications.

## 🚀 Key Features

- **Semantic Context Matching**: Redis-backed semantic search using sentence transformers
- **Self-Evolving Patterns**: Advanced statistical pattern recognition for context relevance
- **Multi-LLM Support**: Seamless integration with Gemini, Claude, and OpenAI
- **Hybrid Context Mode**: Combines semantic and self-evolving context for optimal results
- **Adaptive Context Selection**: Automatically chooses the best context method
- **Auto-Pruning System**: Intelligently manages memory storage and cleanup
- **Semantic Drift Detection**: Monitors and adapts to changing conversation patterns

## 🛠️ Installation

```bash
pip install cortex-memory-sdk
```

## 📖 Quick Start

```python
from cortex_memory import CortexClient

# Initialize the client
client = CortexClient(api_key="your_api_key")

# Generate context-aware responses
response = client.generate_with_context(
    user_id="user123",
    prompt="What did we discuss about AI yesterday?",
    provider="gemini"  # or "claude", "openai", "auto"
)

print(response)
```

## 🔧 Advanced Usage

### Hybrid Context Mode
```python
from cortex_memory.context_manager import generate_with_hybrid_context

response = generate_with_hybrid_context(
    user_id="user123",
    prompt="Explain the latest developments in AI",
    provider="claude"
)
```

### Adaptive Context Selection
```python
from cortex_memory.context_manager import generate_with_adaptive_context

response = generate_with_adaptive_context(
    user_id="user123",
    prompt="What are the key points from our previous meetings?",
    provider="auto"  # Automatically selects best provider
)
```

## 🏗️ Architecture

- **Redis**: High-performance memory storage with semantic embeddings
- **Sentence Transformers**: Dense vector embeddings for semantic similarity
- **Statistical Pattern Recognition**: Robust algorithms for context scoring
- **Multi-Provider LLM Integration**: Unified interface for all major LLM providers

## 📊 Performance

- **Fast Retrieval**: Redis-pipelined operations for sub-second context retrieval
- **Efficient Storage**: Optimized embedding storage and compression
- **Scalable**: Designed for enterprise-scale deployments
- **Cost-Effective**: Intelligent context selection reduces token usage

## 🔒 Security

- API key authentication
- Rate limiting and usage tracking
- Secure Redis connections
- Privacy-focused design

## 📚 Documentation

For detailed documentation, visit: [GitHub Repository](https://github.com/VaishakhVipin/cortex-memory)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/VaishakhVipin/cortex-memory/blob/main/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/VaishakhVipin/cortex-memory/blob/main/LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/VaishakhVipin/cortex-memory/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VaishakhVipin/cortex-memory/discussions)
- **Email**: vaishakh.obelisk@gmail.com

---

**Built with ❤️ by [Vaishakh Vipin](https://github.com/VaishakhVipin)**

Transform your LLM applications with intelligent context management. 🧠✨ 