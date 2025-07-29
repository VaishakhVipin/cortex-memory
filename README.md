# 🧠 Cortex - The Smart Context Layer for Prompt Chains in LLMs

## Overview

Cortex is an advanced semantic context management system that provides intelligent, persistent memory for LLM conversations. It transforms static AI interactions into context-aware, memory-enhanced experiences.

## Key Features

- **Semantic Context Retrieval**: Find relevant conversation history using advanced embeddings
- **Self-Evolving Context Model**: Learns which memories matter most over time
- **Adaptive Context Scoring**: Learns which memories matter most over time
- **Semantic Drift Detection**: Monitors system performance and detects behavioral changes
- **Auto-Pruning**: Removes low-impact memories to reduce memory bloat
- **Advanced Pattern Recognition**: Sophisticated statistical algorithms for context matching

## Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- Google Gemini API key (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/VaishakhVipin/cortex.git
cd cortex

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Basic Usage

```python
from semantic_embeddings import semantic_embeddings
from self_evolving_context import self_evolving_context

# Store a conversation
memory_id = semantic_embeddings.store_conversation_embedding(
    user_id="user123",
    prompt="How do I implement authentication?",
    response="Use JWT tokens with proper validation...",
    metadata={"quality": 0.9}
)

# Find relevant context
similar_contexts = semantic_embeddings.find_semantically_similar_context(
    user_id="user123",
    current_prompt="What's the best way to secure my API?",
    limit=3
)

# Use evolving context for better results
evolving_contexts = self_evolving_context.find_evolving_context(
    user_id="user123",
    current_prompt="What's the best way to secure my API?",
    limit=3
)

print(f"Found {len(similar_contexts)} relevant memories")
```

### API Usage

```python
from context_manager import generate_with_context

# Generate response with context
response = generate_with_context(
    user_id="user123",
    prompt="How do I implement secure authentication?",
    context_method="semantic"  # or "evolving"
)

print(response)
```

## Architecture

### Core Components

1. **Semantic Embeddings System**
   - Generates and stores conversation embeddings
   - Provides semantic similarity search
   - Handles batch processing for efficiency

2. **Self-Evolving Context Model**
   - Adaptive learning system
   - Performance tracking and optimization
   - Auto-pruning of low-impact memories

3. **Semantic Drift Detection**
   - Monitors system performance changes
   - Detects behavioral drift
   - Provides analytics and insights

### Data Flow

```
User Query → Semantic Search → Context Retrieval → Response Generation
                ↓
            Memory Storage ← Learning & Optimization ← Performance Tracking
```

## Advanced Features

### Self-Evolving Context

The system learns from user interactions to improve context relevance:

```python
# Track context effectiveness
self_evolving_context.track_context_effectiveness(
    user_id="user123",
    memory_ids=[memory_id],
    response_quality=0.9,
    user_feedback="very helpful"
)
```

### Auto-Pruning

Automatically removes low-impact memories:

```python
# Prune low-impact memories
pruning_stats = self_evolving_context.auto_pruning.prune_low_impact_memories(
    user_id="user123",
    threshold=0.3
)

print(f"Pruned {pruning_stats['pruned_memories']} low-impact memories")
```

### Drift Detection

Monitor system performance over time:

```python
from semantic_drift_detection import detect_semantic_drift

drift_results = detect_semantic_drift(
    user_id="user123",
    time_window_hours=24
)

print(f"Drift detected: {drift_results['drift_detected']}")
```

## Performance

### Benchmarks

| Method | Avg Time | Context Found | Status |
|--------|----------|---------------|---------|
| **Semantic** | 1.092s | 100% | ✅ **OPTIMIZED** |
| **Evolving** | 4.492s | 100% | ✅ **WORKING** |
| **Keyword** | 1.206s | 100% | ✅ **FAST** |

### Optimization Features

- **Batch Processing**: 1.4s per conversation
- **Redis Pipelining**: Optimized data retrieval
- **Memory Management**: Intelligent pruning of low-impact memories
- **Statistical Algorithms**: No ML dependencies for reliability

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Gemini API (optional)
GEMINI_API_KEY=your_api_key_here

# System Configuration
MAX_EMBEDDINGS_PER_USER=1000
SIMILARITY_THRESHOLD=0.3
AUTO_PRUNING_ENABLED=true
```

### Advanced Configuration

```python
# Custom similarity threshold
similar_contexts = semantic_embeddings.find_semantically_similar_context(
    user_id="user123",
    current_prompt="Your query",
    similarity_threshold=0.5  # Higher threshold for more precise matches
)

# Custom pruning threshold
pruning_stats = self_evolving_context.auto_pruning.prune_low_impact_memories(
    user_id="user123",
    threshold=0.2  # Lower threshold for more aggressive pruning
)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_context_benchmark.py
python test_full_system.py
python test_phase_2_features.py

# Run specific tests
python test_semantic_drift_detection.py
python test_hybrid_approach.py
```

## API Reference

### Core Functions

- `semantic_embeddings.store_conversation_embedding()` - Store conversation
- `semantic_embeddings.find_semantically_similar_context()` - Find similar context
- `self_evolving_context.find_evolving_context()` - Find evolving context
- `self_evolving_context.track_context_effectiveness()` - Track effectiveness
- `semantic_drift_detection.detect_semantic_drift()` - Detect drift

### Advanced Functions

- `self_evolving_context.auto_pruning.prune_low_impact_memories()` - Auto-prune
- `self_evolving_context.get_performance_metrics()` - Get metrics
- `semantic_embeddings.store_conversations_batch()` - Batch storage

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/VaishakhVipin/cortex.git
cd cortex
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linting
python -m flake8 .
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [Wiki](https://github.com/VaishakhVipin/cortex/wiki)
- **Issues**: [GitHub Issues](https://github.com/VaishakhVipin/cortex/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VaishakhVipin/cortex/discussions)

## Roadmap

- [ ] Multi-modal support (images, documents)
- [ ] Real-time learning improvements
- [ ] Advanced analytics dashboard
- [ ] Python and Node.js SDKs
- [ ] Enterprise deployment tools

---

**Cortex: The first truly self-optimizing semantic context system for LLMs.**
