#!/usr/bin/env python3
"""
🧠 Cortex Basic Usage Examples
Demonstrates both direct function usage and API key client usage.
"""

# Example 1: Direct Function Usage (Local/Development)
print("=== Example 1: Direct Function Usage ===")

from cortex import (
    store_conversation,
    generate_with_context,
    generate_with_evolving_context
)

# Store a conversation
memory_id = store_conversation(
    user_id="user123",
    prompt="How do I implement authentication?",
    response="Use JWT tokens with proper validation and secure storage.",
    metadata={"quality": 0.9}
)
print(f"✅ Stored conversation: {memory_id}")

# Generate with semantic context
response1 = generate_with_context(
    user_id="user123",
    prompt="What's the best way to secure my API?"
)
print(f"🤖 Semantic response: {response1[:100]}...")

# Generate with evolving context
response2 = generate_with_evolving_context(
    user_id="user123",
    prompt="How do I implement secure authentication?"
)
print(f"🧠 Evolving response: {response2[:100]}...")

print("\n" + "="*50 + "\n")

# Example 2: API Key Client Usage (Production/Pay-per-use)
print("=== Example 2: API Key Client Usage ===")

from cortex import CortexClient

# Initialize client with API key
# Users get this API key from your frontend
client = CortexClient(api_key="your_api_key_here")

# Store conversation with usage tracking
memory_id = client.store_conversation(
    prompt="How do I implement authentication?",
    response="Use JWT tokens with proper validation and secure storage.",
    metadata={"quality": 0.9}
)
print(f"✅ Stored conversation: {memory_id}")

# Find semantic context
semantic_results = client.find_semantic_context(
    prompt="What's the best way to secure my API?",
    limit=3
)
print(f"🔍 Found {len(semantic_results)} semantic contexts")

# Find evolving context
evolving_results = client.find_evolving_context(
    prompt="How do I implement secure authentication?",
    limit=3
)
print(f"🧠 Found {len(evolving_results)} evolving contexts")

# Generate with automatic context injection
response = client.generate_with_context(
    prompt="How do I implement secure authentication?",
    context_method="evolving"  # or "semantic"
)
print(f"🤖 Generated response: {response[:100]}...")

# Get analytics
analytics = client.get_analytics()
print(f"📊 Analytics: {analytics}")

# Get usage stats
usage_stats = client.get_usage_stats()
print(f"💰 Usage stats: {usage_stats}")

# Get plan info
plan_info = client.get_plan_info()
print(f"📋 Plan info: {plan_info}")

print("\n" + "="*50 + "\n")

# Example 3: Integration with Existing AI System
print("=== Example 3: Integration with Existing AI System ===")

def your_existing_ai_function(prompt: str, user_id: str):
    """
    Your existing AI function that you want to enhance with context.
    """
    # OLD: Direct AI call without context
    # response = call_ai_api(prompt)
    
    # NEW: Context-aware AI call
    response = generate_with_context(
        user_id=user_id,
        prompt=prompt
    )
    return response

# Usage
user_prompt = "How do I implement secure authentication?"
enhanced_response = your_existing_ai_function(user_prompt, "user123")
print(f"🎯 Enhanced response: {enhanced_response[:100]}...")

print("\n" + "="*50 + "\n")

# Example 4: Advanced Usage with Error Handling
print("=== Example 4: Advanced Usage with Error Handling ===")

try:
    client = CortexClient(api_key="your_api_key_here")
    
    # Try to generate response
    response = client.generate_with_context(
        prompt="How do I implement secure authentication?",
        context_method="evolving"
    )
    print(f"✅ Success: {response[:100]}...")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
    # Fallback to direct functions
    print("🔄 Falling back to direct functions...")
    response = generate_with_context("user123", "How do I implement secure authentication?")
    print(f"✅ Fallback response: {response[:100]}...")

print("\n" + "="*50 + "\n")

# Example 5: Batch Processing
print("=== Example 5: Batch Processing ===")

# Store multiple conversations
conversations = [
    ("How do I implement authentication?", "Use JWT tokens with proper validation."),
    ("What's the best database for my app?", "Choose based on your specific requirements."),
    ("How do I deploy to production?", "Use CI/CD pipelines for automated deployment.")
]

for prompt, response in conversations:
    memory_id = store_conversation(
        user_id="user123",
        prompt=prompt,
        response=response
    )
    print(f"📦 Stored: {prompt[:30]}... -> {memory_id}")

print("\n✅ All examples completed!")