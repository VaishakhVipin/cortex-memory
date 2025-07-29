#!/usr/bin/env python3
"""
🧠 Cortex Advanced Integration Example
Shows how to wrap existing AI systems with Cortex for automatic context injection.
"""

import time
from typing import Dict, Any, Optional
from cortex import CortexClient, generate_with_context, generate_with_evolving_context

class CortexWrapper:
    """
    Wrapper class to integrate Cortex with existing AI systems.
    """
    
    def __init__(self, api_key: Optional[str] = None, user_id: str = "default_user"):
        """
        Initialize Cortex wrapper.
        
        Args:
            api_key: Optional API key for production usage
            user_id: User identifier for context management
        """
        self.user_id = user_id
        self.api_key = api_key
        
        if api_key:
            # Production mode with API key
            self.client = CortexClient(api_key=api_key)
            self.mode = "production"
        else:
            # Development mode with direct functions
            self.client = None
            self.mode = "development"
    
    def enhance_ai_response(self, prompt: str, context_method: str = "semantic") -> str:
        """
        Enhance AI response with automatic context injection.
        
        Args:
            prompt: User's prompt
            context_method: "semantic" or "evolving"
            
        Returns:
            Enhanced response with injected context
        """
        try:
            if self.mode == "production":
                # Use API key client
                response = self.client.generate_with_context(
                    prompt=prompt,
                    context_method=context_method
                )
            else:
                # Use direct functions
                if context_method == "semantic":
                    response = generate_with_context(
                        user_id=self.user_id,
                        prompt=prompt
                    )
                else:
                    response = generate_with_evolving_context(
                        user_id=self.user_id,
                        prompt=prompt
                    )
            
            return response
            
        except Exception as e:
            print(f"⚠️ Cortex enhancement failed: {e}")
            # Return original prompt as fallback
            return f"Error: {e}"
    
    def store_interaction(self, prompt: str, response: str, 
                         metadata: Optional[Dict] = None) -> str:
        """
        Store AI interaction for future context.
        
        Args:
            prompt: User's prompt
            response: AI's response
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        try:
            if self.mode == "production":
                memory_id = self.client.store_conversation(
                    prompt=prompt,
                    response=response,
                    metadata=metadata
                )
            else:
                from cortex import store_conversation
                memory_id = store_conversation(
                    user_id=self.user_id,
                    prompt=prompt,
                    response=response,
                    metadata=metadata
                )
            
            return memory_id
            
        except Exception as e:
            print(f"⚠️ Failed to store interaction: {e}")
            return None
    
    def get_context_suggestions(self, prompt: str, limit: int = 3) -> list:
        """
        Get context suggestions for a prompt.
        
        Args:
            prompt: Search prompt
            limit: Number of suggestions
            
        Returns:
            List of context suggestions
        """
        try:
            if self.mode == "production":
                results = self.client.find_semantic_context(
                    prompt=prompt,
                    limit=limit
                )
            else:
                from cortex import semantic_embeddings
                similar_contexts = semantic_embeddings.find_semantically_similar_context(
                    user_id=self.user_id,
                    current_prompt=prompt,
                    limit=limit
                )
                results = []
                for context, score in similar_contexts:
                    results.append({
                        'memory_id': context.get('embedding_id'),
                        'prompt': context.get('prompt'),
                        'response': context.get('response'),
                        'similarity_score': score
                    })
            
            return results
            
        except Exception as e:
            print(f"⚠️ Failed to get context suggestions: {e}")
            return []

# Example 1: Wrap Existing AI Function
def your_existing_ai_function(prompt: str) -> str:
    """
    Your existing AI function (simulated).
    """
    # Simulate AI processing time
    time.sleep(0.1)
    return f"AI Response to: {prompt}"

# Enhanced version with Cortex
def enhanced_ai_function(prompt: str, user_id: str = "user123") -> str:
    """
    Enhanced AI function with automatic context injection.
    """
    # Initialize Cortex wrapper
    cortex = CortexWrapper(user_id=user_id)
    
    # Get enhanced response with context
    enhanced_response = cortex.enhance_ai_response(prompt, context_method="evolving")
    
    # Store the interaction for future context
    cortex.store_interaction(prompt, enhanced_response)
    
    return enhanced_response

# Example 2: Chatbot Integration
class CortexChatbot:
    """
    Chatbot with automatic context memory.
    """
    
    def __init__(self, api_key: Optional[str] = None, user_id: str = "user123"):
        self.cortex = CortexWrapper(api_key=api_key, user_id=user_id)
        self.conversation_history = []
    
    def chat(self, message: str) -> str:
        """
        Process chat message with context awareness.
        """
        # Get context suggestions
        context_suggestions = self.cortex.get_context_suggestions(message, limit=2)
        
        # Generate enhanced response
        response = self.cortex.enhance_ai_response(message, context_method="evolving")
        
        # Store interaction
        memory_id = self.cortex.store_interaction(message, response, {
            'context_suggestions_count': len(context_suggestions),
            'timestamp': time.time()
        })
        
        # Add to conversation history
        self.conversation_history.append({
            'message': message,
            'response': response,
            'memory_id': memory_id,
            'context_suggestions': context_suggestions
        })
        
        return response
    
    def get_conversation_summary(self) -> Dict:
        """
        Get conversation summary and analytics.
        """
        return {
            'total_messages': len(self.conversation_history),
            'conversation_history': self.conversation_history[-5:],  # Last 5 messages
            'user_id': self.cortex.user_id,
            'mode': self.cortex.mode
        }

# Example 3: API Integration
def create_cortex_middleware(api_key: str):
    """
    Create middleware for API integration.
    """
    def cortex_middleware(func):
        def wrapper(*args, **kwargs):
            # Extract prompt from function arguments
            prompt = kwargs.get('prompt') or args[0] if args else None
            
            if prompt:
                # Initialize Cortex
                cortex = CortexWrapper(api_key=api_key)
                
                # Enhance with context
                enhanced_prompt = cortex.enhance_ai_response(prompt)
                
                # Update arguments with enhanced prompt
                if 'prompt' in kwargs:
                    kwargs['prompt'] = enhanced_prompt
                elif args:
                    args = (enhanced_prompt,) + args[1:]
                
                # Call original function
                result = func(*args, **kwargs)
                
                # Store interaction
                cortex.store_interaction(prompt, str(result))
                
                return result
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return cortex_middleware

# Example 4: Usage Examples
if __name__ == "__main__":
    print("🧠 Cortex Advanced Integration Examples")
    print("=" * 50)
    
    # Example 1: Enhanced AI Function
    print("\n1. Enhanced AI Function:")
    user_prompt = "How do I implement secure authentication?"
    
    # Original AI function
    original_response = your_existing_ai_function(user_prompt)
    print(f"Original: {original_response}")
    
    # Enhanced AI function
    enhanced_response = enhanced_ai_function(user_prompt, "user123")
    print(f"Enhanced: {enhanced_response[:100]}...")
    
    # Example 2: Chatbot
    print("\n2. Cortex Chatbot:")
    chatbot = CortexChatbot(user_id="user123")
    
    messages = [
        "How do I implement authentication?",
        "What about JWT tokens?",
        "How do I secure the tokens?"
    ]
    
    for message in messages:
        response = chatbot.chat(message)
        print(f"User: {message}")
        print(f"Bot: {response[:80]}...")
        print()
    
    # Get conversation summary
    summary = chatbot.get_conversation_summary()
    print(f"Conversation Summary: {summary['total_messages']} messages")
    
    # Example 3: Middleware Usage
    print("\n3. Middleware Integration:")
    
    @create_cortex_middleware("your_api_key_here")
    def your_api_function(prompt: str) -> str:
        return f"API Response: {prompt}"
    
    # This will automatically enhance the prompt with context
    api_response = your_api_function("How do I implement authentication?")
    print(f"API Response: {api_response[:100]}...")
    
    print("\n✅ All examples completed!")