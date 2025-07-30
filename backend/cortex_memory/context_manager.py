#!/usr/bin/env python3
"""
🧠 Cortex Context Manager
Manages context fetching and response generation with multiple strategies.
"""

import time
from typing import List, Dict, Tuple, Optional, Any
from .core import store_conversation
from .semantic_embeddings import semantic_embeddings
from .self_evolving_context import self_evolving_context
from .llm_providers import call_llm_api, llm_manager, LLMProvider

def generate_with_context(user_id: str, prompt: str, provider: str = "auto") -> str:
    """
    Generate response with semantic context injection.
    
    Args:
        user_id: User identifier
        prompt: User's prompt
        provider: LLM provider to use ("auto", "gemini", "claude", "openai")
        
    Returns:
        Generated response with semantic context
    """
    try:
        # Find semantically similar context
        similar_contexts = semantic_embeddings.find_semantically_similar_context(
            user_id=user_id,
            current_prompt=prompt,
            limit=3,
            similarity_threshold=0.3
        )
        
        # Build context string
        context_str = ""
        if similar_contexts:
            context_str = "Based on our previous conversations:\n\n"
            for i, (context, score) in enumerate(similar_contexts, 1):
                context_str += f"{i}. Q: {context.get('prompt', 'N/A')}\n"
                context_str += f"   A: {context.get('response', 'N/A')}\n\n"
        
        # Generate enhanced prompt
        enhanced_prompt = f"""
{context_str}
Current Question: {prompt}

Please provide a comprehensive answer that builds upon our previous discussions if relevant.
"""
        
        # Generate response
        response = call_llm_api(enhanced_prompt.strip(), provider=provider)
        
        # Store the conversation
        store_conversation(
            user_id=user_id,
            prompt=prompt,
            response=response,
            metadata={"context_method": "semantic", "contexts_found": len(similar_contexts), "provider": provider}
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Error in semantic context generation: {e}")
        # Fallback to direct response
        return call_llm_api(prompt, provider=provider)

def generate_with_evolving_context(user_id: str, prompt: str, provider: str = "auto") -> str:
    """
    Generate response with self-evolving context injection.
    
    Args:
        user_id: User identifier
        prompt: User's prompt
        provider: LLM provider to use ("auto", "gemini", "claude", "openai")
        
    Returns:
        Generated response with evolving context
    """
    try:
        # Find evolving context
        evolving_contexts = self_evolving_context.find_evolving_context(
            user_id=user_id,
            current_prompt=prompt,
            limit=3,
            similarity_threshold=0.3
        )
        
        # Build context string
        context_str = ""
        if evolving_contexts:
            context_str = "Based on our learning from previous interactions:\n\n"
            for i, (context, score) in enumerate(evolving_contexts, 1):
                context_str += f"{i}. Q: {context.get('prompt', 'N/A')}\n"
                context_str += f"   A: {context.get('response', 'N/A')}\n"
                context_str += f"   Relevance Score: {score:.3f}\n\n"
        
        # Generate enhanced prompt
        enhanced_prompt = f"""
{context_str}
Current Question: {prompt}

Please provide an answer that leverages our learning from previous interactions and adapts to your evolving needs.
"""
        
        # Generate response
        response = call_llm_api(enhanced_prompt.strip(), provider=provider)
        
        # Store the conversation
        store_conversation(
            user_id=user_id,
            prompt=prompt,
            response=response,
            metadata={"context_method": "evolving", "contexts_found": len(evolving_contexts), "provider": provider}
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Error in evolving context generation: {e}")
        # Fallback to direct response
        return call_llm_api(prompt, provider=provider)

def generate_with_hybrid_context(user_id: str, prompt: str, 
                               semantic_weight: float = 0.6, 
                               evolving_weight: float = 0.4,
                               provider: str = "auto") -> str:
    """
    Generate response with hybrid context injection combining semantic and evolving methods.
    
    Args:
        user_id: User identifier
        prompt: User's prompt
        semantic_weight: Weight for semantic context (default: 0.6)
        evolving_weight: Weight for evolving context (default: 0.4)
        provider: LLM provider to use ("auto", "gemini", "claude", "openai")
        
    Returns:
        Generated response with hybrid context
    """
    try:
        # Normalize weights
        total_weight = semantic_weight + evolving_weight
        semantic_weight /= total_weight
        evolving_weight /= total_weight
        
        # Get semantic contexts
        semantic_contexts = semantic_embeddings.find_semantically_similar_context(
            user_id=user_id,
            current_prompt=prompt,
            limit=5,
            similarity_threshold=0.2
        )
        
        # Get evolving contexts
        evolving_contexts = self_evolving_context.find_evolving_context(
            user_id=user_id,
            current_prompt=prompt,
            limit=5,
            similarity_threshold=0.2
        )
        
        # Combine and score contexts
        combined_contexts = {}
        
        for context, score in semantic_contexts:
            mem_id = context.get('embedding_id')
            if mem_id:
                combined_contexts[mem_id] = {
                    'context': context,
                    'semantic_score': score,
                    'evolving_score': 0.0
                }
        
        for context, score in evolving_contexts:
            mem_id = context.get('embedding_id')
            if mem_id:
                if mem_id in combined_contexts:
                    combined_contexts[mem_id]['evolving_score'] = score
                else:
                    combined_contexts[mem_id] = {
                        'context': context,
                        'semantic_score': 0.0,
                        'evolving_score': score
                    }
        
        # Calculate final weighted score
        scored_contexts = []
        for mem_id, data in combined_contexts.items():
            final_score = (data['semantic_score'] * semantic_weight) + \
                          (data['evolving_score'] * evolving_weight)
            scored_contexts.append((data['context'], final_score))
        
        # Sort by final score and select top N
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        top_contexts = scored_contexts[:3]
        
        # Build context string
        context_str = ""
        if top_contexts:
            context_str = "Based on a hybrid analysis of our past interactions and evolving understanding:\n\n"
            for i, (context, score) in enumerate(top_contexts, 1):
                context_str += f"{i}. Q: {context.get('prompt', 'N/A')}\n"
                context_str += f"   A: {context.get('response', 'N/A')}\n"
                context_str += f"   Hybrid Score: {score:.3f}\n\n"
        
        # Generate enhanced prompt
        enhanced_prompt = f"""
{context_str}
Current Question: {prompt}

Please provide a comprehensive answer that combines:
- Semantic similarity from our previous discussions
- Learning patterns from our evolving interactions
- The most relevant context from both approaches
"""
        
        # Generate response
        response = call_llm_api(enhanced_prompt.strip(), provider=provider)
        
        # Store the conversation with hybrid metadata
        store_conversation(
            user_id=user_id,
            prompt=prompt,
            response=response,
            metadata={
                "context_method": "hybrid",
                "semantic_contexts": len(semantic_contexts),
                "evolving_contexts": len(evolving_contexts),
                "combined_contexts": len(top_contexts),
                "semantic_weight": semantic_weight,
                "evolving_weight": evolving_weight,
                "provider": provider,
                "top_context_methods": [ctx['method'] for ctx in top_contexts]
            }
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Error in hybrid context generation: {e}")
        # Fallback to semantic context
        return generate_with_context(user_id, prompt, provider=provider)

def generate_with_adaptive_context(user_id: str, prompt: str, provider: str = "auto") -> str:
    """
    Generate response with adaptive context selection based on query characteristics.
    
    Args:
        user_id: User identifier
        prompt: User's prompt
        provider: LLM provider to use ("auto", "gemini", "claude", "openai")
        
    Returns:
        Generated response with adaptive context
    """
    try:
        # Analyze query characteristics
        query_analysis = _analyze_query_characteristics(prompt)
        
        # Choose context method based on analysis
        if query_analysis['complexity'] == 'high':
            # Complex queries benefit from hybrid approach
            return generate_with_hybrid_context(user_id, prompt, 0.5, 0.5, provider)
        elif query_analysis['type'] == 'followup':
            # Follow-up questions benefit from evolving context
            return generate_with_evolving_context(user_id, prompt, provider)
        elif query_analysis['type'] == 'new_topic':
            # New topics benefit from semantic similarity
            return generate_with_context(user_id, prompt, provider)
        else:
            # Default to hybrid with semantic bias
            return generate_with_hybrid_context(user_id, prompt, 0.7, 0.3, provider)
            
    except Exception as e:
        print(f"❌ Error in adaptive context generation: {e}")
        # Fallback to semantic context
        return generate_with_context(user_id, prompt, provider=provider)

def _analyze_query_characteristics(prompt: str) -> Dict[str, Any]:
    """
    Analyze query characteristics to determine optimal context method.
    
    Args:
        prompt: User's prompt
        
    Returns:
        Query analysis results
    """
    analysis = {
        'complexity': 'medium',
        'type': 'general',
        'word_count': len(prompt.split()),
        'has_technical_terms': False,
        'is_question': prompt.strip().endswith('?'),
        'followup_indicators': []
    }
    
    # Check complexity
    word_count = analysis['word_count']
    if word_count < 10:
        analysis['complexity'] = 'low'
    elif word_count > 30:
        analysis['complexity'] = 'high'
    
    # Check for technical terms
    technical_terms = ['api', 'authentication', 'database', 'deployment', 'algorithm', 
                      'framework', 'library', 'protocol', 'architecture', 'optimization']
    prompt_lower = prompt.lower()
    analysis['has_technical_terms'] = any(term in prompt_lower for term in technical_terms)
    
    # Check for follow-up indicators
    followup_indicators = ['also', 'additionally', 'furthermore', 'moreover', 'besides',
                          'what about', 'how about', 'can you also', 'in addition']
    analysis['followup_indicators'] = [indicator for indicator in followup_indicators 
                                      if indicator in prompt_lower]
    
    # Determine query type
    if analysis['followup_indicators']:
        analysis['type'] = 'followup'
    elif analysis['has_technical_terms'] and analysis['complexity'] == 'high':
        analysis['type'] = 'technical'
    elif not analysis['is_question']:
        analysis['type'] = 'statement'
    else:
        analysis['type'] = 'new_topic'
    
    return analysis

def get_context_analytics(user_id: str) -> Dict[str, Any]:
    """
    Get analytics about context usage and effectiveness.
    
    Args:
        user_id: User identifier
        
    Returns:
        Context analytics
    """
    try:
        # Get semantic analytics
        semantic_metrics = semantic_embeddings.get_precision_recall_metrics(user_id)
        
        # Get evolving analytics
        evolving_metrics = self_evolving_context.get_performance_metrics(user_id)
        
        # Combine analytics
        analytics = {
            'semantic_context': semantic_metrics,
            'evolving_context': evolving_metrics,
            'hybrid_recommendation': _get_hybrid_recommendation(semantic_metrics, evolving_metrics)
        }
        
        return analytics
        
    except Exception as e:
        print(f"❌ Error getting context analytics: {e}")
        return {}

def _get_hybrid_recommendation(semantic_metrics: Dict, evolving_metrics: Dict) -> Dict[str, float]:
    """
    Get recommended weights for hybrid context based on performance metrics.
    
    Args:
        semantic_metrics: Semantic context performance metrics
        evolving_metrics: Evolving context performance metrics
        
    Returns:
        Recommended weights for hybrid approach
    """
    try:
        # Extract key metrics
        semantic_precision = semantic_metrics.get('precision', 0.5)
        semantic_recall = semantic_metrics.get('recall', 0.5)
        semantic_f1 = semantic_metrics.get('f1_score', 0.5)
        
        evolving_accuracy = evolving_metrics.get('context_accuracy', 0.5)
        evolving_adaptation = evolving_metrics.get('adaptation_rate', 0.5)
        
        # Calculate semantic score
        semantic_score = (semantic_precision + semantic_recall + semantic_f1) / 3
        
        # Calculate evolving score
        evolving_score = (evolving_accuracy + evolving_adaptation) / 2
        
        # Normalize scores
        total_score = semantic_score + evolving_score
        if total_score > 0:
            semantic_weight = semantic_score / total_score
            evolving_weight = evolving_score / total_score
        else:
            semantic_weight = 0.6
            evolving_weight = 0.4
        
        return {
            'semantic_weight': round(semantic_weight, 2),
            'evolving_weight': round(evolving_weight, 2),
            'semantic_score': round(semantic_score, 3),
            'evolving_score': round(evolving_score, 3)
        }
        
    except Exception as e:
        print(f"⚠️ Error calculating hybrid recommendation: {e}")
        return {'semantic_weight': 0.6, 'evolving_weight': 0.4} 