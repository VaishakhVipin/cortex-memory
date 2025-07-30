#!/usr/bin/env python3
"""
🧠 Cortex LLM Providers - Multi-Provider Integration
Supports Gemini, Claude, and OpenAI with unified interface.
"""

import os
import json
import time
import requests
from typing import Dict, Optional, List, Any
from enum import Enum
from .config import GEMINI_API_KEY

class LLMProvider(Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"

class LLMConfig:
    """Configuration for LLM providers."""
    
    def __init__(self):
        # API Keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
        self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Default models
        self.gemini_model = "gemini-2.0-flash"
        self.claude_model = "claude-3-5-sonnet-20241022"
        self.openai_model = "gpt-4o-mini"
        
        # Rate limits and timeouts
        self.timeout = 30
        self.max_retries = 3
        self.rate_limit_delay = 1.0

class BaseLLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = "base"
    
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Generate response from LLM."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return False

class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.provider_name = "gemini"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return bool(self.config.gemini_api_key)
    
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Generate response using Gemini API."""
        if not self.is_available():
            raise ValueError("Gemini API key not configured")
        
        url = f"{self.base_url}/models/{self.config.gemini_model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.config.gemini_api_key}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 0.9),
                "topK": kwargs.get("top_k", 40)
            }
        }
        
        try:
            resp = requests.post(
                url, 
                headers=headers, 
                params=params, 
                json=payload,
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Extract response text
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            
            return "No response generated"
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Gemini API error: {e}")
            return f"Error calling Gemini API: {e}"

class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.provider_name = "claude"
        self.base_url = "https://api.anthropic.com/v1"
    
    def is_available(self) -> bool:
        """Check if Claude is available."""
        return bool(self.config.claude_api_key)
    
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Generate response using Claude API."""
        if not self.is_available():
            raise ValueError("Claude API key not configured")
        
        url = f"{self.base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.claude_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config.claude_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        try:
            resp = requests.post(
                url, 
                headers=headers, 
                json=payload,
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Extract response text
            content = data.get("content", [])
            if content:
                return content[0].get("text", "")
            
            return "No response generated"
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Claude API error: {e}")
            return f"Error calling Claude API: {e}"

class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.provider_name = "openai"
        self.base_url = "https://api.openai.com/v1"
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return bool(self.config.openai_api_key)
    
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Generate response using OpenAI API."""
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.openai_api_key}"
        }
        
        payload = {
            "model": self.config.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        try:
            resp = requests.post(
                url, 
                headers=headers, 
                json=payload,
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Extract response text
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            
            return "No response generated"
            
        except requests.exceptions.RequestException as e:
            print(f"❌ OpenAI API error: {e}")
            return f"Error calling OpenAI API: {e}"

class LLMManager:
    """Manager for multiple LLM providers."""
    
    def __init__(self):
        self.config = LLMConfig()
        self.providers = {
            LLMProvider.GEMINI: GeminiProvider(self.config),
            LLMProvider.CLAUDE: ClaudeProvider(self.config),
            LLMProvider.OPENAI: OpenAIProvider(self.config)
        }
        self.default_provider = LLMProvider.GEMINI
        self.fallback_providers = [LLMProvider.CLAUDE, LLMProvider.OPENAI]
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers."""
        return [provider for provider in LLMProvider 
                if self.providers[provider].is_available()]
    
    def generate(self, prompt: str, provider: Optional[LLMProvider] = None, 
                max_tokens: int = 512, use_fallback: bool = True, **kwargs) -> str:
        """
        Generate response using specified or default provider.
        
        Args:
            prompt: Input prompt
            provider: Specific provider to use
            max_tokens: Maximum output tokens
            use_fallback: Whether to try fallback providers if primary fails
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated response text
        """
        # Determine provider to use
        if provider is None:
            provider = self.default_provider
        
        # Try primary provider
        if self.providers[provider].is_available():
            try:
                response = self.providers[provider].generate(
                    prompt, max_tokens, **kwargs
                )
                if not response.startswith("Error calling"):
                    return response
            except Exception as e:
                print(f"⚠️ {provider.value} failed: {e}")
        
        # Try fallback providers if enabled
        if use_fallback:
            for fallback_provider in self.fallback_providers:
                if (fallback_provider != provider and 
                    self.providers[fallback_provider].is_available()):
                    try:
                        response = self.providers[fallback_provider].generate(
                            prompt, max_tokens, **kwargs
                        )
                        if not response.startswith("Error calling"):
                            print(f"🔄 Using fallback provider: {fallback_provider.value}")
                            return response
                    except Exception as e:
                        print(f"⚠️ {fallback_provider.value} fallback failed: {e}")
        
        # All providers failed
        return f"Error: All LLM providers failed. Available: {[p.value for p in self.get_available_providers()]}"
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        status = {}
        for provider in LLMProvider:
            provider_instance = self.providers[provider]
            status[provider.value] = {
                "available": provider_instance.is_available(),
                "model": getattr(self.config, f"{provider.value}_model", "unknown"),
                "has_api_key": bool(getattr(self.config, f"{provider.value}_api_key", None))
            }
        return status

# Global instance
llm_manager = LLMManager()

# Convenience function for backward compatibility
def call_gemini_api(prompt_text: str, max_tokens: int = 512) -> str:
    """Backward compatibility function for Gemini API calls."""
    return llm_manager.generate(prompt_text, LLMProvider.GEMINI, max_tokens)

def call_llm_api(prompt_text: str, provider: str = "auto", max_tokens: int = 512, **kwargs) -> str:
    """
    Unified LLM API call function.
    
    Args:
        prompt_text: Input prompt
        provider: Provider to use ("auto", "gemini", "claude", "openai")
        max_tokens: Maximum output tokens
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Generated response text
    """
    if provider == "auto":
        return llm_manager.generate(prompt_text, max_tokens=max_tokens, **kwargs)
    else:
        try:
            provider_enum = LLMProvider(provider.lower())
            return llm_manager.generate(prompt_text, provider_enum, max_tokens, **kwargs)
        except ValueError:
            return f"Error: Unknown provider '{provider}'. Available: {[p.value for p in LLMProvider]}" 