import os
import requests
from config import GEMINI_API_KEY

def call_gemini_api(prompt_text: str, max_tokens: int = 512) -> str:
    """
    Call Gemini 2.0 Flash API with the given prompt.
    
    Args:
        prompt_text: The text prompt to send to Gemini
        max_tokens: Maximum output tokens (default: 512)
    
    Returns:
        The generated response text from Gemini
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }

    try:
        resp = requests.post(url, headers=headers, params=params, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except requests.exceptions.RequestException as e:
        print(f"❌ Gemini API error: {e}")
        return f"Error calling Gemini API: {e}" 