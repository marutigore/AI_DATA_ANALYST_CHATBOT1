"""
Multi-Provider LLM Factory Engine.
Provides unified instantiation of LLM models with multi-provider fallback support
(Google Gemini -> OpenAI -> Anthropic) to handle rate limits (HTTP 429) or service degradation.
"""
import os
import logging
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

logger = logging.getLogger(__name__)

def get_llm(
    model_name: Optional[str] = None, 
    temperature: float = 0.0
) -> Any:
    """
    Instantiates a primary LLM with configured fallbacks.
    
    Args:
        model_name (str, optional): Target model name. Defaults to gemini-2.5-flash-lite.
        temperature (float): Model sampling temperature.
        
    Returns:
        Any: BaseChatModel instance with .with_fallbacks() configured if secondary keys exist.
    """
    target_model = model_name or "gemini-2.5-flash-lite"
    google_api_key = os.getenv("GOOGLE_API_KEY") or "dummy_key_for_testing"
    
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY is not set in environment. Using fallback configuration.")
        
    primary_llm = ChatGoogleGenerativeAI(
        model=target_model,
        temperature=temperature,
        google_api_key=google_api_key
    )

    
    fallbacks = []
    
    # Secondary Fallback 1: Alternative Gemini model (2.5-flash)
    try:
        alt_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            google_api_key=google_api_key
        )
        fallbacks.append(alt_gemini)
    except Exception as e:
        logger.debug(f"Could not build alt Gemini fallback: {e}")
        
    # Secondary Fallback 2: OpenAI GPT-3.5/GPT-4o if OPENAI_API_KEY present
    openai_key = os.getenv("OPENAI_API_KEY")
    if ChatOpenAI is not None and openai_key:
        try:
            openai_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=temperature,
                api_key=openai_key
            )
            fallbacks.append(openai_llm)
            logger.info("Configured OpenAI as secondary LLM fallback provider.")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI fallback: {e}")
            
    if fallbacks:
        logger.info(f"Configured LLM with {len(fallbacks)} fallback providers.")
        return primary_llm.with_fallbacks(fallbacks)
        
    return primary_llm
