"""Test configuration and fixtures for ALEA LLM Client.

This conftest.py provides rate-limited model fixtures to prevent API rate limit errors
during test execution. Rate limits can be customized via environment variables:

Google API (most restrictive):
    GOOGLE_API_DELAY=2.0        # Seconds between calls (default: 2.0)
    GOOGLE_API_CONCURRENT=1     # Max concurrent calls (default: 1)

Anthropic API:
    ANTHROPIC_API_DELAY=0.5     # Seconds between calls (default: 0.5)
    ANTHROPIC_API_CONCURRENT=3  # Max concurrent calls (default: 3)

OpenAI API:
    OPENAI_API_DELAY=0.2        # Seconds between calls (default: 0.2)
    OPENAI_API_CONCURRENT=5     # Max concurrent calls (default: 5)

xAI/Grok API:
    XAI_API_DELAY=1.0           # Seconds between calls (default: 1.0)
    XAI_API_CONCURRENT=2        # Max concurrent calls (default: 2)

VLLM (local):
    VLLM_API_DELAY=0.1          # Seconds between calls (default: 0.1)
    VLLM_API_CONCURRENT=10      # Max concurrent calls (default: 10)

Example usage:
    export GOOGLE_API_DELAY=3.0  # Slower Google API calls
    pytest tests/test_google.py  # Will automatically use rate limiting
"""

import asyncio
import os
import time
from threading import Lock
from typing import Any

import pytest
from pydantic import BaseModel

from alea_llm_client import (
    AnthropicModel,
    GoogleModel,
    GrokModel,
    OpenAIModel,
    VLLMModel,
)


class PydanticTestModel(BaseModel):
    name: str
    age: int


# Rate limiting configuration
RATE_LIMITS = {
    "google": {
        "delay": float(os.environ.get("GOOGLE_API_DELAY", "2.0")),  # 2 seconds between Google API calls
        "concurrent": int(os.environ.get("GOOGLE_API_CONCURRENT", "1")),  # Max 1 concurrent Google call
    },
    "anthropic": {
        "delay": float(os.environ.get("ANTHROPIC_API_DELAY", "0.5")),  # 0.5 seconds between Anthropic calls
        "concurrent": int(os.environ.get("ANTHROPIC_API_CONCURRENT", "3")),  # Max 3 concurrent Anthropic calls
    },
    "openai": {
        "delay": float(os.environ.get("OPENAI_API_DELAY", "0.2")),  # 0.2 seconds between OpenAI calls
        "concurrent": int(os.environ.get("OPENAI_API_CONCURRENT", "5")),  # Max 5 concurrent OpenAI calls
    },
    "xai": {
        "delay": float(os.environ.get("XAI_API_DELAY", "1.0")),  # 1 second between xAI/Grok calls
        "concurrent": int(os.environ.get("XAI_API_CONCURRENT", "2")),  # Max 2 concurrent xAI calls
    },
    "vllm": {
        "delay": float(os.environ.get("VLLM_API_DELAY", "0.1")),  # Minimal delay for local VLLM
        "concurrent": int(os.environ.get("VLLM_API_CONCURRENT", "10")),  # Higher concurrency for local
    },
}

# Global rate limiter state
_rate_limit_state: dict[str, dict[str, Any]] = {}
_rate_limit_lock = Lock()


def _init_rate_limiter(provider: str) -> None:
    """Initialize rate limiter for a provider."""
    with _rate_limit_lock:
        if provider not in _rate_limit_state:
            config = RATE_LIMITS.get(provider, {"delay": 0.1, "concurrent": 5})
            _rate_limit_state[provider] = {
                "last_call": 0.0,
                "semaphore": asyncio.Semaphore(config["concurrent"]),
                "config": config,
            }


def _apply_rate_limit(provider: str) -> None:
    """Apply rate limiting before making an API call."""
    _init_rate_limiter(provider)

    with _rate_limit_lock:
        state = _rate_limit_state[provider]
        config = state["config"]

        # Calculate how long to wait since last call
        now = time.time()
        time_since_last = now - state["last_call"]
        delay_needed = config["delay"] - time_since_last

        if delay_needed > 0:
            time.sleep(delay_needed)

        # Update last call time
        state["last_call"] = time.time()


class RateLimitedModel:
    """Wrapper to add rate limiting to any model."""

    def __init__(self, model, provider: str):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_provider", provider)
        _init_rate_limiter(provider)

    def __getattr__(self, name):
        """Proxy all attributes to the wrapped model."""
        attr = getattr(self._model, name)

        # Only rate limit specific API-calling methods
        if callable(attr) and name in [
            "chat",
            "chat_async",
            "json",
            "json_async",
            "pydantic",
            "pydantic_async",
        ]:

            def rate_limited_method(*args, **kwargs):
                _apply_rate_limit(self._provider)
                return attr(*args, **kwargs)

            return rate_limited_method

        # For all other attributes/methods, return as-is
        return attr

    def __setattr__(self, name, value):
        """Forward attribute assignments to the wrapped model."""
        if name in ("_model", "_provider"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._model, name, value)

    def __delattr__(self, name):
        """Forward attribute deletions to the wrapped model."""
        if name in ("_model", "_provider"):
            object.__delattr__(self, name)
        else:
            delattr(self._model, name)

    def __getattribute__(self, name):
        """Handle direct attribute access more transparently."""
        # For our internal attributes, use object's __getattribute__
        if name in ("_model", "_provider"):
            return object.__getattribute__(self, name)

        # For everything else, try the model first, then fall back to our __getattr__
        try:
            model = object.__getattribute__(self, "_model")
            return getattr(model, name)
        except AttributeError:
            return object.__getattribute__(self, name)


@pytest.fixture
def anthropic_model():
    model = AnthropicModel()
    return RateLimitedModel(model, "anthropic")


@pytest.fixture
def openai_model():
    model = OpenAIModel()
    return RateLimitedModel(model, "openai")


@pytest.fixture
def google_model():
    # Check if we have Google AI Studio key (preferred) or Vertex AI setup
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    vertex_key = os.environ.get("VERTEX_API_KEY")
    vertex_project = os.environ.get("VERTEX_PROJECT_ID")
    vertex_location = os.environ.get("VERTEX_LOCATION_ID", "us-central1")

    if gemini_key:
        # Use Google AI Studio (Gemini API)
        model = GoogleModel(
            model="gemini-2.0-flash-exp",
            endpoint="https://generativelanguage.googleapis.com",
            api_key=gemini_key,
        )
    elif vertex_key and vertex_project:
        # Use Vertex AI
        model = GoogleModel(
            model="gemini-2.0-flash-exp",
            endpoint=f"https://{vertex_location}-aiplatform.googleapis.com",
            api_key=vertex_key,
            project_id=vertex_project,
            location_id=vertex_location,
        )
    else:
        # Default to Google AI Studio endpoint (tests might skip if no API key)
        model = GoogleModel(
            model="gemini-2.0-flash-exp",
            endpoint="https://generativelanguage.googleapis.com",
        )

    return RateLimitedModel(model, "google")


@pytest.fixture
def grok_model():
    model = GrokModel()
    return RateLimitedModel(model, "xai")


@pytest.fixture
def vllm_model():
    # VLLM typically runs locally, so minimal rate limiting
    model = VLLMModel()
    return RateLimitedModel(model, "vllm")
