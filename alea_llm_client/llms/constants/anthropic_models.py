"""Anthropic model constants and parameter definitions.

This module provides dynamically generated model lists from external configuration,
with fallback to static definitions for reliability.
"""

import logging
from typing import Any

from ..config.loader import get_global_loader, get_models_with_capability, supports_feature_dynamic

logger = logging.getLogger(__name__)


# Dynamic model list generation
def _get_dynamic_anthropic_models() -> dict[str, list[str]]:
    """Generate Anthropic model lists dynamically from configuration."""
    try:
        return get_global_loader().get_anthropic_model_groups()
    except Exception as e:
        logger.warning(f"Failed to load dynamic Anthropic model configuration: {e}")
        return {}


def _get_fallback_anthropic_models() -> dict[str, list[str]]:
    """Fallback static Anthropic model definitions."""
    return {
        "claude_4": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-4-sonnet-20250514",
            "claude-opus-4-1-20250805",
        ],
        "claude_3_7": ["claude-3-7-sonnet-20250219", "claude-3-7-sonnet-latest"],
        "claude_3_5": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ],
        "claude_3": [
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
            "claude-3-opus-latest",
        ],
        "sonnet": [
            "claude-sonnet-4-20250514",
            "claude-4-sonnet-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-latest",
        ],
        "opus": [
            "claude-opus-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-3-opus-20240229",
            "claude-3-opus-latest",
        ],
        "haiku": [
            "claude-3-5-haiku-20241022",
            "claude-3-5-haiku-latest",
            "claude-3-haiku-20240307",
        ],
    }


# Generate model lists with fallback
_dynamic_anthropic_models = _get_dynamic_anthropic_models()
_fallback_anthropic_models = _get_fallback_anthropic_models()


# Use dynamic if available and populated, otherwise fallback
def _get_anthropic_models(category: str) -> list[str]:
    """Get Anthropic models for a category with dynamic/fallback logic."""
    if (
        _dynamic_anthropic_models and category in _dynamic_anthropic_models and _dynamic_anthropic_models[category]
    ):  # Must have content!
        return _dynamic_anthropic_models[category]
    return _fallback_anthropic_models.get(category, [])


# Model family lists (dynamically generated with fallback)
CLAUDE_4_MODELS = _get_anthropic_models("claude_4")
CLAUDE_3_7_MODELS = _get_anthropic_models("claude_3_7")
CLAUDE_3_5_MODELS = _get_anthropic_models("claude_3_5")
CLAUDE_3_MODELS = _get_anthropic_models("claude_3")

# All Claude Models
ALL_CLAUDE_MODELS = CLAUDE_4_MODELS + CLAUDE_3_7_MODELS + CLAUDE_3_5_MODELS + CLAUDE_3_MODELS

# Models with enhanced capabilities
SONNET_MODELS = _get_anthropic_models("sonnet")
OPUS_MODELS = _get_anthropic_models("opus")
HAIKU_MODELS = _get_anthropic_models("haiku")


# Dynamic thinking-enabled models
def _get_thinking_models() -> list[str]:
    """Get models that support thinking dynamically."""
    try:
        return get_models_with_capability("thinking")
    except Exception as e:
        logger.warning(f"Failed to load dynamic thinking models: {e}")
        return CLAUDE_4_MODELS + CLAUDE_3_7_MODELS


THINKING_ENABLED_MODELS = _get_thinking_models()

# Thinking Configuration
DEFAULT_THINKING_CONFIG = {
    "enabled": True,
    "budget_tokens": 1600,  # Minimum 1024, must be < max_tokens
}

THINKING_DISABLED_CONFIG = {"enabled": False}

# API Version that supports thinking
THINKING_API_VERSION = "2023-06-01"  # Update if newer version required


# Model categorization for features (with dynamic fallback)
def supports_thinking(model: str) -> bool:
    """Check if model supports thinking configuration."""
    try:
        return supports_feature_dynamic(model, "thinking")
    except Exception:
        return model in THINKING_ENABLED_MODELS


def is_claude_4(model: str) -> bool:
    """Check if model is Claude 4 family."""
    try:
        model_config = get_global_loader().get_model_config(model)
        return model_config is not None and model_config.family == "claude-4"
    except Exception:
        return model in CLAUDE_4_MODELS


def is_claude_3_7(model: str) -> bool:
    """Check if model is Claude 3.7 family."""
    try:
        model_config = get_global_loader().get_model_config(model)
        return model_config is not None and model_config.family == "claude-3.7"
    except Exception:
        return model in CLAUDE_3_7_MODELS


def get_model_family(model: str) -> str:
    """Get the model family name."""
    try:
        model_config = get_global_loader().get_model_config(model)
        if model_config:
            return model_config.family
    except Exception:
        pass

    # Fallback to static logic
    if model in CLAUDE_4_MODELS:
        return "claude-4"
    elif model in CLAUDE_3_7_MODELS:
        return "claude-3.7"
    elif model in CLAUDE_3_5_MODELS:
        return "claude-3.5"
    elif model in CLAUDE_3_MODELS:
        return "claude-3"
    else:
        return "unknown"


def get_model_size(model: str) -> str:
    """Get the model size (sonnet, opus, haiku)."""
    try:
        model_config = get_global_loader().get_model_config(model)
        if model_config:
            tier = model_config.tier
            # Map tiers to traditional Claude sizes
            if tier in ["small"]:
                return "haiku"
            elif tier in ["medium"]:
                return "sonnet"
            elif tier in ["large"]:
                return "opus"
    except Exception:
        pass

    # Fallback to name-based logic
    if "sonnet" in model:
        return "sonnet"
    elif "opus" in model:
        return "opus"
    elif "haiku" in model:
        return "haiku"
    else:
        return "unknown"


# Dynamic default models with fallback
def _get_dynamic_anthropic_defaults() -> dict[str, str]:
    """Get default Anthropic models dynamically."""
    try:
        loader = get_global_loader()
        providers = loader.get_providers()

        if "anthropic" in providers:
            defaults = providers["anthropic"].defaults
            return {
                "chat": defaults.get("chat", "claude-sonnet-4-20250514"),
                "reasoning": defaults.get("reasoning", "claude-opus-4-20250514"),
                "fast": defaults.get("fast", "claude-3-5-haiku-latest"),
                "thinking": defaults.get("thinking", "claude-sonnet-4-20250514"),
            }
    except Exception as e:
        logger.warning(f"Failed to load dynamic Anthropic defaults: {e}")

    # Fallback to static defaults
    return {
        "chat": "claude-sonnet-4-20250514",
        "reasoning": "claude-opus-4-20250514",
        "fast": "claude-3-5-haiku-latest",
        "thinking": "claude-sonnet-4-20250514",
    }


_anthropic_defaults = _get_dynamic_anthropic_defaults()

# Default models for different use cases
DEFAULT_CHAT_MODEL = _anthropic_defaults["chat"]
DEFAULT_REASONING_MODEL = _anthropic_defaults["reasoning"]
DEFAULT_FAST_MODEL = _anthropic_defaults["fast"]
DEFAULT_THINKING_MODEL = _anthropic_defaults["thinking"]


# Thinking configuration helpers
def create_thinking_config(enabled: bool = True, budget_tokens: int = 1600) -> dict[str, Any]:
    """Create a thinking configuration dictionary."""
    if not enabled:
        return {"enabled": False}

    if budget_tokens < 1024:
        raise ValueError("budget_tokens must be at least 1024")

    return {"enabled": True, "budget_tokens": budget_tokens}


def validate_thinking_config(thinking_config: dict[str, Any], max_tokens: int) -> bool:
    """Validate thinking configuration parameters."""
    if not isinstance(thinking_config, dict):
        return False

    enabled = thinking_config.get("enabled", False)
    if not enabled:
        return True

    budget_tokens = thinking_config.get("budget_tokens", 0)
    if budget_tokens < 1024:
        return False

    return not budget_tokens >= max_tokens
