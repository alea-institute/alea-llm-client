"""OpenAI model constants and parameter definitions.

This module provides dynamically generated model lists from external configuration,
with fallback to static definitions for reliability.
"""

import logging
from typing import Any

from ..config.loader import get_global_loader, get_models_with_capability, supports_feature_dynamic

logger = logging.getLogger(__name__)


# Dynamic model list generation
def _get_dynamic_models() -> dict[str, list[str]]:
    """Generate model lists dynamically from configuration."""
    try:
        return get_global_loader().get_openai_model_groups()
    except Exception as e:
        logger.warning(f"Failed to load dynamic model configuration: {e}")
        return {}


def _get_fallback_models() -> dict[str, list[str]]:
    """Fallback static model definitions."""
    return {
        "gpt_5": [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-chat-latest",
        ],
        "o_series": [
            "o1",
            "o1-mini",
            "o1-pro",
            "o3",
            "o3-mini",
            "o3-pro",
            "o4-mini",
            "o3-deep-research",
            "o3-deep-research-2025-06-26",
            "o4-mini-deep-research",
            "o4-mini-deep-research-2025-06-26",
        ],
        "gpt_4o": [
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "audio": [
            "gpt-4o-audio-preview",
            "gpt-4o-mini-audio-preview",
        ],
        "search": [
            "gpt-4o-search-preview",
            "gpt-4o-mini-search-preview",
        ],
        "computer_use": [
            "computer-use-preview",
            "computer-use-preview-2025-03-11",
        ],
        "realtime": [
            "gpt-4o-realtime-preview",
            "gpt-4o-mini-realtime-preview",
        ],
        "gpt_4_1": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
        ],
    }


# Generate model lists with fallback
_dynamic_models = _get_dynamic_models()
_fallback_models = _get_fallback_models()


# Use dynamic if available and populated, otherwise fallback
def _get_models(category: str) -> list[str]:
    """Get models for a category with dynamic/fallback logic."""
    if _dynamic_models and category in _dynamic_models and _dynamic_models[category]:  # Must have content!
        return _dynamic_models[category]
    return _fallback_models.get(category, [])


# Model family lists (dynamically generated with fallback)
GPT_5_MODELS = _get_models("gpt_5")
GPT_4_1_MODELS = _get_models("gpt_4_1")
O_SERIES_MODELS = _get_models("o_series")
AUDIO_MODELS = _get_models("audio")
SEARCH_MODELS = _get_models("search")
COMPUTER_USE_MODELS = _get_models("computer_use")
REALTIME_MODELS = _get_models("realtime")
GPT_4O_MODELS = _get_models("gpt_4o") + AUDIO_MODELS + SEARCH_MODELS

# Parameter Values
REASONING_EFFORT_VALUES = ["minimal", "low", "medium", "high"]
VERBOSITY_VALUES = ["low", "medium", "high"]
MODALITY_VALUES = ["text", "audio"]


# Dynamic support model lists
def _get_dynamic_support_models() -> dict[str, list[str]]:
    """Get models that support specific features dynamically."""
    try:
        return {
            "reasoning_effort": get_models_with_capability("reasoning_effort"),
            "grammar": get_models_with_capability("grammar"),
            "audio": get_models_with_capability("audio_input"),
            "realtime": get_models_with_capability("realtime"),
            "computer_use": get_models_with_capability("computer_use"),
        }
    except Exception as e:
        logger.warning(f"Failed to load dynamic support models: {e}")
        return {}


_dynamic_support = _get_dynamic_support_models()

# Models that support reasoning effort parameter
REASONING_EFFORT_SUPPORTED_MODELS = _dynamic_support.get("reasoning_effort", O_SERIES_MODELS)

# Models that support verbosity parameter (GPT-5 family)
VERBOSITY_SUPPORTED_MODELS = GPT_5_MODELS + GPT_4_1_MODELS + GPT_4O_MODELS

# Models that support max_completion_tokens (o-series)
MAX_COMPLETION_TOKENS_MODELS = O_SERIES_MODELS

# Models that support audio modality
AUDIO_SUPPORTED_MODELS = _dynamic_support.get("audio", AUDIO_MODELS)

# Models that support realtime streaming
REALTIME_SUPPORTED_MODELS = _dynamic_support.get("realtime", REALTIME_MODELS)

# Models that support computer use
COMPUTER_USE_SUPPORTED_MODELS = _dynamic_support.get("computer_use", COMPUTER_USE_MODELS)

# Models that support grammar constraints (context-free grammars)
GRAMMAR_SUPPORTED_MODELS = _dynamic_support.get("grammar", GPT_5_MODELS)

# Optimization Presets
SPEED_OPTIMIZED: dict[str, Any] = {"reasoning_effort": "minimal", "verbosity": "low"}
BALANCED: dict[str, Any] = {"reasoning_effort": "medium", "verbosity": "medium"}
THOROUGH: dict[str, Any] = {"reasoning_effort": "high", "verbosity": "high"}


# Model categorization for parameter support (with dynamic fallback)
def supports_reasoning_effort(model: str) -> bool:
    """Check if model supports reasoning_effort parameter."""
    try:
        return supports_feature_dynamic(model, "reasoning_effort")
    except Exception:
        return model in REASONING_EFFORT_SUPPORTED_MODELS


def supports_verbosity(model: str) -> bool:
    """Check if model supports verbosity parameter."""
    # Verbosity is a GPT-5+ feature, use family-based logic
    return any(model.startswith(prefix) for prefix in ["gpt-5", "gpt-4.1", "gpt-4o"])


def supports_audio(model: str) -> bool:
    """Check if model supports audio modality."""
    try:
        return supports_feature_dynamic(model, "audio_input")
    except Exception:
        return model in AUDIO_SUPPORTED_MODELS


def requires_max_completion_tokens(model: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens."""
    try:
        return supports_feature_dynamic(model, "max_completion_tokens")
    except Exception:
        return model in MAX_COMPLETION_TOKENS_MODELS


def supports_grammar(model: str) -> bool:
    """Check if model supports grammar constraints (context-free grammars)."""
    try:
        return supports_feature_dynamic(model, "grammar")
    except Exception:
        return model in GRAMMAR_SUPPORTED_MODELS


def supports_realtime(model: str) -> bool:
    """Check if model supports realtime streaming."""
    try:
        return supports_feature_dynamic(model, "realtime")
    except Exception:
        return model in REALTIME_SUPPORTED_MODELS


def supports_computer_use(model: str) -> bool:
    """Check if model supports computer use capabilities."""
    try:
        return supports_feature_dynamic(model, "computer_use")
    except Exception:
        return model in COMPUTER_USE_SUPPORTED_MODELS


# Dynamic default models with fallback
def _get_dynamic_defaults() -> dict[str, str]:
    """Get default models dynamically."""
    try:
        loader = get_global_loader()
        providers = loader.get_providers()

        if "openai" in providers:
            defaults = providers["openai"].defaults
            return {
                "chat": defaults.get("chat", "gpt-5-chat-latest"),
                "reasoning": defaults.get("reasoning", "o3-mini"),
                "fast": defaults.get("fast", "gpt-5-nano"),
            }
    except Exception as e:
        logger.warning(f"Failed to load dynamic defaults: {e}")

    # Fallback to static defaults
    return {
        "chat": "gpt-5-chat-latest",
        "reasoning": "o3-mini",
        "fast": "gpt-5-nano",
    }


_defaults = _get_dynamic_defaults()

# Default models for different use cases
DEFAULT_CHAT_MODEL = _defaults["chat"]
DEFAULT_REASONING_MODEL = _defaults["reasoning"]
DEFAULT_FAST_MODEL = _defaults["fast"]
DEFAULT_AUDIO_MODEL = "gpt-4o-audio-preview"  # Static fallback as not in config
