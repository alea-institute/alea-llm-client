"""Anthropic model constants and parameter definitions."""

from typing import Dict, Any

# Claude 4 Model Family (Latest Generation)
CLAUDE_4_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    "claude-4-sonnet-20250514",
    "claude-opus-4-0",
    "claude-opus-4-20250514",
    "claude-4-opus-20250514",
    "claude-opus-4-1-20250805",
]

# Claude 3.7 Model Family (New)
CLAUDE_3_7_MODELS = ["claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219"]

# Claude 3.5 Model Family (Current)
CLAUDE_3_5_MODELS = [
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-latest",
    "claude-3-5-haiku-20241022",
]

# Claude 3 Model Family (Previous Generation)
CLAUDE_3_MODELS = [
    "claude-3-opus-latest",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]

# All Claude Models
ALL_CLAUDE_MODELS = (
    CLAUDE_4_MODELS + CLAUDE_3_7_MODELS + CLAUDE_3_5_MODELS + CLAUDE_3_MODELS
)

# Models that support thinking configuration
THINKING_ENABLED_MODELS = CLAUDE_4_MODELS + CLAUDE_3_7_MODELS

# Models with enhanced capabilities
SONNET_MODELS = [model for model in ALL_CLAUDE_MODELS if "sonnet" in model]
OPUS_MODELS = [model for model in ALL_CLAUDE_MODELS if "opus" in model]
HAIKU_MODELS = [model for model in ALL_CLAUDE_MODELS if "haiku" in model]

# Thinking Configuration
DEFAULT_THINKING_CONFIG = {
    "enabled": True,
    "budget_tokens": 1600,  # Minimum 1024, must be < max_tokens
}

THINKING_DISABLED_CONFIG = {"enabled": False}

# API Version that supports thinking
THINKING_API_VERSION = "2023-06-01"  # Update if newer version required


# Model categorization for features
def supports_thinking(model: str) -> bool:
    """Check if model supports thinking configuration."""
    return model in THINKING_ENABLED_MODELS


def is_claude_4(model: str) -> bool:
    """Check if model is Claude 4 family."""
    return model in CLAUDE_4_MODELS


def is_claude_3_7(model: str) -> bool:
    """Check if model is Claude 3.7 family."""
    return model in CLAUDE_3_7_MODELS


def get_model_family(model: str) -> str:
    """Get the model family name."""
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
    if "sonnet" in model:
        return "sonnet"
    elif "opus" in model:
        return "opus"
    elif "haiku" in model:
        return "haiku"
    else:
        return "unknown"


# Default models for different use cases
DEFAULT_CHAT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_REASONING_MODEL = "claude-opus-4-20250514"  # Best for complex reasoning
DEFAULT_FAST_MODEL = "claude-3-5-haiku-latest"  # Fastest responses
DEFAULT_THINKING_MODEL = "claude-sonnet-4-20250514"  # Best thinking capabilities


# Thinking configuration helpers
def create_thinking_config(
    enabled: bool = True, budget_tokens: int = 1600
) -> Dict[str, Any]:
    """Create a thinking configuration dictionary."""
    if not enabled:
        return {"enabled": False}

    if budget_tokens < 1024:
        raise ValueError("budget_tokens must be at least 1024")

    return {"enabled": True, "budget_tokens": budget_tokens}


def validate_thinking_config(thinking_config: Dict[str, Any], max_tokens: int) -> bool:
    """Validate thinking configuration parameters."""
    if not isinstance(thinking_config, dict):
        return False

    enabled = thinking_config.get("enabled", False)
    if not enabled:
        return True

    budget_tokens = thinking_config.get("budget_tokens", 0)
    if budget_tokens < 1024:
        return False

    if budget_tokens >= max_tokens:
        return False

    return True
