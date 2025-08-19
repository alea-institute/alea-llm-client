"""OpenAI model constants and parameter definitions."""

from typing import Dict, Any

# GPT-5 Model Family
GPT_5_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5-chat-latest",
]

# GPT-4.1 Model Family
GPT_4_1_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
]

# O-Series Reasoning Models
O_SERIES_MODELS = [
    "o1",
    "o1-2024-12-17",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-pro",
    "o1-pro-2025-03-19",
    "o3",
    "o3-2025-04-16",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o3-pro",
    "o3-pro-2025-06-10",
    "o3-deep-research",
    "o3-deep-research-2025-06-26",
    "o4-mini",
    "o4-mini-2025-04-16",
    "o4-mini-deep-research",
    "o4-mini-deep-research-2025-06-26",
]

# Audio-Capable Models
AUDIO_MODELS = [
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
]

# Search-Enhanced Models
SEARCH_MODELS = [
    "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-mini-search-preview-2025-03-11",
]

# Specialized Models
COMPUTER_USE_MODELS = ["computer-use-preview", "computer-use-preview-2025-03-11"]

REALTIME_MODELS = [
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2025-06-03",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17",
]

# All GPT-4o Models (current generation)
GPT_4O_MODELS = (
    [
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "chatgpt-4o-latest",
    ]
    + AUDIO_MODELS
    + SEARCH_MODELS
)

# Parameter Values
REASONING_EFFORT_VALUES = ["minimal", "low", "medium", "high"]
VERBOSITY_VALUES = ["low", "medium", "high"]
MODALITY_VALUES = ["text", "audio"]

# Models that support reasoning effort parameter
REASONING_EFFORT_SUPPORTED_MODELS = O_SERIES_MODELS

# Models that support verbosity parameter
VERBOSITY_SUPPORTED_MODELS = GPT_5_MODELS + GPT_4_1_MODELS + GPT_4O_MODELS

# Models that support max_completion_tokens (o-series)
MAX_COMPLETION_TOKENS_MODELS = O_SERIES_MODELS

# Models that support audio modality
AUDIO_SUPPORTED_MODELS = AUDIO_MODELS

# Models that support realtime streaming
REALTIME_SUPPORTED_MODELS = REALTIME_MODELS

# Models that support computer use
COMPUTER_USE_SUPPORTED_MODELS = COMPUTER_USE_MODELS

# Models that support grammar constraints (context-free grammars)
GRAMMAR_SUPPORTED_MODELS = GPT_5_MODELS.copy()

# Optimization Presets
SPEED_OPTIMIZED: Dict[str, Any] = {"reasoning_effort": "minimal", "verbosity": "low"}

BALANCED: Dict[str, Any] = {"reasoning_effort": "medium", "verbosity": "medium"}

THOROUGH: Dict[str, Any] = {"reasoning_effort": "high", "verbosity": "high"}


# Model categorization for parameter support
def supports_reasoning_effort(model: str) -> bool:
    """Check if model supports reasoning_effort parameter."""
    return model in REASONING_EFFORT_SUPPORTED_MODELS


def supports_verbosity(model: str) -> bool:
    """Check if model supports verbosity parameter."""
    return model in VERBOSITY_SUPPORTED_MODELS


def supports_audio(model: str) -> bool:
    """Check if model supports audio modality."""
    return model in AUDIO_SUPPORTED_MODELS


def requires_max_completion_tokens(model: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens."""
    return model in MAX_COMPLETION_TOKENS_MODELS


def supports_grammar(model: str) -> bool:
    """Check if model supports grammar constraints (context-free grammars)."""
    return model in GRAMMAR_SUPPORTED_MODELS


def supports_realtime(model: str) -> bool:
    """Check if model supports realtime streaming."""
    return model in REALTIME_SUPPORTED_MODELS


def supports_computer_use(model: str) -> bool:
    """Check if model supports computer use capabilities."""
    return model in COMPUTER_USE_SUPPORTED_MODELS


# Default models for different use cases
DEFAULT_CHAT_MODEL = "gpt-5-chat-latest"
DEFAULT_REASONING_MODEL = "o3-mini"
DEFAULT_FAST_MODEL = "gpt-5-nano"
DEFAULT_AUDIO_MODEL = "gpt-4o-audio-preview-2025-06-03"
