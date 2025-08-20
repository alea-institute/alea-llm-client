"""Comprehensive model registry for all supported LLM providers.

This module provides model listings and capability queries, backed by
the authoritative model capabilities data from model_capabilities.py.

Model names are sourced directly from official SDKs and documentation.

Sources:
- OpenAI: references/openai-python/src/openai/types/shared/chat_model.py
- Anthropic: references/anthropic-sdk-python/src/anthropic/types/model.py
- Google: references/python-genai/ (various test files and docs)
- Grok/xAI: Official xAI API documentation (docs.x.ai)

Last Updated: August 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from alea_llm_client.llms.config.loader import get_global_loader

# Import the comprehensive capabilities - this is our single source of truth
from .model_capabilities import (
    COMPLETE_MODEL_CAPABILITIES,
    DetailedCapabilities,
    ModelStatus,
)


class OpenAIModels(str, Enum):
    """Complete list of OpenAI models.

    Source: references/openai-python/src/openai/types/shared/chat_model.py
    """

    # GPT-5 Models (Latest generation with grammar support)
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_2025_08_07 = "gpt-5-2025-08-07"
    GPT_5_MINI_2025_08_07 = "gpt-5-mini-2025-08-07"
    GPT_5_NANO_2025_08_07 = "gpt-5-nano-2025-08-07"
    GPT_5_CHAT_LATEST = "gpt-5-chat-latest"

    # GPT-4.1 Models
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4_1_2025_04_14 = "gpt-4.1-2025-04-14"
    GPT_4_1_MINI_2025_04_14 = "gpt-4.1-mini-2025-04-14"
    GPT_4_1_NANO_2025_04_14 = "gpt-4.1-nano-2025-04-14"

    # O-Series Reasoning Models (require max_completion_tokens)
    O4_MINI = "o4-mini"
    O4_MINI_2025_04_16 = "o4-mini-2025-04-16"
    O4_MINI_DEEP_RESEARCH = "o4-mini-deep-research"
    O4_MINI_DEEP_RESEARCH_2025_06_26 = "o4-mini-deep-research-2025-06-26"
    O3 = "o3"
    O3_2025_04_16 = "o3-2025-04-16"
    O3_MINI = "o3-mini"
    O3_MINI_2025_01_31 = "o3-mini-2025-01-31"
    O3_PRO = "o3-pro"
    O3_PRO_2025_06_10 = "o3-pro-2025-06-10"
    O3_DEEP_RESEARCH = "o3-deep-research"
    O3_DEEP_RESEARCH_2025_06_26 = "o3-deep-research-2025-06-26"
    O1 = "o1"
    O1_2024_12_17 = "o1-2024-12-17"
    O1_PREVIEW = "o1-preview"
    O1_PREVIEW_2024_09_12 = "o1-preview-2024-09-12"
    O1_MINI = "o1-mini"
    O1_MINI_2024_09_12 = "o1-mini-2024-09-12"
    O1_PRO = "o1-pro"
    O1_PRO_2025_03_19 = "o1-pro-2025-03-19"

    # GPT-4o Models
    GPT_4O = "gpt-4o"
    GPT_4O_2024_11_20 = "gpt-4o-2024-11-20"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"
    CHATGPT_4O_LATEST = "chatgpt-4o-latest"

    # Audio-Capable Models
    GPT_4O_AUDIO_PREVIEW = "gpt-4o-audio-preview"
    GPT_4O_AUDIO_PREVIEW_2024_10_01 = "gpt-4o-audio-preview-2024-10-01"
    GPT_4O_AUDIO_PREVIEW_2024_12_17 = "gpt-4o-audio-preview-2024-12-17"
    GPT_4O_AUDIO_PREVIEW_2025_06_03 = "gpt-4o-audio-preview-2025-06-03"
    GPT_4O_MINI_AUDIO_PREVIEW = "gpt-4o-mini-audio-preview"
    GPT_4O_MINI_AUDIO_PREVIEW_2024_12_17 = "gpt-4o-mini-audio-preview-2024-12-17"

    # Search Models
    GPT_4O_SEARCH_PREVIEW = "gpt-4o-search-preview"
    GPT_4O_MINI_SEARCH_PREVIEW = "gpt-4o-mini-search-preview"
    GPT_4O_SEARCH_PREVIEW_2025_03_11 = "gpt-4o-search-preview-2025-03-11"
    GPT_4O_MINI_SEARCH_PREVIEW_2025_03_11 = "gpt-4o-mini-search-preview-2025-03-11"

    # Realtime Models
    GPT_4O_REALTIME_PREVIEW = "gpt-4o-realtime-preview"
    GPT_4O_MINI_REALTIME_PREVIEW = "gpt-4o-mini-realtime-preview"

    # Computer Use Models
    COMPUTER_USE_PREVIEW = "computer-use-preview"
    COMPUTER_USE_PREVIEW_2025_03_11 = "computer-use-preview-2025-03-11"

    # Legacy GPT-4 Models
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT_4_VISION_PREVIEW = "gpt-4-vision-preview"
    GPT_4 = "gpt-4"
    GPT_4_0314 = "gpt-4-0314"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_32K_0314 = "gpt-4-32k-0314"
    GPT_4_32K_0613 = "gpt-4-32k-0613"

    # Legacy GPT-3.5 Models
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_TURBO_0301 = "gpt-3.5-turbo-0301"
    GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_3_5_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613"
    GPT_3_5_TURBO_INSTRUCT = "gpt-3.5-turbo-instruct"

    # Completion Models (Legacy)
    DAVINCI_002 = "davinci-002"
    BABBAGE_002 = "babbage-002"


class AnthropicModels(str, Enum):
    """Complete list of Anthropic Claude models.

    Source: references/anthropic-sdk-python/src/anthropic/types/model.py
    """

    # Claude 4 Models (Latest generation with thinking capabilities)
    CLAUDE_SONNET_4_20250514 = "claude-sonnet-4-20250514"
    CLAUDE_SONNET_4_0 = "claude-sonnet-4-0"
    CLAUDE_4_SONNET_20250514 = "claude-4-sonnet-20250514"
    CLAUDE_OPUS_4_0 = "claude-opus-4-0"
    CLAUDE_OPUS_4_20250514 = "claude-opus-4-20250514"
    CLAUDE_4_OPUS_20250514 = "claude-4-opus-20250514"
    CLAUDE_OPUS_4_1_20250805 = "claude-opus-4-1-20250805"

    # Claude 3.7 Models (Thinking-capable)
    CLAUDE_3_7_SONNET_LATEST = "claude-3-7-sonnet-latest"
    CLAUDE_3_7_SONNET_20250219 = "claude-3-7-sonnet-20250219"

    # Claude 3.5 Models
    CLAUDE_3_5_SONNET_LATEST = "claude-3-5-sonnet-latest"
    CLAUDE_3_5_SONNET_20241022 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_SONNET_20240620 = "claude-3-5-sonnet-20240620"
    CLAUDE_3_5_HAIKU_LATEST = "claude-3-5-haiku-latest"
    CLAUDE_3_5_HAIKU_20241022 = "claude-3-5-haiku-20241022"

    # Claude 3 Models
    CLAUDE_3_OPUS_LATEST = "claude-3-opus-latest"
    CLAUDE_3_OPUS_20240229 = "claude-3-opus-20240229"
    CLAUDE_3_HAIKU_20240307 = "claude-3-haiku-20240307"


class GoogleModels(str, Enum):
    """Complete list of Google Gemini models.

    Source: references/python-genai/ (various test files and documentation)
    """

    # Gemini 2.5 Models (Latest with thinking capabilities)
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"

    # Gemini 2.0 Models
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_PRO = "gemini-2.0-pro"
    GEMINI_2_0_FLASH_001 = "gemini-2.0-flash-001"
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"

    # Live/Realtime Models
    GEMINI_2_0_FLASH_LIVE_PREVIEW_04_09 = "gemini-2.0-flash-live-preview-04-09"
    GEMINI_LIVE_2_5_FLASH_PREVIEW = "gemini-live-2.5-flash-preview"

    # Gemini 1.5 Models (Previous generation)
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_FLASH_001 = "gemini-1.5-flash-001"
    GEMINI_1_5_FLASH_002 = "gemini-1.5-flash-002"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_PRO_001 = "gemini-1.5-pro-001"
    GEMINI_1_5_PRO_002 = "gemini-1.5-pro-002"

    # Gemini 1.0 Models
    GEMINI_1_0_PRO = "gemini-1.0-pro"
    GEMINI_1_0_PRO_001 = "gemini-1.0-pro-001"

    # Specialized Models
    IMAGEN_3_0_GENERATE_002 = "imagen-3.0-generate-002"
    IMAGE_SEGMENTATION_001 = "image-segmentation-001"


class GrokModels(str, Enum):
    """Complete list of xAI Grok models.

    Source: xAI API documentation (docs.x.ai) and API availability
    """

    # Grok 4 Models (Latest flagship)
    GROK_4 = "grok-4"
    GROK_4_HEAVY = "grok-4-heavy"

    # Grok 3 Models
    GROK_3 = "grok-3"
    GROK_3_MINI = "grok-3-mini"

    # Grok 2 Models
    GROK_2 = "grok-2"
    GROK_2_MINI = "grok-2-mini"
    GROK_2_1212 = "grok-2-1212"  # Default in current implementation
    GROK_2_IMAGE_1212 = "grok-2-image-1212"  # Image generation

    # Beta Models
    GROK_BETA = "grok-beta"


# Legacy compatibility: Simple capabilities alias for backward compatibility
@dataclass
class ModelCapabilities:
    """Legacy model capabilities for backward compatibility."""

    provider: str
    family: str
    size: Optional[str] = None
    supports_grammar: bool = False
    supports_reasoning_effort: bool = False
    supports_thinking: bool = False
    supports_audio: bool = False
    supports_vision: bool = False
    supports_search: bool = False
    supports_realtime: bool = False
    supports_computer_use: bool = False
    requires_max_completion_tokens: bool = False
    is_deprecated: bool = False
    knowledge_cutoff: Optional[str] = None

    @classmethod
    def from_detailed_capabilities(cls, detailed: DetailedCapabilities) -> "ModelCapabilities":
        """Convert from DetailedCapabilities to legacy format."""
        return cls(
            provider=detailed.provider,
            family=detailed.family,
            size=detailed.tier.value,
            supports_grammar=detailed.supports_grammar,
            supports_reasoning_effort=detailed.supports_reasoning_effort,
            supports_thinking=detailed.supports_thinking,
            supports_audio=detailed.supports_audio_input or detailed.supports_audio_output,
            supports_vision=detailed.supports_vision,
            supports_search=detailed.supports_search,
            supports_realtime=detailed.supports_realtime,
            supports_computer_use=detailed.supports_computer_use,
            requires_max_completion_tokens=detailed.requires_max_completion_tokens,
            is_deprecated=detailed.status in [ModelStatus.DEPRECATED, ModelStatus.LEGACY],
            knowledge_cutoff=detailed.knowledge_cutoff,
        )


def get_all_models() -> set[str]:
    """Get all available model names across all providers."""
    # Try to use dynamic configuration first as the authoritative source
    try:
        loader = get_global_loader()
        providers = loader.get_providers()
        models = set()

        for provider_config in providers.values():
            models.update(provider_config.models.keys())

        if models:  # Only use dynamic if we actually loaded models
            return models
    except Exception:
        pass  # Fall back to static capabilities

    # Fallback to the comprehensive model capabilities
    return set(COMPLETE_MODEL_CAPABILITIES.keys())


def get_provider_models(provider: str) -> set[str]:
    """Get all models for a specific provider.

    Args:
        provider: One of 'openai', 'anthropic', 'google', 'xai'

    Returns:
        Set of model names for the provider
    """
    provider = provider.lower()

    # Normalize provider names
    if provider in ["xai", "grok"]:
        provider = "xai"

    # Try to use dynamic configuration first
    try:
        loader = get_global_loader()
        models = loader.get_models_by_provider(provider)

        if models:  # Only use dynamic if we actually loaded models
            return set(models)
    except Exception:
        pass  # Fall back to static capabilities

    # Fallback to static capabilities
    return {name for name, caps in COMPLETE_MODEL_CAPABILITIES.items() if caps.provider.lower() == provider}


def get_model_capabilities(model_name: str) -> Optional[ModelCapabilities]:
    """Get capabilities for a specific model.

    Args:
        model_name: The model identifier

    Returns:
        ModelCapabilities object or None if not found
    """
    detailed = COMPLETE_MODEL_CAPABILITIES.get(model_name)
    if detailed:
        return ModelCapabilities.from_detailed_capabilities(detailed)
    return None


def get_detailed_model_capabilities(model_name: str) -> Optional[DetailedCapabilities]:
    """Get detailed capabilities for a specific model.

    Args:
        model_name: The model identifier

    Returns:
        DetailedCapabilities object or None if not found
    """
    return COMPLETE_MODEL_CAPABILITIES.get(model_name)


def is_model_deprecated(model_name: str) -> bool:
    """Check if a model is deprecated.

    Args:
        model_name: The model identifier

    Returns:
        True if deprecated, False otherwise
    """
    detailed = COMPLETE_MODEL_CAPABILITIES.get(model_name)
    if detailed:
        return detailed.status in [ModelStatus.DEPRECATED, ModelStatus.LEGACY]

    # Fallback heuristics for models not in capabilities data
    if model_name.startswith("gpt-3.5-") or model_name.startswith("gpt-4-"):
        if "turbo" not in model_name and "o" not in model_name:
            return True

    return False


def supports_feature(model_name: str, feature: str) -> bool:
    """Check if a model supports a specific feature.

    Args:
        model_name: The model identifier
        feature: One of 'grammar', 'reasoning_effort', 'thinking', 'audio',
                'vision', 'search', 'realtime', 'computer_use', 'tools',
                'structured_output', 'streaming', 'fine_tuning'

    Returns:
        True if the model supports the feature
    """
    detailed = COMPLETE_MODEL_CAPABILITIES.get(model_name)
    if not detailed:
        # Fallback heuristics for models not in capabilities data
        if feature == "grammar" and model_name.startswith("gpt-5"):
            return True
        if feature == "reasoning_effort" and model_name.startswith(("o1", "o3", "o4")):
            return True
        if feature == "thinking":
            if "claude-4" in model_name or "claude-3-7" in model_name:
                return True
            if "gemini-2.5" in model_name:
                return True
        if feature == "audio" and "audio" in model_name:
            return True
        if feature == "search" and "search" in model_name:
            return True
        if feature == "realtime" and "realtime" in model_name:
            return True
        return bool(feature == "computer_use" and "computer-use" in model_name)

    # Use comprehensive capabilities data
    feature_map = {
        "grammar": detailed.supports_grammar,
        "reasoning_effort": detailed.supports_reasoning_effort,
        "thinking": detailed.supports_thinking,
        "audio": detailed.supports_audio_input or detailed.supports_audio_output,
        "audio_input": detailed.supports_audio_input,
        "audio_output": detailed.supports_audio_output,
        "vision": detailed.supports_vision,
        "search": detailed.supports_search,
        "realtime": detailed.supports_realtime,
        "computer_use": detailed.supports_computer_use,
        "tools": detailed.supports_tools,
        "structured_output": detailed.supports_structured_output,
        "streaming": detailed.supports_streaming,
        "fine_tuning": detailed.supports_fine_tuning,
        "reasoning": detailed.supports_reasoning,
    }

    return feature_map.get(feature, False)


# Default models per provider (dynamically determined from capabilities)
def get_default_models() -> dict[str, str]:
    """Get default model for each provider based on comprehensive capabilities."""
    defaults = {}

    # Try to use dynamic configuration first
    try:
        loader = get_global_loader()
        providers = loader.get_providers()

        for provider_name, provider_config in providers.items():
            chat_default = provider_config.defaults.get("chat")
            if chat_default:
                defaults[provider_name] = chat_default
                continue

    except Exception:
        pass  # Fall back to static defaults

    # Static fallback logic for any missing providers
    for provider in ["openai", "anthropic", "google", "xai"]:
        if provider in defaults:
            continue  # Already have dynamic default

        provider_models = get_provider_models(provider)

        # Find the best available model for each provider
        if provider == "openai":
            # Prefer GPT-5 chat latest, fallback to available GPT-5 models
            for candidate in ["gpt-5-chat-latest", "gpt-5", "gpt-4o"]:
                if candidate in provider_models:
                    defaults[provider] = candidate
                    break
        elif provider == "anthropic":
            # Prefer Claude 4 sonnet, fallback to Claude 3.5 sonnet
            for candidate in ["claude-sonnet-4-20250514", "claude-3-5-sonnet-latest"]:
                if candidate in provider_models:
                    defaults[provider] = candidate
                    break
        elif provider == "google":
            # Prefer Gemini 2.0 flash exp, fallback to 2.0 flash
            for candidate in [
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
            ]:
                if candidate in provider_models:
                    defaults[provider] = candidate
                    break
        elif provider == "xai":
            # Prefer Grok 2-1212, fallback to Grok 4, then Grok 2
            for candidate in ["grok-2-1212", "grok-4", "grok-2"]:
                if candidate in provider_models:
                    defaults[provider] = candidate
                    break

    return defaults


# Legacy compatibility
DEFAULT_MODELS = get_default_models()


# Model size categories dynamically generated from capabilities data
def get_models_by_tier(tier: str) -> list[str]:
    """Get all models of a specific tier/size category.

    Args:
        tier: One of 'nano', 'mini', 'small', 'medium', 'standard', 'large', 'pro', 'flash', 'heavy'

    Returns:
        List of model names in that tier
    """
    tier_models = []

    for name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        if caps.tier.value == tier:
            tier_models.append(name)

    return tier_models


# Legacy compatibility: Dynamic model size categories
def get_model_sizes() -> dict[str, list[str]]:
    """Get model size categories dynamically from capabilities data."""
    return {
        "nano": get_models_by_tier("nano"),
        "mini": get_models_by_tier("mini"),
        "small": get_models_by_tier("small"),
        "medium": get_models_by_tier("medium"),
        "standard": get_models_by_tier("standard"),
        "large": get_models_by_tier("large"),
        "pro": get_models_by_tier("pro"),
        "flash": get_models_by_tier("flash"),
        "heavy": get_models_by_tier("heavy"),
    }


MODEL_SIZES = get_model_sizes()
