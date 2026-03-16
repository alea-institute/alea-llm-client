"""Provider-agnostic LLM kwargs helper.

Translates universal parameters (effort, tier) into provider-specific kwargs
so consumers don't need to know the differences between APIs.

Usage:
    from alea_llm_client.llms.utils.provider_kwargs import get_llm_kwargs

    # From a model instance
    kwargs = get_llm_kwargs(model, effort="low", tier="flex")
    response = model.chat(messages=[...], **kwargs)

    # From a provider string
    kwargs = get_llm_kwargs("openai", effort="high")

    # Override specific keys after
    kwargs = get_llm_kwargs(model, effort="low")
    kwargs["reasoning_effort"] = "xhigh"  # power user override
"""

from typing import Any, Optional, Union

# Effort level mappings per provider
_EFFORT_MAP: dict[str, dict[str, dict[str, Any]]] = {
    "openai": {
        "low": {"reasoning_effort": "none"},
        "medium": {"reasoning_effort": "medium"},
        "high": {"reasoning_effort": "high"},
    },
    "anthropic": {
        "low": {"output_config": {"effort": "low"}},
        "medium": {"output_config": {"effort": "medium"}},
        "high": {"output_config": {"effort": "high"}},
    },
    "google": {
        "low": {"thinking_level": "minimal"},
        "medium": {"thinking_level": "medium"},
        "high": {"thinking_level": "high"},
    },
    "xai": {
        "low": {"reasoning_effort": "none"},
        "medium": {"reasoning_effort": "medium"},
        "high": {"reasoning_effort": "high"},
    },
}

# Service tier mappings per provider
_TIER_MAP: dict[str, dict[str, dict[str, Any]]] = {
    "openai": {
        "flex": {"service_tier": "flex"},
        "standard": {"service_tier": "default"},
        "priority": {"service_tier": "priority"},
    },
    "anthropic": {
        "flex": {"service_tier": "auto"},
        "standard": {"service_tier": "standard_only"},
        "priority": {"service_tier": "auto"},
    },
    # Google and xAI don't have service tier params
}


def detect_provider(model: Any) -> str:
    """Detect the provider from a model instance.

    Args:
        model: A model instance or provider string.

    Returns:
        Provider name: "openai", "anthropic", "google", "xai", "vllm", or "unknown".
    """
    if isinstance(model, str):
        return model.lower()

    # Import here to avoid circular imports
    cls_name = type(model).__name__

    if cls_name == "GrokModel":
        return "xai"
    if cls_name == "OpenAIModel":
        return "openai"
    if cls_name == "AnthropicModel":
        return "anthropic"
    if cls_name == "GoogleModel":
        return "google"
    if cls_name == "VLLMModel":
        return "vllm"

    return "unknown"


def get_llm_kwargs(
    provider: Union[str, Any],
    effort: Optional[str] = None,
    tier: Optional[str] = None,
) -> dict[str, Any]:
    """Build provider-appropriate kwargs from universal settings.

    Translates universal effort/tier into provider-specific parameters.
    Returns a plain dict that can be spread into model calls.

    Args:
        provider: A model instance or provider string ("openai", "anthropic", "google", "xai", "vllm").
        effort: Universal effort level — "low", "medium", or "high".
            - low: minimal thinking, fastest, cheapest
            - medium: balanced (default behavior if omitted)
            - high: thorough reasoning, slowest
        tier: Universal service tier — "flex", "standard", or "priority".
            - flex: cheapest, may queue (OpenAI flex, Anthropic auto)
            - standard: normal pricing and latency
            - priority: fastest, premium pricing

    Returns:
        Dict of provider-specific kwargs. Empty dict for unsupported providers.

    Raises:
        ValueError: If effort or tier values are invalid.

    Examples:
        >>> get_llm_kwargs("openai", effort="low", tier="flex")
        {'reasoning_effort': 'none', 'service_tier': 'flex'}

        >>> get_llm_kwargs("anthropic", effort="high")
        {'output_config': {'effort': 'high'}}

        >>> get_llm_kwargs("google", effort="low")
        {'thinking_level': 'minimal'}

        >>> get_llm_kwargs("vllm", effort="low")
        {}
    """
    provider_name = detect_provider(provider)
    kwargs: dict[str, Any] = {}

    if effort is not None:
        if effort not in ("low", "medium", "high"):
            raise ValueError(f"Invalid effort '{effort}'. Must be one of: low, medium, high")
        effort_kwargs = _EFFORT_MAP.get(provider_name, {}).get(effort, {})
        kwargs.update(effort_kwargs)

    if tier is not None:
        if tier not in ("flex", "standard", "priority"):
            raise ValueError(f"Invalid tier '{tier}'. Must be one of: flex, standard, priority")
        tier_kwargs = _TIER_MAP.get(provider_name, {}).get(tier, {})
        kwargs.update(tier_kwargs)

    return kwargs
