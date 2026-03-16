"""Provider-agnostic LLM kwargs helper.

Translates universal parameters (effort, tier) into provider-specific kwargs
so consumers don't need to know the differences between APIs.

Only emits params that the specific model actually supports — won't send
reasoning_effort to gpt-4.1-mini or service_tier to gpt-4o.

Usage:
    from alea_llm_client.llms.utils.provider_kwargs import get_llm_kwargs

    # From a model instance (recommended — enables model-aware filtering)
    kwargs = get_llm_kwargs(model, effort="low", tier="flex")
    response = model.chat(messages=[...], **kwargs)

    # From a provider string (sends all params for that provider)
    kwargs = get_llm_kwargs("openai", effort="high")

    # Override specific keys after
    kwargs = get_llm_kwargs(model, effort="low")
    kwargs["reasoning_effort"] = "xhigh"  # power user override
"""

from typing import Any, Optional, Union


def _get_model_name(model: Any) -> str | None:
    """Extract model name string from a model instance."""
    if hasattr(model, "model") and isinstance(model.model, str):
        return model.model
    return None


def _openai_supports_effort(model_name: str | None) -> bool:
    """Check if an OpenAI model supports reasoning_effort in Chat API."""
    if model_name is None:
        return False
    # GPT-5.x models support reasoning_effort in Chat Completions API
    return model_name.startswith("gpt-5")


def _openai_supports_tier(model_name: str | None) -> bool:
    """Check if an OpenAI model supports service_tier."""
    if model_name is None:
        return False
    # GPT-5.x and o-series support service_tier
    return model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def _anthropic_supports_effort(model_name: str | None) -> bool:
    """Check if an Anthropic model supports output_config.effort."""
    if model_name is None:
        return True  # assume latest when no model instance
    # Haiku models don't support the effort parameter
    return "haiku" not in model_name


def _google_supports_thinking(model_name: str | None) -> bool:
    """Check if a Google model supports thinking_level."""
    if model_name is None:
        return False
    # Gemini 2.5+ and 3.x support thinking
    return model_name.startswith(("gemini-2.5", "gemini-3"))


# Effort level mappings
_EFFORT_OPENAI = {"low": "none", "medium": "medium", "high": "high"}
_EFFORT_ANTHROPIC = {"low": "low", "medium": "medium", "high": "high"}
_EFFORT_GOOGLE = {"low": "minimal", "medium": "medium", "high": "high"}

# Tier mappings
_TIER_OPENAI = {"flex": "flex", "standard": "default", "priority": "priority"}
_TIER_ANTHROPIC = {"flex": "auto", "standard": "standard_only", "priority": "auto"}


def detect_provider(model: Any) -> str:
    """Detect the provider from a model instance.

    Args:
        model: A model instance or provider string.

    Returns:
        Provider name: "openai", "anthropic", "google", "xai", "vllm", or "unknown".
    """
    if isinstance(model, str):
        return model.lower()

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

    Model-aware: only emits params the specific model supports. For example,
    effort="low" with gpt-4.1-mini returns {} because that model doesn't
    support reasoning_effort, but with gpt-5.4 returns {"reasoning_effort": "none"}.

    When called with a string provider name (no model instance), assumes the
    latest model capabilities for that provider.

    Args:
        provider: A model instance (recommended) or provider string.
        effort: Universal effort level — "low", "medium", or "high".
            - low: minimal thinking, fastest, cheapest
            - medium: balanced (default behavior if omitted)
            - high: thorough reasoning, slowest
        tier: Universal service tier — "flex", "standard", or "priority".
            - flex: cheapest, may queue
            - standard: normal pricing and latency
            - priority: fastest, premium pricing

    Returns:
        Dict of provider-specific kwargs. Empty dict for unsupported
        models/providers.

    Raises:
        ValueError: If effort or tier values are invalid.

    Examples:
        >>> from alea_llm_client import OpenAIModel
        >>> model = OpenAIModel(model="gpt-5.4")
        >>> get_llm_kwargs(model, effort="low", tier="flex")
        {'reasoning_effort': 'none', 'service_tier': 'flex'}

        >>> model = OpenAIModel(model="gpt-4.1-mini")
        >>> get_llm_kwargs(model, effort="low", tier="flex")
        {}

        >>> get_llm_kwargs("anthropic", effort="high")
        {'output_config': {'effort': 'high'}}
    """
    if effort is not None and effort not in ("low", "medium", "high"):
        raise ValueError(f"Invalid effort '{effort}'. Must be one of: low, medium, high")
    if tier is not None and tier not in ("flex", "standard", "priority"):
        raise ValueError(f"Invalid tier '{tier}'. Must be one of: flex, standard, priority")

    provider_name = detect_provider(provider)
    model_name = _get_model_name(provider) if not isinstance(provider, str) else None
    kwargs: dict[str, Any] = {}

    if provider_name in ("openai", "xai"):
        if effort is not None and _openai_supports_effort(model_name):
            kwargs["reasoning_effort"] = _EFFORT_OPENAI[effort]
        if tier is not None and _openai_supports_tier(model_name):
            kwargs["service_tier"] = _TIER_OPENAI[tier]

    elif provider_name == "anthropic":
        if effort is not None and _anthropic_supports_effort(model_name):
            kwargs["output_config"] = {"effort": _EFFORT_ANTHROPIC[effort]}
        if tier is not None:
            kwargs["service_tier"] = _TIER_ANTHROPIC[tier]

    elif provider_name == "google":
        if effort is not None and _google_supports_thinking(model_name):
            kwargs["thinking_level"] = _EFFORT_GOOGLE[effort]
        # Google doesn't have service tier

    # vllm, unknown → empty dict (no-op)

    return kwargs
