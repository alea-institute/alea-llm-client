"""Tests for provider-agnostic LLM kwargs helper."""

import pytest

from alea_llm_client import (
    AnthropicModel,
    GoogleModel,
    GrokModel,
    OpenAIModel,
    detect_provider,
    get_llm_kwargs,
)

# ========== detect_provider ==========


def test_detect_provider_from_string():
    assert detect_provider("openai") == "openai"
    assert detect_provider("Anthropic") == "anthropic"
    assert detect_provider("GOOGLE") == "google"
    assert detect_provider("xai") == "xai"
    assert detect_provider("vllm") == "vllm"


def test_detect_provider_from_model_instance():
    assert detect_provider(OpenAIModel()) == "openai"
    assert detect_provider(AnthropicModel()) == "anthropic"
    assert detect_provider(GrokModel()) == "xai"


def test_detect_provider_unknown():
    assert detect_provider("something_else") == "something_else"
    assert detect_provider(42) == "unknown"


# ========== get_llm_kwargs: OpenAI ==========


def test_openai_effort_low():
    kwargs = get_llm_kwargs("openai", effort="low")
    assert kwargs == {"reasoning_effort": "none"}


def test_openai_effort_medium():
    kwargs = get_llm_kwargs("openai", effort="medium")
    assert kwargs == {"reasoning_effort": "medium"}


def test_openai_effort_high():
    kwargs = get_llm_kwargs("openai", effort="high")
    assert kwargs == {"reasoning_effort": "high"}


def test_openai_tier_flex():
    kwargs = get_llm_kwargs("openai", tier="flex")
    assert kwargs == {"service_tier": "flex"}


def test_openai_tier_standard():
    kwargs = get_llm_kwargs("openai", tier="standard")
    assert kwargs == {"service_tier": "default"}


def test_openai_tier_priority():
    kwargs = get_llm_kwargs("openai", tier="priority")
    assert kwargs == {"service_tier": "priority"}


def test_openai_effort_and_tier():
    kwargs = get_llm_kwargs("openai", effort="low", tier="flex")
    assert kwargs == {"reasoning_effort": "none", "service_tier": "flex"}


# ========== get_llm_kwargs: Anthropic ==========


def test_anthropic_effort_low():
    kwargs = get_llm_kwargs("anthropic", effort="low")
    assert kwargs == {"output_config": {"effort": "low"}}


def test_anthropic_effort_high():
    kwargs = get_llm_kwargs("anthropic", effort="high")
    assert kwargs == {"output_config": {"effort": "high"}}


def test_anthropic_tier_flex():
    kwargs = get_llm_kwargs("anthropic", tier="flex")
    assert kwargs == {"service_tier": "auto"}


def test_anthropic_tier_standard():
    kwargs = get_llm_kwargs("anthropic", tier="standard")
    assert kwargs == {"service_tier": "standard_only"}


# ========== get_llm_kwargs: Google ==========


def test_google_effort_low():
    kwargs = get_llm_kwargs("google", effort="low")
    assert kwargs == {"thinking_level": "minimal"}


def test_google_effort_high():
    kwargs = get_llm_kwargs("google", effort="high")
    assert kwargs == {"thinking_level": "high"}


def test_google_tier_noop():
    kwargs = get_llm_kwargs("google", tier="flex")
    assert kwargs == {}  # Google doesn't have service tier


# ========== get_llm_kwargs: xAI ==========


def test_xai_effort_low():
    kwargs = get_llm_kwargs("xai", effort="low")
    assert kwargs == {"reasoning_effort": "none"}


# ========== get_llm_kwargs: VLLM / unknown ==========


def test_vllm_noop():
    kwargs = get_llm_kwargs("vllm", effort="low", tier="flex")
    assert kwargs == {}


def test_unknown_provider_noop():
    kwargs = get_llm_kwargs("unknown_provider", effort="high", tier="priority")
    assert kwargs == {}


# ========== get_llm_kwargs: no params ==========


def test_no_params():
    kwargs = get_llm_kwargs("openai")
    assert kwargs == {}


# ========== get_llm_kwargs: validation ==========


def test_invalid_effort():
    with pytest.raises(ValueError, match="Invalid effort"):
        get_llm_kwargs("openai", effort="super_duper")


def test_invalid_tier():
    with pytest.raises(ValueError, match="Invalid tier"):
        get_llm_kwargs("openai", tier="ultra")


# ========== get_llm_kwargs: from model instances ==========


def test_from_openai_model_instance():
    model = OpenAIModel()
    kwargs = get_llm_kwargs(model, effort="low", tier="flex")
    assert kwargs == {"reasoning_effort": "none", "service_tier": "flex"}


def test_from_anthropic_model_instance():
    model = AnthropicModel()
    kwargs = get_llm_kwargs(model, effort="high")
    assert kwargs == {"output_config": {"effort": "high"}}


def test_from_grok_model_instance():
    model = GrokModel()
    kwargs = get_llm_kwargs(model, effort="medium")
    assert kwargs == {"reasoning_effort": "medium"}


# ========== Integration: kwargs work with actual API calls ==========


def test_openai_integration():
    """Verify get_llm_kwargs output works in a real API call."""
    model = OpenAIModel()
    kwargs = get_llm_kwargs(model, effort="low", tier="flex")
    response = model.chat(
        messages=[{"role": "user", "content": "Say OK"}],
        **kwargs,
    )
    assert response.text is not None
    assert len(response.text) > 0


def test_anthropic_integration():
    """Verify get_llm_kwargs output works in a real API call."""
    model = AnthropicModel()
    kwargs = get_llm_kwargs(model, effort="low")
    response = model.chat(
        messages=[{"role": "user", "content": "Say OK"}],
        **kwargs,
    )
    assert response.text is not None
    assert len(response.text) > 0


def test_google_integration():
    """Verify get_llm_kwargs output works in a real API call."""
    model = GoogleModel()
    kwargs = get_llm_kwargs(model, effort="low")
    response = model.chat(
        messages=[{"role": "user", "content": "Say OK"}],
        **kwargs,
    )
    assert response.text is not None
    assert len(response.text) > 0
