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


# ========== Model-aware: OpenAI GPT-5.x (supports effort + tier) ==========


def test_openai_gpt5_effort_low():
    model = OpenAIModel(model="gpt-5.4")
    kwargs = get_llm_kwargs(model, effort="low")
    assert kwargs == {"reasoning_effort": "none"}


def test_openai_gpt5_effort_high():
    model = OpenAIModel(model="gpt-5.4")
    kwargs = get_llm_kwargs(model, effort="high")
    assert kwargs == {"reasoning_effort": "high"}


def test_openai_gpt5_tier_flex():
    model = OpenAIModel(model="gpt-5.4")
    kwargs = get_llm_kwargs(model, tier="flex")
    assert kwargs == {"service_tier": "flex"}


def test_openai_gpt5_effort_and_tier():
    model = OpenAIModel(model="gpt-5.4")
    kwargs = get_llm_kwargs(model, effort="low", tier="flex")
    assert kwargs == {"reasoning_effort": "none", "service_tier": "flex"}


# ========== Model-aware: OpenAI GPT-4.x (no effort, no tier) ==========


def test_openai_gpt4_effort_noop():
    """gpt-4.1-mini doesn't support reasoning_effort — should return empty."""
    model = OpenAIModel(model="gpt-4.1-mini")
    kwargs = get_llm_kwargs(model, effort="low")
    assert kwargs == {}


def test_openai_gpt4_tier_noop():
    """gpt-4.1-mini doesn't support service_tier — should return empty."""
    model = OpenAIModel(model="gpt-4.1-mini")
    kwargs = get_llm_kwargs(model, tier="flex")
    assert kwargs == {}


def test_openai_gpt4_effort_and_tier_noop():
    """gpt-4.1-mini supports neither — should return empty."""
    model = OpenAIModel(model="gpt-4.1-mini")
    kwargs = get_llm_kwargs(model, effort="high", tier="priority")
    assert kwargs == {}


# ========== Model-aware: OpenAI o-series (no effort, yes tier) ==========


def test_openai_oseries_effort_noop():
    """o-series doesn't support reasoning_effort in Chat API."""
    model = OpenAIModel(model="o4-mini")
    kwargs = get_llm_kwargs(model, effort="low")
    assert kwargs == {}


def test_openai_oseries_tier_ok():
    """o-series supports service_tier."""
    model = OpenAIModel(model="o4-mini")
    kwargs = get_llm_kwargs(model, tier="flex")
    assert kwargs == {"service_tier": "flex"}


# ========== String provider (no model instance — assumes latest) ==========


def test_string_openai_returns_empty():
    """String provider without model instance returns empty (can't check model)."""
    kwargs = get_llm_kwargs("openai", effort="low")
    assert kwargs == {}


# ========== Anthropic (all models support effort + tier) ==========


def test_anthropic_effort_low():
    model = AnthropicModel()
    kwargs = get_llm_kwargs(model, effort="low")
    assert kwargs == {"output_config": {"effort": "low"}}


def test_anthropic_effort_high():
    model = AnthropicModel()
    kwargs = get_llm_kwargs(model, effort="high")
    assert kwargs == {"output_config": {"effort": "high"}}


def test_anthropic_tier_flex():
    model = AnthropicModel()
    kwargs = get_llm_kwargs(model, tier="flex")
    assert kwargs == {"service_tier": "auto"}


def test_anthropic_tier_standard():
    model = AnthropicModel()
    kwargs = get_llm_kwargs(model, tier="standard")
    assert kwargs == {"service_tier": "standard_only"}


def test_anthropic_haiku_effort_noop():
    """Haiku doesn't support output_config.effort."""
    model = AnthropicModel(model="claude-haiku-4-5")
    kwargs = get_llm_kwargs(model, effort="low")
    assert kwargs == {}


def test_anthropic_haiku_tier_ok():
    """Haiku supports service_tier."""
    model = AnthropicModel(model="claude-haiku-4-5")
    kwargs = get_llm_kwargs(model, tier="flex")
    assert kwargs == {"service_tier": "auto"}


def test_anthropic_string():
    """Anthropic string provider assumes latest (supports effort)."""
    kwargs = get_llm_kwargs("anthropic", effort="low", tier="flex")
    assert kwargs == {"output_config": {"effort": "low"}, "service_tier": "auto"}


# ========== Google (thinking only on 2.5+/3.x) ==========


def test_google_gemini3_effort():
    model = GoogleModel(model="gemini-3.1-pro-preview")
    kwargs = get_llm_kwargs(model, effort="low")
    assert kwargs == {"thinking_level": "minimal"}


def test_google_gemini25_effort():
    model = GoogleModel(model="gemini-2.5-flash")
    kwargs = get_llm_kwargs(model, effort="high")
    assert kwargs == {"thinking_level": "high"}


def test_google_gemini20_effort_noop():
    """gemini-2.0-flash doesn't support thinking."""
    model = GoogleModel(model="gemini-2.0-flash")
    kwargs = get_llm_kwargs(model, effort="high")
    assert kwargs == {}


def test_google_tier_noop():
    model = GoogleModel(model="gemini-3.1-pro-preview")
    kwargs = get_llm_kwargs(model, tier="flex")
    assert kwargs == {}


# ========== xAI/Grok ==========


def test_grok_effort():
    """Grok uses same rules as OpenAI — model name determines support."""
    model = GrokModel(model="grok-4-fast-non-reasoning")
    kwargs = get_llm_kwargs(model, effort="medium")
    assert kwargs == {}  # grok models don't start with gpt-5


# ========== VLLM / unknown ==========


def test_vllm_noop():
    kwargs = get_llm_kwargs("vllm", effort="low", tier="flex")
    assert kwargs == {}


def test_unknown_provider_noop():
    kwargs = get_llm_kwargs("unknown_provider", effort="high", tier="priority")
    assert kwargs == {}


# ========== No params ==========


def test_no_params():
    model = OpenAIModel(model="gpt-5.4")
    kwargs = get_llm_kwargs(model)
    assert kwargs == {}


# ========== Validation ==========


def test_invalid_effort():
    with pytest.raises(ValueError, match="Invalid effort"):
        get_llm_kwargs("openai", effort="super_duper")


def test_invalid_tier():
    with pytest.raises(ValueError, match="Invalid tier"):
        get_llm_kwargs("openai", tier="ultra")


# ========== Integration: kwargs work with actual API calls ==========


def test_openai_gpt5_integration():
    """GPT-5.4 with effort=low tier=flex actually works."""
    model = OpenAIModel(model="gpt-5.4")
    kwargs = get_llm_kwargs(model, effort="low", tier="flex")
    response = model.chat(
        messages=[{"role": "user", "content": "Say OK"}],
        **kwargs,
    )
    assert response.text is not None


def test_openai_gpt4_integration():
    """GPT-4.1-mini with effort/tier — helper returns empty, call works."""
    model = OpenAIModel(model="gpt-4.1-mini")
    kwargs = get_llm_kwargs(model, effort="low", tier="flex")
    assert kwargs == {}
    response = model.chat(
        messages=[{"role": "user", "content": "Say OK"}],
        **kwargs,
    )
    assert response.text is not None


def test_anthropic_integration():
    model = AnthropicModel()
    kwargs = get_llm_kwargs(model, effort="low")
    response = model.chat(
        messages=[{"role": "user", "content": "Say OK"}],
        **kwargs,
    )
    assert response.text is not None


def test_google_integration():
    model = GoogleModel()
    kwargs = get_llm_kwargs(model, effort="low")
    response = model.chat(
        messages=[{"role": "user", "content": "Say OK"}],
        **kwargs,
    )
    assert response.text is not None
