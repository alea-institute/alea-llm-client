"""Tests for the dynamic model registry."""

from alea_llm_client.llms import (
    DEFAULT_MODELS,
    MODEL_SIZES,
    get_all_models,
    get_model_capabilities,
    get_provider_models,
    is_model_deprecated,
    supports_feature,
)


def test_dynamic_model_loading():
    """Test that models are loaded dynamically from JSON config."""
    # Test that key models are available from dynamic loading
    all_models = get_all_models()

    # Check that flagship models from each provider exist
    assert "gpt-5" in all_models, "GPT-5 should be available"
    assert any("claude-sonnet-4" in model for model in all_models), "Claude-4 models should be available"
    assert any("gemini-2" in model for model in all_models), "Gemini-2 models should be available"
    assert any("grok" in model.lower() for model in all_models), "Grok models should be available"


def test_get_all_models():
    """Test getting all models."""
    all_models = get_all_models()

    # Should have models from all providers (updated to realistic expectation)
    assert len(all_models) > 90  # We have 70 OpenAI + 9 Anthropic + 7 Google + 9 xAI = 95 total

    # Check specific models are included
    assert "gpt-5" in all_models
    assert "claude-sonnet-4-20250514" in all_models
    assert "gemini-2.5-flash" in all_models
    assert "grok-2-1212" in all_models


def test_get_provider_models():
    """Test getting models by provider."""
    # OpenAI
    openai_models = get_provider_models("openai")
    assert len(openai_models) > 65  # Real API has 70 models
    assert "gpt-5" in openai_models
    assert "o3-mini" in openai_models

    # Anthropic
    anthropic_models = get_provider_models("anthropic")
    assert len(anthropic_models) > 8  # Real API has 9 models
    assert "claude-sonnet-4-20250514" in anthropic_models

    # Google
    google_models = get_provider_models("google")
    assert len(google_models) > 6  # We have 7 Google models configured
    assert "gemini-2.5-flash" in google_models

    # xAI/Grok (both names should work)
    grok_models = get_provider_models("xai")
    assert len(grok_models) > 8  # Real API has 9 models
    assert "grok-2-1212" in grok_models

    grok_models2 = get_provider_models("grok")
    assert grok_models == grok_models2

    # Invalid provider
    invalid = get_provider_models("invalid")
    assert len(invalid) == 0


def test_model_capabilities():
    """Test getting model capabilities."""
    # Test known model
    caps = get_model_capabilities("gpt-5")
    assert caps is not None
    assert caps.provider == "openai"
    assert caps.family == "gpt-5"
    assert caps.supports_grammar is True
    assert caps.supports_reasoning_effort is False

    # Test O-series
    caps = get_model_capabilities("o3-mini")
    assert caps is not None
    assert caps.supports_reasoning_effort is True
    assert caps.requires_max_completion_tokens is True

    # Test Claude 4
    caps = get_model_capabilities("claude-sonnet-4-20250514")
    assert caps is not None
    assert caps.supports_thinking is True

    # Test unknown model
    caps = get_model_capabilities("unknown-model")
    assert caps is None


def test_supports_feature():
    """Test feature support detection."""
    # GPT-5 supports grammar
    assert supports_feature("gpt-5", "grammar") is True
    assert supports_feature("gpt-5-mini", "grammar") is True
    assert supports_feature("gpt-5", "reasoning_effort") is False

    # O-series supports reasoning_effort
    assert supports_feature("o3-mini", "reasoning_effort") is True
    assert supports_feature("o1", "reasoning_effort") is True
    assert supports_feature("o3-mini", "grammar") is False

    # Claude 4 supports thinking
    assert supports_feature("claude-sonnet-4-20250514", "thinking") is True
    assert supports_feature("claude-3-7-sonnet-latest", "thinking") is True

    # Gemini 2.5 supports thinking
    assert supports_feature("gemini-2.5-flash", "thinking") is True

    # Audio models
    assert supports_feature("gpt-4o-audio-preview", "audio") is True

    # Search models
    assert supports_feature("gpt-4o-search-preview", "search") is True

    # Unknown model/feature
    assert supports_feature("unknown-model", "grammar") is False


def test_is_model_deprecated():
    """Test deprecation detection."""
    # Legacy models should be detected as deprecated
    assert is_model_deprecated("gpt-4-0314") is True  # Old dated version

    # Current models should not be deprecated
    assert is_model_deprecated("gpt-5") is False
    assert is_model_deprecated("claude-sonnet-4-20250514") is False


def test_default_models():
    """Test default model selection."""
    assert DEFAULT_MODELS["openai"] == "gpt-5-chat-latest"
    assert DEFAULT_MODELS["anthropic"] == "claude-sonnet-4-6"
    assert DEFAULT_MODELS["google"] == "gemini-3.1-pro-preview"
    assert DEFAULT_MODELS["xai"] == "grok-4-fast-non-reasoning"


def test_model_sizes():
    """Test model size categorization with dynamic models."""
    # Check mini models contain expected patterns
    mini_models = MODEL_SIZES["mini"]
    mini_model_strings = [str(m) for m in mini_models]
    assert any("gpt-5-mini" in model for model in mini_model_strings), "Should contain GPT-5 mini models"
    assert any("o3-mini" in model for model in mini_model_strings), "Should contain O3 mini models"
    assert any("grok" in model.lower() and "mini" in model for model in mini_model_strings), (
        "Should contain Grok mini models"
    )

    # Check flash models contain expected patterns
    flash_models = MODEL_SIZES["flash"]
    flash_model_strings = [str(m) for m in flash_models]
    assert any("gemini" in model and "flash" in model for model in flash_model_strings), (
        "Should contain Gemini flash models"
    )

    # Check pro models contain expected patterns
    pro_models = MODEL_SIZES["pro"]
    pro_model_strings = [str(m) for m in pro_models]
    assert any("gemini" in model and "pro" in model for model in pro_model_strings), "Should contain Gemini pro models"
    assert any("o3-pro" in model for model in pro_model_strings), "Should contain O3 pro models"


def test_provider_model_completeness():
    """Test that all providers have their key models available."""
    # OpenAI - check key models exist
    openai_models = set(get_provider_models("openai"))
    assert "gpt-5" in openai_models, "GPT-5 should be available"
    assert any("o3-mini" in model for model in openai_models), "O3-mini should be available"
    assert any("gpt-4o" in model for model in openai_models), "GPT-4o should be available"

    # Anthropic - check key models exist
    anthropic_models = set(get_provider_models("anthropic"))
    assert any("claude-sonnet-4" in model for model in anthropic_models), "Claude-4 should be available"
    assert any("claude-3" in model for model in anthropic_models), "Claude-3 should be available"

    # Google - check key models exist
    google_models = set(get_provider_models("google"))
    assert any("gemini-2" in model for model in google_models), "Gemini-2 should be available"
    assert any("gemini" in model and "flash" in model for model in google_models), "Gemini flash should be available"

    # xAI - check key models exist
    xai_models = set(get_provider_models("xai"))
    assert any("grok" in model.lower() for model in xai_models), "Grok models should be available"
