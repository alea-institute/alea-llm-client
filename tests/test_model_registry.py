"""Tests for the model registry."""

from alea_llm_client.llms import (
    OpenAIModels,
    AnthropicModels,
    GoogleModels,
    GrokModels,
    get_all_models,
    get_provider_models,
    get_model_capabilities,
    is_model_deprecated,
    supports_feature,
    DEFAULT_MODELS,
    MODEL_SIZES,
)


def test_enum_values():
    """Test that enum values are strings."""
    assert OpenAIModels.GPT_5.value == "gpt-5"
    assert AnthropicModels.CLAUDE_SONNET_4_20250514.value == "claude-sonnet-4-20250514"
    assert GoogleModels.GEMINI_2_5_FLASH.value == "gemini-2.5-flash"
    assert GrokModels.GROK_2_1212.value == "grok-2-1212"


def test_get_all_models():
    """Test getting all models."""
    all_models = get_all_models()

    # Should have models from all providers
    assert len(all_models) > 100  # We have 76 OpenAI + 17 Anthropic + others

    # Check specific models are included
    assert "gpt-5" in all_models
    assert "claude-sonnet-4-20250514" in all_models
    assert "gemini-2.5-flash" in all_models
    assert "grok-2-1212" in all_models


def test_get_provider_models():
    """Test getting models by provider."""
    # OpenAI
    openai_models = get_provider_models("openai")
    assert len(openai_models) > 70
    assert "gpt-5" in openai_models
    assert "o3-mini" in openai_models

    # Anthropic
    anthropic_models = get_provider_models("anthropic")
    assert len(anthropic_models) > 15
    assert "claude-sonnet-4-20250514" in anthropic_models

    # Google
    google_models = get_provider_models("google")
    assert len(google_models) > 10
    assert "gemini-2.5-flash" in google_models

    # xAI/Grok (both names should work)
    grok_models = get_provider_models("xai")
    assert len(grok_models) > 5
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
    assert is_model_deprecated("gpt-3.5-turbo") is False  # Still supported
    assert is_model_deprecated("gpt-4-0314") is True  # Old dated version

    # Current models should not be deprecated
    assert is_model_deprecated("gpt-5") is False
    assert is_model_deprecated("claude-sonnet-4-20250514") is False


def test_default_models():
    """Test default model selection."""
    assert DEFAULT_MODELS["openai"] == "gpt-5-chat-latest"
    assert DEFAULT_MODELS["anthropic"] == "claude-sonnet-4-20250514"
    assert DEFAULT_MODELS["google"] == "gemini-2.0-flash-exp"
    assert DEFAULT_MODELS["xai"] == "grok-2-1212"


def test_model_sizes():
    """Test model size categorization."""
    # Check mini models
    mini_models = MODEL_SIZES["mini"]
    assert OpenAIModels.GPT_5_MINI in mini_models
    assert OpenAIModels.O3_MINI in mini_models
    assert GrokModels.GROK_3_MINI in mini_models

    # Check flash models
    flash_models = MODEL_SIZES["flash"]
    assert GoogleModels.GEMINI_2_5_FLASH in flash_models
    assert GoogleModels.GEMINI_2_0_FLASH in flash_models

    # Check pro models
    pro_models = MODEL_SIZES["pro"]
    assert GoogleModels.GEMINI_2_5_PRO in pro_models
    assert OpenAIModels.O3_PRO in pro_models


def test_model_enum_completeness():
    """Test that all important models are included."""
    # OpenAI - check key models exist
    openai_values = {m.value for m in OpenAIModels}
    assert "gpt-5" in openai_values
    assert "o3-mini" in openai_values
    assert "gpt-4o" in openai_values
    assert "gpt-4o-audio-preview" in openai_values

    # Anthropic - check key models exist
    anthropic_values = {m.value for m in AnthropicModels}
    assert "claude-sonnet-4-20250514" in anthropic_values
    assert "claude-3-7-sonnet-latest" in anthropic_values
    assert "claude-3-5-sonnet-latest" in anthropic_values

    # Google - check key models exist
    google_values = {m.value for m in GoogleModels}
    assert "gemini-2.5-flash" in google_values
    assert "gemini-2.0-flash-exp" in google_values

    # Grok - check key models exist
    grok_values = {m.value for m in GrokModels}
    assert "grok-4" in grok_values
    assert "grok-2-1212" in grok_values
