"""Tests for the model capabilities matrix."""

from alea_llm_client.llms import (
    ModelStatus,
    ModelTier,
    COMPLETE_MODEL_CAPABILITIES,
    get_models_with_context_window_gte,
    get_models_with_capability,
    get_models_by_tier,
    get_model_details,
    compare_models,
    filter_models,
)


def test_capability_matrix_completeness():
    """Test that major models are in the capability matrix."""
    # Check key models exist
    assert "gpt-5" in COMPLETE_MODEL_CAPABILITIES
    assert "claude-sonnet-4-20250514" in COMPLETE_MODEL_CAPABILITIES
    assert "gemini-2.5-pro" in COMPLETE_MODEL_CAPABILITIES
    assert "grok-4" in COMPLETE_MODEL_CAPABILITIES

    # Should have at least 50 models total
    assert len(COMPLETE_MODEL_CAPABILITIES) >= 50


def test_context_window_query():
    """Test querying models by context window size."""
    # Find models with 1M+ context
    large_context = get_models_with_context_window_gte(1000000)
    assert len(large_context) > 0

    # Verify they actually have large context
    for model in large_context:
        details = get_model_details(model)
        assert details is not None and details.context_window >= 1000000

    # Check specific models are included
    assert "gpt-4.1" in large_context
    assert "gemini-2.5-pro" in large_context


def test_capability_queries():
    """Test querying models by capabilities."""
    # Test tools capability
    tool_models = get_models_with_capability("tools")
    assert len(tool_models) > 20
    assert "gpt-5" in tool_models
    assert "claude-sonnet-4-20250514" in tool_models

    # Test vision capability
    vision_models = get_models_with_capability("vision")
    assert len(vision_models) > 15
    assert "gpt-4o" in vision_models

    # Test audio capability (combined)
    audio_models = get_models_with_capability("audio")
    assert len(audio_models) > 5
    assert "gpt-4o-audio-preview" in audio_models

    # Test reasoning capability
    reasoning_models = get_models_with_capability("reasoning")
    assert len(reasoning_models) > 5
    assert "o3-mini" in reasoning_models
    assert "grok-4" in reasoning_models

    # Test thinking capability
    thinking_models = get_models_with_capability("thinking")
    assert len(thinking_models) > 5
    assert "claude-sonnet-4-20250514" in thinking_models
    assert "gemini-2.5-pro" in thinking_models


def test_tier_queries():
    """Test querying models by tier."""
    # Test mini tier
    mini_models = get_models_by_tier(ModelTier.MINI)
    assert len(mini_models) > 5
    assert "gpt-5-mini" in mini_models
    assert "o3-mini" in mini_models

    # Test pro tier
    pro_models = get_models_by_tier(ModelTier.PRO)
    assert len(pro_models) > 3
    assert "gemini-2.5-pro" in pro_models


def test_model_details():
    """Test getting detailed model information."""
    # Test GPT-5 details
    gpt5 = get_model_details("gpt-5")
    assert gpt5 is not None
    assert gpt5.provider == "openai"
    assert gpt5.family == "gpt-5"
    assert gpt5.tier == ModelTier.STANDARD
    assert gpt5.context_window == 128000
    assert gpt5.supports_tools is True
    assert gpt5.supports_grammar is True
    assert gpt5.supports_reasoning is False

    # Test Claude 4 details
    claude4 = get_model_details("claude-sonnet-4-20250514")
    assert claude4 is not None
    assert claude4.provider == "anthropic"
    assert claude4.context_window == 1000000
    assert claude4.supports_thinking is True
    assert claude4.thinking_budget_min == 1024

    # Test O-series details
    o3 = get_model_details("o3-mini")
    assert o3 is not None
    assert o3.supports_reasoning is True
    assert o3.requires_max_completion_tokens is True
    assert o3.max_completion_tokens == 100000


def test_model_comparison():
    """Test comparing multiple models."""
    models = ["gpt-5", "claude-sonnet-4-20250514", "gemini-2.5-pro"]
    comparison = compare_models(models)

    assert len(comparison) == 3

    # Check GPT-5
    assert comparison["gpt-5"]["provider"] == "openai"
    assert comparison["gpt-5"]["context_window"] == 128000
    assert comparison["gpt-5"]["tools"] is True

    # Check Claude
    assert comparison["claude-sonnet-4-20250514"]["thinking"] is True

    # Check Gemini
    assert comparison["gemini-2.5-pro"]["context_window"] == 2000000


def test_advanced_filtering():
    """Test advanced filtering with multiple criteria."""
    # Find efficient models for production
    efficient = filter_models(
        min_context=100000,
        capabilities=["tools"],
        tiers=[ModelTier.MINI, ModelTier.FLASH],
        exclude_deprecated=True,
    )

    assert len(efficient) > 0
    assert "gpt-5-mini" in efficient
    assert "gemini-2.5-flash" in efficient

    # Verify all match criteria
    for model in efficient:
        details = get_model_details(model)
        assert details is not None
        assert details.context_window >= 100000
        assert details.supports_tools is True
        assert details.tier in [ModelTier.MINI, ModelTier.FLASH]
        assert details.status not in [ModelStatus.DEPRECATED, ModelStatus.LEGACY]


def test_provider_filtering():
    """Test filtering by provider."""
    # Filter for OpenAI models with tools
    openai_tools = filter_models(
        providers=["openai"], capabilities=["tools"], exclude_deprecated=True
    )

    assert len(openai_tools) > 10
    assert all("gpt" in m or m.startswith("o") for m in openai_tools)

    # Filter for Anthropic models
    anthropic_models = filter_models(providers=["anthropic"], min_context=200000)

    assert len(anthropic_models) > 5
    assert all("claude" in m for m in anthropic_models)


def test_output_token_queries():
    """Test querying by output token limits."""
    # Find models with large output capacity
    large_output = filter_models(min_output=32000, exclude_deprecated=True)

    assert len(large_output) > 5

    # Check O-series models with max_completion_tokens
    assert "o3-mini" in large_output
    assert "o1" in large_output

    # Check Claude models
    assert "claude-sonnet-4-20250514" in large_output


def test_special_requirements():
    """Test models with special requirements."""
    # Find models requiring max_completion_tokens
    for model_name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        if caps.requires_max_completion_tokens:
            assert model_name.startswith(("o1", "o3", "o4"))
            assert caps.max_completion_tokens is not None
            assert caps.supports_reasoning is True

    # Find models with thinking budgets
    for model_name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        if caps.supports_thinking:
            if "claude" in model_name:
                assert (
                    caps.thinking_budget_min is not None
                    and caps.thinking_budget_min >= 1024
                )
            if "gemini" in model_name:
                assert (
                    caps.thinking_budget_min is not None
                    and caps.thinking_budget_min >= 4000
                )


def test_modality_support():
    """Test input/output modality queries."""
    # Find truly multimodal models
    multimodal = []
    for model_name, caps in COMPLETE_MODEL_CAPABILITIES.items():
        modalities = set(caps.input_modalities + caps.output_modalities)
        if len(modalities) >= 3:  # text + 2 others
            multimodal.append(model_name)

    assert len(multimodal) > 5
    assert any("gemini" in m for m in multimodal)
    assert any("gpt-4o-audio" in m for m in multimodal)
