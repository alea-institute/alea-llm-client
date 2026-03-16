import os
from typing import cast
from unittest.mock import patch

import pytest
from conftest import PydanticTestModel

from alea_llm_client import AnthropicModel, JSONModelResponse, ModelResponse
from alea_llm_client.core.exceptions import (
    ALEAAuthenticationError,
    ALEARetryExhaustedError,
)


def test_simple_addition(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]
    response: ModelResponse = anthropic_model.chat(messages=prompt)

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


# New tests for Claude 4 and thinking capabilities
def test_claude4_basic_functionality():
    """Test Claude 4 basic functionality with new default model."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Explain the concept of recursion in programming.",
        },
    ]

    response: ModelResponse = model.chat(messages=prompt)

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0
    assert "recursion" in response.text.lower()


def test_claude4_thinking_enabled():
    """Test Claude 4 thinking capabilities with thinking enabled."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Solve this step by step: What is 127 * 83? Show your reasoning.",
        },
    ]

    response: ModelResponse = model.chat(
        messages=prompt,
        thinking=True,  # Enable thinking
        max_tokens=2000,
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0

    # Check if thinking content is available
    if response.thinking:
        assert len(response.thinking) > 0
        print(f"Thinking content length: {len(response.thinking)}")
        print(f"Response text length: {len(response.text)}")

    # Check content blocks
    assert isinstance(response.content_blocks, list)
    if response.content_blocks:
        print(f"Content blocks: {len(response.content_blocks)}")
        for i, block in enumerate(response.content_blocks):
            print(f"Block {i}: type={block.get('type')}")


def test_claude4_thinking_with_config():
    """Test Claude 4 thinking with custom configuration."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Analyze the pros and cons of microservices architecture. Think carefully about this.",
        },
    ]

    thinking_config = {"enabled": True, "budget_tokens": 1800}

    response: ModelResponse = model.chat(messages=prompt, thinking=thinking_config, max_tokens=3000)

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0

    # Thinking should be available
    if response.thinking:
        assert len(response.thinking) > 0
        print("Thinking budget used effectively")


def test_claude4_thinking_disabled():
    """Test Claude 4 with thinking explicitly disabled."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]

    response: ModelResponse = model.chat(
        messages=prompt,
        thinking=False,  # Explicitly disable thinking
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0

    # Thinking should not be present
    assert response.thinking is None or response.thinking == ""


def test_claude4_thinking_disabled_with_config():
    """Test Claude 4 with thinking disabled via config."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Simple question: what color is the sky?",
        },
    ]

    thinking_config = {"enabled": False}

    response: ModelResponse = model.chat(messages=prompt, thinking=thinking_config)

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0
    assert response.thinking is None or response.thinking == ""


def test_complex_reasoning_with_thinking():
    """Test complex reasoning task that benefits from thinking."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Design a distributed system for a social media platform. Consider scalability, consistency, and availability trade-offs. Think through this systematically.",
        },
    ]

    response: ModelResponse = model.chat(
        messages=prompt,
        thinking={"enabled": True, "budget_tokens": 2000},
        max_tokens=4000,
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0

    # Should have substantial content for complex reasoning
    assert len(response.text) > 100  # Expect detailed response

    if response.thinking:
        # Should show reasoning process
        assert len(response.thinking) > 50
        print(f"Complex reasoning - Thinking: {len(response.thinking)} chars, Response: {len(response.text)} chars")


def test_thinking_budget_validation():
    """Test that thinking budget validation works."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    # Test budget too small (should raise ValueError)
    with pytest.raises(ALEARetryExhaustedError):
        model.chat(
            messages=[{"role": "user", "content": "Test"}],
            thinking={"enabled": True, "budget_tokens": 500},
        )

    # Test budget >= max_tokens (should raise ValueError)
    with pytest.raises(ALEARetryExhaustedError):
        model.chat(
            messages=[{"role": "user", "content": "Test"}],
            thinking={"enabled": True, "budget_tokens": 1500},
            max_tokens=1500,
        )


def test_thinking_parameter_validation():
    """Test thinking parameter validation."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    # Test invalid thinking parameter type
    with pytest.raises(ALEARetryExhaustedError):
        model.chat(messages=[{"role": "user", "content": "Test"}], thinking="invalid")


def test_claude_3_7_thinking():
    """Test thinking with a current model (3.7 is deprecated, use sonnet 4.6)."""
    model = AnthropicModel(model="claude-sonnet-4-6")

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Explain the mathematical proof that there are infinitely many prime numbers.",
        },
    ]

    response: ModelResponse = model.chat(messages=prompt, thinking=True, max_tokens=2500)

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0

    # Thinking should be available
    if response.thinking:
        assert len(response.thinking) > 0
        print("Thinking capability confirmed")


def test_content_blocks_parsing():
    """Test that content blocks are properly parsed and preserved."""
    model = AnthropicModel(model="claude-sonnet-4-20250514")

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Create a simple Python function and explain how it works.",
        },
    ]

    response: ModelResponse = model.chat(
        messages=prompt,
        thinking=True,
        max_tokens=2000,  # Ensure max_tokens > thinking budget
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert isinstance(response.content_blocks, list)

    # Should have at least one content block
    if response.content_blocks:
        # Check that content blocks have expected structure
        for block in response.content_blocks:
            assert isinstance(block, dict)
            assert "type" in block
            block_type = block.get("type")
            print(f"Content block type: {block_type}")

            if block_type == "text":
                assert "text" in block
            elif block_type == "thinking":
                assert "thinking" in block or "text" in block


# def test with ignore_cache=True
def test_simple_addition_ignore_cache(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]
    response: ModelResponse = anthropic_model.chat(messages=prompt, ignore_cache=True)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


def test_json_response(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Alice who is 30 years old.",
        },
    ]
    response: JSONModelResponse = anthropic_model.json(messages=prompt, system="Respond in JSON.")

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("name") == "Alice"
    assert response.data.get("age") == 30


def test_format_with_system_prompt(anthropic_model: AnthropicModel):
    messages: list[dict[str, str]] = [
        {"role": "user", "content": "Hello"},
    ]
    system_prompt: str = "You are a helpful assistant."
    formatted_messages: list[dict[str, str]] = anthropic_model.format(
        args=[], kwargs={"messages": messages, "system": system_prompt}
    )

    assert len(formatted_messages) == 1
    assert formatted_messages[0] == messages[0]


def test_chat_with_system_prompt(anthropic_model: AnthropicModel):
    messages: list[dict[str, str]] = [
        {"role": "user", "content": "Hello"},
    ]
    system_prompt: str = "You are a helpful assistant."
    response: ModelResponse = anthropic_model.chat(messages=messages, system=system_prompt)

    assert isinstance(response, ModelResponse)
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response, "text")
    assert isinstance(response.text, str)


def test_unset_api_key():
    from pathlib import Path

    original_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    os.environ["ANTHROPIC_API_KEY"] = ""

    # Temporarily rename the key file if it exists
    key_path = Path.home() / ".alea" / "keys" / "anthropic"
    temp_path = key_path.with_suffix(".tmp")
    key_exists = key_path.exists()
    if key_exists:
        key_path.rename(temp_path)

    try:
        with pytest.raises(ValueError):
            AnthropicModel()
    finally:
        # Restore the key file
        if key_exists:
            temp_path.rename(key_path)
        os.environ["ANTHROPIC_API_KEY"] = original_key


def test_formatter_identity():
    def formatter(args, kwargs):
        if len(args) > 0:
            return args[0]
        else:
            return kwargs.pop("messages")

    anthropic_model = AnthropicModel(formatter=formatter)
    response: ModelResponse = anthropic_model.chat(messages=[{"role": "user", "content": "2+2=?"}])
    assert "4" in response.text or "four" in response.text.lower()


@pytest.mark.asyncio
async def test_chat_async(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 3 + 3?",
        },
    ]
    response: ModelResponse = await anthropic_model.chat_async(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "6" in content, f"Expected '6' in the response, but got: {content}"


@pytest.mark.asyncio
async def test_json_async(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'city' and 'population' for New York City with 8 million people.",
        },
    ]
    response: JSONModelResponse = await anthropic_model.json_async(messages=prompt, system="Respond in JSON.")

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("city") == "New York City"
    assert response.data.get("population") == 8000000


def test_retry_wrapper_exhaustion(anthropic_model: AnthropicModel):
    def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(anthropic_model, "_chat", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            anthropic_model.chat("Test message", ignore_cache=True)


@pytest.mark.asyncio
async def test_retry_wrapper_async_exhaustion(anthropic_model: AnthropicModel):
    async def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(anthropic_model, "_chat_async", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            await anthropic_model.chat_async("Test message", ignore_cache=True)


def test_cache_functionality(anthropic_model: AnthropicModel, tmp_path):
    anthropic_model.cache_path = tmp_path
    anthropic_model.ignore_cache = False  # Enable caching for this test

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]

    # First call should hit the API
    response1: ModelResponse = anthropic_model.chat(messages=prompt)

    # Second call should use cached response
    response2: ModelResponse = anthropic_model.chat(messages=prompt)

    assert response1.text == response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


@pytest.mark.asyncio
async def test_cache_functionality_async(anthropic_model: AnthropicModel, tmp_path):
    anthropic_model.cache_path = tmp_path
    anthropic_model.ignore_cache = False  # Enable caching for this test

    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of Germany?",
        },
    ]

    response1: ModelResponse = await anthropic_model.chat_async(messages=prompt)
    response2: ModelResponse = await anthropic_model.chat_async(messages=prompt)

    assert "Berlin" in response1.text and "Berlin" in response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


def test_pydantic_response(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Bob who is 25 years old.",
        },
    ]
    response = cast(
        PydanticTestModel,
        anthropic_model.pydantic(messages=prompt, system="Respond in JSON.", pydantic_model=PydanticTestModel),
    )

    assert isinstance(response, PydanticTestModel)
    assert response.name == "Bob"
    assert response.age == 25


@pytest.mark.asyncio
async def test_pydantic_async(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Charlie who is 35 years old.",
        },
    ]
    response = cast(
        PydanticTestModel,
        await anthropic_model.pydantic_async(
            messages=prompt, system="Respond in JSON.", pydantic_model=PydanticTestModel
        ),
    )

    assert isinstance(response, PydanticTestModel)
    assert response.name == "Charlie"
    assert response.age == 35


def test_invalid_pydantic_response(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with age as the string 'twenty-five' instead of a number.",
        },
    ]
    with pytest.raises(Exception):
        anthropic_model.pydantic(
            messages=prompt,
            system="Respond in JSON with age as the literal string 'twenty-five'.",
            pydantic_model=PydanticTestModel,
        )


def test_missing_pydantic_model(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object.",
        },
    ]
    with pytest.raises(ALEARetryExhaustedError):
        anthropic_model.pydantic(messages=prompt, system="Respond in JSON.")


def test_custom_model(anthropic_model: AnthropicModel):
    custom_model = "claude-sonnet-4-5"
    anthropic_model.model = custom_model
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Hello",
        },
    ]
    response: ModelResponse = anthropic_model.chat(messages=prompt)
    assert response.metadata.get("model") == custom_model


def test_max_tokens(anthropic_model: AnthropicModel):
    prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": "Write a very long story.",
        },
    ]
    max_tokens = 50
    response: ModelResponse = anthropic_model.chat(messages=prompt, max_tokens=max_tokens)
    assert len(response.text.split()) <= max_tokens


def test_invalid_api_key():
    # unset env value
    model = AnthropicModel(api_key="abc123")
    with pytest.raises(ALEAAuthenticationError):
        print(model.chat(messages=[{"role": "user", "content": "Hello"}], ignore_cache=True))


def test_model_response_to_dict(anthropic_model: AnthropicModel):
    response: ModelResponse = anthropic_model.chat(messages=[{"role": "user", "content": "Hello"}])
    response_dict = response.to_dict()
    assert isinstance(response_dict, dict)
    assert "choices" in response_dict
    assert "metadata" in response_dict
    assert "text" in response_dict


def test_model_response_to_json(anthropic_model: AnthropicModel):
    response: ModelResponse = anthropic_model.chat(messages=[{"role": "user", "content": "Hello"}])
    response_json = response.to_json()
    assert isinstance(response_json, str)
    assert "choices" in response_json
    assert "metadata" in response_json
    assert "text" in response_json


# Tests for New Claude Model Generations
def test_claude_4_models():
    """Test that Claude 4 models are available and working."""
    from alea_llm_client.llms.constants import (
        CLAUDE_4_MODELS,
        is_claude_4,
        supports_thinking,
    )

    # Test key Claude 4 models (using real API model names)
    test_models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-opus-4-1-20250805",
    ]

    for model_name in test_models:
        assert model_name in CLAUDE_4_MODELS, f"{model_name} should be in CLAUDE_4_MODELS"
        assert supports_thinking(model_name), f"{model_name} should support thinking"
        assert is_claude_4(model_name), f"{model_name} should be identified as Claude 4"

        # Test model instantiation
        model = AnthropicModel(model=model_name)
        assert model.model == model_name


def test_claude_3_7_models():
    """Test that Claude 3.7 models are available."""
    from alea_llm_client.llms.constants import (
        CLAUDE_3_7_MODELS,
        is_claude_3_7,
        supports_thinking,
    )

    test_models = ["claude-3-7-sonnet-20250219"]

    for model_name in test_models:
        assert model_name in CLAUDE_3_7_MODELS, f"{model_name} should be in CLAUDE_3_7_MODELS"
        assert supports_thinking(model_name), f"{model_name} should support thinking"
        assert is_claude_3_7(model_name), f"{model_name} should be identified as Claude 3.7"

        # Test model instantiation
        model = AnthropicModel(model=model_name)
        assert model.model == model_name


def test_claude_3_5_models():
    """Test that Claude 3.5 models are available."""
    from alea_llm_client.llms.constants import CLAUDE_3_5_MODELS

    test_models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022",
    ]

    for model_name in test_models:
        assert model_name in CLAUDE_3_5_MODELS, f"{model_name} should be in CLAUDE_3_5_MODELS"

        # Test model instantiation
        model = AnthropicModel(model=model_name)
        assert model.model == model_name


def test_model_family_detection():
    """Test model family detection functions."""
    from alea_llm_client.llms.constants import get_model_family, get_model_size

    # Test Claude 4 family
    assert get_model_family("claude-sonnet-4-20250514") == "claude-4"
    assert get_model_family("claude-opus-4-20250514") == "claude-4"

    # Test Claude 3.7 family
    assert get_model_family("claude-3-7-sonnet-20250219") == "claude-3.7"

    # Test Claude 3.5 family
    assert get_model_family("claude-3-5-sonnet-20241022") == "claude-3.5"
    assert get_model_family("claude-3-5-haiku-20241022") == "claude-3.5"

    # Test Claude 3 family
    assert get_model_family("claude-3-opus-20240229") == "claude-3"

    # Test model size detection
    assert get_model_size("claude-sonnet-4-20250514") == "sonnet"
    assert get_model_size("claude-opus-4-20250514") == "opus"
    assert get_model_size("claude-3-5-haiku-latest") == "haiku"


def test_model_categorization():
    """Test model categorization by size."""
    from alea_llm_client.llms.constants import HAIKU_MODELS, OPUS_MODELS, SONNET_MODELS

    # Test Sonnet models
    sonnet_test_models = [
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
    ]
    for model in sonnet_test_models:
        assert model in SONNET_MODELS, f"{model} should be in SONNET_MODELS"

    # Test Opus models
    opus_test_models = ["claude-opus-4-20250514", "claude-3-opus-20240229"]
    for model in opus_test_models:
        assert model in OPUS_MODELS, f"{model} should be in OPUS_MODELS"

    # Test Haiku models
    haiku_test_models = ["claude-3-5-haiku-20241022", "claude-3-haiku-20240307"]
    for model in haiku_test_models:
        assert model in HAIKU_MODELS, f"{model} should be in HAIKU_MODELS"


def test_thinking_configuration_helpers():
    """Test thinking configuration helper functions."""
    from alea_llm_client.llms.constants import (
        DEFAULT_THINKING_CONFIG,
        THINKING_DISABLED_CONFIG,
        create_thinking_config,
        validate_thinking_config,
    )

    # Test create_thinking_config
    config_enabled = create_thinking_config(enabled=True, budget_tokens=1800)
    assert config_enabled["enabled"] is True
    assert config_enabled["budget_tokens"] == 1800

    config_disabled = create_thinking_config(enabled=False)
    assert config_disabled["enabled"] is False
    assert "budget_tokens" not in config_disabled

    # Test validation with invalid budget
    with pytest.raises(ValueError):
        create_thinking_config(enabled=True, budget_tokens=500)  # Too small

    # Test validate_thinking_config
    assert validate_thinking_config({"enabled": False}, max_tokens=2000) is True
    assert validate_thinking_config({"enabled": True, "budget_tokens": 1500}, max_tokens=2000) is True
    assert (
        validate_thinking_config({"enabled": True, "budget_tokens": 500}, max_tokens=2000) is False
    )  # Budget too small
    assert (
        validate_thinking_config({"enabled": True, "budget_tokens": 2000}, max_tokens=2000) is False
    )  # Budget >= max_tokens

    # Test default configs
    assert DEFAULT_THINKING_CONFIG["enabled"] is True
    assert DEFAULT_THINKING_CONFIG["budget_tokens"] >= 1024
    assert THINKING_DISABLED_CONFIG["enabled"] is False


def test_model_defaults():
    """Test that default model selections are valid."""
    from alea_llm_client.llms.constants import (
        ANTHROPIC_DEFAULT_CHAT_MODEL,
        ANTHROPIC_DEFAULT_FAST_MODEL,
        ANTHROPIC_DEFAULT_REASONING_MODEL,
        ANTHROPIC_DEFAULT_THINKING_MODEL,
        CLAUDE_4_MODELS,
    )

    # Test defaults are in their respective model families
    assert ANTHROPIC_DEFAULT_CHAT_MODEL in CLAUDE_4_MODELS, (
        f"ANTHROPIC_DEFAULT_CHAT_MODEL {ANTHROPIC_DEFAULT_CHAT_MODEL} should be a Claude 4 model"
    )
    assert ANTHROPIC_DEFAULT_REASONING_MODEL in CLAUDE_4_MODELS, (
        f"ANTHROPIC_DEFAULT_REASONING_MODEL {ANTHROPIC_DEFAULT_REASONING_MODEL} should be a Claude 4 model"
    )
    assert ANTHROPIC_DEFAULT_FAST_MODEL in CLAUDE_4_MODELS, (
        f"ANTHROPIC_DEFAULT_FAST_MODEL {ANTHROPIC_DEFAULT_FAST_MODEL} should be a Claude 4 model"
    )
    assert ANTHROPIC_DEFAULT_THINKING_MODEL in CLAUDE_4_MODELS, (
        f"ANTHROPIC_DEFAULT_THINKING_MODEL {ANTHROPIC_DEFAULT_THINKING_MODEL} should be a Claude 4 model"
    )

    # Test that defaults can be instantiated
    chat_model = AnthropicModel(model=ANTHROPIC_DEFAULT_CHAT_MODEL)
    reasoning_model = AnthropicModel(model=ANTHROPIC_DEFAULT_REASONING_MODEL)
    fast_model = AnthropicModel(model=ANTHROPIC_DEFAULT_FAST_MODEL)
    thinking_model = AnthropicModel(model=ANTHROPIC_DEFAULT_THINKING_MODEL)

    assert chat_model.model == ANTHROPIC_DEFAULT_CHAT_MODEL
    assert reasoning_model.model == ANTHROPIC_DEFAULT_REASONING_MODEL
    assert fast_model.model == ANTHROPIC_DEFAULT_FAST_MODEL
    assert thinking_model.model == ANTHROPIC_DEFAULT_THINKING_MODEL


def test_all_claude_models_list():
    """Test that ALL_CLAUDE_MODELS contains all model families."""
    from alea_llm_client.llms.constants import (
        ALL_CLAUDE_MODELS,
        CLAUDE_3_5_MODELS,
        CLAUDE_3_7_MODELS,
        CLAUDE_3_MODELS,
        CLAUDE_4_MODELS,
    )

    # Test that all model families are included
    for model in CLAUDE_4_MODELS:
        assert model in ALL_CLAUDE_MODELS, f"Claude 4 model {model} should be in ALL_CLAUDE_MODELS"

    for model in CLAUDE_3_7_MODELS:
        assert model in ALL_CLAUDE_MODELS, f"Claude 3.7 model {model} should be in ALL_CLAUDE_MODELS"

    for model in CLAUDE_3_5_MODELS:
        assert model in ALL_CLAUDE_MODELS, f"Claude 3.5 model {model} should be in ALL_CLAUDE_MODELS"

    for model in CLAUDE_3_MODELS:
        assert model in ALL_CLAUDE_MODELS, f"Claude 3 model {model} should be in ALL_CLAUDE_MODELS"


def test_thinking_enabled_models():
    """Test that thinking-enabled models are correctly identified."""
    from alea_llm_client.llms.constants import (
        CLAUDE_3_7_MODELS,
        CLAUDE_4_MODELS,
        THINKING_ENABLED_MODELS,
    )

    # Claude 4 models that support thinking (excludes haiku which doesn't)
    for model in CLAUDE_4_MODELS:
        if "haiku" not in model:
            assert model in THINKING_ENABLED_MODELS, f"Claude 4 model {model} should support thinking"

    for model in CLAUDE_3_7_MODELS:
        assert model in THINKING_ENABLED_MODELS, f"Claude 3.7 model {model} should support thinking"


# ========== New model tests ==========


def test_claude_4_6_models():
    """Test that Claude 4.6 models are in the model lists."""
    from alea_llm_client.llms.constants import CLAUDE_4_MODELS, OPUS_MODELS, SONNET_MODELS

    assert "claude-opus-4-6" in CLAUDE_4_MODELS
    assert "claude-sonnet-4-6" in CLAUDE_4_MODELS
    assert "claude-opus-4-6" in OPUS_MODELS
    assert "claude-sonnet-4-6" in SONNET_MODELS


def test_claude_4_5_models():
    """Test that Claude 4.5 models are in the model lists."""
    from alea_llm_client.llms.constants import CLAUDE_4_MODELS, HAIKU_MODELS, OPUS_MODELS, SONNET_MODELS

    assert "claude-opus-4-5" in CLAUDE_4_MODELS
    assert "claude-opus-4-5-20251101" in CLAUDE_4_MODELS
    assert "claude-sonnet-4-5" in CLAUDE_4_MODELS
    assert "claude-sonnet-4-5-20250929" in CLAUDE_4_MODELS
    assert "claude-haiku-4-5" in CLAUDE_4_MODELS
    assert "claude-haiku-4-5-20251001" in CLAUDE_4_MODELS

    assert "claude-opus-4-5" in OPUS_MODELS
    assert "claude-sonnet-4-5" in SONNET_MODELS
    assert "claude-haiku-4-5" in HAIKU_MODELS


def test_claude_4_6_basic():
    """Test Claude 4.6 Sonnet basic functionality."""
    model = AnthropicModel(model="claude-sonnet-4-6")

    response: ModelResponse = model.chat(
        messages=[{"role": "user", "content": "What is 5 + 3?"}],
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert "8" in response.text


def test_claude_4_5_sonnet():
    """Test Claude 4.5 Sonnet model."""
    model = AnthropicModel(model="claude-sonnet-4-5")

    response: ModelResponse = model.chat(
        messages=[{"role": "user", "content": "What is 9 - 4?"}],
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert "5" in response.text


def test_claude_haiku_4_5():
    """Test Claude Haiku 4.5 model."""
    model = AnthropicModel(model="claude-haiku-4-5")

    response: ModelResponse = model.chat(
        messages=[{"role": "user", "content": "What is 2 * 3?"}],
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert "6" in response.text


# ========== service_tier and output_config tests ==========


def test_service_tier_auto():
    """Test service_tier='auto' parameter."""
    model = AnthropicModel(model="claude-sonnet-4-6")

    response: ModelResponse = model.chat(
        messages=[{"role": "user", "content": "Say hello."}],
        service_tier="auto",
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0


def test_service_tier_invalid():
    """Test that invalid service_tier raises ValueError."""
    model = AnthropicModel(model="claude-sonnet-4-6")

    with pytest.raises(ALEARetryExhaustedError):
        model.chat(
            messages=[{"role": "user", "content": "Test"}],
            service_tier="invalid_tier",
        )


def test_output_config_effort():
    """Test output_config with effort parameter."""
    model = AnthropicModel(model="claude-sonnet-4-6")

    response: ModelResponse = model.chat(
        messages=[{"role": "user", "content": "What is the capital of Japan?"}],
        output_config={"effort": "low"},
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0


def test_output_config_invalid_effort():
    """Test that invalid output_config effort raises ValueError."""
    model = AnthropicModel(model="claude-sonnet-4-6")

    with pytest.raises(ALEARetryExhaustedError):
        model.chat(
            messages=[{"role": "user", "content": "Test"}],
            output_config={"effort": "super_duper"},
        )


# ========== Anthropic tool creation helpers tests ==========


def test_anthropic_tool_creation_helpers():
    """Unit tests for all create_anthropic_*_tool functions."""
    from alea_llm_client.llms.constants import (
        create_anthropic_bash_tool,
        create_anthropic_code_execution_tool,
        create_anthropic_computer_use_tool,
        create_anthropic_function_tool,
        create_anthropic_memory_tool,
        create_anthropic_text_editor_tool,
        create_anthropic_tool_search_bm25_tool,
        create_anthropic_tool_search_regex_tool,
        create_anthropic_web_fetch_tool,
        create_anthropic_web_search_tool,
    )

    # web_search
    ws = create_anthropic_web_search_tool()
    assert ws["type"] == "web_search_20260209"
    assert ws["name"] == "web_search"

    ws_domains = create_anthropic_web_search_tool(allowed_domains=["example.com"])
    assert ws_domains["allowed_domains"] == ["example.com"]

    # web_fetch
    wf = create_anthropic_web_fetch_tool(max_content_tokens=5000)
    assert wf["type"] == "web_fetch_20260209"
    assert wf["max_content_tokens"] == 5000

    # code_execution
    ce = create_anthropic_code_execution_tool()
    assert ce["type"] == "code_execution_20260120"

    # bash
    bash = create_anthropic_bash_tool()
    assert bash["type"] == "bash_20250124"

    # text_editor
    te = create_anthropic_text_editor_tool()
    assert te["type"] == "text_editor_20250728"

    # computer_use
    cu = create_anthropic_computer_use_tool(display_width_px=1920, display_height_px=1080)
    assert cu["type"] == "computer_20251124"
    assert cu["display_width_px"] == 1920

    cu_num = create_anthropic_computer_use_tool(display_width_px=1024, display_height_px=768, display_number=1)
    assert cu_num["display_number"] == 1

    # memory
    mem = create_anthropic_memory_tool()
    assert mem["type"] == "memory_20250818"

    # tool_search
    bm25 = create_anthropic_tool_search_bm25_tool()
    assert bm25["type"] == "tool_search_bm25_20251119"

    regex = create_anthropic_tool_search_regex_tool()
    assert regex["type"] == "tool_search_regex_20251119"

    # function tool
    fn = create_anthropic_function_tool(
        name="get_weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        description="Get weather",
    )
    assert fn["name"] == "get_weather"
    assert fn["description"] == "Get weather"


# ========== Anthropic constants tests ==========


def test_anthropic_service_tier_constants():
    """Test ANTHROPIC_SERVICE_TIER_VALUES."""
    from alea_llm_client.llms.constants import ANTHROPIC_SERVICE_TIER_VALUES

    assert ANTHROPIC_SERVICE_TIER_VALUES == ["auto", "standard_only"]


def test_output_effort_constants():
    """Test OUTPUT_EFFORT_VALUES."""
    from alea_llm_client.llms.constants import OUTPUT_EFFORT_VALUES

    assert OUTPUT_EFFORT_VALUES == ["low", "medium", "high", "max"]


def test_deprecated_model_warning():
    """Test that deprecated models emit a DeprecationWarning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        AnthropicModel(model="claude-3-5-sonnet-20241022")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
        assert "claude-sonnet-4-6" in str(w[0].message)


def test_deprecated_model_404_error():
    """Test that deprecated models give a clear error with replacement suggestion."""
    import warnings

    from alea_llm_client.core.exceptions import ALEAModelError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = AnthropicModel(model="claude-3-5-sonnet-20241022")

    with pytest.raises(ALEAModelError) as exc_info:
        model.chat(messages=[{"role": "user", "content": "Test"}])

    # The error should mention the replacement
    assert "deprecated" in str(exc_info.value).lower()
    assert "claude-sonnet-4-6" in str(exc_info.value)


def test_deprecation_replacements():
    """Test the deprecation replacement mapping."""
    from alea_llm_client.llms.constants import get_model_replacement, is_deprecated

    assert is_deprecated("claude-3-5-sonnet-20241022")
    assert get_model_replacement("claude-3-5-sonnet-20241022") == "claude-sonnet-4-6"

    assert is_deprecated("claude-3-opus-20240229")
    assert get_model_replacement("claude-3-opus-20240229") == "claude-opus-4-6"

    assert not is_deprecated("claude-sonnet-4-6")
    assert get_model_replacement("claude-sonnet-4-6") is None


def test_updated_defaults():
    """Test that defaults have been updated to latest models."""
    from alea_llm_client.llms.constants import (
        ANTHROPIC_DEFAULT_CHAT_MODEL,
        ANTHROPIC_DEFAULT_REASONING_MODEL,
        CLAUDE_4_MODELS,
    )

    assert ANTHROPIC_DEFAULT_CHAT_MODEL in CLAUDE_4_MODELS
    assert ANTHROPIC_DEFAULT_REASONING_MODEL in CLAUDE_4_MODELS
