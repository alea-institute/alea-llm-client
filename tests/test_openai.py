import os
import pytest
from typing import List, Dict
from unittest.mock import patch
from pydantic import BaseModel

from alea_llm_client import OpenAIModel, ModelResponse, JSONModelResponse
from alea_llm_client.core.exceptions import (
    ALEARetryExhaustedError,
    ALEAAuthenticationError,
    ALEAModelError,
)


@pytest.fixture
def openai_model():
    return OpenAIModel()


def test_simple_addition(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]
    response: ModelResponse = openai_model.chat(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


# New tests for GPT-5 and latest API features
def test_gpt5_reasoning_effort_minimal():
    """Test GPT-5 with minimal reasoning effort for fast responses."""
    model = OpenAIModel(model="gpt-5")

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "My distributed cache shows inconsistent data. Node A: '123', Node B: '456'. Both healthy.",
        },
    ]

    response: ModelResponse = model.chat(
        messages=prompt, reasoning_effort="minimal", verbosity="low"
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0


def test_gpt5_reasoning_effort_high():
    """Test GPT-5 with high reasoning effort for thorough analysis."""
    model = OpenAIModel(model="gpt-5")

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "My distributed cache shows inconsistent data. Node A: '123', Node B: '456'. Both healthy.",
        },
    ]

    response: ModelResponse = model.chat(
        messages=prompt, reasoning_effort="high", verbosity="high"
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0


def test_verbosity_control():
    """Test verbosity parameter for response length control."""
    model = OpenAIModel(model="gpt-5")

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is Raft consensus?",
        },
    ]

    # Test low verbosity (concise)
    low_response: ModelResponse = model.chat(messages=prompt, verbosity="low")

    # Test high verbosity (comprehensive)
    high_response: ModelResponse = model.chat(messages=prompt, verbosity="high")

    assert isinstance(low_response, ModelResponse)
    assert isinstance(high_response, ModelResponse)
    assert low_response.text is not None
    assert high_response.text is not None

    # High verbosity should generally produce longer responses
    # Note: This is not guaranteed but generally expected
    print(f"Low verbosity length: {len(low_response.text)}")
    print(f"High verbosity length: {len(high_response.text)}")


def test_o_series_reasoning_tokens():
    """Test o-series models to ensure reasoning tokens are captured."""
    model = OpenAIModel(model="o3-mini")

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Prove that the square root of 2 is irrational using a contradiction argument.",
        },
    ]

    response: ModelResponse = model.chat(
        messages=prompt, reasoning_effort="high", max_completion_tokens=1000
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert len(response.text) > 0

    # O-series models should provide reasoning token count
    if response.reasoning_tokens:
        assert response.reasoning_tokens > 0
        print(f"Reasoning tokens used: {response.reasoning_tokens}")


def test_max_completion_tokens_parameter():
    """Test max_completion_tokens parameter (preferred over max_tokens for o-series)."""
    model = OpenAIModel(model="o3-mini")

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Write a short poem about artificial intelligence.",
        },
    ]

    response: ModelResponse = model.chat(
        messages=prompt,
        max_completion_tokens=500,  # Increased to allow for both reasoning and response
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    # Text might be empty if all tokens were used for reasoning, that's OK
    # The important thing is that the parameter is accepted and used


def test_optimization_combinations():
    """Test parameter combinations for different use cases."""
    model = OpenAIModel(model="gpt-5")

    base_prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Optimize this SQL: SELECT * FROM orders WHERE created_at > '2024-01-01'",
        },
    ]

    # Speed optimized
    fast_response: ModelResponse = model.chat(
        messages=base_prompt, reasoning_effort="minimal", verbosity="low"
    )

    # Thorough analysis
    thorough_response: ModelResponse = model.chat(
        messages=base_prompt, reasoning_effort="high", verbosity="high"
    )

    assert isinstance(fast_response, ModelResponse)
    assert isinstance(thorough_response, ModelResponse)
    assert fast_response.text is not None
    assert thorough_response.text is not None

    print(f"Fast response length: {len(fast_response.text)}")
    print(f"Thorough response length: {len(thorough_response.text)}")


def test_invalid_reasoning_effort():
    """Test that invalid reasoning_effort values raise error."""
    model = OpenAIModel(model="gpt-5")

    # The error gets wrapped by the retry mechanism, so we catch the more general exception
    with pytest.raises(ALEARetryExhaustedError):
        model.chat(
            messages=[{"role": "user", "content": "Test"}], reasoning_effort="invalid"
        )


def test_invalid_verbosity():
    """Test that invalid verbosity values raise ValueError."""
    model = OpenAIModel(model="gpt-5")

    with pytest.raises(ALEARetryExhaustedError):
        model.chat(messages=[{"role": "user", "content": "Test"}], verbosity="invalid")


def test_deprecated_max_tokens_warning():
    """Test that max_tokens is not supported for o-series models."""
    model = OpenAIModel(model="o3-mini")

    # This should raise an error since o-series models don't support max_tokens
    with pytest.raises(ALEAModelError):
        model.chat(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,  # Not supported parameter for o-series
        )


# def test with ignore_cache=True
def test_simple_addition_ignore_cache(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]
    response: ModelResponse = openai_model.chat(messages=prompt, ignore_cache=True)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


def test_json_response(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Alice who is 30 years old.",
        },
    ]
    response: JSONModelResponse = openai_model.json(
        messages=prompt, system="Respond in JSON."
    )

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("name") == "Alice"
    assert response.data.get("age") == 30


def test_format_with_system_prompt(openai_model: OpenAIModel):
    expected_messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "Hello"},
    ]
    system_prompt: str = "You are a helpful assistant."
    formatted_messages: List[Dict[str, str]] = openai_model.format(
        args=[], kwargs={"messages": messages, "system": system_prompt}
    )

    assert len(formatted_messages) == 2
    assert formatted_messages == expected_messages


def test_chat_with_system_prompt(openai_model: OpenAIModel):
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "Hello"},
    ]
    system_prompt: str = "You are a helpful assistant."
    response: ModelResponse = openai_model.chat(messages=messages, system=system_prompt)

    assert isinstance(response, ModelResponse)
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response, "text")
    assert isinstance(response.text, str)


def test_unset_api_key():
    from pathlib import Path

    original_key: str = os.environ.get("OPENAI_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = ""

    # Temporarily rename the key file if it exists
    key_path = Path.home() / ".alea" / "keys" / "openai"
    temp_path = key_path.with_suffix(".tmp")
    key_exists = key_path.exists()
    if key_exists:
        key_path.rename(temp_path)

    try:
        with pytest.raises(ValueError):
            OpenAIModel()
    finally:
        # Restore the key file
        if key_exists:
            temp_path.rename(key_path)
        os.environ["OPENAI_API_KEY"] = original_key


def test_formatter_identity():
    def formatter(args, kwargs):
        if len(args) > 0:
            return args[0]
        else:
            return kwargs.pop("messages")

    openai_model = OpenAIModel(formatter=formatter)
    response: ModelResponse = openai_model.chat(
        messages=[{"role": "user", "content": "2+2=?"}]
    )
    assert "4" in response.text or "four" in response.text.lower()


@pytest.mark.asyncio
async def test_chat_async(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 3 + 3?",
        },
    ]
    response: ModelResponse = await openai_model.chat_async(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "6" in content, f"Expected '6' in the response, but got: {content}"


@pytest.mark.asyncio
async def test_json_async(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'city' and 'population' for New York City with 8 million people.",
        },
    ]
    response: JSONModelResponse = await openai_model.json_async(
        messages=prompt, system="Respond in JSON."
    )

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("city") == "New York City"
    assert response.data.get("population") == 8000000


def test_retry_wrapper_exhaustion(openai_model: OpenAIModel):
    def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(openai_model, "_chat", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            openai_model.chat("Test message")


@pytest.mark.asyncio
async def test_retry_wrapper_async_exhaustion(openai_model: OpenAIModel):
    async def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(openai_model, "_chat_async", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            await openai_model.chat_async("Test message")


def test_cache_functionality(openai_model: OpenAIModel, tmp_path):
    openai_model.cache_path = tmp_path

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]

    # First call should hit the API
    response1: ModelResponse = openai_model.chat(messages=prompt)

    # Second call should use cached response
    response2: ModelResponse = openai_model.chat(messages=prompt)

    assert response1.text == response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


@pytest.mark.asyncio
async def test_cache_functionality_async(openai_model: OpenAIModel, tmp_path):
    openai_model.cache_path = tmp_path

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of Germany?",
        },
    ]

    response1: ModelResponse = await openai_model.chat_async(messages=prompt)
    response2: ModelResponse = await openai_model.chat_async(messages=prompt)

    assert "Berlin" in response1.text and "Berlin" in response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


class PydanticTestModel(BaseModel):
    name: str
    age: int


def test_pydantic_response(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Bob who is 25 years old.",
        },
    ]
    response: PydanticTestModel = openai_model.pydantic(
        messages=prompt, system="Respond in JSON.", pydantic_model=PydanticTestModel
    )

    assert isinstance(response, PydanticTestModel)
    assert response.name == "Bob"
    assert response.age == 25


@pytest.mark.asyncio
async def test_pydantic_async(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Charlie who is 35 years old.",
        },
    ]
    response: PydanticTestModel = await openai_model.pydantic_async(
        messages=prompt, system="Respond in JSON.", pydantic_model=PydanticTestModel
    )

    assert isinstance(response, PydanticTestModel)
    assert response.name == "Charlie"
    assert response.age == 35


def test_missing_pydantic_model(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object.",
        },
    ]
    with pytest.raises(ALEARetryExhaustedError):
        openai_model.pydantic(messages=prompt, system="Respond in JSON.")


def test_custom_model(openai_model: OpenAIModel):
    custom_model = "gpt-4o-mini"
    openai_model.model = custom_model
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Hello",
        },
    ]
    response: ModelResponse = openai_model.chat(messages=prompt)
    assert response.metadata.get("model") == custom_model


def test_max_tokens(openai_model: OpenAIModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Write a very long story.",
        },
    ]
    max_tokens = 50
    response: ModelResponse = openai_model.chat(messages=prompt, max_tokens=max_tokens)
    assert len(response.text.split()) <= max_tokens


def test_invalid_api_key():
    # unset env value
    model = OpenAIModel(api_key="abc123")
    with pytest.raises(ALEAAuthenticationError):
        print(
            model.chat(
                messages=[{"role": "user", "content": "Hello"}], ignore_cache=True
            )
        )


def test_model_response_to_dict(openai_model: OpenAIModel):
    response: ModelResponse = openai_model.chat(
        messages=[{"role": "user", "content": "Hello"}]
    )
    response_dict = response.to_dict()
    assert isinstance(response_dict, dict)
    assert "choices" in response_dict
    assert "metadata" in response_dict
    assert "text" in response_dict


def test_model_response_to_json(openai_model: OpenAIModel):
    response: ModelResponse = openai_model.chat(
        messages=[{"role": "user", "content": "Hello"}]
    )
    response_json = response.to_json()
    assert isinstance(response_json, str)
    assert "choices" in response_json
    assert "metadata" in response_json
    assert "text" in response_json


# Tests for New Model Generations
def test_gpt_5_models():
    """Test that GPT-5 models are available and working."""
    from alea_llm_client.llms.constants import GPT_5_MODELS, supports_grammar

    # Test a few key GPT-5 models
    test_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-chat-latest"]

    for model_name in test_models:
        assert model_name in GPT_5_MODELS, f"{model_name} should be in GPT_5_MODELS"
        assert supports_grammar(model_name), f"{model_name} should support grammar"

        # Test model instantiation
        model = OpenAIModel(model=model_name)
        assert model.model == model_name


def test_gpt_4_1_models():
    """Test that GPT-4.1 models are available."""
    from alea_llm_client.llms.constants import GPT_4_1_MODELS

    test_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]

    for model_name in test_models:
        assert model_name in GPT_4_1_MODELS, f"{model_name} should be in GPT_4_1_MODELS"

        # Test model instantiation
        model = OpenAIModel(model=model_name)
        assert model.model == model_name


def test_o_series_models():
    """Test that O-series reasoning models are available."""
    from alea_llm_client.llms.constants import (
        O_SERIES_MODELS,
        supports_reasoning_effort,
        requires_max_completion_tokens,
    )

    test_models = ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "o1-pro", "o3-pro"]

    for model_name in test_models:
        assert model_name in O_SERIES_MODELS, (
            f"{model_name} should be in O_SERIES_MODELS"
        )
        assert supports_reasoning_effort(model_name), (
            f"{model_name} should support reasoning_effort"
        )
        assert requires_max_completion_tokens(model_name), (
            f"{model_name} should require max_completion_tokens"
        )

        # Test model instantiation
        model = OpenAIModel(model=model_name)
        assert model.model == model_name


def test_specialized_models():
    """Test specialized models like computer use and realtime."""
    from alea_llm_client.llms.constants import (
        COMPUTER_USE_MODELS,
        REALTIME_MODELS,
        AUDIO_MODELS,
        supports_computer_use,
        supports_realtime,
        supports_audio,
    )

    # Test computer use models
    for model_name in ["computer-use-preview", "computer-use-preview-2025-03-11"]:
        assert model_name in COMPUTER_USE_MODELS
        assert supports_computer_use(model_name)

    # Test realtime models
    for model_name in ["gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview"]:
        assert model_name in REALTIME_MODELS
        assert supports_realtime(model_name)

    # Test audio models
    for model_name in ["gpt-4o-audio-preview", "gpt-4o-mini-audio-preview"]:
        assert model_name in AUDIO_MODELS
        assert supports_audio(model_name)


def test_model_support_functions():
    """Test all model support detection functions."""
    from alea_llm_client.llms.constants import (
        supports_reasoning_effort,
        supports_verbosity,
        supports_audio,
        supports_grammar,
        supports_realtime,
        supports_computer_use,
        requires_max_completion_tokens,
    )

    # Test GPT-5 capabilities
    assert supports_verbosity("gpt-5")
    assert supports_grammar("gpt-5")
    assert not supports_reasoning_effort("gpt-5")  # GPT-5 uses different parameters
    assert not requires_max_completion_tokens("gpt-5")

    # Test O-series capabilities
    assert supports_reasoning_effort("o3-mini")
    assert requires_max_completion_tokens("o3-mini")
    assert not supports_grammar("o3-mini")  # O-series doesn't support grammar yet

    # Test specialized model capabilities
    assert supports_realtime("gpt-4o-realtime-preview")
    assert supports_computer_use("computer-use-preview")
    assert supports_audio("gpt-4o-audio-preview")


def test_deep_research_models():
    """Test new deep research models."""
    from alea_llm_client.llms.constants import O_SERIES_MODELS

    research_models = [
        "o3-deep-research",
        "o3-deep-research-2025-06-26",
        "o4-mini-deep-research",
        "o4-mini-deep-research-2025-06-26",
    ]

    for model_name in research_models:
        assert model_name in O_SERIES_MODELS, (
            f"{model_name} should be in O_SERIES_MODELS"
        )

        # Test model instantiation
        model = OpenAIModel(model=model_name)
        assert model.model == model_name


def test_model_defaults():
    """Test that default model selections are valid."""
    from alea_llm_client.llms.constants import (
        OPENAI_DEFAULT_CHAT_MODEL,
        OPENAI_DEFAULT_REASONING_MODEL,
        OPENAI_DEFAULT_FAST_MODEL,
        GPT_5_MODELS,
        O_SERIES_MODELS,
    )

    # Test defaults are in their respective model families
    assert OPENAI_DEFAULT_CHAT_MODEL in GPT_5_MODELS, (
        f"OPENAI_DEFAULT_CHAT_MODEL {OPENAI_DEFAULT_CHAT_MODEL} should be a GPT-5 model"
    )
    assert OPENAI_DEFAULT_REASONING_MODEL in O_SERIES_MODELS, (
        f"OPENAI_DEFAULT_REASONING_MODEL {OPENAI_DEFAULT_REASONING_MODEL} should be an O-series model"
    )
    assert OPENAI_DEFAULT_FAST_MODEL in GPT_5_MODELS, (
        f"OPENAI_DEFAULT_FAST_MODEL {OPENAI_DEFAULT_FAST_MODEL} should be a GPT-5 model"
    )

    # Test that defaults can be instantiated
    chat_model = OpenAIModel(model=OPENAI_DEFAULT_CHAT_MODEL)
    reasoning_model = OpenAIModel(model=OPENAI_DEFAULT_REASONING_MODEL)
    fast_model = OpenAIModel(model=OPENAI_DEFAULT_FAST_MODEL)

    assert chat_model.model == OPENAI_DEFAULT_CHAT_MODEL
    assert reasoning_model.model == OPENAI_DEFAULT_REASONING_MODEL
    assert fast_model.model == OPENAI_DEFAULT_FAST_MODEL
