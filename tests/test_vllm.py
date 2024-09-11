import pytest
from typing import List, Dict
from unittest.mock import patch
from pydantic import BaseModel

from alea_llm_client import VLLMModel, ModelResponse, JSONModelResponse
from alea_llm_client.core.exceptions import ALEARetryExhaustedError


@pytest.fixture
def vllm_model():
    return VLLMModel()


def test_simple_addition(vllm_model: VLLMModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]
    response: ModelResponse = vllm_model.chat(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


# def test with ignore_cache=True
def test_simple_addition_ignore_cache(vllm_model: VLLMModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]
    response: ModelResponse = vllm_model.chat(messages=prompt, ignore_cache=True)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


def test_json_response(vllm_model: VLLMModel):
    prompt = "Give me a JSON object with keys 'name' and 'age' for a person named Alice who is 30 years old."
    response: JSONModelResponse = vllm_model.json(prompt, system="Respond in JSON.")

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("name") == "Alice"
    assert response.data.get("age") == 30


def test_format_with_system_prompt(vllm_model: VLLMModel):
    expected_messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "Hello"},
    ]
    system_prompt: str = "You are a helpful assistant."
    formatted_messages: List[Dict[str, str]] = vllm_model.format(
        args=[], kwargs={"messages": messages, "system": system_prompt}
    )

    assert len(formatted_messages) == 2
    assert formatted_messages == expected_messages


def test_chat_with_system_prompt(vllm_model: VLLMModel):
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "Hello"},
    ]
    system_prompt: str = "You are a helpful assistant."
    response: ModelResponse = vllm_model.chat(messages=messages, system=system_prompt)

    assert isinstance(response, ModelResponse)
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response, "text")
    assert isinstance(response.text, str)


def test_formatter_identity():
    def formatter(args, kwargs):
        if len(args) > 0:
            return args[0]
        else:
            return kwargs.pop("messages")

    vllm_model = VLLMModel(formatter=formatter)
    response: ModelResponse = vllm_model.chat(
        messages=[{"role": "user", "content": "2+2=?"}]
    )
    assert "4" in response.text or "four" in response.text.lower()


@pytest.mark.asyncio
async def test_chat_async(vllm_model: VLLMModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 3 + 3?",
        },
    ]
    response: ModelResponse = await vllm_model.chat_async(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "6" in content, f"Expected '6' in the response, but got: {content}"


@pytest.mark.asyncio
async def test_json_async(vllm_model: VLLMModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'city' and 'population' for New York City with 8 million people.",
        },
    ]
    response: JSONModelResponse = await vllm_model.json_async(
        messages=prompt, system="Respond in JSON."
    )

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("city") == "New York City"
    assert response.data.get("population") in ["8000000", 8000000]


def test_retry_wrapper_exhaustion(vllm_model: VLLMModel):
    def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(vllm_model, "_chat", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            vllm_model.chat("Test message")


@pytest.mark.asyncio
async def test_retry_wrapper_async_exhaustion(vllm_model: VLLMModel):
    async def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(vllm_model, "_chat_async", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            await vllm_model.chat_async("Test message")


def test_cache_functionality(vllm_model: VLLMModel, tmp_path):
    vllm_model.cache_path = tmp_path

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]

    # First call should hit the API
    response1: ModelResponse = vllm_model.chat(messages=prompt)

    # Second call should use cached response
    response2: ModelResponse = vllm_model.chat(messages=prompt)

    assert response1.text == response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


@pytest.mark.asyncio
async def test_cache_functionality_async(vllm_model: VLLMModel, tmp_path):
    vllm_model.cache_path = tmp_path

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of Germany?",
        },
    ]

    response1: ModelResponse = await vllm_model.chat_async(messages=prompt)
    response2: ModelResponse = await vllm_model.chat_async(messages=prompt)

    assert "Berlin" in response1.text and "Berlin" in response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


class TestPydanticModel(BaseModel):
    name: str
    age: int


def test_pydantic_response(vllm_model: VLLMModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Bob who is 25 years old.",
        },
    ]
    response: TestPydanticModel = vllm_model.pydantic(
        messages=prompt, system="Respond in JSON.", pydantic_model=TestPydanticModel
    )

    assert isinstance(response, TestPydanticModel)
    assert response.name == "Bob"
    assert response.age == 25


@pytest.mark.asyncio
async def test_pydantic_async(vllm_model: VLLMModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Charlie who is 35 years old.",
        },
    ]
    response: TestPydanticModel = await vllm_model.pydantic_async(
        messages=prompt, system="Respond in JSON.", pydantic_model=TestPydanticModel
    )

    assert isinstance(response, TestPydanticModel)
    assert response.name == "Charlie"
    assert response.age == 35


def test_missing_pydantic_model(vllm_model: VLLMModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object.",
        },
    ]
    with pytest.raises(ALEARetryExhaustedError):
        vllm_model.pydantic(messages=prompt, system="Respond in JSON.")


def test_max_tokens(vllm_model: VLLMModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Write a very long story.",
        },
    ]
    max_tokens = 50
    response: ModelResponse = vllm_model.chat(messages=prompt, max_tokens=max_tokens)
    assert len(response.text.split()) <= max_tokens


def test_model_response_to_dict(vllm_model: VLLMModel):
    response: ModelResponse = vllm_model.chat(
        messages=[{"role": "user", "content": "Hello"}]
    )
    response_dict = response.to_dict()
    assert isinstance(response_dict, dict)
    assert "choices" in response_dict
    assert "metadata" in response_dict
    assert "text" in response_dict


def test_model_response_to_json(vllm_model: VLLMModel):
    response: ModelResponse = vllm_model.chat(
        messages=[{"role": "user", "content": "Hello"}]
    )
    response_json = response.to_json()
    assert isinstance(response_json, str)
    assert "choices" in response_json
    assert "metadata" in response_json
    assert "text" in response_json
