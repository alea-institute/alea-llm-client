import pytest
from typing import List, Dict
from unittest.mock import patch
from pydantic import BaseModel

from alea_llm_client import GrokModel, ModelResponse, JSONModelResponse
from alea_llm_client.core.exceptions import ALEARetryExhaustedError


@pytest.fixture
def grok_model():
    return GrokModel()


def test_simple_addition(grok_model: GrokModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]
    response: ModelResponse = grok_model.chat(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


# def test with ignore_cache=True
def test_simple_addition_ignore_cache(grok_model: GrokModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 2 + 2?",
        },
    ]
    response: ModelResponse = grok_model.chat(messages=prompt, ignore_cache=True)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


def test_json_response(grok_model: GrokModel):
    prompt = "Give me a JSON object with keys 'name' and 'age' for a person named Alice who is 30 years old."
    response: JSONModelResponse = grok_model.json(prompt, system="Respond in JSON.")

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("name") == "Alice"
    assert response.data.get("age") == 30


def test_format_with_system_prompt(grok_model: GrokModel):
    expected_messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "Hello"},
    ]
    system_prompt: str = "You are a helpful assistant."
    formatted_messages: List[Dict[str, str]] = grok_model.format(
        args=[], kwargs={"messages": messages, "system": system_prompt}
    )

    assert len(formatted_messages) == 2
    assert formatted_messages == expected_messages


def test_chat_with_system_prompt(grok_model: GrokModel):
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "Hello"},
    ]
    system_prompt: str = "You are a helpful assistant."
    response: ModelResponse = grok_model.chat(messages=messages, system=system_prompt)

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

    grok_model = GrokModel(formatter=formatter)
    response: ModelResponse = grok_model.chat(
        messages=[{"role": "user", "content": "2+2=?"}]
    )
    assert "4" in response.text or "four" in response.text.lower()


@pytest.mark.asyncio
async def test_chat_async(grok_model: GrokModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is 3 + 3?",
        },
    ]
    response: ModelResponse = await grok_model.chat_async(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "6" in content, f"Expected '6' in the response, but got: {content}"


@pytest.mark.asyncio
async def test_json_async(grok_model: GrokModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'city' and 'population' for New York City with 8 million people.",
        },
    ]
    response: JSONModelResponse = await grok_model.json_async(
        messages=prompt, system="Respond in JSON."
    )

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("city") == "New York City"
    assert response.data.get("population") in ["8000000", 8000000]


def test_retry_wrapper_exhaustion(grok_model: GrokModel):
    def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(grok_model, "_chat", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            grok_model.chat("Test message")


@pytest.mark.asyncio
async def test_retry_wrapper_async_exhaustion(grok_model: GrokModel):
    async def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(grok_model, "_chat_async", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            await grok_model.chat_async("Test message")


def test_cache_functionality(grok_model: GrokModel, tmp_path):
    grok_model.cache_path = tmp_path

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ]

    # First call should hit the API
    response1: ModelResponse = grok_model.chat(messages=prompt)

    # Second call should use cached response
    response2: ModelResponse = grok_model.chat(messages=prompt)

    assert response1.text == response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


@pytest.mark.asyncio
async def test_cache_functionality_async(grok_model: GrokModel, tmp_path):
    grok_model.cache_path = tmp_path

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "What is the capital of Germany?",
        },
    ]

    response1: ModelResponse = await grok_model.chat_async(messages=prompt)
    response2: ModelResponse = await grok_model.chat_async(messages=prompt)

    assert "Berlin" in response1.text and "Berlin" in response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


class TestPydanticModel(BaseModel):
    name: str
    age: int


def test_pydantic_response(grok_model: GrokModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Bob who is 25 years old.",
        },
    ]
    response: TestPydanticModel = grok_model.pydantic(
        messages=prompt, system="Respond in JSON.", pydantic_model=TestPydanticModel
    )

    assert isinstance(response, TestPydanticModel)
    assert response.name == "Bob"
    assert response.age == 25


@pytest.mark.asyncio
async def test_pydantic_async(grok_model: GrokModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object with keys 'name' and 'age' for a person named Charlie who is 35 years old.",
        },
    ]
    response: TestPydanticModel = await grok_model.pydantic_async(
        messages=prompt, system="Respond in JSON.", pydantic_model=TestPydanticModel
    )

    assert isinstance(response, TestPydanticModel)
    assert response.name == "Charlie"
    assert response.age == 35


def test_missing_pydantic_model(grok_model: GrokModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Give me a JSON object.",
        },
    ]
    with pytest.raises(ALEARetryExhaustedError):
        grok_model.pydantic(messages=prompt, system="Respond in JSON.")


def test_max_tokens(grok_model: GrokModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": "Write a very long story.",
        },
    ]
    max_tokens = 50
    response: ModelResponse = grok_model.chat(messages=prompt, max_tokens=max_tokens)
    assert len(response.text.split()) <= max_tokens


def test_model_response_to_dict(grok_model: GrokModel):
    response: ModelResponse = grok_model.chat(
        messages=[{"role": "user", "content": "Hello"}]
    )
    response_dict = response.to_dict()
    assert isinstance(response_dict, dict)
    assert "choices" in response_dict
    assert "metadata" in response_dict
    assert "text" in response_dict


def test_model_response_to_json(grok_model: GrokModel):
    response: ModelResponse = grok_model.chat(
        messages=[{"role": "user", "content": "Hello"}]
    )
    response_json = response.to_json()
    assert isinstance(response_json, str)
    assert "choices" in response_json
    assert "metadata" in response_json
    assert "text" in response_json
