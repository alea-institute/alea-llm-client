import os

import pytest
from typing import List, Dict
from unittest.mock import patch
from pydantic import BaseModel

from alea_llm_client import GoogleModel, ModelResponse, JSONModelResponse
from alea_llm_client.core.exceptions import ALEARetryExhaustedError


@pytest.fixture
def google_model():
    access_token = os.environ.get("VERTEX_ACCESS_TOKEN")
    project_id = os.environ.get("VERTEX_PROJECT_ID")
    location_id = os.environ.get("VERTEX_LOCATION_ID")
    api_endpoint = f"https://{location_id}-aiplatform.googleapis.com"

    return GoogleModel(
        model="gemini-2.0-flash-exp",
        api_key=access_token,
        endpoint=api_endpoint,
        project_id=project_id,
        location_id=location_id,
    )


def test_simple_addition(google_model: GoogleModel):
    prompt: List[Dict[str, str]] = [
        {"role": "user", "parts": [{"text": "What is 2 + 2?"}]},
    ]
    response: ModelResponse = google_model.chat(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


# def test with ignore_cache=True
def test_simple_addition_ignore_cache(google_model: GoogleModel):
    prompt: List[Dict[str, str]] = [
        {"role": "user", "parts": [{"text": "What is 2 + 2?"}]},
    ]
    response: ModelResponse = google_model.chat(messages=prompt, ignore_cache=True)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "4" in content, f"Expected '4' in the response, but got: {content}"


def test_json_response(google_model: GoogleModel):
    prompt = "Give me a JSON object with keys 'name' and 'age' for a person named Alice who is 30 years old."
    response: JSONModelResponse = google_model.json(prompt)

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("name") == "Alice"
    assert response.data.get("age") == 30


def test_chat_with_system_prompt(google_model: GoogleModel):
    messages: List[Dict[str, str]] = [
        {"role": "user", "parts": [{"text": "What is 2 + 2?"}]},
    ]
    system_instruction: dict = {"parts": [{"text": "You are a helpful assistant."}]}
    response: ModelResponse = google_model.chat(
        messages=messages, system_instruction=system_instruction
    )

    assert isinstance(response, ModelResponse)
    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert hasattr(response, "text")
    assert isinstance(response.text, str)


@pytest.mark.asyncio
async def test_chat_async(google_model: GoogleModel):
    prompt: List[Dict[str, str]] = [
        {"role": "user", "parts": [{"text": "What is 2 * 3?"}]},
    ]
    response: ModelResponse = await google_model.chat_async(messages=prompt)

    assert isinstance(response, ModelResponse)
    content: str = response.text.strip()
    assert "6" in content, f"Expected '6' in the response, but got: {content}"


@pytest.mark.asyncio
async def test_json_async(google_model: GoogleModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Give me a JSON object with keys 'city' and 'population' for New York City with 8 million people."
                }
            ],
        },
    ]
    response: JSONModelResponse = await google_model.json_async(messages=prompt)

    assert isinstance(response, JSONModelResponse)
    assert isinstance(response.data, dict)
    assert response.data.get("city") == "New York City"
    assert response.data.get("population") in ["8000000", 8000000]


def test_retry_wrapper_exhaustion(google_model: GoogleModel):
    def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(google_model, "_chat", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            google_model.chat("Test message")


@pytest.mark.asyncio
async def test_retry_wrapper_async_exhaustion(google_model: GoogleModel):
    async def failing_function(*args, **kwargs):
        raise Exception("Simulated failure")

    with patch.object(google_model, "_chat_async", side_effect=failing_function):
        with pytest.raises(ALEARetryExhaustedError):
            await google_model.chat_async("Test message")


def test_cache_functionality(google_model: GoogleModel, tmp_path):
    google_model.cache_path = tmp_path

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            # "content": "What is the capital of France?",
            "parts": [{"text": "What is the capital of France?"}],
        },
    ]

    # First call should hit the API
    response1: ModelResponse = google_model.chat(messages=prompt)

    # Second call should use cached response
    response2: ModelResponse = google_model.chat(messages=prompt)

    assert response1.text == response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


@pytest.mark.asyncio
async def test_cache_functionality_async(google_model: GoogleModel, tmp_path):
    google_model.cache_path = tmp_path

    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            # "content": "What is the capital of Germany?",
            "parts": [{"text": "What is the capital of Germany?"}],
        },
    ]

    response1: ModelResponse = await google_model.chat_async(messages=prompt)
    response2: ModelResponse = await google_model.chat_async(messages=prompt)

    assert "Berlin" in response1.text and "Berlin" in response2.text
    assert len(list(tmp_path.glob("*.json.gz"))) == 1  # Check if cache file was created


class TestPydanticModel(BaseModel):
    name: str
    age: int


def test_pydantic_response(google_model: GoogleModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            # "content": "Give me a JSON object with keys 'name' and 'age' for a person named Bob who is 25 years old.",
            "parts": [
                {
                    "text": "Give me a JSON object with keys 'name' and 'age' for a person named Bob who is 25 years old."
                }
            ],
        },
    ]
    response: TestPydanticModel = google_model.pydantic(
        messages=prompt, pydantic_model=TestPydanticModel
    )

    assert isinstance(response, TestPydanticModel)
    assert response.name == "Bob"
    assert response.age == 25


@pytest.mark.asyncio
async def test_pydantic_async(google_model: GoogleModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            # "content": "Give me a JSON object with keys 'name' and 'age' for a person named Charlie who is 35 years old.",
            "parts": [
                {
                    "text": "Give me a JSON object with keys 'name' and 'age' for a person named Charlie who is 35 years old."
                }
            ],
        },
    ]
    response: TestPydanticModel = await google_model.pydantic_async(
        messages=prompt, pydantic_model=TestPydanticModel
    )

    assert isinstance(response, TestPydanticModel)
    assert response.name == "Charlie"
    assert response.age == 35


def test_missing_pydantic_model(google_model: GoogleModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            # "content": "Give me a JSON object.",
            "parts": [{"text": "Give me a JSON object."}],
        },
    ]
    with pytest.raises(ALEARetryExhaustedError):
        google_model.pydantic(messages=prompt)


def test_max_tokens(google_model: GoogleModel):
    prompt: List[Dict[str, str]] = [
        {
            "role": "user",
            # "content": "Write a very long story.",
            "parts": [{"text": "Write a very long story."}],
        },
    ]
    max_tokens = 50
    response: ModelResponse = google_model.chat(
        messages=prompt, generation_config={"maxOutputTokens": max_tokens}
    )
    assert len(response.text.split()) <= max_tokens


def test_model_response_to_dict(google_model: GoogleModel):
    response: ModelResponse = google_model.chat(
        # messages=[{"role": "user", "content": "Hello"}]
        messages=[{"role": "user", "parts": [{"text": "Hello"}]}]
    )
    response_dict = response.to_dict()
    assert isinstance(response_dict, dict)
    assert "choices" in response_dict
    assert "metadata" in response_dict
    assert "text" in response_dict


def test_model_response_to_json(google_model: GoogleModel):
    response: ModelResponse = google_model.chat(
        # messages=[{"role": "user", "content": "Hello"}]
        messages=[{"role": "user", "parts": [{"text": "Hello"}]}]
    )
    response_json = response.to_json()
    assert isinstance(response_json, str)
    assert "choices" in response_json
    assert "metadata" in response_json
    assert "text" in response_json
