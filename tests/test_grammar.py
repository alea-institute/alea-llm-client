"""Tests for grammar parameter functionality in GPT-5 models."""

import pytest

from alea_llm_client import OpenAIModel, ModelResponse
from alea_llm_client.core.exceptions import ALEARetryExhaustedError
from alea_llm_client.llms.constants import (
    EXAMPLE_GRAMMARS,
    validate_grammar_definition,
    create_grammar_response_format,
    supports_grammar,
    GRAMMAR_SUPPORTED_MODELS,
)


@pytest.fixture
def gpt5_model():
    """Create a GPT-5 model for testing."""
    return OpenAIModel(model="gpt-5", ignore_cache=True)


def test_grammar_validation_valid_lark():
    """Test validation of valid Lark grammar."""
    grammar = 'start: "YES" | "NO"'
    is_valid, error = validate_grammar_definition(grammar, "lark")
    assert is_valid
    assert error is None


def test_grammar_validation_valid_regex():
    """Test validation of valid regex grammar."""
    grammar = r"^\d+$"
    is_valid, error = validate_grammar_definition(grammar, "regex")
    assert is_valid
    assert error is None


def test_grammar_validation_invalid_syntax():
    """Test validation with invalid syntax type."""
    grammar = 'start: "YES" | "NO"'
    is_valid, error = validate_grammar_definition(grammar, "invalid")
    assert not is_valid
    assert error is not None and "Invalid grammar syntax" in error


def test_grammar_validation_empty_definition():
    """Test validation with empty grammar definition."""
    is_valid, error = validate_grammar_definition("", "lark")
    assert not is_valid
    assert error is not None and "non-empty string" in error


def test_grammar_validation_invalid_regex():
    """Test validation with invalid regex pattern."""
    grammar = "[unclosed"
    is_valid, error = validate_grammar_definition(grammar, "regex")
    assert not is_valid
    assert error is not None and "Invalid regex pattern" in error


def test_create_grammar_response_format():
    """Test creating grammar response format dictionary."""
    grammar = 'start: "YES" | "NO"'
    format_dict = create_grammar_response_format("lark", grammar)

    expected = {"type": "grammar", "grammar": grammar, "syntax": "lark"}
    assert format_dict == expected


def test_create_grammar_response_format_invalid():
    """Test creating grammar response format with invalid input."""
    with pytest.raises(ValueError):
        create_grammar_response_format("invalid", "test")


def test_create_grammar_tool():
    """Test creating grammar tool for Responses API."""
    from alea_llm_client.llms.constants import create_grammar_tool

    grammar = 'start: "YES" | "NO"'
    tool = create_grammar_tool("lark", grammar, "test_tool", "Test description")

    expected = {
        "type": "custom",
        "name": "test_tool",
        "description": "Test description",
        "format": {"type": "grammar", "syntax": "lark", "definition": grammar},
    }
    assert tool == expected


def test_create_grammar_tool_defaults():
    """Test creating grammar tool with default parameters."""
    from alea_llm_client.llms.constants import create_grammar_tool

    grammar = r"^\d+$"
    tool = create_grammar_tool("regex", grammar)

    expected = {
        "type": "custom",
        "name": "grammar_response",
        "description": "Grammar-constrained response",
        "format": {"type": "grammar", "syntax": "regex", "definition": grammar},
    }
    assert tool == expected


def test_create_grammar_tool_invalid():
    """Test creating grammar tool with invalid input."""
    from alea_llm_client.llms.constants import create_grammar_tool

    with pytest.raises(ValueError):
        create_grammar_tool("invalid", "test")


def test_supports_grammar_function():
    """Test the supports_grammar utility function."""
    assert supports_grammar("gpt-5")
    assert supports_grammar("gpt-5-mini")
    assert not supports_grammar("gpt-4o")
    assert not supports_grammar("claude-3-opus")


def test_grammar_supported_models_list():
    """Test that GRAMMAR_SUPPORTED_MODELS contains expected models."""
    assert "gpt-5" in GRAMMAR_SUPPORTED_MODELS
    assert "gpt-5-mini" in GRAMMAR_SUPPORTED_MODELS
    assert "gpt-5-chat-latest" in GRAMMAR_SUPPORTED_MODELS


def test_binary_choice_grammar(gpt5_model: OpenAIModel):
    """Test binary choice grammar with GPT-5 using Responses API."""
    input_text = "Is 2 + 2 equal to 4?"

    binary_grammar = EXAMPLE_GRAMMARS["binary_choice_lark"]

    response: ModelResponse = gpt5_model.responses(
        input=input_text,
        grammar=binary_grammar["definition"],
        grammar_syntax=binary_grammar["syntax"],
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    # Response should follow the grammar (accept various yes/no formats)
    response_text = response.text.strip().lower().replace(".", "")
    assert response_text in ["yes", "no"], (
        f"Expected yes/no response, got: {response.text}"
    )


def test_number_regex_grammar(gpt5_model: OpenAIModel):
    """Test number regex grammar with GPT-5 using Responses API."""
    input_text = "What is 5 + 3? Respond with just the number."

    number_grammar = EXAMPLE_GRAMMARS["number_regex"]

    response: ModelResponse = gpt5_model.responses(
        input=input_text,
        grammar=number_grammar["definition"],
        grammar_syntax=number_grammar["syntax"],
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert response.text.strip().isdigit()


def test_email_regex_grammar(gpt5_model: OpenAIModel):
    """Test email regex grammar with GPT-5 using Responses API."""
    input_text = (
        "Generate a valid email address for a user named John Doe at example.com"
    )

    email_grammar = EXAMPLE_GRAMMARS["email_regex"]

    response: ModelResponse = gpt5_model.responses(
        input=input_text,
        grammar=email_grammar["definition"],
        grammar_syntax=email_grammar["syntax"],
        timeout=300,
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert "@" in response.text
    assert "." in response.text


def test_grammar_with_default_syntax(gpt5_model: OpenAIModel):
    """Test grammar parameter with default lark syntax using Responses API."""
    input_text = "Answer yes or no: Is Python a programming language?"

    response: ModelResponse = gpt5_model.responses(
        input=input_text,
        grammar='start: "yes" | "no"',  # No syntax specified, should default to lark
        timeout=300,
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert response.text.strip().lower() in ["yes", "no"]


def test_invalid_grammar_raises_error():
    """Test that invalid grammar raises appropriate error during validation."""
    # Test the validation function directly since grammar validation
    # happens before API calls but may get caught by retry logic
    from alea_llm_client.llms.constants import validate_grammar_definition

    # Empty grammar should fail validation
    is_valid, error = validate_grammar_definition("", "lark")
    assert not is_valid
    assert error is not None and "non-empty string" in error

    # Invalid syntax should fail
    is_valid, error = validate_grammar_definition("test", "invalid_syntax")
    assert not is_valid
    assert error is not None and "Invalid grammar syntax" in error


def test_invalid_grammar_syntax_raises_error(gpt5_model: OpenAIModel):
    """Test that invalid grammar syntax raises appropriate error."""
    input_text = "Test message"

    # Grammar validation happens before API call, so should raise ValueError immediately
    with pytest.raises((ValueError, ALEARetryExhaustedError)):
        gpt5_model.responses(
            input=input_text,
            grammar='start: "test"',
            grammar_syntax="invalid_syntax",  # Invalid syntax type
        )


def test_grammar_syntax_only_no_definition():
    """Test that providing grammar_syntax without grammar definition does nothing."""
    model = OpenAIModel(model="gpt-5", ignore_cache=True)
    input_text = "Hello"

    # This should work normally without grammar constraints
    response: ModelResponse = model.responses(
        input=input_text,
        grammar_syntax="lark",  # No grammar definition provided
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None


def test_complex_json_grammar(gpt5_model: OpenAIModel):
    """Test complex JSON object grammar using Responses API."""
    input_text = "Create a JSON object with name 'Alice' and age 30"

    json_grammar = EXAMPLE_GRAMMARS["json_object_lark"]

    response: ModelResponse = gpt5_model.responses(
        input=input_text,
        grammar=json_grammar["definition"],
        grammar_syntax=json_grammar["syntax"],
        timeout=300,
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert "{" in response.text and "}" in response.text


def test_grammar_performance_warning_logged(gpt5_model: OpenAIModel, caplog):
    """Test that grammar usage logs a performance warning using Responses API."""
    input_text = "Answer yes or no"

    gpt5_model.responses(input=input_text, grammar='start: "YES" | "NO"', timeout=300)

    # Check that performance warning was logged
    assert "latency overhead" in caplog.text
    assert "timeout" in caplog.text


def test_grammar_with_other_parameters(gpt5_model: OpenAIModel):
    """Test grammar parameter combined with other GPT-5 parameters using Responses API."""
    input_text = "Is artificial intelligence useful? Answer yes or no."

    response: ModelResponse = gpt5_model.responses(
        input=input_text,
        grammar='start: "YES" | "NO"',
        reasoning_effort="minimal",  # Should work with other parameters
        verbosity="low",
        timeout=300,
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert response.text.strip() in ["YES", "NO"]


@pytest.mark.asyncio
async def test_grammar_async(gpt5_model: OpenAIModel):
    """Test grammar parameter with async responses method."""
    input_text = "Should we use async programming? Answer yes or no."

    response: ModelResponse = await gpt5_model.responses_async(
        input=input_text, grammar='start: "yes" | "no"', timeout=300
    )

    assert isinstance(response, ModelResponse)
    assert response.text is not None
    assert response.text.strip().lower() in ["yes", "no"]


def test_example_grammars_all_valid():
    """Test that all example grammars are valid."""
    for name, grammar_spec in EXAMPLE_GRAMMARS.items():
        is_valid, error = validate_grammar_definition(
            grammar_spec["definition"], grammar_spec["syntax"]
        )
        assert is_valid, f"Example grammar '{name}' is invalid: {error}"
