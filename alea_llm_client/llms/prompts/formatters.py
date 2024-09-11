"""LLM prompt formatters to handle message/chat templates and special token translation across various models."""

# imports
from enum import Enum


# Token type enumeration
class TokenType(Enum):
    """Enumeration of token types for special token translation."""

    BEGIN_OF_TEXT = "begin_of_text"
    END_OF_TEXT = "end_of_text"
    START_HEADER = "start_header"
    END_HEADER = "end_header"
    JSON_MARKER = "json_marker"


# Token mappings
TOKEN_MAPPING = {
    "llama3": {
        TokenType.BEGIN_OF_TEXT: "<|begin_of_text|>",
        TokenType.END_OF_TEXT: "<|eot_id|>",
        TokenType.START_HEADER: "<|start_header_id|>",
        TokenType.END_HEADER: "<|end_header_id|>",
        TokenType.JSON_MARKER: "```json",
    },
}


def format_prompt(
    args: list,
    kwargs: dict,
) -> str:
    """
    Format a model prompt based on the upstream *args and **kwargs, including
    a model=... argument to match the token enum.
    """
    # ensure that we have a model argument
    if "model" not in kwargs:
        raise ValueError("Model argument not provided.")

    # get the model argument and remove it from the kwargs
    model = kwargs.pop("model")

    # get the token mapping for the model
    token_mapping = TOKEN_MAPPING.get(model, None)
    if token_mapping is None:
        raise ValueError(f"Token mapping not found for model: {model}")

    # check for args and kwargs
    text = None
    messages = None
    if "text" in kwargs or "messages" in kwargs:
        text = kwargs.pop("text", None)
        messages = kwargs.pop("messages", None)
    elif len(args) > 0:
        if isinstance(args[0], str):
            text = args[0]
            messages = None
        elif isinstance(args[0], list):
            text = None
            messages = args[0]
    else:
        raise ValueError("No text or messages provided in args or kwargs.")

    # check for system prompt
    system_prompt = kwargs.pop("system", None)

    # check that we have either text or messages
    if text is not None and messages is not None:
        raise ValueError(
            "Upstream caller to format() must provide either 'text' or 'messages', not both."
        )

    if text is None and messages is None:
        raise ValueError(
            "Upstream caller to format() must provide either 'text' or 'messages'."
        )

    json_output = False
    formatted_prompt = token_mapping[TokenType.BEGIN_OF_TEXT]

    if system_prompt:
        formatted_prompt += (
            f"{token_mapping.get(TokenType.START_HEADER)}system{token_mapping.get(TokenType.END_HEADER)}"
            f"\n\n{system_prompt}{token_mapping.get(TokenType.END_OF_TEXT)}"
        )

    if text:
        # format text into prompt if provided
        text = text.strip()
        if text.endswith(token_mapping.get(TokenType.JSON_MARKER)):
            text = text[: -len(token_mapping.get(TokenType.JSON_MARKER))].strip()
            json_output = True
        formatted_prompt += (
            f"{token_mapping.get(TokenType.START_HEADER)}user{token_mapping.get(TokenType.END_HEADER)}"
            f"\n\n{text}{token_mapping.get(TokenType.END_OF_TEXT)}"
        )
    elif messages:
        # format messages into prompt if provided
        for message in messages:
            role = message.get("role", "user").lower()
            content = message.get("content", "")
            formatted_prompt += (
                f"{token_mapping.get(TokenType.START_HEADER)}{role}{token_mapping.get(TokenType.END_HEADER)}"
                f"\n\n{content}{token_mapping.get(TokenType.END_OF_TEXT)}"
            )

        if messages[-1]["role"].lower() == "user" and messages[-1][
            "content"
        ].strip().endswith(token_mapping.get(TokenType.JSON_MARKER)):
            json_output = True

    # add assistant header
    formatted_prompt += (
        f"{token_mapping.get(TokenType.START_HEADER)}"
        f"assistant{token_mapping.get(TokenType.END_HEADER)}\n\n"
    )

    # add JSON marker if needed
    if json_output:
        formatted_prompt += f"{token_mapping.get(TokenType.JSON_MARKER)}\n"

    return formatted_prompt


def format_prompt_llama3(
    args: list,
    kwargs: dict,
) -> str:
    """
    Format a model prompt for the llama3 model based on the upstream *args and **kwargs.
    """
    kwargs["model"] = "llama3"
    return format_prompt(*args, **kwargs)
