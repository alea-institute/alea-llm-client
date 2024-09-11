import json
import re

from alea_llm_client.core.logging import setup_logger

# Get an instance of a logger
LOGGER = setup_logger(__name__)


def normalize_json_response(response: str) -> str:
    """Normalize and clean JSON responses.

    This function removes backtick language indicators and ensures
    the response is a valid JSON or JSONL object.

    Args:
        response (str): The raw response string from the model.

    Returns:
        str: Cleaned and normalized JSON string.

    Raises:
        ValueError: If the cleaned response is not a valid JSON or JSONL.
    """
    # Remove backticks and language indicators
    cleaned = re.sub(
        r"^```(?:json)?\n?|\n?```$", "", response.strip(), flags=re.MULTILINE
    )

    # Attempt to parse as JSON
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Try to parse from the first { or [
    try:
        first_pos_list = [
            pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0
        ]
        if not first_pos_list:
            raise ValueError(
                "The cleaned response is not a valid JSON or JSONL object: %s",
                cleaned,
            )
        first_pos = min(first_pos_list)
        cleaned = cleaned[first_pos:]
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Now try to trim from the last } or ]
    try:
        last_pos_list = [
            pos for pos in (cleaned.rfind("}"), cleaned.rfind("]")) if pos >= 0
        ]
        if not last_pos_list:
            raise ValueError(
                "The cleaned response is not a valid JSON or JSONL object: %s"
                % (cleaned,)
            )
        last_pos = max(last_pos_list)
        cleaned = cleaned[: last_pos + 1]
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Attempt to parse as JSONL
    try:
        for line in cleaned.split("\n"):
            if line.strip():
                json.loads(line)
        return cleaned
    except json.JSONDecodeError:
        LOGGER.error(f"Invalid JSON or JSONL response: {cleaned}")
        raise ValueError("The cleaned response is not a valid JSON or JSONL object.")


def replace_jsons_refs_with_enum(schema, enum_def):
    """
    Replace all JSON Schema $refs with an enum definition.

    Args:
        schema (dict): The JSON Schema to update.
        enum_def (dict): The enum definition to use.

    Returns:
        dict: The updated JSON Schema
    """
    if isinstance(schema, dict):
        keys_to_update = []
        for key, value in schema.items():
            if key == "$ref" and value == "#/$defs/ContentType":
                keys_to_update.append(key)
            else:
                replace_jsons_refs_with_enum(value, enum_def)

        for key in keys_to_update:
            schema.pop(key)
            schema.update(enum_def)

    elif isinstance(schema, list):
        for item in schema:
            replace_jsons_refs_with_enum(item, enum_def)

    return schema
