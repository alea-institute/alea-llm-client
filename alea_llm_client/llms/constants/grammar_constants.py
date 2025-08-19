"""Grammar constants for OpenAI GPT-5 context-free grammar support.

This module provides constants and validation for grammar-constrained generation,
supporting both Lark and Regex grammar syntax types.
"""

from typing import Literal, Dict, Any, Optional
from enum import Enum

# Grammar syntax types
GRAMMAR_SYNTAX_TYPES = ["lark", "regex"]
GrammarSyntaxType = Literal["lark", "regex"]

# Response format type for grammar
GRAMMAR_RESPONSE_FORMAT_TYPE = "grammar"


class GrammarSyntax(Enum):
    """Supported grammar syntax types."""

    LARK = "lark"
    REGEX = "regex"


# Performance considerations
GRAMMAR_LATENCY_MULTIPLIER = 10  # 8-10x latency overhead
GRAMMAR_DEFAULT_TIMEOUT = 300  # 5 minutes recommended timeout
GRAMMAR_MIN_TIMEOUT = 120  # Minimum recommended timeout

# Validation constants
MIN_GRAMMAR_DEFINITION_LENGTH = 1
MAX_GRAMMAR_DEFINITION_LENGTH = 10000  # Reasonable limit to prevent issues


def validate_grammar_syntax(syntax: str) -> bool:
    """Validate that the grammar syntax is supported.

    Args:
        syntax: The grammar syntax type to validate.

    Returns:
        True if syntax is valid, False otherwise.
    """
    return syntax in GRAMMAR_SYNTAX_TYPES


def validate_grammar_definition(
    definition: str, syntax: str
) -> tuple[bool, Optional[str]]:
    """Validate a grammar definition string.

    Args:
        definition: The grammar definition to validate.
        syntax: The syntax type ("lark" or "regex").

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not definition or not isinstance(definition, str):
        return False, "Grammar definition must be a non-empty string"

    if len(definition) < MIN_GRAMMAR_DEFINITION_LENGTH:
        return (
            False,
            f"Grammar definition must be at least {MIN_GRAMMAR_DEFINITION_LENGTH} character(s)",
        )

    if len(definition) > MAX_GRAMMAR_DEFINITION_LENGTH:
        return (
            False,
            f"Grammar definition cannot exceed {MAX_GRAMMAR_DEFINITION_LENGTH} characters",
        )

    if not validate_grammar_syntax(syntax):
        return (
            False,
            f"Invalid grammar syntax '{syntax}'. Must be one of: {', '.join(GRAMMAR_SYNTAX_TYPES)}",
        )

    # Basic syntax-specific validation
    if syntax == "regex":
        try:
            import re

            re.compile(definition)
        except re.error as e:
            return False, f"Invalid regex pattern: {str(e)}"
    elif syntax == "lark":
        # Basic Lark grammar validation (checking for common patterns)
        if not any(char in definition for char in [":", "|", '"', "'"]):
            return (
                False,
                "Lark grammar should contain rules with colons (:) and terminals",
            )

    return True, None


def create_grammar_response_format(
    syntax: GrammarSyntaxType, definition: str
) -> Dict[str, Any]:
    """Create a grammar response format dictionary.

    Args:
        syntax: The grammar syntax type.
        definition: The grammar definition string.

    Returns:
        Dictionary representing the grammar response format.

    Raises:
        ValueError: If syntax or definition is invalid.
    """
    is_valid, error = validate_grammar_definition(definition, syntax)
    if not is_valid:
        raise ValueError(error)

    return {
        "type": GRAMMAR_RESPONSE_FORMAT_TYPE,
        "grammar": definition,
        "syntax": syntax,
    }


def create_grammar_tool(
    syntax: GrammarSyntaxType,
    definition: str,
    name: str = "grammar_response",
    description: str = "Grammar-constrained response",
) -> Dict[str, Any]:
    """Create a grammar tool for OpenAI Responses API.

    Args:
        syntax: The grammar syntax type ("lark" or "regex").
        definition: The grammar definition string.
        name: The tool name.
        description: The tool description.

    Returns:
        Dictionary representing the grammar tool for Responses API.

    Raises:
        ValueError: If syntax or definition is invalid.
    """
    is_valid, error = validate_grammar_definition(definition, syntax)
    if not is_valid:
        raise ValueError(error)

    return {
        "type": "custom",
        "name": name,
        "description": description,
        "format": {"type": "grammar", "syntax": syntax, "definition": definition},
    }


# Example grammar definitions for testing and documentation
EXAMPLE_GRAMMARS = {
    "binary_choice_lark": {"syntax": "lark", "definition": 'start: "YES" | "NO"'},
    "number_regex": {"syntax": "regex", "definition": r"^\d+$"},
    "json_object_lark": {
        "syntax": "lark",
        "definition": r"""
        start: "{" pair ("," pair)* "}"
        pair: STRING ":" value
        value: STRING | NUMBER | "true" | "false" | "null"
        STRING: /"[^"]*"/
        NUMBER: /-?\d+(\.\d+)?/
        """,
    },
    "email_regex": {
        "syntax": "regex",
        "definition": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    },
}
