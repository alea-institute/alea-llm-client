"""Tool creation helpers for Anthropic Messages API tool types.

This module provides helper functions to create tool definition dicts
for Anthropic server tools and custom tools. Each returns a plain dict.
"""

from typing import Any, Optional


def create_anthropic_web_search_tool(
    max_uses: Optional[int] = None,
    user_location: Optional[dict[str, Any]] = None,
    allowed_domains: Optional[list[str]] = None,
    blocked_domains: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Create a web_search tool for the Anthropic Messages API.

    Args:
        max_uses: Optional maximum number of search uses per request.
        user_location: Optional user location dict.
        allowed_domains: Optional list of allowed domains.
        blocked_domains: Optional list of blocked domains.

    Returns:
        Tool definition dict.

    Raises:
        ValueError: If both allowed_domains and blocked_domains are provided.
    """
    if allowed_domains is not None and blocked_domains is not None:
        raise ValueError("allowed_domains and blocked_domains are mutually exclusive")

    tool: dict[str, Any] = {
        "type": "web_search_20260209",
        "name": "web_search",
    }
    if max_uses is not None:
        tool["max_uses"] = max_uses
    if user_location is not None:
        tool["user_location"] = user_location
    if allowed_domains is not None:
        tool["allowed_domains"] = allowed_domains
    if blocked_domains is not None:
        tool["blocked_domains"] = blocked_domains
    return tool


def create_anthropic_web_fetch_tool(
    max_content_tokens: Optional[int] = None,
    allowed_domains: Optional[list[str]] = None,
    blocked_domains: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Create a web_fetch tool for the Anthropic Messages API.

    Args:
        max_content_tokens: Optional maximum content tokens per fetch.
        allowed_domains: Optional list of allowed domains.
        blocked_domains: Optional list of blocked domains.

    Returns:
        Tool definition dict.

    Raises:
        ValueError: If both allowed_domains and blocked_domains are provided.
    """
    if allowed_domains is not None and blocked_domains is not None:
        raise ValueError("allowed_domains and blocked_domains are mutually exclusive")

    tool: dict[str, Any] = {
        "type": "web_fetch_20260209",
        "name": "web_fetch",
    }
    if max_content_tokens is not None:
        tool["max_content_tokens"] = max_content_tokens
    if allowed_domains is not None:
        tool["allowed_domains"] = allowed_domains
    if blocked_domains is not None:
        tool["blocked_domains"] = blocked_domains
    return tool


def create_anthropic_code_execution_tool() -> dict[str, Any]:
    """Create a code_execution tool for the Anthropic Messages API.

    Returns:
        Tool definition dict.
    """
    return {"type": "code_execution_20260120", "name": "code_execution"}


def create_anthropic_bash_tool() -> dict[str, Any]:
    """Create a bash tool for the Anthropic Messages API.

    Returns:
        Tool definition dict.
    """
    return {"type": "bash_20250124", "name": "bash"}


def create_anthropic_text_editor_tool() -> dict[str, Any]:
    """Create a text_editor tool for the Anthropic Messages API.

    Returns:
        Tool definition dict.
    """
    return {"type": "text_editor_20250728", "name": "text_editor"}


def create_anthropic_computer_use_tool(
    display_width_px: int,
    display_height_px: int,
    display_number: Optional[int] = None,
) -> dict[str, Any]:
    """Create a computer use tool for the Anthropic Messages API.

    Args:
        display_width_px: Display width in pixels.
        display_height_px: Display height in pixels.
        display_number: Optional X display number.

    Returns:
        Tool definition dict.
    """
    tool: dict[str, Any] = {
        "type": "computer_20251124",
        "name": "computer",
        "display_width_px": display_width_px,
        "display_height_px": display_height_px,
    }
    if display_number is not None:
        tool["display_number"] = display_number
    return tool


def create_anthropic_memory_tool() -> dict[str, Any]:
    """Create a memory tool for the Anthropic Messages API.

    Returns:
        Tool definition dict.
    """
    return {"type": "memory_20250818", "name": "memory"}


def create_anthropic_tool_search_bm25_tool() -> dict[str, Any]:
    """Create a tool_search_bm25 tool for the Anthropic Messages API.

    Returns:
        Tool definition dict.
    """
    return {"type": "tool_search_bm25_20251119", "name": "tool_search_bm25"}


def create_anthropic_tool_search_regex_tool() -> dict[str, Any]:
    """Create a tool_search_regex tool for the Anthropic Messages API.

    Returns:
        Tool definition dict.
    """
    return {"type": "tool_search_regex_20251119", "name": "tool_search_regex"}


def create_anthropic_function_tool(
    name: str,
    input_schema: dict[str, Any],
    description: Optional[str] = None,
) -> dict[str, Any]:
    """Create a custom function tool for the Anthropic Messages API.

    Args:
        name: Function name.
        input_schema: JSON Schema describing the function input.
        description: Optional function description.

    Returns:
        Tool definition dict.
    """
    tool: dict[str, Any] = {
        "type": "custom",
        "name": name,
        "input_schema": input_schema,
    }
    if description is not None:
        tool["description"] = description
    return tool
