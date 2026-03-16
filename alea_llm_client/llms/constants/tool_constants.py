"""Tool creation helpers for OpenAI Responses API tool types.

This module provides helper functions to create tool definition dicts
for all 15 Responses API tool types. Each returns a plain dict with
no new classes or dependencies.
"""

from typing import Any, Optional


def create_web_search_tool(
    search_context_size: str = "medium",
    user_location: Optional[dict[str, Any]] = None,
    filters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a web_search tool for the Responses API.

    Args:
        search_context_size: Size of search context ("low", "medium", "high").
        user_location: Optional user location dict with type, city, region, country, timezone.
        filters: Optional search filters.

    Returns:
        Tool definition dict.
    """
    tool: dict[str, Any] = {
        "type": "web_search",
        "search_context_size": search_context_size,
    }
    if user_location is not None:
        tool["user_location"] = user_location
    if filters is not None:
        tool["filters"] = filters
    return tool


def create_web_search_preview_tool(
    search_context_size: str = "medium",
    user_location: Optional[dict[str, Any]] = None,
    filters: Optional[dict[str, Any]] = None,
    search_content_types: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Create a web_search_preview tool for the Responses API.

    Args:
        search_context_size: Size of search context ("low", "medium", "high").
        user_location: Optional user location dict.
        filters: Optional search filters.
        search_content_types: Optional list of content types to search.

    Returns:
        Tool definition dict.
    """
    tool: dict[str, Any] = {
        "type": "web_search_preview",
        "search_context_size": search_context_size,
    }
    if user_location is not None:
        tool["user_location"] = user_location
    if filters is not None:
        tool["filters"] = filters
    if search_content_types is not None:
        tool["search_content_types"] = search_content_types
    return tool


def create_file_search_tool(
    vector_store_ids: list[str],
    max_num_results: Optional[int] = None,
    filters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a file_search tool for the Responses API.

    Args:
        vector_store_ids: List of vector store IDs to search.
        max_num_results: Optional maximum number of results.
        filters: Optional search filters.

    Returns:
        Tool definition dict.
    """
    tool: dict[str, Any] = {
        "type": "file_search",
        "vector_store_ids": vector_store_ids,
    }
    if max_num_results is not None:
        tool["max_num_results"] = max_num_results
    if filters is not None:
        tool["filters"] = filters
    return tool


def create_code_interpreter_tool(
    container: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a code_interpreter tool for the Responses API.

    Args:
        container: Optional container configuration dict.

    Returns:
        Tool definition dict.
    """
    tool: dict[str, Any] = {"type": "code_interpreter"}
    if container is not None:
        tool["container"] = container
    return tool


def create_computer_tool() -> dict[str, Any]:
    """Create a computer tool for the Responses API.

    Returns:
        Tool definition dict.
    """
    return {"type": "computer"}


def create_computer_use_preview_tool(
    display_width: int = 1024,
    display_height: int = 768,
    environment: str = "browser",
) -> dict[str, Any]:
    """Create a computer_use_preview tool for the Responses API.

    Args:
        display_width: Display width in pixels.
        display_height: Display height in pixels.
        environment: Environment type ("browser", "mac", "windows", "linux").

    Returns:
        Tool definition dict.
    """
    return {
        "type": "computer_use_preview",
        "display_width": display_width,
        "display_height": display_height,
        "environment": environment,
    }


def create_image_generation_tool(
    model: str = "gpt-image-1",
    quality: str = "auto",
    size: str = "auto",
    output_format: str = "png",
) -> dict[str, Any]:
    """Create an image_generation tool for the Responses API.

    Args:
        model: Image generation model to use.
        quality: Image quality ("auto", "low", "medium", "high").
        size: Image size ("auto", "1024x1024", "1536x1024", "1024x1536").
        output_format: Output format ("png", "webp", "jpeg").

    Returns:
        Tool definition dict.
    """
    return {
        "type": "image_generation",
        "model": model,
        "quality": quality,
        "size": size,
        "output_format": output_format,
    }


def create_mcp_tool(
    server_label: str,
    server_url: Optional[str] = None,
    connector_id: Optional[str] = None,
    allowed_tools: Optional[list[str]] = None,
    require_approval: Optional[str] = None,
) -> dict[str, Any]:
    """Create an mcp tool for the Responses API.

    Args:
        server_label: Label for the MCP server.
        server_url: URL of the MCP server (mutually exclusive with connector_id).
        connector_id: Connector ID for managed MCP (mutually exclusive with server_url).
        allowed_tools: Optional list of allowed tool names.
        require_approval: Approval setting ("always", "never").

    Returns:
        Tool definition dict.

    Raises:
        ValueError: If neither server_url nor connector_id is provided.
    """
    if server_url is None and connector_id is None:
        raise ValueError("Either server_url or connector_id must be provided")

    tool: dict[str, Any] = {
        "type": "mcp",
        "server_label": server_label,
    }
    if server_url is not None:
        tool["server_url"] = server_url
    if connector_id is not None:
        tool["connector_id"] = connector_id
    if allowed_tools is not None:
        tool["allowed_tools"] = allowed_tools
    if require_approval is not None:
        tool["require_approval"] = require_approval
    return tool


def create_function_tool(
    name: str,
    parameters: dict[str, Any],
    description: Optional[str] = None,
    strict: Optional[bool] = None,
) -> dict[str, Any]:
    """Create a function tool for the Responses API.

    Args:
        name: Function name.
        parameters: JSON Schema describing the function parameters.
        description: Optional function description.
        strict: Optional strict mode for structured output.

    Returns:
        Tool definition dict.
    """
    tool: dict[str, Any] = {
        "type": "function",
        "name": name,
        "parameters": parameters,
    }
    if description is not None:
        tool["description"] = description
    if strict is not None:
        tool["strict"] = strict
    return tool


def create_local_shell_tool() -> dict[str, Any]:
    """Create a local_shell tool for the Responses API.

    Returns:
        Tool definition dict.
    """
    return {"type": "local_shell"}


def create_tool_search_tool(
    execution: str = "server",
) -> dict[str, Any]:
    """Create a tool_search tool for the Responses API.

    Args:
        execution: Where tools execute ("server" or "client").

    Returns:
        Tool definition dict.
    """
    return {
        "type": "tool_search",
        "execution": execution,
    }


def create_apply_patch_tool() -> dict[str, Any]:
    """Create an apply_patch tool for the Responses API.

    Returns:
        Tool definition dict.
    """
    return {"type": "apply_patch"}
