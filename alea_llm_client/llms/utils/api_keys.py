"""Shared API key retrieval utilities.

This module provides centralized API key management functionality
to eliminate duplication across model classes.
"""

import os
from pathlib import Path
from typing import Any, Optional

from alea_llm_client.core.logging import setup_logger

logger = setup_logger(__name__)

# Default key storage location
DEFAULT_KEY_BASE_PATH = Path.home() / ".alea" / "keys"


def get_api_key(
    init_kwargs: dict[str, Any],
    provider_name: str,
    env_var_names: list[str],
    key_file_name: Optional[str] = None,
    fallback_value: Optional[str] = None,
    required: bool = True,
) -> str:
    """Generic API key retrieval utility.

    This function implements the standard API key retrieval pattern:
    1. Check init_kwargs["api_key"]
    2. Check environment variables (in order)
    3. Check key file at ~/.alea/keys/{key_file_name}
    4. Return fallback_value or raise error

    Args:
        init_kwargs: The initialization kwargs dict from the model
        provider_name: Human readable provider name for error messages
        env_var_names: List of environment variable names to check (in order)
        key_file_name: Name of the key file (defaults to provider_name.lower())
        fallback_value: Value to return if no key found (instead of raising error)
        required: Whether to raise error if no key found (ignored if fallback_value provided)

    Returns:
        The API key string

    Raises:
        ValueError: If required=True and no API key found
    """
    # Step 1: Check if api_key was provided in init_kwargs
    api_key = init_kwargs.get("api_key")
    if api_key and api_key.strip():
        logger.debug(f"Using {provider_name} API key from init_kwargs")
        return api_key.strip()

    # Step 2: Check environment variables
    logger.debug(f"Attempting to get {provider_name} API key from environment variables")
    for env_var in env_var_names:
        api_key = os.environ.get(env_var, None)
        if api_key and api_key.strip():
            logger.debug(f"Found {provider_name} API key in {env_var}")
            return api_key.strip()

    # Step 3: Check key file
    if key_file_name is None:
        key_file_name = provider_name.lower()

    key_path = DEFAULT_KEY_BASE_PATH / key_file_name
    logger.debug(f"Attempting to get {provider_name} API key from key file: {key_path}")

    if key_path.exists():
        try:
            api_key = key_path.read_text().strip()
            if api_key:
                logger.debug(f"Found {provider_name} API key in key file")
                return api_key
        except OSError as e:
            logger.warning(f"Error reading {provider_name} key file {key_path}: {e}")

    # Step 4: Return fallback or raise error
    if fallback_value is not None:
        logger.debug(f"Using fallback value for {provider_name} API key")
        return fallback_value

    if not required:
        logger.debug(f"No {provider_name} API key found, but not required")
        return ""

    # Construct helpful error message
    env_vars_str = ", ".join(env_var_names)
    raise ValueError(
        f"{provider_name} API key not found. Tried: "
        f"init_kwargs['api_key'], environment variables ({env_vars_str}), "
        f"and key file ({key_path})"
    )


def get_anthropic_api_key(init_kwargs: dict[str, Any]) -> str:
    """Get Anthropic API key using standard retrieval pattern."""
    return get_api_key(
        init_kwargs=init_kwargs,
        provider_name="Anthropic",
        env_var_names=["ANTHROPIC_API_KEY"],
        key_file_name="anthropic",
    )


def get_openai_api_key(init_kwargs: dict[str, Any]) -> str:
    """Get OpenAI API key using standard retrieval pattern."""
    return get_api_key(
        init_kwargs=init_kwargs,
        provider_name="OpenAI",
        env_var_names=["OPENAI_API_KEY"],
        key_file_name="openai",
    )


def get_google_api_key(init_kwargs: dict[str, Any]) -> str:
    """Get Google/Vertex API key using standard retrieval pattern."""
    return get_api_key(
        init_kwargs=init_kwargs,
        provider_name="Google",
        env_var_names=["GOOGLE_API_KEY", "VERTEX_API_KEY", "GEMINI_API_KEY"],
        key_file_name="google",
    )


def get_grok_api_key(init_kwargs: dict[str, Any]) -> str:
    """Get Grok API key using standard retrieval pattern."""
    return get_api_key(
        init_kwargs=init_kwargs,
        provider_name="Grok",
        env_var_names=["GROK_API_KEY", "XAI_API_KEY"],
        key_file_name="grok",
    )


def get_vllm_api_key(init_kwargs: dict[str, Any]) -> str:
    """Get VLLM API key (placeholder - VLLM typically doesn't need real keys)."""
    return get_api_key(
        init_kwargs=init_kwargs,
        provider_name="VLLM",
        env_var_names=["VLLM_API_KEY"],
        key_file_name="vllm",
        fallback_value="key",  # VLLM uses placeholder key
        required=False,
    )


def get_openai_compatible_api_key(init_kwargs: dict[str, Any], endpoint: str) -> str:
    """Get API key for OpenAI-compatible endpoints."""
    # For generic OpenAI-compatible endpoints, try common env vars
    return get_api_key(
        init_kwargs=init_kwargs,
        provider_name="OpenAI-Compatible",
        env_var_names=["OPENAI_API_KEY", "API_KEY"],
        key_file_name="openai_compatible",
        fallback_value="key",  # Many local models use placeholder keys
        required=False,
    )
