"""Anthropic model implementation for the ALEA LLM client.

This module provides an implementation of the BaseAIModel for Anthropic's API.
It includes classes and methods for both synchronous and asynchronous chat
and JSON completions using Anthropic's language models.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel

from alea_llm_client.core.exceptions import ALEAAuthenticationError, ALEAModelError

# project imports
from alea_llm_client.core.logging import LoggerMixin
from alea_llm_client.llms.models.base_ai_model import (
    BaseAIModel,
    JSONModelResponse,
    ModelResponse,
)
from alea_llm_client.llms.utils.api_keys import get_anthropic_api_key

DEFAULT_ENDPOINT = "https://api.anthropic.com/"
DEFAULT_CACHE_PATH = Path.home() / ".alea" / "cache" / "anthropic"
DEFAULT_VERSION = "2023-06-01"
DEFAULT_KWARGS = {"max_tokens": 128}


class AnthropicModel(BaseAIModel, LoggerMixin):
    """
    Anthropic model implementation.

    This class implements the BaseAIModel for Anthropic's API, providing methods
    for both synchronous and asynchronous chat and JSON completions.
    """

    COMPLETION_ENDPOINT = "/v1/complete"
    CHAT_ENDPOINT = "/v1/messages"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        endpoint: str = DEFAULT_ENDPOINT,
        formatter: Optional[Callable] = None,
        cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Anthropic model.

        Args:
            api_key: The API key for Anthropic. If None, it will be retrieved from environment variables.
            model: The name of the Anthropic model to use.
            endpoint: The API endpoint URL (if different from default).
            formatter: A function to format input messages.
            cache_path: The path to the cache directory for storing model responses.
        """
        # Save init kwargs before calling parent constructor (needed for get_api_key)
        self.init_kwargs = kwargs.copy()
        self.init_kwargs.update(
            {
                "api_key": api_key,
                "model": model,
                "endpoint": endpoint,
                "formatter": formatter,
                "cache_path": cache_path,
            }
        )

        BaseAIModel.__init__(
            self,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            formatter=formatter,
            cache_path=cache_path,
            **kwargs,
        )
        self.logger.info(f"Initialized AnthropicModel with model: {model} and endpoint: {endpoint}")

    def get_api_key(self) -> str:
        """
        Retrieve the API key for Anthropic from the environment.

        Returns:
            The Anthropic API key.

        Raises:
            ValueError: If the ANTHROPIC_API_KEY is not found in environment variables.
        """
        return get_anthropic_api_key(self.init_kwargs)

    def _initialize_client(self) -> httpx.Client:
        """Initialize and return the synchronous HTTP client for Anthropic API.

        Returns:
            httpx.Client: The initialized synchronous client.
        """
        return httpx.Client(
            base_url=self.endpoint,
            http2=True,
            timeout=self.init_kwargs.get("timeout", 600),
        )

    def _initialize_async_client(self) -> httpx.AsyncClient:
        """Initialize and return the asynchronous HTTP client for Anthropic API.

        Returns:
            httpx.AsyncClient: The initialized asynchronous client.
        """
        return httpx.AsyncClient(
            base_url=self.endpoint,
            http2=True,
            timeout=self.init_kwargs.get("timeout", 600),
        )

    def _make_request(self, *args: Any, **kwargs: Any) -> httpx.Response:
        """
        Make a request to the API.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The response from the API.
        """
        # we must have a url to proceed
        url = kwargs.pop("url", args[0] if len(args) > 0 else None)
        if not url:
            raise ValueError("No URL provided for request; must provide either as args[0] or url=... keyword arg.")

        # pop headers from kwargs
        headers = kwargs.pop("headers", {})

        # set Authorization if not already set
        if "Authorization" not in headers:
            headers["x-api-key"] = self.get_api_key()
            headers["anthropic-version"] = self.init_kwargs.get("anthropic_version", DEFAULT_VERSION)

        self.logger.debug("Making request to the API")

        # Build request body with parameter validation
        request_body = {"model": self.model}

        # Handle thinking configuration parameter (Claude 4+ models)
        if "thinking" in kwargs:
            thinking_config = kwargs.pop("thinking")

            # Handle different thinking parameter formats
            if isinstance(thinking_config, bool):
                if thinking_config:
                    request_body["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": 1600,
                    }
                    self.logger.debug("Enabling thinking configuration with default budget")
                # Don't include thinking config if disabled
            elif isinstance(thinking_config, dict):
                # Validate thinking configuration
                if thinking_config.get("enabled", False):
                    budget_tokens = thinking_config.get("budget_tokens", 1600)
                    if budget_tokens < 1024:
                        raise ValueError("thinking budget_tokens must be at least 1024")

                    # Check against max_tokens if present
                    max_tokens_key = "max_tokens" if "max_tokens" in kwargs else None
                    if max_tokens_key and kwargs[max_tokens_key] <= budget_tokens:
                        raise ValueError("thinking budget_tokens must be less than max_tokens")

                    request_body["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                    }
                    self.logger.debug(f"Enabling thinking with budget_tokens: {budget_tokens}")
                # Don't include thinking config if disabled
            else:
                raise ValueError("thinking parameter must be a boolean or dictionary")

        try:
            # set any missing DEFAULT_KWARGS
            for k, v in DEFAULT_KWARGS.items():
                if k not in kwargs:
                    kwargs[k] = v

            # Add remaining parameters
            request_body.update(kwargs)

            # make and raise here
            response = self.client.post(
                url,
                json=request_body,
                headers=headers,
            )

            # check for 400 to return the right ALEA error
            if response.status_code == 400:
                error_message = response.json().get("error", {}).get("message", response.json())
                raise ALEAModelError(f"Model error: {error_message}")

            # check for 401 to return the right ALEA error
            if response.status_code == 401:
                error_message = response.json().get("error", {}).get("message", response.json())
                raise ALEAAuthenticationError(f"Authentication error: {error_message}")

            response.raise_for_status()
            return response
        except (ALEAModelError, ALEAAuthenticationError):
            raise
        except Exception as e:
            self.logger.exception(f"Error in request: {e!s}")
            raise ALEAModelError(f"Error in request: {e!s}") from e

    async def _make_request_async(self, *args: Any, **kwargs: Any) -> httpx.Response:
        """
        Make an asynchronous request to the API.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The response from the API.
        """
        # we must have a url to proceed
        url = kwargs.pop("url", args[0] if len(args) > 0 else None)
        if not url:
            raise ValueError("No URL provided for request; must provide either as args[0] or url=... keyword arg.")

        # pop headers from kwargs
        headers = kwargs.pop("headers", {})

        # set Authorization if not already set
        if "Authorization" not in headers:
            headers["x-api-key"] = self.get_api_key()
            headers["anthropic-version"] = self.init_kwargs.get("anthropic_version", DEFAULT_VERSION)

        self.logger.debug("Making asynchronous request to the API")

        # Build request body with parameter validation
        request_body = {"model": self.model}

        # Handle thinking configuration parameter (Claude 4+ models)
        if "thinking" in kwargs:
            thinking_config = kwargs.pop("thinking")

            # Handle different thinking parameter formats
            if isinstance(thinking_config, bool):
                if thinking_config:
                    request_body["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": 1600,
                    }
                    self.logger.debug("Enabling thinking configuration with default budget")
                # Don't include thinking config if disabled
            elif isinstance(thinking_config, dict):
                # Validate thinking configuration
                if thinking_config.get("enabled", False):
                    budget_tokens = thinking_config.get("budget_tokens", 1600)
                    if budget_tokens < 1024:
                        raise ValueError("thinking budget_tokens must be at least 1024")

                    # Check against max_tokens if present
                    max_tokens_key = "max_tokens" if "max_tokens" in kwargs else None
                    if max_tokens_key and kwargs[max_tokens_key] <= budget_tokens:
                        raise ValueError("thinking budget_tokens must be less than max_tokens")

                    request_body["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget_tokens,
                    }
                    self.logger.debug(f"Enabling thinking with budget_tokens: {budget_tokens}")
                # Don't include thinking config if disabled
            else:
                raise ValueError("thinking parameter must be a boolean or dictionary")

        try:
            # set any missing DEFAULT_KWARGS
            for k, v in DEFAULT_KWARGS.items():
                if k not in kwargs:
                    kwargs[k] = v

            # Add remaining parameters
            request_body.update(kwargs)

            # make and raise here
            response = await self.async_client.post(
                url,
                json=request_body,
                headers=headers,
            )

            # check for 400 to return the right ALEA error
            if response.status_code == 400:
                error_message = response.json().get("error", {}).get("message", response.json())
                raise ALEAModelError(f"Model error: {error_message}")

            # check for 401 to return the right ALEA error
            if response.status_code == 401:
                error_message = response.json().get("error", {}).get("message", response.json())
                raise ALEAAuthenticationError(f"Authentication error: {error_message}")

            response.raise_for_status()
            return response
        except (ALEAModelError, ALEAAuthenticationError):
            raise
        except Exception as e:
            self.logger.exception(f"Error in asynchronous request: {e!s}")
            raise ALEAModelError(f"Error in asynchronous request: {e!s}") from e

    def format(self, args: Any, kwargs: Any) -> list[dict[str, str]]:
        """Format inputs or outputs using the specified formatter.

        This method formats the input messages for the chat completion.
        If a custom formatter is provided, it will be used. Otherwise,
        it formats the input as a list of message dictionaries.

        Args:
            args (Any): Positional arguments passed to the chat method.
            kwargs (Any): Keyword arguments passed to the chat method.

        Returns:
            List[Dict[str, str]]: A list of formatted message dictionaries.

        Raises:
            ValueError: If no messages are provided for chat completion.
        """
        self.logger.debug("Formatting input for Anthropic API")
        if self.formatter:
            return self.formatter(args, kwargs)

        # Handle messages
        messages = kwargs.pop("messages", None)
        if not messages:
            if len(args) > 0:
                messages = [{"role": "user", "content": args[0]}]
            else:
                self.logger.error("No messages provided for chat completion")
                raise ValueError("No messages provided for chat completion.")

        self.logger.debug(f"Formatted messages: {messages}")
        return messages

    @staticmethod
    def _get_complete_choices(response_data: dict) -> list[str | dict]:
        """
        Get the response choices from the response data.

        Args:
            response_data: The response data.

        Returns:
            The response choices.
        """
        return [choice.get("text") for choice in response_data.get("choices", [])]

    @staticmethod
    def _get_chat_choices(response_data: dict) -> list[str | dict]:
        """
        Get the chat response choices from the response data.
        Enhanced to handle thinking blocks and multiple content types.

        Args:
            response_data: The response data.

        Returns:
            The response choices (text content only).
        """
        content_blocks = response_data.get("content", [])
        if not content_blocks:
            return [""]

        # Extract text content from all text blocks
        text_parts = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        # Return concatenated text or empty string if no text blocks
        return ["\n".join(text_parts) if text_parts else ""]

    @staticmethod
    def _extract_thinking_content(response_data: dict) -> Optional[str]:
        """
        Extract thinking content from the response data.

        Args:
            response_data: The response data.

        Returns:
            The thinking content, if available.
        """
        content_blocks = response_data.get("content", [])
        thinking_parts = []

        for block in content_blocks:
            if block.get("type") == "thinking":
                thinking_text = block.get("thinking", "") or block.get("text", "")
                if thinking_text:
                    thinking_parts.append(thinking_text)

        return "\n".join(thinking_parts) if thinking_parts else None

    @staticmethod
    def _extract_content_blocks(response_data: dict) -> list[dict[str, Any]]:
        """
        Extract all content blocks from the response data.

        Args:
            response_data: The response data.

        Returns:
            List of content blocks.
        """
        return response_data.get("content", [])

    def _handle_chat_response(self, response: httpx.Response) -> ModelResponse:
        """
        Handle the response from the chat completion.
        Enhanced to extract thinking content and content blocks for Anthropic models.

        Args:
            response: The response from the chat completion.

        Returns:
            The model response with thinking content and content blocks populated.
        """
        self.logger.debug("Handling Anthropic chat response")
        response_data = response.json()
        choices = self._get_chat_choices(response_data)

        # Extract additional Anthropic-specific content
        thinking_content = self._extract_thinking_content(response_data)
        content_blocks = self._extract_content_blocks(response_data)

        # Extract usage information and check for reasoning tokens (for future models)
        usage = response_data.get("usage", {})
        reasoning_tokens = usage.get("reasoning_tokens")

        model_response = ModelResponse(
            choices=choices,
            metadata={"model": self.model, "usage": usage},
            text=choices[0] if len(choices) > 0 else "",
            thinking=thinking_content,
            reasoning_tokens=reasoning_tokens,
            content_blocks=content_blocks,
        )

        self.logger.debug(f"Anthropic model response with {len(content_blocks)} content blocks")
        if thinking_content:
            self.logger.debug(f"Response includes thinking content ({len(thinking_content)} characters)")

        return model_response

    def _complete(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """
        Synchronous completion method.
        Note: Anthropic API uses chat/messages endpoint for all completions.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The response from the completion.
        """
        self.logger.debug("Initiating synchronous completion")
        prompt = kwargs.pop("prompt", None) or (args[0] if len(args) > 0 else None)
        if not prompt:
            raise ValueError("No prompt provided for completion.")

        # Convert prompt to messages format for Anthropic API
        messages = [{"role": "user", "content": prompt}]
        return self._handle_chat_response(
            self._make_request(
                url=self.CHAT_ENDPOINT,
                messages=messages,
                **kwargs,
            )
        )

    async def _complete_async(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """
        Asynchronous completion method.
        Note: Anthropic API uses chat/messages endpoint for all completions.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The response from the completion.
        """
        self.logger.debug("Initiating asynchronous completion")
        prompt = kwargs.pop("prompt", None) or (args[0] if len(args) > 0 else None)
        if not prompt:
            raise ValueError("No prompt provided for completion.")

        # Convert prompt to messages format for Anthropic API
        messages = [{"role": "user", "content": prompt}]
        return self._handle_chat_response(
            await self._make_request_async(
                url=self.CHAT_ENDPOINT,
                messages=messages,
                **kwargs,
            )
        )

    def _chat(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """
        Synchronous chat completion method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The response from the chat completion.
        """
        self.logger.debug("Initiating synchronous chat completion")
        messages = self.format(args, kwargs)
        return self._handle_chat_response(
            self._make_request(
                url=self.CHAT_ENDPOINT,
                messages=messages,
                **kwargs,
            )
        )

    async def _chat_async(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """
        Asynchronous chat completion method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The response from the chat completion.
        """
        self.logger.debug("Initiating asynchronous chat completion")
        messages = self.format(args, kwargs)
        return self._handle_chat_response(
            await self._make_request_async(
                url=self.CHAT_ENDPOINT,
                messages=messages,
                **kwargs,
            )
        )

    def _handle_json_response(self, response: httpx.Response) -> JSONModelResponse:
        """
        Handle the response from the JSON completion.

        Args:
            response: The response from the JSON completion.

        Returns:
            The JSON response from the JSON completion.
        """
        self.logger.debug("Handling JSON response")

        # get response and raw response choice output
        response_data = response.json()
        choices = self._get_chat_choices(response_data)
        try:
            raw_json = choices[0]
        except IndexError:
            raw_json = "{}"
        json_data = self.parse_json(raw_json)
        json_model_response = JSONModelResponse(
            choices=choices,
            metadata={"model": self.model, "usage": response_data.get("usage", {})},
            text=raw_json,
            data=json_data,
        )
        self.logger.debug(f"JSON model response: {json_model_response}")
        return json_model_response

    def _json(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """
        Synchronous JSON completion method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The JSON response from the completion.
        """
        self.logger.debug("Initiating synchronous JSON completion")
        messages = self.format(args, kwargs)
        return self._handle_json_response(
            self._make_request(
                url=self.CHAT_ENDPOINT,
                messages=messages,
                **kwargs,
            )
        )

    async def _json_async(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """
        Asynchronous JSON completion method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The JSON response from the completion.
        """
        self.logger.debug("Initiating asynchronous JSON completion")
        messages = self.format(args, kwargs)
        return self._handle_json_response(
            await self._make_request_async(
                url=self.CHAT_ENDPOINT,
                messages=messages,
                **kwargs,
            )
        )

    def _handle_pydantic_response(self, response: httpx.Response, pydantic_model: BaseModel) -> BaseModel:
        """
        Handle the response from the Pydantic completion.

        Args:
            response: The response from the Pydantic completion.
            pydantic_model: The Pydantic model to validate the response.

        Returns:
            The Pydantic model response from the Pydantic completion.
        """
        self.logger.debug("Handling Pydantic response")

        # get response and raw response choice output
        response_data = response.json()
        choices = self._get_chat_choices(response_data)
        try:
            raw_json = choices[0]
        except IndexError:
            raw_json = "{}"
        json_data = self.parse_json(raw_json)
        pydantic_response = pydantic_model.model_validate(json_data)
        self.logger.debug("Pydantic completion successful")
        return pydantic_response

    def _pydantic(self, *args: Any, **kwargs: Any) -> BaseModel:
        """
        Synchronous Pydantic completion method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The Pydantic model response from the completion.

        Raises:
            ValueError: If Pydantic model is not provided.
        """
        self.logger.debug("Initiating synchronous Pydantic completion")

        # pydantic model with result field
        pydantic_model: Optional[BaseModel] = kwargs.pop("pydantic_model", None)
        if not pydantic_model:
            raise ValueError("Pydantic model not provided for Pydantic completion.")

        # get the response
        messages = self.format(args, kwargs)
        return self._handle_pydantic_response(
            self._make_request(
                url=self.CHAT_ENDPOINT,
                messages=messages,
                **kwargs,
            ),
            pydantic_model,
        )

    async def _pydantic_async(self, *args: Any, **kwargs: Any) -> BaseModel:
        """
        Asynchronous Pydantic completion method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The Pydantic model response from the completion.

        Raises:
            ValueError: If Pydantic model is not provided.
        """
        self.logger.debug("Initiating asynchronous Pydantic completion")

        # pydantic model with result field
        pydantic_model: Optional[BaseModel] = kwargs.pop("pydantic_model", None)
        if not pydantic_model:
            raise ValueError("Pydantic model not provided for Pydantic completion.")

        # get the response
        messages = self.format(args, kwargs)
        return self._handle_pydantic_response(
            await self._make_request_async(
                url=self.CHAT_ENDPOINT,
                messages=messages,
                **kwargs,
            ),
            pydantic_model,
        )

    def _responses(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """
        Synchronous responses completion method.
        Note: Anthropic doesn't have a direct responses API equivalent.
        This delegates to regular chat completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The response from the responses completion.
        """
        self.logger.debug("Initiating synchronous responses completion (delegating to chat)")
        self.logger.warning("Anthropic doesn't have responses API - using chat endpoint")
        return self._chat(*args, **kwargs)

    async def _responses_async(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """
        Asynchronous responses completion method.
        Note: Anthropic doesn't have a direct responses API equivalent.
        This delegates to regular chat completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The response from the responses completion.
        """
        self.logger.debug("Initiating asynchronous responses completion (delegating to chat)")
        self.logger.warning("Anthropic doesn't have responses API - using chat endpoint")
        return await self._chat_async(*args, **kwargs)
