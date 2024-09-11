"""Anthropic model implementation for the ALEA LLM client.

This module provides an implementation of the BaseAIModel for Anthropic's API.
It includes classes and methods for both synchronous and asynchronous chat
and JSON completions using Anthropic's language models.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Dict

import httpx

# project imports
from alea_llm_client.core.logging import LoggerMixin
from alea_llm_client.core.exceptions import ALEAModelError, ALEAAuthenticationError
from alea_llm_client.llms.models.openai_compatible_model import OpenAICompatibleModel

DEFAULT_ENDPOINT = "https://api.anthropic.com/"
DEFAULT_CACHE_PATH = Path.home() / ".alea" / "cache" / "anthropic"
DEFAULT_KEY_PATH = Path.home() / ".alea" / "keys" / "anthropic"
DEFAULT_VERSION = "2023-06-01"
DEFAULT_KWARGS = {"max_tokens": 128}


class AnthropicModel(OpenAICompatibleModel, LoggerMixin):
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
        model: str = "claude-3-5-sonnet-20240620",
        endpoint: Optional[str] = DEFAULT_ENDPOINT,
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
        OpenAICompatibleModel.__init__(
            self,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            formatter=formatter,
            cache_path=cache_path,
            **kwargs,
        )
        if endpoint is None:
            self.logger.info(f"Initialized AnthropicModel with model: {model}")
        else:
            self.logger.info(
                f"Initialized AnthropicModel with model: {model} and endpoint: {endpoint}"
            )

    def get_api_key(self) -> str:
        """
        Retrieve the API key for Anthropic from the environment.

        Returns:
            The Anthropic API key.

        Raises:
            ValueError: If the ANTHROPIC_API_KEY is not found in environment variables.
        """
        # check if api_key is set
        if self.init_kwargs.get("api_key", None):
            return self.init_kwargs["api_key"]

        self.logger.debug(
            "Attempting to get Anthropic API key from environment variables"
        )
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if api_key:
            return api_key

        # try to load from key path
        self.logger.debug("Attempting to get Anthropic API key from key file")
        if DEFAULT_KEY_PATH.exists():
            api_key = DEFAULT_KEY_PATH.read_text().strip()
            if len(api_key) > 0:
                return api_key

        raise ValueError(
            "ANTHROPIC_API_KEY not found in environment variables or key file."
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
            raise ValueError(
                "No URL provided for request; must provide either as args[0] or url=... keyword arg."
            )

        # pop headers from kwargs
        headers = kwargs.pop("headers", {})

        # set Authorization if not already set
        if "Authorization" not in headers:
            headers["x-api-key"] = self.get_api_key()
            headers["anthropic-version"] = self.init_kwargs.get(
                "anthropic_version", DEFAULT_VERSION
            )

        self.logger.debug("Making request to the API")
        try:
            # set any missing DEFAULT_KWARGS
            for k, v in DEFAULT_KWARGS.items():
                if k not in kwargs:
                    kwargs[k] = v

            # make and raise here
            response = self.client.post(
                url,
                json={
                    "model": self.model,
                    **kwargs,
                },
                headers=headers,
            )

            # check for 400 to return the right ALEA error
            if response.status_code == 400:
                error_message = (
                    response.json().get("error", {}).get("message", response.json())
                )
                raise ALEAModelError(f"Model error: {error_message}")

            # check for 401 to return the right ALEA error
            if response.status_code == 401:
                error_message = (
                    response.json().get("error", {}).get("message", response.json())
                )
                raise ALEAAuthenticationError(f"Authentication error: {error_message}")

            response.raise_for_status()
            return response
        except (ALEAModelError, ALEAAuthenticationError) as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error in request: {str(e)}")
            raise ALEAModelError(f"Error in request: {str(e)}") from e

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
            raise ValueError(
                "No URL provided for request; must provide either as args[0] or url=... keyword arg."
            )

        # pop headers from kwargs
        headers = kwargs.pop("headers", {})

        # set Authorization if not already set
        if "Authorization" not in headers:
            headers["x-api-key"] = self.get_api_key()
            headers["anthropic-version"] = self.init_kwargs.get(
                "anthropic_version", DEFAULT_VERSION
            )

        self.logger.debug("Making asynchronous request to the API")
        try:
            # set any missing DEFAULT_KWARGS
            for k, v in DEFAULT_KWARGS.items():
                if k not in kwargs:
                    kwargs[k] = v

            # make and raise here
            response = await self.async_client.post(
                url,
                json={
                    "model": self.model,
                    **kwargs,
                },
                headers=headers,
            )

            # check for 400 to return the right ALEA error
            if response.status_code == 400:
                error_message = (
                    response.json().get("error", {}).get("message", response.json())
                )
                raise ALEAModelError(f"Model error: {error_message}")

            # check for 401 to return the right ALEA error
            if response.status_code == 401:
                error_message = (
                    response.json().get("error", {}).get("message", response.json())
                )
                raise ALEAAuthenticationError(f"Authentication error: {error_message}")

            response.raise_for_status()
            return response
        except (ALEAModelError, ALEAAuthenticationError) as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error in asynchronous request: {str(e)}")
            raise ALEAModelError(f"Error in asynchronous request: {str(e)}") from e

    def format(self, args: Any, kwargs: Any) -> List[Dict[str, str]]:
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
    def _get_complete_choices(response_data: dict) -> List[str | dict]:
        """
        Get the response choices from the response data.

        Args:
            response_data: The response data.

        Returns:
            The response choices.
        """
        return [choice.get("text") for choice in response_data.get("choices", [])]

    @staticmethod
    def _get_chat_choices(response_data: dict) -> List[str | dict]:
        """
        Get the chat response choices from the response data.

        Args:
            response_data: The response data.

        Returns:
            The response choices.
        """
        messages = response_data.get("content", [])
        return [messages[0].get("text", "") if len(messages) > 0 else ""]
