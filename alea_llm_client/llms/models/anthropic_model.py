"""Anthropic model implementation for the LLM.

This module provides an implementation of the BaseAIModel for Anthropic's API.
It includes classes and methods for both synchronous and asynchronous chat
and JSON completions using Anthropic's language models.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Third-party imports
from anthropic import Anthropic, AsyncAnthropic, AuthenticationError
from anthropic.types import Message
from pydantic import BaseModel

# Local imports
from alea_llm_client.core.exceptions import ALEAAuthenticationError
from alea_llm_client.llms.models.base_ai_model import (
    BaseAIModel,
    JSONModelResponse,
    ModelResponse,
)

# Required constants
DEFAULT_MAX_TOKENS = 4096


DEFAULT_CACHE_PATH = Path.home() / ".alea" / "cache" / "anthropic"


class AnthropicModel(BaseAIModel):
    """
    Anthropic model implementation.

    This class implements the BaseAIModel for Anthropic's API, providing methods
    for both synchronous and asynchronous chat and JSON completions.

    Attributes:
        api_key (str): The API key for authenticating with Anthropic's API.
        model (str): The name of the Anthropic model to use.
        endpoint (Optional[str]): The API endpoint URL (if different from default).
        formatter (Optional[Callable]): A function to format input messages.
        client (Anthropic): The Anthropic client for making synchronous API calls.
        async_client (AsyncAnthropic): The Anthropic client for making asynchronous API calls.

    Example:
        >>> model = AnthropicModel(api_key="your-api-key")
        >>> response = model.chat("Hello, how are you?")
        >>> print(response.text)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20240620",
        endpoint: Optional[str] = None,
        formatter: Optional[Callable] = None,
        cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
    ) -> None:
        """
        Initialize the AnthropicModel.

        Args:
            api_key (Optional[str]): The API key for Anthropic. If None, it will be retrieved from environment variables.
            model (str): The name of the Anthropic model to use.
            endpoint (Optional[str]): The API endpoint URL (if different from default).
            formatter (Optional[Callable]): A function to format input messages.
            cache_path (Optional[Path]): The path to the cache directory.

        Raises:
            ValueError: If the API key is not provided and cannot be found in environment variables.
        """
        BaseAIModel.__init__(
            self,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            formatter=formatter,
            cache_path=cache_path,
        )
        self.cache_path = cache_path
        self.logger.info(f"Initialized AnthropicModel with model: {model}")

    def _initialize_client(self) -> Anthropic:
        """Initialize and return the Anthropic client.

        Returns:
            Anthropic: An instance of the Anthropic client for synchronous operations.
        """
        self.logger.debug("Initializing Anthropic client")
        return Anthropic(api_key=self.api_key)

    def _initialize_async_client(self) -> AsyncAnthropic:
        """Initialize and return the AsyncAnthropic client.

        Returns:
            AsyncAnthropic: An instance of the AsyncAnthropic client for asynchronous operations.
        """
        self.logger.debug("Initializing AsyncAnthropic client")
        return AsyncAnthropic(api_key=self.api_key)

    def get_api_key(self) -> str:
        """Retrieve the API key for Anthropic from the environment.

        Returns:
            str: The Anthropic API key.

        Raises:
            ValueError: If the ANTHROPIC_API_KEY is not found in environment variables.
        """
        self.logger.debug("Retrieving Anthropic API key from environment")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.logger.error("ANTHROPIC_API_KEY not found in environment variables")
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        return api_key

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

    def _handle_chat_response(self, response: Message) -> ModelResponse:
        """Handle the chat response from the Anthropic API.

        Args:
            response (Message): The response message from the API.

        Returns:
            ModelResponse: The formatted model response.
        """
        self.logger.debug("Handling chat response from Anthropic API")
        model_response = ModelResponse(
            choices=[response.content[0].text],
            metadata={"model": self.model, "usage": response.usage.to_dict()},
            text=response.content[0].text,
        )
        self.logger.debug(f"Model response: {model_response}")
        return model_response

    def _chat(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Synchronous chat completion method.

        This method provides a synchronous interface to the chat completion
        functionality. It internally uses the Anthropic client to send a request
        to the API and process the response.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the chat completion.

        Raises:
            Exception: Any exceptions raised by the Anthropic API.

        Example:
            >>> model = AnthropicModel(api_key="your-api-key")
            >>> response = model.chat("What is the capital of France?")
            >>> print(response.text)
        """
        self.logger.info("Initiating synchronous chat completion")
        messages = self.format(args, kwargs)
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None) or []
        try:
            response: Message = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                system=system,
                **kwargs,
            )
            return self._handle_chat_response(response)
        except AuthenticationError as e:
            self.logger.error(f"Error in synchronous chat completion: {e}")
            raise ALEAAuthenticationError(
                "Error in synchronous chat completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in synchronous chat completion: {e}")
            raise e

    async def _chat_async(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Asynchronous chat completion method.

        This method provides an asynchronous interface to the chat completion
        functionality. It internally uses the AsyncAnthropic client to send a
        request to the API and process the response.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the chat completion.

        Raises:
            Exception: Any exceptions raised by the Anthropic API.

        Example:
            >>> model = AnthropicModel(api_key="your-api-key")
            >>> response = await model.chat_async("What is the capital of France?")
            >>> print(response.text)
        """
        self.logger.info("Initiating asynchronous chat completion")
        messages = self.format(args, kwargs)
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None) or []
        try:
            response: Message = await self.async_client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                system=system,
                **kwargs,
            )
            return self._handle_chat_response(response)
        except AuthenticationError as e:
            self.logger.error(f"Error in asynchronous chat completion: {e}")
            raise ALEAAuthenticationError(
                "Error in asynchronous chat completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in asynchronous chat completion: {e}")
            raise e

    def _handle_json_response(self, response: Message) -> JSONModelResponse:
        """
        Handle the response from the JSON completion.

        Args:
            response (Message): The response message from the API.

        Returns:
            The JSON response from the JSON completion.
        """
        self.logger.debug("Handling JSON response")
        json_data = self.parse_json(response.content[0].text)
        json_model_response = JSONModelResponse(
            choices=[response.content[0].text],
            metadata={"model": self.model, "usage": response.usage.model_dump()},
            text=response.content[0].text,
            data=json_data,
        )
        self.logger.debug(f"JSON model response: {json_model_response}")
        return json_model_response

    def _json(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """
        Perform a JSON completion using the Anthropic API.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The JSON response from the JSON completion.
        """
        self.logger.info("Initiating JSON completion")
        messages = self.format(args, kwargs)
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None) or []
        try:
            response: Message = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                system=system,
                **kwargs,
            )
            return self._handle_json_response(response)
        except AuthenticationError as e:
            self.logger.error(f"Error in JSON completion: {e}")
            raise ALEAAuthenticationError("Error in JSON completion: {e}") from e
        except Exception as e:
            self.logger.error(f"Error in JSON completion: {e}")
            raise e

    async def _json_async(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """
        Perform an asynchronous JSON completion using the Anthropic API.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The JSON response from the JSON completion.
        """
        self.logger.info("Initiating asynchronous JSON completion")
        messages = self.format(args, kwargs)
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None) or []
        try:
            response: Message = await self.async_client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                system=system,
                **kwargs,
            )
            return self._handle_json_response(response)
        except AuthenticationError as e:
            self.logger.error(f"Error in asynchronous JSON completion: {e}")
            raise ALEAAuthenticationError(
                "Error in asynchronous JSON completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in asynchronous JSON completion: {e}")
            raise e

    def _handle_pydantic_response(
        self, response: Message, pydantic_model: BaseModel
    ) -> BaseModel:
        """
        Handle the response from the Pydantic completion.

        Args:
            response (Message): The response message from the API.
            pydantic_model (BaseModel): The Pydantic model to use for parsing the response.

        Returns:
            The Pydantic response from the Pydantic completion.
        """
        self.logger.debug("Handling Pydantic response")
        data = self.parse_json(response.content[0].text)
        pydantic_response = pydantic_model.model_validate(data)
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
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None) or []
        try:
            response: Message = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                system=system,
                **kwargs,
            )
            return self._handle_pydantic_response(response, pydantic_model)
        except AuthenticationError as e:
            self.logger.error(f"Error in pydantic completion: {e}")
            raise ALEAAuthenticationError("Error in pydantic completion: {e}") from e
        except Exception as e:
            self.logger.error(f"Error in pydantic completion: {str(e)}")
            raise

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
        max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        system = kwargs.pop("system", None) or []
        try:
            response: Message = await self.async_client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                system=system,
                **kwargs,
            )
            return self._handle_pydantic_response(response, pydantic_model)
        except AuthenticationError as e:
            self.logger.error(f"Error in pydantic completion: {e}")
            raise ALEAAuthenticationError("Error in pydantic completion: {e}") from e
        except Exception as e:
            self.logger.error(f"Error in JSON completion: {str(e)}")
            raise
