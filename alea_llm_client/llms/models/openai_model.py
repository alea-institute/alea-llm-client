"""OpenAI model implementation for the LLM.

This module provides an implementation of the BaseAIModel for OpenAI's API.
It includes classes and methods for both synchronous and asynchronous chat
and JSON completions using OpenAI's language models.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Third-party imports
from openai import AsyncOpenAI, OpenAI, AuthenticationError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from alea_llm_client.core.exceptions import ALEAAuthenticationError
from alea_llm_client.core.logging import LoggerMixin

# Local imports
from alea_llm_client.llms.models.base_ai_model import (
    BaseAIModel,
    JSONModelResponse,
    ModelResponse,
)

DEFAULT_CACHE_PATH = Path.home() / ".alea" / "cache" / "openai"


class OpenAIModel(BaseAIModel, LoggerMixin):
    """
    OpenAI model implementation.

    This class implements the BaseAIModel for OpenAI's API, providing methods
    for both synchronous and asynchronous chat and JSON completions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        endpoint: Optional[str] = None,
        formatter: Optional[Callable] = None,
        cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
    ) -> None:
        """
        Initialize the OpenAIModel.

        Args:
            api_key: The API key for OpenAI. If None, it will be retrieved from environment variables.
            model: The name of the OpenAI model to use.
            endpoint: The API endpoint URL (if different from default).
            formatter: A function to format input messages.
            cache_path: The path to the cache directory for storing model responses.
        """
        BaseAIModel.__init__(
            self,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            formatter=formatter,
            cache_path=cache_path,
        )
        if endpoint is None:
            self.logger.info(f"Initialized OpenAIModel with model: {model}")
        else:
            self.logger.info(
                f"Initialized OpenAIModel with model: {model} and endpoint: {endpoint}"
            )

    def get_api_key(self) -> str:
        """
        Retrieve the API key for OpenAI from the environment.

        Returns:
            The OpenAI API key.

        Raises:
            ValueError: If the OPENAI_API_KEY is not found in environment variables.
        """
        self.logger.debug("Retrieving OpenAI API key from environment")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        return api_key

    def _initialize_client(self) -> OpenAI:
        """
        Initialize and return the OpenAI client.

        Returns:
            An instance of the OpenAI client.
        """
        self.logger.debug("Initializing OpenAI client")
        args = {"api_key": self.api_key}
        if self.endpoint:
            args["base_url"] = self.endpoint
        return OpenAI(**args)

    def _initialize_async_client(self) -> AsyncOpenAI:
        """
        Initialize and return the AsyncOpenAI client.

        Returns:
            An instance of the AsyncOpenAI client.
        """
        self.logger.debug("Initializing AsyncOpenAI client")
        args = {"api_key": self.api_key}
        if self.endpoint:
            args["base_url"] = self.endpoint
        return AsyncOpenAI(**args)

    def format(self, args: Any, kwargs: Any) -> List[Dict[str, str]]:
        """
        Format inputs or outputs using the specified formatter.

        Args:
            args: Positional arguments passed to the chat method.
            kwargs: Keyword arguments passed to the chat method.

        Returns:
            A list of formatted message dictionaries.

        Raises:
            ValueError: If no messages are provided for chat completion.
        """
        self.logger.debug("Formatting input for OpenAI API")
        if self.formatter:
            return self.formatter(args, kwargs)

        messages = kwargs.pop("messages", None)
        if not messages:
            if len(args) > 0:
                messages = [{"role": "user", "content": args[0]}]
            else:
                self.logger.error("No messages provided for chat completion")
                raise ValueError("No messages provided for chat completion.")

        system = kwargs.pop("system", None)
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(messages)

        self.logger.debug(f"Formatted messages: {formatted_messages}")
        return formatted_messages

    def _handle_chat_response(self, response: ChatCompletion) -> ModelResponse:
        """
        Handle the response from the chat completion.

        Args:
            response: The response from the chat completion.

        Returns:
            The model response from the chat completion.
        """
        self.logger.debug("Handling chat response")
        model_response = ModelResponse(
            choices=[choice.message.content for choice in response.choices],
            metadata={"model": self.model, "usage": response.usage.model_dump()},
            text=response.choices[0].message.content,
        )
        self.logger.debug(f"Model response: {model_response}")
        return model_response

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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            return self._handle_chat_response(response)
        except AuthenticationError as e:
            self.logger.error(f"Error in synchronous chat completion: {e}")
            raise ALEAAuthenticationError(
                "Error in synchronous chat completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in chat completion: {str(e)}")
            raise

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
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            return self._handle_chat_response(response)
        except AuthenticationError as e:
            self.logger.error(f"Error in asynchronous chat completion: {e}")
            raise ALEAAuthenticationError(
                "Error in asynchronous chat completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in asynchronous chat completion: {str(e)}")
            raise

    def _handle_json_response(self, response: ChatCompletion) -> JSONModelResponse:
        """
        Handle the response from the JSON completion.

        Args:
            response: The response from the JSON completion.

        Returns:
            The JSON response from the JSON completion.
        """
        self.logger.debug("Handling JSON response")
        json_data = self.parse_json(response.choices[0].message.content)
        json_model_response = JSONModelResponse(
            choices=[choice.message.content for choice in response.choices],
            metadata={"model": self.model, "usage": response.usage.model_dump()},
            text=response.choices[0].message.content,
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
        kwargs.pop("response_format", None)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs,
            )
            return self._handle_json_response(response)
        except AuthenticationError as e:
            self.logger.error(f"Error in synchronous JSON completion: {e}")
            raise ALEAAuthenticationError(
                "Error in synchronous JSON completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in JSON completion: {str(e)}")
            raise

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
        kwargs.pop("response_format", None)
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs,
            )
            return self._handle_json_response(response)
        except AuthenticationError as e:
            self.logger.error(f"Error in asynchronous JSON completion: {e}")
            raise ALEAAuthenticationError(
                "Error in asynchronous JSON completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in asynchronous JSON completion: {str(e)}")
            raise

    def _handle_pydantic_response(
        self, response: ChatCompletion, pydantic_model: BaseModel
    ) -> BaseModel:
        """
        Handle the response from the Pydantic completion.

        Args:
            response: The response from the Pydantic completion.
            pydantic_model: The Pydantic model to validate the response.

        Returns:
            The Pydantic model response from the Pydantic completion.
        """
        self.logger.debug("Handling Pydantic response")
        json_data = self.parse_json(response.choices[0].message.content)
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
        kwargs.pop("response_format", None)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs,
            )
            return self._handle_pydantic_response(response, pydantic_model)
        except AuthenticationError as e:
            self.logger.error(f"Error in synchronous Pydantic completion: {e}")
            raise ALEAAuthenticationError(
                "Error in synchronous Pydantic completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in JSON completion: {str(e)}")
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
        kwargs.pop("response_format", None)
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs,
            )
            self.logger.debug(f"Received response: {response}")
            return self._handle_pydantic_response(response, pydantic_model)
        except AuthenticationError as e:
            self.logger.error(f"Error in asynchronous Pydantic completion: {e}")
            raise ALEAAuthenticationError(
                "Error in asynchronous Pydantic completion: {e}"
            ) from e
        except Exception as e:
            self.logger.error(f"Error in JSON completion: {str(e)}")
            raise
