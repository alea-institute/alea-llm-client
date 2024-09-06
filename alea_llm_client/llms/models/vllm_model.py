"""VLLM model implementation for the LLM.

This module provides an implementation of the BaseAIModel for VLLM's OpenAI-compatible
API. It includes classes and methods for both synchronous and asynchronous chat
and JSON completions using OpenAI's language models.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Callable, Optional

# packages
from pydantic import BaseModel

# Local imports
from alea_llm_client.llms.models import OpenAIModel
from alea_llm_client.llms.models.base_ai_model import JSONModelResponse


DEFAULT_CACHE_PATH = Path.home() / ".alea" / "cache" / "vllm"


class VLLMModel(OpenAIModel):
    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        endpoint: Optional[str] = "http://localhost:8000/v1",
        formatter: Optional[Callable] = None,
        cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
    ) -> None:
        """
        Initialize the VLLMModel with the specified model and endpoint.

        Args:
            model: The name of the OpenAI model to use.
            endpoint: The API endpoint URL (if different from default).
            formatter: A function to format input messages.
            cache_path: The path to the cache directory for storing model responses.
        """

        # append to the cache path
        cache_path = cache_path / endpoint / model
        cache_path.mkdir(parents=True, exist_ok=True)

        super().__init__(
            model=model,
            endpoint=endpoint,
            formatter=formatter,
            cache_path=cache_path,
            api_key="key",
        )

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

        # check if last message doesn't end with ```json
        if not messages[-1]["content"].endswith("\n```json\n"):
            messages[-1]["content"] += "\n```json\n"

        try:
            # ensure that we set the ```json start and \n``` end tokens
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                stop=["\n```\n"],
                **kwargs,
            )
            return self._handle_json_response(response)
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

        # check if last message doesn't end with ```json
        if not messages[-1]["content"].endswith("\n```json\n"):
            messages[-1]["content"] += "\n```json\n"

        try:
            # ensure that we set the ```json start and \n``` end tokens
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                stop=["\n```\n"],
                **kwargs,
            )
            return self._handle_json_response(response)
        except Exception as e:
            self.logger.error(f"Error in asynchronous JSON completion: {str(e)}")
            raise

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

        # check if last message doesn't end with ```json
        if not messages[-1]["content"].endswith("\n```json\n"):
            messages[-1]["content"] += "\n```json\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                stop=["\n```\n"],
                **kwargs,
            )
            return self._handle_pydantic_response(response, pydantic_model)
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

        # check if last message doesn't end with ```json
        if not messages[-1]["content"].endswith("\n```json\n"):
            messages[-1]["content"] += "\n```json\n"

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                stop=["\n```\n"],
                **kwargs,
            )
            self.logger.debug(f"Received response: {response}")
            return self._handle_pydantic_response(response, pydantic_model)
        except Exception as e:
            self.logger.error(f"Error in JSON completion: {str(e)}")
            raise

    def __str__(self) -> str:
        return f"VLLMModel(model={self.model}, endpoint={self.endpoint})"
