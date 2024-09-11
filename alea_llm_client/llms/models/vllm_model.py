"""VLLM model implementation for the LLM.

This module provides an implementation of the BaseAIModel for OpenAI-compatible
VLLM API. It includes classes and methods for both synchronous and asynchronous
chat and JSON completions using OpenAI's language models.
"""

import hashlib

# Standard library imports
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

# project imports
from alea_llm_client.core.logging import LoggerMixin
from alea_llm_client.llms.models.base_ai_model import (
    JSONModelResponse,
)
from alea_llm_client.llms.models.openai_compatible_model import OpenAICompatibleModel

DEFAULT_ENDPOINT = "http://localhost:8000/"
DEFAULT_CACHE_PATH = Path.home() / ".alea" / "cache" / "vllm"
DEFAULT_KEY_PATH = Path.home() / ".alea" / "keys" / "vllm"


class VLLMModel(OpenAICompatibleModel, LoggerMixin):
    """
    VLLM model implementation.

    This class implements the BaseAIModel for OpenAI-compatible VLLM API, providing
    methods for both synchronous and asynchronous chat and JSON completions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        endpoint: Optional[str] = DEFAULT_ENDPOINT,
        formatter: Optional[Callable] = None,
        cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the VLLM model.

        Args:
            api_key: The API key for OpenAI. If None, it will be retrieved from environment variables.
            model: The name of the VLLM model to use.
            endpoint: The API endpoint URL (if different from default).
            formatter: A function to format input messages.
            cache_path: The path to the cache directory for storing model responses.
        """
        # add the endpoint and model hashes to the cache path
        endpoint_hash = hashlib.blake2b(endpoint.encode()).hexdigest()
        model_hash = hashlib.blake2b(model.encode()).hexdigest()
        cache_path = cache_path / endpoint_hash / model_hash
        cache_path.mkdir(parents=True, exist_ok=True)

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
            self.logger.info(f"Initialized VLLMModel with model: {model}")
        else:
            self.logger.info(
                f"Initialized VLLMModel with model: {model} and endpoint: {endpoint}"
            )

    def get_api_key(self) -> str:
        """
        Retrieve the API key for VLLM from the environment.

        Returns:
            The VLLM API key, which is just a placeholder.
        """
        # we don't need one for VLLM
        return "key"

    def _json(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """
        Perform a JSON completion with the VLLM model.

        Args:
            *args: The input arguments to pass to the model.
            **kwargs: The input keyword arguments to pass to the model.

        Returns:
            JSONModelResponse: The response from the model.
        """
        # add response_format to kwargs and call super method
        kwargs["response_format"] = {"type": "json_object"}
        return super()._json(*args, **kwargs)

    async def _json_async(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """
        Perform an asynchronous JSON completion with the VLLM model.

        Args:
            *args: The input arguments to pass to the model.
            **kwargs: The input keyword arguments to pass to the model.

        Returns:
            JSONModelResponse: The response from the model.
        """
        # add response_format to kwargs and call super method
        kwargs["response_format"] = {"type": "json_object"}
        return await super()._json_async(*args, **kwargs)

    def _pydantic(self, *args: Any, **kwargs: Any) -> BaseModel:
        """
        Perform a Pydantic completion with the VLLM model.

        Args:
            *args: The input arguments to pass to the model.
            **kwargs: The input keyword arguments to pass to the model.

        Returns:
            BaseModel: The response from the model.
        """
        # add response_format to kwargs and call super method
        kwargs["response_format"] = {"type": "json_object"}
        return super()._pydantic(*args, **kwargs)

    async def _pydantic_async(self, *args: Any, **kwargs: Any) -> BaseModel:
        """
        Perform an asynchronous Pydantic completion with the VLLM model.

        Args:
            *args: The input arguments to pass to the model.
            **kwargs: The input keyword arguments to pass to the model.

        Returns:
            BaseModel: The response from the model.
        """
        # add response_format to kwargs and call super method
        kwargs["response_format"] = {"type": "json_object"}
        return await super()._pydantic_async(*args, **kwargs)
