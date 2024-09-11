"""Base model for AI language models.

This module defines the abstract base class for AI language models and related data structures.
It provides a common interface for different AI model implementations, such as OpenAI and Anthropic.
"""

# Standard library imports
import abc
import asyncio
import gzip
import hashlib
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# packages
import pydantic_core
from pydantic import BaseModel

# project imports
from alea_llm_client.core.exceptions import (
    ALEARetryExhaustedError,
    ALEAAuthenticationError,
    ALEAModelError,
)
from alea_llm_client.core.logging import LoggerMixin
from alea_llm_client.llms.utils.json import normalize_json_response


# Default keys to exclude from the cache key
DEFAULT_EXCLUDE_KEYS = ("ignore_cache",)


# enum for response type: ModelResponse, JSONModelResponse, PydanticModelResponse
class ResponseType(Enum):
    TEXT = "MODEL"
    JSON = "JSON"
    PYDANTIC = "PYDANTIC"


@dataclass
class ModelResponse:
    """Represents a response from an AI model.

    Attributes:
        choices (List[str]): A list of response choices from the model.
        metadata (Dict): Additional metadata about the response.
        text (Optional[str]): The primary text response, if applicable.
    """

    choices: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ModelResponse to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the ModelResponse.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Convert the ModelResponse to a JSON string.

        Returns:
            str: A JSON string representation of the ModelResponse.
        """
        return json.dumps(self.to_dict(), default=str)


@dataclass
class JSONModelResponse(ModelResponse):
    """Represents a JSON response from an AI model.

    Inherits from ModelResponse and adds a data field for structured JSON data.

    Attributes:
        data (Dict): The structured JSON data returned by the model.
    """

    data: Dict[str, Any] = field(default_factory=dict)


class BaseAIModel(abc.ABC, LoggerMixin):
    """
    Abstract base class for AI models.

    This class defines a common interface for various AI model implementations,
    such as OpenAI, Anthropic, and HuggingFace models.

    Attributes:
        api_key (str): The API key for authenticating with the model service.
        model (str): The specific model identifier to use.
        endpoint (Optional[str]): The API endpoint URL, if applicable.
        client (Any): The client instance for interacting with the API.
        async_client (Any): The asynchronous client instance for interacting with the API.
        formatter (Optional[Callable]): A function to format inputs/outputs.
        cache_path (Optional[Path]): The path to the cache directory; if none, caching is disabled.
        retry_limit (int): The number of times to retry the request on failure.
        retry_delay (float): The initial delay between retries, in seconds.
    """

    # The size of the cache digest hash
    CACHE_DIGEST_SIZE = 8

    def __init__(
        self,
        api_key: str,
        model: str,
        endpoint: Optional[str] = None,
        formatter: Optional[Callable] = None,
        cache_path: Optional[Path] = None,
        ignore_cache: bool = False,
        retry_limit: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the BaseAIModel.

        Args:
            api_key (str): The API key for authenticating with the model service.
            model (str): The specific model identifier to use.
            endpoint (Optional[str]): The API endpoint URL, if applicable.
            formatter (Optional[Callable]): A function to format inputs/outputs.
            cache_path (Optional[Path]): The path to the cache directory.
            ignore_cache (bool): Whether to ignore the cache and always make a request.
            retry_limit (int): The number of times to retry the request on failure.
            retry_delay (float): The initial delay between retries, in seconds.
        """
        LoggerMixin.__init__(self)
        self.api_key = api_key or self.get_api_key()
        self.model = model
        self.endpoint = endpoint
        self.client = self._initialize_client()
        self.async_client = self._initialize_async_client()
        self.formatter = formatter
        self.cache_path = cache_path
        self.ignore_cache = ignore_cache
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay

        # ensure cache path exists if provided
        if self.cache_path and not self.cache_path.exists():
            # create the cache directory if it doesn't exist
            self.cache_path.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize and return the client for the AI model.

        Returns:
            Any: The initialized client instance.

        Raises:
            AIModelError: If there's an error initializing the client.
        """
        pass

    @abc.abstractmethod
    def _initialize_async_client(self) -> Any:
        """Initialize and return the asynchronous client for the AI model.

        Returns:
            Any: The initialized asynchronous client instance.

        Raises:
            AIModelError: If there's an error initializing the async client.
        """
        pass

    def _short_hash(self, data: str | bytes) -> str:
        """Generate a short hash using the BLAKE2b algorithm.

        Args:
            data (str | bytes): The data to hash.

        Returns:
            str: The short hash of the data.
        """
        if len(data) == 0:
            return "0" * self.CACHE_DIGEST_SIZE

        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.blake2b(data, digest_size=self.CACHE_DIGEST_SIZE).hexdigest()

    def get_object_cache_path(
        self, args: dict, exclude_keys: tuple[str, ...] = DEFAULT_EXCLUDE_KEYS
    ) -> Path:
        """Generate a cache path for the given object.

        Args:
            args (dict): The input arguments to the model.
            exclude_keys (tuple[str, ...]): A tuple of keys to exclude from the cache key.

        Returns:
            Path: The cache path for the object.
        """
        # get a string representation of a stable sorted tuple of the arguments
        args_tuple = tuple((str(k), str(v)) for k, v in sorted(args.items()))
        arg_str = str(args_tuple).encode("utf-8")

        # generate a short hash of the arguments to use as the cache key
        hash_str = self._short_hash(arg_str)
        return self.cache_path / f"{hash_str}.json.gz"

    def get_cached_object(self, args: dict) -> Optional[Union[dict, list]]:
        """Retrieve a cached object from the cache directory.

        Args:
            args (dict): The input arguments to the model.

        Returns:
            Optional[Union[dict, list]]: The cached object, if found.
        """
        cache_path = self.get_object_cache_path(args)
        if cache_path.exists():
            self.logger.info(f"Loading cached object from {cache_path}")
            with gzip.open(cache_path, "rt") as file:
                return json.load(file)

        return None

    def set_cached_object(self, args: dict, obj: Union[dict, list]) -> None:
        """Save an object to the cache directory.

        Args:
            args (dict): The input arguments to the model.
            obj (Union[dict, list]): The object to save to the cache.
        """
        cache_path = self.get_object_cache_path(args)
        self.logger.info(f"Saving object to cache: {cache_path}")
        with gzip.open(cache_path, "wt") as file:
            json.dump(obj, file)

    def parse_json(self, data: str | bytes) -> Optional[Union[dict, list]]:
        """Parse JSON data from a string or bytes object.

        Args:
            data (str | bytes): The JSON data to parse.

        Returns:
            Optional[Union[dict, list]]: The parsed JSON data, if valid.
        """
        if isinstance(data, bytes):
            normalized_data = data.decode("utf-8")
        else:
            normalized_data = data

        # normalize the JSON data
        normalized_data = normalize_json_response(normalized_data)

        # try to parse with pydantic_core
        try:
            return pydantic_core.from_json(normalized_data, allow_partial=True)
        except Exception as e:
            self.logger.error(f"Error parsing JSON data: {str(e)}:\n{normalized_data}")
            return None

    @abc.abstractmethod
    def get_api_key(self) -> str:
        """Retrieve the API key for the model.

        Returns:
            str: The API key for authenticating with the model service.

        Raises:
            ValueError: If the API key is not found or invalid.
        """
        pass

    @abc.abstractmethod
    def format(self, args: Any, kwargs: Any) -> List[Dict[str, str]]:
        """Format inputs or outputs using the specified formatter.

        Args:
            args (Any): Positional arguments to be formatted.
            kwargs (Any): Keyword arguments to be formatted.

        Returns:
            List[Dict[str, str]]: A list of formatted message dictionaries.

        Raises:
            ValueError: If the input cannot be properly formatted.
        """
        pass

    def _retry_wrapper(
        self, func: Callable, response_type: ResponseType, *args: Any, **kwargs: Any
    ) -> Any:
        """Retry a function call a specified number of times on failure.

        Args:
            func (Callable): The function to call.
            response_type (ResponseType): The type of response to return.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the function call.

        Raises:
            RetryExhaustedError: If all retry attempts fail.
        """
        for attempt in range(self.retry_limit):
            try:
                # check the cache
                ignore_cache = kwargs.pop("ignore_cache", None) or self.ignore_cache
                pydantic_model: Optional[BaseModel] = kwargs.get("pydantic_model", None)
                cache_args = {
                    "response_type": response_type,
                    "model": self.model,
                    "args": args,
                    "kwargs": kwargs.copy(),
                }
                if not ignore_cache:
                    cached_response = self.get_cached_object(cache_args)
                    if cached_response:
                        if response_type == ResponseType.TEXT:
                            return ModelResponse(**cached_response)
                        elif response_type == ResponseType.JSON:
                            return JSONModelResponse(**cached_response)
                        elif response_type == ResponseType.PYDANTIC:
                            return pydantic_model.model_validate(cached_response)
                        else:
                            raise ValueError(f"Invalid return type: {response_type}")

                # call the function
                result = func(*args, **kwargs)

                # save the result to the cache and return it
                if self.cache_path and not ignore_cache:
                    if response_type in (ResponseType.TEXT, ResponseType.JSON):
                        self.set_cached_object(cache_args, result.to_dict())
                    elif response_type == ResponseType.PYDANTIC:
                        self.set_cached_object(cache_args, result.model_dump())
                return result
            except ALEAAuthenticationError as e:
                self.logger.error(f"Authentication error: {str(e)}")
                raise e
            except ALEAModelError as e:
                self.logger.error(f"Model error: {str(e)}")
                raise e
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}\n{traceback.format_exc()}"
                )
                kwargs["ignore_cache"] = True
                if attempt == self.retry_limit - 1:
                    raise ALEARetryExhaustedError(
                        f"All {self.retry_limit} retry attempts failed"
                    ) from e
            time.sleep(self.retry_delay * (2**attempt))

    async def _retry_wrapper_async(
        self, func: Callable, response_type: ResponseType, *args: Any, **kwargs: Any
    ) -> Any:
        """Retry an asynchronous function call a specified number of times on failure.

        Args:
            func (Callable): The asynchronous function to call.
            response_type (ResponseType): The type of response to return.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the asynchronous function call.

        Raises:
            RetryExhaustedError: If all retry attempts fail.
        """
        for attempt in range(self.retry_limit):
            try:
                # check the cache
                ignore_cache = kwargs.pop("ignore_cache", None) or self.ignore_cache
                pydantic_model: Optional[BaseModel] = kwargs.get("pydantic_model", None)
                cache_args = {
                    "response_type": response_type,
                    "model": self.model,
                    "args": args,
                    "kwargs": kwargs.copy(),
                }
                if not ignore_cache:
                    cached_response = self.get_cached_object(cache_args)
                    if cached_response:
                        if response_type == ResponseType.TEXT:
                            return ModelResponse(**cached_response)
                        elif response_type == ResponseType.JSON:
                            return JSONModelResponse(**cached_response)
                        elif response_type == ResponseType.PYDANTIC:
                            return pydantic_model.model_validate(cached_response)

                # call the function
                result = await func(*args, **kwargs)

                # save the result to the cache and return it
                if self.cache_path and not ignore_cache:
                    if response_type in (ResponseType.TEXT, ResponseType.JSON):
                        self.set_cached_object(cache_args, result.to_dict())
                    elif response_type == ResponseType.PYDANTIC:
                        self.set_cached_object(cache_args, result.model_dump())
                return await func(*args, **kwargs)
            except ALEAAuthenticationError as e:
                self.logger.error(f"Authentication error: {str(e)}")
                raise e
            except ALEAModelError as e:
                self.logger.error(f"Model error: {str(e)}")
                raise e
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}\n{traceback.format_exc()}"
                )
                kwargs["ignore_cache"] = True
                if attempt == self.retry_limit - 1:
                    raise ALEARetryExhaustedError(
                        f"All {self.retry_limit} retry attempts failed"
                    ) from e
            await asyncio.sleep(self.retry_delay * (2**attempt))

    # complete hooks
    def complete(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Perform a synchronous completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the completion.

        Raises:
            AIModelError: If there's an error during the completion process.
        """
        return self._retry_wrapper(self._complete, ResponseType.TEXT, *args, **kwargs)

    @abc.abstractmethod
    def _complete(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Perform a synchronous completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the completion.

        Raises:
            AIModelError: If there's an error during the completion process.
        """
        pass

    async def complete_async(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Perform an asynchronous completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the completion.

        Raises:
            AIModelError: If there's an error during the asynchronous completion process.
        """
        return await self._retry_wrapper_async(
            self._complete_async, ResponseType.TEXT, *args, **kwargs
        )

    @abc.abstractmethod
    async def _complete_async(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Perform an asynchronous completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the completion.

        Raises:
            AIModelError: If there's an error during the asynchronous completion process.
        """
        pass

    def chat(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Perform a synchronous chat completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the chat completion.

        Raises:
            AIModelError: If there's an error during the chat completion process.
        """
        return self._retry_wrapper(self._chat, ResponseType.TEXT, *args, **kwargs)

    @abc.abstractmethod
    def _chat(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Perform a synchronous chat completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the chat completion.

        Raises:
            AIModelError: If there's an error during the chat completion process.
        """
        pass

    async def chat_async(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Perform an asynchronous chat completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the chat completion.

        Raises:
            AIModelError: If there's an error during the asynchronous chat completion process.
        """
        return await self._retry_wrapper_async(
            self._chat_async, ResponseType.TEXT, *args, **kwargs
        )

    @abc.abstractmethod
    async def _chat_async(self, *args: Any, **kwargs: Any) -> ModelResponse:
        """Perform an asynchronous chat completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ModelResponse: The response from the chat completion.

        Raises:
            AIModelError: If there's an error during the asynchronous chat completion process.
        """
        pass

    def json(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """Perform a synchronous JSON completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            JSONModelResponse: The JSON response from the completion.

        Raises:
            AIModelError: If there's an error during the JSON completion process.
        """
        return self._retry_wrapper(self._json, ResponseType.JSON, *args, **kwargs)

    @abc.abstractmethod
    def _json(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """Perform a synchronous JSON completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            JSONModelResponse: The JSON response from the completion.

        Raises:
            AIModelError: If there's an error during the JSON completion process.
        """
        pass

    async def json_async(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """Perform an asynchronous JSON completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            JSONModelResponse: The JSON response from the completion.

        Raises:
            AIModelError: If there's an error during the asynchronous JSON completion process.
        """
        return await self._retry_wrapper_async(
            self._json_async, ResponseType.JSON, *args, **kwargs
        )

    @abc.abstractmethod
    async def _json_async(self, *args: Any, **kwargs: Any) -> JSONModelResponse:
        """Perform an asynchronous JSON completion.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            JSONModelResponse: The JSON response from the completion.

        Raises:
            AIModelError: If there's an error during the asynchronous JSON completion process.
        """
        pass

    def pydantic(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Perform a synchronous JSON completion with Pydantic validation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pydantic.BaseModel: The Pydantic model response from the completion.

        Raises:
            AIModelError: If there's an error during the Pydantic validation process.
        """
        return self._retry_wrapper(
            self._pydantic, ResponseType.PYDANTIC, *args, **kwargs
        )

    @abc.abstractmethod
    def _pydantic(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Perform a synchronous JSON completion with Pydantic validation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pydantic.BaseModel: The Pydantic model response from the completion.

        Raises:
            AIModelError: If there's an error during the Pydantic validation process.
        """
        pass

    async def pydantic_async(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Perform an asynchronous JSON completion with Pydantic validation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pydantic.BaseModel: The Pydantic model response from the completion.

        Raises:
            AIModelError: If there's an error during the asynchronous Pydantic validation process.
        """
        return await self._retry_wrapper_async(
            self._pydantic_async, ResponseType.PYDANTIC, *args, **kwargs
        )

    @abc.abstractmethod
    async def _pydantic_async(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Perform an asynchronous JSON completion with Pydantic validation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pydantic.BaseModel: The Pydantic model response from the completion.

        Raises:
            AIModelError: If there's an error during the asynchronous Pydantic validation process.
        """
        pass
