"""
Core functionality used across alea_llm_client modules.
"""

from .exceptions import (
    ALEAAuthenticationError,
    ALEAError,
    ALEAModelError,
    ALEARetryExhaustedError,
)
from .logging import DEFAULT_LOGGER, LoggerMixin, setup_logger

__all__ = [
    "DEFAULT_LOGGER",
    "ALEAAuthenticationError",
    "ALEAError",
    "ALEAModelError",
    "ALEARetryExhaustedError",
    "LoggerMixin",
    "setup_logger",
]
