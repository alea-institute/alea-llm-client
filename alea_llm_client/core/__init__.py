"""
Core functionality used across alea_llm_client modules.
"""

from .exceptions import (
    ALEAModelError,
    ALEAError,
    ALEARetryExhaustedError,
    ALEAAuthenticationError,
)
from .logging import LoggerMixin, setup_logger, DEFAULT_LOGGER

__all__ = [
    "ALEAModelError",
    "ALEAError",
    "ALEARetryExhaustedError",
    "ALEAAuthenticationError",
    "LoggerMixin",
    "setup_logger",
    "DEFAULT_LOGGER",
]
