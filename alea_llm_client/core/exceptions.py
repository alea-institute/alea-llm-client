"""Custom exception classes for the alea package."""


class ALEAError(Exception):
    """Base exception class for all alea package errors."""

    pass


class ALEAModelError(ALEAError):
    """Exception raised for errors in AI model operations."""

    pass


class ALEAAuthenticationError(ALEAModelError):
    """Exception raised for authentication errors."""

    pass


class ALEARetryExhaustedError(ALEAModelError):
    """Exception raised when all retry attempts have been exhausted."""

    pass
