"""
This module contains a mixin class that adds logging functionality to any class.
"""

# project imports
from alea_llm_client.core.logging import setup_logger


class LoggerMixin:
    """A mixin class to add logging functionality to any class."""

    def __init__(self, *args, **kwargs):
        """Initialize the LoggerMixin.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Attributes:
            logger_name (str): The name of the logger.
            logger (Logger): The logger instance.

        Example:
            >>> class MyClass(LoggerMixin):
            ...     def __init__(self):
            ...         super().__init__()
        """
        # Set up the logger
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # Create a logger instance
        self.logger = setup_logger(logger_name)

        # Call the superclass constructor
        super().__init__(*args, **kwargs)
