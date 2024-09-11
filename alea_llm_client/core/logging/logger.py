"""
This module provides a logger factory and default logger for the project.
"""

# imports
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

# Create a default logger for the entire project
DEFAULT_LOG_DIR = Path.home() / ".alea" / "logs"
if not DEFAULT_LOG_DIR.exists():
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "alea_llm_client.log"


def setup_logger(
    name: str, log_file: Optional[Path] = DEFAULT_LOG_FILE, level: int = logging.WARN
) -> logging.Logger:
    """Set up and return a logger with the given name and configuration.

    Args:
        name (str): The name of the logger.
        log_file (Optional[Path]): The path to the log file. If None, logs will only be printed to console.
        level (int): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        # Create a file handler for logging to a file
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file.absolute()), maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create the default logger for the project
DEFAULT_LOGGER = setup_logger("alea", DEFAULT_LOG_FILE)
