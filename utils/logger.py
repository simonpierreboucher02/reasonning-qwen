"""
Logging utilities for the reasoning model.

This module provides simple logging configuration for tracking model
training, evaluation, and inference.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "reasoning_model",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup and configure a logger.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("my_model", level=logging.DEBUG)
        >>> logger.info("Model initialized")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "reasoning_model") -> logging.Logger:
    """Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger()
        >>> logger.info("Processing batch...")
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger
