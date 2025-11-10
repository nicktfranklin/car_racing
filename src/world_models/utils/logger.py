"""
Centralized logging configuration for World Models.

This module provides a consistent logging interface across the codebase.
Use logger.info() for important events, logger.debug() for detailed info.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "world_models",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "world_models") -> logging.Logger:
    """
    Get a logger instance. Will use the root logger configuration.

    Args:
        name: Logger name (use __name__ from calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
