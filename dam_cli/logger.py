"""
Logging utilities for the DAM CLI Wrapper
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file, if None, logs only to console
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("dam_cli")
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_error(logger: logging.Logger, message: str) -> None:
    """
    Log an error message.
    
    Args:
        logger: Logger instance
        message: Error message
    """
    logger.error(message)


def log_warning(logger: logging.Logger, message: str) -> None:
    """
    Log a warning message.
    
    Args:
        logger: Logger instance
        message: Warning message
    """
    logger.warning(message)


def log_info(logger: logging.Logger, message: str) -> None:
    """
    Log an info message.
    
    Args:
        logger: Logger instance
        message: Info message
    """
    logger.info(message)


def log_debug(logger: logging.Logger, message: str) -> None:
    """
    Log a debug message.
    
    Args:
        logger: Logger instance
        message: Debug message
    """
    logger.debug(message)


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
