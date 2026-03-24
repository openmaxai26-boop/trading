"""
Structured logging utility for the trading system.
Uses loguru for rich, structured logs with file rotation.
"""

import sys
import os
from loguru import logger


def setup_logger(log_level: str = "INFO", log_dir: str = "./logs") -> None:
    """Configure loguru logger with console and file handlers."""
    os.makedirs(log_dir, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — general logs
    logger.add(
        os.path.join(log_dir, "trading_{time:YYYY-MM-DD}.log"),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        rotation="00:00",       # Rotate at midnight
        retention="30 days",
        compression="gz",
    )

    # Error-only file
    logger.add(
        os.path.join(log_dir, "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}\n{exception}",
        rotation="00:00",
        retention="90 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
    )


def get_logger(name: str):
    """Return a logger bound to a specific module name."""
    return logger.bind(name=name)


# Initialize with defaults on import
setup_logger(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "./logs"),
)
