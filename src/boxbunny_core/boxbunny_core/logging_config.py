"""Logging configuration for BoxBunny.

Sets up rotating file handlers and structured formatting.
No print() statements allowed in production code -- use this logger.
"""
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_dir: str = "~/.boxbunny/logs",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure BoxBunny logging with rotating file handler."""
    log_dir = os.path.expanduser(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "boxbunny.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper()))

    root_logger = logging.getLogger("boxbunny")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    root_logger.info("BoxBunny logging initialized")
    return root_logger
