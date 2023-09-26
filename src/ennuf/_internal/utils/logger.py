#  (C) Crown Copyright, Met Office, 2023.
"""Module for defining logger creation for the project."""
import logging
import os
from typing import Callable


def create_logger(level: int = logging.INFO) -> logging.Logger:
    """Returns a logger with the name 'ennuf' with the specified log level."""
    logger = logging.getLogger("ennuf")

    logger.setLevel(level)

    logger.addHandler(_get_handler())

    return logger


def _get_handler() -> logging.Handler:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s %(parent_process)d:%(process)d %(levelname)s - %(message)s (%(pathname)s:%(lineno)s)"
    )
    handler.setFormatter(formatter)

    factory = _get_log_record_factory()
    logging.setLogRecordFactory(factory)

    return handler


def _get_log_record_factory() -> Callable[[], logging.LogRecord]:
    base_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = base_factory(*args, **kwargs)
        record.parent_process = os.getppid()
        return record

    return factory
