#  (C) Crown Copyright, Met Office, 2023.
"""Module for project configuration."""
from ennuf._internal.formatters import UMFormatter
from ennuf._internal.utils.logger import create_logger


class _EnnufConfig:
    """Defines the project configuration."""

    default_formatter = UMFormatter()
    logger = create_logger()
    compiler = "gfortran"


CONFIG = _EnnufConfig()
"""Global config variable for ennuf. All config options can be set through this."""
