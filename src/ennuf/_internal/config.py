#  (C) Crown Copyright, Met Office, 2023.
from ennuf._internal.formatters import UMFormatter
from ennuf._internal.utils.logger import create_logger


class _EnnufConfig:
    default_formatter = UMFormatter()
    logger = create_logger()
    compiler = "gfortran"


CONFIG = _EnnufConfig()
