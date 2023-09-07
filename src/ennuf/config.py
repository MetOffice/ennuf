#  (C) Crown Copyright, Met Office, 2023.
from ennuf.formatters.um_formatter import UMFormatter
from ennuf.utils.logger import create_logger


class _EnnufConfig:
    default_formatter = UMFormatter()
    logger = create_logger()


CONFIG = _EnnufConfig()
