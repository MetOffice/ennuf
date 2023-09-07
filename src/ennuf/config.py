#  (C) Crown Copyright, Met Office, 2023.
from ennuf.formatters.um_formatter import UMFormatter


class _EnnufConfig:
    default_formatter = UMFormatter()


CONFIG = _EnnufConfig()
