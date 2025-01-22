#  (C) Crown Copyright, Met Office, 2025.

from ennuf._internal.formatters.base_formatter import BaseFormatter


class MetOfficeMinimalistFormatter(BaseFormatter):
    """Formatter which only adds the bare minimum to the Fortran file for it to compile, no project-specific stuff."""

    _maxlinelength = 80

    @property
    def default_dtype(self) -> str:
        return "4"

    def required_file_header(self) -> str:
        return self._get_met_office_copyright()
