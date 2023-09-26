#  (C) Crown Copyright, Met Office, 2023.
from abc import ABC, abstractmethod

import numpy as np


class BaseFormatter(ABC):
    @abstractmethod
    def format_line(self, line: str) -> str:
        pass

    @abstractmethod
    def format_data_statement(self, varname: str, data: np.ndarray) -> str:
        pass

    @property
    def default_dtype(self) -> str:
        return ""

    def required_file_header(self) -> str:
        """Inserted at the top of the file"""
        return ""

    def required_module_imports(self, *args, **kwargs) -> str:
        """Inserted after MODULE statement."""
        return ""

    def required_module_declarations(self, *args, **kwargs) -> str:
        """Inserted after imports."""
        return ""

    def required_subroutine_imports(self, *args, **kwargs) -> str:
        """Inserted after SUBROUTINE statement."""
        return ""

    def required_subroutine_declarations(self, *args, **kwargs) -> str:
        """Inserted after SUBROUTINE statement."""
        return ""

    def required_subroutine_opening_actions(self, *args, **kwargs) -> str:
        """Inserted after declarations and initialisations but before code."""
        return ""

    def required_subroutine_closing_actions(self, *args, **kwargs) -> str:
        """Inserted after declarations and initialisations but before code."""
        return ""
