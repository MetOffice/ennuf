#  (C) Crown Copyright, Met Office, 2023.
from abc import ABC

import numpy as np

from ennuf._internal.utils.string_utils import split_except_in_single_quotes


class BaseFormatter(ABC):
    """Abstract Base Class for formatters"""

    _maxlinelength = 80

    @property
    def default_dtype(self) -> str:
        """Gives the default dtype, as it would be written in Fortran"""
        return ""

    @staticmethod
    def _get_met_office_copyright() -> str:
        return (
            "! *****************************COPYRIGHT*******************************\n"
            "! (C) Crown copyright Met Office. All rights reserved.\n"
            "! For further details please refer to the file COPYRIGHT.txt\n"
            "! which you should have received as part of this distribution.\n"
            "! *****************************COPYRIGHT*******************************\n"
        )

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

    def format_line(self, line: str) -> str:
        """Takes some Fortran and formats it into lines with appropriate styling according to the formatter."""
        iscomment = line.lstrip()[0] == "!"
        if iscomment:
            return self._format_comment_line(line)
        return self._format_code_line(line)

    def format_data_statement(self, varname: str, data: np.ndarray) -> str:
        """Formats an array and its name into Fortran DATA statements"""
        match len(data.shape):
            case 1:
                data_stmt = f"DATA {varname} / "
                for val in data:
                    data_stmt = f"{data_stmt}{val}, "
                data_stmt = data_stmt.rstrip().rstrip(",")
                data_stmt += " /"
                return self.format_line(data_stmt)
            case 2:
                data_stmts = "\n"
                for i, row in enumerate(data):
                    next_stmt = self.format_data_statement(varname=f"{varname}({i + 1}, :)", data=row)
                    data_stmts = f"{data_stmts}{next_stmt}\n"
                return data_stmts
            case 3:
                data_stmts = "\n"
                for i, row in enumerate(data):
                    for j, col in enumerate(row):
                        next_stmt = self.format_data_statement(varname=f"{varname}({i + 1}, {j + 1}, :)", data=col)
                        data_stmts = f"{data_stmts}{next_stmt}\n"
                return data_stmts
            case _:
                raise NotImplementedError(
                    f"Could not format array {varname}: "
                    f"no implementation for formatting array initialisation for arrays with "
                    f"the following number of dimensions: {len(data.shape)}"
                )

    def _format_comment_line(self, line: str):
        pieces = line.split()
        formatted_line = "! "
        code_block = ""
        for piece in pieces:
            # first check if this piece is too big to fit on one line even by itself
            min_possible_len = len(f"! {piece}")
            if min_possible_len > self._maxlinelength:
                raise ValueError(
                    "Length of comment fragment exceeded maximum permitted line"
                    " length and contained no "
                    "whitespace for the formatter to know where to split it.\n"
                    "Consider adding whitespace at appropriate place(s) if possible.\n"
                    f'Line length (with !): "{min_possible_len}"'
                    f' of maximum permitted "{self._maxlinelength}"\n'
                    f'Comment fragment:\n"{piece}"'
                )

            # Propose this piece being the end of this line.
            proposed_line = f"{formatted_line} {piece}".lstrip()

            # If this line is impossible because it is too long, we should start a new line.
            if len(proposed_line) > self._maxlinelength:
                formatted_line = f"{formatted_line}".lstrip()
                code_block = f"{code_block}{formatted_line}\n"
                # now we can reset formatted_line and use it for the next line
                formatted_line = "! "
            formatted_line = f"{formatted_line} {piece}"
        # append the last line (which we've already checked is below the max line length limit)
        if formatted_line:
            formatted_line = formatted_line.lstrip()
            code_block = f"{code_block}{formatted_line}\n"
        return code_block

    def _format_code_line(self, line: str):
        pieces = split_except_in_single_quotes(line)
        formatted_line = ""
        code_block = ""
        for piece in pieces:
            # first check if this piece is too big to fit on one line even by itself
            min_possible_len = len(f"{piece} &")
            if min_possible_len > self._maxlinelength:
                raise ValueError(
                    "Length of code fragment exceeded maximum permitted line length and contained no "
                    "whitespace for the formatter to know where to split it.\n"
                    "Consider adding whitespace at appropriate place(s) if possible.\n"
                    f'Line length (with &): "{min_possible_len}" of maximum permitted "{self._maxlinelength}"\n'
                    f'Code fragment:\n"{piece}"'
                )

            # Propose this piece being the end of this line.
            proposed_line = f"{formatted_line} {piece} &".lstrip()

            # If this line is impossible because it is too long, we should start a new line.
            if len(proposed_line) > self._maxlinelength:
                formatted_line = f"{formatted_line} &".lstrip()
                code_block = f"{code_block}{formatted_line}\n"
                # now we can reset formatted_line and use it for the next line
                formatted_line = ""
            formatted_line = f"{formatted_line} {piece}"
        # append the last line (which we've already checked is below the max line length limit)
        if formatted_line:
            formatted_line = formatted_line.lstrip()
            code_block = f"{code_block}{formatted_line}\n"
        return code_block
