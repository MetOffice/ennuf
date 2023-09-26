#  (C) Crown Copyright, Met Office, 2023.

from ennuf._internal.formatters.base_formatter import BaseFormatter


class UMFormatter(BaseFormatter):
    """Formatter for making Fortran files in a UM-compatible way"""

    _maxlinelength = 80

    @property
    def default_dtype(self) -> str:
        return "real_umphys"

    def required_subroutine_opening_actions(self, *args, **kwargs) -> str:
        return "IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName,zhook_in,zhook_handle)\n"

    def required_subroutine_closing_actions(self, *args, **kwargs) -> str:
        return "IF (lhook) CALL dr_hook(ModuleName//':'//RoutineName,zhook_out,zhook_handle)\n"

    def required_subroutine_imports(self) -> str:
        yomhook_import = "USE yomhook,               ONLY: lhook, dr_hook"
        parkind1_import = "USE parkind1,              ONLY: jprb, jpim"
        return f"{yomhook_import}\n" f"{parkind1_import}\n"

    def required_subroutine_declarations(self, subroutine_name: str) -> str:
        zhook_in_decl = "INTEGER(KIND=jpim), PARAMETER :: zhook_in  = 0"
        zhook_out_decl = "INTEGER(KIND=jpim), PARAMETER :: zhook_out = 1"
        zhook_handle_decl = "REAL(KIND=jprb)               :: zhook_handle"
        routine_name_decl = f"CHARACTER(LEN=*), PARAMETER :: RoutineName='{subroutine_name.upper()}'"
        return f"{zhook_in_decl}\n" f"{zhook_out_decl}\n" f"{zhook_handle_decl}\n" f"\n" f"{routine_name_decl}\n"

    def required_module_imports(self, *args, **kwargs) -> str:
        use_umphys_statement = "USE um_types, ONLY: real_umphys"
        return f"{use_umphys_statement}\n"

    def required_module_declarations(self, module_name: str, **kwargs) -> str:
        module_name_stmt = self.format_line(
            f"CHARACTER(LEN=*), PARAMETER, PRIVATE :: ModuleName = '{module_name.upper()}'"
        )
        return f"{module_name_stmt}\n"

    def required_file_header(self) -> str:
        return self._get_met_office_copyright()
