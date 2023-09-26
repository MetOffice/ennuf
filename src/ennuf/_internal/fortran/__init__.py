#  (C) Crown Copyright, Met Office, 2023.
"""Package containing Fortran code and functions to access that Fortran code and move it elsewhere"""
import shutil
from pathlib import Path


def copy_neural_net_mod(dest_dir: Path) -> None:
    """Copies the neural net mod fortran file to the destination."""
    fortran_dir = Path(__file__).parent
    neural_net_mod_file = fortran_dir.joinpath("neural_net_mod.f90")
    shutil.copy(neural_net_mod_file, dest_dir.joinpath("neural_net_mod.f90"))
