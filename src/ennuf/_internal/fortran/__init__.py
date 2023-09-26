#  (C) Crown Copyright, Met Office, 2023.
import shutil
from pathlib import Path


def copy_neural_net_mod(dest_dir: Path):
    fortran_dir = Path(__file__).parent
    neural_net_mod_file = fortran_dir.joinpath("neural_net_mod.f90")
    shutil.copy(neural_net_mod_file, dest_dir.joinpath("neural_net_mod.f90"))
