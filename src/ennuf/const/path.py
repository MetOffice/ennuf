#  (C) Crown Copyright, Met Office, 2023.
from pathlib import Path

_PROJ_DIR = Path(__file__).parent.parent.parent.parent
_EXAMPLES_DIR = _PROJ_DIR.joinpath('example')
KERAS_FUNCTIONAL_EXAMPLES_DIR = _EXAMPLES_DIR.joinpath('keras_functional')
