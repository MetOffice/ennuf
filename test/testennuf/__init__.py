#  (C) Crown Copyright, Met Office, 2023.
from pathlib import Path

TESTS_ROOT_DIR = Path(__file__).parent
WEIGHTS_DIR = TESTS_ROOT_DIR.joinpath("weights")
TMPDIR = TESTS_ROOT_DIR.joinpath("tmp")
TMPDIR.mkdir(exist_ok=True)
RANDOM_SEED = 43
