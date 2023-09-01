#  (C) Crown Copyright, Met Office, 2023.
from typing import Set, List

import numpy as np

from ennuf.ml_model.layer import Layer


class Model:
    layers: List[Layer] = []
    """
    The model's layers. Note this is a list rather than a set, so that when displayed to a user the layers
    can appear easily in the same order they specified them; but this is *not* guarunteed to be the ordering of
    the layers of a sequential model.
    """
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __str__(self):
        output = f'An ML model with dtype {self.dtype} the following layers:\n'
        for layer in self.layers:
            output += str(layer) + ';\n'
        return output
