#  (C) Crown Copyright, Met Office, 2023.
from typing import Tuple

from ennuf.ml_model.layer import Layer


class InputLayer(Layer):
    def __init__(self, shape: Tuple, name: str, input_name=None):
        self.shape = shape
        super().__init__(name, input_name)

    def __str__(self):
        return f'An input layer of shape {self.shape}'
