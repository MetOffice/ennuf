#  (C) Crown Copyright, Met Office, 2023.
import numpy as np
import numpy as np
from ennuf.ml_model.activation import Activation
from ennuf.ml_model.layer import Layer


class Dense(Layer):
    def __init__(
            self,
            name: str,
            input_name: str,
            units: int,
            weights: np.ndarray,
            biases: np.ndarray | None = None,
            activation: Activation = None,
            use_bias: bool = True,
    ):
        self.units = units
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.use_bias = use_bias
        super().__init__(name, input_name)

    def __str__(self):
        return f'Dense layer "{self.name}" with size {self.units},' \
               f' activation {str(self.activation)}' \
               f' {"with" if self.use_bias else "without"} bias' \
               f' and input "{self.input_name}"'
