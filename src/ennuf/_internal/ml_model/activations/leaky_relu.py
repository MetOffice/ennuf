#  (C) Crown Copyright, Met Office, 2023.
from ennuf._internal.ml_model.base_activation import BaseActivation


class LeakyRelu(BaseActivation):
    """Ennuf representation of Leaky ReLU activation function"""

    def fortran_id(self) -> str:
        """The string identifier of the activation function used in Fortran"""
        return f"leaky_relu({self.alpha})"

    def __str__(self):
        return f"LeakyRelu(alpha={self.alpha:.3f})"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
