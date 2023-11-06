#  (C) Crown Copyright, Met Office, 2023.
from ennuf._internal.ml_model.base_activation import BaseActivation


class LeakyRelu(BaseActivation):
    """Ennuf representation of Leaky ReLU activation function"""

    def fortran_id(self) -> str:
        """The string identifier of the activation function used in Fortran"""
        return f"'leakyrelu '"

    def __str__(self):
        return f"LeakyRelu(alpha={self.alpha:.5f})"

    def __init__(self, alpha: float):
        self.alpha = alpha
