#  (C) Crown Copyright, Met Office, 2023.
from ennuf._internal.ml_model.base_activation import BaseActivation


class LeakyRelu(BaseActivation):
    def fortran_id(self) -> str:
        return f"leaky_relu({self.alpha})"

    def __str__(self):
        return f"LeakyRelu(alpha={self.alpha:.3f})"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
