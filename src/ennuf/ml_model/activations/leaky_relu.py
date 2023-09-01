#  (C) Crown Copyright, Met Office, 2023.
from ennuf.ml_model.activation import Activation


class LeakyRelu(Activation):
    def __str__(self):
        return f'LeakyRelu(alpha={self.alpha:.3f})'

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
