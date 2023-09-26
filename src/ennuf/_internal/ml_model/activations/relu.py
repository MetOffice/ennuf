#  (C) Crown Copyright, Met Office, 2023.
from ennuf._internal.ml_model.base_activation import BaseActivation


class Relu(BaseActivation):
    """Ennuf representation of ReLU activation function"""

    def fortran_id(self) -> str:
        """The string identifier of the activation function used in Fortran"""
        return "'relu      '"

    def __str__(self):
        return "relu"
