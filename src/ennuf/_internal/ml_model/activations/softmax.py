#  (C) Crown Copyright, Met Office, 2025.
from ennuf._internal.ml_model.base_activation import BaseActivation


class Softmax(BaseActivation):
    """Ennuf representation of Softmax activation function"""

    def fortran_id(self) -> str:
        """The string identifier of the activation function used in Fortran"""
        return "'softmax   '"

    def __str__(self):
        return "softmax"
