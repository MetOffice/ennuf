#  (C) Crown Copyright, Met Office, 2023.
from ennuf._internal.ml_model.base_activation import BaseActivation


class Linear(BaseActivation):
    """Ennuf representation of linear activation function"""

    def fortran_id(self) -> str:
        """The string identifier of the activation function used in Fortran"""
        return "'linear    '"

    def __str__(self):
        return "linear"
