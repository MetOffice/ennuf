#  (C) Crown Copyright, Met Office, 2023.
from ennuf.ml_model.base_activation import BaseActivation


class Relu(BaseActivation):
    def fortran_id(self) -> str:
        return "'relu      '"

    def __str__(self):
        return 'relu'

