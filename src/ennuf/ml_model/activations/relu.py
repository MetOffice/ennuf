#  (C) Crown Copyright, Met Office, 2023.
from ennuf.ml_model.activation import Activation


class Relu(Activation):
    def fortran_id(self) -> str:
        return "'relu      '"

    def __str__(self):
        return 'relu'

