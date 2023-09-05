#  (C) Crown Copyright, Met Office, 2023.
from ennuf.ml_model.activation import Activation


class Sigmoid(Activation):
    def fortran_id(self) -> str:
        return "'sigmoid   '"

    def __str__(self):
        return 'sigmoid'

