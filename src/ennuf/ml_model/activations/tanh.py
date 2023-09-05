#  (C) Crown Copyright, Met Office, 2023.
from ennuf.ml_model.activation import Activation


class Tanh(Activation):
    def fortran_id(self) -> str:
        return "'tanh      '"

    def __str__(self):
        return 'tanh'

