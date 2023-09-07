#  (C) Crown Copyright, Met Office, 2023.
from ennuf.ml_model.base_activation import BaseActivation


class Tanh(BaseActivation):
    def fortran_id(self) -> str:
        return "'tanh      '"

    def __str__(self):
        return 'tanh'

