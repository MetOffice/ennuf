#  (C) Crown Copyright, Met Office, 2023.
from ennuf.ml_model.activation import Activation


class Tanh(Activation):
    def __str__(self):
        return 'tanh'

