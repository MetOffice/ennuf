#  (C) Crown Copyright, Met Office, 2023.
from ennuf._internal.ml_model.base_activation import BaseActivation


class Sigmoid(BaseActivation):
    def fortran_id(self) -> str:
        return "'sigmoid   '"

    def __str__(self):
        return "sigmoid"
