#  (C) Crown Copyright, Met Office, 2023.
from ennuf._internal.ml_model.base_activation import BaseActivation


class Linear(BaseActivation):
    def __str__(self):
        return "linear"

    def fortran_id(self) -> str:
        return "'linear    '"
