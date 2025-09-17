#  (C) Crown Copyright, Met Office, 2023.
import numpy as np

import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers.input_layer import InputLayer


class Pooling1d(BaseLayer):
    """ENNUF representation of a 1d pooling layer"""

    def get_fortran_data_initialisation(self) -> str:
        return ""

    def __init__(
            self,
            name: str,
            inputs: BaseLayer,
            parent_model: model.Model,
            pool_size: int,
            type_of_pooling: str,
            padding: int,
            stride: int
    ):
        self.pool_size = pool_size
        self.type_of_pooling = type_of_pooling
        self.padding = padding
        self.stride = stride
        channels = inputs.shape[0]
        l_in = inputs.shape[1]
        if (l_in + 2 * self.padding - self.pool_size) % self.stride != 0:
            raise ValueError(f"Cannot integer divide by stride: {(l_in + 2 * self.padding - self.pool_size)=} and {self.stride=}.\n Modulus should be zero but is: {(l_in + 2 * self.padding - self.pool_size) % self.stride}")
        l_out = int(1 + (l_in + 2 * self.padding - self.pool_size) / self.stride)
        super().__init__(name, (channels, l_out), inputs, parent_model)

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        subroutine_name = self.fortran_id()
        x_in = self.inputs.output_name
        y_out = self.output_name
        channels = self.shape[0]
        l_in = self.inputs.shape[1]
        l_out = self.shape[1]
        pool_size = self.pool_size
        type_of_pooling = f"'{self.type_of_pooling}'"
        padding = self.padding
        stride = self.stride
        call_stmt = self.parent_model.formatter.format_line(
            f"CALL {subroutine_name}({x_in}, {y_out}, {channels}, {l_in}, {l_out}, {type_of_pooling}, {pool_size}, {padding}, {stride})"
        )
        return call_stmt

    @staticmethod
    def fortran_id() -> str:
        return "pooling_1d"

    def __str__(self):
        return (
            f' 1D Pooling layer "{self.name}" of type "{self.type_of_pooling}",'
            f' with window  of size {self.pool_size}'
            f' and inputs "{self.inputs.name}"'
        )

    def get_fortran_type_declaration(self, dtype: str) -> str:
        channels = self.shape[0]
        l_out = self.shape[1]
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}({channels},{l_out})"
        )
        return output_typedecl
