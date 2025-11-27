#  (C) Crown Copyright, Met Office, 2025.
from typing import Tuple, List

from ennuf._internal.ml_model.base_layer import BaseLayer
import ennuf._internal.ml_model.model as model


class Reshape(BaseLayer):
    def __init__(self, name: str, shape: int | Tuple[int, ...], inputs: BaseLayer,
                 parent_model: model.Model):
        super().__init__(name, shape, inputs, parent_model)
        if len(self.inputs.shape) != len(self.shape):
            raise NotImplementedError(f"Have only implemented reshape layers where the final shape"
                                      f" is the same dimensionality as the initial shape."
                                      f"Got initial shape of {self.inputs.shape}"
                                      f"and final shape of {self.shape} in layer {self.name}")

    def __str__(self):
        return (
            f"Reshape layer {self.name}"
            f" with input shape {self.inputs.shape}"
            f" with output shape {self.shape}"
            f" and inputs {self.inputs.name}"
        )

    def get_fortran_type_declaration(self, dtype: str) -> str:
        input_shape = self.inputs.shape
        if input_shape is None:
            raise ValueError("Input shape of reshape layer cannot be None")
        output_shape = self.shape if len(self.shape) != 1 else f"({str(self.shape[0])})"
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}{output_shape}"
        )
        return f"{output_typedecl}\n"

    def get_fortran_data_initialisation(self) -> str:
        return ""

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        if len(self.shape) !=1:
            call_stmt = self.parent_model.formatter.format_line(
                f"{self.output_name} = RESHAPE({self.inputs.output_name}, (/{str(self.shape)[1:-1]}/), ORDER=[{str(tuple(i+1 for i in reversed(range(len(self.shape)))))[1:-1]}])"
            )
        else:
            call_stmt = self.parent_model.formatter.format_line(
                f"{self.output_name} = RESHAPE(RESHAPE({self.inputs.output_name}, (/{str(tuple(reversed(self.inputs.shape)))[1:-1]}/), ORDER=[{str(tuple(i+1 for i in reversed(range(len(self.inputs.shape)))))[1:-1]}]), (/{str(self.shape[0])}/))"
            )
        return f"{call_stmt}\n"
