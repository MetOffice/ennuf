from typing import Tuple, List

import numpy as np

from ennuf._internal.ml_model.base_layer import BaseLayer
import ennuf._internal.ml_model.model as model


class Flatten(BaseLayer):
    def __init__(self, name: str, inputs: BaseLayer,
                 parent_model: model.Model):
        target_shape = (np.prod(inputs.shape),)
        super().__init__(name, target_shape, inputs, parent_model)

    def __str__(self):
        return (
            f"Flatten layer {self.name}"
            f"with input shape {self.inputs.shape}"
            f"with output shape {self.shape}"
            f"and inputs {self.inputs.name}"
        )

    def get_fortran_type_declaration(self, dtype: str) -> str:
        input_shape = self.inputs.shape
        if input_shape is None:
            raise ValueError("Input shape of flatten layer cannot be None")
        output_shape = self.shape if len(self.shape) != 1 else f"({str(self.shape[0])})"
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}{output_shape}"
        )
        return f"{output_typedecl}\n"

    def get_fortran_data_initialisation(self) -> str:
        return ""

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        call_stmt = self.parent_model.formatter.format_line(
            f"{self.output_name} = RESHAPE("
            f"RESHAPE({self.inputs.output_name}, "
            f"(/{str(tuple(reversed(self.inputs.shape)))[1:-1]}/), "
            f"ORDER=[{str(tuple(i+1 for i in reversed(range(len(self.inputs.shape)))))[1:-1]}]), "
            # f"(/{str(self.shape)[1:-1]}/))"
            f"(/{str(self.shape[0])}/))"
        )
        return f"{call_stmt}\n"
