#  (C) Crown Copyright, Met Office, 2023.
from typing import Tuple

from ennuf._internal.ml_model.base_layer import BaseLayer


class InputLayer(BaseLayer):
    """Ennuf representation of a set of inputs to a neural network."""

    def __init__(self, shape: Tuple[int], name: str, parent_model):
        super().__init__(
            name=name,
            shape=shape,
            input_name=None,
            input_layer=None,
            parent_model=parent_model,
        )
        self.output_name = self.name

    def __str__(self):
        return f'Input layer "{self.name}" with shape {self.shape}'

    def get_fortran_type_declaration(self, dtype: str) -> str:
        shape_str = ""
        for dim in self.shape:
            if shape_str:
                shape_str = f"{shape_str}, {dim}"
            else:
                shape_str = f"{dim}"
        input_typedecl = self.parent_model.formatter.format_line(f"REAL(KIND={dtype}) :: {self.name}({shape_str})")
        return f"{input_typedecl}\n"

    def get_fortran_data_initialisation(self) -> str:
        return ""

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        return ""
