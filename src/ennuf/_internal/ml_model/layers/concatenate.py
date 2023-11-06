#  (C) Crown Copyright, Met Office, 2023.
from typing import List, Tuple

from ennuf._internal.ml_model import model
from ennuf._internal.ml_model.base_layer import BaseLayer


class Concatenate(BaseLayer):
    def __init__(
            self,
            name: str,
            shape: int | Tuple[int] | None,
            inputs: List[BaseLayer],
            axis: int,
            parent_model: model.Model,
    ):
        self.axis = axis
        super().__init__(name, shape, inputs, parent_model)
        if len(self.shape) != 1:
            raise NotImplementedError(
                f'Output shape expected to be {self.shape} but have not yet implemented'
                f' Concatenate layers for shapes that are not 1d.'
                )
        if len(inputs) != 2:
            raise NotImplementedError(
                'Not yet implemented Concatenate layers for concatenating more than two layers'
                f'at once, but recieved {len(inputs)} inputs.'
                )

    def __str__(self):
        return f'Concatenate layer concatenating "{tuple([inp.name for inp in self.inputs])}"' \
               f'along axis "{self.axis}" giving an output shape of "{self.shape}"'

    def get_fortran_type_declaration(self, dtype: str) -> str:
        output_shape = self.shape[0]
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}({output_shape})"
        )
        return output_typedecl

    def get_fortran_data_initialisation(self) -> str:
        return ""

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        subroutine_name = self.fortran_id()
        x_in1 = self.inputs[0].output_name
        x_in2 = self.inputs[1].output_name
        y_out = self.output_name
        call_stmt = self.parent_model.formatter.format_line(
            f"CALL {subroutine_name}({x_in1}, {x_in2}, {y_out})"
        )
        return call_stmt

    @staticmethod
    def fortran_id() -> str | None:
        return "concatenate_1d"
