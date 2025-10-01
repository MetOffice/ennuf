#  (C) Crown Copyright, Met Office, 2023.
from typing import Tuple

from ennuf._internal.ml_model.base_layer import BaseLayer


class InputLayer(BaseLayer):
    """Ennuf representation of a set of inputs to a neural network."""

    def __init__(self, shape: Tuple[int, ...], name: str, parent_model, has_channels: bool = False,):
        """
        :param shape: the shape of input data
        :param name: a unique id for this layer, which must be a valid Fortran identifier
        :param has_channels: True if the first dimension of the input shape should be treated as a number of channels
            for e.g. convolutional layers after this one.
        :param parent_model: the ennuf model this layer belongs to
        """
        shape_with_channels = shape if has_channels else (1,) + shape
        super().__init__(
            name=name,
            shape=shape_with_channels,
            inputs=None,
            parent_model=parent_model,
        )
        self.shape_without_channels_if_not_provided = shape
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
