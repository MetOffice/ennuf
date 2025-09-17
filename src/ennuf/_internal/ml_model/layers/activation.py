#  (C) Crown Copyright, Met Office, 2023.
from typing import Tuple

import numpy as np

import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.base_activation import BaseActivation

class Activation(BaseLayer):
    """Ennuf representation of the activation function layer"""
    def __init__(
        self,
        name: str,
        shape: int | Tuple[int, ...],
        inputs: BaseLayer,
        parent_model: model.Model,
        activation: BaseActivation,
    ):
        self.activation=activation
        if isinstance(shape, tuple) and len(shape) == 2:
            shape_with_channels = shape
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape_with_channels = (1, shape[0])
        elif isinstance(shape, int):
            shape_with_channels = (1, shape)
        else:
            raise ValueError(f"Activation layer cannot have more than two dimensions, {shape=} requested")
        super().__init__(name, shape_with_channels, inputs, parent_model)

    def get_fortran_type_declaration(self, dtype: str) -> str:
        channels = self.shape[0]
        output_shape = self.shape[1]
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}({channels},{output_shape})"
        )
        return output_typedecl

    def get_fortran_data_initialisation(self) -> str:
        return ""

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        subroutine_name = self.fortran_id()
        x_in = self.inputs.output_name
        y_out = self.output_name
        channels = self.shape[0]
        length = self.inputs.shape[1]
        activation_id = self.activation.fortran_id() 
        if hasattr(self.activation, "alpha"):
            alpha = self.activation.alpha
            call_stmt = self.parent_model.formatter.format_line(f"CALL {subroutine_name}({x_in}, {y_out}, {channels}, {length}, {activation_id}, {alpha})")
        else:
            call_stmt = self.parent_model.formatter.format_line(f"CALL {subroutine_name}({x_in}, {y_out}, {channels}, {length}, {activation_id})")
        
        return call_stmt

    @staticmethod
    def fortran_id() -> str:
        return "activation_function"

    def __str__(self):
        return (
            f' Activation Function layer,'
            f' with activation "{str(self.activation)}"'
            f' and inputs "{self.inputs.name}"'
        )
