#  (C) Crown Copyright, Met Office, 2023.
from typing import Tuple

import numpy as np

import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers.input_layer import InputLayer


class Dense(BaseLayer):
    """Ennuf representation of a dense layer"""

    def __init__(
        self,
        name: str,
        inputs: BaseLayer,
        parent_model: model.Model,
        shape: int | Tuple[int, ...],
        weights: np.ndarray,
        biases: np.ndarray | None,
        use_bias: bool = True,
    ):
        if isinstance(shape, tuple) and len(shape) == 2:
            shape_with_channels = shape
        elif isinstance(shape, tuple) and len(shape) == 1:
            shape_with_channels = (1, shape[0])
        elif isinstance(shape, int):
            shape_with_channels = (1, shape)
        else:
            raise ValueError(f"Dense layer cannot have more than two dimensions, {shape=} requested")
        self.units = shape_with_channels[1]
        self.weights = weights
        if use_bias and biases is None:
            raise ValueError("biases may not be None while use_bias is True")
        self.biases = biases
        self.use_bias = use_bias
        super().__init__(name, shape_with_channels, inputs, parent_model)
        self._weights_name = f"w_{self.name}"
        self._bias_name = f"b_{self.name}"

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        subroutine_name = self.fortran_id()
        x_in = self.inputs.output_name
        y_out = self.output_name
        channels = self.shape[0]
        l_in = self.weights.shape[0]
        l_out = self.weights.shape[1]
        weights = self._weights_name
        biases = self._bias_name
        call_stmt = self.parent_model.formatter.format_line(
            f"CALL {subroutine_name}({x_in}, {y_out}, {channels}, {l_in}, {l_out}, {weights}, {biases})"
        ) if self.use_bias else self.parent_model.formatter.format_line(
            f"CALL {subroutine_name}({x_in}, {y_out}, {channels}, {l_in}, {l_out}, {weights})"
        )
        return call_stmt

    @staticmethod
    def fortran_id() -> str:
        return "dense"

    def __str__(self):
        return (
            f'Dense layer "{self.name}" with size {self.units},'
            f' {"with" if self.use_bias else "without"} bias'
            f' and inputs "{self.inputs.name}"'
        )

    def get_fortran_type_declaration(self, dtype: str) -> str:
        input_shape = self.weights.shape[0]
        output_shape = self.shape[1]
        channels = self.shape[0]
        weights_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self._weights_name}({output_shape}, {input_shape})"
        )
        bias_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self._bias_name}({output_shape})"
        ) if self.use_bias else ""
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}({channels},{output_shape})"
        )
        typedecl = f"{weights_typedecl}{bias_typedecl}{output_typedecl}\n"
        return typedecl

    def get_fortran_data_initialisation(self) -> str:
        weights_inits = self.parent_model.formatter.format_data_statement(varname=self._weights_name, data=self.weights.T)
        if self.use_bias:
            bias_init = self.parent_model.formatter.format_data_statement(varname=self._bias_name, data=self.biases)
            return f"{bias_init}\n{weights_inits}"
        else:
            return weights_inits
