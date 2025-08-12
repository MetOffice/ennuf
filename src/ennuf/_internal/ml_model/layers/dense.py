#  (C) Crown Copyright, Met Office, 2023.
import numpy as np

import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.base_layer import BaseLayer


class Dense(BaseLayer):
    """Ennuf representation of a dense layer"""

    def __init__(
        self,
        name: str,
        inputs: BaseLayer,
        parent_model: model.Model,
        units: int,
        weights: np.ndarray,
        biases: np.ndarray,
        use_bias: bool = True,
    ):
        self.units = units
        self.weights = weights
        self.biases = biases
        self.use_bias = use_bias
        if not use_bias:
            raise NotImplementedError("Not yet implemented dense layers without bias.")
        super().__init__(name, self.weights.shape[1], inputs, parent_model)
        self._weights_name = f"w_{self.name}"
        self._bias_name = f"b_{self.name}"

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        subroutine_name = self.fortran_id()
        x_in = self.inputs.output_name
        y_out = self.output_name
        channels = self.inputs.shape[0]
        l_in = self.weights.shape[1]
        l_out = self.weights.shape[0]
        weights = self._weights_name
        biases = self._bias_name
        call_stmt = self.parent_model.formatter.format_line(
            f"CALL {subroutine_name}({x_in}, {y_out}, {channels}, {l_in}, {l_out}, {weights}, {biases})"
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
        output_shape = self.shape[0]
        channels = self.inputs.shape[0]
        weights_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self._weights_name}({output_shape}, {input_shape})"
        )
        bias_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self._bias_name}({output_shape})"
        )
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}({channels},{output_shape})"
        )
        return f"{weights_typedecl}{bias_typedecl}{output_typedecl}\n"

    def get_fortran_data_initialisation(self) -> str:
        bias_init = self.parent_model.formatter.format_data_statement(varname=self._bias_name, data=self.biases)
        weights_inits = self.parent_model.formatter.format_data_statement(varname=self._weights_name, data=self.weights.T)
        return f"{bias_init}\n{weights_inits}"
