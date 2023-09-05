#  (C) Crown Copyright, Met Office, 2023.
import numpy as np

from ennuf.config import FORMATTER
from ennuf.ml_model.activation import Activation
from ennuf.ml_model.activations.leaky_relu import LeakyRelu
from ennuf.ml_model.activations.linear import Linear
from ennuf.ml_model.layer import Layer


class Dense(Layer):
    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        subroutine_name = self.fortran_id()
        x_in = self.input_layer.output_name
        y_out = self.output_name
        n_in = self.weights.shape[0]
        n_out = self.weights.shape[1]
        weights = self._weights_name
        biases = self._bias_name
        activation = self.activation.fortran_id()
        call_stmt = FORMATTER.format_line(
            f'CALL {subroutine_name}({x_in}, {y_out}, {n_in}, {n_out}, {weights}, {biases}, {activation})'
            )
        return call_stmt

    @staticmethod
    def fortran_id() -> str:
        return 'dense'

    def __str__(self):
        return f'Dense layer "{self.name}" with size {self.units},' \
               f' activation {str(self.activation)}' \
               f' {"with" if self.use_bias else "without"} bias' \
               f' and input "{self.input_name}"'

    def __init__(
            self,
            name: str,
            input_name: str,
            input_layer: Layer | None,
            units: int,
            weights: np.ndarray,
            biases: np.ndarray,
            activation: Activation = None,
            use_bias: bool = True,
    ):
        self.units = units
        self.weights = weights
        self.biases = biases
        self.activation = activation
        if activation is None:
            self.activation = Linear()
        self.use_bias = use_bias
        if not use_bias:
            raise NotImplementedError('Not yet implemented dense layers without bias.')
        if isinstance(self.activation, LeakyRelu) or isinstance(self.activation, Linear):
            raise NotImplementedError(f'Not yet implemented activation {str(self.activation)}')
        super().__init__(name, input_name, input_layer)
        self._weights_name = f'w_{self.name}'
        self._bias_name = f'b_{self.name}'

    def get_fortran_type_declaration(self, dtype: str) -> str:
        input_shape = self.weights.shape[0]
        output_shape = self.weights.shape[1]
        weights_typedecl = FORMATTER.format_line(
            f'REAL(KIND={dtype}) :: {self._weights_name}({input_shape}, {output_shape})'
        )
        bias_typedecl = FORMATTER.format_line(f'REAL(KIND={dtype}) :: {self._bias_name}({output_shape})')
        output_typedecl = FORMATTER.format_line(f'REAL(KIND={dtype}) :: {self.output_name}({output_shape})')
        return f'{weights_typedecl}{bias_typedecl}{output_typedecl}\n'

    def get_fortran_data_initialisation(self) -> str:
        bias_init = FORMATTER.format_data_statement(varname=self._bias_name, data=self.biases)
        weights_inits = FORMATTER.format_data_statement(varname=self._weights_name, data=self.weights)
        return f'{bias_init}\n{weights_inits}'

    def get_additional_fortran_imports(self) -> str:
        if isinstance(self.activation, LeakyRelu):
            return 'leaky_relu'
        return ''
