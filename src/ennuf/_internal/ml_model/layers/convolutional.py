#  (C) Crown Copyright, Met Office, 2023.
from enum import Enum

import numpy as np

import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers.input_layer import InputLayer


class PaddingMode(Enum):
    NONE = "none   "
    ZEROS = "zeros  "
    REFLECT = "reflect"

    @classmethod
    def from_keras_padding_mode(cls, keras_padding_mode):
        match keras_padding_mode:
            case "valid":
                return PaddingMode.NONE
            case "same":
                return PaddingMode.ZEROS
            case _:
                raise NotImplementedError(f"Unsupported padding mode: {keras_padding_mode}")

    @classmethod
    def from_torch_padding_mode(cls, torch_padding_mode):
        match torch_padding_mode:
            case "zeros":
                return PaddingMode.ZEROS
            case "reflect":
                return PaddingMode.REFLECT
            case _:
                raise NotImplementedError(f"Unsupported padding mode: {torch_padding_mode}")


class Conv1d(BaseLayer):
    """ENNUF representation of a 1d convolutional layer"""

    def __init__(
            self,
            name: str,
            inputs: BaseLayer,
            parent_model: model.Model,
            weights: np.ndarray,
            biases: np.ndarray,
            padding_mode: PaddingMode,
            padding: int,
            stride: int,
            dilation: int,
            use_bias: bool = True,
    ):
        self.pad_mode = padding_mode
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.weights = weights
        self.biases = biases
        self.use_bias = use_bias
        if not use_bias:
            raise NotImplementedError("Not yet implemented convolutional layers without bias.")
        self.kernel_size = self.weights.shape[2]
        c_out = self.weights.shape[0]
        l_in = inputs.shape[1]
        if (self.dilation * (self.kernel_size - 1) - 1) % self.stride != 0:
            raise ValueError(
                f"Cannot integer divide by stride: {(self.dilation * (self.kernel_size - 1) - 1)=} and {self.stride=}.\n Modulus should be zero but is: {(self.dilation * (self.kernel_size - 1) - 1) % self.stride}")
        l_out = int(1 + ((l_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride))
        if l_out < 1:
            raise ValueError(f"Output of convolutional layer predicted to have a dimension of length less than 1 ({l_out} in a shape of {(c_out, l_out)}).")
        super().__init__(name, (c_out, l_out), inputs, parent_model)
        self._weights_name = f"w_{self.name}"
        self._bias_name = f"b_{self.name}"

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        subroutine_name = self.fortran_id()
        x_in = self.inputs.output_name
        y_out = self.output_name
        c_in = self.weights.shape[1]
        l_in = self.inputs.shape[1]
        c_out = self.weights.shape[0]
        l_out = self.shape[1]
        k_size = self.weights.shape[2]
        pad_mode = f"'{self.pad_mode.value}'"
        padding = self.padding
        stride = self.stride
        dilation = self.dilation
        weights = self._weights_name
        biases = self._bias_name
        call_stmt = self.parent_model.formatter.format_line(
            f"CALL {subroutine_name}({x_in}, {y_out}, {c_in}, {c_out}, {l_in}, {l_out}, {k_size}, {weights}, {biases}, {pad_mode}, {padding}, {stride}, {dilation})"
        )
        return call_stmt

    @staticmethod
    def fortran_id() -> str:
        return "conv_1d"

    def __str__(self):
        return (
            f' 1D Convolutional layer "{self.name}" with kernel of size {self.kernel_size},'
            f' {"with" if self.use_bias else "without"} bias'
            f' and inputs "{self.inputs.name}"'
        )

    def get_fortran_type_declaration(self, dtype: str) -> str:
        channels_out = self.weights.shape[0]
        channels_in = self.weights.shape[1]
        kernel_size = self.weights.shape[2]
        length_out = self.shape[1]
        weights_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self._weights_name}({channels_out}, {channels_in}, {kernel_size})"
        )
        bias_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self._bias_name}({channels_out})"
        )
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}({channels_out},{length_out})"
        )
        return f"{weights_typedecl}{bias_typedecl}{output_typedecl}\n"

    def get_fortran_data_initialisation(self) -> str:
        bias_init = self.parent_model.formatter.format_data_statement(varname=self._bias_name, data=self.biases)
        weights_init = self.parent_model.formatter.format_data_statement(varname=self._weights_name, data=self.weights)
        return f"{bias_init}\n{weights_init}"
