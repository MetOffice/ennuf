from ennuf._internal.ml_model.base_layer import BaseLayer


class Reshape(BaseLayer):
    def __str__(self):
        return (
            f"Reshape layer {self.name}"
            f"with input shape {self.inputs.shape}"
            f"with output shape {self.shape}"
            f"and inputs {self.inputs.name}"
        )

    def get_fortran_type_declaration(self, dtype: str) -> str:
        input_shape = self.inputs.shape
        if input_shape is None:
            raise ValueError("Input shape of reshape layer cannot be None")
        output_shape = self.shape
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: {self.output_name}{output_shape}"
        )
        return f"{output_typedecl}\n"

    def get_fortran_data_initialisation(self) -> str:
        return ""

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        call_stmt = self.parent_model.formatter.format_line(
            f"{self.output_name} = RESHAPE({self.inputs.output_name}, (/{str(self.shape)[1:-1]}/), ORDER=[{str(tuple(i+1 for i in reversed(range(len(self.shape)))))[1:-1]}])"
        )
        return f"{call_stmt}\n"
