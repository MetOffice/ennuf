import numpy as np

import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.base_layer import BaseLayer


class SVR_ENNUF(BaseLayer):
    """Ennuf representation of support vector regression"""

    def __init__(
            self,
            name:str,
            parent_model: model.Model,
            dual_coef: np.array,
            support_vectors: np.array,
            intercept: float
    ):
        self.dual_coef = dual_coef
        self.support_vectors = support_vectors
        self.intercept = intercept
        super().__init__(name, support_vectors.shape[1], None, parent_model)

    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        subroutine_name = self.fortran_id()
        x_in="inputs"
        y_out="y_outputs"
        n_dims=self.shape
        dual_coef=self.dual_coef
        support_vectors=self.support_vectors
        intercept=self.intercept
        n_support_vectors=support_vectors.shape[0]
        call_stmt = self.parent_model.formatter.format_line(
            f"CALL {subroutine_name}({x_in},{support_vectors},{dual_coef},{n_support_vectors}, {n_dims}, {y_out}, {intercept})"
        )
        return call_stmt

    @staticmethod
    def fortran_id() -> str:
        return "svr"
    
    def __str__(self):
        return (
            f'Support vector regression, with {self.shape}-dimensional input,'
            f'and {self.support_vectors.shape[0]} support vectors.'
        )
    
    def get_fortran_type_declaration(self, dtype: str) -> str:
        input_shape = self.shape[0]
        output_shape = 1
        dual_coef_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: dual_coef({self.support_vectors.shape[0]})"
        )
        support_vectors_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: support_vectors({self.support_vectors.shape[0]},{self.support_vectors.shape[1]})"
        )
        output_typedecl = self.parent_model.formatter.format_line(
            f"REAL(KIND={dtype}) :: y_outputs)"
        )
        return f"{dual_coef_typedecl}{support_vectors_typedecl}{output_typedecl}\n"

    def get_fortran_data_initialisation(self) -> str:
        dual_coef_init = self.parent_model.formatter.format_data_statement(varname="dual_coef", data=self.dual_coef)
        support_vectors_init = self.parent_model.formatter.format_data_statement(varname="support_vectors", data=self.support_vectors)
        return f"{dual_coef_init}\n{support_vectors_init}"