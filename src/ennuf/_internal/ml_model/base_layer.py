#  (C) Crown Copyright, Met Office, 2023.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List

import ennuf._internal.ml_model.model as model


class BaseLayer(ABC):
    """Abstract base class for ennuf layers"""

    def __init__(
        self,
        name: str,
        shape: int | Tuple[int] | None,
        inputs: List[BaseLayer] | BaseLayer | None,
        parent_model: model.Model,
    ):
        self.inputs = inputs
        if isinstance(shape, Tuple):
            self.shape: Tuple[int] | None = shape
        else:
            self.shape = (shape,)
        self.name = name
        self._overriden_output_name = None
        self.parent_model = parent_model

    @property
    def output_name(self):
        return f"y_{self.name}" if self._overriden_output_name is None else self._overriden_output_name

    @output_name.setter
    def output_name(self, value):
        self._overriden_output_name = value

    @abstractmethod
    def __str__(self):
        """
        Just a description of the layer for debugging purposes, e.g. "Dense layer with {N} neurons"
        """
        pass

    @abstractmethod
    def get_fortran_type_declaration(self, dtype: str) -> str:
        """
        All type declarations that will be required if the layer is used in a model module.
        Would not normally include input arguments to the layer since these are normally outputs of other layers.

        For example, if the Fortran subroutine representing this layer type takes arguments
        `input1, input2, internal_arg1, internal_arg2, internal_arg3, output`, then in the scope that calls
        this subroutine, we need to have declared the three internal args and the output. The inputs are presumed
        already declared since they should be the output of some other layers.
        """
        pass

    @abstractmethod
    def get_fortran_data_initialisation(self) -> str:
        """
        If the layer contains internal data such as weights, then this needs to be initialised.
        Formatters should help with this, should be able to just pass those the arrays we need to initalise to
        get the Fortran initialisation of them.
        """
        pass

    @abstractmethod
    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        """Returns a Fortran statement calling the relevant subroutine with the required arguments."""
        pass

    @staticmethod
    def fortran_id() -> str | None:
        """The name of the corresponding Fortran subroutine."""
        return None

    def get_additional_fortran_imports(self) -> str:
        return ""
