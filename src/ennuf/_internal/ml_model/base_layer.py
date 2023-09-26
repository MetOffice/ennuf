#  (C) Crown Copyright, Met Office, 2023.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import ennuf._internal.ml_model.model as model


class BaseLayer(ABC):
    """Abstract base class for ennuf layers"""

    def __init__(
        self,
        name: str,
        shape: int | Tuple[int] | None,
        input_name: str | None,
        input_layer: BaseLayer | None,
        parent_model: model.Model,
    ):
        self.input_layer = input_layer
        self.input_name = input_name
        if isinstance(shape, Tuple):
            self.shape: Tuple[int] | None = shape
        else:
            self.shape = (shape,)
        self.name = name
        self.output_name = f"y_{self.name}"
        self.parent_model = parent_model

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_fortran_type_declaration(self, dtype: str) -> str:
        pass

    @abstractmethod
    def get_fortran_data_initialisation(self) -> str:
        pass

    @abstractmethod
    def get_fortran_layer_subroutine_call_stmt(self) -> str:
        pass

    @staticmethod
    def fortran_id() -> str | None:
        return None

    def get_additional_fortran_imports(self) -> str:
        return ""
