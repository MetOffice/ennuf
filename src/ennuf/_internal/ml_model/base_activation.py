#  (C) Crown Copyright, Met Office, 2023.
from abc import ABC, abstractmethod


class BaseActivation(ABC):
    """Abstract base class for ennuf activations"""

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def fortran_id(self) -> str:
        """The string identifier of the activation function used in Fortran"""
