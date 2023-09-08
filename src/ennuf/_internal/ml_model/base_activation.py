#  (C) Crown Copyright, Met Office, 2023.
from abc import ABC, abstractmethod


class BaseActivation(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def fortran_id(self) -> str:
        pass
