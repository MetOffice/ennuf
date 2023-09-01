#  (C) Crown Copyright, Met Office, 2023.
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, name: str, input_name: str):
        self.input_name = input_name
        self.name = name

    @abstractmethod
    def __str__(self):
        pass
