#  (C) Crown Copyright, Met Office, 2024.
from typing import Dict, Type

from ennuf._internal.ml_model.activations.linear import Linear
from ennuf._internal.ml_model.base_activation import BaseActivation
from ennuf._internal.ml_model.activations.leaky_relu import LeakyRelu
from ennuf._internal.ml_model.activations.relu import Relu
from ennuf._internal.ml_model.activations.sigmoid import Sigmoid
from ennuf._internal.ml_model.activations.tanh import Tanh
from ennuf._internal.ml_model.activations.softmax import Softmax


class SupportedActivations:
    """Class specifying which activations are supported and checking whether a given activation is."""

    @staticmethod
    def ids() -> Dict[str, Type[BaseActivation]]:
        """Returns a dict of all possible string ids and their associated activation functions"""
        return {
            "relu": Relu,
            "sigmoid": Sigmoid,
            "tanh": Tanh,
            "linear": Linear,
            "LeakyReLU": LeakyRelu,
            "softmax": Softmax,
        }

    @classmethod
    def from_identifier(cls, id_: str) -> BaseActivation | None:
        """Takes a string id and returns an ennuf activation class"""
        if id_ not in cls.ids():
            raise NotImplementedError(f"Unsupported activation identifier: {id_}")
        activationtype = cls.ids()[id_]
        if activationtype is LeakyRelu:
            raise NotImplementedError("pretty sure you are never supposed to reach here")
        if activationtype is None:
            return None
        return activationtype()

    @classmethod
    def from_serialized_dict(cls, serialized_dict: Dict) -> BaseActivation | None:
        """
        Takes a dictionary representation of an activation function of the form that might be returned by Keras,
        and returns an ennuf activation class.
        """
        id_ = serialized_dict["class_name"]
        if id_ not in cls.ids():
            raise NotImplementedError(f"Unsupported activation identifier: {id_}")
        activation_type = cls.ids()[id_]
        if activation_type is None:
            return None
        if activation_type is LeakyRelu:
            try:
                alpha = serialized_dict["config"]["negative_slope"]
            except KeyError:
                alpha = serialized_dict["config"]["alpha"]
            return LeakyRelu(alpha=alpha)
        return None
