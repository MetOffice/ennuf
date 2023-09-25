#  (C) Crown Copyright, Met Office, 2023.
from typing import Dict, Type

from ennuf._internal.ml_model.activations.linear import Linear
from ennuf._internal.ml_model.base_activation import BaseActivation
from ennuf._internal.ml_model.activations.leaky_relu import LeakyRelu
from ennuf._internal.ml_model.activations.relu import Relu
from ennuf._internal.ml_model.activations.sigmoid import Sigmoid
from ennuf._internal.ml_model.activations.tanh import Tanh


class SupportedActivations:
    @staticmethod
    def ids() -> Dict[str, Type[BaseActivation] | None]:
        return {'relu': Relu, 'sigmoid': Sigmoid, 'tanh': Tanh, 'linear': Linear, 'LeakyReLU': LeakyRelu}

    @classmethod
    def from_identifier(cls, id_: str) -> BaseActivation | None:
        if id_ not in cls.ids():
            raise NotImplementedError(f'Unsupported activation identifier: {id_}')
        activationtype = cls.ids()[id_]
        if activationtype is None:
            return None
        return activationtype()

    @classmethod
    def from_serialized_keras_dict(cls, seralized_dict: Dict) -> BaseActivation | None:
        id_ = seralized_dict['class_name']
        if id_ not in cls.ids():
            raise NotImplementedError(f'Unsupported activation identifier: {id_}')
        activationtype = cls.ids()[id_]
        if activationtype is None:
            return None
        if activationtype is LeakyRelu:
            alpha = seralized_dict['config']['alpha']
            return LeakyRelu(alpha=alpha)
