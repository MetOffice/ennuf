#  (C) Crown Copyright, Met Office, 2023.
from typing import Dict

import keras.activations
import keras.layers
import tensorflow as tf

from ennuf.ml_model.layer import Layer
from ennuf.ml_model.layers.dense import Dense
from ennuf.ml_model.layers.input_layer import InputLayer
from ennuf.ml_model.model import Model
from ennuf.ml_model.supported_activations import SupportedActivations


def from_layer(parent_ennuf_model: Model, layer) -> Layer:
    if isinstance(layer, keras.layers.Dense) or isinstance(layer, tf.keras.layers.Dense):
        layer: keras.layers.Dense
        use_bias = False if layer.bias is None else True
        biases = layer.get_weights()[1] if use_bias else None
        kas_activation = keras.activations.serialize(layer.activation)
        if isinstance(kas_activation, str):
            activation = SupportedActivations.from_identifier(kas_activation)
        elif isinstance(kas_activation, Dict):
            activation = SupportedActivations.from_serialized_keras_dict(kas_activation)
        else:
            activation = None
        input_name: str = layer.input.name
        # keras layer names are like, if the previous layer was dense layer "dense3", that's internally several layers
        # potentially, perhaps ending with "dense3/BiasAdd:0" or "dense3/leaky_re_lu/LeakyReLu:0" but the internal ones
        # use / here it works perfectly fine to split them by / until either a user uses names with / or the tf
        # internals change and remove this (by which point hopefully we're no longer using ENNUF)
        input_name = input_name.split('/')[0]
        return Dense(
            name=layer.name,
            input_name=input_name,
            input_layer=parent_ennuf_model.layer_dict[input_name],
            units=layer.units,
            weights=layer.get_weights()[0],
            biases=biases,
            activation=activation,
            use_bias=use_bias,
        )
    if isinstance(layer, keras.layers.InputLayer) or isinstance(layer, tf.keras.layers.InputLayer):
        layer: keras.layers.InputLayer
        shape = layer.input_shape[0][1:]
        return InputLayer(name=layer.name, shape=shape)
    raise NotImplementedError(f'Could not match a supported layer type to type {type(layer)}')
