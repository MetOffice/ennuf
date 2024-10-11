#  (C) Crown Copyright, Met Office, 2024.
"""Module for defining how Keras layers should be translated to ennuf layers"""
from typing import Dict, List

import tensorflow as tf

from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers.concatenate import Concatenate
from ennuf._internal.ml_model.layers.dense import Dense
from ennuf._internal.ml_model.layers.input_layer import InputLayer
import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.supported_activations import SupportedActivations


def from_layer(parent_ennuf_model: model.Model, layer) -> BaseLayer:
    """Takes a keras model and a keras layer and returns an equivalent ennuf layer."""
    if isinstance(layer, tf.keras.layers.Dense):
        layer: tf.keras.layers.Dense
        use_bias = layer.bias is not None
        biases = layer.get_weights()[1] if use_bias else None
        kas_activation = tf.keras.activations.serialize(layer.activation)
        if isinstance(kas_activation, str):
            activation = SupportedActivations.from_identifier(kas_activation)
        elif isinstance(kas_activation, Dict):
            activation = SupportedActivations.from_serialized_keras_dict(kas_activation)
        else:
            activation = None
        input_name: str = layer.input.name
        # keras layer names are like,
        # if the previous layer was dense layer "dense3", that's internally several layers
        # potentially, perhaps ending with "dense3/BiasAdd:0"
        # or "dense3/leaky_re_lu/LeakyReLu:0" but the internal ones
        # use / here it works perfectly fine to split them by / until
        # either a user uses names with / or the tf
        # internals change and remove this
        # (by which point hopefully we're no longer using ENNUF)
        # *** update for tensorflow 2.17 ***
        # keras layer names now are separate from their input and output names.
        # layer.output.name will match the layer.input.name of the next layer,
        # but layer.name tells you the name of the layer itself, which is not
        # useful.

        input_name = input_name.split("/")[0]
        layer_name = layer.output.name
        return Dense(
            name=layer_name,
            inputs=parent_ennuf_model.layer_dict[input_name],
            parent_model=parent_ennuf_model,
            units=layer.units,
            weights=layer.get_weights()[0],
            biases=biases,
            activation=activation,
            use_bias=use_bias,
        )
    if isinstance(layer, tf.keras.layers.InputLayer):
        layer: tf.keras.layers.InputLayer
        shape = layer.batch_shape[1:]
        return InputLayer(name=layer.output.name, shape=shape, parent_model=parent_ennuf_model)
    if isinstance(layer, tf.keras.layers.Concatenate):
        layer: tf.keras.layers.Concatenate
        # See Dense above for description of why the weird split is needed
        input_names: List[str] = [inp.name.split("/")[0] for inp in layer.input]
        inputs: List[BaseLayer] = [parent_ennuf_model.layer_dict[name] for name in input_names]
        return Concatenate(
            name=layer.output.name,
            # the [1:] is needed to skip the None dimension Keras layers have in position 0.
            shape=layer.output.shape[1:],
            inputs=inputs,
            axis=layer.axis,
            parent_model=parent_ennuf_model,
        )
    raise NotImplementedError(f"Could not match a supported layer type to type {type(layer)}")
