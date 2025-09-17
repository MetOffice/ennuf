#  (C) Crown Copyright, Met Office, 2024.
"""Module for defining how Keras layers should be translated to ennuf layers"""
from typing import Dict, List

import numpy as np
import tensorflow as tf

from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers.concatenate import Concatenate
from ennuf._internal.ml_model.layers.dense import Dense
from ennuf._internal.ml_model.layers.convolutional import Conv1d, PaddingMode
from ennuf._internal.ml_model.layers.pooling import Pooling1d
from ennuf._internal.ml_model.layers.activation import Activation
from ennuf._internal.ml_model.activations.linear import Linear
from ennuf._internal.ml_model.layers.reshape import Reshape
from ennuf._internal.ml_model.layers.flatten import Flatten
from ennuf._internal.ml_model.layers.input_layer import InputLayer
import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.supported_activations import SupportedActivations
from ennuf._internal.ml_model.activations.relu import Relu
from ennuf._internal.ml_model.activations.leaky_relu import LeakyRelu
from ennuf._internal.ml_model.activations.tanh import Tanh
from ennuf._internal.ml_model.activations.sigmoid import Sigmoid


def from_layer(parent_ennuf_model: model.Model, layer, input_layer_channels: str | None, previous_layer_name=None) -> \
List[
    BaseLayer]:
    """Takes a keras model and a keras layer and returns an equivalent ennuf layer."""
    if isinstance(layer, tf.keras.layers.Dense):
        layer: tf.keras.layers.Dense
        use_bias = layer.bias is not None
        biases = layer.get_weights()[1] if use_bias else None
        kas_activation = tf.keras.activations.serialize(layer.activation)
        if isinstance(kas_activation, str):
            activation = SupportedActivations.from_identifier(kas_activation)
        elif isinstance(kas_activation, Dict):
            activation = SupportedActivations.from_serialized_dict(kas_activation)
        else:
            activation = None
        input_name: str = layer.input.name if previous_layer_name is None else previous_layer_name
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

        input_name = input_name.split("/")[0] if previous_layer_name is None else previous_layer_name
        layer_name = layer.output.name
        input_layer = parent_ennuf_model.layer_dict[input_name]
        if not isinstance(input_layer.shape, int) and len(input_layer.shape) == 2:
            shape = (input_layer.shape[0], layer.units)
        else:
            shape = layer.units
        ennuf_dense_layer = Dense(
            name=layer_name,
            inputs=input_layer,
            parent_model=parent_ennuf_model,
            shape=shape,
            weights=layer.get_weights()[0],
            biases=biases,
            use_bias=use_bias,
        )
        if activation is None or isinstance(activation, Linear):
            return [ennuf_dense_layer]
        else:
            ennuf_activation_layer = Activation(
                name=f"{layer_name}_activation",
                shape=layer.get_weights()[0].shape[1],
                inputs=ennuf_dense_layer,
                parent_model=parent_ennuf_model,
                activation=activation,
            )
            return [
                ennuf_dense_layer,
                ennuf_activation_layer,
            ]
    if isinstance(layer, tf.keras.layers.InputLayer):
        layer: tf.keras.layers.InputLayer
        shape = layer.batch_shape[1:]
        # if input_layer_channels == "last" and len(shape) > 1:
        #     shape = (shape[-1],) + shape[:-1]
        has_channels = False if input_layer_channels is None else True
        return [InputLayer(name=layer.output.name, shape=shape, has_channels=has_channels,
                           parent_model=parent_ennuf_model)]
    if isinstance(layer, tf.keras.layers.Concatenate):
        layer: tf.keras.layers.Concatenate
        # See Dense above for description of why the weird split is needed
        input_names: List[str] = [inp.name.split("/")[0] for inp in layer.input]
        inputs: List[BaseLayer] = [parent_ennuf_model.layer_dict[name] for name in input_names]
        return [Concatenate(
            name=layer.output.name,
            # the [1:] is needed to skip the None dimension Keras layers have in position 0.
            shape=layer.output.shape[1:],
            inputs=inputs,
            axis=layer.axis,
            parent_model=parent_ennuf_model,
        )]
    if isinstance(layer, tf.keras.layers.Reshape):
        layer: tf.keras.layers.Reshape
        input_name = previous_layer_name
        if input_name is None:
            raise NotImplementedError(f"Cannot fetch name of input to reshape layer other than when provided directly.")
        return [Reshape(
            name=layer.output.name,
            shape=layer.target_shape,
            inputs=parent_ennuf_model.layer_dict[input_name],
            parent_model=parent_ennuf_model
        )]
    if isinstance(layer, tf.keras.layers.ReLU):
        input_name = layer.input.name if previous_layer_name is None else previous_layer_name
        if hasattr(layer, "negative_slope") and layer.negative_slope != 0.0:
            return [Activation(
                name=layer.output.name,
                shape=layer.output.shape[1:],
                inputs=parent_ennuf_model.layer_dict[input_name],
                parent_model=parent_ennuf_model,
                activation=LeakyRelu(layer.negative_slope),
            )]
        return [Activation(
            name=layer.output.name,
            shape=layer.output.shape[1:],
            inputs=parent_ennuf_model.layer_dict[input_name],
            parent_model=parent_ennuf_model,
            activation=Relu(),
        )]
    if isinstance(layer, tf.keras.layers.Activation):
        input_name = layer.input.name if previous_layer_name is None else previous_layer_name
        kas_activation = tf.keras.activations.serialize(layer)["config"]["activation"]
        if isinstance(kas_activation, str):
            activation = SupportedActivations.from_identifier(kas_activation)
        elif isinstance(kas_activation, Dict):
            activation = SupportedActivations.from_serialized_dict(kas_activation)
        else:
            activation = None
        return [Activation(
            name=layer.output.name,
            shape=layer.output.shape[1:],
            inputs=parent_ennuf_model.layer_dict[input_name],
            parent_model=parent_ennuf_model,
            activation=activation,
        )]
    if isinstance(layer, tf.keras.layers.Conv1D):
        input_name = layer.input.name if previous_layer_name is None else previous_layer_name
        kas_activation = tf.keras.activations.serialize(layer.activation)
        if isinstance(kas_activation, str):
            activation = SupportedActivations.from_identifier(kas_activation)
        elif isinstance(kas_activation, Dict):
            activation = SupportedActivations.from_serialized_dict(kas_activation)
        else:
            activation = None
        input_layer = parent_ennuf_model.layer_dict[input_name]
        kernel_weights = layer.get_weights()[0].T
        layer_name = layer.output.name
        use_bias = layer.bias is not None
        biases = layer.get_weights()[1] if use_bias else None
        padding_mode = PaddingMode.from_keras_padding_mode(layer.padding)
        stride = layer.strides[0] if len(layer.strides) == 1 else None
        if stride is None:
            raise NotImplementedError(f"Conv1d supports only strides of length 1 but got {len(layer.strides)}")
        if padding_mode is PaddingMode.NONE:
            padding = 0
        elif layer.padding == "same":
            padding = _compute_same_padding(input_layer.shape[0], kernel_weights.shape[2], stride)
        else:
            raise NotImplementedError(f"Unsupported padding {layer.padding}")
        dilation = layer.dilation_rate[0] if len(layer.dilation_rate) == 1 else None
        if dilation is None:
            raise NotImplementedError(
                f"Conv1d supports only dilation_rate of length 1 but got {len(layer.dilation_rate)}.")
        conv_input_layer = input_layer
        layers = []
        if input_layer_channels == "last" and len(input_layer.shape) > 1:
            fixed_shape = (input_layer.shape[-1],)+ils if \
                len(ils := input_layer.shape[:-1]) != 1 else (input_layer.shape[-1], ils[0])
            pre_reshape_layer = Reshape(
                name=f"{layer.output.name}_pre_reshape",
                shape=fixed_shape,
                inputs=input_layer,
                parent_model=parent_ennuf_model,
            )
            conv_input_layer = pre_reshape_layer
            layers.append(pre_reshape_layer)
        ennuf_conv_layer = Conv1d(
            name=layer_name,
            inputs=conv_input_layer,
            parent_model=parent_ennuf_model,
            weights=kernel_weights,
            biases=biases,
            padding_mode=padding_mode,
            padding=padding,
            stride=stride,
            dilation=dilation,
            use_bias=use_bias,
        )
        layers.append(ennuf_conv_layer)
        activation_input_layer = ennuf_conv_layer
        if input_layer_channels == "last" and len(ennuf_conv_layer.shape) > 1:
            fixed_shape = ils + (ennuf_conv_layer.shape[0],) if \
            len(ils := ennuf_conv_layer.shape[1:]) != 1 else (ils[0],ennuf_conv_layer.shape[0])
            post_reshape_layer = Reshape(
                name=f"{layer.output.name}_post_reshape",
                shape=fixed_shape,
                inputs=ennuf_conv_layer,
                parent_model=parent_ennuf_model,
            )
            activation_input_layer = post_reshape_layer
            layers.append(post_reshape_layer)
        if not (activation is None or isinstance(activation, Linear)):
            ennuf_activation_layer = Activation(
                name=f"{layer_name}_activation",
                shape=activation_input_layer.shape,
                inputs=activation_input_layer,
                parent_model=parent_ennuf_model,
                activation=activation,
            )
            layers.append(ennuf_activation_layer)
        return layers
    if isinstance(layer, tf.keras.layers.Flatten):
        input_name = previous_layer_name
        if input_name is None:
            raise NotImplementedError(f"Cannot fetch name of input to flatten layer other than when provided directly.")
        input_layer = parent_ennuf_model.layer_dict[input_name]
        # if input_layer_channels == "last" and len(input_layer.shape)>1:
        #     fixed_shape =  ils + (input_layer.shape[0],) if len(ils := input_layer.shape[1:]) != 1 else (ils[0], input_layer.shape[0])
        #     reshape_layer = Reshape(
        #         name=f"{layer.output.name}_fix_channels",
        #         shape=fixed_shape,
        #         inputs=input_layer,
        #         parent_model=parent_ennuf_model,
        #     )
        #     flatten_layer = Flatten(
        #         name=layer.output.name,
        #         inputs=reshape_layer,
        #         parent_model=parent_ennuf_model,
        #     )
        #     return [reshape_layer, flatten_layer]
        return [Flatten(
            name=layer.output.name,
            inputs=input_layer,
            parent_model=parent_ennuf_model
        )]
    is_avg = isinstance(layer, tf.keras.layers.AvgPool1D)
    if (is_max := isinstance(layer, tf.keras.layers.MaxPool1D)) or is_avg:
        input_name = previous_layer_name
        if input_name is None:
            raise NotImplementedError(f"Cannot fetch name of input to pooling layer other than when provided directly.")
        input_layer = parent_ennuf_model.layer_dict[input_name]
        if len(layer.pool_size) > 1:
            raise ValueError(f"Unexpected pool size shape for 1d pooling, got {layer.pool_size=}")
        pool_size = layer.pool_size[0]
        padding_mode = PaddingMode.from_keras_padding_mode(layer.padding)
        stride = layer.strides[0] if len(layer.strides) == 1 else None
        if stride is None:
            raise NotImplementedError(f"Conv1d supports only strides of length 1 but got {len(layer.strides)}")
        if padding_mode is PaddingMode.NONE:
            padding = 0
        elif layer.padding == "same":
            padding = _compute_same_padding(input_layer.shape[0], pool_size, stride)
        else:
            raise NotImplementedError(f"Unsupported padding {layer.padding}")
        type_of_pooling = "MAX" if is_max else "AVG" if is_avg else None
        if type_of_pooling is None:
            raise ValueError("Pooling type should be MAX or AVG.")
        layers = []
        pooling_input_layer = input_layer
        if input_layer_channels == "last" and len(input_layer.shape) > 1:
            fixed_shape = (input_layer.shape[-1],)+ils if \
                len(ils := input_layer.shape[:-1]) != 1 else (input_layer.shape[-1], ils[0])
            pre_reshape_layer = Reshape(
                name=f"{layer.output.name}_pre_reshape",
                shape=fixed_shape,
                inputs=input_layer,
                parent_model=parent_ennuf_model,
            )
            pooling_input_layer = pre_reshape_layer
            layers.append(pre_reshape_layer)
        ennuf_pooling_layer = Pooling1d(name=layer.output.name, inputs=pooling_input_layer, parent_model=parent_ennuf_model,
                                        pool_size=pool_size,
                                        type_of_pooling=type_of_pooling, padding=padding, stride=stride, )
        layers.append(ennuf_pooling_layer)
        if input_layer_channels == "last" and len(ennuf_pooling_layer.shape) > 1:
            fixed_shape = ils + (ennuf_pooling_layer.shape[0],) if \
            len(ils := ennuf_pooling_layer.shape[1:]) != 1 else (ils[0],ennuf_pooling_layer.shape[0])
            post_reshape_layer = Reshape(
                name=f"{layer.output.name}_post_reshape",
                shape=fixed_shape,
                inputs=ennuf_pooling_layer,
                parent_model=parent_ennuf_model,
            )
            layers.append(post_reshape_layer)
        return layers
    raise NotImplementedError(f"Could not match a supported layer type to type {type(layer)}")


def _compute_same_padding(input_length, kernel_size, stride):
    # Output length for 'same' padding
    output_length = int((input_length + stride - 1) // stride)
    # Total padding
    padding = max(0, (output_length - 1) * stride + kernel_size - input_length)
    # Split padding to left and right
    pad_left = padding // 2
    pad_right = padding - pad_left
    if pad_left != pad_right:
        raise NotImplementedError(f"padding only supported if equal on all sides, but got {pad_left=}, {pad_right=}")
    return pad_left
