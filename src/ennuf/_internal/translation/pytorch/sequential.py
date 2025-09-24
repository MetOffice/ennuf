# (C) Crown Copyright, Met Office, 2025.
import numpy as np
import torch
import torch.nn as nn
from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers.concatenate import Concatenate
from ennuf._internal.ml_model.layers.dense import Dense
from ennuf._internal.ml_model.layers.flatten import Flatten
from ennuf._internal.ml_model.layers.convolutional import Conv1d
from ennuf._internal.ml_model.layers.pooling import Pooling1d
from ennuf._internal.ml_model.layers.reshape import Reshape
from ennuf._internal.ml_model.layers.convolutional import PaddingMode
from ennuf._internal.ml_model.layers.input_layer import InputLayer
import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.supported_activations import SupportedActivations
import ennuf._internal.ml_model.model as ennufmodel
import warnings
import re
from ennuf._internal.ml_model.layers.activation import Activation
from ennuf._internal.ml_model.activations.linear import Linear


def from_sequential(
        pytorch_model: torch.nn.Module,
        input_shape: tuple[int, ...],
        name: str = "placeholder",
        long_name: str = "Auto-generated module by ENNUF",
        input_layers_have_channels: bool = False,
        dtype: np.dtype | None = None,
) -> ennufmodel.Model:
    for layer in pytorch_model.children():
        if hasattr(layer, "weight"):
            dtype = layer.weight.detach().numpy().dtype
            break
    layer_names = ["input"]
    for i, layer in enumerate(pytorch_model.children()):
        if hasattr(layer, "weight"):
            layer_names.append(str(layer).split("(")[0] + str(i))
    if dtype is None:
        raise ValueError("Unable to find dtype.")
    ennuf_model = ennufmodel.Model(
        name=name,
        long_name=long_name,
        output_names=[layer_names[-1]],
        dtype=dtype,
    )
    input_layer = InputLayer(name="input", has_channels=input_layers_have_channels, shape=input_shape,
                             parent_model=ennuf_model)
    ennuf_model.layers.append(input_layer)
    processed_layers = []
    original_layers = list(pytorch_model.children())
    for i, layer in enumerate(original_layers):
        layers_added = 1  # overwrite this later if not true
        layer_name = str(layer).split("(")[0] + str(i)
        if processed_layers == []:
            input_name = "input"
        else:
            input_name = processed_layers[-1].name
        if isinstance(layer, nn.Linear):
            use_bias = layer.bias is not None
            biases = layer.bias.detach().numpy() if use_bias else None
            previous_layer = ennuf_model.layer_dict[input_name]
            weights = layer.weight.detach().numpy().T
            channels = previous_layer.shape[0] if len(previous_layer.shape) > 1 else 1
            ennuf_dense_layer = Dense(
                name=layer_name,
                inputs=previous_layer,
                parent_model=ennuf_model,
                shape=(channels, weights.shape[1]),
                weights=weights,
                biases=biases,
                use_bias=use_bias,
            )
            processed_layers.append(ennuf_dense_layer)
        elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.Softmax)):
            if isinstance(layer, nn.LeakyReLU):
                str_fun = re.split("\(|\)", str(layer)[:-1])
                dic = {"class_name": str_fun[0],
                       "config": {temp.split("=")[0]: temp.split("=")[1] for temp in str_fun[1:]}}
                activation = SupportedActivations.from_serialized_dict(dic)
            else:
                activation = str(layer).split("(")[0].lower()
                activation = SupportedActivations.from_identifier(activation)
            previous_layer = ennuf_model.layer_dict[input_name]
            activation_shape = previous_layer.shape
            ennuf_activation_layer = Activation(
                name=f"{layer_name}_activation",
                shape=activation_shape,
                inputs=previous_layer,
                parent_model=ennuf_model,
                activation=activation,
            )
            processed_layers.append(ennuf_activation_layer)
        elif isinstance(layer, nn.Flatten):
            ennuf_layer = Flatten(
                name=layer_name,
                inputs=ennuf_model.layer_dict[input_name],
                parent_model=ennuf_model,
            )
            processed_layers.append(ennuf_layer)
        elif isinstance(layer, nn.Conv1d):
            weights = layer.weight.detach().numpy()
            biases = layer.bias
            padding_mode = PaddingMode.from_torch_padding_mode(layer.padding_mode)
            padding = layer.padding[0] if len(layer.padding) == 1 else None
            if padding is None:
                raise NotImplementedError(f"Unsupported padding, expected 1d padding shape but got {layer.padding}")
            stride = layer.stride[0] if len(layer.stride) == 1 else None
            if stride is None:
                raise NotImplementedError(f"Unsupported stride, expected 1d padding stride but got {layer.stride}")
            dilation = layer.dilation[0] if len(layer.dilation) == 1 else None
            if dilation is None:
                raise NotImplementedError(f"Unsupported dilation, expected 1d dilation shape but got {layer.dilation}")
            use_bias = False if layer.bias is None else True
            ennuf_layer = Conv1d(
                name=layer_name,
                inputs=ennuf_model.layer_dict[input_name],
                parent_model=ennuf_model,
                weights=weights,
                biases=biases,
                padding_mode=padding_mode,
                padding=padding,
                stride=stride,
                dilation=dilation,
                use_bias=use_bias,
            )
            processed_layers.append(ennuf_layer)
        elif isinstance(layer, (nn.MaxPool1d, nn.AvgPool1d)):
            pool_size=ks if isinstance(ks:= layer.kernel_size, int) else ks[0] if len(ks) == 1 else None
            if pool_size is None:
                raise NotImplementedError(f"Unsupported pool size, expected 1d pool size shape but got {layer.kernel_size}")
            type_of_pooling = "MAX" if isinstance(layer, nn.MaxPool1d) else "AVG"
            padding = layer.padding if isinstance(layer.padding, int) else layer.padding[0] if len(layer.padding) == 1 else None
            if padding is None:
                raise NotImplementedError(f"Unsupported padding, expected 1d padding shape but got {layer.padding}")
            stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0] if len(layer.stride) == 1 else None
            if stride is None:
                raise NotImplementedError(f"Unsupported stride, expected 1d padding stride but got {layer.stride}")
            ennuf_layer = Pooling1d(
                name=layer_name,
                inputs=ennuf_model.layer_dict[input_name],
                parent_model=ennuf_model,
                pool_size=pool_size,
                type_of_pooling=type_of_pooling,
                padding=padding,
                stride=stride,
            )
            processed_layers.append(ennuf_layer)
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)=} ({layer=})")
        [ennuf_model.layers.append(processed_layer) for processed_layer in processed_layers[-layers_added:]]
    ennuf_model.output_names = [ennuf_model.layers[-1].output_name.strip("y_")]
    return ennuf_model
