import torch
import torch.nn as nn
from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers.concatenate import Concatenate
from ennuf._internal.ml_model.layers.dense import Dense
from ennuf._internal.ml_model.layers.input_layer import InputLayer
import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.supported_activations import SupportedActivations
import ennuf._internal.ml_model.model as ennufmodel
import warnings
import re


def from_sequential(
    pytorch_model: torch.nn.Module,
    input_shape: list,
    name: str = "placeholder",
    long_name: str = "Auto-generated module by ENNUF",
) -> ennufmodel.Model:
    for layer in pytorch_model.children():
        if hasattr(layer,"weight"):
            dtype=layer.weight.detach().numpy().dtype
            break
    layer_names=["input"]
    for i, layer in enumerate(pytorch_model.children()):
        if hasattr(layer,"weight"):
            layer_names.append(str(layer).split("(")[0]+str(i))
    ennuf_model = ennufmodel.Model(
        name=name,
        long_name=long_name,
        output_names=[layer_names[-1]],
        dtype=dtype,
    )
    input_layer=InputLayer(name="input", shape=input_shape, parent_model=ennuf_model)
    ennuf_model.layers.append(input_layer)
    processed_layers=[]
    original_layers=list(pytorch_model.children())
    i=0
    while i < len(original_layers):
        layer=original_layers[i]
        layer_name=str(layer).split("(")[0]+str(i)
        if isinstance(layer, nn.Linear):
            use_bias = layer.bias is not None
            biases = layer.bias.detach().numpy() if use_bias else None
            if i<len(original_layers)-1 and isinstance(original_layers[i+1],(nn.ReLU, nn.Tanh, nn.Sigmoid)):
                activation=str(original_layers[i+1]).split("(")[0].lower()
                activation = SupportedActivations.from_identifier(activation)
                i+=2
            elif i<len(original_layers)-1 and isinstance(original_layers[i+1], nn.LeakyReLU):
                str_fun=re.split("\(|\)",str(original_layers[i+1])[:-1])
                dic={"class_name":str_fun[0], "config": {temp.split("=")[0]:temp.split("=")[1] for temp in str_fun[1:]}}
                activation=SupportedActivations.from_serialized_dict(dic)
                i+=2
            else:
                warnings.warn("Valid activation function not found")
                activation = SupportedActivations.from_identifier(None)
                i+=1
            if processed_layers==[]:
                input_name="input"
            else:
                input_name=processed_layers[-1].name
            processed_layers.append(
                Dense(
                    name=layer_name,
                    inputs=ennuf_model.layer_dict[input_name],
                    parent_model=ennuf_model,
                    units=layer.weight.shape[0],
                    weights=layer.weight.detach().numpy().T,
                    biases=biases,
                    activation=activation,
                    use_bias=use_bias,
                )
            )
        ennuf_model.layers.append(processed_layers[-1])
    return ennuf_model

