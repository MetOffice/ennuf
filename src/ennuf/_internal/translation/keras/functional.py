#  (C) Crown Copyright, Met Office, 2024.
"""Module for the translation of models defined with the Keras Functional API"""
import tensorflow as tf

import ennuf._internal.ml_model.model as ennufmodel
from ennuf._internal.translation.keras.keras_layer import from_layer


def from_functional(
    keras_model: tf.keras.Model,
    name: str = "placeholder",
    long_name: str = "Auto-generated module by ENNUF",
    input_layers_have_channels = False,
) -> ennufmodel.Model:
    """
    Takes a keras functional model and returns an equivalent ENNUF model.

    Parameters
    ----------
    keras_model
        A model created with the Keras functional API (NOT Sequential)
    name
        The name of the model as you wish it to be
        referred to in Fortran. Must be a valid Fortran identifier.
        Should be written in snake_case, and not be a pre-existing
        Fortran keyword such as random_number or allocate.
    long_name
        A longer, more descriptive name of the model.
        This will only appear in comments, so any valid text to appear
        in Fortran comments is fine.
    input_layers_have_channels
        Whether the inputs to the model are assumed to have the first dimension be a channels dimension (common in CNNs).
        Defaults to false.
    Returns
    -------
    Returns
        An ENNUF model, which is a simplified representation of advanced models
        of the sort created by TensorFlow or PyTorch.
        An ENNUF model supports only a small subset of the possible
        network architectures out there - only the ones we have written
        ways of representing in Fortran. The model
        will have methods on it which can be used to write a Fortran module based on it.
    """
    dtype = keras_model.variable_dtype
    model = ennufmodel.Model(
        name=name,
        long_name=long_name,
        output_names=[""],
        dtype=dtype,
    )
    for layer in keras_model.layers:
        previous_layer_name = None
        if hasattr(layer, "input"):
            if hasattr(layer.input, "name"):
                keras_input_name = layer.input.name
                previous_layer_name = keras_input_name
                for potential_previous_layer in keras_model.layers:
                    if potential_previous_layer.output.name == keras_input_name and hasattr(potential_previous_layer, "activation"):
                        if not (potential_previous_layer.activation is tf.keras.activations.linear):
                            previous_layer_name = f"{keras_input_name}_activation"
        ennuf_layers = from_layer(parent_ennuf_model=model, layer=layer, input_layers_have_channels=input_layers_have_channels, previous_layer_name=previous_layer_name)
        for ennuf_layer in ennuf_layers:
            model.layers.append(ennuf_layer)
    output_names = []
    for layer in model.layers:
        layer_is_output = True
        for possibly_next_layer in model.layers:
            if possibly_next_layer.inputs is not None:
                if isinstance(possibly_next_layer.inputs, list):
                    if layer in possibly_next_layer.inputs:
                        layer_is_output = False
                        continue
                else:
                    if layer == possibly_next_layer.inputs:
                        layer_is_output = False
                        continue
        if layer_is_output:
            output_names.append(layer.output_name.strip("y_"))
    model.output_names = output_names
    return model
