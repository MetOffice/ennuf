#  (C) Crown Copyright, Met Office, 2023.
"""Module for the translation of models defined with the Keras Functional API"""
import tensorflow as tf

import ennuf._internal.ml_model.model as ennufmodel
from ennuf._internal.translation.keras.keras_layer import from_layer


def from_functional(
    keras_model: tf.keras.Model,
    name: str = "placeholder",
    long_name: str = "Auto-generated module by ENNUF",
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
    if isinstance(keras_model.output_shape, dict):
        if set(keras_model.output_names) != set(keras_model.output_shape.keys()):
            # TODO: handle this in KGO testing somehow
            raise NotImplementedError(
                "Currently you must name your model outputs the same name as the layers they are connected to. "
                "This only matters for automated KGO testing and should not impact functionality "
                "if no fix was implemented."
            )

    dtype = keras_model.variable_dtype
    model = ennufmodel.Model(
        name=name,
        long_name=long_name,
        output_names=keras_model.output_names,
        dtype=dtype,
    )
    for layer in keras_model.layers:
        model.layers.append(from_layer(parent_ennuf_model=model, layer=layer))
    return model
