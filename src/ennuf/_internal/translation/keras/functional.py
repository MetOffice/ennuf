#  (C) Crown Copyright, Met Office, 2023.
import tensorflow as tf

import ennuf._internal.ml_model.model as ennufmodel
from ennuf._internal.translation.keras.keras_layer import from_layer


def from_keras_functional(keras_model: tf.keras.Model) -> ennufmodel.Model:
    """Takes a keras functional model and returns an equivalent ennuf model."""
    dtype = keras_model.variable_dtype
    model = ennufmodel.Model(id_='placeholder', long_name='Placeholder Name', output_names=keras_model.output_names, dtype=dtype)
    for layer in keras_model.layers:
        model.layers.append(from_layer(parent_ennuf_model=model, layer=layer))
    return model
