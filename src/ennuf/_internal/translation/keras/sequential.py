import tensorflow as tf
import ennuf._internal.ml_model.model as ennufmodel
from ennuf._internal.translation.keras.keras_layer import from_layer
from ennuf._internal.ml_model.layers.input_layer import InputLayer
def from_sequential(
        keras_model: tf.keras.Sequential,
        name: str = "placeholder",
        long_name: str = "Auto-generated module by ENNUF",
) -> ennufmodel.Model:
    dtype = keras_model.variable_dtype
    if len(keras_model.outputs) != 1:
        raise ValueError(f"Expected Sequential model to have a single output layer, "
                         f"got {len(keras_model.outputs)}: ({keras_model.outputs})")
    output_name = keras_model.outputs[0].name
    model = ennufmodel.Model(
        name=name,
        long_name=long_name,
        output_names=[output_name],
        dtype=dtype,
    )
    if len(keras_model.inputs) != 1:
        raise ValueError(f"Expected Sequential model to have a single input layer, "
                         f"got {len(keras_model.inputs)}: ({keras_model.inputs})")
    input_layer = keras_model.inputs[0]
    input_layer_shape = input_layer.shape[1:]  # strips leading None
    model.layers.append(InputLayer(name=input_layer.name, shape=input_layer_shape, parent_model=model))
    for layer in keras_model.layers:
        model.layers.append(from_layer(parent_ennuf_model=model, layer=layer, previous_layer_name=model.layers[-1].name))
    return model
