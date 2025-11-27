# Usage

How you make use of ennuf depends on what other libraries you want to use it with, if any.

## Using ENNUF with pytorch

### Translating a `nn.Sequential` model

`ennuf.pytorch` provides the `from_sequential` method for translating pytorch models written using `torch.nn.Sequential`.
Example usage:

```python
from ennuf.pytorch import from_sequential

import torch.nn as nn

torch_model = nn.Sequential(
        nn.Conv1d(3, 12, 4, dilation=5),
        nn.Linear(5, 7),
        nn.LeakyReLU(0.7),
    )  # example network
model = from_sequential(torch_model, input_shape=(3, 20), name="my_first_ennuf_nn", input_layers_have_channels=True)
model_mod_path = "/path/to/where/you/want/your/fortran/files"
model.create_fortran_module(model_mod_path)
```

### Translating other pytorch models

You will need to manually specify the architecture of your ML model, 
[see below](usage.md#using-ennuf-without-external-libraries).

The reason for this is that
non-sequential torch models are not guaranteed to have the metadata required to reconstruct them
in a different format available as non-private fields in the classes themselves. 
While they may do so when exported to file, ENNUF does not currently
support translation from saved model files.

## Using ENNUF with tensorflow/keras

### Translating a `Sequential` model

```python
from ennuf.keras import from_sequential

import tensorflow as tf

keras_model = tf.keras.Sequential([
        tf.keras.Input((10,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="tanh"),
        tf.keras.layers.Dense(2)
    ])  # example network
model = from_sequential(keras_model, name="my_first_ennuf_nn")
model_mod_path = "/path/to/where/you/want/your/fortran/files"
model.create_fortran_module(model_mod_path)
```

### Translating a `Functional` model

As above, but with a model created using the functional API,
and `from ennuf.keras import from_functional` instead of `from_sequential`.

## Using ENNUF without external libraries

It's nice to be able to automatically translate your model in a couple of lines of python,
but you can also manually specify your network architecture and generate Fortran based on that.


```python
from ennuf.ml_model import (
    Activation, Concatenate, Conv1d, Dense, Flatten, InputLayer,
    Model, Pooling1d, Reshape, SupportedActivations
)
import numpy as np

ennuf_model = Model(
    name="my_first_ennuf_nn", 
    output_names=["",], 
    description="A model for demonstrating how to use the ENNUF manual API",
    dtype=np.float32,
)
ennuf_model.layers=[
    InputLayer((4,))
]
```
