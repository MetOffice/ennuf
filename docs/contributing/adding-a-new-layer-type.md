# Adding a new layer type

New layers should have a Python component and a Fortran component.

## Fortran component

The Fortran implementation of the layer should go in `ennuf/_internal/fortran/neural_net_mod.f90`.
Typically, this means adding a new subroutine to the module.

## Python component

### The `layer` class

This is our internal representation of the layer that acts as a middle step between
the Python implementation (in Keras, PyTorch, etc.) and Fortran.

In `ennuf/_internal/ml_model/layers` you should add `yourlayer.py`, containing a class
extending the layer abstract base class `BaseLayer`, so `YourLayer(BaseLayer)`.

> `layer` classes are NOT implementations of those layers. For example, a `Dense` layer
> class should not implement any matrix multiplication logic for doing forward passes
> through a neural network, applying activations, backward passes during training, etc. 
> 
> It simply holds the minimum amount of information to
> recreate that layer in our Fortran implementation (number of neurons, 
> activation type, etc);
> and has methods for automatically generating that Fortran.


All abstract methods must be implemented.

### Translation

In addition to the above, we need to implement a way to translate between supported
Python machine learning libraries and our internal representation of the layer.

When adding a new layer, support needs to be added in `ennuf/_internal/translation`.
This should detect if a matching layer type from the external library is used, and
extract all relevant information to create an instance of the `layer` class above.