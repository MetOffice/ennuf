# ENNUF

## What is it?

**E**asy **N**eural **N**etworks for the **U**M in **F**ortran.

Born out of the need for a temporary solution to deal with the fact that we currently 
have no good way of running machine learning models (which are typically designed and
trained in python using TensorFlow or PyTorch) in the UM (which is of course written
in Fortran). Calling python code from the UM has been tried and massively slows it down.
We'd like a way of bridging the gap that isn't a fully-fledged TensorFlow/PyTorch 
library rewritten in Fortran, since we absolutely do not currently have the resources 
in MSE to create and maintain something like that.

However, we only currently *need* a few things: to run pre-trained models with a small 
subset of the range of features such models are permitted to exhibit within the python
libraries.

We could read in the weights and biases of these models from some external files, which
could require a lot of changes to the UM, and IO control is naturally not a part of the
codebase we'd normally touch.

However, you can also just initialise the weights directly in a Fortran file!

The above realisations led Cyril Morcrette to come up with ENNUF - something that takes
a python model and pastes all the weights and biases directly into a Fortran file.

At first this was a more manually controlled process, with neural network structure
(layer sizes, types etc.) all needing to be specified in a Fortran head and tail files,
the weights then being pasted inbetween and the files being stitched together.

However, automating this process as much as possible is desirable.

## TODO:

- Add INTENT statements