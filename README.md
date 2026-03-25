# ENNUF

**E**asy **N**eural **N**etworks for **U**se in **F**ortran.

Welcome to the ENNUF documentation!

## What is ENNUF?

ENNUF is software for translating machine learning models defined in python into Fortran code.
This is useful because machine learning models are often developed and trained in python, while
Earth System Models (ESMs) are typically written in Fortran.

### Isn't there other software that does that? Why ENNUF?

Yes, there's lots of software out there for helping you run ML models in Fortran. 
A useful list of other such software
can be found at https://github.com/TRACCS-COMPACT/hybrid_physics_AI_awesome_list.

ENNUF came into existence at the UK Met Office in early 2023 to meet a particular set of requirements.

These were:

1. software for calling machine learning models within a traditional 
physics-based ESM was needed.
2. the solution must not require python code 
to be interpreted at runtime of the ESM, as this will almost certainly be a bottleneck 
for the speed of that model.
3. the solution must work on all compilers and architectures the Met Office models are ran on.
4. the solution must also work on future Met Office HPC systems, 
as these are regularly upgraded every few years.
5. the solution may not involve adding additional dependencies to any Met Office Fortran repositories.
6. if developed in-house, the solution has limits on the amount of staff resources than
can be put towards developing and
subsequently maintaining it, and must therefore be sufficiently easy to maintain and 
limited in scope to only what is required by its users.
7. the solution will not require training of ML models in Fortran, only their inference.
8. the solution will only require those ML model components required by the users, 
with more being added on an as-need basis rather than preemptively.
9. the solution should avoid I/O within the ESM it is deployed, if at all possible.

With all this in mind, existing solutions seemed unsuitable. Cyril Morcrette then had the idea for
ENNUF - to write the components of a Neural Network in Fortran, and to write software that would
then automatically read a python model, and generate some Fortran that calls the right components in
the right order with the right arguments. To avoid point 9 above, the weights of the network 
would be pasted directly into the Fortran code as data statements, and compiled into the
ESM executable.

ENNUF now has all the functionality envisioned at that time, and meets all the above requirements.
If you have a model in python, then (provided it was written in a supported library and 
contains only supported components),
generating a Fortran subroutine that does exactly the same thing as your model in python is as simple
as writing:

```python
from ennuf.your_library import translate
ennuf_model = translate(your_model)
ennuf_model.to_fortran("path/to/where/you/want/your/fortran/files")
```

You'll then get some Fortran files you can include in your Fortran project, whatever it is, and that's
it!

If translating python models to Fortran sounds useful to you,
and the above list of requirements is similar to your own, then consider using ENNUF. 

If you do want to use ENNUF and have questions, or want to contribute, please reach out to us.
