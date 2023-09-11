#  (C) Crown Copyright, Met Office, 2023.
import ennuf
from ennuf import from_keras_functional
from ennuf.formatters import MinimalistFormatter
from example_package._simple_mlp import SimpleMLP


def main():
    model = SimpleMLP.build()
    model = from_keras_functional(model)
    print(model)
    model.create_fortran_module('um_test.f90', overwrite=True)
    model.formatter = MinimalistFormatter()
    model.create_fortran_module('test.f90', overwrite=True)
    print(f"Created fortran files for model {model.long_name}")
    model = ennuf.Model('scratch_ml', 'ML_FROM_SCRATCH', ['output'], )
    sqmodel = SimpleMLP.build_sequential()
    model = from_keras_functional(sqmodel)
    print(model)
    return


if __name__ == '__main__':
    main()
