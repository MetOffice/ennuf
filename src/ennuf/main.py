#  (C) Crown Copyright, Met Office, 2023.
import ennuf
from ennuf.formatters.minimalist_formatter import MinimalistFormatter
from ennuf.simple_mlp import SimpleMLP
from ennuf.translation.keras.functional import from_keras_functional


def main():
    model = SimpleMLP.build()
    model = from_keras_functional(model)
    print(model)
    model.create_fortran_module('um_test.f90')
    model.formatter = MinimalistFormatter()
    model.create_fortran_module('test.f90')
    return


if __name__ == '__main__':
    main()
