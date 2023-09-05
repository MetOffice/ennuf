#  (C) Crown Copyright, Met Office, 2023.
import ast

from ennuf.const.path import KERAS_FUNCTIONAL_EXAMPLES_DIR
from ennuf.python_parser.keras_functional import KerasFunctionalModelAnalyzer
from ennuf.simple_mlp import SimpleMLP
from ennuf.translation.keras.functional import from_keras_functional


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    model = SimpleMLP.build()
    model = from_keras_functional(model)
    print(model)
    fortran_file = ''
    fortran_file += model.fortran_file_head()
    fortran_file += model.fortran_module_head()
    fortran_file += model.fortran_subroutine()
    fortran_file += model.fortran_module_tail()
    with open('test.f90', 'w') as file:
        file.write(fortran_file)

    return
    with open(KERAS_FUNCTIONAL_EXAMPLES_DIR.iterdir().__next__(), 'r') as source:
        tree = ast.parse(source.read())
    visitor = KerasFunctionalModelAnalyzer()
    visitor.visit(tree)  #
    # visitor.report()
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
