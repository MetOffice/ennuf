#  (C) Crown Copyright, Met Office, 2023.
import shutil
import subprocess

import numpy as np
import pytest

import ennuf
from testennuf import TMPDIR, RANDOM_SEED
from testennuf.kgo.template import template_test_kgo


#
# def test_kgo():
#     from ennuf.formatters import MinimalistFormatter
#     from ennuf.keras import from_functional
#     from testennuf.example_models.simple_mlp import SimpleMLP
#
#     test_kgo_dir = TMPDIR
#     keras_model = SimpleMLP.build_functional()
#     model = from_functional(keras_model)
#     model.formatter = MinimalistFormatter()
#     modelpath = test_kgo_dir.joinpath(f'{model.name}_mod.f90')
#     model.create_fortran_module(
#         modelpath,
#         overwrite=True,
#         include_neural_net_mod=True
#     )
#     rng = np.random.default_rng(RANDOM_SEED)
#     input_data = {}
#     for input_layer in model.inputs:
#         name = input_layer.name
#         shape = input_layer.shape
#         random_input_data = rng.random(size=shape, dtype=np.float32)
#         input_data[name] = random_input_data[None]
#         np.savetxt(test_kgo_dir.joinpath(f'{name}.txt'), random_input_data, fmt='%.8f', delimiter=' ')
#         random_input_data.T.tofile(test_kgo_dir.joinpath(f'{name}.dat'))
#     # for output_layer in model.outputs:
#     #     output_data = np.fromfile('otest.dat', dtype=np.float32, count=3 * 2 * 5)
#     #     output_data = output_data.reshape(shape, order='F')
#     #     print(output_data)
#
#     txt_reader_mod_path = TMPDIR.joinpath('matrix_txt_reader_mod.f90')
#     neural_net_mod_path = TMPDIR.joinpath('neural_net_mod.f90')
#     main_path = TMPDIR.joinpath('main.f90')
#     executablepath = test_kgo_dir.joinpath(f"run_{model.name}")
#     f90_files = [modelpath, txt_reader_mod_path, neural_net_mod_path, main_path]
#     object_files = [f90_file.with_suffix('.o') for f90_file in f90_files]
#     command_compile_objects = [ennuf.CONFIG.compiler, '-c', *f90_files]
#     command_make_executable = [ennuf.CONFIG.compiler, '-o', executablepath, *object_files]
#     command_run_executable = [f'./{executablepath.name}']
#     subprocess.call(command_compile_objects, cwd=test_kgo_dir)
#     subprocess.call(command_make_executable, cwd=test_kgo_dir)
#     subprocess.call(command_run_executable, cwd=test_kgo_dir)
#     outputs = keras_model.predict(input_data)
#     print(outputs)  # this is a dict with keys 'outputs_1', 'outputs_2'
#     # TODO: Automatically validate output of python model with output of Fortran model.


def template_test_keras_functional(keras_model):
    from ennuf.keras import from_functional

    model = from_functional(keras_model)
    dir_ = TMPDIR.joinpath('keras', f'{model.name}')
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, keras_model.predict)
    shutil.rmtree(dir_)


def test_simple_mlp():
    from testennuf.example_models.simple_mlp import SimpleMLP

    keras_model = SimpleMLP.build_functional_easy()
    template_test_keras_functional(keras_model)

    keras_model = SimpleMLP.build_functional()
    template_test_keras_functional(keras_model)

    keras_model = SimpleMLP.build_functional_2()
    template_test_keras_functional(keras_model)

    keras_model = SimpleMLP.build_functional_1_1()
    with pytest.raises(NotImplementedError):
        template_test_keras_functional(keras_model)
