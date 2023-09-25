#  (C) Crown Copyright, Met Office, 2023.
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

import ennuf
from testennuf import RANDOM_SEED
from testennuf.kgo.test_kgo_program_writer import TestKGOProgramWriter


def template_test_kgo(model: ennuf.Model, dir_: Path, kgo_fn: Callable):
    from ennuf.formatters import MinimalistFormatter

    model.formatter = MinimalistFormatter()
    model_mod_path = dir_.joinpath(f'{model.id_}_mod.f90')
    model.create_fortran_module(
        model_mod_path,
        overwrite=True,
        include_neural_net_mod=True
    )
    rng = np.random.default_rng(RANDOM_SEED)
    input_data = {}
    for input_layer in model.inputs:
        name = input_layer.name
        shape = input_layer.shape
        random_input_data = rng.random(size=shape, dtype=np.float32)
        input_data[name] = random_input_data[None]
        random_input_data.T.tofile(dir_.joinpath(f'in_{name}.dat'))

    main_path = dir_.joinpath(f'{model.id_}_kgo_test_program.f90')
    writer = TestKGOProgramWriter(model)
    writer.write(main_path)
    neural_net_mod_path = dir_.joinpath('neural_net_mod.f90')
    executablepath = dir_.joinpath(f"run_{model.id_}")
    f90_files = [model_mod_path, neural_net_mod_path, main_path]
    object_files = [f90_file.with_suffix('.o') for f90_file in f90_files]
    command_compile_nnmod = [ennuf.CONFIG.compiler, '-c', neural_net_mod_path]
    command_compile_mmod = [ennuf.CONFIG.compiler, '-c', model_mod_path]
    command_compile_main = [ennuf.CONFIG.compiler, '-c', main_path]
    command_make_executable = [ennuf.CONFIG.compiler, '-o', executablepath, *object_files]
    command_run_executable = [f'./{executablepath.name}']
    subprocess.call(command_compile_nnmod, cwd=dir_)
    subprocess.call(command_compile_mmod, cwd=dir_)
    subprocess.call(command_compile_main, cwd=dir_)
    subprocess.call(command_make_executable, cwd=dir_)
    subprocess.call(command_run_executable, cwd=dir_)
    hopefully_good_output = {}
    for output_layer in model.outputs:
        name = output_layer.name
        shape = output_layer.shape
        output_data = np.fromfile(dir_.joinpath(f'out_{name}.dat'), dtype=np.float32, count=np.product(shape))
        output_data = output_data.reshape(shape, order='F')
        hopefully_good_output[name] = output_data
    kgo = kgo_fn(input_data)
    for key in kgo:
        assert np.isclose(kgo[key], hopefully_good_output[key]).all()
