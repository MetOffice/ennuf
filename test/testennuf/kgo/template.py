#  (C) Crown Copyright, Met Office, 2023.
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import ennuf
from testennuf import RANDOM_SEED
from testennuf.kgo.test_kgo_program_writer import KGOProgramWriterTester


def template_test_kgo(model: ennuf.Model, dir_: Path, kgo_fn: Callable, model_type: str, atol=1.0e-7, rtol=1.0e-4, testing_convolution=False, original_model=None):
    from ennuf.formatters import MinimalistFormatter

    model.formatter = MinimalistFormatter()
    model_mod_path = dir_.joinpath(f"{model.name}_mod.f90")
    rng = np.random.default_rng(RANDOM_SEED)
    
    if model_type=="keras":
        model.create_fortran_module(model_mod_path, overwrite=True, include_neural_net_mod=True, include_svr_mod=False)
        input_data = {}
        for input_layer in model.inputs:
            name = input_layer.name
            shape = input_layer.shape_without_channels_if_not_provided
            random_input_data = rng.random(size=shape, dtype=np.float32)
            input_data[name] = random_input_data[None]
            random_input_data.T.tofile(dir_.joinpath(f"in_{name}.dat"))
    elif model_type=="pytorch":
        model.create_fortran_module(model_mod_path, overwrite=True, include_neural_net_mod=True, include_svr_mod=False)
        name = list(model.inputs)[0].name
        shape = list(model.inputs)[0].shape_without_channels_if_not_provided
        random_input_data = rng.random(size=shape, dtype=np.float32)
        input_data=random_input_data.copy()
        random_input_data.T.tofile(dir_.joinpath(f"in_{name}.dat"))
    elif model_type=="sklearn":
        model.create_fortran_module(model_mod_path, overwrite=True, include_neural_net_mod=False, include_svr_mod=True)
        name = list(model.inputs)[0].name
        shape = list(model.inputs)[0].shape
        random_input_data = rng.random(size=shape, dtype=np.float32)
        input_data=[random_input_data.copy()]
        random_input_data.T.tofile(dir_.joinpath(f"in_{name}.dat"))
    else:
        raise Exception(f"Unknown model type: {model_type}")


    main_path = dir_.joinpath(f"{model.name}_kgo_test_program.f90")
    writer = KGOProgramWriterTester(model)
    writer.write(main_path)
    neural_net_mod_path = dir_.joinpath("neural_net_mod.f90")
    svr_mod_path = dir_.joinpath("svr_mod.f90")
    executablepath = dir_.joinpath(f"run_{model.name}")
    if model_type=="sklearn":
        f90_files = [model_mod_path, svr_mod_path, main_path]
    else:
        f90_files = [model_mod_path, neural_net_mod_path, main_path]
    object_files = [f90_file.with_suffix(".o") for f90_file in f90_files]

    command_compile_nnmod = [ennuf.CONFIG.compiler, "-c", neural_net_mod_path]
    command_compile_svr = [ennuf.CONFIG.compiler, "-c", svr_mod_path]
    command_compile_mmod = [ennuf.CONFIG.compiler, "-c", model_mod_path]
    command_compile_main = [ennuf.CONFIG.compiler, "-c", main_path]
    command_make_executable = [
        ennuf.CONFIG.compiler,
        "-o",
        executablepath,
        *object_files,
    ]
    command_run_executable = [f"./{executablepath.name}"]
    if model_type=="sklearn":
        subprocess.call(command_compile_svr, cwd=dir_)
    else:
        subprocess.call(command_compile_nnmod, cwd=dir_)

    subprocess.call(command_compile_mmod, cwd=dir_)
    subprocess.call(command_compile_main, cwd=dir_)
    subprocess.call(command_make_executable, cwd=dir_)
    subprocess.call(command_run_executable, cwd=dir_)
    hopefully_good_output = {}
    for output_layer in model.outputs:
        name = output_layer.name
        shape = output_layer.shape
        output_data = np.fromfile(dir_.joinpath(f"out_{name}.dat"), dtype=np.float32, count=np.prod(shape))
        output_data = output_data.reshape(shape, order="F")
        hopefully_good_output[name] = output_data
    if model_type=="pytorch":
        kgo = kgo_fn(torch.from_numpy(input_data)).detach().numpy()
    elif testing_convolution:
        # kgo = kgo_fn({k: v.T.squeeze()[None] for k, v in input_data.items()}) # .T.squeeze()[None] is a temp fix for convolution testing
        kgo = kgo_fn(input_data)
    else:
        kgo = kgo_fn(input_data)
    if isinstance(kgo, dict):
        for key in kgo:
            kgo_data: np.ndarray = kgo[key].squeeze()
            possible_keys = []
            out_data = None
            if key in hopefully_good_output:
                out_data = hopefully_good_output[key].squeeze()
            else:
                # keys are different between fortran and python, need to search
                for out_key, possibly_matching_out_data in hopefully_good_output.items():
                    if possibly_matching_out_data.squeeze().shape == kgo_data.shape:
                        possible_keys.append(out_key)
                if len(possible_keys) == 0:
                    print(f"{kgo_data.shape=} matched none of the following: {[{a_tuple[0]: a_tuple[1].squeeze().shape} for a_tuple in hopefully_good_output.items()]}", file=sys.stderr)
                    raise AssertionError("no outputs of hopefully good output matched shape of kgo data")
                for possible_key in possible_keys:
                    try:
                        out_data = hopefully_good_output[possible_key].squeeze()
                        compare_data(atol, kgo_data, out_data)
                    except AssertionError:
                        out_data = None
                        continue
                assert out_data is not None  # if test fails here, no data matched
            compare_data(atol, kgo_data, out_data)
    else:
        assert len(hopefully_good_output.keys()) == 1
        for key in hopefully_good_output:
            print(f"{kgo=}\n {hopefully_good_output[key]=}")
            print(kgo.squeeze().shape, hopefully_good_output[key].squeeze().shape)
            assert kgo.squeeze().shape == hopefully_good_output[key].squeeze().shape
            assert np.isclose(kgo.squeeze(), hopefully_good_output[key].squeeze(), atol=atol, rtol=rtol).all()


def compare_data(atol, kgo_data, out_data):
    if kgo_data.shape != out_data.shape:
        print(f"kgo shape: {kgo_data.shape}, out shape: {out_data.shape}")
    assert kgo_data.shape == out_data.shape
    if not np.isclose(kgo_data, out_data, atol=atol).all():
        for i, kgo_datum in enumerate(kgo_data):
            out_datum = out_data[i]
            # print(f'kgo: [{kgo_datum}], out: [{out_data[i]}]')
            diff = np.abs(kgo_datum - out_datum)
            print(f"diff: [{diff}], reldiff: [{np.abs(diff / kgo_datum)}]")
    print(f"{kgo_data=}\n {out_data=}")
    print(f"{kgo_data.shape=}, {out_data.shape=}")
    assert np.isclose(kgo_data, out_data, atol=atol).all()