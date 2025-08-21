#  (C) Crown Copyright, Met Office, 2024.

import shutil

import pytest

from testennuf import TMPDIR, WEIGHTS_DIR
from testennuf.example_models.sequential import SequentialExamples
from testennuf.example_models.simple_mlp import SimpleMLP
from testennuf.example_models.t_anomaly_covariances_01 import TAnomalyCovariances01
from testennuf.kgo.template import template_test_kgo


def template_test_keras_functional(keras_model):
    from ennuf.keras import from_functional

    model = from_functional(keras_model)
    dir_ = TMPDIR.joinpath("keras", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, keras_model.predict, model_type="keras")
    shutil.rmtree(dir_)

def template_test_keras_sequential(keras_model):
    from ennuf.keras import from_sequential

    model = from_sequential(keras_model)
    dir_ = TMPDIR.joinpath("keras", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, keras_model.predict, model_type="keras")
    shutil.rmtree(dir_)


def test_keras_sequential():
    keras_model = SequentialExamples.simple_mlp()
    template_test_keras_sequential(keras_model)

def test_keras_sequential_with_activations():
    keras_model = SequentialExamples.simple_mlp_with_activations()
    template_test_keras_sequential(keras_model)

def test_keras_sequential_with_activation_layers():
    keras_model = SequentialExamples.simple_mlp_with_activations_as_layers()
    template_test_keras_sequential(keras_model)

def test_keras_with_reshape():
    keras_model = SequentialExamples.simple_mlp_with_reshape()
    template_test_keras_sequential(keras_model)

def test_keras_functional_examples():
    keras_model = SimpleMLP.build_functional_easy()
    template_test_keras_functional(keras_model)


def test_functional():
    keras_model = SimpleMLP.build_functional()
    template_test_keras_functional(keras_model)


def test_functional_2():
    keras_model = SimpleMLP.build_functional_2()
    template_test_keras_functional(keras_model)


def test_functional_1_1_raises_not_implemented_error():
    keras_model = SimpleMLP.build_functional_1_1()
    with pytest.raises(NotImplementedError):
        template_test_keras_functional(keras_model)

def test_cov_pred_sep_outputs_small():
    keras_model = SimpleMLP.build_covariance_predictor_separate_outputs_small()
    template_test_keras_functional(keras_model)

def test_cov_pred_sep_outputs():
    keras_model = SimpleMLP.build_covariance_predictor_separate_outputs()
    template_test_keras_functional(keras_model)
    #
    # keras_model = SimpleMLP.build_covariance_predictor()
    # template_test_keras_functional(keras_model)


def test_t_anomaly_covariances_01():
    keras_model = TAnomalyCovariances01.build_function(0.1)
    vn = 1
    name = f"TAC01v{vn:03d}"
    weights_dir = WEIGHTS_DIR.joinpath(name)
    try:
        keras_model.load_weights(weights_dir)
    except ValueError as e:
        if not ("File format not supported" in str(e)):
            raise e
        print("File format not supported, skipping test")
    template_test_keras_functional(keras_model)
