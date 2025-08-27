#  (C) Crown Copyright, Met Office, 2024.

import shutil

import pytest
import tensorflow as tf

from testennuf import TMPDIR, RANDOM_SEED
from testennuf.example_models.sequential import SequentialExamples
from testennuf.example_models.simple_mlp import SimpleMLP
from testennuf.kgo.template import template_test_kgo

@pytest.fixture
def set_seed():
    tf.random.set_seed(RANDOM_SEED)

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


def test_keras_sequential(set_seed):
    keras_model = SequentialExamples.simple_mlp()
    template_test_keras_sequential(keras_model)

def test_keras_sequential_with_activations(set_seed):
    keras_model = SequentialExamples.simple_mlp_with_activations()
    template_test_keras_sequential(keras_model)

def test_keras_sequential_with_activation_layers(set_seed):
    keras_model = SequentialExamples.simple_mlp_with_activations_as_layers()
    template_test_keras_sequential(keras_model)

def test_keras_with_reshape(set_seed):
    keras_model = SequentialExamples.simple_mlp_with_reshape()
    template_test_keras_sequential(keras_model)

def test_keras_functional_examples(set_seed):
    keras_model = SimpleMLP.build_functional_easy()
    template_test_keras_functional(keras_model)


def test_functional(set_seed):
    keras_model = SimpleMLP.build_functional()
    template_test_keras_functional(keras_model)


def test_functional_2(set_seed):
    keras_model = SimpleMLP.build_functional_2()
    template_test_keras_functional(keras_model)


def test_functional_1_1(set_seed):
    keras_model = SimpleMLP.build_functional_1_1()
    template_test_keras_functional(keras_model)

def test_cov_pred_sep_outputs_small(set_seed):
    keras_model = SimpleMLP.build_covariance_predictor_separate_outputs_small()
    template_test_keras_functional(keras_model)

def test_cov_pred_sep_outputs(set_seed):
    keras_model = SimpleMLP.build_covariance_predictor_separate_outputs()
    template_test_keras_functional(keras_model)
    #
    # keras_model = SimpleMLP.build_covariance_predictor()
    # template_test_keras_functional(keras_model)
