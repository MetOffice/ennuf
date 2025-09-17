#  (C) Crown Copyright, Met Office, 2024.

import shutil

import pytest
import tensorflow as tf

from testennuf import TMPDIR, RANDOM_SEED
from testennuf.example_models.keras_convolutional import KerasConvolutional
from testennuf.example_models.sequential import SequentialExamples
from testennuf.example_models.simple_mlp import SimpleMLP
from testennuf.kgo.template import template_test_kgo


@pytest.fixture
def set_seed():
    tf.random.set_seed(RANDOM_SEED)


def template_test_keras_functional(keras_model, input_layer_channels=None, isconv=False):
    from ennuf.keras import from_functional

    model = from_functional(keras_model, input_layer_channels=input_layer_channels)
    dir_ = TMPDIR.joinpath("keras", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, keras_model.predict, model_type="keras", testing_convolution=isconv, original_model=keras_model)
    shutil.rmtree(dir_)


def template_test_keras_sequential(keras_model, input_layer_channels=None, isconv=False):
    from ennuf.keras import from_sequential

    model = from_sequential(keras_model, input_layer_channels=input_layer_channels)
    dir_ = TMPDIR.joinpath("keras", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, keras_model.predict, model_type="keras", testing_convolution=isconv)
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

def test_flatten(set_seed):
    keras_model = KerasConvolutional.only_flatten()
    template_test_keras_functional(keras_model)

def test_only_conv(set_seed):
    keras_model = KerasConvolutional.build_only_conv()
    template_test_keras_functional(keras_model, "last", isconv=True)

def test_conv_with_dense(set_seed):
    keras_model = KerasConvolutional.build_conv_with_dense()
    template_test_keras_functional(keras_model, "last", isconv=True)

def test_simple_conv(set_seed):
    keras_model = KerasConvolutional.build_simple_conv()
    template_test_keras_functional(keras_model, "last", isconv=True)




def test_deep_conv(set_seed):
    keras_model = KerasConvolutional.build_deep_conv()
    template_test_keras_functional(keras_model, "last", isconv=True)


def test_conv_multi_output(set_seed):
    keras_model = KerasConvolutional.build_conv_multi_output()
    template_test_keras_functional(keras_model, "last", isconv=True)


def test_conv_dropout(set_seed):
    keras_model = KerasConvolutional.build_conv_with_dropout()
    template_test_keras_functional(keras_model, "last", isconv=True)
