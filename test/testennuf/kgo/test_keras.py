#  (C) Crown Copyright, Met Office, 2024.

import shutil

import pytest
import tensorflow as tf

from testennuf import TMPDIR, RANDOM_SEED
from testennuf.example_models.keras_convolutional import KerasConvolutional
from testennuf.example_models.keras_sequential import SequentialExamples
from testennuf.example_models.keras_simple_mlp import SimpleMLP
from testennuf.kgo.template import template_test_kgo


@pytest.fixture
def set_seed():
    tf.random.set_seed(RANDOM_SEED)


def template_test_keras_functional(keras_model, input_layer_channels=None):
    from ennuf.keras import from_functional

    model = from_functional(keras_model, input_layer_channels=input_layer_channels)
    dir_ = TMPDIR.joinpath("keras", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, keras_model.predict, model_type="keras", original_model=keras_model)
    shutil.rmtree(dir_)


def template_test_keras_sequential(keras_model, input_layer_channels=None):
    from ennuf.keras import from_sequential

    model = from_sequential(keras_model, input_layer_channels=input_layer_channels)
    dir_ = TMPDIR.joinpath("keras", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, keras_model.predict, model_type="keras", original_model=keras_model)
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


def test_keras_sequential_with_several_layers(set_seed):
    keras_model = SimpleMLP.build_sequential()
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
    template_test_keras_functional(keras_model, "last")


def test_only_conv_no_bias(set_seed):
    keras_model = KerasConvolutional.build_only_conv_no_bias()
    with pytest.raises(NotImplementedError):
        template_test_keras_functional(keras_model, "last")


def test_conv_with_dense(set_seed):
    keras_model = KerasConvolutional.build_conv_with_dense()
    template_test_keras_functional(keras_model, "last")


def test_simple_conv(set_seed):
    keras_model = KerasConvolutional.build_simple_conv()
    template_test_keras_functional(keras_model, "last")


def test_deep_conv(set_seed):
    keras_model = KerasConvolutional.build_deep_conv()
    template_test_keras_functional(keras_model, "last")


def test_conv_multi_output(set_seed):
    keras_model = KerasConvolutional.build_conv_multi_output()
    template_test_keras_functional(keras_model, "last")


def test_conv_dropout(set_seed):
    keras_model = KerasConvolutional.build_conv_with_dropout()
    with pytest.raises(NotImplementedError):
        template_test_keras_functional(keras_model, "last")


def test_channels_first(set_seed):
    keras_model = tf.keras.Sequential([
        tf.keras.Input((2, 4)),
        tf.keras.layers.Conv1D(2, 3, activation=None, padding="same", data_format="channels_first"),
    ])
    template_test_keras_sequential(keras_model, "first")


@pytest.mark.parametrize(
    "in_c, kernel_size, padding, stride",
    [
        (1, 1, "valid", 1),
        (3, 5, "same", 1),
        (16, 6, "same", 2),
    ]
)
def test_various_cnns_sequential(in_c, kernel_size, padding, stride):
    keras_model = tf.keras.Sequential([
        tf.keras.Input((12, in_c), name="inputs"),
        tf.keras.layers.Conv1D(5, kernel_size, padding=padding, strides=stride),
    ])
    template_test_keras_sequential(keras_model, "last")


@pytest.mark.parametrize(
    "in_c, kernel_size, padding, stride",
    [
        (1, 1, "valid", 1),
        (3, 5, "same", 1),
        (16, 6, "same", 2),
    ]
)
def test_various_cnns_functional(in_c, kernel_size, padding, stride):
    inputs = tf.keras.Input((12, in_c), name="inputs")
    x = tf.keras.layers.Conv1D(5, kernel_size, padding=padding, strides=stride)(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=x)
    template_test_keras_functional(keras_model, "last")
