import shutil

import numpy as np
import pytest
from testennuf import TMPDIR, RANDOM_SEED
from testennuf.example_models.pytorch_convolutional import PytorchConvolutional
from testennuf.example_models.pytorch_simple_mlp import SimpleMLP, LessSimpleMLP

from testennuf.kgo.template import template_test_kgo
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def template_test_torch_sequential(torch_model, input_size: tuple[int, ...], input_has_channels=True):
    from ennuf.pytorch import from_sequential

    model = from_sequential(torch_model, input_size, input_layers_have_channels=input_has_channels, dtype=np.float32)
    dir_ = TMPDIR.joinpath("torch", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    torch_model.eval()
    template_test_kgo(model, dir_, torch_model, "pytorch")
    shutil.rmtree(dir_)


def test_pytorch_sequential_1():
    torch_model = SimpleMLP.build_sequential_simple()
    template_test_torch_sequential(torch_model, (1,), input_has_channels=False)


def test_pytorch_sequential_2():
    torch_model = LessSimpleMLP.build_sequential()
    template_test_torch_sequential(torch_model, (2,), input_has_channels=False)


def test_only_flatten():
    torch_model = PytorchConvolutional.only_flatten()
    template_test_torch_sequential(torch_model, (3, 4))


def test_only_conv():
    torch_model = PytorchConvolutional.build_only_conv()
    template_test_torch_sequential(torch_model, (3, 4))


def test_only_conv_no_bias():
    torch_model = PytorchConvolutional.build_only_conv_no_bias()
    with pytest.raises(NotImplementedError):
        template_test_torch_sequential(torch_model, (3, 4))


def test_simple_conv():
    torch_model = PytorchConvolutional.build_simple_conv()
    template_test_torch_sequential(torch_model, (3, 4))


def test_deep_conv():
    torch_model = PytorchConvolutional.build_deep_conv()
    template_test_torch_sequential(torch_model, (3, 32))


def test_dropout_conv():
    torch_model = PytorchConvolutional.build_conv_with_dropout()
    with pytest.raises(NotImplementedError):
        template_test_torch_sequential(torch_model, (1, 32))


@pytest.mark.parametrize(
    "in_c, kernel_size, padding, stride",
    [
        (1, 1, 0, 1),
        (3, 4, 1, 1),
        (16, 6, 5, 2),
    ]
)
def test_various_cnns(in_c, kernel_size, padding, stride):
    torch_model = nn.Sequential(
        nn.Conv1d(in_c, 12, kernel_size, padding=padding, padding_mode="reflect", stride=stride)
    )
    template_test_torch_sequential(torch_model, input_size=(in_c, 10))


def test_conv_followed_by_dense():
    torch_model = nn.Sequential(
        nn.Conv1d(3, 12, 4, dilation=5),
        nn.Linear(5, 7),
        nn.LeakyReLU(0.7),
    )
    template_test_torch_sequential(torch_model, input_size=(3, 20))


def test_complicated_cnn():
    torch_model = nn.Sequential(
        nn.Conv1d(3, 12, 4, dilation=3),
        nn.Sigmoid(),
        nn.AvgPool1d(2, stride=1),
        nn.Conv1d(12, 14, 5, padding=4, padding_mode="reflect"),
        nn.MaxPool1d(4, padding=2, stride=7),
        nn.Tanh(),
        nn.Linear(3, 23),
        nn.Flatten(0),
        nn.LeakyReLU(0.2),
        nn.Linear(322, 4),
    )
    template_test_torch_sequential(torch_model, input_size=(3, 20))

def test_manual():
    from ennuf.ml_model import (
        Activation, Concatenate, Conv1d, Dense, Flatten, InputLayer,
        Model, Pooling1d, SupportedActivations, PaddingMode
    )
    rng = np.random.default_rng(RANDOM_SEED)
    # fetch the weights for each of your model's layers. In this example we'll randomly generate some,
    # but you'll want to get yours from your trained network.
    conv_weights = rng.random((3,1))
    conv_biases = rng.random((3,))
    dense_weights = rng.random((4,4))
    dense_biases = rng.random(4)

    ennuf_model = Model(
        name="my_first_ennuf_nn",
        output_names=["", ],
        description="A model for demonstrating how to use the ENNUF manual API",
        dtype=np.float32,
    )
    ennuf_model.layers = [
        input_1 := InputLayer((4,), name="input_1", parent_model=ennuf_model),
        input_2 := InputLayer((3, 20,), name="input_2", parent_model=ennuf_model),
        conv_1 := Conv1d(
            name="conv_1",
            inputs=input_2,
            parent_model=ennuf_model,
            weights=conv_weights,
            biases=conv_biases,
            padding_mode=PaddingMode.ZEROS,
            padding=1,
            stride=1,
            dilation=1,
        ),
        pool_1 := Pooling1d(
            name="pool_1",
            inputs=conv_1,
            parent_model=ennuf_model,
            pool_size=2,
            type_of_pooling="AVG",
            padding=0,
            stride=1,
        ),
        tanh := Activation(name="tanh", shape=(3,4), inputs=pool_1, parent_model=ennuf_model, activation=SupportedActivations.from_identifier("tanh")),
        flattener := Flatten(name="flattener", inputs=tanh, parent_model=ennuf_model),
        concat_1 := Concatenate(name="concat_1", shape=9, inputs=[input_1, flattener], axis=0, parent_model=ennuf_model),
        dense_1 := Dense(
            name="dense_1",
            inputs=concat_1,
            parent_model=ennuf_model,
            shape=9,
            weights=dense_weights,
            biases=dense_biases,
        ),
        leaky_relu := Activation(name="leaky_relu", shape=9, inputs=dense_1, parent_model=ennuf_model, activation=SupportedActivations.ids()["LeakyReLU"](0.2)),
    ]
