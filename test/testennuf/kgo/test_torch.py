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
