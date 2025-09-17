import shutil

from testennuf import TMPDIR
from testennuf.example_models.pytorch_convolutional import PytorchConvolutional
from testennuf.example_models.simple_mlp_pytorch import SimpleMLP, LessSimpleMLP

from testennuf.kgo.template import template_test_kgo


def template_test_torch_sequential(torch_model, input_size: tuple[int, ...]):
    from ennuf.pytorch import from_sequential

    model = from_sequential(torch_model, input_size)
    dir_ = TMPDIR.joinpath("torch", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    torch_model.eval()
    template_test_kgo(model, dir_, torch_model, "pytorch")
    shutil.rmtree(dir_)


def test_pytorch_sequential_1():
    torch_model = SimpleMLP.build_sequential_simple()
    template_test_torch_sequential(torch_model, (1,))


def test_pytorch_sequential_2():
    torch_model = LessSimpleMLP.build_sequential()
    template_test_torch_sequential(torch_model, (2,))


def test_only_conv():
    torch_model = PytorchConvolutional.build_only_conv()
    template_test_torch_sequential(torch_model, (3, 4))

def test_simple_conv():
    torch_model = PytorchConvolutional.build_simple_conv()
    template_test_torch_sequential(torch_model, (3, 16))


def test_deep_conv():
    torch_model = PytorchConvolutional.build_deep_conv()
    template_test_torch_sequential(torch_model, (3, 32))


def test_dropout_conv():
    torch_model = PytorchConvolutional.build_conv_with_dropout()
    template_test_torch_sequential(torch_model, (1, 32))
