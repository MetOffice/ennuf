import shutil

import pytest

# from ennuf.test.testennuf import TMPDIR, WEIGHTS_DIR
from testennuf.example_models.simple_mlp_pytorch import SimpleMLP
from testennuf.example_models.t_anomaly_covariances_01 import TAnomalyCovariances01
from testennuf.kgo.template import template_test_kgo


def template_test_torch_sequential(torch_model):
    from ennuf.pytorch import from_sequential

    model = from_sequential(torch_model)
    dir_ = TMPDIR.joinpath("torch", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, keras_model.predict)
    shutil.rmtree(dir_)

def test_pytorch_sequential_examples():
    torch_model = SimpleMLP.build_functional_easy()
    template_test_torch_functional(torch_model)

test_pytorch_sequential_examples()