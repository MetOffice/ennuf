#  (C) Crown Copyright, Met Office, 2023.
import numpy as np
from pathlib import Path

import tensorflow as tf
import tensorflow as tf
import os.path
import shutil

import pytest

import ennuf.keras
from ennuf._internal.formatters import MinimalistFormatter
from testennuf import TMPDIR, TESTS_ROOT_DIR, WEIGHTS_DIR
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
    template_test_kgo(model, dir_, keras_model.predict)
    shutil.rmtree(dir_)


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


def test_cov_pred_sep_outputs():
    keras_model = SimpleMLP.build_covariance_predictor_separate_outputs()
    template_test_keras_functional(keras_model)
    #
    # keras_model = SimpleMLP.build_covariance_predictor()
    # template_test_keras_functional(keras_model)


def test_t_anomaly_covariances_01():
    import numpy as np

    keras_model = TAnomalyCovariances01.build_function(0.1)
    vn = 1
    name = f"TAC01v{vn:03d}"
    weights_dir = WEIGHTS_DIR.joinpath(name)
    keras_model.load_weights(weights_dir)
    inputs = np.asarray([[1.0, 0.12, 0.02, 0.64, 0.9, 0.91]])
    from ennuf import keras

    model = ennuf.keras.from_functional(keras_model, name=name.lower(), long_name="T Anomaly Covariances 01")
    model.formatter = MinimalistFormatter()
    model.create_fortran_module("/data/users/hreid/src/ENNUF/ref/tac001v001/tac01v001_mod.f90")
    template_test_keras_functional(keras_model)
