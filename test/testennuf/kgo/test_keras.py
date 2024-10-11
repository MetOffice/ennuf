#  (C) Crown Copyright, Met Office, 2024.

import shutil

import pytest

from testennuf import TMPDIR, WEIGHTS_DIR
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
    try:
        keras_model.load_weights(weights_dir)
    except ValueError as e:
        if not ("File format not supported" in str(e)):
            raise e
        print("File format not supported, skipping test")
    inputs = np.asarray([[1.0, 0.12, 0.02, 0.64, 0.9, 0.91]])
    kgo = keras_model.predict(inputs)

    def get_covariance_matrix(diags):
        nlev = 70
        cov = np.zeros((nlev, nlev))
        for i in range(nlev):
            cov[i, i] = diags[i]
        for i in range(nlev - 1):
            cov[i, i + 1] = diags[nlev + i]
            cov[i + 1, i] = diags[nlev + i]
        for i in range(nlev - 2):
            cov[i, i + 2] = diags[nlev + nlev - 1 + i]
            cov[i + 2, i] = diags[nlev + nlev - 1 + i]
        return cov

    def get_covariance_matrix_alternative(diags):
        def alpha(i, j):
            if j < 2:
                return 1
            else:
                return 0

        nlev = 70
        cov = np.zeros((nlev, nlev))
        for i in range(nlev):
            cov[i, i] = diags[i]
        for j in range(nlev - 1):
            for i in range(nlev - j):
                cov[i, i + j] = alpha(i, j) * diags[i] * diags[i + j]
                cov[i + j, i] = alpha(i, j) * diags[i] * diags[i + j]
        return cov

    A = get_covariance_matrix(kgo[0])

    # FIXME: A should be positive semidefinite but isn't
    # model = ennuf.keras.from_functional(keras_model, name=name.lower(), long_name="T Anomaly Covariances 01")
    # model.formatter = MinimalistFormatter()
    # def check_cholesky_matrix_gives_positive_definite_covariance_matrix(L):
    #     if np.isclose(np.linalg.det(L), 0):
    #         raise ValueError('determinant of L may not be zero')
    #     A = np.matmul(L, L.T)
    #     rng.multivariate_normal(np.zeros(L.shape[0]), cov=A, check_valid='raise')
    #     print('all fine')
    #
    # model.create_fortran_module("/data/users/hreid/src/ENNUF/ref/tac001v001/tac01v001_mod.f90")
    template_test_keras_functional(keras_model)
