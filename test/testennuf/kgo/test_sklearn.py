#  (C) Crown Copyright, Met Office, 2024.

import shutil

import pytest
from sklearn.svm import SVR
import numpy as np

from testennuf import TMPDIR
from testennuf.kgo.template import template_test_kgo

def template_test_svr(sklearn_model):
    from ennuf.sklearn import from_svr

    model = from_svr(sklearn_model)
    dir_ = TMPDIR.joinpath("sklearn", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    template_test_kgo(model, dir_, sklearn_model.predict, "sklearn")
    shutil.rmtree(dir_)

def test_svr():
    sklearn_model = SVR(kernel='rbf')
    X = np.sort(np.random.rand(40, 1),
            axis=0)
    y = np.sin(X).ravel()

    # add some noise to the data
    # y[::5] += (0.5 - np.random.rand(8))

    sklearn_model.fit(X, y)
    template_test_svr(sklearn_model)

test_svr()