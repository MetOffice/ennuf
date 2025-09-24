#  (C) Crown Copyright, Met Office, 2024.

import shutil

import pytest
from sklearn.svm import SVR
import numpy as np

from testennuf import TMPDIR, RANDOM_SEED
from testennuf.kgo.template import template_test_kgo

def template_test_svr(sklearn_model):
    from ennuf.sklearn import from_svr

    model = from_svr(sklearn_model)
    dir_ = TMPDIR.joinpath("sklearn", f"{model.name}")
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir(parents=True)
    # FIXME: Investigate causes of the slight discrepancies between the Fortran and python SVRs that necessitate
    #  higher tolerances (floating point related?)
    template_test_kgo(model, dir_, sklearn_model.predict, "sklearn", atol=1.0e-4)
    shutil.rmtree(dir_)

def test_svr():
    sklearn_model = SVR(kernel='rbf')
    rng = np.random.default_rng(seed=RANDOM_SEED)
    X = np.sort(rng.random((40, 1)),
            axis=0)
    y = np.sin(X).ravel()

    # add some noise to the data
    # y[::5] += (0.5 - rng.random(8))

    sklearn_model.fit(X, y)
    with pytest.raises(NotImplementedError):
        template_test_svr(sklearn_model)

test_svr()
