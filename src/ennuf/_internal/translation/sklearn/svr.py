#  (C) Crown Copyright, Met Office, 2025.
from sklearn.svm import SVR

from ennuf._internal.ml_model.base_layer import BaseLayer
from ennuf._internal.ml_model.layers.concatenate import Concatenate
from ennuf._internal.ml_model.layers.svr import SVR_ENNUF
from ennuf._internal.ml_model.layers.input_layer import InputLayer
import ennuf._internal.ml_model.model as model
from ennuf._internal.ml_model.supported_activations import SupportedActivations
import ennuf._internal.ml_model.model as ennufmodel
import warnings
import re


def from_svr(
    sklearn_model:SVR,
    name: str="placeholder",
    long_name: str = "Auto-generated module by ENNUF",
    formatter = None,
    ignore_warning = False,
) -> ennufmodel.Model:
    if not ignore_warning:
        raise NotImplementedError("SVRs are currently unsupported and still in development.")
    layer_names=["input","svr"]
    dtype=sklearn_model.dual_coef_.dtype
    ennuf_model = ennufmodel.Model(
        name=name,
        output_names=[layer_names[-1]],
        description=long_name,
        dtype=dtype,
        formatter=formatter
    )
    input_layer=InputLayer(name="input", has_channels=True, shape=sklearn_model.support_vectors_.shape[1], parent_model=ennuf_model)
    ennuf_model.layers.append(input_layer)
    svr_layer=SVR_ENNUF(
        name="svr",
        parent_model=ennuf_model,
        inputs=input_layer,
        dual_coef=sklearn_model.dual_coef_,
        support_vectors=sklearn_model.support_vectors_,
        intercept=sklearn_model.intercept_
    )
    ennuf_model.layers.append(svr_layer)
    return ennuf_model


# import matplotlib.pyplot as plt
# import numpy as np
# import random
# import sklearn.datasets
# from sklearn.svm import SVR
# X, y =sklearn.datasets.make_regression(n_samples=100, n_features=2, n_informative=2)
# svr_lin = SVR(kernel="linear", C=100, gamma="auto")
# random.seed(10)
# trained_model=svr_lin.fit(X,y)
# print(from_svr(trained_model))