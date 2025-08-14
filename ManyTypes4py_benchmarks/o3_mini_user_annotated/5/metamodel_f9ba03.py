#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad.common.typing as tp
from . import utils
from .base import registry
from . import callbacks


class MetaModelFailure(ValueError):
    """Sometimes the optimum of the metamodel is at infinity."""


def learn_on_k_best(
    archive: utils.Archive[utils.MultiValue],
    k: int,
    algorithm: str = "quad",
    degree: int = 2,
    shape: tp.Any = None,
    para: tp.Any = None,
) -> tp.ArrayLike:
    """Approximate optimum learnt from the k best.

    Parameters
    ----------
    archive : utils.Archive[utils.MultiValue]
        The archive containing the values to learn from.
    k : int
        Number of best individuals to use for learning.
    algorithm : str, optional
        The algorithm used for meta-modelling, by default "quad".
    degree : int, optional
        Polynomial degree for feature transformation, by default 2.
    shape : tp.Any, optional
        Shape for "image" algorithm, by default None.
    para : tp.Any, optional
        Parameter object used when algorithm is "image", by default None.

    Returns
    -------
    tp.ArrayLike
        The estimated optimum.
    """
    items = list(archive.items_as_arrays())
    dimension: int = len(items[0][0])
    if algorithm == "image":
        k = len(archive) // 6
    # Select the k best.
    first_k_individuals = sorted(
        items, key=lambda indiv: archive[indiv[0]].get_estimation("average")
    )[:k]
    if algorithm == "image":
        assert para is not None
        new_first_k_individuals: list[np.ndarray] = []
        for i in first_k_individuals:
            new_child = para.spawn_child()
            new_child.set_standardized_data(i[0])
            new_first_k_individuals += [new_child.value.flatten()]
    else:
        new_first_k_individuals = first_k_individuals
    assert len(first_k_individuals) == k

    # Recenter the best.
    middle = np.array(sum(p[0] for p in first_k_individuals) / k)
    normalization: float = 1e-15 + np.sqrt(
        np.sum((first_k_individuals[-1][0] - first_k_individuals[0][0]) ** 2)
    )
    if algorithm == "image":
        middle = 0.0 * middle
        normalization = 1.0

    y: np.ndarray = np.asarray(
        [archive[c[0]].get_estimation("pessimistic") for c in first_k_individuals]
    )
    if algorithm == "image":
        X: np.ndarray = np.asarray(
            [(c - middle) / normalization for c in new_first_k_individuals]
        )
    else:
        X = np.asarray(
            [(c[0] - middle) / normalization for c in new_first_k_individuals]
        )

    from sklearn.preprocessing import PolynomialFeatures

    polynomial_features = PolynomialFeatures(degree=degree)

    def trans(X_in: np.ndarray) -> np.ndarray:
        if degree > 1:
            return polynomial_features.fit_transform(X_in)
        return X_in

    X2: np.ndarray = trans(X)
    if not max(y) - min(y) > 1e-20:  # better use "not" for dealing with nans
        raise MetaModelFailure
    y = (y - min(y)) / (max(y) - min(y))
    if algorithm == "neural":
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(hidden_layer_sizes=(16, 16), solver="lbfgs")
    elif algorithm in ["image"]:
        from sklearn.svm import SVR
        import scipy.ndimage as ndimage

        def rephrase(x: np.ndarray) -> np.ndarray:
            if shape is None:
                return x
            radii: list[int] = [1 + int(0.3 * np.sqrt(shape[i])) for i in range(len(shape))]
            newx = ndimage.convolve(x.reshape(shape), np.ones(radii) / np.prod(radii))
            return newx

        def my_kernel(x: np.ndarray, y_in: np.ndarray) -> np.ndarray:
            k_mat = np.zeros((len(x), len(y_in)))
            for i in range(len(x)):
                for j in range(len(y_in)):
                    diff = rephrase(x[i]) - rephrase(y_in[j])
                    k_mat[i][j] = np.exp(-50.0 * np.sum(diff ** 2) / (len(x[0])))
            return k_mat

        model = SVR(kernel=my_kernel, C=1e4, tol=1e-3)
    elif algorithm in ["svm", "svr"]:
        from sklearn.svm import SVR

        model = SVR()
    elif algorithm == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()
    else:
        assert algorithm == "quad", f"Metamodelling algorithm {algorithm} not recognized."
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()

    model.fit(X2, y)
    model_outputs: np.ndarray = model.predict(X2)
    indices = np.argsort(y)
    ordered_model_outputs = [model_outputs[i] for i in indices]
    success_rate = np.average(0.5 + 0.5 * np.sign(np.diff(ordered_model_outputs)))
    if not np.all(np.diff(ordered_model_outputs) > 0) and algorithm != "image":
        raise MetaModelFailure("Unlearnable objective function.")
    if success_rate < 0.6:
        raise MetaModelFailure("Unlearnable objective function.")

    def loss_function_sm(x: np.ndarray) -> float:
        x_array = np.asarray(x, dtype=X[0].dtype).flatten()[None, :]
        return float(model.predict(trans(x_array)))

    try:
        Powell = registry["Powell"]
        DE = registry["DE"]
        DiscreteLenglerOnePlusOne = registry["DiscreteLenglerOnePlusOne"]

        optimizer_classes: tuple = (Powell, DE) if algorithm != "image" else (DiscreteLenglerOnePlusOne,)
        for cls in optimizer_classes:
            optimizer = cls(
                parametrization=para if (para is not None and algorithm == "image") else dimension,
                budget=45 * dimension + 30,
            )
            optimizer.register_callback("ask", callbacks.EarlyStopping.timer(20))
            if "image" in algorithm:
                optimizer.suggest(X2[0].reshape(shape))
                optimizer.suggest(X2[1].reshape(shape))
                optimizer.suggest(X2[2].reshape(shape))
            try:
                minimum_point = optimizer.minimize(loss_function_sm)
                minimum = minimum_point.value
            except RuntimeError:
                assert cls == Powell, "Only Powell is allowed to crash here."
            else:
                break
    except ValueError as e:
        raise MetaModelFailure(f"Infinite meta-model optimum in learn_on_k_best: {e}.")
    if loss_function_sm(minimum) > y[len(y) // 3] and algorithm == "image" and success_rate < 0.9:
        raise MetaModelFailure("Not a good proposal.")
    if loss_function_sm(minimum) > y[0] and algorithm != "image":
        raise MetaModelFailure("Not a good proposal.")
    if algorithm == "image":
        minimum = minimum_point.get_standardized_data(reference=para)
    if np.sum(minimum**2) > 1.0 and algorithm != "image":
        raise MetaModelFailure("huge meta-model optimum in learn_on_k_best.")
    return middle + normalization * minimum.flatten()