import numpy as np
import nevergrad.common.typing as tp
from typing import Optional, Tuple, Any, List
from . import utils
from .base import registry
from . import callbacks

class MetaModelFailure(ValueError):
    """Sometimes the optimum of the metamodel is at infinity."""

def learn_on_k_best(
    archive: utils.Archive[utils.Value],
    k: int,
    algorithm: str = 'quad',
    degree: int = 2,
    shape: Optional[Tuple[int, ...]] = None,
    para: Optional[Any] = None
) -> np.ndarray:
    """Approximate optimum learnt from the k best.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
    """
    items: List[Tuple[np.ndarray, utils.Value]] = list(archive.items_as_arrays())
    dimension: int = len(items[0][0])
    if algorithm == 'image':
        k = len(archive) // 6
    first_k_individuals: List[Tuple[np.ndarray, utils.Value]] = sorted(
        items,
        key=lambda indiv: archive[indiv[0]].get_estimation('average')
    )[:k]
    if algorithm == 'image':
        assert para is not None
        new_first_k_individuals: List[np.ndarray] = []
        for i in first_k_individuals:
            new_child = para.spawn_child()
            new_child.set_standardized_data(i[0])
            new_first_k_individuals += [new_child.value.flatten()]
    else:
        new_first_k_individuals = [indiv for indiv in first_k_individuals]
    assert len(first_k_individuals) == k
    middle: np.ndarray = np.array(sum((p[0] for p in first_k_individuals)) / k)
    normalization: float = 1e-15 + np.sqrt(np.sum((first_k_individuals[-1][0] - first_k_individuals[0][0]) ** 2))
    if 'image' == algorithm:
        middle = 0.0 * middle
        normalization = 1.0
    y: np.ndarray = np.asarray([archive[c[0]].get_estimation('pessimistic') for c in first_k_individuals])
    if algorithm == 'image':
        X: np.ndarray = np.asarray([(c - middle) / normalization for c in new_first_k_individuals])
    else:
        X: np.ndarray = np.asarray([(c[0] - middle) / normalization for c in new_first_k_individuals])
    from sklearn.preprocessing import PolynomialFeatures
    polynomial_features = PolynomialFeatures(degree=degree)

    def trans(X_input: np.ndarray) -> np.ndarray:
        if degree > 1:
            return polynomial_features.fit_transform(X_input)
        return X_input

    X2: np.ndarray = trans(X)
    if not max(y) - min(y) > 1e-20:
        raise MetaModelFailure
    y = (y - min(y)) / (max(y) - min(y))
    if algorithm == 'neural':
        from sklearn.neural_network import MLPRegressor
        model: Any = MLPRegressor(hidden_layer_sizes=(16, 16), solver='lbfgs')
    elif algorithm in ['image']:
        from sklearn.svm import SVR
        import scipy.ndimage as ndimage

        def rephrase(x: np.ndarray) -> np.ndarray:
            if shape is None:
                return x
            radii = [1 + int(0.3 * np.sqrt(shape[i])) for i in range(len(shape))]
            newx = ndimage.convolve(x.reshape(shape), np.ones(radii) / np.prod(radii))
            return newx

        def my_kernel(x: np.ndarray, y_: np.ndarray) -> float:
            k = 0.0
            for i in range(len(x)):
                k += np.exp(-50.0 * np.sum((rephrase(x[i]) - rephrase(y_[i])) ** 2) / len(x[0]))
            return k

        model = SVR(kernel=my_kernel, C=10000.0, tol=0.001)
    elif algorithm in ['svm', 'svr']:
        from sklearn.svm import SVR
        model: Any = SVR()
    elif algorithm == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model: Any = RandomForestRegressor()
    else:
        assert algorithm == 'quad', f'Metamodelling algorithm {algorithm} not recognized.'
        from sklearn.linear_model import LinearRegression
        model: Any = LinearRegression()
    model.fit(X2, y)
    model_outputs: np.ndarray = model.predict(X2)
    indices: np.ndarray = np.argsort(y)
    ordered_model_outputs: List[float] = [model_outputs[i] for i in indices]
    success_rate: float = np.average(0.5 + 0.5 * np.sign(np.diff(ordered_model_outputs)))
    if not np.all(np.diff(ordered_model_outputs) > 0) and 'image' != algorithm:
        raise MetaModelFailure('Unlearnable objective function.')
    if np.average(0.5 + 0.5 * np.sign(np.diff(ordered_model_outputs))) < 0.6:
        raise MetaModelFailure('Unlearnable objective function.')

    try:
        Powell = registry['Powell']
        DE = registry['DE']
        DiscreteLenglerOnePlusOne = registry['DiscreteLenglerOnePlusOne']

        def loss_function_sm(x: tp.ArrayLike) -> float:
            return float(model.predict(trans(np.asarray(x, dtype=X[0].dtype).flatten()[None, :])))
        
        optimizers = (Powell, DE) if algorithm != 'image' else (DiscreteLenglerOnePlusOne,)
        for cls in optimizers:
            optimizer = cls(
                parametrization=para if para is not None and algorithm == 'image' else dimension,
                budget=45 * dimension + 30
            )
            optimizer.register_callback('ask', callbacks.EarlyStopping.timer(20))
            if 'image' in algorithm:
                optimizer.suggest(X2[0].reshape(shape))
                optimizer.suggest(X2[1].reshape(shape))
                optimizer.suggest(X2[2].reshape(shape))
            try:
                minimum_point = optimizer.minimize(loss_function_sm)
                minimum = minimum_point.value
            except RuntimeError:
                assert cls == Powell, 'Only Powell is allowed to crash here.'
            else:
                break
    except ValueError as e:
        raise MetaModelFailure(f'Infinite meta-model optimum in learn_on_k_best: {e}.')
    if loss_function_sm(minimum) > y[len(y) // 3] and algorithm == 'image' and (success_rate < 0.9):
        raise MetaModelFailure('Not a good proposal.')
    if loss_function_sm(minimum) > y[0] and algorithm != 'image':
        raise MetaModelFailure('Not a good proposal.')
    if algorithm == 'image':
        minimum = minimum_point.get_standardized_data(reference=para)
    if np.sum(minimum ** 2) > 1.0 and algorithm != 'image':
        raise MetaModelFailure('huge meta-model optimum in learn_on_k_best.')
    return middle + normalization * minimum.flatten()
