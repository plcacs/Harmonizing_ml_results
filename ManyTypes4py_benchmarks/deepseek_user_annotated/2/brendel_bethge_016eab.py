# mypy: allow-untyped-defs, no-strict-optional

from typing import Union, Optional, Tuple, Any, List, Dict, Callable, TypeVar, cast
from typing_extensions import Literal
from abc import ABC, abstractmethod
import numpy as np
import eagerpy as ep
import logging
import warnings
from ..devutils import flatten
from . import LinearSearchBlendedUniformNoiseAttack
from ..tensorboard import TensorBoard
from .base import Model
from .base import MinimizationAttack
from .base import get_is_adversarial
from .base import get_criterion
from .base import T
from ..criteria import Misclassification, TargetedMisclassification
from .base import raise_if_kwargs
from .base import verify_input_bounds
from ..distances import l0, l1, l2, linf


try:
    from numba.experimental import jitclass  # type: ignore
    import numba
    from numba.core.types import ClassType  # type: ignore
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    # delay the error until the attack is initialized
    NUMBA_IMPORT_ERROR = e

    def jitclass(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(c: Any) -> Any:
            return c

        return decorator

else:
    NUMBA_IMPORT_ERROR = None

EPS = 1e-10


class Optimizer(object):  # pragma: no cover
    """Base class for the trust-region optimization."""

    def __init__(self) -> None:
        self.bfgsb: BFGSB = BFGSB()  # a box-constrained BFGS solver

    def solve(self, x0: np.ndarray, x: np.ndarray, b: np.ndarray, min_: float, max_: float, c: float, r: float) -> np.ndarray:
        x0, x, b = x0.astype(np.float64), x.astype(np.float64), b.astype(np.float64)
        cmax, cmaxnorm = self._max_logit_diff(x, b, min_, max_, c)

        if np.abs(cmax) < np.abs(c):
            if np.sqrt(cmaxnorm) < r:
                _delta = self.optimize_boundary_s_t_trustregion(
                    x0, x, b, min_, max_, c, r
                )
            else:
                _delta = self.optimize_boundary_s_t_trustregion(
                    x0, x, b, min_, max_, c, r
                )
        else:
            if cmaxnorm < r:
                _delta = self.optimize_distance_s_t_boundary_and_trustregion(
                    x0, x, b, min_, max_, c, r
                )
            else:
                bnorm = np.linalg.norm(b)
                minnorm = self._minimum_norm_to_boundary(x, b, min_, max_, c, bnorm)

                if minnorm <= r:
                    _delta = self.optimize_distance_s_t_boundary_and_trustregion(
                        x0, x, b, min_, max_, c, r
                    )
                else:
                    _delta = self.optimize_boundary_s_t_trustregion(
                        x0, x, b, min_, max_, c, r
                    )

        return _delta

    def _max_logit_diff(self, x: np.ndarray, b: np.ndarray, _ell: float, _u: float, c: float) -> Tuple[float, float]:
        N = x.shape[0]
        cmax = 0.0
        norm = 0.0

        if c > 0:
            for n in range(N):
                if b[n] > 0:
                    cmax += b[n] * (_u - x[n])
                    norm += (_u - x[n]) ** 2
                else:
                    cmax += b[n] * (_ell - x[n])
                    norm += (x[n] - _ell) ** 2
        else:
            for n in range(N):
                if b[n] > 0:
                    cmax += b[n] * (_ell - x[n])
                    norm += (x[n] - _ell) ** 2
                else:
                    cmax += b[n] * (_u - x[n])
                    norm += (_u - x[n]) ** 2

        return cmax, np.sqrt(norm)

    def _minimum_norm_to_boundary(self, x: np.ndarray, b: np.ndarray, _ell: float, _u: float, c: float, bnorm: float) -> float:
        N = x.shape[0]

        lambda_lower = 2 * c / (bnorm**2 + EPS)
        lambda_upper = np.sign(c) * np.inf
        _lambda = lambda_lower
        k = 0

        while True:
            k += 1
            _c = 0
            norm = 0

            if c > 0:
                for n in range(N):
                    lam_step = _lambda * b[n] / 2
                    if b[n] > 0:
                        max_step = _u - x[n]
                        delta_step = min(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step**2
                    else:
                        max_step = _ell - x[n]
                        delta_step = max(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step**2
            else:
                for n in range(N):
                    lam_step = _lambda * b[n] / 2
                    if b[n] > 0:
                        max_step = _ell - x[n]
                        delta_step = max(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step**2
                    else:
                        max_step = _u - x[n]
                        delta_step = min(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step**2

            if np.abs(_c) < np.abs(c):
                if np.isinf(lambda_upper):
                    _lambda *= 2
                else:
                    lambda_lower = _lambda
                    _lambda = (lambda_upper - lambda_lower) / 2 + lambda_lower
            else:
                lambda_upper = _lambda
                _lambda = (lambda_upper - lambda_lower) / 2 + lambda_lower

            if 0.999 * np.abs(c) - EPS < np.abs(_c) < 1.001 * np.abs(c) + EPS:
                break

        return np.sqrt(norm)

    def optimize_distance_s_t_boundary_and_trustregion(
        self, x0: np.ndarray, x: np.ndarray, b: np.ndarray, min_: float, max_: float, c: float, r: float
    ) -> np.ndarray:
        params0 = np.array([0.0, 0.0])
        bounds = np.array([(-np.inf, np.inf), (0, np.inf)])
        args = (x0, x, b, min_, max_, c, r)

        qk = self.bfgsb.solve(self.fun_and_jac, params0, bounds, args)
        return self._get_final_delta(
            qk[0], qk[1], x0, x, b, min_, max_, c, r, touchup=True
        )

    @abstractmethod
    def fun_and_jac(self, params: np.ndarray, x0: np.ndarray, x: np.ndarray, b: np.ndarray, min_: float, max_: float, c: float, r: float) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _get_final_delta(self, lam: float, mu: float, x0: np.ndarray, x: np.ndarray, b: np.ndarray, min_: float, max_: float, c: float, r: float, touchup: bool = True) -> np.ndarray:
        raise NotImplementedError

    def optimize_boundary_s_t_trustregion_fun_and_jac(
        self, params: np.ndarray, x0: np.ndarray, x: np.ndarray, b: np.ndarray, min_: float, max_: float, c: float, r: float
    ) -> Tuple[float, np.ndarray]:
        N = x0.shape[0]
        s = -np.sign(c)
        _mu = params[0]
        t = 1 / (2 * _mu + EPS)

        g = -_mu * r**2
        grad_mu = -(r**2)

        for n in range(N):
            d = -s * b[n] * t

            if d < min_ - x[n]:
                d = min_ - x[n]
            elif d > max_ - x[n]:
                d = max_ - x[n]
            else:
                grad_mu += (b[n] + 2 * _mu * d) * (b[n] / (2 * _mu**2 + EPS))

            grad_mu += d**2
            g += (b[n] + _mu * d) * d

        return -g, -np.array([grad_mu])

    def safe_div(self, nominator: float, denominator: float) -> float:
        if np.abs(denominator) > EPS:
            return nominator / denominator
        elif denominator >= 0:
            return nominator / EPS
        else:
            return -nominator / EPS

    def optimize_boundary_s_t_trustregion(self, x0: np.ndarray, x: np.ndarray, b: np.ndarray, min_: float, max_: float, c: float, r: float) -> np.ndarray:
        params0 = np.array([1.0])
        args = (x0, x, b, min_, max_, c, r)
        bounds = np.array([(0, np.inf)])

        qk = self.bfgsb.solve(
            self.optimize_boundary_s_t_trustregion_fun_and_jac, params0, bounds, args
        )

        _delta = self.safe_div(-b, 2 * qk[0])

        for n in range(x0.shape[0]):
            if _delta[n] < min_ - x[n]:
                _delta[n] = min_ - x[n]
            elif _delta[n] > max_ - x[n]:
                _delta[n] = max_ - x[n]

        return _delta


class BrendelBethgeAttack(MinimizationAttack, ABC):
    def __init__(
        self,
        init_attack: Optional[MinimizationAttack] = None,
        overshoot: float = 1.1,
        steps: int = 1000,
        lr: float = 1e-3,
        lr_decay: float = 0.5,
        lr_num_decay: int = 20,
        momentum: float = 0.8,
        tensorboard: Union[Literal[False], None, str] = False,
        binary_search_steps: int = 10,
    ) -> None:

        if NUMBA_IMPORT_ERROR is not None:
            raise NUMBA_IMPORT_ERROR  # pragma: no cover

        if "0.49." in numba.__version__:
            warnings.warn(
                "There are known issues with numba version 0.49 and we suggest using numba 0.50 or newer."
            )

        self.init_attack = init_attack
        self.overshoot = overshoot
        self.steps = steps
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_num_decay = lr_num_decay
        self.momentum = momentum
        self.tensorboard = tensorboard
        self.binary_search_steps = binary_search_steps

        self._optimizer: Optimizer = self.instantiate_optimizer()

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[TargetedMisclassification, Misclassification, T],
        *,
        starting_points: Optional[ep.Tensor] = None,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        del kwargs

        tb = TensorBoard(logdir=self.tensorboard)

        originals, restore_type = ep.astensor_(inputs)
        del inputs

        verify_input_bounds(originals, model)

        criterion_ = get_criterion(criterion)
        del criterion
        is_adversarial = get_is_adversarial(criterion_, model)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack()
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            starting_points = init_attack.run(model, originals, criterion_)

        best_advs = ep.astensor(starting_points)
        assert is_adversarial(best_advs).all()

        N = len(originals)
        rows = range(N)
        bounds = model.bounds
        min_, max_ = bounds

        x0 = originals
        x0_np_flatten = x0.numpy().reshape((N, -1))
        x1 = best_advs

        lower_bound = ep.zeros(x0, shape=(N,))
        upper_bound = ep.ones(x0, shape=(N,))

        for _ in range(self.binary_search_steps):
            epsilons = (lower_bound + upper_bound) / 2
            mid_points = self.mid_points(x0, x1, epsilons, bounds)
            is_advs = is_adversarial(mid_points)
            lower_bound = ep.where(is_advs, lower_bound, epsilons)
            upper_bound = ep.where(is_advs, epsilons, upper_bound)

        starting_points = self.mid_points(x0, x1, upper_bound, bounds)

        tb.scalar("batchsize", N, 0)

        def loss_fun(x: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
            logits = model(x)

            if targeted:
                c_minimize = best_other_classes(logits, classes)
                c_maximize = classes
            else:
                c_minimize = classes
                c_maximize = best_other_classes(logits, classes)

            logits_diffs = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert logits_diffs.shape == (N,)

            return logits_diffs.sum(), logits_diffs

        value_and_grad = ep.value_and_grad_fn(x0, loss_fun, has_aux=True)

        def logits_diff_and_grads(x: ep.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            _, logits_diffs, boundary = value_and_grad(x)
            return logits_diffs.numpy(), boundary.numpy().copy()

        x = starting_points
        lrs = self.lr * np.ones(N)
        lr_reduction_interval = max(1, int(self.steps / self.lr_num_decay))
        converged = np.zeros(N, dtype=np.bool_)
        rate_normalization = np.prod(x.shape) * (max_ - min_)
        original_shape = x.shape
        _best_advs = best_advs.numpy()

        for step in range(1, self.steps + 1):
            if converged.all():
                break  # pragma: no cover

            logits_diffs, _boundary = logits_diff_and_grads(x)

            distances = self.norms(originals - x)
            source_norms = self.norms(originals - best_advs)

            closer = distances < source_norms
            is_advs = logits_diffs < 0
            closer = closer.logical_and(ep.from_numpy(x, is_advs))

            x_np_flatten = x.numpy().reshape((N, -1))

            if closer.any():
                _best_advs = best_advs.numpy().copy()
                _closer = closer.numpy().flatten()
                for idx in np.arange(N)[_closer]:
                    _best_advs[idx] = x_np_flatten[idx].reshape(original_shape[1:])

            best_advs = ep.from_numpy(x, _best_advs)

            if step == 1:
                boundary = _boundary
            else:
                boundary = (1 - self.momentum) * _boundary + self.momentum * boundary

            if (step + 1) % lr_reduction_interval == 0:
                lrs *= self.lr_decay

            x = x.reshape((N, -1))
            region = lrs * rate_normalization

            corr_logits_diffs = np.where(
                -logits_diffs < 0,
                -self.overshoot * logits_diffs,
                -(2 - self.overshoot) * logits_diffs,
            )

            deltas, k = [], 0

            for sample in range(N):
                if converged[sample]:
                    deltas.append(
                        np.zeros_like(x0_np_flatten[sample])
                    )  # pragma: no cover
                else:
                    _x0 = x0_np_flatten[sample]
                    _x = x_np_flatten[sample]
                    _b = boundary[k].flatten()
                    _c = corr_logits_diffs[k]
                    r = region[sample]

                    delta = self._optimizer.solve(  # type: ignore
                        _x0, _x, _b, bounds[0], bounds[1], _c, r
                    )
                    deltas.append(delta)

                    k += 1  # idx of masked