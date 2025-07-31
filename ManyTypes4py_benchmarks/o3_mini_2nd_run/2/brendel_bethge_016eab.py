#!/usr/bin/env python3
from typing import Union, Optional, Tuple, Any, Callable, List
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
    from numba.experimental import jitclass
    import numba
except (ModuleNotFoundError, ImportError) as e:
    NUMBA_IMPORT_ERROR = e

    def jitclass(*args, **kwargs):
        def decorator(c):
            return c
        return decorator
else:
    NUMBA_IMPORT_ERROR = None
EPS: float = 1e-10

class Optimizer(object):
    """Base class for the trust‐region optimization."""

    def __init__(self) -> None:
        self.bfgsb: BFGSB = BFGSB()

    def solve(self, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
              min_: float, max_: float, c: float, r: float) -> np.ndarray:
        x0, x, b = (x0.astype(np.float64), x.astype(np.float64), b.astype(np.float64))
        cmax, cmaxnorm = self._max_logit_diff(x, b, min_, max_, c)
        if np.abs(cmax) < np.abs(c):
            if np.sqrt(cmaxnorm) < r:
                _delta = self.optimize_boundary_s_t_trustregion(x0, x, b, min_, max_, c, r)
            else:
                _delta = self.optimize_boundary_s_t_trustregion(x0, x, b, min_, max_, c, r)
        elif cmaxnorm < r:
            _delta = self.optimize_distance_s_t_boundary_and_trustregion(x0, x, b, min_, max_, c, r)
        else:
            bnorm = np.linalg.norm(b)
            minnorm = self._minimum_norm_to_boundary(x, b, min_, max_, c, bnorm)
            if minnorm <= r:
                _delta = self.optimize_distance_s_t_boundary_and_trustregion(x0, x, b, min_, max_, c, r)
            else:
                _delta = self.optimize_boundary_s_t_trustregion(x0, x, b, min_, max_, c, r)
        return _delta

    def _max_logit_diff(self, x: np.ndarray, b: np.ndarray, _ell: float, _u: float,
                        c: float) -> Tuple[float, float]:
        N: int = x.shape[0]
        cmax: float = 0.0
        norm: float = 0.0
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
        return (cmax, np.sqrt(norm))

    def _minimum_norm_to_boundary(self, x: np.ndarray, b: np.ndarray, _ell: float, _u: float,
                                  c: float, bnorm: float) -> float:
        N: int = x.shape[0]
        lambda_lower: float = 2 * c / (bnorm ** 2 + EPS)
        lambda_upper: float = np.sign(c) * np.inf
        _lambda: float = lambda_lower
        k: int = 0
        while True:
            k += 1
            _c: float = 0.0
            norm: float = 0.0
            if c > 0:
                for n in range(N):
                    lam_step: float = _lambda * b[n] / 2
                    if b[n] > 0:
                        max_step: float = _u - x[n]
                        delta_step: float = min(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
                    else:
                        max_step = _ell - x[n]
                        delta_step = max(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
            else:
                for n in range(N):
                    lam_step = _lambda * b[n] / 2
                    if b[n] > 0:
                        max_step = _ell - x[n]
                        delta_step = max(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
                    else:
                        max_step = _u - x[n]
                        delta_step = min(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
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

    def optimize_distance_s_t_boundary_and_trustregion(self, x0: np.ndarray, x: np.ndarray,
                                                       b: np.ndarray, min_: float,
                                                       max_: float, c: float, r: float) -> np.ndarray:
        params0: np.ndarray = np.array([0.0, 0.0])
        bounds: np.ndarray = np.array([(-np.inf, np.inf), (0, np.inf)])
        args: Tuple[Any, ...] = (x0, x, b, min_, max_, c, r)
        qk: np.ndarray = self.bfgsb.solve(self.fun_and_jac, params0, bounds, args)
        return self._get_final_delta(qk[0], qk[1], x0, x, b, min_, max_, c, r, touchup=True)

    def optimize_boundary_s_t_trustregion_fun_and_jac(self, params: np.ndarray, x0: np.ndarray, x: np.ndarray,
                                                      b: np.ndarray, min_: float, max_: float,
                                                      c: float, r: float) -> Tuple[float, np.ndarray]:
        N: int = x0.shape[0]
        s: float = -np.sign(c)
        _mu: float = params[0]
        t: float = 1 / (2 * _mu + EPS)
        g: float = -_mu * r ** 2
        grad_mu: float = -r ** 2
        for n in range(N):
            d: float = -s * b[n] * t
            if d < min_ - x[n]:
                d = min_ - x[n]
            elif d > max_ - x[n]:
                d = max_ - x[n]
            else:
                grad_mu += (b[n] + 2 * _mu * d) * (b[n] / (2 * _mu ** 2 + EPS))
            grad_mu += d ** 2
            g += (b[n] + _mu * d) * d
        return (-g, -np.array([grad_mu]))

    def safe_div(self, nominator: float, denominator: float) -> float:
        if np.abs(denominator) > EPS:
            return nominator / denominator
        elif denominator >= 0:
            return nominator / EPS
        else:
            return -nominator / EPS

    def optimize_boundary_s_t_trustregion(self, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                                          min_: float, max_: float, c: float, r: float) -> np.ndarray:
        params0: np.ndarray = np.array([1.0])
        args: Tuple[Any, ...] = (x0, x, b, min_, max_, c, r)
        bounds: np.ndarray = np.array([(0, np.inf)])
        qk: np.ndarray = self.bfgsb.solve(self.optimize_boundary_s_t_trustregion_fun_and_jac,
                                          params0, bounds, args)
        _delta: np.ndarray = self.safe_div(-b, 2 * qk[0])
        for n in range(x0.shape[0]):
            if _delta[n] < min_ - x[n]:
                _delta[n] = min_ - x[n]
            elif _delta[n] > max_ - x[n]:
                _delta[n] = max_ - x[n]
        return _delta

class BrendelBethgeAttack(MinimizationAttack, ABC):
    def __init__(self, init_attack: Optional[Any] = None, overshoot: float = 1.1, steps: int = 1000,
                 lr: float = 0.001, lr_decay: float = 0.5, lr_num_decay: int = 20, momentum: float = 0.8,
                 tensorboard: Union[bool, str] = False, binary_search_steps: int = 10) -> None:
        if NUMBA_IMPORT_ERROR is not None:
            raise NUMBA_IMPORT_ERROR
        if '0.49.' in numba.__version__:
            warnings.warn('There are known issues with numba version 0.49 and we suggest using numba 0.50 or newer.')
        self.init_attack: Optional[Any] = init_attack
        self.overshoot: float = overshoot
        self.steps: int = steps
        self.lr: float = lr
        self.lr_decay: float = lr_decay
        self.lr_num_decay: int = lr_num_decay
        self.momentum: float = momentum
        self.tensorboard: Union[bool, str] = tensorboard
        self.binary_search_steps: int = binary_search_steps
        self._optimizer: Optimizer = self.instantiate_optimizer()

    def run(self, model: Model, inputs: Any, criterion: Any, *,
            starting_points: Optional[Any] = None, early_stop: Optional[Any] = None, **kwargs: Any) -> Any:
        raise_if_kwargs(kwargs)
        del kwargs
        tb: TensorBoard = TensorBoard(logdir=self.tensorboard)
        originals, restore_type = ep.astensor_(inputs)
        del inputs
        verify_input_bounds(originals, model)
        criterion_ = get_criterion(criterion)
        del criterion
        is_adversarial = get_is_adversarial(criterion_, model)
        if isinstance(criterion_, Misclassification):
            targeted: bool = False
            classes = criterion_.labels
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
        else:
            raise ValueError('unsupported criterion')
        if starting_points is None:
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack()
                logging.info(f'Neither starting_points nor init_attack given. Falling back to {init_attack!r} for initialization.')
            else:
                init_attack = self.init_attack
            starting_points = init_attack.run(model, originals, criterion_)
        best_advs = ep.astensor(starting_points)
        assert is_adversarial(best_advs).all()
        N: int = len(originals)
        rows = range(N)
        bounds: Tuple[float, float] = model.bounds
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
        tb.scalar('batchsize', N, 0)

        def loss_fun(x: Any) -> Tuple[Any, Any]:
            logits = model(x)
            if targeted:
                c_minimize = best_other_classes(logits, classes)
                c_maximize = classes
            else:
                c_minimize = classes
                c_maximize = best_other_classes(logits, classes)
            logits_diffs = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert logits_diffs.shape == (N,)
            return (logits_diffs.sum(), logits_diffs)
        value_and_grad = ep.value_and_grad_fn(x0, loss_fun, has_aux=True)

        def logits_diff_and_grads(x: Any) -> Tuple[np.ndarray, np.ndarray]:
            _, logits_diffs, boundary = value_and_grad(x)
            return (logits_diffs.numpy(), boundary.numpy().copy())
        x = starting_points
        lrs: np.ndarray = self.lr * np.ones(N)
        lr_reduction_interval: int = max(1, int(self.steps / self.lr_num_decay))
        converged: np.ndarray = np.zeros(N, dtype=np.bool_)
        rate_normalization: float = np.prod(x.shape) * (max_ - min_)
        original_shape = x.shape
        _best_advs: np.ndarray = best_advs.numpy()
        for step in range(1, self.steps + 1):
            if converged.all():
                break
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
            corr_logits_diffs = np.where(-logits_diffs < 0, -self.overshoot * logits_diffs, -(2 - self.overshoot) * logits_diffs)
            deltas: List[np.ndarray] = []
            k: int = 0
            for sample in range(N):
                if converged[sample]:
                    deltas.append(np.zeros_like(x0_np_flatten[sample]))
                else:
                    _x0 = x0_np_flatten[sample]
                    _x = x_np_flatten[sample]
                    _b = boundary[k].flatten()
                    _c = corr_logits_diffs[k]
                    r_sample: float = region[sample]
                    delta = self._optimizer.solve(_x0, _x, _b, bounds[0], bounds[1], _c, r_sample)
                    deltas.append(delta)
                    k += 1
            deltas_np = np.stack(deltas)
            deltas_ep = ep.from_numpy(x, deltas_np.astype(np.float32))
            x = (x + ep.astensor(deltas_ep)).reshape(original_shape)
            tb.probability('converged', ep.from_numpy(x, converged.astype(np.bool_)), step)
            tb.histogram('norms', source_norms, step)
            tb.histogram('candidates/distances', distances, step)
        tb.close()
        return restore_type(best_advs)

    @abstractmethod
    def instantiate_optimizer(self) -> Optimizer:
        raise NotImplementedError

    @abstractmethod
    def norms(self, x: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def mid_points(self, x0: Any, x1: Any, epsilons: Any, bounds: Tuple[float, float]) -> Any:
        raise NotImplementedError

def best_other_classes(logits: Any, exclude: Any) -> Any:
    other_logits = logits - ep.onehot_like(logits, exclude, value=np.inf)
    return other_logits.argmax(axis=-1)

class L2BrendelBethgeAttack(BrendelBethgeAttack):
    distance = l2

    def instantiate_optimizer(self) -> Optimizer:
        from warnings import warn
        if len(L2Optimizer._ctor.signatures) == 0:
            warn('At the first initialisation the optimizer needs to be compiled. This may take between 20 to 60 seconds.')
        return L2Optimizer()

    def norms(self, x: Any) -> Any:
        return flatten(x).norms.l2(axis=-1)

    def mid_points(self, x0: Any, x1: Any, epsilons: Any, bounds: Tuple[float, float]) -> Any:
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        return epsilons * x1 + (1 - epsilons) * x0

class LinfinityBrendelBethgeAttack(BrendelBethgeAttack):
    distance = linf

    def instantiate_optimizer(self) -> Optimizer:
        return LinfOptimizer()

    def norms(self, x: Any) -> Any:
        return flatten(x).norms.linf(axis=-1)

    def mid_points(self, x0: Any, x1: Any, epsilons: Any, bounds: Tuple[float, float]) -> Any:
        delta = x1 - x0
        min_, max_ = bounds
        s = max_ - min_
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        clipped_delta = ep.where(delta < -epsilons * s, -epsilons * s, delta)
        clipped_delta = ep.where(clipped_delta > epsilons * s, epsilons * s, clipped_delta)
        return x0 + clipped_delta

class L1BrendelBethgeAttack(BrendelBethgeAttack):
    distance = l1

    def instantiate_optimizer(self) -> Optimizer:
        return L1Optimizer()

    def norms(self, x: Any) -> Any:
        return flatten(x).norms.l1(axis=-1)

    def mid_points(self, x0: Any, x1: Any, epsilons: Any, bounds: Tuple[float, float]) -> Any:
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        threshold = (bounds[1] - bounds[0]) * (1 - epsilons)
        mask = (x1 - x0).abs() > threshold
        new_x = ep.where(mask, x0 + (x1 - x0).sign() * ((x1 - x0).abs() - threshold), x0)
        return new_x

class L0BrendelBethgeAttack(BrendelBethgeAttack):
    distance = l0

    def instantiate_optimizer(self) -> Optimizer:
        return L0Optimizer()

    def norms(self, x: Any) -> Any:
        return (flatten(x).abs() > 0.0001).sum(axis=-1)

    def mid_points(self, x0: Any, x1: Any, epsilons: Any, bounds: Tuple[float, float]) -> Any:
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        threshold = (bounds[1] - bounds[0]) * epsilons
        mask = ep.abs(x1 - x0) < threshold
        new_x = ep.where(mask, x1, x0)
        return new_x

@jitclass(spec=[])
class BFGSB(object):
    def __init__(self) -> None:
        pass

    def solve(self, fun_and_jac: Callable[..., Tuple[float, np.ndarray]], q0: np.ndarray,
              bounds: np.ndarray, args: Tuple[Any, ...],
              ftol: float = 1e-10, pgtol: float = -1e-05,
              maxiter: Optional[int] = None) -> np.ndarray:
        N: int = q0.shape[0]
        if maxiter is None:
            maxiter = N * 200
        l: np.ndarray = bounds[:, 0]
        u: np.ndarray = bounds[:, 1]
        func_calls: int = 0
        old_fval, gfk = fun_and_jac(q0, *args)
        func_calls += 1
        k: int = 0
        Hk: np.ndarray = np.eye(N)
        qk: np.ndarray = q0
        old_old_fval: float = old_fval + np.linalg.norm(gfk) / 2
        _gfk: np.ndarray = gfk
        while k < maxiter:
            pg_norm: float = 0.0
            for v in range(N):
                if _gfk[v] < 0:
                    gv: float = max(qk[v] - u[v], _gfk[v])
                else:
                    gv = min(qk[v] - l[v], _gfk[v])
                if pg_norm < np.abs(gv):
                    pg_norm = np.abs(gv)
            if pg_norm < pgtol:
                break
            x_cp: np.ndarray = self._cauchy_point(qk, l, u, _gfk.copy(), Hk)
            qk1: np.ndarray = self._subspace_min(qk, l, u, x_cp, _gfk.copy(), Hk)
            pk: np.ndarray = qk1 - qk
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1, fnev = self._line_search_wolfe(fun_and_jac, qk, pk, _gfk, old_fval, old_old_fval, l, u, args)
            func_calls += fnev
            if alpha_k is None:
                break
            if np.abs(old_fval - old_old_fval) <= ftol + ftol * np.abs(old_fval):
                break
            qkp1: np.ndarray = self._project(qk + alpha_k * pk, l, u)
            if gfkp1 is None:
                _, gfkp1 = fun_and_jac(qkp1, *args)
            sk: np.ndarray = qkp1 - qk
            qk = qkp1
            yk: np.ndarray = np.zeros_like(qk)
            for k3 in range(N):
                yk[k3] = gfkp1[k3] - _gfk[k3]
                if np.abs(yk[k3]) < 0.0001:
                    yk[k3] = -0.0001
            _gfk = gfkp1
            k += 1
            Hk_sk: np.ndarray = Hk.dot(sk)
            sk_yk: float = 0.0
            sk_Hk_sk: float = 0.0
            for v in range(N):
                sk_yk += sk[v] * yk[v]
                sk_Hk_sk += sk[v] * Hk_sk[v]
            if np.abs(sk_yk) >= 1e-08:
                rhok: float = 1.0 / sk_yk
            else:
                rhok = 100000.0
            if np.abs(sk_Hk_sk) >= 1e-08:
                rsk_Hk_sk: float = 1.0 / sk_Hk_sk
            else:
                rsk_Hk_sk = 100000.0
            for v in range(N):
                for w in range(N):
                    Hk[v, w] += yk[v] * yk[w] * rhok - Hk_sk[v] * Hk_sk[w] * rsk_Hk_sk
        return qk

    def _cauchy_point(self, x: np.ndarray, l: np.ndarray, u: np.ndarray, g: np.ndarray,
                       B: np.ndarray) -> np.ndarray:
        n: int = x.shape[0]
        t: np.ndarray = np.zeros_like(x)
        d: np.ndarray = np.zeros_like(x)
        for i in range(n):
            if g[i] < 0:
                t[i] = (x[i] - u[i]) / (g[i] - EPS)
            elif g[i] > 0:
                t[i] = (x[i] - l[i]) / (g[i] + EPS)
            elif g[i] == 0:
                t[i] = np.inf
            if t[i] == 0:
                d[i] = 0
            else:
                d[i] = -g[i]
        ts: np.ndarray = t.copy()
        ts = ts[ts != 0]
        ts = np.sort(ts)
        df: float = g.dot(d)
        d2f: float = d.dot(B.dot(d))
        if d2f < 1e-10:
            return x
        dt_min: float = -df / d2f
        t_old: float = 0.0
        i: int = 0
        z: np.ndarray = np.zeros_like(x)
        while i < ts.shape[0] and dt_min >= ts[i] - t_old:
            ind = ts[i] < t
            d[~ind] = 0
            z = z + (ts[i] - t_old) * d
            df = g.dot(d) + d.dot(B.dot(z))
            d2f = d.dot(B.dot(d))
            dt_min = df / (d2f + 1e-08)
            t_old = ts[i]
            i += 1
        dt_min = max(dt_min, 0)
        t_old = t_old + dt_min
        x_cp: np.ndarray = x - t_old * g
        temp: np.ndarray = x - t * g
        x_cp[t_old > t] = temp[t_old > t]
        return x_cp

    def _subspace_min(self, x: np.ndarray, l: np.ndarray, u: np.ndarray, x_cp: np.ndarray,
                      d: np.ndarray, G: np.ndarray) -> np.ndarray:
        n: int = x.shape[0]
        Z: np.ndarray = np.eye(n)
        fixed: np.ndarray = (x_cp <= l + 1e-08) + (x_cp >= u - 100000000.0)
        if np.all(fixed):
            x = x_cp
            return x
        Z = Z[:, ~fixed]
        rgc: np.ndarray = Z.T.dot(d + G.dot(x_cp - x))
        rB: np.ndarray = Z.T.dot(G.dot(Z)) + 1e-10 * np.eye(Z.shape[1])
        d[~fixed] = np.linalg.solve(rB, rgc)
        d[~fixed] = -d[~fixed]
        alpha: float = 1.0
        temp1: float = alpha
        for i in np.arange(n)[~fixed]:
            dk: float = d[i]
            if dk < 0:
                temp2: float = l[i] - x_cp[i]
                if temp2 >= 0:
                    temp1 = 0
                elif dk * alpha < temp2:
                    temp1 = temp2 / (dk - EPS)
                else:
                    temp2 = u[i] - x_cp[i]
            else:
                temp2 = u[i] - x_cp[i]
                if temp1 <= 0:
                    temp1 = 0
                elif dk * alpha > temp2:
                    temp1 = temp2 / (dk + EPS)
            alpha = min(temp1, alpha)
        return x_cp + alpha * Z.dot(d[~fixed])

    def _project(self, q: np.ndarray, l: np.ndarray, u: np.ndarray) -> np.ndarray:
        N: int = q.shape[0]
        for k in range(N):
            if q[k] < l[k]:
                q[k] = l[k]
            elif q[k] > u[k]:
                q[k] = u[k]
        return q

    def _line_search_armijo(self, fun_and_jac: Callable[..., Tuple[float, np.ndarray]],
                            pt: np.ndarray, dpt: np.ndarray, func_calls: int, m: float,
                            gk: np.ndarray, l: np.ndarray, u: np.ndarray,
                            x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                            min_: float, max_: float, c: float, r: float
                            ) -> Tuple[float, np.ndarray, float, np.ndarray, int]:
        ls_rho: float = 0.6
        ls_c: float = 0.0001
        ls_alpha: float = 1.0
        t: float = m * ls_c
        for k2 in range(100):
            ls_pt: np.ndarray = self._project(pt + ls_alpha * dpt, l, u)
            gkp1, dgkp1 = fun_and_jac(ls_pt, x0, x, b, min_, max_, c, r)
            func_calls += 1
            if gk - gkp1 >= ls_alpha * t:
                break
            else:
                ls_alpha *= ls_rho
        return (ls_alpha, ls_pt, gkp1, dgkp1, func_calls)

    def _line_search_wolfe(self, fun_and_jac: Callable[..., Tuple[float, np.ndarray]],
                           xk: np.ndarray, pk: np.ndarray, gfk: np.ndarray, old_fval: float,
                           old_old_fval: float, l: np.ndarray, u: np.ndarray,
                           args: Tuple[Any, ...]
                           ) -> Tuple[Optional[float], int, int, float, float, Optional[np.ndarray], int]:
        c1: float = 0.0001
        c2: float = 0.9
        N: int = xk.shape[0]
        _ls_fc: int = 0
        _ls_ingfk: Optional[np.ndarray] = None
        alpha0: float = 0.0
        phi0: float = old_fval
        derphi0: float = 0.0
        for v in range(N):
            derphi0 += gfk[v] * pk[v]
        if derphi0 == 0:
            derphi0 = 1e-08
        elif np.abs(derphi0) < 1e-08:
            derphi0 = np.sign(derphi0) * 1e-08
        alpha1: float = min(1.0, 1.01 * 2 * (phi0 - old_old_fval) / derphi0)
        if alpha1 == 0:
            alpha_star: Optional[float] = None
            fval_star: float = old_fval
            old_fval = old_old_fval
            fprime_star: Optional[np.ndarray] = None
        _xkp1: np.ndarray = self._project(xk + alpha1 * pk, l, u)
        phi_a1, _ls_ingfk = fun_and_jac(_xkp1, *args)
        _ls_fc += 1
        phi_a0: float = phi0
        derphi_a0: float = derphi0
        i: int = 1
        maxiter: int = 10
        while True:
            if alpha1 == 0:
                break
            if phi_a1 > phi0 + c1 * alpha1 * derphi0 or (phi_a1 >= phi_a0 and i > 1):
                k: int = 0
                delta1: float = 0.2
                delta2: float = 0.1
                phi_rec: float = phi0
                a_rec: float = 0.0
                a_hi: float = alpha1
                a_lo: float = alpha0
                phi_lo: float = phi_a0
                phi_hi: float = phi_a1
                derphi_lo: float = derphi_a0
                while True:
                    dalpha: float = a_hi - a_lo
                    if dalpha < 0:
                        a, b_var = (a_hi, a_lo)
                    else:
                        a, b_var = (a_lo, a_hi)
                    if k > 0:
                        cchk: float = delta1 * dalpha
                        a_j = self._cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
                    else:
                        a_j = None
                    if k == 0 or a_j is None or a_j > b_var - cchk or (a_j < a + cchk):
                        qchk: float = delta2 * dalpha
                        a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                        if a_j is None or a_j > b_var - qchk or a_j < a + qchk:
                            a_j = a_lo + 0.5 * dalpha
                    _xkp1 = self._project(xk + a_j * pk, l, u)
                    phi_aj, _ls_ingfk = fun_and_jac(_xkp1, *args)
                    derphi_aj: float = 0.0
                    for v in range(N):
                        derphi_aj += _ls_ingfk[v] * pk[v]
                    if phi_aj > phi0 + c1 * a_j * derphi0 or phi_aj >= phi_lo:
                        phi_rec = phi_hi
                        a_rec = a_hi
                        a_hi = a_j
                        phi_hi = phi_aj
                    else:
                        if abs(derphi_aj) <= -c2 * derphi0:
                            a_star = a_j
                            val_star = phi_aj
                            valprime_star = _ls_ingfk
                            break
                        if derphi_aj * (a_hi - a_lo) >= 0:
                            phi_rec = phi_hi
                            a_rec = a_hi
                            a_hi = a_lo
                            phi_hi = phi_lo
                        else:
                            phi_rec = phi_lo
                            a_rec = a_lo
                        a_lo = a_j
                        phi_lo = phi_aj
                        derphi_lo = derphi_aj
                    k += 1
                    if k > maxiter:
                        a_star = a_j
                        val_star = phi_aj
                        valprime_star = None
                        break
                alpha_star = a_star
                fval_star = val_star
                fprime_star = valprime_star
                fnev = k
                _ls_fc += fnev
                break
            i += 1
            if i > maxiter:
                break
            _xkp1 = self._project(xk + alpha1 * pk, l, u)
            _, _ls_ingfk = fun_and_jac(_xkp1, *args)
            derphi_a1 = 0.0
            for v in range(N):
                derphi_a1 += _ls_ingfk[v] * pk[v]
            _ls_fc += 1
            if abs(derphi_a1) <= -c2 * derphi0:
                alpha_star = alpha1
                fval_star = phi_a1
                fprime_star = _ls_ingfk
                break
            if derphi_a1 >= 0:
                maxiter = 10
                k = 0
                delta1 = 0.2
                delta2 = 0.1
                phi_rec = phi0
                a_rec = 0.0
                a_hi = alpha0
                a_lo = alpha1
                phi_lo = phi_a1
                phi_hi = phi_a0
                derphi_lo = derphi_a1
                while True:
                    dalpha = a_hi - a_lo
                    if dalpha < 0:
                        a, b_var = (a_hi, a_lo)
                    else:
                        a, b_var = (a_lo, a_hi)
                    if k > 0:
                        cchk = delta1 * dalpha
                        a_j = self._cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
                    else:
                        a_j = None
                    if k == 0 or a_j is None or a_j > b_var - cchk or (a_j < a + cchk):
                        qchk = delta2 * dalpha
                        a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                        if a_j is None or a_j > b_var - qchk or a_j < a + qchk:
                            a_j = a_lo + 0.5 * dalpha
                    _xkp1 = self._project(xk + a_j * pk, l, u)
                    phi_aj, _ls_ingfk = fun_and_jac(_xkp1, *args)
                    derphi_aj = 0.0
                    for v in range(N):
                        derphi_aj += _ls_ingfk[v] * pk[v]
                    if phi_aj > phi0 + c1 * a_j * derphi0 or phi_aj >= phi_lo:
                        phi_rec = phi_hi
                        a_rec = a_hi
                        a_hi = a_j
                        phi_hi = phi_aj
                    else:
                        if abs(derphi_aj) <= -c2 * derphi0:
                            a_star = a_j
                            val_star = phi_aj
                            valprime_star = _ls_ingfk
                            break
                        if derphi_aj * (a_hi - a_lo) >= 0:
                            phi_rec = phi_hi
                            a_rec = a_hi
                            a_hi = a_lo
                            phi_hi = phi_lo
                        else:
                            phi_rec = phi_lo
                            a_rec = a_lo
                        a_lo = a_j
                        phi_lo = phi_aj
                        derphi_lo = derphi_aj
                    k += 1
                    if k > maxiter:
                        a_star = a_j
                        val_star = phi_aj
                        valprime_star = None
                        break
                alpha_star = a_star
                fval_star = val_star
                fprime_star = valprime_star
                fnev = k
                _ls_fc += fnev
                break
            alpha2 = 2 * alpha1
            i = i + 1
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            _xkp1 = self._project(xk + alpha1 * pk, l, u)
            phi_a1, _ls_ingfk = fun_and_jac(_xkp1, *args)
            _ls_fc += 1
            derphi_a0 = derphi_a1
            if i > maxiter:
                alpha_star = alpha1
                fval_star = phi_a1
                fprime_star = None
                break
        return (alpha_star, _ls_fc, _ls_fc, fval_star, old_fval, fprime_star, _ls_fc)

    def _cubicmin(self, a: float, fa: float, fpa: float, b: float, fb: float, c: float, fc: float) -> Optional[float]:
        C: float = fpa
        db: float = b - a
        dc: float = c - a
        if db == 0 or dc == 0 or b == c:
            return None
        denom: float = (db * dc) ** 2 * (db - dc)
        A: float = dc ** 2 * (fb - fa - C * db) - db ** 2 * (fc - fa - C * dc)
        B: float = -dc ** 3 * (fb - fa - C * db) + db ** 3 * (fc - fa - C * dc)
        A /= denom
        B /= denom
        radical: float = B * B - 3 * A * C
        if radical < 0:
            return None
        if A == 0:
            return None
        xmin: float = a + (-B + np.sqrt(radical)) / (3 * A)
        return xmin

    def _quadmin(self, a: float, fa: float, fpa: float, b: float, fb: float) -> Optional[float]:
        D: float = fa
        C: float = fpa
        db: float = b - a * 1.0
        if db == 0:
            return None
        B: float = (fb - D - C * db) / (db * db)
        if B <= 0:
            return None
        xmin: float = a - C / (2.0 * B)
        return xmin

if NUMBA_IMPORT_ERROR is None:
    spec = [('bfgsb', BFGSB.class_type.instance_type)]
else:
    spec = []

@jitclass(spec=spec)
class L2Optimizer(Optimizer):
    def optimize_distance_s_t_boundary_and_trustregion(self, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                                                       min_: float, max_: float, c: float, r: float) -> np.ndarray:
        N: int = x0.shape[0]
        clamp_c: float = 0.0
        clamp_norm: float = 0.0
        ck: float = c
        rk: float = r
        masked_values: int = 0
        mask: np.ndarray = np.zeros(N, dtype=np.uint8)
        delta: np.ndarray = np.empty_like(x0)
        dx: np.ndarray = x0 - x
        for k in range(20):
            bnorm: float = 1e-08
            bdotDx: float = 0.0
            for i in range(N):
                if mask[i] == 0:
                    bnorm += b[i] * b[i]
                    bdotDx += b[i] * dx[i]
            bdotDx = bdotDx / bnorm
            ck_bnorm: float = ck / bnorm
            b_scale: float = -bdotDx + ck / bnorm
            new_masked_values: int = 0
            delta_norm: float = 0.0
            descent_norm: float = 0.0
            boundary_step_norm: float = 0.0
            for i in range(N):
                if mask[i] == 0:
                    delta[i] = dx[i] + b[i] * b_scale
                    boundary_step_norm += b[i] * ck_bnorm * b[i] * ck_bnorm
                    delta_norm += delta[i] * delta[i]
                    descent_norm += (dx[i] - b[i] * bdotDx) ** 2
            if boundary_step_norm > rk * rk:
                for i in range(N):
                    if mask[i] == 0:
                        delta[i] = b[i] * ck_bnorm
            elif delta_norm > rk * rk:
                region_correct: float = np.sqrt(rk * rk - boundary_step_norm)
                region_correct = region_correct / (np.sqrt(descent_norm) + 1e-08)
                b_scale = -region_correct * bdotDx + ck / bnorm
                for i in range(N):
                    if mask[i] == 0:
                        delta[i] = region_correct * dx[i] + b[i] * b_scale
            for i in range(N):
                if mask[i] == 0:
                    if x[i] + delta[i] <= min_:
                        mask[i] = 1
                        delta[i] = min_ - x[i]
                        new_masked_values += 1
                        clamp_norm += delta[i] * delta[i]
                        clamp_c += b[i] * delta[i]
                    if x[i] + delta[i] >= max_:
                        mask[i] = 1
                        delta[i] = max_ - x[i]
                        new_masked_values += 1
                        clamp_norm += delta[i] * delta[i]
                        clamp_c += b[i] * delta[i]
            if new_masked_values == 0:
                break
            masked_values += new_masked_values
            if clamp_norm < r * r:
                rk = np.sqrt(r * r - clamp_norm)
            else:
                rk = 0.0
            ck = c - clamp_c
            if masked_values == N:
                break
        return delta

    def fun_and_jac(self, params: np.ndarray, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                    min_: float, max_: float, c: float, r: float) -> Tuple[float, np.ndarray]:
        lam: float = params[0]
        mu: float = params[1]
        N: int = x0.shape[0]
        g: float = 0.0
        d_g_d_lam: float = 0.0
        d_g_d_mu: float = 0.0
        distance: float = 0.0
        b_dot_d: float = 0.0
        d_norm: float = 0.0
        t: float = 1 / (2 * mu + 2)
        for n in range(N):
            dx = x0[n] - x[n]
            bn = b[n]
            d = (2 * dx - lam * bn) * t
            if d + x[n] > max_:
                d = max_ - x[n]
            elif d + x[n] < min_:
                d = min_ - x[n]
            else:
                prefac: float = 2 * (d - dx) + 2 * mu * d + lam * bn
                d_g_d_lam -= prefac * bn * t
                d_g_d_mu -= prefac * 2 * d * t
            distance += (d - dx) ** 2
            b_dot_d += bn * d
            d_norm += d ** 2
            g += (dx - d) ** 2 + mu * d ** 2 + lam * bn * d
            d_g_d_lam += bn * d
            d_g_d_mu += d ** 2
        g += -mu * r ** 2 - lam * c
        d_g_d_lam -= c
        d_g_d_mu -= r ** 2
        return (-g, -np.array([d_g_d_lam, d_g_d_mu]))

    def _get_final_delta(self, lam: float, mu: float, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                         min_: float, max_: float, c: float, r: float, touchup: bool = True) -> np.ndarray:
        delta: np.ndarray = np.empty_like(x0)
        N: int = x0.shape[0]
        t: float = 1 / (2 * mu + 2)
        for n in range(N):
            d: float = (2 * (x0[n] - x[n]) - lam * b[n]) * t
            if d + x[n] > max_:
                d = max_ - x[n]
            elif d + x[n] < min_:
                d = min_ - x[n]
            delta[n] = d
        return delta

    def _distance(self, x0: np.ndarray, x: np.ndarray) -> float:
        return np.linalg.norm(x0 - x) ** 2

@jitclass(spec=spec)
class L1Optimizer(Optimizer):
    def fun_and_jac(self, params: np.ndarray, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                    min_: float, max_: float, c: float, r: float) -> Tuple[float, np.ndarray]:
        lam: float = params[0]
        mu: float = params[1]
        N: int = x0.shape[0]
        g: float = 0.0
        d_g_d_lam: float = 0.0
        d_g_d_mu: float = 0.0
        if mu > 0:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                t: float = 1 / (2 * mu + EPS)
                u: float = -lam * bn * t - dx
                if np.abs(u) - t < 0:
                    d = dx
                else:
                    d = np.sign(u) * (np.abs(u) - t) + dx
                    if d + x[n] < min_:
                        d = min_ - x[n]
                    elif d + x[n] > max_:
                        d = max_ - x[n]
                    else:
                        prefac: float = np.sign(d - dx) + 2 * mu * d + lam * bn
                        d_g_d_lam -= prefac * bn * t
                        d_g_d_mu -= prefac * 2 * d * t
                g += np.abs(dx - d) + mu * d ** 2 + lam * bn * d
                d_g_d_lam += bn * d
                d_g_d_mu += d ** 2
        else:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                if np.abs(lam * bn) < 1:
                    d = dx
                elif np.sign(lam * bn) < 0:
                    d = max_ - x[n]
                else:
                    d = min_ - x[n]
                g += np.abs(dx - d) + mu * d ** 2 + lam * bn * d
                d_g_d_lam += bn * d
                d_g_d_mu += d ** 2
        g += -mu * r ** 2 - lam * c
        d_g_d_lam -= c
        d_g_d_mu -= r ** 2
        return (-g, -np.array([d_g_d_lam, d_g_d_mu]))

    def _get_final_delta(self, lam: float, mu: float, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                         min_: float, max_: float, c: float, r: float, touchup: bool = True) -> np.ndarray:
        delta: np.ndarray = np.empty_like(x0)
        N: int = x0.shape[0]
        b_dot_d: float = 0.0
        norm_d: float = 0.0
        distance: float = 0.0
        if mu > 0:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                t: float = 1 / (2 * mu + EPS)
                u: float = -lam * bn * t - dx
                if np.abs(u) - t < 0:
                    d = dx
                else:
                    d = np.sign(u) * (np.abs(u) - t) + dx
                    if d + x[n] < min_:
                        d = min_ - x[n]
                    elif d + x[n] > max_:
                        d = max_ - x[n]
                delta[n] = d
                b_dot_d += b[n] * d
                norm_d += d ** 2
                distance += np.abs(d - dx)
        else:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                if np.abs(lam * bn) < 1:
                    d = dx
                elif np.sign(lam * bn) < 0:
                    d = max_ - x[n]
                else:
                    d = min_ - x[n]
                delta[n] = d
                b_dot_d += b[n] * d
                norm_d += d ** 2
                distance += np.abs(d - dx)
        if touchup:
            dc: float = c - b_dot_d
            k: int = 0
            min_distance: float = np.inf
            min_distance_idx: int = 0
            for n in range(N):
                if np.abs(b[n]) > 0:
                    dx = x0[n] - x[n]
                    old_d: float = delta[n]
                    new_d: float = old_d + dc / (b[n] + np.sign(b[n]) * EPS)
                    if x[n] + new_d <= max_ and x[n] + new_d >= min_ and (norm_d - old_d ** 2 + new_d ** 2 <= r ** 2):
                        if k == 0:
                            min_distance = distance - np.abs(old_d - dx) + np.abs(new_d - dx)
                            min_distance_idx = n
                            k += 1
                        else:
                            new_distance = distance - np.abs(old_d - dx) + np.abs(new_d - dx)
                            if min_distance > new_distance:
                                min_distance = new_distance
                                min_distance_idx = n
            if k > 0:
                idx: int = min_distance_idx
                old_d = delta[idx]
                new_d = old_d + dc / (b[idx] + np.sign(b[idx]) * EPS)
                delta[idx] = new_d
        return delta

    def _distance(self, x0: np.ndarray, x: np.ndarray) -> float:
        return np.abs(x0 - x).sum()

@jitclass(spec=spec)
class LinfOptimizer(Optimizer):
    def optimize_distance_s_t_boundary_and_trustregion(self, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                                                       min_: float, max_: float, c: float, r: float) -> np.ndarray:
        params0: np.ndarray = np.array([0.0, 0.0])
        bounds: np.ndarray = np.array([(-np.inf, np.inf), (0, np.inf)])
        return self.binary_search(params0, bounds, x0, x, b, min_, max_, c, r)

    def binary_search(self, q0: np.ndarray, bounds: np.ndarray, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                      min_: float, max_: float, c: float, r: float, etol: float = 1e-06,
                      maxiter: int = 1000) -> np.ndarray:
        epsilon: float = (max_ - min_) / 2.0
        eps_low: float = min_
        eps_high: float = max_
        func_calls: int = 0
        bnorm: float = np.linalg.norm(b)
        lambda0: float = 2 * c / (bnorm ** 2 + EPS)
        k: int = 0
        while eps_high - eps_low > etol:
            fun_val, nfev, _lambda0 = self.fun(epsilon, x0, x, b, min_, max_, c, r, lambda0=lambda0)
            func_calls += nfev
            if fun_val > -np.inf:
                eps_high = epsilon
                lambda0 = _lambda0
            else:
                eps_low = epsilon
            k += 1
            epsilon = (eps_high - eps_low) / 2.0 + eps_low
            if k > 20:
                break
        delta: np.ndarray = self._get_final_delta(lambda0, eps_high, x0, x, b, min_, max_, c, r, touchup=True)
        return delta

    def _Linf_bounds(self, x0: np.ndarray, epsilon: float, ell: float, u: float) -> Tuple[np.ndarray, np.ndarray]:
        N: int = x0.shape[0]
        _ell: np.ndarray = np.empty_like(x0)
        _u: np.ndarray = np.empty_like(x0)
        for i in range(N):
            nx: float = x0[i] - epsilon
            px: float = x0[i] + epsilon
            if nx > ell:
                _ell[i] = nx
            else:
                _ell[i] = ell
            if px < u:
                _u[i] = px
            else:
                _u[i] = u
        return (_ell, _u)

    def fun(self, epsilon: float, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
            ell: float, u: float, c: float, r: float, lambda0: Optional[float] = None
            ) -> Tuple[float, int, float]:
        N: int = x.shape[0]
        _ell, _u = self._Linf_bounds(x0, epsilon, ell, u)
        _lambda: float = lambda0 if lambda0 is not None else 2 * c / ((np.linalg.norm(b) ** 2) + EPS)
        k: int = 0
        lambda_max: float = 1e10
        lambda_min: float = -1e10
        max_c: float = 0.0
        min_c: float = 0.0
        for n in range(N):
            if b[n] > 0:
                max_c += b[n] * (_u[n] - x[n])
                min_c += b[n] * (_ell[n] - x[n])
            else:
                max_c += b[n] * (_ell[n] - x[n])
                min_c += b[n] * (_u[n] - x[n])
        if c > max_c or c < min_c:
            return (-np.inf, k, _lambda)
        while True:
            k += 1
            _c: float = 0.0
            norm: float = 0.0
            _active_bnorm: float = 0.0
            for n in range(N):
                lam_step: float = _lambda * b[n] / 2
                if lam_step + x[n] < _ell[n]:
                    delta_step: float = _ell[n] - x[n]
                elif lam_step + x[n] > _u[n]:
                    delta_step = _u[n] - x[n]
                else:
                    delta_step = lam_step
                    _active_bnorm += b[n] ** 2
                _c += b[n] * delta_step
                norm += delta_step ** 2
            if 0.9999 * np.abs(c) - EPS < np.abs(_c) < 1.0001 * np.abs(c) + EPS:
                if norm > r ** 2:
                    return (-np.inf, k, _lambda)
                else:
                    return (-epsilon, k, _lambda)
            else:
                if _c > c:
                    lambda_max = _lambda
                else:
                    lambda_min = _lambda
                if _active_bnorm == 0:
                    _lambda = (lambda_max - lambda_min) / 2 + lambda_min
                else:
                    _lambda += 2 * (c - _c) / (_active_bnorm + EPS)
                dlambda: float = lambda_max - lambda_min
                if _lambda > lambda_max - 0.1 * dlambda or _lambda < lambda_min + 0.1 * dlambda:
                    _lambda = (lambda_max - lambda_min) / 2 + lambda_min

    def _get_final_delta(self, lam: float, eps: float, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                         min_: float, max_: float, c: float, r: float, touchup: bool = True) -> np.ndarray:
        N: int = x.shape[0]
        delta: np.ndarray = np.empty_like(x0)
        _ell, _u = self._Linf_bounds(x0, eps, min_, max_)
        for n in range(N):
            lam_step: float = lam * b[n] / 2
            if lam_step + x[n] < _ell[n]:
                delta[n] = _ell[n] - x[n]
            elif lam_step + x[n] > _u[n]:
                delta[n] = _u[n] - x[n]
            else:
                delta[n] = lam_step
        return delta

    def _distance(self, x0: np.ndarray, x: np.ndarray) -> float:
        return np.abs(x0 - x).max()

@jitclass(spec=spec)
class L0Optimizer(Optimizer):
    def optimize_distance_s_t_boundary_and_trustregion(self, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                                                       min_: float, max_: float, c: float, r: float) -> np.ndarray:
        params0: np.ndarray = np.array([0.0, 0.0])
        bounds: np.ndarray = np.array([(-np.inf, np.inf), (0, np.inf)])
        return self.minimize(params0, bounds, x0, x, b, min_, max_, c, r)

    def minimize(self, q0: np.ndarray, bounds: np.ndarray, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                 min_: float, max_: float, c: float, r: float, ftol: float = 1e-09, xtol: float = -1e-05,
                 maxiter: int = 1000) -> np.ndarray:
        delta, delta_norm = self.minimize_without_trustregion(x0, x, b, c, r, min_, max_)
        if delta_norm <= r:
            return delta
        else:
            args: Tuple[Any, ...] = (x0, x, b, min_, max_, c, r)
            results: np.ndarray = self._nelder_mead_algorithm(q0, bounds, args=args, tol_f=ftol, tol_x=xtol, max_iter=maxiter)
            delta = self._get_final_delta(results[0], results[1], x0, x, b, min_, max_, c, r, touchup=True)
        return delta

    def minimize_without_trustregion(self, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                                     c: float, r: float, ell: float, u: float) -> Tuple[np.ndarray, float]:
        delta: np.ndarray = x0 - x
        total: np.ndarray = np.empty_like(x0)
        total_b: np.ndarray = np.empty_like(x0)
        bdotdelta: float = b.dot(delta)
        delta_bdotdelta: float = c - bdotdelta
        for k in range(x0.shape[0]):
            if b[k] > 0 and delta_bdotdelta > 0:
                total_b[k] = (u - x0[k]) * b[k]
                total[k] = u - x0[k]
            elif b[k] > 0 and delta_bdotdelta < 0:
                total_b[k] = (ell - x0[k]) * b[k]
                total[k] = ell - x0[k]
            elif b[k] < 0 and delta_bdotdelta > 0:
                total_b[k] = (ell - x0[k]) * b[k]
                total[k] = ell - x0[k]
            else:
                total_b[k] = (u - x0[k]) * b[k]
                total[k] = u - x0[k]
        b_argsort: np.ndarray = np.argsort(np.abs(total_b))[::-1]
        for idx in b_argsort:
            if np.abs(c - bdotdelta) > np.abs(total_b[idx]):
                delta[idx] += total[idx]
                bdotdelta += total_b[idx]
            else:
                delta[idx] += (c - bdotdelta) / (b[idx] + 1e-20)
                break
        delta_norm: float = np.linalg.norm(delta)
        return (delta, delta_norm)

    def _nelder_mead_algorithm(self, q0: np.ndarray, bounds: np.ndarray, args: Tuple[Any, ...],
                                ρ: float = 1.0, χ: float = 2.0, γ: float = 0.5, σ: float = 0.5,
                                tol_f: float = 1e-08, tol_x: float = 1e-08, max_iter: int = 1000) -> np.ndarray:
        vertices: np.ndarray = self._initialize_simplex(q0)
        n: int = vertices.shape[1]
        self._check_params(ρ, χ, γ, σ, bounds, n)
        nit: int = 0
        ργ: float = ρ * γ
        ρχ: float = ρ * χ
        σ_n: float = σ ** n
        f_val: np.ndarray = np.empty(n + 1, dtype=np.float64)
        for i in range(n + 1):
            f_val[i] = self._neg_bounded_fun(bounds, vertices[i], args=args)
        sort_ind: np.ndarray = f_val.argsort()
        LV_ratio: float = 1.0
        x_bar: np.ndarray = vertices[sort_ind[:n]].sum(axis=0) / n
        while True:
            shrink: bool = False
            fail: bool = nit >= max_iter
            best_val_idx: int = sort_ind[0]
            worst_val_idx: int = sort_ind[n]
            term_f: bool = f_val[worst_val_idx] - f_val[best_val_idx] < tol_f
            term_x: bool = LV_ratio < tol_x
            if term_x or term_f or fail:
                break
            x_r: np.ndarray = x_bar + ρ * (x_bar - vertices[worst_val_idx])
            f_r: float = self._neg_bounded_fun(bounds, x_r, args=args)
            if f_r >= f_val[best_val_idx] and f_r < f_val[sort_ind[n - 1]]:
                vertices[worst_val_idx] = x_r
                LV_ratio *= ρ
            elif f_r < f_val[best_val_idx]:
                x_e: np.ndarray = x_bar + χ * (x_r - x_bar)
                f_e: float = self._neg_bounded_fun(bounds, x_e, args=args)
                if f_e < f_r:
                    vertices[worst_val_idx] = x_e
                    LV_ratio *= ρχ
                else:
                    vertices[worst_val_idx] = x_r
                    LV_ratio *= ρ
            else:
                if f_r < f_val[worst_val_idx]:
                    x_c: np.ndarray = x_bar + γ * (x_r - x_bar)
                    LV_ratio_update: float = ργ
                else:
                    x_c = x_bar - γ * (x_r - x_bar)
                    LV_ratio_update = γ
                f_c: float = self._neg_bounded_fun(bounds, x_c, args=args)
                if f_c < min(f_r, f_val[worst_val_idx]):
                    vertices[worst_val_idx] = x_c
                    LV_ratio *= LV_ratio_update
                else:
                    shrink = True
                    for i in sort_ind[1:]:
                        vertices[i] = vertices[best_val_idx] + σ * (vertices[i] - vertices[best_val_idx])
                        f_val[i] = self._neg_bounded_fun(bounds, vertices[i], args=args)
                    sort_ind[1:] = f_val[sort_ind[1:]].argsort() + 1
                    x_bar = vertices[best_val_idx] + σ * (x_bar - vertices[best_val_idx]) + (vertices[worst_val_idx] - vertices[sort_ind[n]]) / n
                    LV_ratio *= σ_n
            if not shrink:
                f_val[worst_val_idx] = self._neg_bounded_fun(bounds, vertices[worst_val_idx], args=args)
                for i, j in enumerate(sort_ind):
                    if f_val[worst_val_idx] < f_val[j]:
                        sort_ind[i + 1:] = sort_ind[i:-1]
                        sort_ind[i] = worst_val_idx
                        break
                x_bar += (vertices[worst_val_idx] - vertices[sort_ind[n]]) / n
            nit += 1
        return vertices[sort_ind[0]]

    def _initialize_simplex(self, x0: np.ndarray) -> np.ndarray:
        n: int = x0.size
        vertices: np.ndarray = np.empty((n + 1, n), dtype=np.float64)
        vertices[:] = x0
        nonzdelt: float = 0.05
        zdelt: float = 0.00025
        for i in range(n):
            if vertices[i + 1, i] != 0.0:
                vertices[i + 1, i] *= 1 + nonzdelt
            else:
                vertices[i + 1, i] = zdelt
        return vertices

    def _check_params(self, ρ: float, χ: float, γ: float, σ: float, bounds: np.ndarray, n: int) -> None:
        if ρ < 0:
            raise ValueError('ρ must be strictly greater than 0.')
        if χ < 1:
            raise ValueError('χ must be strictly greater than 1.')
        if χ < ρ:
            raise ValueError('χ must be strictly greater than ρ.')
        if γ < 0 or γ > 1:
            raise ValueError('γ must be strictly between 0 and 1.')
        if σ < 0 or σ > 1:
            raise ValueError('σ must be strictly between 0 and 1.')
        if not (bounds.shape == (0, 2) or bounds.shape == (n, 2)):
            raise ValueError('The shape of `bounds` is not valid.')
        if (np.atleast_2d(bounds)[:, 0] > np.atleast_2d(bounds)[:, 1]).any():
            raise ValueError('Lower bounds must be greater than upper bounds.')

    def _check_bounds(self, x: np.ndarray, bounds: np.ndarray) -> bool:
        if bounds.shape == (0, 2):
            return True
        else:
            return (np.atleast_2d(bounds)[:, 0] <= x).all() and (x <= np.atleast_2d(bounds)[:, 1]).all()

    def _neg_bounded_fun(self, bounds: np.ndarray, x: np.ndarray, args: Tuple[Any, ...] = ()) -> float:
        if self._check_bounds(x, bounds):
            return -self.fun(x, *args)
        else:
            return np.inf

    def fun(self, params: np.ndarray, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
            min_: float, max_: float, c: float, r: float) -> float:
        lam: float = params[0]
        mu: float = params[1]
        N: int = x0.shape[0]
        g: float = -mu * r ** 2 - lam * c
        if mu > 0:
            t: float = 1 / (2 * mu + EPS)
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                case1: float = lam * bn * dx + mu * dx ** 2
                optd: float = -lam * bn * t
                if optd < min_ - x[n]:
                    optd = min_ - x[n]
                elif optd > max_ - x[n]:
                    optd = max_ - x[n]
                case2: float = 1 + lam * bn * optd + mu * optd ** 2
                if case1 <= case2:
                    g += mu * dx ** 2 + lam * bn * dx
                else:
                    g += 1 + mu * optd ** 2 + lam * bn * optd
        else:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                case1: float = lam * bn * dx
                case2: float = 1 + lam * bn * (min_ - x[n])
                case3: float = 1 + lam * bn * (max_ - x[n])
                if case1 <= case2 and case1 <= case3:
                    g += mu * dx ** 2 + lam * bn * dx
                elif case2 < case3:
                    g += 1 + mu * (min_ - x[n]) ** 2 + lam * bn * (min_ - x[n])
                else:
                    g += 1 + mu * (max_ - x[n]) ** 2 + lam * bn * (max_ - x[n])
        return g

    def _get_final_delta(self, lam: float, mu: float, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                         min_: float, max_: float, c: float, r: float, touchup: bool = True) -> Optional[np.ndarray]:
        if touchup:
            delta = self.__get_final_delta(lam, mu, x0, x, b, min_, max_, c, r)
            if delta is not None:
                return delta
            else:
                params = [(lam + 1e-05, mu), (lam, mu + 1e-05), (lam - 1e-05, mu), (lam, mu - 1e-05),
                          (lam + 1e-05, mu + 1e-05), (lam - 1e-05, mu - 1e-05),
                          (lam + 1e-05, mu - 1e-05), (lam - 1e-05, mu + 1e-05)]
                for param in params:
                    delta = self.__get_final_delta(param[0], param[1], x0, x, b, min_, max_, c, r)
                    if delta is not None:
                        return delta
                return self.__get_final_delta(lam, mu, x0, x, b, min_, max_, c, r, False)
        else:
            return self.__get_final_delta(lam, mu, x0, x, b, min_, max_, c, r, False)

    def __get_final_delta(self, lam: float, mu: float, x0: np.ndarray, x: np.ndarray, b: np.ndarray,
                          min_: float, max_: float, c: float, r: float, touchup: bool = True) -> Optional[np.ndarray]:
        delta: np.ndarray = np.empty_like(x0)
        N: int = x0.shape[0]
        b_dot_d: float = 0.0
        norm_d: float = 0.0
        distance: float = 0.0
        if mu > 0:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                t: float = 1 / (2 * mu + EPS)
                case1: float = lam * bn * dx + mu * dx ** 2
                optd: float = -lam * bn * t
                if optd < min_ - x[n]:
                    optd = min_ - x[n]
                elif optd > max_ - x[n]:
                    optd = max_ - x[n]
                case2: float = 1 + lam * bn * optd + mu * optd ** 2
                if case1 <= case2:
                    d = dx
                else:
                    d = optd
                    distance += 1
                delta[n] = d
                b_dot_d += bn * d
                norm_d += d ** 2
        else:
            for n in range(N):
                dx = x0[n] - x[n]
                bn = b[n]
                case1 = lam * bn * dx
                case2 = 1 + lam * bn * (min_ - x[n])
                case3 = 1 + lam * bn * (max_ - x[n])
                if case1 <= case2 and case1 <= case3:
                    d = dx
                elif case2 < case3:
                    d = min_ - x[n]
                    distance += 1
                else:
                    d = max_ - x[n]
                    distance += 1
                delta[n] = d
                norm_d += d ** 2
                b_dot_d += bn * d
        if touchup:
            dc: float = c - b_dot_d
            k: int = 0
            min_distance: float = np.inf
            min_norm: float = np.inf
            min_distance_idx: int = 0
            for n in range(N):
                if np.abs(b[n]) > 0:
                    dx = x0[n] - x[n]
                    old_d = delta[n]
                    new_d = old_d + dc / (b[n] + np.sign(b[n]) * EPS)
                    if x[n] + new_d <= max_ and x[n] + new_d >= min_ and (norm_d - old_d ** 2 + new_d ** 2 <= r ** 2):
                        if k == 0:
                            min_distance = (distance - (np.abs(old_d - dx) > 1e-10) + (np.abs(new_d - dx) > 1e-10))
                            min_distance_idx = n
                            min_norm = norm_d - old_d ** 2 + new_d ** 2
                            k += 1
                        else:
                            new_distance = (distance - (np.abs(old_d - dx) > 1e-10) + (np.abs(new_d - dx) > 1e-10))
                            if min_distance > new_distance or (min_distance == new_distance and min_norm > norm_d - old_d ** 2 + new_d ** 2):
                                min_distance = new_distance
                                min_norm = norm_d - old_d ** 2 + new_d ** 2
                                min_distance_idx = n
            if k > 0:
                idx: int = min_distance_idx
                old_d = delta[idx]
                new_d = old_d + dc / (b[idx] + np.sign(b[idx]) * EPS)
                delta[idx] = new_d
                return delta
            else:
                return None
        return delta

    def _distance(self, x0: np.ndarray, x: np.ndarray) -> int:
        return np.sum(np.abs(x - x0) > EPS)
