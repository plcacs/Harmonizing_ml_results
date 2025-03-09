from typing import Union, Optional, Tuple, Any, List, Callable, Dict
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
EPS = 1e-10

class Optimizer:

    def __init__(self):
        self.bfgsb = BFGSB()

    def solve(self, x0, x, b, min_, max_, c, r):
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

    def _max_logit_diff(self, x, b, _ell, _u, c):
        """Tests whether the (estimated) boundary can be reached within trust region."""
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
        return (cmax, np.sqrt(norm))

    def _minimum_norm_to_boundary(self, x, b, _ell, _u, c, bnorm):
        """Computes the minimum norm necessary to reach the boundary. More precisely, we aim to solve the
        following optimization problem

            min ||delta||_2^2 s.t. lower <= x + delta <= upper AND b.dot(delta) = c

        Lets forget about the box constraints for a second, i.e.

            min ||delta||_2^2 s.t. b.dot(delta) = c

        The dual of this problem is quite straight-forward to solve,

            g(lambda, delta) = ||delta||_2^2 + lambda * (c - b.dot(delta))

        The minimum of this Lagrangian is delta^* = lambda * b / 2, and so

            inf_delta g(lambda, delta) = lambda^2 / 4 ||b||_2^2 + lambda * c

        and so the optimal lambda, which maximizes inf_delta g(lambda, delta), is given by

            lambda^* = 2c / ||b||_2^2

        which in turn yields the optimal delta:

            delta^* = c * b / ||b||_2^2

        To take into account the box-constraints we perform a binary search over lambda and apply the box
        constraint in each step.
        """
        N = x.shape[0]
        lambda_lower = 2 * c / (bnorm ** 2 + EPS)
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

    def optimize_distance_s_t_boundary_and_trustregion(self, x0, x, b, min_, max_, c, r):
        """Find the solution to the optimization problem

        min_delta ||dx - delta||_p^p s.t. ||delta||_2^2 <= r^2 AND b^T delta = c AND min_ <= x + delta <= max_
        """
        params0 = np.array([0.0, 0.0])
        bounds = np.array([(-np.inf, np.inf), (0, np.inf)])
        args = (x0, x, b, min_, max_, c, r)
        qk = self.bfgsb.solve(self.fun_and_jac, params0, bounds, args)
        return self._get_final_delta(qk[0], qk[1], x0, x, b, min_, max_, c, r, touchup=True)

    def optimize_boundary_s_t_trustregion_fun_and_jac(self, params, x0, x, b, min_, max_, c, r):
        N = x0.shape[0]
        s = -np.sign(c)
        _mu = params[0]
        t = 1 / (2 * _mu + EPS)
        g = -_mu * r ** 2
        grad_mu = -r ** 2
        for n in range(N):
            d = -s * b[n] * t
            if d < min_ - x[n]:
                d = min_ - x[n]
            elif d > max_ - x[n]:
                d = max_ - x[n]
            else:
                grad_mu += (b[n] + 2 * _mu * d) * (b[n] / (2 * _mu ** 2 + EPS))
            grad_mu += d ** 2
            g += (b[n] + _mu * d) * d
        return (-g, -np.array([grad_mu]))

    def safe_div(self, nominator, denominator):
        if np.abs(denominator) > EPS:
            return nominator / denominator
        elif denominator >= 0:
            return nominator / EPS
        else:
            return -nominator / EPS

    def optimize_boundary_s_t_trustregion(self, x0, x, b, min_, max_, c, r):
        """Find the solution to the optimization problem

        min_delta sign(c) b^T delta s.t. ||delta||_2^2 <= r^2 AND min_ <= x + delta <= max_

        Note: this optimization problem is independent of the Lp norm being optimized.

        Lagrangian: g(delta) = sign(c) b^T delta + mu * (||delta||_2^2 - r^2)
        Optimal delta: delta = - sign(c) * b / (2 * mu)
        """
        params0 = np.array([1.0])
        args = (x0, x, b, min_, max_, c, r)
        bounds = np.array([(0, np.inf)])
        qk = self.bfgsb.solve(self.optimize_boundary_s_t_trustregion_fun_and_jac, params0, bounds, args)
        _delta = self.safe_div(-b, 2 * qk[0])
        for n in range(x0.shape[0]):
            if _delta[n] < min_ - x[n]:
                _delta[n] = min_ - x[n]
            elif _delta[n] > max_ - x[n]:
                _delta[n] = max_ - x[n]
        return _delta

class BrendelBethgeAttack(MinimizationAttack, ABC):

    def __init__(self, init_attack=None, overshoot=1.1, steps=1000, lr=0.001, lr_decay=0.5, lr_num_decay=20, momentum=0.8, tensorboard=False, binary_search_steps=10):
        if NUMBA_IMPORT_ERROR is not None:
            raise NUMBA_IMPORT_ERROR
        if '0.49.' in numba.__version__:
            warnings.warn('There are known issues with numba version 0.49 and we suggest using numba 0.50 or newer.')
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

    def run(self, model, inputs, criterion, *, starting_points: Optional[ep.Tensor]=None, early_stop: Optional[float]=None, **kwargs: Any):
        """Applies the Brendel & Bethge attack.

        Parameters
        ----------
        inputs : Tensor that matches model type
            The original clean inputs.
        criterion : Callable
            A callable that returns true if the given logits of perturbed
            inputs should be considered adversarial w.r.t. to the given labels
            and unperturbed inputs.
        starting_point : Tensor of same type and shape as inputs
            Adversarial inputs to use as a starting points, in particular
            for targeted attacks.
        """
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
            raise ValueError('unsupported criterion')
        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack()
                logging.info(f'Neither starting_points nor init_attack given. Falling back to {init_attack!r} for initialization.')
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
        tb.scalar('batchsize', N, 0)

        def loss_fun(x):
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

        def logits_diff_and_grads(x):
            _, logits_diffs, boundary = value_and_grad(x)
            return (logits_diffs.numpy(), boundary.numpy().copy())
        x = starting_points
        lrs = self.lr * np.ones(N)
        lr_reduction_interval = max(1, int(self.steps / self.lr_num_decay))
        converged = np.zeros(N, dtype=np.bool_)
        rate_normalization = np.prod(x.shape) * (max_ - min_)
        original_shape = x.shape
        _best_advs = best_advs.numpy()
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
            deltas, k = ([], 0)
            for sample in range(N):
                if converged[sample]:
                    deltas.append(np.zeros_like(x0_np_flatten[sample]))
                else:
                    _x0 = x0_np_flatten[sample]
                    _x = x_np_flatten[sample]
                    _b = boundary[k].flatten()
                    _c = corr_logits_diffs[k]
                    r = region[sample]
                    delta = self._optimizer.solve(_x0, _x, _b, bounds[0], bounds[1], _c, r)
                    deltas.append(delta)
                    k += 1
            deltas = np.stack(deltas)
            deltas = ep.from_numpy(x, deltas.astype(np.float32))
            x = (x + ep.astensor(deltas)).reshape(original_shape)
            tb.probability('converged', ep.from_numpy(x, converged.astype(np.bool_)), step)
            tb.histogram('norms', source_norms, step)
            tb.histogram('candidates/distances', distances, step)
        tb.close()
        return restore_type(best_advs)

    @abstractmethod
    def instantiate_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def norms(self, x):
        raise NotImplementedError

    @abstractmethod
    def mid_points(self, x0, x1, epsilons, bounds):
        raise NotImplementedError

def best_other_classes(logits, exclude):
    other_logits = logits - ep.onehot_like(logits, exclude, value=np.inf)
    return other_logits.argmax(axis=-1)

class L2BrendelBethgeAttack(BrendelBethgeAttack):
    distance = l2

    def instantiate_optimizer(self):
        if len(L2Optimizer._ctor.signatures) == 0:
            warnings.warn('At the first initialisation the optimizer needs to be compiled. This may take between 20 to 60 seconds.')
        return L2Optimizer()

    def norms(self, x):
        return flatten(x).norms.l2(axis=-1)

    def mid_points(self, x0, x1, epsilons, bounds):
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        return epsilons * x1 + (1 - epsilons) * x0

class LinfinityBrendelBethgeAttack(BrendelBethgeAttack):
    distance = linf

    def instantiate_optimizer(self):
        return LinfOptimizer()

    def norms(self, x):
        return flatten(x).norms.linf(axis=-1)

    def mid_points(self, x0, x1, epsilons, bounds):
        delta = x1 - x0
        min_, max_ = bounds
        s = max_ - min_
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        clipped_delta = ep.where(delta < -epsilons * s, -epsilons * s, delta)
        clipped_delta = ep.where(clipped_delta > epsilons * s, epsilons * s, clipped_delta)
        return x0 + clipped_delta

class L1BrendelBethgeAttack(BrendelBethgeAttack):
    distance = l1

    def instantiate_optimizer(self):
        return L1Optimizer()

    def norms(self, x):
        return flatten(x).norms.l1(axis=-1)

    def mid_points(self, x0, x1, epsilons, bounds):
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        threshold = (bounds[1] - bounds[0]) * (1 - epsilons)
        mask = (x1 - x0).abs() > threshold
        new_x = ep.where(mask, x0 + (x1 - x0).sign() * ((x1 - x0).abs() - threshold), x0)
        return new_x

class L0BrendelBethgeAttack(BrendelBethgeAttack):
    distance = l0

    def instantiate_optimizer(self):
        return L0Optimizer()

    def norms(self, x):
        return (flatten(x).abs() > 0.0001).sum(axis=-1)

    def mid_points(self, x0, x1, epsilons, bounds):
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        threshold = (bounds[1] - bounds[0]) * epsilons
        mask = ep.abs(x1 - x0) < threshold
        new_x = ep.where(mask, x1, x0)
        return new_x

@jitclass(spec=[])
class BFGSB:

    def __init__(self):
        pass

    def solve(self, fun_and_jac, q0, bounds, args, ftol=1e-10, pgtol=-1e-05, maxiter=None):
        N = q0.shape[0]
        if maxiter is None:
            maxiter = N * 200
        l = bounds[:, 0]
        u = bounds[:, 1]
        func_calls = 0
        old_fval, gfk = fun_and_jac(q0, *args)
        func_calls += 1
        k = 0
        Hk = np.eye(N)
        qk = q0
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2
        _gfk = gfk
        while k < maxiter:
            pg_norm = 0
            for v in range(N):
                if _gfk[v] < 0:
                    gv = max(qk[v] - u[v], _gfk[v])
                else:
                    gv = min(qk[v] - l[v], _gfk[v])
                if pg_norm < np.abs(gv):
                    pg_norm = np.abs(gv)
            if pg_norm < pgtol:
                break
            x_cp = self._cauchy_point(qk, l, u, _gfk.copy(), Hk)
            qk1 = self._subspace_min(qk, l, u, x_cp, _gfk.copy(), Hk)
            pk = qk1 - qk
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1, fnev = self._line_search_wolfe(fun_and_jac, qk, pk, _gfk, old_fval, old_old_fval, l, u, args)
            func_calls += fnev
            if alpha_k is None:
                break
            if np.abs(old_fval - old_old_fval) <= ftol + ftol * np.abs(old_fval):
                break
            qkp1 = self._project(qk + alpha_k * pk, l, u)
            if gfkp1 is None:
                _, gfkp1 = fun_and_jac(qkp1, *args)
            sk = qkp1 - qk
            qk = qkp1
            yk = np.zeros_like(qk)
            for k3 in range(N):
                yk[k3] = gfkp1[k3] - _gfk[k3]
                if np.abs(yk[k3]) < 0.0001:
                    yk[k3] = -0.0001
            _gfk = gfkp1
            k += 1
            Hk_sk = Hk.dot(sk)
            sk_yk = 0
            sk_Hk_sk = 0
            for v in range(N):
                sk_yk += sk[v] * yk[v]
                sk_Hk_sk += sk[v] * Hk_sk[v]
            if np.abs(sk_yk) >= 1e-08:
                rhok = 1.0 / sk_yk
            else:
                rhok = 100000.0
            if np.abs(sk_Hk_sk) >= 1e-08:
                rsk_Hk_sk = 1.0 / sk_Hk_sk
            else:
                rsk_Hk_sk = 100000.0
            for v in range(N):
                for w in range(N):
                    Hk[v, w] += yk[v] * yk[w] * rhok - Hk_sk[v] * Hk_sk[w] * rsk_Hk_sk
        return qk

    def _cauchy_point(self, x, l, u, g, B):
        n = x.shape[0]
        t = np.zeros_like(x)
        d = np.zeros_like(x)
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
        ts = t.copy()
        ts = ts[ts != 0]
        ts = np.sort(ts)
        df = g.dot(d)
        d2f = d.dot(B.dot(d))
        if d2f < 1e-10:
            return x
        dt_min = -df / d2f
        t_old = 0
        i = 0
        z = np.zeros_like(x)
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
        x_cp = x - t_old * g
        temp = x - t * g
        x_cp[t_old > t] = temp[t_old > t]
        return x_cp

    def _subspace_min(self, x, l, u, x_cp, d, G):
        n = x.shape[0]
        Z = np.eye(n)
        fixed = (x_cp <= l + 1e-08) + (x_cp >= u - 100000000.0)
        if np.all(fixed):
            x = x_cp
            return x
        Z = Z[:, ~fixed]
        rgc = Z.T.dot(d + G.dot(x_cp - x))
        rB = Z.T.dot(G.dot(Z)) + 1e-10 * np.eye(Z.shape[1])
        d[~fixed] = np.linalg.solve(rB, rgc)
        d[~fixed] = -d[~fixed]
        alpha = 1
        temp1 = alpha
        for i in np.arange(n)[~fixed]:
            dk = d[i]
            if dk < 0:
                temp2 = l[i] - x_cp[i]
                if temp2 >= 0:
                    temp1 = 0
                elif dk * alpha < temp2:
                    temp