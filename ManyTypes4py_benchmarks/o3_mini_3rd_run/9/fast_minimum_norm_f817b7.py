#!/usr/bin/env python3
import math
from abc import abstractmethod, ABC
from typing import Union, Optional, Any, Tuple, Callable
import eagerpy as ep
from eagerpy.astensor import T
from .base import MinimizationAttack, raise_if_kwargs, get_criterion, get_is_adversarial
from .gradient_descent_base import uniform_l1_n_balls, normalize_lp_norms, uniform_l2_n_balls
from .. import Model, Misclassification, TargetedMisclassification
from ..devutils import atleast_kd, flatten
from ..distances import l1, linf, l2, l0, LpDistance

# Maps for Lp norms, where p can be integer or float
ps: dict = {l0: 0, l1: 1, l2: 2, linf: ep.inf, LpDistance: ep.nan}
duals: dict = {l0: ep.nan, l1: ep.inf, l2: 2, linf: 1, LpDistance: ep.nan}


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits: ep.Tensor = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)


def project_onto_l1_ball(x: ep.Tensor, eps: ep.Tensor) -> ep.Tensor:
    """
    Computes Euclidean projection onto the L1 ball for a batch. [Duchi08]_

    Adapted from the pytorch version by Tony Duan:
    https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55

    Args:
        x: Batch of arbitrary-size tensors to project, possibly on GPU.
        eps: Radius of l-1 ball to project onto.
    """
    original_shape: Tuple[int, ...] = x.shape
    x = flatten(x)
    mask: ep.Tensor = (ep.norms.l1(x, axis=1) <= eps).astype(x.dtype).expand_dims(1)
    mu: ep.Tensor = ep.flip(ep.sort(ep.abs(x)), axis=-1).astype(x.dtype)
    cumsum: ep.Tensor = ep.cumsum(mu, axis=-1)
    arange: ep.Tensor = ep.arange(x, 1, x.shape[1] + 1).astype(x.dtype)
    rho: ep.Tensor = ep.max((mu * arange > cumsum - eps.expand_dims(1)).astype(x.dtype) * arange, axis=-1) - 1
    rho = ep.maximum(rho, 0)
    theta: ep.Tensor = (cumsum[ep.arange(x, x.shape[0]), rho.astype(ep.arange(x, 1).dtype)] - eps) / (rho + 1.0)
    proj: ep.Tensor = (ep.abs(x) - theta.expand_dims(1)).clip(min_=0, max_=ep.inf)
    x = mask * x + (1 - mask) * proj * ep.sign(x)
    return x.reshape(original_shape)


class FMNAttackLp(MinimizationAttack, ABC):
    """
    The Fast Minimum Norm adversarial attack, in Lp norm. [Pintor21]_

    Args:
        steps: Number of iterations.
        max_stepsize: Initial stepsize for the gradient update.
        min_stepsize: Final stepsize for the gradient update. The
            stepsize will be reduced with a cosine annealing policy.
        gamma: Initial stepsize for the epsilon update. It will
            be updated with a cosine annealing reduction up to 0.001.
        init_attack: Optional initial attack. If an initial attack
            is specified (or initial points are provided in the run), the
            attack will first try to search for the boundary between the
            initial point and the points in a class that satisfies the
            adversarial criterion.
        binary_search_steps: Number of steps to use for the search
            from the adversarial points. If no initial attack or adversarial
            starting point is provided, this parameter will be ignored.
    """

    def __init__(
        self,
        *,
        steps: int = 100,
        max_stepsize: float = 1.0,
        min_stepsize: Optional[float] = None,
        gamma: float = 0.05,
        init_attack: Optional[MinimizationAttack] = None,
        binary_search_steps: int = 10,
    ):
        self.steps: int = steps
        self.max_stepsize: float = max_stepsize
        self.init_attack: Optional[MinimizationAttack] = init_attack
        if min_stepsize is not None:
            self.min_stepsize: float = min_stepsize
        else:
            self.min_stepsize = max_stepsize / 100
        self.binary_search_steps: int = binary_search_steps
        self.gamma: float = gamma
        self.p: Union[int, float] = ps[self.distance]  # type: ignore[attr-defined]
        self.dual: Union[int, float] = duals[self.distance]  # type: ignore[attr-defined]

    def run(
        self,
        model: Model,
        inputs: Any,
        criterion: Any,
        *,
        starting_points: Optional[ep.Tensor] = None,
        early_stop: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        raise_if_kwargs(kwargs)
        criterion_ = get_criterion(criterion)
        if isinstance(criterion_, Misclassification):
            targeted: bool = False
            classes: ep.Tensor = criterion_.labels  # type: ignore[attr-defined]
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes  # type: ignore[attr-defined]
        else:
            raise ValueError("unsupported criterion")

        # rows will be assigned later, so we declare loss_fn with closure capturing it.
        def loss_fn(inputs_inner: ep.Tensor, labels: ep.Tensor) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            # 'rows' is nonlocal, defined later in run.
            logits: ep.Tensor = model(inputs_inner)
            if targeted:
                c_minimize: ep.Tensor = best_other_classes(logits, labels)
                c_maximize: ep.Tensor = labels
            else:
                c_minimize = labels
                c_maximize = best_other_classes(logits, labels)
            # Use rows from the outer scope.
            loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            return (-loss.sum(), (logits, loss))
        
        x, restore_type = ep.astensor_(inputs)  # type: Tuple[ep.Tensor, Callable[[ep.Tensor], Any]]
        del inputs, criterion, kwargs
        N: int = len(x)
        initialized: bool = False
        if starting_points is not None:
            x1: ep.Tensor = starting_points
            initialized = True
        elif self.init_attack is not None:
            x1 = self.init_attack.run(model, x, criterion_)
            initialized = True
        if initialized is True:
            is_adv: Callable[[ep.Tensor], ep.Tensor] = get_is_adversarial(criterion_, model)
            assert is_adv(x1).all()
            lower_bound: ep.Tensor = ep.zeros(x, shape=(N,))
            upper_bound: ep.Tensor = ep.ones(x, shape=(N,))
            for _ in range(self.binary_search_steps):
                epsilons: ep.Tensor = (lower_bound + upper_bound) / 2
                mid_points: ep.Tensor = self.mid_points(x, x1, epsilons, model.bounds)  # type: ignore[attr-defined]
                is_advs: ep.Tensor = is_adv(mid_points)
                lower_bound = ep.where(is_advs, lower_bound, epsilons)
                upper_bound = ep.where(is_advs, epsilons, upper_bound)
            starting_points = self.mid_points(x, x1, upper_bound, model.bounds)
            delta: ep.Tensor = starting_points - x
        else:
            delta = ep.zeros_like(x)
        if classes.shape != (N,):
            name: str = "target_classes" if targeted else "labels"
            raise ValueError(f"expected {name} to have shape ({N},), got {classes.shape}")
        min_bound, max_bound = model.bounds  # type: Tuple[float, float]
        rows: range = range(N)
        grad_and_logits: Callable[[ep.Tensor, ep.Tensor], Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor], ep.Tensor]] = ep.value_and_grad_fn(x, loss_fn, has_aux=True)  # type: ignore

        if self.p != 0:
            epsilon: ep.Tensor = ep.inf * ep.ones(x, len(x))
        else:
            epsilon = ep.maximum(ep.ones(x, len(x)), ep.norms.l0(flatten(delta), axis=-1))
        if self.p != 0:
            worst_norm: ep.Tensor = ep.norms.lp(flatten(ep.maximum(x - min_bound, max_bound - x)), p=self.p, axis=-1)  # type: ignore
        else:
            worst_norm = flatten(ep.ones_like(x)).bool().sum(axis=1).float32()
        best_lp: ep.Tensor = worst_norm
        best_delta: ep.Tensor = delta
        adv_found: ep.Tensor = ep.zeros(x, len(x)).bool()
        for i in range(self.steps):
            stepsize: float = self.min_stepsize + (self.max_stepsize - self.min_stepsize) * (1 + math.cos(math.pi * i / self.steps)) / 2
            gamma: float = 0.001 + (self.gamma - 0.001) * (1 + math.cos(math.pi * (i / self.steps))) / 2
            x_adv: ep.Tensor = x + delta
            loss, (logits, loss_batch), gradients = grad_and_logits(x_adv, classes)
            is_adversarial: ep.Tensor = criterion_(x_adv, logits)  # type: ignore
            lp: ep.Tensor = ep.norms.lp(flatten(delta), p=self.p, axis=-1)  # type: ignore
            is_smaller: ep.Tensor = lp <= best_lp
            is_both: ep.Tensor = ep.logical_and(is_adversarial, is_smaller)
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_lp = ep.where(is_both, lp, best_lp)
            best_delta = ep.where(atleast_kd(is_both, x.ndim), delta, best_delta)
            if self.p != 0:
                distance_to_boundary: ep.Tensor = abs(loss_batch) / ep.norms.lp(flatten(gradients), p=self.dual, axis=-1)  # type: ignore
                epsilon = ep.where(
                    is_adversarial,
                    ep.minimum(epsilon * (1 - gamma), ep.norms.lp(flatten(best_delta), p=self.p, axis=-1)),  # type: ignore
                    ep.where(adv_found, epsilon * (1 + gamma), ep.norms.lp(flatten(delta), p=self.p, axis=-1) + distance_to_boundary)  # type: ignore
                )
            else:
                epsilon = ep.where(
                    is_adversarial,
                    ep.minimum(
                        ep.minimum(epsilon - 1, (epsilon * (1 - gamma)).astype(ep.arange(x, 1).dtype).astype(epsilon.dtype)),
                                   ep.norms.lp(flatten(best_delta), p=self.p, axis=-1)),  # type: ignore
                    ep.maximum(epsilon + 1, (epsilon * (1 + gamma)).astype(ep.arange(x, 1).dtype).astype(epsilon.dtype))
                    )
                )
                epsilon = ep.maximum(1, epsilon).astype(epsilon.dtype)
            epsilon = ep.minimum(epsilon, worst_norm)
            grad_: ep.Tensor = self.normalize(gradients) * stepsize
            delta = delta + grad_
            delta = self.project(x=x + delta, x0=x, epsilon=epsilon) - x
            delta = ep.clip(x + delta, min_bound, max_bound) - x
        x_adv = x + best_delta
        return restore_type(x_adv)

    def normalize(self, gradients: ep.Tensor) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=2)

    @abstractmethod
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: ep.Tensor) -> ep.Tensor:
        ...

    @abstractmethod
    def mid_points(self, x0: ep.Tensor, x1: ep.Tensor, epsilons: ep.Tensor, bounds: Tuple[float, float]) -> ep.Tensor:
        raise NotImplementedError


class L1FMNAttack(FMNAttackLp):
    """
    The L1 Fast Minimum Norm adversarial attack, in Lp norm. [Pintor21]_

    Args:
        steps: Number of iterations.
        max_stepsize: Initial stepsize for the gradient update.
        min_stepsize: Final stepsize for the gradient update. The
            stepsize will be reduced with a cosine annealing policy.
        gamma: Initial stepsize for the epsilon update. It will
            be updated with a cosine annealing reduction up to 0.001.
        init_attack: Optional initial attack. If an initial attack
            is specified (or initial points are provided in the run), the
            attack will first try to search for the boundary between the
            initial point and the points in a class that satisfies the
            adversarial criterion.
        binary_search_steps: Number of steps to use for the search
            from the adversarial points. If no initial attack or adversarial
            starting point is provided, this parameter will be ignored.
    """
    distance = l1  # type: Any
    p: int = 1
    dual: Union[int, float] = ep.inf

    def get_random_start(self, x0: ep.Tensor, epsilon: ep.Tensor) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r: ep.Tensor = uniform_l1_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: ep.Tensor) -> ep.Tensor:
        return x0 + project_onto_l1_ball(x - x0, epsilon)

    def mid_points(self, x0: ep.Tensor, x1: ep.Tensor, epsilons: ep.Tensor, bounds: Tuple[float, float]) -> ep.Tensor:
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        threshold: ep.Tensor = (bounds[1] - bounds[0]) * (1 - epsilons)
        mask: ep.Tensor = (x1 - x0).abs() > threshold
        new_x: ep.Tensor = ep.where(mask, x0 + (x1 - x0).sign() * ((x1 - x0).abs() - threshold), x0)
        return new_x


class L2FMNAttack(FMNAttackLp):
    """
    The L2 Fast Minimum Norm adversarial attack, in Lp norm. [Pintor21]_

    Args:
        steps: Number of iterations.
        max_stepsize: Initial stepsize for the gradient update.
        min_stepsize: Final stepsize for the gradient update. The
            stepsize will be reduced with a cosine annealing policy.
        gamma: Initial stepsize for the epsilon update. It will
            be updated with a cosine annealing reduction up to 0.001.
        init_attack: Optional initial attack. If an initial attack
            is specified (or initial points are provided in the run), the
            attack will first try to search for the boundary between the
            initial point and the points in a class that satisfies the
            adversarial criterion.
        binary_search_steps: Number of steps to use for the search
            from the adversarial points. If no initial attack or adversarial
            starting point is provided, this parameter will be ignored.
    """
    distance = l2  # type: Any
    p: int = 2
    dual: int = 2

    def get_random_start(self, x0: ep.Tensor, epsilon: ep.Tensor) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r: ep.Tensor = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: ep.Tensor) -> ep.Tensor:
        norms: ep.Tensor = flatten(x).norms.l2(axis=-1)
        norms = ep.maximum(norms, 1e-12)
        # The factor here is computed to scale the perturbation; note that the current code
        # uses norms/norms so factor==1. This is a placeholder for the projection logic.
        factor: ep.Tensor = ep.minimum(1, norms / norms)
        factor = atleast_kd(factor, x.ndim)
        return x0 + (x - x0) * factor

    def mid_points(self, x0: ep.Tensor, x1: ep.Tensor, epsilons: ep.Tensor, bounds: Tuple[float, float]) -> ep.Tensor:
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        return epsilons * x1 + (1 - epsilons) * x0


class LInfFMNAttack(FMNAttackLp):
    """
    The L-infinity Fast Minimum Norm adversarial attack, in Lp norm. [Pintor21]_

    Args:
        steps: Number of iterations.
        max_stepsize: Initial stepsize for the gradient update.
        min_stepsize: Final stepsize for the gradient update. The
            stepsize will be reduced with a cosine annealing policy.
        gamma: Initial stepsize for the epsilon update. It will
            be updated with a cosine annealing reduction up to 0.001.
        init_attack: Optional initial attack. If an initial attack
            is specified (or initial points are provided in the run), the
            attack will first try to search for the boundary between the
            initial point and the points in a class that satisfies the
            adversarial criterion.
        binary_search_steps: Number of steps to use for the search
            from the adversarial points. If no initial attack or adversarial
            starting point is provided, this parameter will be ignored.
    """
    distance = linf  # type: Any
    p: Any = ep.inf
    dual: int = 1

    def get_random_start(self, x0: ep.Tensor, epsilon: ep.Tensor) -> ep.Tensor:
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: ep.Tensor) -> ep.Tensor:
        clipped = ep.maximum(flatten(x - x0).T, -epsilon)
        clipped = ep.minimum(clipped, epsilon).T
        return x0 + clipped.reshape(x0.shape)

    def mid_points(self, x0: ep.Tensor, x1: ep.Tensor, epsilons: ep.Tensor, bounds: Tuple[float, float]) -> ep.Tensor:
        delta: ep.Tensor = x1 - x0
        min_bound, max_bound = bounds
        s: float = max_bound - min_bound
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        clipped_delta: ep.Tensor = ep.where(delta < -epsilons * s, -epsilons * s, delta)
        clipped_delta = ep.where(clipped_delta > epsilons * s, epsilons * s, clipped_delta)
        return x0 + clipped_delta


class L0FMNAttack(FMNAttackLp):
    """
    The L0 Fast Minimum Norm adversarial attack, in Lp norm. [Pintor21]_

    Args:
        steps: Number of iterations.
        max_stepsize: Initial stepsize for the gradient update.
        min_stepsize: Final stepsize for the gradient update. The
            stepsize will be reduced with a cosine annealing policy.
        gamma: Initial stepsize for the epsilon update. It will
            be updated with a cosine annealing reduction up to 0.001.
        init_attack: Optional initial attack. If an initial attack
            is specified (or initial points are provided in the run), the
            attack will first try to search for the boundary between the
            initial point and the points in a class that satisfies the
            adversarial criterion.
        binary_search_steps: Number of steps to use for the search
            from the adversarial points. If no initial attack or adversarial
            starting point is provided, this parameter will be ignored.
    """
    distance = l0  # type: Any
    p: int = 0

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: ep.Tensor) -> ep.Tensor:
        flatten_delta: ep.Tensor = flatten(x - x0)
        n, d = flatten_delta.shape
        abs_delta: ep.Tensor = abs(flatten_delta)
        epsilon = epsilon.astype(ep.arange(x, 1).dtype)
        rows = range(n)
        idx_sorted: ep.Tensor = ep.flip(ep.argsort(abs_delta, axis=1), -1)[rows, epsilon]
        thresholds: ep.Tensor = (ep.ones_like(flatten_delta).T * abs_delta[rows, idx_sorted]).T
        clipped: ep.Tensor = ep.where(abs_delta >= thresholds, flatten_delta, 0)
        return x0 + clipped.reshape(x0.shape).astype(x0.dtype)

    def mid_points(self, x0: ep.Tensor, x1: ep.Tensor, epsilons: ep.Tensor, bounds: Tuple[float, float]) -> ep.Tensor:
        n_features: ep.Tensor = flatten(ep.ones_like(x0)).bool().sum(axis=1).float32()
        new_x: ep.Tensor = self.project(x1, x0, n_features * epsilons)
        return new_x
