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

ps = {l0: 0, l1: 1, l2: 2, linf: ep.inf, LpDistance: ep.nan}
duals = {l0: ep.nan, l1: ep.inf, l2: 2, linf: 1, LpDistance: ep.nan}

def best_other_classes(logits: T, exclude: T) -> T:
    other_logits = logits - ep.onehot_like(logits, exclude, value=ep.inf)
    return other_logits.argmax(axis=-1)

def project_onto_l1_ball(x: T, eps: T) -> T:
    original_shape = x.shape
    x = flatten(x)
    mask = (ep.norms.l1(x, axis=1) <= eps).astype(x.dtype).expand_dims(1)
    mu = ep.flip(ep.sort(ep.abs(x)), axis=-1).astype(x.dtype)
    cumsum = ep.cumsum(mu, axis=-1)
    arange = ep.arange(x, 1, x.shape[1] + 1).astype(x.dtype)
    rho = ep.max((mu * arange > cumsum - eps.expand_dims(1)).astype(x.dtype) * arange, axis=-1) - 1
    rho = ep.maximum(rho, 0)
    theta = (cumsum[ep.arange(x, x.shape[0]), rho.astype(ep.arange(x, 1).dtype)] - eps) / (rho + 1.0)
    proj = (ep.abs(x) - theta.expand_dims(1)).clip(min_=0, max_=ep.inf)
    x = mask * x + (1 - mask) * proj * ep.sign(x)
    return x.reshape(original_shape)

class FMNAttackLp(MinimizationAttack, ABC):
    def __init__(self, *, steps: int = 100, max_stepsize: float = 1.0, min_stepsize: Optional[float] = None, gamma: float = 0.05, init_attack: Optional[MinimizationAttack] = None, binary_search_steps: int = 10) -> None:
        self.steps = steps
        self.max_stepsize = max_stepsize
        self.init_attack = init_attack
        if min_stepsize is not None:
            self.min_stepsize = min_stepsize
        else:
            self.min_stepsize = max_stepsize / 100
        self.binary_search_steps = binary_search_steps
        self.gamma = gamma
        self.p = ps[self.distance]
        self.dual = duals[self.distance]

    def run(self, model: Model, inputs: T, criterion: Any, *, starting_points: Optional[T] = None, early_stop: Optional[Callable[[T], bool]] = None, **kwargs: Any) -> T:
        raise_if_kwargs(kwargs)
        criterion_ = get_criterion(criterion)
        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
        else:
            raise ValueError('unsupported criterion')

        def loss_fn(inputs: T, labels: T) -> Tuple[T, Tuple[T, T]]:
            logits = model(inputs)
            if targeted:
                c_minimize = best_other_classes(logits, labels)
                c_maximize = labels
            else:
                c_minimize = labels
                c_maximize = best_other_classes(logits, labels)
            loss = logits[rows, c_minimize] - logits[rows, c_maximize]
            return (-loss.sum(), (logits, loss))

        x, restore_type = ep.astensor_(inputs)
        del inputs, criterion, kwargs
        N = len(x)
        initialized = False
        if starting_points is not None:
            x1 = starting_points
            initialized = True
        elif self.init_attack is not None:
            x1 = self.init_attack.run(model, x, criterion_)
            initialized = True
        if initialized is True:
            is_adv = get_is_adversarial(criterion_, model)
            assert is_adv(x1).all()
            lower_bound = ep.zeros(x, shape=(N,))
            upper_bound = ep.ones(x, shape=(N,))
            for _ in range(self.binary_search_steps):
                epsilons = (lower_bound + upper_bound) / 2
                mid_points = self.mid_points(x, x1, epsilons, model.bounds)
                is_advs = is_adv(mid_points)
                lower_bound = ep.where(is_advs, lower_bound, epsilons)
                upper_bound = ep.where(is_advs, epsilons, upper_bound)
            starting_points = self.mid_points(x, x1, upper_bound, model.bounds)
            delta = starting_points - x
        else:
            delta = ep.zeros_like(x)
        if classes.shape != (N,):
            name = 'target_classes' if targeted else 'labels'
            raise ValueError(f'expected {name} to have shape ({N},), got {classes.shape}')
        min_, max_ = model.bounds
        rows = range(N)
        grad_and_logits = ep.value_and_grad_fn(x, loss_fn, has_aux=True)
        if self.p != 0:
            epsilon = ep.inf * ep.ones(x, len(x))
        else:
            epsilon = ep.maximum(ep.ones(x, len(x)), ep.norms.l0(flatten(delta), axis=-1))
        if self.p != 0:
            worst_norm = ep.norms.lp(flatten(ep.maximum(x - min_, max_ - x)), p=self.p, axis=-1)
        else:
            worst_norm = flatten(ep.ones_like(x)).bool().sum(axis=1).float32()
        best_lp = worst_norm
        best_delta = delta
        adv_found = ep.zeros(x, len(x)).bool()
        for i in range(self.steps):
            stepsize = self.min_stepsize + (self.max_stepsize - self.min_stepsize) * (1 + math.cos(math.pi * i / self.steps)) / 2
            gamma = 0.001 + (self.gamma - 0.001) * (1 + math.cos(math.pi * (i / self.steps))) / 2
            x_adv = x + delta
            loss, (logits, loss_batch), gradients = grad_and_logits(x_adv, classes)
            is_adversarial = criterion_(x_adv, logits)
            lp = ep.norms.lp(flatten(delta), p=self.p, axis=-1)
            is_smaller = lp <= best_lp
            is_both = ep.logical_and(is_adversarial, is_smaller)
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_lp = ep.where(is_both, lp, best_lp)
            best_delta = ep.where(atleast_kd(is_both, x.ndim), delta, best_delta)
            if self.p != 0:
                distance_to_boundary = abs(loss_batch) / ep.norms.lp(flatten(gradients), p=self.dual, axis=-1)
                epsilon = ep.where(is_adversarial, ep.minimum(epsilon * (1 - gamma), ep.norms.lp(flatten(best_delta), p=self.p, axis=-1)), ep.where(adv_found, epsilon * (1 + gamma), ep.norms.lp(flatten(delta), p=self.p, axis=-1) + distance_to_boundary))
            else:
                epsilon = ep.where(is_adversarial, ep.minimum(ep.minimum(epsilon - 1, (epsilon * (1 - gamma)).astype(ep.arange(x, 1).dtype).astype(epsilon.dtype)), ep.norms.lp(flatten(best_delta), p=self.p, axis=-1)), ep.maximum(epsilon + 1, (epsilon * (1 + gamma)).astype(ep.arange(x, 1).dtype).astype(epsilon.dtype)))
                epsilon = ep.maximum(1, epsilon).astype(epsilon.dtype)
            epsilon = ep.minimum(epsilon, worst_norm)
            grad_ = self.normalize(gradients) * stepsize
            delta = delta + grad_
            delta = self.project(x=x + delta, x0=x, epsilon=epsilon) - x
            delta = ep.clip(x + delta, *model.bounds) - x
        x_adv = x + best_delta
        return restore_type(x_adv)

    def normalize(self, gradients: T) -> T:
        return normalize_lp_norms(gradients, p=2)

    @abstractmethod
    def project(self, x: T, x0: T, epsilon: T) -> T:
        ...

    @abstractmethod
    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:
        raise NotImplementedError

class L1FMNAttack(FMNAttackLp):
    distance = l1
    p = 1
    dual = ep.inf

    def get_random_start(self, x0: T, epsilon: T) -> T:
        batch_size, n = flatten(x0).shape
        r = uniform_l1_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def project(self, x: T, x0: T, epsilon: T) -> T:
        return x0 + project_onto_l1_ball(x - x0, epsilon)

    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        threshold = (bounds[1] - bounds[0]) * (1 - epsilons)
        mask = (x1 - x0).abs() > threshold
        new_x = ep.where(mask, x0 + (x1 - x0).sign() * ((x1 - x0).abs() - threshold), x0)
        return new_x

class L2FMNAttack(FMNAttackLp):
    distance = l2
    p = 2
    dual = 2

    def get_random_start(self, x0: T, epsilon: T) -> T:
        batch_size, n = flatten(x0).shape
        r = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def project(self, x: T, x0: T, epsilon: T) -> T:
        norms = flatten(x).norms.l2(axis=-1)
        norms = ep.maximum(norms, 1e-12)
        factor = ep.minimum(1, norms / norms)
        factor = atleast_kd(factor, x.ndim)
        return x0 + (x - x0) * factor

    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        return epsilons * x1 + (1 - epsilons) * x0

class LInfFMNAttack(FMNAttackLp):
    distance = linf
    p = ep.inf
    dual = 1

    def get_random_start(self, x0: T, epsilon: T) -> T:
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

    def project(self, x: T, x0: T, epsilon: T) -> T:
        clipped = ep.maximum(flatten(x - x0).T, -epsilon)
        clipped = ep.minimum(clipped, epsilon).T
        return x0 + clipped.reshape(x0.shape)

    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:
        delta = x1 - x0
        min_, max_ = bounds
        s = max_ - min_
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))
        clipped_delta = ep.where(delta < -epsilons * s, -epsilons * s, delta)
        clipped_delta = ep.where(clipped_delta > epsilons * s, epsilons * s, clipped_delta)
        return x0 + clipped_delta

class L0FMNAttack(FMNAttackLp):
    distance = l0
    p = 0

    def project(self, x: T, x0: T, epsilon: T) -> T:
        flatten_delta = flatten(x - x0)
        n, d = flatten_delta.shape
        abs_delta = abs(flatten_delta)
        epsilon = epsilon.astype(ep.arange(x, 1).dtype)
        rows = range(n)
        idx_sorted = ep.flip(ep.argsort(abs_delta, axis=1), -1)[rows, epsilon]
        thresholds = (ep.ones_like(flatten_delta).T * abs_delta[rows, idx_sorted]).T
        clipped = ep.where(abs_delta >= thresholds, flatten_delta, 0)
        return x0 + clipped.reshape(x0.shape).astype(x0.dtype)

    def mid_points(self, x0: T, x1: T, epsilons: T, bounds: Tuple[T, T]) -> T:
        n_features = flatten(ep.ones_like(x0)).bool().sum(axis=1).float32()
        new_x = self.project(x1, x0, n_features * epsilons)
        return new_x
