from typing import Union, Any, Optional, Callable, Tuple, Type, TypeVar, cast
from abc import ABC, abstractmethod
import eagerpy as ep
from ..devutils import flatten
from ..devutils import atleast_kd
from ..types import Bounds
from ..models.base import Model
from ..criteria import Misclassification, TargetedMisclassification
from ..distances import l1, l2, linf
from .base import FixedEpsilonAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds

class Optimizer(ABC):

    def __init__(self, x: ep.Tensor) -> None:
        pass

    @abstractmethod
    def __call__(self, gradient: ep.Tensor) -> ep.Tensor:
        pass

class AdamOptimizer(Optimizer):

    def __init__(self, x: ep.Tensor, stepsize: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-08) -> None:
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: ep.Tensor = ep.zeros_like(x)
        self.v: ep.Tensor = ep.zeros_like(x)
        self.t: int = 0

    def __call__(self, gradient: ep.Tensor) -> ep.Tensor:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        bias_correction_1 = 1 - self.beta1 ** self.t
        bias_correction_2 = 1 - self.beta2 ** self.t
        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2
        return self.stepsize * m_hat / (ep.sqrt(v_hat) + self.epsilon)

class GDOptimizer(Optimizer):

    def __init__(self, x: ep.Tensor, stepsize: float) -> None:
        self.stepsize = stepsize

    def __call__(self, gradient: ep.Tensor) -> ep.Tensor:
        return self.stepsize * gradient

class BaseGradientDescent(FixedEpsilonAttack, ABC):

    def __init__(self, *, rel_stepsize: float, abs_stepsize: Optional[float] = None, steps: int, random_start: bool) -> None:
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start

    def get_loss_fn(self, model: Model, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:

        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.crossentropy(logits, labels).sum()
        return loss_fn

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return GDOptimizer(x, stepsize)

    def value_and_grad(self, loss_fn: Callable[[ep.Tensor], ep.Tensor], x: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
        return ep.value_and_grad(loss_fn, x)

    def run(self, model: Model, inputs: Any, criterion: Union[Misclassification, TargetedMisclassification, Any], *, epsilon: float, **kwargs: Any) -> Any:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs
        verify_input_bounds(x0, model)
        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, 'target_classes'):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError('unsupported criterion')
        loss_fn = self.get_loss_fn(model, classes)
        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize
        optimizer = self.get_optimizer(x0, stepsize)
        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0
        for _ in range(self.steps):
            _, gradients = self.value_and_grad(loss_fn, x)
            gradients = self.normalize(gradients, x=x, bounds=model.bounds)
            x = x + gradient_step_sign * optimizer(gradients)
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model.bounds)
        return restore_type(x)

    @abstractmethod
    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...

    @abstractmethod
    def normalize(self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds) -> ep.Tensor:
        ...

    @abstractmethod
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...

def clip_lp_norms(x: ep.Tensor, *, norm: float, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)
    factor = ep.minimum(1, norm / norms)
    factor = atleast_kd(factor, x.ndim)
    return x * factor

def normalize_lp_norms(x: ep.Tensor, *, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor

def uniform_l1_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    u = ep.uniform(dummy, (batch_size, n))
    v = u.sort(axis=-1)
    vp = ep.concatenate([ep.zeros(v, (batch_size, 1)), v[:, :n - 1]], axis=-1)
    assert v.shape == vp.shape
    x = v - vp
    sign = ep.uniform(dummy, (batch_size, n), low=-1.0, high=1.0).sign()
    return sign * x

def uniform_l2_n_spheres(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    x = ep.normal(dummy, (batch_size, n + 1))
    r = x.norms.l2(axis=-1, keepdims=True)
    s = x / r
    return s

def uniform_l2_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    """Sampling from the n-ball

    Implementation of the algorithm proposed by Voelker et al. [#Voel17]_

    References:
        .. [#Voel17] Voelker et al., 2017, Efficiently sampling vectors and coordinates
            from the n-sphere and n-ball
            http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    s = uniform_l2_n_spheres(dummy, batch_size, n + 1)
    b = s[:, :n]
    return b

class L1BaseGradientDescent(BaseGradientDescent):
    distance = l1

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l1_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=1)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=1)

class L2BaseGradientDescent(BaseGradientDescent):
    distance = l2

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=2)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=2)

class LinfBaseGradientDescent(BaseGradientDescent):
    distance = linf

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

    def normalize(self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds) -> ep.Tensor:
        return gradients.sign()

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.clip(x - x0, -epsilon, epsilon)
