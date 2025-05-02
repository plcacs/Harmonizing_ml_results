from typing import Union, Any, Optional, Callable, Tuple
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

    def __init__(self, x):
        pass

    @abstractmethod
    def __call__(self, gradient):
        pass

class AdamOptimizer(Optimizer):

    def __init__(self, x, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = ep.zeros_like(x)
        self.v = ep.zeros_like(x)
        self.t = 0

    def __call__(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        bias_correction_1 = 1 - self.beta1 ** self.t
        bias_correction_2 = 1 - self.beta2 ** self.t
        m_hat = self.m / bias_correction_1
        v_hat = self.v / bias_correction_2
        return self.stepsize * m_hat / (ep.sqrt(v_hat) + self.epsilon)

class GDOptimizer(Optimizer):

    def __init__(self, x, stepsize):
        self.stepsize = stepsize

    def __call__(self, gradient):
        return self.stepsize * gradient

class BaseGradientDescent(FixedEpsilonAttack, ABC):

    def __init__(self, *, rel_stepsize, abs_stepsize=None, steps, random_start):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start

    def get_loss_fn(self, model, labels):

        def loss_fn(inputs):
            logits = model(inputs)
            return ep.crossentropy(logits, labels).sum()
        return loss_fn

    def get_optimizer(self, x, stepsize):
        return GDOptimizer(x, stepsize)

    def value_and_grad(self, loss_fn, x):
        return ep.value_and_grad(loss_fn, x)

    def run(self, model, inputs, criterion, *, epsilon, **kwargs):
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
            classes = criterion_.target_classes
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
    def get_random_start(self, x0, epsilon):
        ...

    @abstractmethod
    def normalize(self, gradients, *, x, bounds):
        ...

    @abstractmethod
    def project(self, x, x0, epsilon):
        ...

def clip_lp_norms(x, *, norm, p):
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)
    factor = ep.minimum(1, norm / norms)
    factor = atleast_kd(factor, x.ndim)
    return x * factor

def normalize_lp_norms(x, *, p):
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor

def uniform_l1_n_balls(dummy, batch_size, n):
    u = ep.uniform(dummy, (batch_size, n))
    v = u.sort(axis=-1)
    vp = ep.concatenate([ep.zeros(v, (batch_size, 1)), v[:, :n - 1]], axis=-1)
    assert v.shape == vp.shape
    x = v - vp
    sign = ep.uniform(dummy, (batch_size, n), low=-1.0, high=1.0).sign()
    return sign * x

def uniform_l2_n_spheres(dummy, batch_size, n):
    x = ep.normal(dummy, (batch_size, n + 1))
    r = x.norms.l2(axis=-1, keepdims=True)
    s = x / r
    return s

def uniform_l2_n_balls(dummy, batch_size, n):
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

    def get_random_start(self, x0, epsilon):
        batch_size, n = flatten(x0).shape
        r = uniform_l1_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(self, gradients, *, x, bounds):
        return normalize_lp_norms(gradients, p=1)

    def project(self, x, x0, epsilon):
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=1)

class L2BaseGradientDescent(BaseGradientDescent):
    distance = l2

    def get_random_start(self, x0, epsilon):
        batch_size, n = flatten(x0).shape
        r = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(self, gradients, *, x, bounds):
        return normalize_lp_norms(gradients, p=2)

    def project(self, x, x0, epsilon):
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=2)

class LinfBaseGradientDescent(BaseGradientDescent):
    distance = linf

    def get_random_start(self, x0, epsilon):
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

    def normalize(self, gradients, *, x, bounds):
        return gradients.sign()

    def project(self, x, x0, epsilon):
        return x0 + ep.clip(x - x0, -epsilon, epsilon)