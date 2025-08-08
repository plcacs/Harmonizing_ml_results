from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep
from ..devutils import flatten, atleast_kd
from ..types import Bounds
from ..models.base import Model
from ..criteria import Misclassification, TargetedMisclassification
from ..distances import l1, l2, linf
from .base import FixedEpsilonAttack, T, get_criterion, raise_if_kwargs, verify_input_bounds

class Optimizer(ABC):

    def __init__(self, x: ep.Tensor) -> None:
        pass

    @abstractmethod
    def __call__(self, gradient: ep.Tensor) -> ep.Tensor:

class AdamOptimizer(Optimizer):

    def __init__(self, x: ep.Tensor, stepsize: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-08) -> None:
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = ep.zeros_like(x)
        self.v = ep.zeros_like(x)
        self.t = 0

    def __call__(self, gradient: ep.Tensor) -> ep.Tensor:

class GDOptimizer(Optimizer):

    def __init__(self, x: ep.Tensor, stepsize: float) -> None:
        self.stepsize = stepsize

    def __call__(self, gradient: ep.Tensor) -> ep.Tensor:

class BaseGradientDescent(FixedEpsilonAttack, ABC):

    def __init__(self, *, rel_stepsize: float, abs_stepsize: Optional[float] = None, steps: int, random_start: bool) -> None:

    def get_loss_fn(self, model: Model, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> GDOptimizer:

    def value_and_grad(self, loss_fn: Callable[[ep.Tensor], ep.Tensor], x: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:

    def run(self, model: Model, inputs: ep.Tensor, criterion: Union[Misclassification, TargetedMisclassification], *, epsilon: float, **kwargs: Any) -> ep.Tensor:

    @abstractmethod
    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:

    @abstractmethod
    def normalize(self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds) -> ep.Tensor:

    @abstractmethod
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:

def clip_lp_norms(x: ep.Tensor, *, norm: float, p: float) -> ep.Tensor:

def normalize_lp_norms(x: ep.Tensor, *, p: float) -> ep.Tensor:

def uniform_l1_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:

def uniform_l2_n_spheres(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:

def uniform_l2_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:

class L1BaseGradientDescent(BaseGradientDescent):
    distance: Callable = l1

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:

    def normalize(self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds) -> ep.Tensor:

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:

class L2BaseGradientDescent(BaseGradientDescent):
    distance: Callable = l2

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:

    def normalize(self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds) -> ep.Tensor:

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:

class LinfBaseGradientDescent(BaseGradientDescent):
    distance: Callable = linf

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:

    def normalize(self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds) -> ep.Tensor:

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
