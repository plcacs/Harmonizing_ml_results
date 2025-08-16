from typing import Optional
from .gradient_descent_base import L1BaseGradientDescent, AdamOptimizer, Optimizer
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent
import eagerpy as ep

class L1ProjectedGradientDescentAttack(L1BaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.025, abs_stepsize: Optional[float] = None, steps: int = 50, random_start: bool = True) -> None:

class L2ProjectedGradientDescentAttack(L2BaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.025, abs_stepsize: Optional[float] = None, steps: int = 50, random_start: bool = True) -> None:

class LinfProjectedGradientDescentAttack(LinfBaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.01 / 0.3, abs_stepsize: Optional[float] = None, steps: int = 40, random_start: bool = True) -> None:

class L1AdamProjectedGradientDescentAttack(L1ProjectedGradientDescentAttack):
    def __init__(self, *, rel_stepsize: float = 0.025, abs_stepsize: Optional[float] = None, steps: int = 50, random_start: bool = True, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08) -> None:
    def get_optimizer(self, x, stepsize) -> AdamOptimizer:

class L2AdamProjectedGradientDescentAttack(L2ProjectedGradientDescentAttack):
    def __init__(self, *, rel_stepsize: float = 0.025, abs_stepsize: Optional[float] = None, steps: int = 50, random_start: bool = True, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08) -> None:
    def get_optimizer(self, x, stepsize) -> AdamOptimizer:

class LinfAdamProjectedGradientDescentAttack(LinfProjectedGradientDescentAttack):
    def __init__(self, *, rel_stepsize: float = 0.025, abs_stepsize: Optional[float] = None, steps: int = 50, random_start: bool = True, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08) -> None:
    def get_optimizer(self, x, stepsize) -> AdamOptimizer:
