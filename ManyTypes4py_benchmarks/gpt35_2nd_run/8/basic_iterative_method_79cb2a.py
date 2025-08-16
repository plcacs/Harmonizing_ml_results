from typing import Optional
from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent
from .gradient_descent_base import AdamOptimizer, Optimizer
import eagerpy as ep

class L1BasicIterativeAttack(L1BaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.2, abs_stepsize: Optional[float] = None, steps: int = 10, random_start: bool = False) -> None:
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class L2BasicIterativeAttack(L2BaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.2, abs_stepsize: Optional[float] = None, steps: int = 10, random_start: bool = False) -> None:
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class LinfBasicIterativeAttack(LinfBaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.2, abs_stepsize: Optional[float] = None, steps: int = 10, random_start: bool = False) -> None:
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class L1AdamBasicIterativeAttack(L1BaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.2, abs_stepsize: Optional[float] = None, steps: int = 10, random_start: bool = False, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08) -> None:
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

class L2AdamBasicIterativeAttack(L2BaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.2, abs_stepsize: Optional[float] = None, steps: int = 10, random_start: bool = False, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08) -> None:
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

class LinfAdamBasicIterativeAttack(LinfBaseGradientDescent):
    def __init__(self, *, rel_stepsize: float = 0.2, abs_stepsize: Optional[float] = None, steps: int = 10, random_start: bool = False, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08) -> None:
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x, stepsize):
        return AdamOptimizer(x, stepsize, self.adam_beta1, self.adam_beta2, self.adam_epsilon)
