from typing import Optional
from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent
from .gradient_descent_base import AdamOptimizer, Optimizer
import eagerpy as ep

class L1BasicIterativeAttack(L1BaseGradientDescent):
    """L1 Basic Iterative Method

    Args:
        rel_stepsize: Stepsize relative to epsilon.
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps.
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class L2BasicIterativeAttack(L2BaseGradientDescent):
    """L2 Basic Iterative Method

    Args:
        rel_stepsize: Stepsize relative to epsilon.
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps.
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class LinfBasicIterativeAttack(LinfBaseGradientDescent):
    """L-infinity Basic Iterative Method

    Args:
        rel_stepsize: Stepsize relative to epsilon.
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps.
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class L1AdamBasicIterativeAttack(L1BaseGradientDescent):
    """L1 Basic Iterative Method with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon.
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps.
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

class L2AdamBasicIterativeAttack(L2BaseGradientDescent):
    """L2 Basic Iterative Method with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon.
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps.
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

class LinfAdamBasicIterativeAttack(LinfBaseGradientDescent):
    """L-infinity Basic Iterative Method with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon.
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps.
        random_start : Controls whether to randomly start within allowed epsilon ball.
    """

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x, stepsize):
        return AdamOptimizer(x, stepsize, self.adam_beta1, self.adam_beta2, self.adam_epsilon)