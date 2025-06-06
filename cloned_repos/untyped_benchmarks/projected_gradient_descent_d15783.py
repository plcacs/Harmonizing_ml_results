from typing import Optional
from .gradient_descent_base import L1BaseGradientDescent, AdamOptimizer, Optimizer
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent
import eagerpy as ep

class L1ProjectedGradientDescentAttack(L1BaseGradientDescent):
    """L1 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(self, *, rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class L2ProjectedGradientDescentAttack(L2BaseGradientDescent):
    """L2 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(self, *, rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class LinfProjectedGradientDescentAttack(LinfBaseGradientDescent):
    """Linf Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon (defaults to 0.01 / 0.3).
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(self, *, rel_stepsize=0.01 / 0.3, abs_stepsize=None, steps=40, random_start=True):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)

class L1AdamProjectedGradientDescentAttack(L1ProjectedGradientDescentAttack):
    """L1 Projected Gradient Descent with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
        adam_beta1 : beta_1 parameter of Adam optimizer
        adam_beta2 : beta_2 parameter of Adam optimizer
        adam_epsilon : epsilon parameter of Adam optimizer responsible for numerical stability
    """

    def __init__(self, *, rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x, stepsize):
        return AdamOptimizer(x, stepsize, self.adam_beta1, self.adam_beta2, self.adam_epsilon)

class L2AdamProjectedGradientDescentAttack(L2ProjectedGradientDescentAttack):
    """L2 Projected Gradient Descent with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
        adam_beta1 : beta_1 parameter of Adam optimizer
        adam_beta2 : beta_2 parameter of Adam optimizer
        adam_epsilon : epsilon parameter of Adam optimizer responsible for numerical stability
    """

    def __init__(self, *, rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x, stepsize):
        return AdamOptimizer(x, stepsize, self.adam_beta1, self.adam_beta2, self.adam_epsilon)

class LinfAdamProjectedGradientDescentAttack(LinfProjectedGradientDescentAttack):
    """Linf Projected Gradient Descent with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
        adam_beta1 : beta_1 parameter of Adam optimizer
        adam_beta2 : beta_2 parameter of Adam optimizer
        adam_epsilon : epsilon parameter of Adam optimizer responsible for numerical stability
    """

    def __init__(self, *, rel_stepsize=0.025, abs_stepsize=None, steps=50, random_start=True, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x, stepsize):
        return AdamOptimizer(x, stepsize, self.adam_beta1, self.adam_beta2, self.adam_epsilon)