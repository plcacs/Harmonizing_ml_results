from .gradient_descent_base import L1BaseGradientDescent
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent
from ..models.base import Model
from ..criteria import Misclassification, TargetedMisclassification
from .base import T
from typing import Union, Any

class L1FastGradientAttack(L1BaseGradientDescent):
    def __init__(self, *, random_start: bool = False) -> None:
        super().__init__(rel_stepsize=1.0, steps=1, random_start=random_start)

    def run(self, model: Model, inputs: Any, criterion: Union[Misclassification, TargetedMisclassification], *, epsilon: float, **kwargs: Any) -> Any:
        if hasattr(criterion, 'target_classes'):
            raise ValueError('unsupported criterion')
        return super().run(model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs)

class L2FastGradientAttack(L2BaseGradientDescent):
    def __init__(self, *, random_start: bool = False) -> None:
        super().__init__(rel_stepsize=1.0, steps=1, random_start=random_start)

    def run(self, model: Model, inputs: Any, criterion: Union[Misclassification, TargetedMisclassification], *, epsilon: float, **kwargs: Any) -> Any:
        if hasattr(criterion, 'target_classes'):
            raise ValueError('unsupported criterion')
        return super().run(model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs)

class LinfFastGradientAttack(LinfBaseGradientDescent):
    def __init__(self, *, random_start: bool = False) -> None:
        super().__init__(rel_stepsize=1.0, steps=1, random_start=random_start)

    def run(self, model: Model, inputs: Any, criterion: Union[Misclassification, TargetedMisclassification], *, epsilon: float, **kwargs: Any) -> Any:
        if hasattr(criterion, 'target_classes'):
            raise ValueError('unsupported criterion')
        return super().run(model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs)
