from typing import Union, Tuple, Any, Optional
import numpy as np
import eagerpy as ep
from ..types import Bounds
from ..models import Model
from ..criteria import Misclassification, TargetedMisclassification
from .base import MinimizationAttack, T, get_criterion, raise_if_kwargs, verify_input_bounds
from .gradient_descent_base import AdamOptimizer

class L2CarliniWagnerAttack(MinimizationAttack):
    distance: Callable[[ep.TensorType, ep.TensorType], ep.TensorType] = l2

    def __init__(self, binary_search_steps: int = 9, steps: int = 10000, stepsize: float = 0.01, confidence: float = 0, initial_const: float = 0.001, abort_early: bool = True) -> None:
        self.binary_search_steps: int = binary_search_steps
        self.steps: int = steps
        self.stepsize: float = stepsize
        self.confidence: float = confidence
        self.initial_const: float = initial_const
        self.abort_early: bool = abort_early

    def run(self, model: Model, inputs: ep.TensorType, criterion: Union[Misclassification, TargetedMisclassification], *, early_stop: Optional[Any] = None, **kwargs: Any) -> ep.TensorType:
        ...

def best_other_classes(logits: ep.TensorType, exclude: ep.TensorType) -> ep.TensorType:
    ...

def _to_attack_space(x: ep.TensorType, *, bounds: Bounds) -> ep.TensorType:
    ...

def _to_model_space(x: ep.TensorType, *, bounds: Bounds) -> ep.TensorType:
    ...
