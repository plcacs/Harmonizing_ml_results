import logging
from typing import Union, Any, Optional, Callable, List
from typing_extensions import Literal
import math
import eagerpy as ep
import numpy as np
from foolbox.attacks import LinearSearchBlendedUniformNoiseAttack
from foolbox.tensorboard import TensorBoard
from ..models import Model
from ..criteria import Criterion
from ..distances import l1
from ..devutils import atleast_kd, flatten
from .base import MinimizationAttack, get_is_adversarial
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from .base import verify_input_bounds
from ..distances import l2, linf

class HopSkipJumpAttack(MinimizationAttack):
    distance: Callable[[ep.Tensor, ep.Tensor], ep.Tensor] = l1

    def __init__(self, init_attack: Optional[MinimizationAttack] = None, steps: int = 64, initial_gradient_eval_steps: int = 100, max_gradient_eval_steps: int = 10000, stepsize_search: Literal['geometric_progression', 'grid_search'] = 'geometric_progression', gamma: float = 1.0, tensorboard: Union[bool, str] = False, constraint: Literal['l2', 'linf'] = 'l2') -> None:
        ...

    def run(self, model: Model, inputs: ep.Tensor, criterion: Criterion, *, early_stop: Optional[Any] = None, starting_points: Optional[List[ep.Tensor]] = None, **kwargs: Any) -> ep.Tensor:
        ...

    def approximate_gradients(self, is_adversarial: Callable[[ep.Tensor], ep.Tensor], x_advs: ep.Tensor, steps: int, delta: ep.Tensor) -> ep.Tensor:
        ...

    def _project(self, originals: ep.Tensor, perturbed: ep.Tensor, epsilons: ep.Tensor) -> ep.Tensor:
        ...

    def _binary_search(self, is_adversarial: Callable[[ep.Tensor], ep.Tensor], originals: ep.Tensor, perturbed: ep.Tensor) -> ep.Tensor:
        ...

    def select_delta(self, originals: ep.Tensor, distances: ep.Tensor, step: int) -> ep.Tensor:
        ...
