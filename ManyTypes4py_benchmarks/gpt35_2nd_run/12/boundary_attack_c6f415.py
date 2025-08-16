from typing import Union, Tuple, Optional, Any
from typing_extensions import Literal
import numpy as np
import eagerpy as ep
import logging
from ..devutils import flatten
from ..devutils import atleast_kd
from ..types import Bounds
from ..models import Model
from ..criteria import Criterion
from ..distances import l2
from ..tensorboard import TensorBoard
from .blended_noise import LinearSearchBlendedUniformNoiseAttack
from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import get_is_adversarial
from .base import raise_if_kwargs
from .base import verify_input_bounds

class BoundaryAttack(MinimizationAttack):
    distance: Any = l2

    def __init__(self, init_attack: Optional[MinimizationAttack] = None, steps: int = 25000, spherical_step: float = 0.01, source_step: float = 0.01, source_step_convergance: float = 1e-07, step_adaptation: float = 1.5, tensorboard: Union[bool, None] = False, update_stats_every_k: int = 10):
        ...

    def run(self, model: Model, inputs: Any, criterion: Criterion, *, early_stop: Optional[Any] = None, starting_points: Optional[Any] = None, **kwargs: Any) -> Any:
        ...

class ArrayQueue:
    def __init__(self, maxlen: int, N: int):
        ...

    @property
    def maxlen(self) -> int:
        ...

    @property
    def N(self) -> int:
        ...

    def append(self, x: Any) -> None:
        ...

    def clear(self, dims: Any) -> None:
        ...

    def mean(self) -> Any:
        ...

    def isfull(self) -> Any:
        ...
