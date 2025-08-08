from typing import Union, Optional, Tuple, Any, Callable
from typing_extensions import Literal
import eagerpy as ep
import logging
from abc import ABC
from abc import abstractmethod
from ..devutils import flatten
from ..devutils import atleast_kd
from ..models import Model
from ..criteria import Criterion
from ..distances import l2, linf
from .base import MinimizationAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds

class DeepFoolAttack(MinimizationAttack, ABC):
    def __init__(self, *, steps: int = 50, candidates: Optional[int] = 10, overshoot: float = 0.02, loss: Literal['logits', 'crossentropy'] = 'logits') -> None:
        self.steps: int = steps
        self.candidates: Optional[int] = candidates
        self.overshoot: float = overshoot
        self.loss: Literal['logits', 'crossentropy'] = loss

    def _get_loss_fn(self, model: Model, classes: ep.Tensor) -> Callable[[ep.Tensor, int], Tuple[float, Tuple[ep.Tensor, ep.Tensor]]]:
        ...

    def run(self, model: Model, inputs: ep.Tensor, criterion: Union[Criterion, Callable[[ep.Tensor, ep.Tensor], ep.Tensor]], *, early_stop: Optional[bool] = None) -> ep.Tensor:
        ...

    @abstractmethod
    def get_distances(self, losses: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        ...

    @abstractmethod
    def get_perturbations(self, distances: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:

class L2DeepFoolAttack(DeepFoolAttack):
    distance = l2

    def get_distances(self, losses: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        ...

    def get_perturbations(self, distances: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:

class LinfDeepFoolAttack(DeepFoolAttack):
    distance = linf

    def get_distances(self, losses: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
        ...

    def get_perturbations(self, distances: ep.Tensor, grads: ep.Tensor) -> ep.Tensor:
