from typing import Callable, TypeVar, Any, Union, Optional, Sequence, List, Tuple, Dict, Iterable
from typing_extensions import final, overload
from abc import ABC, abstractmethod
import eagerpy as ep
from ..models import Model
from ..criteria import Criterion, Misclassification
from ..devutils import atleast_kd
from ..distances import Distance

T = TypeVar('T')
CriterionType = TypeVar('CriterionType', bound=Criterion)

class Attack(ABC):

    @overload
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    @overload
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def repeat(self, times: int) -> Any:
        ...

    def __repr__(self) -> str:
        ...

class AttackWithDistance(Attack):

    @property
    @abstractmethod
    def distance(self) -> Distance:
        ...

    def repeat(self, times: int) -> Any:
        ...

class Repeated(AttackWithDistance):

    def __init__(self, attack: Attack, times: int):
        ...

    @property
    def distance(self) -> Distance:
        ...

    @overload
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    @overload
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    def repeat(self, times: int) -> Any:
        ...

class FixedEpsilonAttack(AttackWithDistance):

    @abstractmethod
    def run(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilon: float, **kwargs: Any) -> Any:
        ...

    @overload
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    @overload
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

class MinimizationAttack(AttackWithDistance):

    @abstractmethod
    def run(self, model: Model, inputs: Any, criterion: CriterionType, *, early_stop: Optional[float], **kwargs: Any) -> Any:
        ...

    @overload
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    @overload
    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

    def __call__(self, model: Model, inputs: Any, criterion: CriterionType, *, epsilons: Union[float, List[float]], **kwargs: Any) -> Any:
        ...

class FlexibleDistanceMinimizationAttack(MinimizationAttack):

    def __init__(self, *, distance: Optional[Distance] = None):
        ...

    @property
    def distance(self) -> Distance:
        ...

def get_is_adversarial(criterion: Criterion, model: Model) -> Callable[[Any], Any]:
    ...

def get_criterion(criterion: Union[Criterion, Any]) -> Criterion:
    ...

def get_channel_axis(model: Model, ndim: int) -> Optional[int]:
    ...

def raise_if_kwargs(kwargs: Dict[str, Any]) -> None:
    ...

def verify_input_bounds(input: Any, model: Model) -> None:
    ...
