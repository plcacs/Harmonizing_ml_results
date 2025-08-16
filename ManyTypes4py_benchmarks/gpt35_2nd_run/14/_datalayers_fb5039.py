import warnings
import functools
import numpy as np
from nevergrad.common.typing import TypeVar, Optional, bool
from nevergrad.common import errors
from . import _layering
from ._layering import Int
from . import data
from .data import Data
from .core import Parameter
from . import discretization
from . import transforms as trans
from . import utils

D = TypeVar('D', bound=Data)
Op = TypeVar('Op', bound='Operation')
BL = TypeVar('BL', bound='BoundLayer')

class Operation(_layering.Layered, _layering.Filterable):
    _LAYER_LEVEL: _layering.Level = _layering.Level.OPERATION
    _LEGACY: bool = False

    def __init__(self, *args: Parameter, **kwargs: Parameter) -> None:
        super().__init__()
        if any((isinstance(x, Parameter) for x in args + tuple(kwargs.values()))):
            raise errors.NevergradTypeError('Operation with Parameter instances are not supported')

class BoundLayer(Operation):
    _LAYER_LEVEL: _layering.Level = _layering.Level.OPERATION

    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None, uniform_sampling: Optional[bool] = None) -> None:
        ...

class Modulo(BoundLayer):
    def __init__(self, module: Union[np.ndarray, np.float64, np.int_, float, int]) -> None:
        ...

class ForwardableOperation(Operation):
    def _layered_get_value(self) -> np.ndarray:
        ...

    def _layered_set_value(self, value: np.ndarray) -> None:
        ...

    def forward(self, value: np.ndarray) -> np.ndarray:
        ...

    def backward(self, value: np.ndarray) -> np.ndarray:
        ...

class Exponent(ForwardableOperation):
    def __init__(self, base: float) -> None:
        ...

    def forward(self, value: np.ndarray) -> np.ndarray:
        ...

    def backward(self, value: np.ndarray) -> np.ndarray:
        ...

class Power(ForwardableOperation):
    def __init__(self, power: float) -> None:
        ...

    def forward(self, value: np.ndarray) -> np.ndarray:
        ...

    def backward(self, value: np.ndarray) -> np.ndarray:
        ...

class Add(ForwardableOperation):
    def __init__(self, offset: float) -> None:
        ...

    def forward(self, value: np.ndarray) -> np.ndarray:
        ...

    def backward(self, value: np.ndarray) -> np.ndarray:
        ...

class Multiply(ForwardableOperation):
    def __init__(self, value: float) -> None:
        ...

    def forward(self, value: np.ndarray) -> np.ndarray:
        ...

    def backward(self, value: np.ndarray) -> np.ndarray:
        ...

class Bound(BoundLayer):
    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None, method: str = 'bouncing', uniform_sampling: Optional[bool] = None) -> None:
        ...

class SoftmaxSampling(Int):
    def __init__(self, arity: int, deterministic: bool = False) -> None:
        ...

    def _get_name(self) -> str:
        ...

    def _layered_get_value(self) -> np.ndarray:
        ...

    def _layered_set_value(self, value: np.ndarray) -> None:
        ...

class AngleOp(Operation):
    def _layered_get_value(self) -> np.ndarray:
        ...

    def _layered_set_value(self, value: np.ndarray) -> None:
        ...

def Angles(init: Optional[np.ndarray] = None, shape: Optional[Sequence[int]] = None, deg: bool = False, bound_method: Optional[str] = None) -> Array:
    ...
