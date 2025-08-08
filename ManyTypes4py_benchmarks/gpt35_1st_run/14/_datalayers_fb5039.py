import warnings
import functools
import numpy as np
from nevergrad.common.typing import TypeVar, Optional, Tuple, Union
from nevergrad.common import errors
from . import _layering
from ._layering import Int
from . import data as _data
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

    def _normalizer(self) -> trans.Affine:
        ...

    def __call__(self, data: Data, inplace: bool = False) -> Data:
        ...

    def _layered_sample(self) -> Data:
        ...

    def set_normalized_value(self, value: np.ndarray) -> None:
        ...

    def get_normalized_value(self) -> np.ndarray:
        ...

    def _check(self, value: np.ndarray) -> None:
        ...

class Modulo(BoundLayer):
    ...

class ForwardableOperation(Operation):
    ...

class Exponent(ForwardableOperation):
    ...

class Power(ForwardableOperation):
    ...

class Add(ForwardableOperation):
    ...

class Multiply(ForwardableOperation):
    ...

class Bound(BoundLayer):
    ...

class SoftmaxSampling(Int):
    ...

class AngleOp(Operation):
    ...

def Angles(init: Optional[Union[np.ndarray, float]], shape: Optional[Tuple[int, ...]], deg: bool = False, bound_method: Optional[str] = None) -> Data:
    ...
