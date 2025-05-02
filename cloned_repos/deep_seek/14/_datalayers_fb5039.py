import warnings
import functools
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from . import _layering
from ._layering import Int as Int
from . import data as _data
from .data import Data
from .core import Parameter
from . import discretization
from . import transforms as trans
from . import utils
from typing import Any, Optional, Tuple, TypeVar, Union, Dict, Sequence

D = tp.TypeVar('D', bound=Data)
Op = tp.TypeVar('Op', bound='Operation')
BL = tp.TypeVar('BL', bound='BoundLayer')

class Operation(_layering.Layered, _layering.Filterable):
    _LAYER_LEVEL: _layering.Level = _layering.Level.OPERATION
    _LEGACY: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if any((isinstance(x, Parameter) for x in args + tuple(kwargs.values()))):
            raise errors.NevergradTypeError('Operation with Parameter instances are not supported')

class BoundLayer(Operation):
    _LAYER_LEVEL: _layering.Level = _layering.Level.OPERATION

    def __init__(
        self,
        lower: Optional[Union[float, np.ndarray]] = None,
        upper: Optional[Union[float, np.ndarray]] = None,
        uniform_sampling: Optional[bool] = None
    ) -> None:
        super().__init__(lower, upper, uniform_sampling)
        self.bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = tuple((None if a is None else trans.bound_to_array(a) for a in (lower, upper)))
        both_bounds: bool = all((b is not None for b in self.bounds))
        self.uniform_sampling: Optional[bool] = uniform_sampling
        if uniform_sampling is None:
            self.uniform_sampling = both_bounds
        if self.uniform_sampling and (not both_bounds):
            raise errors.NevergradValueError('Cannot use full range sampling if both bounds are not set')
        if not (lower is None or upper is None):
            if (self.bounds[0] >= self.bounds[1]).any():
                raise errors.NevergradValueError(f'Lower bounds {lower} should be strictly smaller than upper bounds {upper}')

    def _normalizer(self) -> trans.Affine:
        if any((b is None for b in self.bounds)):
            raise RuntimeError('Cannot use normalized value for not-fully bounded Parameter')
        return trans.Affine(self.bounds[1] - self.bounds[0], self.bounds[0]).reverted()

    def __call__(self, data: D, inplace: bool = False) -> D:
        new: D = data if inplace else data.copy()
        value: Any = new.value
        new.add_layer(self.copy())
        try:
            new.value = value
        except ValueError as e:
            raise errors.NevergradValueError('Current value is not within bounds, please update it first') from e
        if all((x is not None for x in self.bounds)):
            tests: List[D] = [data.copy() for _ in range(2)]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for test, bound in zip(tests, self.bounds):
                    val: Any = bound * np.ones(value.shape) if isinstance(value, np.ndarray) else bound[0]
                    test.value = val
            state: np.ndarray = tests[0].get_standardized_data(reference=tests[1])
            min_dist: float = np.min(np.abs(state))
            if min_dist < 3.0:
                warnings.warn(f'Bounds are {min_dist} sigma away from each other at the closest, you should aim for at least 3 for better quality.', errors.NevergradRuntimeWarning)
        return new

    def _layered_sample(self) -> 'Data':
        if not self.uniform_sampling:
            return super()._layered_sample()
        root: Any = self._layers[0]
        if not isinstance(root, Data):
            raise errors.NevergradTypeError(f'BoundLayer {self} on a non-Data root {root}')
        shape: Tuple[int, ...] = super()._layered_get_value().shape
        child: Data = root.spawn_child()
        new_val: np.ndarray = self.random_state.uniform(size=shape)
        del child.value
        child._layers[self._layer_index].set_normalized_value(new_val)
        return child

    def set_normalized_value(self, value: np.ndarray) -> None:
        new_val: np.ndarray = self._normalizer().backward(value)
        self._layers[self._layer_index]._layered_set_value(new_val)

    def get_normalized_value(self) -> np.ndarray:
        value: np.ndarray = self._layers[self._layer_index]._layered_get_value()
        return self._normalizer().forward(value)

    def _check(self, value: np.ndarray) -> None:
        if not utils.BoundChecker(*self.bounds)(value):
            raise errors.NevergradValueError('New value does not comply with bounds')

class Modulo(BoundLayer):
    def __init__(self, module: Union[np.ndarray, np.float64, np.int_, float, int]) -> None:
        super().__init__(lower=0, upper=module)
        if not isinstance(module, (np.ndarray, np.float64, np.int_, float, int)):
            raise TypeError(f'Unsupported type {type(module)} for module')
        self._module: Union[np.ndarray, np.float64, np.int_, float, int] = module

    def _layered_get_value(self) -> np.ndarray:
        return super()._layered_get_value() % self._module

    def _layered_set_value(self, value: np.ndarray) -> None:
        self._check(value)
        current: np.ndarray = super()._layered_get_value()
        super()._layered_set_value(current - current % self._module + value)

class ForwardableOperation(Operation):
    def _layered_get_value(self) -> np.ndarray:
        return self.forward(super()._layered_get_value())

    def _layered_set_value(self, value: np.ndarray) -> None:
        super()._layered_set_value(self.backward(value))

    def forward(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Exponent(ForwardableOperation):
    def __init__(self, base: float) -> None:
        super().__init__(base)
        if base <= 0:
            raise errors.NevergradValueError('Exponent must be strictly positive')
        self._base: float = base
        self._name: str = f'exp={base:.2f}'

    def forward(self, value: np.ndarray) -> np.ndarray:
        return self._base ** value

    def backward(self, value: np.ndarray) -> np.ndarray:
        return np.log(value) / np.log(self._base)

class Power(ForwardableOperation):
    def __init__(self, power: float) -> None:
        super().__init__(power)
        self._power: float = power

    def forward(self, value: np.ndarray) -> np.ndarray:
        return value ** self._power

    def backward(self, value: np.ndarray) -> np.ndarray:
        return value ** (1.0 / self._power)

class Add(ForwardableOperation):
    def __init__(self, offset: np.ndarray) -> None:
        super().__init__(offset)
        self._offset: np.ndarray = offset

    def forward(self, value: np.ndarray) -> np.ndarray:
        return self._offset + value

    def backward(self, value: np.ndarray) -> np.ndarray:
        return value - self._offset

class Multiply(ForwardableOperation):
    def __init__(self, value: np.ndarray) -> None:
        super().__init__(value)
        self._mult: np.ndarray = value
        self.name: str = f'Mult({value})'

    def forward(self, value: np.ndarray) -> np.ndarray:
        return self._mult * value

    def backward(self, value: np.ndarray) -> np.ndarray:
        return value / self._mult

class Bound(BoundLayer):
    def __init__(
        self,
        lower: Optional[Union[float, np.ndarray]] = None,
        upper: Optional[Union[float, np.ndarray]] = None,
        method: str = 'bouncing',
        uniform_sampling: Optional[bool] = None
    ) -> None:
        super().__init__(lower=lower, upper=upper, uniform_sampling=uniform_sampling)
        transforms: Dict[str, Any] = dict(clipping=trans.Clipping, arctan=trans.ArctanBound, tanh=trans.TanhBound, gaussian=trans.CumulativeDensity)
        transforms['bouncing'] = functools.partial(trans.Clipping, bounce=True)
        if method not in transforms:
            raise errors.NevergradValueError(f'Unknown method {method}, available are: {transforms.keys()}\nSee docstring for more help.')
        self._method: str = method
        self._transform: Any = transforms[method](*self.bounds)
        self.set_name(self._transform.name)

    def _layered_get_value(self) -> np.ndarray:
        deep_value: np.ndarray = super()._layered_get_value()
        value: np.ndarray = self._transform.forward(deep_value)
        if deep_value is not value and self._method in ('clipping', 'bouncing'):
            super()._layered_set_value(value)
        return value

    def _layered_set_value(self, value: np.ndarray) -> None:
        super()._layered_set_value(self._transform.backward(value))

class SoftmaxSampling(Int):
    def __init__(self, arity: int, deterministic: bool = False) -> None:
        super().__init__(deterministic=deterministic)
        self.arity: int = arity
        self.ordered: bool = False

    def _get_name(self) -> str:
        tag: str = '{det}' if self.deterministic else ''
        return self.__class__.__name__ + tag

    def _layered_get_value(self) -> np.ndarray:
        if self._cache is None:
            value: np.ndarray = _layering.Layered._layered_get_value(self)
            if value.ndim != 2 or value.shape[1] != self.arity:
                raise ValueError(f'Dimension 1 should be the arity {self.arity}')
            encoder: discretization.Encoder = discretization.Encoder(value, rng=self.random_state)
            self._cache: np.ndarray = encoder.encode(deterministic=self.deterministic)
        return self._cache

    def _layered_set_value(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray) and (not value.dtype == int):
            raise TypeError('Expected an integer array, got {value}')
        if self.arity is None:
            raise RuntimeError('Arity is not initialized')
        self._cache: np.ndarray = value
        out: np.ndarray = np.zeros((value.size, self.arity), dtype=float)
        coeff: float = discretization.weight_for_reset(self.arity)
        out[np.arange(value.size, dtype=int), value] = coeff
        super()._layered_set_value(out)

class AngleOp(Operation):
    def _layered_get_value(self) -> np.ndarray:
        x: np.ndarray = super()._layered_get_value()
        if x.shape[0] != 2:
            raise ValueError(f'First dimension should be 2, got {x.shape}')
        return np.angle(x[0, ...] + 1j * x[1, ...])

    def _layered_set_value(self, value: np.ndarray) -> None:
        out: np.ndarray = np.stack([fn(value) for fn in (np.cos, np.sin)], axis=0)
        super()._layered_set_value(out)

def Angles(
    init: Optional[np.ndarray] = None,
    shape: Optional[Sequence[int]] = None,
    deg: bool = False,
    bound_method: Optional[str] = None
) -> _data.Array:
    if sum((x is None for x in (init, shape))) != 1:
        raise ValueError('Exactly 1 of init or shape must be provided')
    out_shape: Tuple[int, ...] = tuple(shape) if shape is not None else np.array(init).shape
    ang: _data.Array = _data.Array(shape=(2,) + out_shape)
    if bound_method is not None:
        Bound(-2, 2, method=bound_method)(ang, inplace=True)
    ang.add_layer(AngleOp())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=errors.NevergradRuntimeWarning)
        Bound(-np.pi, np.pi)(ang, inplace=True)
    if deg:
        ang = ang * (180 / np.pi)
    if init is not None:
        ang.value = init
    return ang
