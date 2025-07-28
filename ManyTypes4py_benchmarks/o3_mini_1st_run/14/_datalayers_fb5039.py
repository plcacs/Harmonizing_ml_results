#!/usr/bin/env python3
import warnings
import functools
from typing import Any, Optional, Union, Sequence
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from . import _layering
from ._layering import Int
from . import data as _data
from .data import Data
from .core import Parameter
from . import discretization
from . import transforms as trans
from . import utils

D = tp.TypeVar("D", bound=Data)
Op = tp.TypeVar("Op", bound="Operation")
BL = tp.TypeVar("BL", bound="BoundLayer")


class Operation(_layering.Layered, _layering.Filterable):
    _LAYER_LEVEL = _layering.Level.OPERATION
    _LEGACY = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if any((isinstance(x, Parameter) for x in args + tuple(kwargs.values()))):
            raise errors.NevergradTypeError("Operation with Parameter instances are not supported")


class BoundLayer(Operation):
    _LAYER_LEVEL = _layering.Level.OPERATION

    def __init__(
        self,
        lower: Optional[Union[float, np.ndarray]] = None,
        upper: Optional[Union[float, np.ndarray]] = None,
        uniform_sampling: Optional[bool] = None,
    ) -> None:
        """Bounds all real values into [lower, upper]"""
        super().__init__(lower, upper, uniform_sampling)
        self.bounds = tuple(
            (None if a is None else trans.bound_to_array(a) for a in (lower, upper))
        )
        both_bounds = all((b is not None for b in self.bounds))
        self.uniform_sampling = uniform_sampling
        if uniform_sampling is None:
            self.uniform_sampling = both_bounds
        if self.uniform_sampling and (not both_bounds):
            raise errors.NevergradValueError(
                "Cannot use full range sampling if both bounds are not set"
            )
        if not (lower is None or upper is None):
            if (self.bounds[0] >= self.bounds[1]).any():
                raise errors.NevergradValueError(
                    f"Lower bounds {lower} should be strictly smaller than upper bounds {upper}"
                )

    def _normalizer(self) -> trans.Affine:
        if any((b is None for b in self.bounds)):
            raise RuntimeError("Cannot use normalized value for not-fully bounded Parameter")
        return trans.Affine(self.bounds[1] - self.bounds[0], self.bounds[0]).reverted()

    def __call__(self, data: Data, inplace: bool = False) -> Data:
        """Creates a new Data instance with bounds"""
        new = data if inplace else data.copy()
        value = new.value
        new.add_layer(self.copy())
        try:
            new.value = value
        except ValueError as e:
            raise errors.NevergradValueError("Current value is not within bounds, please update it first") from e
        if all((x is not None for x in self.bounds)):
            tests = [data.copy() for _ in range(2)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for test, bound in zip(tests, self.bounds):
                    val = bound * np.ones(value.shape) if isinstance(value, np.ndarray) else bound[0]
                    test.value = val
            state = tests[0].get_standardized_data(reference=tests[1])
            min_dist = np.min(np.abs(state))
            if min_dist < 3.0:
                warnings.warn(
                    f"Bounds are {min_dist} sigma away from each other at the closest, you should aim for at least 3 for better quality.",
                    errors.NevergradRuntimeWarning,
                )
        return new

    def _layered_sample(self) -> Data:
        if not self.uniform_sampling:
            return super()._layered_sample()
        root = self._layers[0]
        if not isinstance(root, Data):
            raise errors.NevergradTypeError(f"BoundLayer {self} on a non-Data root {root}")
        shape = super()._layered_get_value().shape
        child = root.spawn_child()
        new_val = self.random_state.uniform(size=shape)
        del child.value
        child._layers[self._layer_index].set_normalized_value(new_val)
        return child

    def set_normalized_value(self, value: np.ndarray) -> None:
        """Sets a value normalized between 0 and 1"""
        new_val = self._normalizer().backward(value)
        self._layers[self._layer_index]._layered_set_value(new_val)

    def get_normalized_value(self) -> np.ndarray:
        """Gets a value normalized between 0 and 1"""
        value = self._layers[self._layer_index]._layered_get_value()
        return self._normalizer().forward(value)

    def _check(self, value: np.ndarray) -> None:
        if not utils.BoundChecker(*self.bounds)(value):
            raise errors.NevergradValueError("New value does not comply with bounds")


class Modulo(BoundLayer):
    """Applies a modulo operation on the array
    CAUTION: WIP
    """

    def __init__(self, module: Union[np.ndarray, np.float64, np.int_, float, int]) -> None:
        super().__init__(lower=0, upper=module)
        if not isinstance(module, (np.ndarray, np.float64, np.int_, float, int)):
            raise TypeError(f"Unsupported type {type(module)} for module")
        self._module = module

    def _layered_get_value(self) -> np.ndarray:
        value = super()._layered_get_value()
        return value % self._module

    def _layered_set_value(self, value: np.ndarray) -> None:
        self._check(value)
        current = super()._layered_get_value()
        super()._layered_set_value(current - current % self._module + value)


class ForwardableOperation(Operation):
    """Operation with simple forward and backward methods (simplifies chaining)"""

    def _layered_get_value(self) -> np.ndarray:
        return self.forward(super()._layered_get_value())

    def _layered_set_value(self, value: np.ndarray) -> None:
        super()._layered_set_value(self.backward(value))

    def forward(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Exponent(ForwardableOperation):
    """Applies an array as exponent of a float"""

    def __init__(self, base: float) -> None:
        super().__init__(base)
        if base <= 0:
            raise errors.NevergradValueError("Exponent must be strictly positive")
        self._base = base
        self._name = f"exp={base:.2f}"

    def forward(self, value: np.ndarray) -> np.ndarray:
        return self._base ** value

    def backward(self, value: np.ndarray) -> np.ndarray:
        return np.log(value) / np.log(self._base)


class Power(ForwardableOperation):
    """Applies a float as exponent of a Data parameter"""

    def __init__(self, power: float) -> None:
        super().__init__(power)
        self._power = power

    def forward(self, value: np.ndarray) -> np.ndarray:
        return value ** self._power

    def backward(self, value: np.ndarray) -> np.ndarray:
        return value ** (1.0 / self._power)


class Add(ForwardableOperation):
    """Applies an offset to the value"""

    def __init__(self, offset: float) -> None:
        super().__init__(offset)
        self._offset = offset

    def forward(self, value: np.ndarray) -> np.ndarray:
        return self._offset + value

    def backward(self, value: np.ndarray) -> np.ndarray:
        return value - self._offset


class Multiply(ForwardableOperation):
    """Multiplies the value by a constant"""

    def __init__(self, value: float) -> None:
        super().__init__(value)
        self._mult = value
        self.name = f"Mult({value})"

    def forward(self, value: np.ndarray) -> np.ndarray:
        return self._mult * value

    def backward(self, value: np.ndarray) -> np.ndarray:
        return value / self._mult


class Bound(BoundLayer):
    def __init__(
        self,
        lower: Optional[Union[float, np.ndarray]] = None,
        upper: Optional[Union[float, np.ndarray]] = None,
        method: str = "bouncing",
        uniform_sampling: Optional[bool] = None,
    ) -> None:
        """Bounds all real values into [lower, upper] using a provided method"""
        super().__init__(lower=lower, upper=upper, uniform_sampling=uniform_sampling)
        transforms = dict(
            clipping=trans.Clipping,
            arctan=trans.ArctanBound,
            tanh=trans.TanhBound,
            gaussian=trans.CumulativeDensity,
        )
        transforms["bouncing"] = functools.partial(trans.Clipping, bounce=True)
        if method not in transforms:
            raise errors.NevergradValueError(
                f"Unknown method {method}, available are: {list(transforms.keys())}\nSee docstring for more help."
            )
        self._method = method
        self._transform = transforms[method](*self.bounds)
        self.set_name(self._transform.name)

    def _layered_get_value(self) -> np.ndarray:
        deep_value = super()._layered_get_value()
        value = self._transform.forward(deep_value)
        if deep_value is not value and self._method in ("clipping", "bouncing"):
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
        tag = "{det}" if self.deterministic else ""
        return self.__class__.__name__ + tag

    def _layered_get_value(self) -> np.ndarray:
        if self._cache is None:
            value = _layering.Layered._layered_get_value(self)
            if value.ndim != 2 or value.shape[1] != self.arity:
                raise ValueError(f"Dimension 1 should be the arity {self.arity}")
            encoder = discretization.Encoder(value, rng=self.random_state)
            self._cache = encoder.encode(deterministic=self.deterministic)
        return self._cache

    def _layered_set_value(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray) and (not value.dtype == int):
            raise TypeError(f"Expected an integer array, got {value}")
        if self.arity is None:
            raise RuntimeError("Arity is not initialized")
        self._cache = value
        out = np.zeros((value.size, self.arity), dtype=float)
        coeff = discretization.weight_for_reset(self.arity)
        out[np.arange(value.size, dtype=int), value] = coeff
        super()._layered_set_value(out)


class AngleOp(Operation):
    """Converts to and from angles from -pi to pi"""

    def _layered_get_value(self) -> np.ndarray:
        x = super()._layered_get_value()
        if x.shape[0] != 2:
            raise ValueError(f"First dimension should be 2, got {x.shape}")
        return np.angle(x[0, ...] + 1j * x[1, ...])

    def _layered_set_value(self, value: np.ndarray) -> None:
        out = np.stack([fn(value) for fn in (np.cos, np.sin)], axis=0)
        super()._layered_set_value(out)


def Angles(
    init: Optional[Union[np.ndarray, Sequence[float]]] = None,
    shape: Optional[Sequence[int]] = None,
    deg: bool = False,
    bound_method: Optional[str] = None,
) -> Data:
    """Creates an Array parameter representing an angle from -pi to pi (deg=False)
    or -180 to 180 (deg=True).
    Internally, this keeps track of coordinates which are transformed to an angle.
    """
    if sum((x is None for x in (init, shape))) != 1:
        raise ValueError("Exactly 1 of init or shape must be provided")
    out_shape = tuple(shape) if shape is not None else np.array(init).shape
    ang = _data.Array(shape=(2,) + out_shape)
    if bound_method is not None:
        Bound(-2, 2, method=bound_method)(ang, inplace=True)
    ang.add_layer(AngleOp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=errors.NevergradRuntimeWarning)
        Bound(-np.pi, np.pi)(ang, inplace=True)
    if deg:
        ang = ang * (180 / np.pi)
    if init is not None:
        ang.value = init
    return ang
