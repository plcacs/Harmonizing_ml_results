#!/usr/bin/env python3
import uuid
import itertools
import numpy as np
from scipy import stats
import nevergrad.common.typing as tp
from . import utils

BoundType = tp.Optional[tp.Union[tp.ArrayLike, float]]


def bound_to_array(x: tp.Union[tp.ArrayLike, float]) -> np.ndarray:
    """Updates type of bounds to use arrays"""
    if isinstance(x, (tuple, list, np.ndarray)):
        return np.asarray(x)
    else:
        return np.array([x], dtype=float)


def _f(x: tp.Any) -> tp.Union[float, int, np.ndarray]:
    """Format for prints:
    array with one scalars are converted to floats
    """
    if isinstance(x, (np.ndarray, list, tuple)):
        x = np.asarray(x)
        if x.shape == (1,):
            x = float(x[0])
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return x


class Transform:
    """Base class for transforms implementing a forward and a backward (inverse)
    method.
    This provide a default representation, and a short representation should be implemented
    for each transform.
    """

    def __init__(self) -> None:
        self.name: str = uuid.uuid4().hex

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reverted(self) -> "Reverted":
        return Reverted(self)

    def __repr__(self) -> str:
        args = ', '.join((f'{x}={y}' for x, y in sorted(self.__dict__.items()) if not x.startswith('_')))
        return f'{self.__class__.__name__}({args})'


class Reverted(Transform):
    """Inverse of a transform.

    Parameters
    ----------
    transform: Transform
    """

    def __init__(self, transform: Transform) -> None:
        super().__init__()
        self.transform: Transform = transform
        self.name = f'Rv({self.transform.name})'

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.transform.backward(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return self.transform.forward(y)


class Affine(Transform):
    """Affine transform a * x + b

    Parameters
    ----------
    a: float or ArrayLike
    b: float or ArrayLike
    """

    def __init__(self, a: tp.Union[float, tp.ArrayLike], b: tp.Union[float, tp.ArrayLike]) -> None:
        super().__init__()
        self.a: np.ndarray = bound_to_array(a)
        self.b: np.ndarray = bound_to_array(b)
        if not np.any(self.a):
            raise ValueError('"a" parameter should be non-zero to prevent information loss.')
        self.name = f'Af({a},{b})'

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.a * x + self.b

    def backward(self, y: np.ndarray) -> np.ndarray:
        return (y - self.b) / self.a


class Exponentiate(Transform):
    """Exponentiation transform base ** (coeff * x)
    This can for instance be used for to get a logarithmicly distruted values 10**(-[1, 2, 3]).

    Parameters
    ----------
    base: float
    coeff: float
    """

    def __init__(self, base: float = 10.0, coeff: float = 1.0) -> None:
        super().__init__()
        self.base: float = base
        self.coeff: float = coeff
        self.name = f'Ex({self.base},{self.coeff})'

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.base ** (float(self.coeff) * x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.log(y) / (float(self.coeff) * np.log(self.base))


class BoundTransform(Transform):
    def __init__(self, a_min: BoundType = None, a_max: BoundType = None) -> None:
        super().__init__()
        self.a_min: tp.Optional[np.ndarray] = None
        self.a_max: tp.Optional[np.ndarray] = None
        for name, value in [('a_min', a_min), ('a_max', a_max)]:
            if value is not None:
                isarray = isinstance(value, (tuple, list, np.ndarray))
                setattr(self, name, np.asarray(value) if isarray else np.array([value]))
        if not (self.a_min is None or self.a_max is None):
            if (self.a_min >= self.a_max).any():
                raise ValueError(f'Lower bounds {a_min} should be strictly smaller than upper bounds {a_max}')
        if self.a_min is None and self.a_max is None:
            raise ValueError('At least one bound must be specified')
        self.shape: tp.Tuple[int, ...] = self.a_min.shape if self.a_min is not None else self.a_max.shape

    def _check_shape(self, x: np.ndarray) -> None:
        for dims in itertools.zip_longest(x.shape, self.shape, fillvalue=1):
            if dims[0] != dims[1] and (not any((dim == 1 for dim in dims))):
                raise ValueError(f'Shapes do not match: {self.shape} and {x.shape}')


class TanhBound(BoundTransform):
    """Bounds all real values into [a_min, a_max] using a tanh transform.
    Beware, tanh goes very fast to its limits.

    Parameters
    ----------
    a_min: float
    a_max: float
    """

    def __init__(self, a_min: float, a_max: float) -> None:
        super().__init__(a_min=a_min, a_max=a_max)
        if self.a_min is None or self.a_max is None:
            raise ValueError('Both bounds must be specified')
        self._b: np.ndarray = 0.5 * (self.a_max + self.a_min)
        self._a: np.ndarray = 0.5 * (self.a_max - self.a_min)
        self.name = f'Th({_f(a_min)},{_f(a_max)})'

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._check_shape(x)
        return self._b + self._a * np.tanh(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        self._check_shape(y)
        if (y > self.a_max).any() or (y < self.a_min).any():
            raise ValueError(f'Only data between {self.a_min} and {self.a_max} can be transformed back (bounds lead to infinity).')
        return np.arctanh((y - self._b) / self._a)


class Clipping(BoundTransform):
    """Bounds all real values into [a_min, a_max] using clipping (not bijective).

    Parameters
    ----------
    a_min: float or None
        lower bound
    a_max: float or None
        upper bound
    bounce: bool
        bounce (once) on borders instead of just clipping
    """

    def __init__(self, a_min: BoundType = None, a_max: BoundType = None, bounce: bool = False) -> None:
        super().__init__(a_min=a_min, a_max=a_max)
        self._bounce: bool = bounce
        b = ',b' if bounce else ''
        self.name = f'Cl({_f(a_min)},{_f(a_max)}{b})'
        self.checker: tp.Callable[[np.ndarray], bool] = utils.BoundChecker(self.a_min, self.a_max)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._check_shape(x)
        if self.checker(x):
            return x
        out: np.ndarray = np.clip(x, self.a_min, self.a_max)
        if self._bounce:
            out = np.clip(2 * out - x, self.a_min, self.a_max)
        return out

    def backward(self, y: np.ndarray) -> np.ndarray:
        self._check_shape(y)
        if not self.checker(y):
            raise ValueError(f'Only data between {self.a_min} and {self.a_max} can be transformed back.\nGot: {y}')
        return y


class ArctanBound(BoundTransform):
    """Bounds all real values into [a_min, a_max] using an arctan transform.
    This is a much softer approach compared to tanh.

    Parameters
    ----------
    a_min: float
    a_max: float
    """

    def __init__(self, a_min: float, a_max: float) -> None:
        super().__init__(a_min=a_min, a_max=a_max)
        if self.a_min is None or self.a_max is None:
            raise ValueError('Both bounds must be specified')
        self._b: np.ndarray = 0.5 * (self.a_max + self.a_min)
        self._a: np.ndarray = (self.a_max - self.a_min) / np.pi
        self.name = f'At({_f(a_min)},{_f(a_max)})'

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._check_shape(x)
        return self._b + self._a * np.arctan(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        self._check_shape(y)
        if (y > self.a_max).any() or (y < self.a_min).any():
            raise ValueError(f'Only data between {self.a_min} and {self.a_max} can be transformed back.')
        return np.tan((y - self._b) / self._a)


class CumulativeDensity(BoundTransform):
    """Bounds all real values into [0, 1] using a gaussian cumulative density function (cdf)
    Beware, cdf goes very fast to its limits.

    Parameters
    ----------
    lower: float
        lower bound
    upper: float
        upper bound
    eps: float
        small values to avoid hitting the bounds
    scale: float
        scaling factor of the density
    density: str
        either gaussian, or cauchy distributions
    """

    def __init__(self, lower: float = 0.0, upper: float = 1.0, eps: float = 1e-09, scale: float = 1.0, density: str = 'gaussian') -> None:
        super().__init__(a_min=lower, a_max=upper)
        self._b: float = lower
        self._a: float = upper - lower
        self._eps: float = eps
        self._scale: float = scale
        self.name = f'Cd({_f(lower)},{_f(upper)})'
        if density not in ('gaussian', 'cauchy'):
            raise ValueError('Unknown density')
        if density == 'gaussian':
            self._forw = stats.norm.cdf
            self._back = stats.norm.ppf
        else:
            self._forw = stats.cauchy.cdf
            self._back = stats.cauchy.ppf

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._a * self._forw(x / self._scale) + self._b

    def backward(self, y: np.ndarray) -> np.ndarray:
        if (y > self.a_max).any() or (y < self.a_min).any():
            raise ValueError(f'Only data between {self.a_min} and {self.a_max} can be transformed back.\nGot: {y}')
        y_clipped: np.ndarray = np.clip((y - self._b) / self._a, self._eps, 1 - self._eps)
        return self._scale * self._back(y_clipped)


class Fourrier(Transform):
    def __init__(self, axes: tp.Union[int, tp.Iterable[int]]) -> None:
        super().__init__()
        if isinstance(axes, int):
            self.axes: tp.Tuple[int, ...] = (axes,)
        else:
            self.axes = tuple(axes)
        self.name = f'F({axes})'

    def forward(self, x: np.ndarray) -> np.ndarray:
        if any((x.shape[a] % 2 for a in self.axes)):
            raise ValueError(f'Only even shapes are allowed for Fourrier transform, got {x.shape}')
        return np.fft.rfftn(x, axes=self.axes, norm='ortho')

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.fft.irfftn(y, axes=self.axes, norm='ortho')
