import uuid
import itertools
import numpy as np
from scipy import stats
import nevergrad.common.typing as tp
from . import utils
from typing import Optional, Union, Sequence, Tuple, Any, List, Dict, Callable

def bound_to_array(x):
    """Updates type of bounds to use arrays"""
    if isinstance(x, (tuple, list, np.ndarray)):
        return np.asarray(x)
    else:
        return np.array([x], dtype=float)

class Transform:
    """Base class for transforms implementing a forward and a backward (inverse)
    method.
    This provide a default representation, and a short representation should be implemented
    for each transform.
    """

    def __init__(self):
        self.name: str = uuid.uuid4().hex

    def forward(self, x):
        raise NotImplementedError

    def backward(self, y):
        raise NotImplementedError

    def reverted(self):
        return Reverted(self)

    def __repr__(self):
        args = ', '.join((f'{x}={y}' for x, y in sorted(self.__dict__.items()) if not x.startswith('_')))
        return f'{self.__class__.__name__}({args})'

class Reverted(Transform):
    """Inverse of a transform.

    Parameters
    ----------
    transform: Transform
    """

    def __init__(self, transform):
        super().__init__()
        self.transform: Transform = transform
        self.name: str = f'Rv({self.transform.name})'

    def forward(self, x):
        return self.transform.backward(x)

    def backward(self, y):
        return self.transform.forward(y)

class Affine(Transform):
    """Affine transform a * x + b

    Parameters
    ----------
    a: float
    b: float
    """

    def __init__(self, a, b):
        super().__init__()
        self.a: np.ndarray = bound_to_array(a)
        self.b: np.ndarray = bound_to_array(b)
        if not np.any(self.a):
            raise ValueError('"a" parameter should be non-zero to prevent information loss.')
        self.name: str = f'Af({a},{b})'

    def forward(self, x):
        return self.a * x + self.b

    def backward(self, y):
        return (y - self.b) / self.a

class Exponentiate(Transform):
    """Exponentiation transform base ** (coeff * x)
    This can for instance be used for to get a logarithmicly distruted values 10**(-[1, 2, 3]).

    Parameters
    ----------
    base: float
    coeff: float
    """

    def __init__(self, base=10.0, coeff=1.0):
        super().__init__()
        self.base: float = base
        self.coeff: float = coeff
        self.name: str = f'Ex({self.base},{self.coeff})'

    def forward(self, x):
        return self.base ** (float(self.coeff) * x)

    def backward(self, y):
        return np.log(y) / (float(self.coeff) * np.log(self.base))
BoundType = Optional[Union[tp.ArrayLike, float]]

def _f(x):
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

class BoundTransform(Transform):

    def __init__(self, a_min=None, a_max=None):
        super().__init__()
        self.a_min: Optional[np.ndarray] = None
        self.a_max: Optional[np.ndarray] = None
        for name, value in [('a_min', a_min), ('a_max', a_max)]:
            if value is not None:
                isarray = isinstance(value, (tuple, list, np.ndarray))
                setattr(self, name, np.asarray(value) if isarray else np.array([value]))
        if not (self.a_min is None or self.a_max is None):
            if (self.a_min >= self.a_max).any():
                raise ValueError(f'Lower bounds {a_min} should be strictly smaller than upper bounds {a_max}')
        if self.a_min is None and self.a_max is None:
            raise ValueError('At least one bound must be specified')
        self.shape: Tuple[int, ...] = self.a_min.shape if self.a_min is not None else self.a_max.shape

    def _check_shape(self, x):
        for dims in itertools.zip_longest(x.shape, self.shape, fillvalue=1):
            if dims[0] != dims[1] and (not any((x == 1 for x in dims))):
                raise ValueError(f'Shapes do not match: {self.shape} and {x.shape}')

class TanhBound(BoundTransform):
    """Bounds all real values into [a_min, a_max] using a tanh transform.
    Beware, tanh goes very fast to its limits.

    Parameters
    ----------
    a_min: float
    a_max: float
    """

    def __init__(self, a_min, a_max):
        super().__init__(a_min=a_min, a_max=a_max)
        if self.a_min is None or self.a_max is None:
            raise ValueError('Both bounds must be specified')
        self._b: np.ndarray = 0.5 * (self.a_max + self.a_min)
        self._a: np.ndarray = 0.5 * (self.a_max - self.a_min)
        self.name: str = f'Th({_f(a_min)},{_f(a_max)})'

    def forward(self, x):
        self._check_shape(x)
        return self._b + self._a * np.tanh(x)

    def backward(self, y):
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

    def __init__(self, a_min=None, a_max=None, bounce=False):
        super().__init__(a_min=a_min, a_max=a_max)
        self._bounce: bool = bounce
        b: str = ',b' if bounce else ''
        self.name: str = f'Cl({_f(a_min)},{_f(a_max)}{b})'
        self.checker: utils.BoundChecker = utils.BoundChecker(self.a_min, self.a_max)

    def forward(self, x):
        self._check_shape(x)
        if self.checker(x):
            return x
        out: np.ndarray = np.clip(x, self.a_min, self.a_max)
        if self._bounce:
            out = np.clip(2 * out - x, self.a_min, self.a_max)
        return out

    def backward(self, y):
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

    def __init__(self, a_min, a_max):
        super().__init__(a_min=a_min, a_max=a_max)
        if self.a_min is None or self.a_max is None:
            raise ValueError('Both bounds must be specified')
        self._b: np.ndarray = 0.5 * (self.a_max + self.a_min)
        self._a: np.ndarray = (self.a_max - self.a_min) / np.pi
        self.name: str = f'At({_f(a_min)},{_f(a_max)})'

    def forward(self, x):
        self._check_shape(x)
        return self._b + self._a * np.arctan(x)

    def backward(self, y):
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

    def __init__(self, lower=0.0, upper=1.0, eps=1e-09, scale=1.0, density='gaussian'):
        super().__init__(a_min=lower, a_max=upper)
        self._b: float = lower
        self._a: float = upper - lower
        self._eps: float = eps
        self._scale: float = scale
        self.name: str = f'Cd({_f(lower)},{_f(upper)})'
        if density not in ('gaussian', 'cauchy'):
            raise ValueError('Unknown density')
        if density == 'gaussian':
            self._forw: Callable[[np.ndarray], np.ndarray] = stats.norm.cdf
            self._back: Callable[[np.ndarray], np.ndarray] = stats.norm.ppf
        else:
            self._forw: Callable[[np.ndarray], np.ndarray] = stats.cauchy.cdf
            self._back: Callable[[np.ndarray], np.ndarray] = stats.cauchy.ppf

    def forward(self, x):
        return self._a * self._forw(x / self._scale) + self._b

    def backward(self, y):
        if (y > self.a_max).any() or (y < self.a_min).any():
            raise ValueError(f'Only data between {self.a_min} and {self.a_max} can be transformed back.\nGot: {y}')
        y = np.clip((y - self._b) / self._a, self._eps, 1 - self._eps)
        return self._scale * self._back(y)

class Fourrier(Transform):

    def __init__(self, axes=0):
        super().__init__()
        self.axes: Tuple[int, ...] = (axes,) if isinstance(axes, int) else tuple(axes)
        self.name: str = f'F({axes})'

    def forward(self, x):
        if any((x.shape[a] % 2 for a in self.axes)):
            raise ValueError(f'Only even shapes are allowed for Fourrier transform, got {x.shape}')
        return np.fft.rfftn(x, axes=self.axes, norm='ortho')

    def backward(self, y):
        return np.fft.irfftn(y, axes=self.axes, norm='ortho')