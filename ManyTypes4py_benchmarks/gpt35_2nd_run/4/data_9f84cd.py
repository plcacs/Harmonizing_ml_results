import numpy as np
from nevergrad.common.typing import TypeVar
from . import _layering
from . import core
from .container import Dict
from . import utils

D = TypeVar('D', bound='Data')
P = TypeVar('P', bound=core.Parameter)

def _param_string(parameters: Dict) -> str:
    """Hacky helper for nice-visualization"""
    substr = f'[{parameters._get_parameters_str()}]'
    if substr == '[]':
        substr = ''
    return substr

class Data(core.Parameter):
    """Array/scalar parameter

    Parameters
    ----------
    init: np.ndarray, or None
        initial value of the array (defaults to 0, with a provided shape)
    shape: tuple of ints, or None
        shape of the array, to be provided iff init is not provided
    lower: array, float or None
        minimum value
    upper: array, float or None
        maximum value
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Note
    ----
    - More specific behaviors can be obtained throught the following methods:
      set_bounds, set_mutation, set_integer_casting
    - if both lower and upper bounds are provided, sigma will be adapted so that the range spans 6 sigma.
      Also, if init is not provided, it will be set to the middle value.
    """

    def __init__(self, *, init=None, shape=None, lower=None, upper=None, mutable_sigma=False) -> None:
        super().__init__()
        sigma = np.array([1.0])
        if isinstance(lower, (list, tuple)):
            lower = np.array(lower, dtype=float)
        if isinstance(upper, (list, tuple)):
            upper = np.array(upper, dtype=float)
        if sum((x is None for x in [init, shape])) != 1:
            raise ValueError('Exactly one of "init" or "shape" must be provided')
        if init is not None:
            init = np.asarray(init, dtype=float)
        else:
            assert isinstance(shape, (list, tuple)) and all((isinstance(n, int) for n in shape)), f'Incorrect shape: {shape} (type: {type(shape)}).'
            init = np.zeros(shape, dtype=float)
            if lower is not None and upper is not None:
                init += (lower + upper) / 2.0
        self._value = init
        self.add_layer(_layering.ArrayCasting())
        num_bounds = sum((x is not None for x in (lower, upper)))
        layer = None
        if num_bounds:
            from . import _datalayers
            layer = _datalayers.Bound(lower=lower, upper=upper, uniform_sampling=init is None and num_bounds == 2)
            if num_bounds == 2:
                sigma = (layer.bounds[1] - layer.bounds[0]) / 6
        sigma = sigma[0] if sigma.size == 1 else sigma
        if mutable_sigma:
            siginit = sigma
            base = float(np.exp(1.0 / np.sqrt(2 * init.size)))
            sigma = base ** (Array if isinstance(sigma, np.ndarray) else Scalar)(init=siginit, mutable_sigma=False)
            sigma.value = siginit
        self.parameters = Dict(sigma=sigma)
        self.parameters._ignore_in_repr = dict(sigma='1.0')
        if layer is not None:
            layer(self, inplace=True)

    @property
    def bounds(self) -> tuple:
        """Estimate of the bounds (None if unbounded)

        Note
        ----
        This may be inaccurate (WIP)
        """

    @property
    def dimension(self) -> int:
        return int(np.prod(self._value.shape))

    def _get_name(self) -> str:

    @property
    def sigma(self) -> core.Parameter:

    def _layered_sample(self) -> 'Data':

    def set_bounds(self, lower=None, upper=None, method='bouncing', full_range_sampling=None) -> 'Data':

    def set_mutation(self, sigma=None, exponent=None) -> 'Data':

    def set_integer_casting(self) -> 'Data':

    @property
    def integer(self) -> bool:

    def _internal_set_standardized_data(self, data, reference):

    def _internal_get_standardized_data(self, reference):

    def _to_reduced_space(self, value=None):

    def _layered_recombine(self, *others):

    def copy(self) -> 'Data':

    def _layered_set_value(self, value):

    def _layered_get_value(self):

    def _new_with_data_layer(self, name, *args, **kwargs):

    def __mod__(self, module):

    def __rpow__(self, base):

    def __add__(self, offset):

    def __sub__(self, offset):

    def __radd__(self, offset):

    def __mul__(self, value):

    def __rmul__(self, value):

    def __truediv__(self, value):

    def __rtruediv__(self, value):

    def __pow__(self, power):

    def __neg__(self):

def _fix_legacy(parameter: Data) -> None:

class Array(Data):
    value: core.ValueProperty

class Scalar(Data):
    value: core.ValueProperty

class Log(Scalar):

