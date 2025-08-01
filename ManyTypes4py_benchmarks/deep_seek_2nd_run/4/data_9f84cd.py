import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from . import _layering
from . import core
from .container import Dict
from . import utils
from typing import Any, Optional, Tuple, Union, List, TypeVar, Generic, cast

D = TypeVar('D', bound='Data')
P = TypeVar('P', bound=core.Parameter)

def _param_string(parameters: core.Parameter) -> str:
    """Hacky helper for nice-visualizatioon"""
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

    def __init__(
        self,
        *,
        init: Optional[Union[np.ndarray, List[float], Tuple[float, ...]]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        lower: Optional[Union[np.ndarray, float]] = None,
        upper: Optional[Union[np.ndarray, float]] = None,
        mutable_sigma: bool = False
    ) -> None:
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
        self._value: np.ndarray = init
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
    def bounds(self) -> Tuple[Optional[Union[np.ndarray, float]], Optional[Union[np.ndarray, float]]]:
        """Estimate of the bounds (None if unbounded)

        Note
        ----
        This may be inaccurate (WIP)
        """
        from . import _datalayers
        bound_layers = _datalayers.BoundLayer.filter_from(self)
        if not bound_layers:
            return (None, None)
        bounds = bound_layers[-1].bounds
        forwardable = _datalayers.ForwardableOperation.filter_from(self)
        forwardable = [x for x in forwardable if x._layer_index > bound_layers[-1]._layer_index]
        for f in forwardable:
            bounds = tuple((None if b is None else f.forward(b) for b in bounds))
        return bounds

    @property
    def dimension(self) -> int:
        return int(np.prod(self._value.shape))

    def _get_name(self) -> str:
        cls = self.__class__.__name__
        descriptors = []
        if self._value.shape != (1,):
            descriptors.append(str(self._value.shape).replace(' ', ''))
        descriptors += [layer.name for layer in self._layers[1:] if not isinstance(layer, (_layering.ArrayCasting, _layering._ScalarCasting))]
        description = '' if not descriptors else '{{{}}}'.format(','.join(descriptors))
        return f'{cls}{description}' + _param_string(self.parameters)

    @property
    def sigma(self) -> core.Parameter:
        """Value for the standard deviation used to mutate the parameter"""
        return self.parameters['sigma']

    def _layered_sample(self) -> 'Data':
        child = self.spawn_child()
        from . import helpers
        with helpers.deterministic_sampling(child):
            child.mutate()
        return child

    def set_bounds(
        self,
        lower: Optional[Union[np.ndarray, float]] = None,
        upper: Optional[Union[np.ndarray, float]] = None,
        method: str = 'bouncing',
        full_range_sampling: Optional[bool] = None
    ) -> 'Data':
        """Bounds all real values into [lower, upper] using a provided method

        Parameters
        ----------
        lower: array, float or None
            minimum value
        upper: array, float or None
            maximum value
        method: str
            One of the following choices:

            - "bouncing": bounce on border (at most once). This is a variant of clipping,
               avoiding bounds over-samping (default).
            - "clipping": clips the values inside the bounds. This is efficient but leads
              to over-sampling on the bounds.
            - "constraint": adds a constraint (see register_cheap_constraint) which leads to rejecting mutations
              reaching beyond the bounds. This avoids oversampling the boundaries, but can be inefficient in large
              dimension.
            - "arctan": maps the space [lower, upper] to to all [-inf, inf] using arctan transform. This is efficient
              but it completely reshapes the space (a mutation in the center of the space will be larger than a mutation
              close to the bounds), and reaching the bounds is equivalent to reaching the infinity.
            - "tanh": same as "arctan", but with a "tanh" transform. "tanh" saturating much faster than "arctan", it can lead
              to unexpected behaviors.
        full_range_sampling: Optional bool
            Changes the default behavior of the "sample" method (aka creating a child and mutating it from the current instance)
            or the sampling optimizers, to creating a child with a value sampled uniformly (or log-uniformly) within
            the while range of the bounds. The "sample" method is used by some algorithms to create an initial population.
            This is activated by default if both bounds are provided.

        Notes
        -----
        - "tanh" reaches the boundaries really quickly, while "arctan" is much softer
        - only "clipping" accepts partial bounds (None values)
        """
        from . import _datalayers
        value = self.value
        if method == 'constraint':
            layer = _datalayers.BoundLayer(lower=lower, upper=upper, uniform_sampling=full_range_sampling)
            checker = utils.BoundChecker(*layer.bounds)
            self.register_cheap_constraint(checker)
        else:
            layer = _datalayers.Bound(lower=lower, upper=upper, method=method, uniform_sampling=full_range_sampling)
        layer._LEGACY = True
        layer(self, inplace=True)
        _fix_legacy(self)
        try:
            self.value = value
        except ValueError as e:
            raise errors.NevergradValueError('Current value is not within bounds, please update it first') from e
        return self

    def set_mutation(
        self,
        sigma: Optional[Union[core.Parameter, float]] = None,
        exponent: Optional[float] = None
    ) -> 'Data':
        """Output will be cast to integer(s) through deterministic rounding.

        Parameters
        ----------
        sigma: Array/Log or float
            The standard deviation of the mutation. If a Parameter is provided, it will replace the current
            value. If a float is provided, it will either replace a previous float value, or update the value
            of the Parameter.
        exponent: float
            exponent for the logarithmic mode. With the default sigma=1, using exponent=2 will perform
            x2 or /2 "on average" on the value at each mutation.

        Returns
        -------
        self
        """
        if sigma is not None:
            if isinstance(sigma, core.Parameter) or isinstance(self.parameters._content['sigma'], core.Constant):
                self.parameters._content['sigma'] = core.as_parameter(sigma)
            else:
                self.sigma.value = sigma
        if exponent is not None:
            from . import _datalayers
            if exponent <= 0.0:
                raise ValueError('Only exponents strictly higher than 0.0 are allowed')
            value = self.value
            layer = _datalayers.Exponent(base=exponent)
            layer._LEGACY = True
            self.add_layer(layer)
            _fix_legacy(self)
            try:
                self.value = value
            except ValueError as e:
                raise errors.NevergradValueError('Cannot convert to logarithmic mode with current non-positive value, please update it firstp.') from e
        return self

    def set_integer_casting(self) -> 'Data':
        """Output will be cast to integer(s) through deterministic rounding.

        Returns
        -------
        self

        Note
        ----
        Using integer casting makes the parameter discrete which can make the optimization more
        difficult. It is especially ill-advised to use this with a range smaller than 10, or
        a sigma lower than 1. In those cases, you should rather use a TransitionChoice instead.
        """
        return self.add_layer(_layering.Int())

    @property
    def integer(self) -> bool:
        return any((isinstance(x, _layering.Int) for x in self._layers))

    def _internal_set_standardized_data(self, data: np.ndarray, reference: 'Data') -> None:
        assert isinstance(data, np.ndarray)
        sigma = reference.sigma.value
        data_reduc = sigma * (data + reference._to_reduced_space()).reshape(reference._value.shape)
        self._value = data_reduc
        self.value

    def _internal_get_standardized_data(self, reference: 'Data') -> np.ndarray:
        return reference._to_reduced_space(self._value - reference._value)

    def _to_reduced_space(self, value: Optional[np.ndarray] = None) -> np.ndarray:
        """Converts array with appropriate shapes to reduced (uncentered) space
        by applying log scaling and sigma scaling
        """
        if value is None:
            value = self._value
        reduced = value / self.sigma.value
        return reduced.ravel()

    def _layered_recombine(self, *others: 'Data') -> None:
        all_params = [self] + list(others)
        all_arrays = [p.get_standardized_data(reference=self) for p in all_params]
        mean = np.mean(all_arrays, axis=0)
        self.set_standardized_data(mean)

    def copy(self) -> 'Data':
        child = super().copy()
        child._value = np.array(self._value, copy=True)
        return child

    def _layered_set_value(self, value: np.ndarray) -> None:
        self._check_frozen()
        if self._value.shape != value.shape:
            raise ValueError(f'Cannot set array of shape {self._value.shape} with value of shape {value.shape}')
        self._value = value

    def _layered_get_value(self) -> np.ndarray:
        return self._value

    def _new_with_data_layer(self, name: str, *args: Any, **kwargs: Any) -> 'Data':
        from . import _datalayers
        new = self.copy()
        new.add_layer(getattr(_datalayers, name)(*args, **kwargs))
        return new

    def __mod__(self, module: float) -> 'Data':
        return self._new_with_data_layer('Modulo', module)

    def __rpow__(self, base: float) -> 'Data':
        return self._new_with_data_layer('Exponent', base)

    def __add__(self, offset: float) -> 'Data':
        return self._new_with_data_layer('Add', offset)

    def __sub__(self, offset: float) -> 'Data':
        return self.__add__(-offset)

    def __radd__(self, offset: float) -> 'Data':
        return self.__add__(offset)

    def __mul__(self, value: float) -> 'Data':
        return self._new_with_data_layer('Multiply', value)

    def __rmul__(self, value: float) -> 'Data':
        return self.__mul__(value)

    def __truediv__(self, value: float) -> 'Data':
        return self.__mul__(1.0 / value)

    def __rtruediv__(self, value: float) -> 'Data':
        return value * self ** (-1)

    def __pow__(self, power: float) -> 'Data':
        return self._new_with_data_layer('Power', power)

    def __neg__(self) -> 'Data':
        return self.__mul__(-1.0)

def _fix_legacy(parameter: Data) -> None:
    """Ugly hack for keeping the legacy behaviors with considers bounds always after the exponent
    and can still sample "before" the exponent (log-uniform).
    """
    from . import _datalayers
    legacy = [x for x in _datalayers.Operation.filter_from(parameter) if x._LEGACY]
    if len(legacy) < 2:
        return
    if len(legacy) > 2:
        raise errors.NevergradRuntimeError('More than 2 legacy layers, this should not happen, open an issue')
    value = parameter.value
    layers_inds = tuple((leg._layer_index for leg in legacy))
    if abs(layers_inds[0] - layers_inds[1]) > 1:
        raise errors.NevergradRuntimeError('Non-legacy layers between 2 legacy layers')
    parameter._layers = [x for x in parameter._layers if x._layer_index not in layers_inds]
    for k, sub in enumerate(parameter._layers):
        sub._layer_index = k
        sub._layers = parameter._layers
    parameter.value = value
    bound_ind = int(isinstance(legacy[0], _datalayers.Exponent))
    bound = legacy[bound_ind]
    exp = legacy[(bound_ind + 1) % 2]
    bound.bounds = tuple((None if b is None else exp.backward(b) for b in bound.bounds))
    if isinstance(bound, _datalayers.Bound):
        bound = _datalayers.Bound(lower=bound.bounds[0], upper=bound.bounds[1], method=bound._method, uniform_sampling=bound.uniform_sampling)
    for lay in (bound, exp):
        lay._layer_index = 0
        lay._layers = [lay]
        parameter.add_layer(lay)
    return

class Array(Data):
    value = core.ValueProperty()

class Scalar(Data):
    """Parameter representing a scalar.

    Parameters
    ----------
    init: optional float
        initial value of the scalar (defaults to 0.0 if both bounds are not provided)
    lower: optional float
        minimum value if any
    upper: optional float
        maximum value if any
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Notes
    -----
    - by default, this is an unbounded scalar with Gaussian mutations.
    - if both lower and upper bounds are provided, sigma will be adapted so that the range spans 6 sigma.
      Also, if init is not provided, it will be set to the middle value.
    - More specific behaviors can be obtained throught the following methods:
      :code:`set_bounds`, :code:`set_mutation`, :code:`set_integer_casting`
    """
    value = core.ValueProperty()

    def __init__(
        self,
        init: Optional[float] = None,
        *,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        mutable_sigma: bool = True
    ) -> None:
        bounded = all((a is not None for a in (lower, upper)))
        no_init = init is None
        if bounded:
            if init is None:
                init = (lower + upper) / 2.0
        if init is None:
            init = 0.0
        super().__init__(init=np.array([init]), mutable_sigma=mutable_sigma)
        if bounded:
            self.set_mutation(sigma=(upper - lower) / 6)
        if any((a is not None for a in (lower, upper))):
            self.set_bounds(lower=lower, upper=upper, full_range_sampling=bounded