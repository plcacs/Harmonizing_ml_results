import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from . import _layering
from . import core
from .container import Dict
from . import utils
D = tp.TypeVar('D', bound='Data')
P = tp.TypeVar('P', bound=core.Parameter)

def _param_string(parameters):
    """Hacky helper for nice-visualizatioon"""
    substr = f'[{parameters._get_parameters_str()}]'
    if substr == '[]':
        substr = ''
    return substr

class Data(core.Parameter):

    def __init__(self, *, init: tp.Optional[tp.ArrayLike]=None, shape: tp.Optional[tp.Tuple[int, ...]]=None, lower: tp.BoundValue=None, upper: tp.BoundValue=None, mutable_sigma: bool=False):
        super().__init__()
        sigma: tp.Any = np.array([1.0])
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
        layer: tp.Any = None
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
    def bounds(self):
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
    def dimension(self):
        return int(np.prod(self._value.shape))

    def _get_name(self):
        cls = self.__class__.__name__
        descriptors: tp.List[str] = []
        if self._value.shape != (1,):
            descriptors.append(str(self._value.shape).replace(' ', ''))
        descriptors += [layer.name for layer in self._layers[1:] if not isinstance(layer, (_layering.ArrayCasting, _layering._ScalarCasting))]
        description = '' if not descriptors else '{{{}}}'.format(','.join(descriptors))
        return f'{cls}{description}' + _param_string(self.parameters)

    @property
    def sigma(self):
        return self.parameters['sigma']

    def _layered_sample(self):
        child = self.spawn_child()
        from . import helpers
        with helpers.deterministic_sampling(child):
            child.mutate()
        return child

    def set_bounds(self, lower=None, upper=None, method='bouncing', full_range_sampling=None):
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

    def set_mutation(self, sigma=None, exponent=None):
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

    def set_integer_casting(self):
        return self.add_layer(_layering.Int())

    @property
    def integer(self):
        return any((isinstance(x, _layering.Int) for x in self._layers))

    def _internal_set_standardized_data(self, data, reference):
        assert isinstance(data, np.ndarray)
        sigma = reference.sigma.value
        data_reduc = sigma * (data + reference._to_reduced_space()).reshape(reference._value.shape)
        self._value = data_reduc
        self.value

    def _internal_get_standardized_data(self, reference):
        return reference._to_reduced_space(self._value - reference._value)

    def _to_reduced_space(self, value=None):
        if value is None:
            value = self._value
        reduced = value / self.sigma.value
        return reduced.ravel()

    def _layered_recombine(self, *others: D):
        all_params = [self] + list(others)
        all_arrays = [p.get_standardized_data(reference=self) for p in all_params]
        mean: np.ndarray = np.mean(all_arrays, axis=0)
        self.set_standardized_data(mean)

    def copy(self):
        child = super().copy()
        child._value = np.array(self._value, copy=True)
        return child

    def _layered_set_value(self, value):
        self._check_frozen()
        if self._value.shape != value.shape:
            raise ValueError(f'Cannot set array of shape {self._value.shape} with value of shape {value.shape}')
        self._value = value

    def _layered_get_value(self):
        return self._value

    def _new_with_data_layer(self, name, *args: tp.Any, **kwargs: tp.Any):
        from . import _datalayers
        new = self.copy()
        new.add_layer(getattr(_datalayers, name)(*args, **kwargs))
        return new

    def __mod__(self, module):
        return self._new_with_data_layer('Modulo', module)

    def __rpow__(self, base):
        return self._new_with_data_layer('Exponent', base)

    def __add__(self, offset):
        return self._new_with_data_layer('Add', offset)

    def __sub__(self, offset):
        return self.__add__(-offset)

    def __radd__(self, offset):
        return self.__add__(offset)

    def __mul__(self, value):
        return self._new_with_data_layer('Multiply', value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __truediv__(self, value):
        return self.__mul__(1.0 / value)

    def __rtruediv__(self, value):
        return value * self ** (-1)

    def __pow__(self, power):
        return self._new_with_data_layer('Power', power)

    def __neg__(self):
        return self.__mul__(-1.0)

def _fix_legacy(parameter):
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
    bound: _datalayers.BoundLayer = legacy[bound_ind]
    exp: _datalayers.Exponent = legacy[(bound_ind + 1) % 2]
    bound.bounds = tuple((None if b is None else exp.backward(b) for b in bound.bounds))
    if isinstance(bound, _datalayers.Bound):
        bound = _datalayers.Bound(lower=bound.bounds[0], upper=bound.bounds[1], method=bound._method, uniform_sampling=bound.uniform_sampling)
    for lay in (bound, exp):
        lay._layer_index = 0
        lay._layers = [lay]
        parameter.add_layer(lay)
    return

class Array(Data):
    value: core.ValueProperty[tp.ArrayLike, np.ndarray] = core.ValueProperty()

class Scalar(Data):
    value: core.ValueProperty[float, float] = core.ValueProperty()

    def __init__(self, init=None, *, lower: tp.Optional[float]=None, upper: tp.Optional[float]=None, mutable_sigma: bool=True):
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
            self.set_bounds(lower=lower, upper=upper, full_range_sampling=bounded and no_init)
        self.add_layer(_layering._ScalarCasting())

class Log(Scalar):

    def __init__(self, *, init: tp.Optional[float]=None, exponent: tp.Optional[float]=None, lower: tp.Optional[float]=None, upper: tp.Optional[float]=None, mutable_sigma: bool=False):
        no_init = init is None
        bounded = all((a is not None for a in (lower, upper)))
        if bounded:
            if init is None:
                init = float(np.sqrt(lower * upper))
            if exponent is None:
                exponent = float(np.exp((np.log(upper) - np.log(lower)) / 6.0))
        if init is None:
            raise ValueError('You must define either a init value or both lower and upper bounds')
        if exponent is None:
            exponent = 2.0
        from . import _datalayers
        exp_layer = _datalayers.Exponent(exponent)
        raw_bounds = tuple((None if x is None else np.array([x], dtype=float) for x in (lower, upper)))
        bounds = tuple((None if x is None else exp_layer.backward(x) for x in raw_bounds))
        init = exp_layer.backward(np.array([init]))
        super().__init__(init=init[0], mutable_sigma=mutable_sigma)
        if any((x is not None for x in bounds)):
            bound_layer = _datalayers.Bound(*bounds, uniform_sampling=bounded and no_init)
            bound_layer(self, inplace=True)
        self.add_layer(exp_layer)