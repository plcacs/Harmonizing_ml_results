import importlib
import math
import types
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union,
    cast, overload
)
import numpy as np
from hypothesis import strategies as st
from hypothesis._settings import note_deprecation
from hypothesis.errors import HypothesisException, InvalidArgument
from hypothesis.extra._array_helpers import (
    NDIM_MAX, BasicIndex, BasicIndexStrategy, BroadcastableShapes, Shape,
    array_shapes, broadcastable_shapes, check_argument, check_valid_dims,
    mutually_broadcastable_shapes as _mutually_broadcastable_shapes,
    order_check, valid_tuple_axes as _valid_tuple_axes
)
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import proxies
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.lazy import unwrap_strategies
from hypothesis.strategies._internal.numbers import Real
from hypothesis.strategies._internal.strategies import Ex, MappedStrategy, T, check_strategy
from hypothesis.strategies._internal.utils import defines_strategy

def _try_import(mod_name: str, attr_name: str) -> Any:
    assert '.' not in attr_name
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name, None)
    except ImportError:
        return None

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray
else:
    NDArray = _try_import('numpy.typing', 'NDArray')
    
ArrayLike = _try_import('numpy.typing', 'ArrayLike')
_NestedSequence = _try_import('numpy._typing._nested_sequence', '_NestedSequence')
_SupportsArray = _try_import('numpy._typing._array_like', '_SupportsArray')

__all__ = [
    'BroadcastableShapes', 'array_dtypes', 'array_shapes', 'arrays', 'basic_indices',
    'boolean_dtypes', 'broadcastable_shapes', 'byte_string_dtypes', 'complex_number_dtypes',
    'datetime64_dtypes', 'floating_dtypes', 'from_dtype', 'integer_array_indices',
    'integer_dtypes', 'mutually_broadcastable_shapes', 'nested_dtypes', 'scalar_dtypes',
    'timedelta64_dtypes', 'unicode_string_dtypes', 'unsigned_integer_dtypes',
    'valid_tuple_axes'
]

TIME_RESOLUTIONS = tuple('Y  M  D  h  m  s  ms  us  ns  ps  fs  as'.split())
NP_FIXED_UNICODE = tuple((int(x) for x in np.__version__.split('.')[:2])) >= (1, 19)

@defines_strategy(force_reusable_values=True)
def from_dtype(
    dtype: np.dtype,
    *,
    alphabet: Optional[st.SearchStrategy[str]] = None,
    min_size: int = 0,
    max_size: Optional[int] = None,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    allow_subnormal: Optional[bool] = None,
    exclude_min: Optional[bool] = None,
    exclude_max: Optional[bool] = None,
    min_magnitude: int = 0,
    max_magnitude: Optional[float] = None
) -> st.SearchStrategy[Any]:
    """Creates a strategy which can generate any value of the given dtype."""
    check_type(np.dtype, dtype, 'dtype')
    kwargs = {k: v for k, v in locals().items() if k != 'dtype' and v is not None}
    if dtype.names is not None and dtype.fields is not None:
        subs = [from_dtype(dtype.fields[name][0], **kwargs) for name in dtype.names]
        return st.tuples(*subs)
    if dtype.subdtype is not None:
        subtype, shape = dtype.subdtype
        return arrays(subtype, shape, elements=kwargs)

    def compat_kw(*args: str, **kw: Any) -> Dict[str, Any]:
        """Update default args to the strategy with user-supplied keyword args."""
        assert {'min_value', 'max_value', 'max_size'}.issuperset(kw)
        for key in set(kwargs).intersection(kw):
            msg = f'dtype {dtype!r} requires {key}={kwargs[key]!r} to be %s {kw[key]!r}'
            if kw[key] is not None:
                if key.startswith('min_') and kw[key] > kwargs[key]:
                    raise InvalidArgument(msg % ('at least',))
                elif key.startswith('max_') and kw[key] < kwargs[key]:
                    raise InvalidArgument(msg % ('at most',))
        kw.update({k: v for k, v in kwargs.items() if k in args or k in kw})
        return kw

    if dtype.kind == 'b':
        result = st.booleans()
    elif dtype.kind == 'f':
        result = st.floats(
            width=cast(Literal[16, 32, 64], min(8 * dtype.itemsize, 64)),
            **compat_kw('min_value', 'max_value', 'allow_nan', 'allow_infinity',
                       'allow_subnormal', 'exclude_min', 'exclude_max')
        )
    elif dtype.kind == 'c':
        result = st.complex_numbers(
            width=cast(Literal[32, 64, 128], min(8 * dtype.itemsize, 128)),
            **compat_kw('min_magnitude', 'max_magnitude', 'allow_nan',
                       'allow_infinity', 'allow_subnormal')
        )
    elif dtype.kind in ('S', 'a'):
        max_size = dtype.itemsize or None
        result = st.binary(**compat_kw('min_size', max_size=max_size)).filter(
            lambda b: b[-1:] != b'\x00'
        )
    elif dtype.kind == 'u':
        kw = compat_kw(min_value=0, max_value=2 ** (8 * dtype.itemsize) - 1)
        result = st.integers(**kw)
    elif dtype.kind == 'i':
        overflow = 2 ** (8 * dtype.itemsize - 1)
        result = st.integers(**compat_kw(min_value=-overflow, max_value=overflow - 1))
    elif dtype.kind == 'U':
        max_size = (dtype.itemsize or 0) // 4 or None
        if NP_FIXED_UNICODE and 'alphabet' not in kwargs:
            kwargs['alphabet'] = st.characters()
        result = st.text(**compat_kw('alphabet', 'min_size', max_size=max_size)).filter(
            lambda b: b[-1:] != '\x00'
        )
    elif dtype.kind in ('m', 'M'):
        if '[' in dtype.str:
            res = st.just(dtype.str.split('[')[-1][:-1])
        else:
            res = st.sampled_from(TIME_RESOLUTIONS)
        if allow_nan is not False:
            elems = st.integers(-2 ** 63, 2 ** 63 - 1) | st.just('NaT')
        else:
            elems = st.integers(-2 ** 63 + 1, 2 ** 63 - 1)
        result = st.builds(dtype.type, elems, res)
    else:
        raise InvalidArgument(f'No strategy inference for {dtype}')
    return result.map(dtype.type)

class ArrayStrategy(st.SearchStrategy[NDArray[Any]]):
    def __init__(
        self,
        element_strategy: st.SearchStrategy[Any],
        shape: Shape,
        dtype: np.dtype,
        fill: st.SearchStrategy[Any],
        unique: bool
    ):
        self.shape = tuple(shape)
        self.fill = fill
        self.array_size = int(np.prod(shape))
        self.dtype = dtype
        self.element_strategy = element_strategy
        self.unique = unique
        self._check_elements = dtype.kind not in ('O', 'V')

    def __repr__(self) -> str:
        return f'ArrayStrategy({self.element_strategy!r}, shape={self.shape}, dtype={self.dtype!r}, fill={self.fill!r}, unique={self.unique!r})'

    def set_element(self, val: Any, result: NDArray[Any], idx: int, *, fill: bool = False) -> None:
        try:
            result[idx] = val
        except TypeError as err:
            raise InvalidArgument(
                f'Could not add element={val!r} of {val.dtype!r} to array of {result.dtype!r} - '
                'possible mismatch of time units in dtypes?'
            ) from err
        try:
            elem_changed = self._check_elements and val != result[idx] and (val == val)
        except Exception as err:
            raise HypothesisException(
                'Internal error when checking element=%r of %r to array of %r' %
                (val, val.dtype, result.dtype)
            ) from err
        if elem_changed:
            strategy = self.fill if fill else self.element_strategy
            if self.dtype.kind == 'f':
                try:
                    is_subnormal = 0 < abs(val) < np.finfo(self.dtype).tiny
                except Exception:
                    is_subnormal = False
                if is_subnormal:
                    raise InvalidArgument(
                        f'Generated subnormal float {val} from strategy {strategy} resulted in '
                        f'{result[idx]!r}, probably as a result of NumPy being built with '
                        'flush-to-zero compiler options. Consider passing allow_subnormal=False.'
                    )
            raise InvalidArgument(
                'Generated array element %r from %r cannot be represented as dtype %r - '
                'instead it becomes %r (type %r). Consider using a more precise strategy, '
                'for example passing the `width` argument to `floats()`.' %
                (val, strategy, self.dtype, result[idx], type(result[idx]))
            )

    def do_draw(self, data: cu.ConjectureData) -> NDArray[Any]:
        if 0 in self.shape:
            return np.zeros(dtype=self.dtype, shape=self.shape)
        unsized_string_dtype = self.dtype.kind in ('S', 'a', 'U') and self.dtype.itemsize == 0
        result = np.zeros(shape=self.array_size, dtype=object if unsized_string_dtype else self.dtype)
        if self.fill.is_empty:
            if self.unique:
                elems = st.lists(
                    self.element_strategy,
                    min_size=self.array_size,
                    max_size=self.array_size,
                    unique=True
                )
                for i, v in enumerate(data.draw(elems)):
                    self.set_element(v, result, i)
            else:
                for i in range(len(result)):
                    self.set_element(data.draw(self.element_strategy), result, i)
        else:
            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                average_size=min(0.9 * self.array_size, max(10, math.sqrt(self.array_size)))
            )
            needs_fill = np.full(self.array_size, True)
            seen: Set[Any] = set()
            while elements.more():
                i = data.draw_integer(0, self.array_size - 1)
                if not needs_fill[i]:
                    elements.reject()
                    continue
                self.set_element(data.draw(self.element_strategy), result, i)
                if self.unique:
                    if result[i] in seen:
                        elements.reject()
                        continue
                    else:
                        seen.add(result[i])
                needs_fill[i] = False
            if needs_fill.any():
                one_element = np.zeros(shape=1, dtype=object if unsized_string_dtype else self.dtype)
                self.set_element(data.draw(self.fill), one_element, 0, fill=True)
                if unsized_string_dtype:
                    one_element = one_element.astype(self.dtype)
                fill_value = one_element[0]
                if self.unique:
                    try:
                        is_nan = np.isnan(fill_value)
                    except TypeError:
                        is_nan = False
                    if not is_nan:
                        raise InvalidArgument(
                            f'Cannot fill unique array with non-NaN value {fill_value!r}'
                        )
                np.putmask(result, needs_fill, one_element)
        if unsized_string_dtype:
            out = result.astype(self.dtype)
            mismatch = out != result
            if mismatch.any():
                raise InvalidArgument(
                    'Array elements %r cannot be represented as dtype %r - instead they become %r. '
                    'Use a more precise strategy, e.g. without trailing null bytes, as this will be '
                    'an error future versions.' % (result[mismatch], self.dtype, out[mismatch])
                )
            result = out
        result = result.reshape(self.shape).copy()
        assert result.base is None
        return result

def fill_for(
    elements: st.SearchStrategy[Any],
    unique: bool,
    fill: Optional[st.SearchStrategy[Any]],
    name: str = ''
) -> st.SearchStrategy[Any]:
    if fill is None:
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    else:
        check_strategy(fill, f'{name}.fill' if name else 'fill')
    return fill

D = TypeVar('D', bound='DTypeLike')
G = TypeVar('G', bound='np.generic')

@overload
@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: np.dtype,
    shape: Shape,
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False
) -> st.SearchStrategy[NDArray[Any]]: ...

@overload
@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: st.SearchStrategy[np.dtype],
    shape: Shape,
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False
) -> st.SearchStrategy[NDArray[Any]]: ...

@overload
@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: np.dtype,
    shape: st.SearchStrategy[Shape],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False
) -> st.SearchStrategy[NDArray[Any]]: ...

@overload
@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: st.SearchStrategy[np.dtype],
    shape: st.SearchStrategy[Shape],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False
) -> st.SearchStrategy[NDArray[Any]]: ...

@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: Union[np.dtype, st.SearchStrategy[np.dtype]],
    shape: Union[Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[st.SearchStrategy[Any], Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False
) -> st.SearchStrategy[NDArray[Any]]:
    """Returns a strategy for generating :class:`numpy:numpy.ndarray`\\ s."""
    if type(dtype) in (getattr(types, 'UnionType', object()), Union):
        dtype = st.one_of(*(from_dtype(np.dtype(d)) for d in dtype.__args__))
    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(lambda d: arrays(d, shape, elements=elements, fill=fill, unique=unique))
    if isinstance(shape, st.SearchStrategy):
        return shape.flatmap(lambda s: arrays(dtype, s, elements=elements, fill=fill, unique=unique))
    dtype = np.dtype(dtype)
    assert isinstance(dtype, np.dtype)
    if elements is None or isinstance(elements, Mapping):
        if dtype.kind in ('m', 'M') and '[' not in dtype.str:
            return st.sampled_from(TIME_RESOLUTIONS).map(
                (dtype.str + '[{}]').format
            ).flatmap(lambda d: arrays(d, shape=shape, fill=fill, unique=unique))
        elements = from_dtype(dtype, **elements or {})
    check_strategy(elements, 'elements')
    unwrapped = unwrap_strategies(elements)
    if isinstance(unwrapped, MappedStrategy) and unwrapped.pack == dtype.type:
        elements = unwrapped.mapped_strategy
        if getattr(unwrapped, 'force_has_reusable_values', False):
            elements.force_has_reusable_values = True
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    check_argument(
        all((isinstance(s, int) for s in shape)),
        'Array shape must be integer in each dimension, provided shape was {}',
        shape
    )
    fill = fill_for(elements=elements, unique=unique, fill=fill)
    return ArrayStrategy(elements, shape, dtype, fill, unique)

@defines_strategy()
def scalar_dtypes() -> st.SearchStrategy[np.dtype]:
    """Return a strategy that can return any non-flexible scalar dtype."""
    return st.one_of(
        boolean_dtypes(),
        integer_dtypes(),
        unsigned_integer_dtypes(),
        floating