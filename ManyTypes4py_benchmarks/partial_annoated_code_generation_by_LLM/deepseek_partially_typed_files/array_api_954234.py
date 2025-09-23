import math
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from numbers import Real
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, TypeVar, Union, get_args, cast
from warnings import warn
from weakref import WeakValueDictionary
from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.extra._array_helpers import NDIM_MAX, BasicIndex, BasicIndexStrategy, BroadcastableShapes, Shape, array_shapes, broadcastable_shapes, check_argument, check_valid_dims, mutually_broadcastable_shapes as _mutually_broadcastable_shapes, order_check, valid_tuple_axes as _valid_tuple_axes
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check_function
from hypothesis.internal.floats import next_down
from hypothesis.internal.reflection import proxies
from hypothesis.internal.validation import check_type, check_valid_bound, check_valid_integer, check_valid_interval
from hypothesis.strategies._internal.strategies import check_strategy
from hypothesis.strategies._internal.utils import defines_strategy
if TYPE_CHECKING:
    from typing import TypeAlias

__all__ = ['make_strategies_namespace']

RELEASED_VERSIONS = ('2021.12', '2022.12', '2023.12', '2024.12')
NOMINAL_VERSIONS = (*RELEASED_VERSIONS, 'draft')
assert sorted(NOMINAL_VERSIONS) == list(NOMINAL_VERSIONS)
NominalVersion = Literal['2021.12', '2022.12', '2023.12', '2024.12', 'draft']
assert get_args(NominalVersion) == NOMINAL_VERSIONS

INT_NAMES = ('int8', 'int16', 'int32', 'int64')
UINT_NAMES = ('uint8', 'uint16', 'uint32', 'uint64')
ALL_INT_NAMES = INT_NAMES + UINT_NAMES
FLOAT_NAMES = ('float32', 'float64')
REAL_NAMES = ALL_INT_NAMES + FLOAT_NAMES
COMPLEX_NAMES = ('complex64', 'complex128')
NUMERIC_NAMES = REAL_NAMES + COMPLEX_NAMES
DTYPE_NAMES = ('bool', *NUMERIC_NAMES)
DataType = TypeVar('DataType')

@check_function
def check_xp_attributes(xp: Any, attributes: Sequence[str]) -> None:
    missing_attrs = [attr for attr in attributes if not hasattr(xp, attr)]
    if len(missing_attrs) > 0:
        f_attrs = ', '.join(missing_attrs)
        raise InvalidArgument(f'Array module {xp.__name__} does not have required attributes: {f_attrs}')

def partition_attributes_and_stubs(xp: Any, attributes: Sequence[str]) -> tuple[list[Any], list[str]]:
    non_stubs: list[Any] = []
    stubs: list[str] = []
    for attr in attributes:
        try:
            non_stubs.append(getattr(xp, attr))
        except AttributeError:
            stubs.append(attr)
    return (non_stubs, stubs)

def warn_on_missing_dtypes(xp: Any, stubs: Sequence[str]) -> None:
    f_stubs = ', '.join(stubs)
    warn(f'Array module {xp.__name__} does not have the following dtypes in its namespace: {f_stubs}', HypothesisWarning, stacklevel=3)

def find_castable_builtin_for_dtype(xp: Any, api_version: NominalVersion, dtype: Optional[DataType]) -> type:
    """Returns builtin type which can have values that are castable to the given
    dtype, according to :xp-ref:`type promotion rules <type_promotion.html>`.

    For floating dtypes we always return ``float``, even though ``int`` is also castable.
    """
    stubs: list[str] = []
    try:
        bool_dtype = xp.bool
        if dtype == bool_dtype:
            return bool
    except AttributeError:
        stubs.append('bool')
    (int_dtypes, int_stubs) = partition_attributes_and_stubs(xp, ALL_INT_NAMES)
    if dtype in int_dtypes:
        return int
    (float_dtypes, float_stubs) = partition_attributes_and_stubs(xp, FLOAT_NAMES)
    if dtype is not None and dtype in float_dtypes:
        return float
    stubs.extend(int_stubs)
    stubs.extend(float_stubs)
    if api_version > '2021.12':
        (complex_dtypes, complex_stubs) = partition_attributes_and_stubs(xp, COMPLEX_NAMES)
        if dtype in complex_dtypes:
            return complex
        stubs.extend(complex_stubs)
    if len(stubs) > 0:
        warn_on_missing_dtypes(xp, stubs)
    raise InvalidArgument(f'dtype={dtype} not recognised in {xp.__name__}')

@check_function
def dtype_from_name(xp: Any, name: str) -> DataType:
    if name in DTYPE_NAMES:
        try:
            return cast(DataType, getattr(xp, name))
        except AttributeError as e:
            raise InvalidArgument(f'Array module {xp.__name__} does not have dtype {name} in its namespace') from e
    else:
        f_valid_dtypes = ', '.join(DTYPE_NAMES)
        raise InvalidArgument(f'{name} is not a valid Array API data type (pick from: {f_valid_dtypes})')

def _from_dtype(xp: Any, api_version: NominalVersion, dtype: Union[DataType, str], *, min_value: Optional[Union[int, float]]=None, max_value: Optional[Union[int, float]]=None, allow_nan: Optional[bool]=None, allow_infinity: Optional[bool]=None, allow_subnormal: Optional[bool]=None, exclude_min: Optional[bool]=None, exclude_max: Optional[bool]=None) -> st.SearchStrategy[Union[bool, int, float, complex]]:
    """Return a strategy for any value of the given dtype.

    Values generated are of the Python scalar which is
    :xp-ref:`promotable <type_promotion.html>` to ``dtype``, where the values do
    not exceed its bounds.

    * ``dtype`` may be a dtype object or the string name of a
      :xp-ref:`valid dtype <data_types.html>`.

    Compatible ``**kwargs`` are passed to the inferred strategy function for
    integers and floats.  This allows you to customise the min and max values,
    and exclude non-finite numbers. This is particularly useful when kwargs are
    passed through from :func:`arrays()`, as it seamlessly handles the ``width``
    or other representable bounds for you.
    """
    check_xp_attributes(xp, ['iinfo', 'finfo'])
    if isinstance(dtype, str):
        dtype = dtype_from_name(xp, dtype)
    builtin = find_castable_builtin_for_dtype(xp, api_version, dtype)

    def check_valid_minmax(prefix: str, val: Union[int, float], info_obj: Any) -> None:
        name = f'{prefix}_value'
        check_valid_bound(val, name)
        check_argument(val >= info_obj.min, f'dtype={dtype} requires {name}={val} to be at least {info_obj.min}')
        check_argument(val <= info_obj.max, f'dtype={dtype} requires {name}={val} to be at most {info_obj.max}')
    if builtin is bool:
        return st.booleans()
    elif builtin is int:
        iinfo = xp.iinfo(dtype)
        if min_value is None:
            min_value = iinfo.min
        if max_value is None:
            max_value = iinfo.max
        check_valid_integer(min_value, 'min_value')
        check_valid_integer(max_value, 'max_value')
        assert isinstance(min_value, int)
        assert isinstance(max_value, int)
        check_valid_minmax('min', min_value, iinfo)
        check_valid_minmax('max', max_value, iinfo)
        check_valid_interval(min_value, max_value, 'min_value', 'max_value')
        return st.integers(min_value=min_value, max_value=max_value)
    elif builtin is float:
        finfo = xp.finfo(dtype)
        kw: dict[str, Any] = {}
        if min_value is not None:
            check_valid_bound(min_value, 'min_value')
            assert isinstance(min_value, Real)
            check_valid_minmax('min', min_value, finfo)
            kw['min_value'] = min_value
        if max_value is not None:
            check_valid_bound(max_value, 'max_value')
            assert isinstance(max_value, Real)
            check_valid_minmax('max', max_value, finfo)
            if min_value is not None:
                check_valid_interval(min_value, max_value, 'min_value', 'max_value')
            kw['max_value'] = max_value
        if allow_subnormal is not None:
            kw['allow_subnormal'] = allow_subnormal
        else:
            subnormal = next_down(float(finfo.smallest_normal), width=finfo.bits)
            ftz = bool(xp.asarray(subnormal, dtype=dtype) == 0)
            if ftz:
                kw['allow_subnormal'] = False
        if allow_nan is not None:
            kw['allow_nan'] = allow_nan
        if allow_infinity is not None:
            kw['allow_infinity'] = allow_infinity
        if exclude_min is not None:
            kw['exclude_min'] = exclude_min
        if exclude_max is not None:
            kw['exclude_max'] = exclude_max
        return st.floats(width=finfo.bits, **kw)
    else:
        finfo = xp.finfo(dtype)
        if allow_subnormal is None:
            subnormal = next_down(float(finfo.smallest_normal), width=finfo.bits)
            x = xp.asarray(complex(subnormal, subnormal), dtype=dtype)
            builtin_x = complex(x)
            allow_subnormal = builtin_x.real != 0 and builtin_x.imag != 0
        return st.complex_numbers(allow_nan=allow_nan, allow_infinity=allow_infinity, allow_subnormal=allow_subnormal, width=finfo.bits * 2)

class ArrayStrategy(st.SearchStrategy):

    def __init__(self, *, xp: Any, api_version: NominalVersion, elements_strategy: st.SearchStrategy, dtype: DataType, shape: Shape, fill: st.SearchStrategy, unique: bool) -> None:
        self.xp = xp
        self.elements_strategy = elements_strategy
        self.dtype = dtype
        self.shape = shape
        self.fill = fill
        self.unique = unique
        self.array_size = math.prod(shape)
        self.builtin = find_castable_builtin_for_dtype(xp, api_version, dtype)
        self.finfo = None if self.builtin is not float else xp.finfo(self.dtype)

    def check_set_value(self, val: Any, val_0d: Any, strategy: st.SearchStrategy) -> None:
        if val == val and self.builtin(val_0d) != val:
            if self.builtin is float:
                assert self.finfo is not None
                try:
                    is_subnormal = 0 < abs(val) < self.finfo.smallest_normal
                except Exception:
                    is_subnormal = False
                if is_subnormal:
                    raise InvalidArgument(f'Generated subnormal float {val} from strategy {strategy} resulted in {val_0d!r}, probably as a result of array module {self.xp.__name__} being built with flush-to-zero compiler options. Consider passing allow_subnormal=False.')
            raise InvalidArgument(f'Generated array element {val!r} from strategy {strategy} cannot be represented with dtype {self.dtype}. Array module {self.xp.__name__} instead represents the element as {val_0d}. Consider using a more precise elements strategy, for example passing the width argument to floats().')

    def do_draw(self, data: Any) -> Any:
        if 0 in self.shape:
            return self.xp.zeros(self.shape, dtype=self.dtype)
        if self.fill.is_empty:
            elems = data.draw(st.lists(self.elements_strategy, min_size=self.array_size, max_size=self.array_size, unique=self.unique))
            try:
                result = self.xp.asarray(elems, dtype=self.dtype)
            except Exception as e:
                if len(elems) <= 6:
                    f_elems = str(elems)
                else:
                    f_elems = f'[{elems[0]}, {elems[1]}, ..., {elems[-2]}, {elems[-1]}]'
                types = tuple(sorted({type(e) for e in elems}, key=lambda t: t.__name__))
                f_types = f'type {types[0]}' if len(types) == 1 else f'types {types}'
                raise InvalidArgument(f'Generated elements {f_elems} from strategy {self.elements_strategy} could not be converted to array of dtype {self.dtype}. Consider if elements of {f_types} are compatible with {self.dtype}.') from e
            for i in range(self.array_size):
                self.check_set_value(elems[i], result[i], self.elements_strategy)
        else:
            fill_val = data.draw(self.fill)
            result_obj = [fill_val for _ in range(self.array_size)]
            fill_mask = [True for _ in range(self.array_size)]
            elements = cu.many(data, min_size=0, max_size=self.array_size, average_size=min(0.9 * self.array_size, max(10, math.sqrt(self.array_size))))
            assigned: set[int] = set()
            seen: set[Any] = set()
            while elements.more():
                i = data.draw_integer(0, self.array_size - 1)
                if i in assigned:
                    elements.reject("chose an array index we've already used")
                    continue
                val = data.draw(self.elements_strategy)
                if self.unique:
                    if val in seen:
                        elements.reject("chose an element we've already used")
                        continue
                    else:
                        seen.add(val)
                result_obj[i] = val
                assigned.add(i)
                fill_mask[i] = False
            try:
                result = self.xp.asarray(result_obj, dtype=self.dtype)
            except Exception as e:
                f_expr = f'xp.asarray({result_obj}, dtype={self.dtype})'
                raise InvalidArgument(f'Could not create array via {f_expr}') from e
            for (i, val) in enumerate(result_obj):
                val_0d = result[i]
                if fill_mask[i] and self.unique:
                    if not self.xp.isnan(val_0d):
                        raise InvalidArgument(f'Array module {self.xp.__name__} did not recognise fill value {fill_val!r} as NaN - instead got {val_0d!r}. Cannot fill unique array with non-NaN values.')
                else:
                    self.check_set_value(val, val_0d, self.elements_strategy)
        return self.xp.reshape(result, self.shape)

def _arrays(xp: Any, api_version: NominalVersion, dtype: Union[DataType, str, st.SearchStrategy[DataType], st.SearchStrategy[str]], shape: Union[int, Shape, st.SearchStrategy[Shape]], *, elements: Optional[Union[Mapping[str, Any], st.SearchStrategy]]=None, fill: Optional[st.SearchStrategy[Any]]=None, unique: bool=False) -> st.SearchStrategy:
    """Returns a strategy for :xp-ref:`arrays <array_object.html>`.

    * ``dtype`` may be a :xp-ref:`valid dtype <data_types.html>` object or name,
      or a strategy that generates such values.
    * ``shape`` may be an integer >= 0, a tuple of such integers, or a strategy
      that generates such values.
    * ``elements`` is a strategy for values to put in the array. If ``None``
      then a suitable value will be inferred based on the dtype, which may give
      any legal value (including e.g. NaN for floats). If a mapping, it will be
      passed as ``**kwargs`` to :func:`from_dtype()` when inferring based on the dtype.
    * ``fill`` is a strategy that may be used to generate a single background
      value for the array. If ``None``, a suitable default will be inferred
      based on the other arguments. If set to
      :func:`~hypothesis.strategies.nothing` then filling behaviour will be
      disabled entirely and every element will be generated independently.
    * ``unique`` specifies if the elements of the array should all be distinct
      from one another; if fill is also set, the only valid values for fill to
      return are NaN values.

    Arrays of specified ``dtype`` and ``shape`` are generated for example
    like this:

    .. code-block:: pycon

      >>> from numpy import array_api as xp
      >>> xps.arrays(xp, xp.int8, (2, 3)).example()
      Array([[-8,  6,  3],
             [-6,  4,  6]], dtype=int8)

    Specifying element boundaries by a :obj:`python:dict` of the kwargs to pass
    to :func:`from_dtype` will ensure ``dtype`` bounds will be respected.

    .. code-block:: pycon

      >>> xps.arrays(xp, xp.int8, 3, elements={"min_value": 10}).example()
      Array([125, 13, 79], dtype=int8)

    Refer to :doc:`What you can generate and how <data>` for passing
    your own elements strategy.

    .. code-block:: pycon

      >>> xps.arrays(xp, xp.float32, 3, elements=floats(0, 1, width=32)).example()
      Array([ 0.88974794,  0.77387938,  0.1977879 ], dtype=float32)

    Array values