from __future__ import annotations
from collections.abc import Callable, Hashable, Iterator
from datetime import timedelta
import operator
from sys import getsizeof
from typing import TYPE_CHECKING, Any, Literal, cast, overload, Union
import numpy as np
from pandas._libs import index as libindex, lib
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import cache_readonly, doc, set_module
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import ensure_platform_int, ensure_python_int, is_float, is_integer, is_scalar, is_signed_integer_dtype
from pandas.core.dtypes.generic import ABCTimedeltaIndex
from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.ops.common import unpack_zerodim_and_defer
if TYPE_CHECKING:
    from pandas._typing import Axis, Dtype, JoinHow, NaPosition, NumpySorter, Self, npt
    from pandas import Series
_empty_range: range = range(0)
_dtype_int64: np.dtype = np.dtype(np.int64)

def min_fitting_element(start: int, step: int, lower_limit: int) -> int:
    """Returns the smallest element greater than or equal to the limit"""
    no_steps: int = -(-(lower_limit - start) // abs(step))
    return start + abs(step) * no_steps

@set_module('pandas')
class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of an Index limited to representing
    monotonic ranges with a 64-bit dtype. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
        The end value of the range (exclusive).
    step : int (default: 1)
        The step size of the range.
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.

    Examples
    --------
    >>> list(pd.RangeIndex(5))
    [0, 1, 2, 3, 4]

    >>> list(pd.RangeIndex(-2, 4))
    [-2, -1, 0, 1, 2, 3]

    >>> list(pd.RangeIndex(0, 10, 2))
    [0, 2, 4, 6, 8]

    >>> list(pd.RangeIndex(2, -10, -3))
    [2, -1, -4, -7]

    >>> list(pd.RangeIndex(0))
    []

    >>> list(pd.RangeIndex(1, 0))
    []
    """
    _typ: str = 'rangeindex'
    _dtype_validation_metadata: tuple = (is_signed_integer_dtype, 'signed integer')
    _range: range
    _values: np.ndarray

    @property
    def _engine_type(self) -> type:
        return libindex.Int64Engine

    def __new__(cls, start: Any = None, stop: Any = None, step: Any = None, dtype: Any = None, copy: bool = False, name: Any = None) -> RangeIndex:
        cls._validate_dtype(dtype)
        name = maybe_extract_name(name, start, cls)
        if isinstance(start, cls):
            return start.copy(name=name)
        elif isinstance(start, range):
            return cls._simple_new(start, name=name)
        if com.all_none(start, stop, step):
            raise TypeError('RangeIndex(...) must be called with integers')
        start = ensure_python_int(start) if start is not None else 0
        if stop is None:
            (start, stop) = (0, start)
        else:
            stop = ensure_python_int(stop)
        step = ensure_python_int(step) if step is not None else 1
        if step == 0:
            raise ValueError('Step must not be zero')
        rng: range = range(start, stop, step)
        return cls._simple_new(rng, name=name)

    @classmethod
    def from_range(cls, data: range, name: Any = None, dtype: Any = None) -> RangeIndex:
        """
        Create :class:`pandas.RangeIndex` from a ``range`` object.

        This method provides a way to create a :class:`pandas.RangeIndex` directly
        from a Python ``range`` object. The resulting :class:`RangeIndex` will have
        the same start, stop, and step values as the input ``range`` object.
        It is particularly useful for constructing indices in an efficient and
        memory-friendly manner.

        Parameters
        ----------
        data : range
            The range object to be converted into a RangeIndex.
        name : str, default None
            Name to be stored in the index.
        dtype : Dtype or None
            Data type for the RangeIndex. If None, the default integer type will
            be used.

        Returns
        -------
        RangeIndex

        See Also
        --------
        RangeIndex : Immutable Index implementing a monotonic integer range.
        Index : Immutable sequence used for indexing and alignment.

        Examples
        --------
        >>> pd.RangeIndex.from_range(range(5))
        RangeIndex(start=0, stop=5, step=1)

        >>> pd.RangeIndex.from_range(range(2, -10, -3))
        RangeIndex(start=2, stop=-10, step=-3)
        """
        if not isinstance(data, range):
            raise TypeError(f'{cls.__name__}(...) must be called with object coercible to a range, {data!r} was passed')
        cls._validate_dtype(dtype)
        return cls._simple_new(data, name=name)

    @classmethod
    def _simple_new(cls, values: range, name: Any = None) -> RangeIndex:
        result: RangeIndex = object.__new__(cls)
        assert isinstance(values, range)
        result._range = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        result._references = None
        return result

    @classmethod
    def _validate_dtype(cls, dtype: Any) -> None:
        if dtype is None:
            return
        (validation_func, expected) = cls._dtype_validation_metadata
        if not validation_func(dtype):
            raise ValueError(f'Incorrect `dtype` passed: expected {expected}, received {dtype}')

    @cache_readonly
    def _constructor(self) -> type:
        """return the class to use for construction"""
        return Index

    @cache_readonly
    def _data(self) -> np.ndarray:
        """
        An int array that for performance reasons is created only when needed.

        The constructed array is saved in ``_cache``.
        """
        return np.arange(self.start, self.stop, self.step, dtype=np.int64)

    def _get_data_as_items(self) -> list[tuple[str, int]]:
        """return a list of tuples of start, stop, step"""
        rng: range = self._range
        return [('start', rng.start), ('stop', rng.stop), ('step', rng.step)]

    def __reduce__(self) -> tuple:
        d: dict = {'name': self._name}
        d.update(dict(self._get_data_as_items()))
        return (ibase._new_Index, (type(self), d), None)

    def _format_attrs(self) -> list[tuple[str, str | int]]:
        """
        Return a list of tuples of the (attr, formatted_value)
        """
        attrs: list[tuple[str, str | int]] = cast('list[tuple[str, str | int]]', self._get_data_as_items())
        if self._name is not None:
            attrs.append(('name', ibase.default_pprint(self._name)))
        return attrs

    def _format_with_header(self, *, header: list[str], na_rep: str) -> list[str]:
        if not len(self._range):
            return header
        first_val_str: str = str(self._range[0])
        last_val_str: str = str(self._range[-1])
        max_length: int = max(len(first_val_str), len(last_val_str))
        return header + [f'{x:<{max_length}}' for x in self._range]

    @property
    def start(self) -> int:
        """
        The value of the `start` parameter (``0`` if this was not supplied).

        This property returns the starting value of the `RangeIndex`. If the `start`
        value is not explicitly provided during the creation of the `RangeIndex`,
        it defaults to 0.

        See Also
        --------
        RangeIndex : Immutable index implementing a range-based index.
        RangeIndex.stop : Returns the stop value of the `RangeIndex`.
        RangeIndex.step : Returns the step value of the `RangeIndex`.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.start
        0

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.start
        2
        """
        return self._range.start

    @property
    def stop(self) -> int:
        """
        The value of the `stop` parameter.

        This property returns the `stop` value of the RangeIndex, which defines the
        upper (or lower, in case of negative steps) bound of the index range. The
        `stop` value is exclusive, meaning the RangeIndex includes values up to but
        not including this value.

        See Also
        --------
        RangeIndex : Immutable index representing a range of integers.
        RangeIndex.start : The start value of the RangeIndex.
        RangeIndex.step : The step size between elements in the RangeIndex.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.stop
        5

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.stop
        -10
        """
        return self._range.stop

    @property
    def step(self) -> int:
        """
        The value of the `step` parameter (``1`` if this was not supplied).

        The ``step`` parameter determines the increment (or decrement in the case
        of negative values) between consecutive elements in the ``RangeIndex``.

        See Also
        --------
        RangeIndex : Immutable index implementing a range-based index.
        RangeIndex.stop : Returns the stop value of the RangeIndex.
        RangeIndex.start : Returns the start value of the RangeIndex.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.step
        1

        >>> idx = pd.RangeIndex(2, -10, -3)
        >>> idx.step
        -3

        Even if :class:`pandas.RangeIndex` is empty, ``step`` is still ``1`` if
        not supplied.

        >>> idx = pd.RangeIndex(1, 0)
        >>> idx.step
        1
        """
        return self._range.step

    @cache_readonly
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.
        """
        rng: range = self._range
        return getsizeof(rng) + sum((getsizeof(getattr(rng, attr_name)) for attr_name in ['start', 'stop', 'step']))

    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self.nbytes

    @property
    def dtype(self) -> np.dtype:
        return _dtype_int64

    @property
    def is_unique(self) -> bool:
        """return if the index has unique values"""
        return True

    @cache_readonly
    def is_monotonic_increasing(self) -> bool:
        return self._range.step > 0 or len(self) <= 1

    @cache_readonly
    def is_monotonic_decreasing(self) -> bool:
        return self._range.step < 0 or len(self) <= 1

    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            key = ensure_python_int(key)
        except (TypeError, OverflowError):
            return False
        return key in self._range

    @property
    def inferred_type(self) -> str:
        return 'integer'

    @doc(Index.get_loc)
    def get_loc(self, key: Any) -> int:
        if is_integer(key) or (is_float(key) and key.is_integer()):
            new_key: int = int(key)
            try:
                return self._range.index(new_key)
            except ValueError as err:
                raise KeyError(key) from err
        if isinstance(key, Hashable):
            raise KeyError(key)
        self._check_indexing_error(key)
        raise KeyError(key)

    def _get_indexer(self, target: Index, method: str | None = None, limit: int | None = None, tolerance: Any = None) -> npt.NDArray[np.intp]:
        if com.any_not_none(method, tolerance, limit):
            return super()._get_indexer(target, method=method, tolerance=tolerance, limit=limit)
        if self.step > 0:
            (start, stop, step) = (self.start, self.stop, self.step)
        else:
            reverse: range = self._range[::-1]
            (start, stop, step) = (reverse.start, reverse.stop, reverse.step)
        target_array: np.ndarray = np.asarray(target)
        locs: np.ndarray = target_array - start
        valid: np.ndarray = (locs % step == 0) & (locs >= 0) & (target_array < stop)
        locs[~valid] = -1
        locs[valid] = locs[valid] / step
        if step != self.step:
            locs[valid] = len(self) - 1 - locs[valid]
        return ensure_platform_int(locs)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        """
        Should an integer key be treated as positional?
        """
        return False

    def tolist(self) -> list[int]:
        return list(self._range)

    @doc(Index.__iter__)
    def __iter__(self) -> Iterator[int]:
        yield from self._range

    @doc(Index._shallow_copy)
    def _shallow_copy(self, values: np.ndarray, name: Hashable = no_default) -> Index:
        name = self._name if name is no_default else name
        if values.dtype.kind == 'f':
            return Index(values, name=name, dtype=np.float64)
        if values.dtype.kind == 'i' and values.ndim == 1:
            if len(values) == 1:
                start: int = values[0]
                new_range: range = range(start, start + self.step, self.step)
                return type(self)._simple_new(new_range, name=name)
            maybe_range: range | np.ndarray = ibase.maybe_sequence_to_range(values)
            if isinstance(maybe_range, range):
                return type(self)._simple_new(maybe_range, name=name)
        return self._constructor._simple_new(values, name=name)

    def _view(self) -> Self:
        result: Self = type(self)._simple_new(self._range, name=self._name)
        result._cache = self._cache
        return result

    def _wrap_reindex_result(self, target: Index, indexer: npt.NDArray[np.intp] | None, preserve_names: bool) -> tuple[Index, npt.NDArray[np.intp] | None]:
        if not isinstance(target, type(self)) and target.dtype.kind == 'i':
            target = self._shallow_copy(target._values, name=target.name)
        return super()._wrap_reindex_result(target, indexer, preserve_names)

    @doc(Index.copy)
    def copy(self, name: Hashable | None = None, deep: bool = False) -> Self:
        name = self._validate_names(name=name, deep=deep)[0]
        new_index: Self = self._rename(name=name)
        return new_index

    def _minmax(self, meth: Literal['min', 'max']) -> int | float:
        no_steps: int = len(self) - 1
        if no_steps == -1:
            return np.nan
        elif meth == 'min' and self.step > 0 or (meth == 'max' and self.step < 0):
            return self.start
        return self.start + self.step * no_steps

    def min(self, axis: Any = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int | float:
        """The minimum value of the RangeIndex"""
        nv.validate_minmax_axis(axis)
        nv.validate_min(args, kwargs)
        return self._minmax('min')

   