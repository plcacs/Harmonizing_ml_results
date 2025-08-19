from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, overload, Callable, Iterator, Sequence
import warnings
import numpy as np
from pandas._libs import algos as libalgos, lib
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import Appender, Substitution, cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg, validate_insert_loc
from pandas.core.dtypes.cast import maybe_cast_pointwise_result
from pandas.core.dtypes.common import is_list_like, is_scalar, pandas_dtype
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCSeries
from pandas.core.dtypes.missing import isna
from pandas.core import arraylike, missing, roperator
from pandas.core.algorithms import duplicated, factorize_array, isin, map_array, mode, rank, unique
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.missing import _fill_limit_area_1d
from pandas.core.sorting import nargminmax, nargsort
if TYPE_CHECKING:
    from pandas._libs.missing import NAType
    from pandas._typing import ArrayLike, AstypeArg, AxisInt, Dtype, DtypeObj, FillnaOptions, InterpolateOptions, NumpySorter, NumpyValueArrayLike, PositionalIndexer, ScalarIndexer, Self, SequenceIndexer, Shape, SortKind, TakeIndexer, npt
    from pandas import Index
_extension_array_shared_docs: dict[str, str] = {}

class ExtensionArray:
    """
    Abstract base class for custom 1-D array types.

    pandas will recognize instances of this class as proper arrays
    with a custom type and will not attempt to coerce them to objects. They
    may be stored directly inside a :class:`DataFrame` or :class:`Series`.

    Attributes
    ----------
    dtype
    nbytes
    ndim
    shape

    Methods
    -------
    argsort
    astype
    copy
    dropna
    duplicated
    factorize
    fillna
    equals
    insert
    interpolate
    isin
    isna
    ravel
    repeat
    searchsorted
    shift
    take
    tolist
    unique
    view
    _accumulate
    _concat_same_type
    _explode
    _formatter
    _from_factorized
    _from_sequence
    _from_sequence_of_strings
    _hash_pandas_object
    _pad_or_backfill
    _reduce
    _values_for_argsort
    _values_for_factorize
    """
    _typ: ClassVar[str] = 'extension'
    __pandas_priority__: ClassVar[int] = 1000

    @classmethod
    def _from_sequence(cls, scalars: Sequence[Any], *, dtype: Any | None = None, copy: bool = False) -> ExtensionArray:
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_scalars(cls, scalars: Sequence[Any], *, dtype: ExtensionDtype) -> ExtensionArray:
        """
        Strict analogue to _from_sequence, allowing only sequences of scalars
        that should be specifically inferred to the given dtype.
        """
        try:
            return cls._from_sequence(scalars, dtype=dtype, copy=False)
        except (ValueError, TypeError):
            raise
        except Exception:
            warnings.warn('_from_scalars should only raise ValueError or TypeError. Consider overriding _from_scalars where appropriate.', stacklevel=find_stack_level())
            raise

    @classmethod
    def _from_sequence_of_strings(cls, strings: Sequence[str], *, dtype: ExtensionDtype, copy: bool = False) -> ExtensionArray:
        """
        Construct a new ExtensionArray from a sequence of strings.
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: ExtensionArray) -> ExtensionArray:
        """
        Reconstruct an ExtensionArray after factorization.
        """
        raise AbstractMethodError(cls)

    @overload
    def __getitem__(self, item: int) -> Any:
        ...

    @overload
    def __getitem__(self, item: slice | np.ndarray | list[int]) -> ExtensionArray:
        ...

    def __getitem__(self, item: Any) -> Any:
        """
        Select a subset of self.
        """
        raise AbstractMethodError(self)

    def __setitem__(self, key: int | np.ndarray | slice, value: Any) -> None:
        """
        Set one or more values inplace.
        """
        raise NotImplementedError(f'{type(self)} does not implement __setitem__.')

    def __len__(self) -> int:
        """
        Length of this array
        """
        raise AbstractMethodError(self)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over elements of the array.
        """
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: Any) -> bool:
        """
        Return for `item in self`.
        """
        if is_scalar(item) and isna(item):
            if not self._can_hold_na:
                return False
            elif item is self.dtype.na_value or isinstance(item, self.dtype.type):
                return self._hasna
            else:
                return False
        else:
            return (item == self).any()

    def __eq__(self, other: Any) -> ExtensionArray | np.ndarray:
        """
        Return for `self == other` (element-wise equality).
        """
        raise AbstractMethodError(self)

    def __ne__(self, other: Any) -> ExtensionArray | np.ndarray:
        """
        Return for `self != other` (element-wise in-equality).
        """
        return ~(self == other)

    def to_numpy(self, dtype: np.dtype[Any] | str | None = None, copy: bool = False, na_value: Any = lib.no_default) -> np.ndarray:
        """
        Convert to a NumPy ndarray.
        """
        result = np.asarray(self, dtype=dtype)
        if copy or na_value is not lib.no_default:
            result = result.copy()
        if na_value is not lib.no_default:
            result[self.isna()] = na_value
        return result

    @property
    def dtype(self) -> ExtensionDtype:
        """
        An instance of ExtensionDtype.
        """
        raise AbstractMethodError(self)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return a tuple of the array dimensions.
        """
        return (len(self),)

    @property
    def size(self) -> int:
        """
        The number of elements in the array.
        """
        return int(np.prod(self.shape))

    @property
    def ndim(self) -> int:
        """
        Extension Arrays are only allowed to be 1-dimensional.
        """
        return 1

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        raise AbstractMethodError(self)

    @overload
    def astype(self, dtype: Any, copy: bool = ...) -> np.ndarray | ExtensionArray:
        ...

    @overload
    def astype(self, dtype: Any, copy: bool = ...) -> np.ndarray | ExtensionArray:
        ...

    @overload
    def astype(self, dtype: Any, copy: bool = ...) -> np.ndarray | ExtensionArray:
        ...

    def astype(self, dtype: Any, copy: bool = True) -> np.ndarray | ExtensionArray:
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.
        """
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if not copy:
                return self
            else:
                return self.copy()
        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            return cls._from_sequence(self, dtype=dtype, copy=copy)
        elif lib.is_np_dtype(dtype, 'M'):
            from pandas.core.arrays import DatetimeArray
            return DatetimeArray._from_sequence(self, dtype=dtype, copy=copy)
        elif lib.is_np_dtype(dtype, 'm'):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._from_sequence(self, dtype=dtype, copy=copy)
        if not copy:
            return np.asarray(self, dtype=dtype)
        else:
            return np.array(self, dtype=dtype, copy=copy)

    def isna(self) -> np.ndarray | ExtensionArray:
        """
        A 1-D array indicating if each value is missing.
        """
        raise AbstractMethodError(self)

    @property
    def _hasna(self) -> bool:
        """
        Equivalent to `self.isna().any()`.
        """
        return bool(self.isna().any())

    def _values_for_argsort(self) -> np.ndarray:
        """
        Return values for sorting.
        """
        return np.array(self)

    def argsort(self, *, ascending: bool = True, kind: str = 'quicksort', na_position: Literal['first', 'last'] = 'last', **kwargs: Any) -> np.ndarray:
        """
        Return the indices that would sort this array.
        """
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)
        values = self._values_for_argsort()
        return nargsort(values, kind=kind, ascending=ascending, na_position=na_position, mask=np.asarray(self.isna()))

    def argmin(self, skipna: bool = True) -> int:
        """
        Return the index of minimum value.
        """
        validate_bool_kwarg(skipna, 'skipna')
        if not skipna and self._hasna:
            raise ValueError('Encountered an NA value with skipna=False')
        return int(nargminmax(self, 'argmin'))

    def argmax(self, skipna: bool = True) -> int:
        """
        Return the index of maximum value.
        """
        validate_bool_kwarg(skipna, 'skipna')
        if not skipna and self._hasna:
            raise ValueError('Encountered an NA value with skipna=False')
        return int(nargminmax(self, 'argmax'))

    def interpolate(self, *, method: str, axis: int, index: Index, limit: int | None, limit_direction: Literal['forward', 'backward', 'both'], limit_area: Literal['inside', 'outside'] | None, copy: bool, **kwargs: Any) -> ExtensionArray:
        """
        Fill NaN values using an interpolation method.
        """
        raise NotImplementedError(f'{type(self).__name__} does not implement interpolate')

    def _pad_or_backfill(self, *, method: Literal['backfill', 'bfill', 'pad', 'ffill'], limit: int | None = None, limit_area: Literal['inside', 'outside'] | None = None, copy: bool = True) -> ExtensionArray:
        """
        Pad or backfill values, used by Series/DataFrame ffill and bfill.
        """
        mask = self.isna()
        if mask.any():
            meth = missing.clean_fill_method(method)
            npmask = np.asarray(mask)
            if limit_area is not None and (not npmask.all()):
                _fill_limit_area_1d(npmask, limit_area)
            if meth == 'pad':
                indexer = libalgos.get_fill_indexer(npmask, limit=limit)
                return self.take(indexer, allow_fill=True)
            else:
                indexer = libalgos.get_fill_indexer(npmask[::-1], limit=limit)[::-1]
                return self[::-1].take(indexer, allow_fill=True)
        else:
            if not copy:
                return self
            new_values = self.copy()
        return new_values

    def fillna(self, value: Any, limit: int | None = None, copy: bool = True) -> ExtensionArray:
        """
        Fill NA/NaN values using the specified method.
        """
        mask = self.isna()
        if limit is not None and limit < len(self):
            modify = mask.cumsum() > limit
            if modify.any():
                mask = mask.copy()
                mask[modify] = False
        value = missing.check_value_size(value, mask, len(self))
        if mask.any():
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
            new_values[mask] = value
        elif not copy:
            new_values = self[:]
        else:
            new_values = self.copy()
        return new_values

    def dropna(self) -> ExtensionArray:
        """
        Return ExtensionArray without NA values.
        """
        return self[~self.isna()]

    def duplicated(self, keep: Literal['first', 'last', False] = 'first') -> np.ndarray:
        """
        Return boolean ndarray denoting duplicate values.
        """
        mask = self.isna().astype(np.bool_, copy=False)
        return duplicated(values=self, keep=keep, mask=mask)

    def shift(self, periods: int = 1, fill_value: Any | None = None) -> ExtensionArray:
        """
        Shift values by desired number.
        """
        if not len(self) or periods == 0:
            return self.copy()
        if isna(fill_value):
            fill_value = self.dtype.na_value
        empty = self._from_sequence([fill_value] * min(abs(periods), len(self)), dtype=self.dtype)
        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods):]
            b = empty
        return self._concat_same_type([a, b])

    def unique(self) -> ExtensionArray:
        """
        Compute the ExtensionArray of unique values.
        """
        uniques = unique(self.astype(object))
        return self._from_sequence(uniques, dtype=self.dtype)

    def searchsorted(self, value: Any, side: Literal['left', 'right'] = 'left', sorter: np.ndarray | None = None) -> np.ndarray | int:
        """
        Find indices where elements should be inserted to maintain order.
        """
        arr = self.astype(object)
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        return arr.searchsorted(value, side=side, sorter=sorter)

    def equals(self, other: ExtensionArray) -> bool:
        """
        Return if another array is equivalent to this array.
        """
        if type(self) != type(other):
            return False
        other = cast(ExtensionArray, other)
        if self.dtype != other.dtype:
            return False
        elif len(self) != len(other):
            return False
        else:
            equal_values = self == other
            if isinstance(equal_values, ExtensionArray):
                equal_values = equal_values.fillna(False)
            equal_na = self.isna() & other.isna()
            return bool((equal_values | equal_na).all())

    def isin(self, values: np.ndarray | ExtensionArray) -> np.ndarray:
        """
        Pointwise comparison for set containment in the given values.
        """
        return isin(np.asarray(self), values)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """
        Return an array and missing value suitable for factorization.
        """
        return (self.astype(object), np.nan)

    def factorize(self, use_na_sentinel: bool = True) -> tuple[np.ndarray, ExtensionArray]:
        """
        Encode the extension array as an enumerated type.
        """
        arr, na_value = self._values_for_factorize()
        codes, uniques = factorize_array(arr, use_na_sentinel=use_na_sentinel, na_value=na_value)
        uniques_ea = self._from_factorized(uniques, self)
        return (codes, uniques_ea)
    _extension_array_shared_docs['repeat'] = "\n        Repeat elements of a %(klass)s.\n\n        Returns a new %(klass)s where each element of the current %(klass)s\n        is repeated consecutively a given number of times.\n\n        Parameters\n        ----------\n        repeats : int or array of ints\n            The number of repetitions for each element. This should be a\n            non-negative integer. Repeating 0 times will return an empty\n            %(klass)s.\n        axis : None\n            Must be ``None``. Has no effect but is accepted for compatibility\n            with numpy.\n\n        Returns\n        -------\n        %(klass)s\n            Newly created %(klass)s with repeated elements.\n\n        See Also\n        --------\n        Series.repeat : Equivalent function for Series.\n        Index.repeat : Equivalent function for Index.\n        numpy.repeat : Similar method for :class:`numpy.ndarray`.\n        ExtensionArray.take : Take arbitrary positions.\n\n        Examples\n        --------\n        >>> cat = pd.Categorical(['a', 'b', 'c'])\n        >>> cat\n        ['a', 'b', 'c']\n        Categories (3, object): ['a', 'b', 'c']\n        >>> cat.repeat(2)\n        ['a', 'a', 'b', 'b', 'c', 'c']\n        Categories (3, object): ['a', 'b', 'c']\n        >>> cat.repeat([1, 2, 3])\n        ['a', 'b', 'b', 'c', 'c', 'c']\n        Categories (3, object): ['a', 'b', 'c']\n        "

    @Substitution(klass='ExtensionArray')
    @Appender(_extension_array_shared_docs['repeat'])
    def repeat(self, repeats: int | Sequence[int] | np.ndarray, axis: None = None) -> ExtensionArray:
        nv.validate_repeat((), {'axis': axis})
        ind = np.arange(len(self)).repeat(repeats)  # type: ignore[arg-type]
        return self.take(ind)

    def take(self, indices: Sequence[int] | np.ndarray, *, allow_fill: bool = False, fill_value: Any | None = None) -> ExtensionArray:
        """
        Take elements from an array.
        """
        raise AbstractMethodError(self)

    def copy(self) -> ExtensionArray:
        """
        Return a copy of the array.
        """
        raise AbstractMethodError(self)

    def view(self, dtype: Any | None = None) -> ExtensionArray | np.ndarray:
        """
        Return a view on the array.
        """
        if dtype is not None:
            raise NotImplementedError(dtype)
        return self[:]

    def __repr__(self) -> str:
        if self.ndim > 1:
            return self._repr_2d()
        from pandas.io.formats.printing import format_object_summary
        data = format_object_summary(self, self._formatter(), indent_for_name=False).rstrip(', \n')
        class_name = f'<{type(self).__name__}>\n'
        footer = self._get_repr_footer()
        return f'{class_name}{data}\n{footer}'

    def _get_repr_footer(self) -> str:
        if self.ndim > 1:
            return f'Shape: {self.shape}, dtype: {self.dtype}'
        return f'Length: {len(self)}, dtype: {self.dtype}'

    def _repr_2d(self) -> str:
        from pandas.io.formats.printing import format_object_summary
        lines = [format_object_summary(x, self._formatter(), indent_for_name=False).rstrip(', \n') for x in self]
        data = ',\n'.join(lines)
        class_name = f'<{type(self).__name__}>'
        footer = self._get_repr_footer()
        return f'{class_name}\n[\n{data}\n]\n{footer}'

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str]:
        """
        Formatting function for scalar values.
        """
        if boxed:
            return str
        return repr

    def transpose(self, *axes: Any) -> ExtensionArray:
        """
        Return a transposed view on this array.
        """
        return self[:]

    @property
    def T(self) -> ExtensionArray:
        return self.transpose()

    def ravel(self, order: None | Literal['C', 'F', 'A', 'K'] = 'C') -> ExtensionArray:
        """
        Return a flattened view on this array.
        """
        return self

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[ExtensionArray]) -> ExtensionArray:
        """
        Concatenate multiple array of this dtype.
        """
        raise AbstractMethodError(cls)

    @cache_readonly
    def _can_hold_na(self) -> bool:
        return self.dtype._can_hold_na

    def _accumulate(self, name: Literal['cummin', 'cummax', 'cumsum', 'cumprod'], *, skipna: bool = True, **kwargs: Any) -> ExtensionArray:
        """
        Return an ExtensionArray performing an accumulation operation.
        """
        raise NotImplementedError(f'cannot perform {name} with type {self.dtype}')

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any) -> Any | np.ndarray:
        """
        Return a scalar result of performing the reduction operation.
        """
        meth = getattr(self, name, None)
        if meth is None:
            raise TypeError(f"'{type(self).__name__}' with dtype {self.dtype} does not support operation '{name}'")
        result = meth(skipna=skipna, **kwargs)
        if keepdims:
            if name in ['min', 'max']:
                result = self._from_sequence([result], dtype=self.dtype)
            else:
                result = np.array([result])
        return result

    def _values_for_json(self) -> np.ndarray:
        """
        Specify how to render our entries in to_json.
        """
        return np.asarray(self)

    def _hash_pandas_object(self, *, encoding: str, hash_key: str, categorize: bool) -> np.ndarray:
        """
        Hook for hash_pandas_object.
        """
        from pandas.core.util.hashing import hash_array
        values, _ = self._values_for_factorize()
        return hash_array(values, encoding=encoding, hash_key=hash_key, categorize=categorize)

    def _explode(self) -> tuple[ExtensionArray, np.ndarray]:
        """
        Transform each element of list-like to a row.
        """
        values = self.copy()
        counts = np.ones(shape=(len(self),), dtype=np.uint64)
        return (values, counts)

    def tolist(self) -> list[Any]:
        """
        Return a list of the values.
        """
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self)

    def delete(self, loc: int | slice | np.ndarray | Sequence[int]) -> ExtensionArray:
        indexer = np.delete(np.arange(len(self)), loc)
        return self.take(indexer)

    def insert(self, loc: int, item: Any) -> ExtensionArray:
        """
        Insert an item at the given position.
        """
        loc = validate_insert_loc(loc, len(self))
        item_arr = type(self)._from_sequence([item], dtype=self.dtype)
        return type(self)._concat_same_type([self[:loc], item_arr, self[loc:]])

    def _putmask(self, mask: np.ndarray, value: Any) -> None:
        """
        Analogue to np.putmask(self, mask, value)
        """
        if is_list_like(value):
            val = value[mask]
        else:
            val = value
        self[mask] = val

    def _where(self, mask: np.ndarray, value: Any) -> ExtensionArray:
        """
        Analogue to np.where(mask, self, value)
        """
        result = self.copy()
        if is_list_like(value):
            val = value[~mask]
        else:
            val = value
        result[~mask] = val
        return result

    def _rank(self, *, axis: int = 0, method: str = 'average', na_option: str = 'keep', ascending: bool = True, pct: bool = False) -> ExtensionArray | np.ndarray:
        """
        See Series.rank.__doc__.
        """
        if axis != 0:
            raise NotImplementedError
        return rank(self, axis=axis, method=method, na_option=na_option, ascending=ascending, pct=pct)

    @classmethod
    def _empty(cls, shape: tuple[int, ...], dtype: ExtensionDtype) -> ExtensionArray:
        """
        Create an ExtensionArray with the given shape and dtype.
        """
        obj = cls._from_sequence([], dtype=dtype)
        taker = np.broadcast_to(np.intp(-1), shape)
        result = obj.take(taker, allow_fill=True)
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(f"Default 'empty' implementation is invalid for dtype='{dtype}'")
        return result

    def _quantile(self, qs: np.ndarray, interpolation: str) -> ExtensionArray:
        """
        Compute the quantiles of self for each quantile in `qs`.
        """
        mask = np.asarray(self.isna())
        arr = np.asarray(self)
        fill_value = np.nan
        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)
        return type(self)._from_sequence(res_values)

    def _mode(self, dropna: bool = True) -> ExtensionArray:
        """
        Returns the mode(s) of the ExtensionArray.
        """
        return mode(self, dropna=dropna)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if any((isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)) for other in inputs)):
            return NotImplemented
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result
        if 'out' in kwargs:
            return arraylike.dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        if method == 'reduce':
            result = arraylike.dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
            if result is not NotImplemented:
                return result
        return arraylike.default_array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def map(self, mapper: Callable[[Any], Any] | dict[Any, Any] | ABCSeries, na_action: Literal[None, 'ignore'] = None) -> np.ndarray | "Index" | ExtensionArray:
        """
        Map values using an input mapping or function.
        """
        return map_array(self, mapper, na_action=na_action)

    def _groupby_op(self, *, how: str, has_dropped_na: bool, min_count: int, ngroups: int, ids: np.ndarray, **kwargs: Any) -> np.ndarray | ExtensionArray:
        """
        Dispatch GroupBy reduction or transformation operation.
        """
        from pandas.core.arrays.string_ import StringDtype
        from pandas.core.groupby.ops import WrappedCythonOp
        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)
        if isinstance(self.dtype, StringDtype):
            if op.how in ['prod', 'mean', 'median', 'cumsum', 'cumprod', 'std', 'sem', 'var', 'skew', 'kurt']:
                raise TypeError(f"dtype '{self.dtype}' does not support operation '{how}'")
            if op.how not in ['any', 'all']:
                op._get_cython_function(op.kind, op.how, np.dtype(object), False)
            npvalues = self.to_numpy(object, na_value=np.nan)
        else:
            raise NotImplementedError(f'function is not implemented for this dtype: {self.dtype}')
        res_values = op._cython_op_ndim_compat(npvalues, min_count=min_count, ngroups=ngroups, comp_ids=ids, mask=None, **kwargs)
        if op.how in op.cast_blocklist:
            return res_values
        if isinstance(self.dtype, StringDtype):
            dtype = self.dtype
            string_array_cls = dtype.construct_array_type()
            return string_array_cls._from_sequence(res_values, dtype=dtype)
        else:
            raise NotImplementedError

class ExtensionArraySupportsAnyAll(ExtensionArray):

    @overload
    def any(self, *, skipna: bool = ...) -> bool:
        ...

    @overload
    def any(self, *, skipna: bool) -> bool:
        ...

    def any(self, *, skipna: bool = True) -> bool:
        raise AbstractMethodError(self)

    @overload
    def all(self, *, skipna: bool = ...) -> bool:
        ...

    @overload
    def all(self, *, skipna: bool) -> bool:
        ...

    def all(self, *, skipna: bool = True) -> bool:
        raise AbstractMethodError(self)

class ExtensionOpsMixin:
    """
    A base class for linking the operators to their dunder names.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_arithmetic_method(cls, op: Callable[[Any, Any], Any]) -> Callable[..., Any]:
        raise AbstractMethodError(cls)

    @classmethod
    def _add_arithmetic_ops(cls) -> None:
        setattr(cls, '__add__', cls._create_arithmetic_method(operator.add))
        setattr(cls, '__radd__', cls._create_arithmetic_method(roperator.radd))
        setattr(cls, '__sub__', cls._create_arithmetic_method(operator.sub))
        setattr(cls, '__rsub__', cls._create_arithmetic_method(roperator.rsub))
        setattr(cls, '__mul__', cls._create_arithmetic_method(operator.mul))
        setattr(cls, '__rmul__', cls._create_arithmetic_method(roperator.rmul))
        setattr(cls, '__pow__', cls._create_arithmetic_method(operator.pow))
        setattr(cls, '__rpow__', cls._create_arithmetic_method(roperator.rpow))
        setattr(cls, '__mod__', cls._create_arithmetic_method(operator.mod))
        setattr(cls, '__rmod__', cls._create_arithmetic_method(roperator.rmod))
        setattr(cls, '__floordiv__', cls._create_arithmetic_method(operator.floordiv))
        setattr(cls, '__rfloordiv__', cls._create_arithmetic_method(roperator.rfloordiv))
        setattr(cls, '__truediv__', cls._create_arithmetic_method(operator.truediv))
        setattr(cls, '__rtruediv__', cls._create_arithmetic_method(roperator.rtruediv))
        setattr(cls, '__divmod__', cls._create_arithmetic_method(divmod))
        setattr(cls, '__rdivmod__', cls._create_arithmetic_method(roperator.rdivmod))

    @classmethod
    def _create_comparison_method(cls, op: Callable[[Any, Any], Any]) -> Callable[..., Any]:
        raise AbstractMethodError(cls)

    @classmethod
    def _add_comparison_ops(cls) -> None:
        setattr(cls, '__eq__', cls._create_comparison_method(operator.eq))
        setattr(cls, '__ne__', cls._create_comparison_method(operator.ne))
        setattr(cls, '__lt__', cls._create_comparison_method(operator.lt))
        setattr(cls, '__gt__', cls._create_comparison_method(operator.gt))
        setattr(cls, '__le__', cls._create_comparison_method(operator.le))
        setattr(cls, '__ge__', cls._create_comparison_method(operator.ge))

    @classmethod
    def _create_logical_method(cls, op: Callable[[Any, Any], Any]) -> Callable[..., Any]:
        raise AbstractMethodError(cls)

    @classmethod
    def _add_logical_ops(cls) -> None:
        setattr(cls, '__and__', cls._create_logical_method(operator.and_))
        setattr(cls, '__rand__', cls._create_logical_method(roperator.rand_))
        setattr(cls, '__or__', cls._create_logical_method(operator.or_))
        setattr(cls, '__ror__', cls._create_logical_method(roperator.ror_))
        setattr(cls, '__xor__', cls._create_logical_method(operator.xor))
        setattr(cls, '__rxor__', cls._create_logical_method(roperator.rxor))

class ExtensionScalarOpsMixin(ExtensionOpsMixin):
    """
    A mixin for defining ops on an ExtensionArray.

    It is assumed that the underlying scalar objects have the operators
    already defined.
    """

    @classmethod
    def _create_method(cls, op: Callable[[Any, Any], Any], coerce_to_dtype: bool = True, result_dtype: Any | None = None) -> Callable[..., Any]:
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.
        """

        def _binop(self: ExtensionArray, other: Any) -> Any:

            def convert_values(param: Any) -> Any:
                if isinstance(param, ExtensionArray) or is_list_like(param):
                    ovalues = param
                else:
                    ovalues = [param] * len(self)
                return ovalues
            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                return NotImplemented
            lvalues = self
            rvalues = convert_values(other)
            res = [op(a, b) for a, b in zip(lvalues, rvalues)]

            def _maybe_convert(arr: Any) -> Any:
                if coerce_to_dtype:
                    res_inner = maybe_cast_pointwise_result(arr, self.dtype, same_dtype=False)
                    if not isinstance(res_inner, type(self)):
                        res_inner = np.asarray(arr)
                else:
                    res_inner = np.asarray(arr, dtype=result_dtype)
                return res_inner
            if op.__name__ in {'divmod', 'rdivmod'}:
                a, b = zip(*res)
                return (_maybe_convert(a), _maybe_convert(b))
            return _maybe_convert(res)
        op_name = f'__{op.__name__}__'
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op: Callable[[Any, Any], Any]) -> Callable[..., Any]:
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op: Callable[[Any, Any], Any]) -> Callable[..., Any]:
        return cls._create_method(op, coerce_to_dtype=False, result_dtype=bool)