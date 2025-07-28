from __future__ import annotations

import textwrap
from typing import Any, Callable, Generic, Iterator, List, Optional, overload, TypeVar, Union
from typing import TYPE_CHECKING, cast, final
from typing_extensions import Literal

import numpy as np
from pandas._libs import lib
from pandas._typing import AxisInt, DtypeObj, IndexLabel, NDFrameT, Self, Shape, npt
from pandas.compat import PYPY
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly, doc
from pandas.core.dtypes.cast import can_hold_element
from pandas.core.dtypes.common import is_object_dtype, is_scalar
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndex, ABCMultiIndex, ABCSeries
from pandas.core.dtypes.missing import isna, remove_na_arraylike
from pandas.core import algorithms, nanops, ops
from pandas.core.accessor import DirNamesMixin
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike, extract_array

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Iterator as IteratorType
    from pandas._typing import DropKeep, NumpySorter, NumpyValueArrayLike, ScalarLike_co
    from pandas import DataFrame, Index, Series

_T = TypeVar("_T", bound=NDFrameT)

class PandasObject(DirNamesMixin):
    """
    Baseclass for various pandas objects.
    """

    _cache: dict[str, Any]

    @property
    def _constructor(self) -> type[Self]:
        """
        Class constructor (for this class it's just `__class__`).
        """
        return type(self)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular object.
        """
        return object.__repr__(self)

    def _reset_cache(self, key: Optional[Any] = None) -> None:
        """
        Reset cached properties. If ``key`` is passed, only clears that key.
        """
        if not hasattr(self, '_cache'):
            return
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __sizeof__(self) -> int:
        """
        Generates the total memory usage for an object that returns
        either a value or Series of values
        """
        memory_usage = getattr(self, 'memory_usage', None)
        if memory_usage:
            mem = memory_usage(deep=True)
            return int(mem if is_scalar(mem) else mem.sum())
        return super().__sizeof__()

class NoNewAttributesMixin:
    """
    Mixin which prevents adding new attributes.

    Prevents additional attributes via xxx.attribute = "something" after a
    call to `self.__freeze()`. Mainly used to prevent the user from using
    wrong attributes on an accessor (`Series.cat/.str/.dt`).

    If you really want to add a new attribute at a later time, you need to use
    `object.__setattr__(self, key, value)`.
    """

    def _freeze(self) -> None:
        """
        Prevents setting additional attributes.
        """
        object.__setattr__(self, '__frozen', True)

    def __setattr__(self, key: str, value: Any) -> None:
        if getattr(self, '__frozen', False) and (not (key == '_cache' or key in type(self).__dict__ or getattr(self, key, None) is not None)):
            raise AttributeError(f"You cannot add any new attribute '{key}'")
        object.__setattr__(self, key, value)

class SelectionMixin(Generic[_T]):
    """
    mixin implementing the selection & aggregation interface on a group-like
    object sub-classes need to define: obj, exclusions
    """
    _selection: Optional[Union[List[Any], tuple[Any, ...], ABCSeries, ABCIndex, np.ndarray]] = None
    _internal_names: List[str] = ['_cache', '__setstate__']
    _internal_names_set = set(_internal_names)

    # Attributes that should be provided by subclass:
    obj: _T
    exclusions: List[Any]

    @final
    @property
    def _selection_list(self) -> List[Any]:
        if not isinstance(self._selection, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            return [self._selection]  # type: ignore
        return list(self._selection)  # type: ignore

    @cache_readonly
    def _selected_obj(self) -> _T:
        if self._selection is None or isinstance(self.obj, ABCSeries):
            return self.obj
        else:
            return self.obj[self._selection]  # type: ignore

    @final
    @cache_readonly
    def ndim(self) -> int:
        return self._selected_obj.ndim  # type: ignore

    @final
    @cache_readonly
    def _obj_with_exclusions(self) -> _T:
        if isinstance(self.obj, ABCSeries):
            return self.obj
        if self._selection is not None:
            return self.obj[self._selection_list]  # type: ignore
        if len(self.exclusions) > 0:
            return self.obj._drop_axis(self.exclusions, axis=1, only_slice=True)  # type: ignore
        else:
            return self.obj

    def __getitem__(self, key: Any) -> _T:
        if self._selection is not None:
            raise IndexError(f'Column(s) {self._selection} already selected')
        if isinstance(key, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            if len(self.obj.columns.intersection(key)) != len(set(key)):  # type: ignore
                bad_keys = list(set(key).difference(self.obj.columns))  # type: ignore
                raise KeyError(f'Columns not found: {str(bad_keys)[1:-1]}')
            return self._gotitem(list(key), ndim=2)
        else:
            if key not in self.obj:  # type: ignore
                raise KeyError(f'Column not found: {key}')
            ndim = self.obj[key].ndim  # type: ignore
            return self._gotitem(key, ndim=ndim)

    def _gotitem(self, key: Any, ndim: int, subset: Optional[Any] = None) -> _T:
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        raise AbstractMethodError(self)

    @final
    def _infer_selection(self, key: Any, subset: _T) -> Optional[Any]:
        """
        Infer the `selection` to pass to our constructor in _gotitem.
        """
        selection = None
        if subset.ndim == 2 and (lib.is_scalar(key) and key in subset or lib.is_list_like(key)):
            selection = key
        elif subset.ndim == 1 and lib.is_scalar(key) and (key == subset.name):  # type: ignore
            selection = key
        return selection

    def aggregate(self, func: Any, *args: Any, **kwargs: Any) -> _T:
        raise AbstractMethodError(self)

    agg = aggregate

class IndexOpsMixin(OpsMixin):
    """
    Common ops mixin to support a unified interface / docs for Series / Index
    """
    __array_priority__ = 1000
    _hidden_attrs = frozenset(['tolist'])

    @property
    def dtype(self) -> DtypeObj:
        raise AbstractMethodError(self)

    @property
    def _values(self) -> Any:
        raise AbstractMethodError(self)

    @final
    def transpose(self, *args: Any, **kwargs: Any) -> Self:
        """
        Return the transpose, which is by definition self.

        Returns
        -------
        %(klass)s
        """
        nv.validate_transpose(args, kwargs)
        return self  # type: ignore

    T = property(transpose, doc="\n        Return the transpose, which is by definition self.\n\n        See Also\n        --------\n        Index : Immutable sequence used for indexing and alignment.\n\n        Examples\n        --------\n        For Series:\n\n        >>> s = pd.Series(['Ant', 'Bear', 'Cow'])\n        >>> s\n        0     Ant\n        1    Bear\n        2     Cow\n        dtype: object\n        >>> s.T\n        0     Ant\n        1    Bear\n        2     Cow\n        dtype: object\n\n        For Index:\n\n        >>> idx = pd.Index([1, 2, 3])\n        >>> idx.T\n        Index([1, 2, 3], dtype='int64')\n        ")

    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the shape of the underlying data.

        See Also
        --------
        Series.ndim : Number of dimensions of the underlying data.
        Series.size : Return the number of elements in the underlying data.
        Series.nbytes : Return the number of bytes in the underlying data.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s.shape
        (3,)
        """
        return self._values.shape

    def __len__(self) -> int:
        raise AbstractMethodError(self)

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the underlying data, by definition 1.

        See Also
        --------
        Series.size: Return the number of elements in the underlying data.
        Series.shape: Return a tuple of the shape of the underlying data.
        Series.dtype: Return the dtype object of the underlying data.
        Series.values: Return Series as ndarray or ndarray-like depending on the dtype.

        Examples
        --------
        >>> s = pd.Series(["Ant", "Bear", "Cow"])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.ndim
        1

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.ndim
        1
        """
        return 1

    @final
    def item(self) -> Any:
        """
        Return the first element of the underlying data as a Python scalar.

        Returns
        -------
        scalar
            The first element of Series or Index.

        Raises
        ------
        ValueError
            If the data is not length = 1.

        See Also
        --------
        Index.values : Returns an array representing the data in the Index.
        Series.head : Returns the first `n` rows.

        Examples
        --------
        >>> s = pd.Series([1])
        >>> s.item()
        1

        For an index:

        >>> s = pd.Series([1], index=["a"])
        >>> s.index.item()
        'a'
        """
        if len(self) == 1:
            return next(iter(self))
        raise ValueError('can only convert an array of size 1 to a Python scalar')

    @property
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.

        See Also
        --------
        Series.ndim : Number of dimensions of the underlying data.
        Series.size : Return the number of elements in the underlying data.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["Ant", "Bear", "Cow"])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.nbytes
        24

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.nbytes
        24
        """
        return self._values.nbytes

    @property
    def size(self) -> int:
        """
        Return the number of elements in the underlying data.

        See Also
        --------
        Series.ndim: Number of dimensions of the underlying data, by definition 1.
        Series.shape: Return a tuple of the shape of the underlying data.
        Series.dtype: Return the dtype object of the underlying data.
        Series.values: Return Series as ndarray or ndarray-like depending on the dtype.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["Ant", "Bear", "Cow"])
        >>> s
        0     Ant
        1    Bear
        2     Cow
        dtype: object
        >>> s.size
        3

        For Index:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.size
        3
        """
        return len(self._values)

    @property
    def array(self) -> ExtensionArray[Any]:
        """
        The ExtensionArray of the data backing this Series or Index.

        This property provides direct access to the underlying array data of a
        Series or Index without requiring conversion to a NumPy array. It
        returns an ExtensionArray, which is the native storage format for
        pandas extension dtypes.

        Returns
        -------
        ExtensionArray
            An ExtensionArray of the values stored within. For extension
            types, this is the actual array. For NumPy native types, this
            is a thin (no copy) wrapper around :class:`numpy.ndarray`.

            ``.array`` differs from ``.values``, which may require converting
            the data to a different form.

        See Also
        --------
        Index.to_numpy : Similar method that always returns a NumPy array.
        Series.to_numpy : Similar method that always returns a NumPy array.

        Notes
        -----
        This table lays out the different array types for each extension
        dtype within pandas.

        ================== =============================
        dtype              array type
        ================== =============================
        category           Categorical
        period             PeriodArray
        interval           IntervalArray
        IntegerNA          IntegerArray
        string             StringArray
        boolean            BooleanArray
        datetime64[ns, tz] DatetimeArray
        ================== =============================
        """
        raise AbstractMethodError(self)

    def to_numpy(self, dtype: Optional[Union[str, np.dtype]] = None, copy: bool = False,
                 na_value: Any = lib.no_default, **kwargs: Any) -> np.ndarray:
        """
        A NumPy ndarray representing the values in this Series or Index.
        """
        if isinstance(self.dtype, ExtensionDtype):
            return self.array.to_numpy(dtype, copy=copy, na_value=na_value, **kwargs)
        elif kwargs:
            bad_keys = next(iter(kwargs.keys()))
            raise TypeError(f"to_numpy() got an unexpected keyword argument '{bad_keys}'")
        fillna = na_value is not lib.no_default and (not (na_value is np.nan and np.issubdtype(self.dtype, np.floating)))  # type: ignore
        values = self._values
        if fillna and self.hasnans:
            if not can_hold_element(values, na_value):
                values = np.asarray(values, dtype=dtype)
            else:
                values = values.copy()
            values[np.asanyarray(isna(self))] = na_value
        result = np.asarray(values, dtype=dtype)
        if copy and (not fillna) or not copy:
            if np.shares_memory(self._values[:2], result[:2]):
                if not copy:
                    result = result.view()
                    result.flags.writeable = False
                else:
                    result = result.copy()
        return result

    @final
    @property
    def empty(self) -> bool:
        """
        Indicator whether Index is empty.
        """
        return not self.size

    @doc(op='max', oppose='min', value='largest')
    def argmax(self, axis: Optional[Any] = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int:
        """
        Return int position of the {value} value in the Series.
        """
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)
        if isinstance(delegate, ExtensionArray):
            return delegate.argmax(skipna=skipna)
        else:
            result = nanops.nanargmax(delegate, skipna=skipna)
            return result

    @doc(argmax, op='min', oppose='max', value='smallest')
    def argmin(self, axis: Optional[Any] = None, skipna: bool = True, *args: Any, **kwargs: Any) -> int:
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)
        if isinstance(delegate, ExtensionArray):
            return delegate.argmin(skipna=skipna)
        else:
            result = nanops.nanargmin(delegate, skipna=skipna)
            return result

    def tolist(self) -> List[Any]:
        """
        Return a list of the values.
        """
        return self._values.tolist()

    to_list = tolist

    def __iter__(self) -> Iterator[Any]:
        """
        Return an iterator of the values.
        """
        if not isinstance(self._values, np.ndarray):
            return iter(self._values)
        else:
            return map(self._values.item, range(self._values.size))

    @cache_readonly
    def hasnans(self) -> bool:
        """
        Return True if there are any NaNs.
        """
        return bool(isna(self).any())

    @final
    def _map_values(self, mapper: Union[Callable[[Any], Any], dict[Any, Any]], 
                    na_action: Optional[str] = None) -> Any:
        """
        An internal function that maps values using the input
        correspondence (which can be a dict, Series, or function).

        Returns
        -------
        Union[Index, MultiIndex], inferred
        """
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.map(mapper, na_action=na_action)
        return algorithms.map_array(arr, mapper, na_action=na_action)

    def value_counts(self, normalize: bool = False, sort: bool = True, ascending: bool = False,
                     bins: Optional[int] = None, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of unique values.
        """
        return algorithms.value_counts_internal(self, sort=sort, ascending=ascending, normalize=normalize, bins=bins, dropna=dropna)

    def unique(self) -> Any:
        values = self._values
        if not isinstance(values, np.ndarray):
            result = values.unique()
        else:
            result = algorithms.unique1d(values)
        return result

    @final
    def nunique(self, dropna: bool = True) -> int:
        """
        Return number of unique elements in the object.
        """
        uniqs = self.unique()
        if dropna:
            uniqs = remove_na_arraylike(uniqs)
        return len(uniqs)

    @property
    def is_unique(self) -> bool:
        """
        Return True if values in the object are unique.
        """
        return self.nunique(dropna=False) == len(self)

    @property
    def is_monotonic_increasing(self) -> bool:
        """
        Return True if values in the object are monotonically increasing.
        """
        from pandas import Index
        return Index(self).is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return True if values in the object are monotonically decreasing.
        """
        from pandas import Index
        return Index(self).is_monotonic_decreasing

    @final
    def _memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of the values.
        """
        if hasattr(self.array, 'memory_usage'):
            return self.array.memory_usage(deep=deep)
        v = self.array.nbytes
        if deep and is_object_dtype(self.dtype) and (not PYPY):
            values = cast(np.ndarray, self._values)
            v += lib.memory_usage_of_objects(values)
        return v

    @doc(algorithms.factorize, values='', order='', size_hint='', sort=textwrap.dedent('            sort : bool, default False\n                Sort `uniques` and shuffle `codes` to maintain the\n                relationship.\n            '))
    def factorize(self, sort: bool = False, use_na_sentinel: bool = True) -> tuple[np.ndarray, Index]:
        codes, uniques = algorithms.factorize(self._values, sort=sort, use_na_sentinel=use_na_sentinel)
        if uniques.dtype == np.float16:
            uniques = uniques.astype(np.float32)
        if isinstance(self, ABCMultiIndex):
            uniques = self._constructor(uniques)  # type: ignore
        else:
            from pandas import Index
            try:
                uniques = Index(uniques, dtype=self.dtype)
            except NotImplementedError:
                uniques = Index(uniques)
        return (codes, uniques)
    _shared_docs: dict[str, str] = {}

    @overload
    def searchsorted(self, value: Any, side: Literal['left', 'right'] = 'left',
                       sorter: Optional[Union[Any, np.ndarray]] = None) -> int: ...
    @overload
    def searchsorted(self, value: List[Any], side: Literal['left', 'right'] = 'left',
                       sorter: Optional[Union[Any, np.ndarray]] = None) -> np.ndarray: ...

    @doc(_shared_docs.get('searchsorted', ""), klass='Index')
    def searchsorted(self, value: Any, side: Literal['left', 'right'] = 'left', sorter: Optional[Union[Any, np.ndarray]] = None) -> Union[int, np.ndarray]:
        if isinstance(value, ABCDataFrame):
            msg = f'Value must be 1-D array-like or scalar, {type(value).__name__} is not supported'
            raise ValueError(msg)
        values = self._values
        if not isinstance(values, np.ndarray):
            return values.searchsorted(value, side=side, sorter=sorter)
        return algorithms.searchsorted(values, value, side=side, sorter=sorter)

    def drop_duplicates(self, *, keep: str = 'first') -> Self:
        duplicated = self._duplicated(keep=keep)
        return self[~duplicated]

    @final
    def _duplicated(self, keep: str = 'first') -> np.ndarray:
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.duplicated(keep=keep)
        return algorithms.duplicated(arr, keep=keep)

    def _arith_method(self, other: Any, op: Any) -> Self:
        res_name = ops.get_op_result_name(self, other)
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        rvalues = ops.maybe_prepare_scalar_for_op(rvalues, lvalues.shape)
        rvalues = ensure_wrapped_if_datetimelike(rvalues)
        if isinstance(rvalues, range):
            rvalues = np.arange(rvalues.start, rvalues.stop, rvalues.step)
        with np.errstate(all='ignore'):
            result = ops.arithmetic_op(lvalues, rvalues, op)
        return self._construct_result(result, name=res_name)

    def _construct_result(self, result: Any, name: Any) -> Self:
        """
        Construct an appropriately-wrapped result from the ArrayLike result
        of an arithmetic-like operation.
        """
        raise AbstractMethodError(self)