"""
Base and utility classes for pandas objects.
"""
from __future__ import annotations
import textwrap
from typing import (
    TYPE_CHECKING, Any, Generic, Literal, cast, final, overload, 
    Union, Optional, List, Tuple, Dict, Set, FrozenSet, TypeVar, 
    Sequence, Iterable, Callable, Type, Mapping, Hashable
)
import numpy as np
from pandas._libs import lib
from pandas._typing import (
    AxisInt, DtypeObj, IndexLabel, NDFrameT, Self, Shape, npt,
    ScalarLike_co, NumpySorter, NumpyValueArrayLike, DropKeep
)
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
    from collections.abc import Hashable, Iterator
    from pandas import DataFrame, Index, Series

T = TypeVar('T')
NDFrameT_co = TypeVar('NDFrameT_co', covariant=True)

_shared_docs: Dict[str, str] = {}

class PandasObject(DirNamesMixin):
    """
    Baseclass for various pandas objects.
    """

    @property
    def _constructor(self) -> Type[Self]:
        """
        Class constructor (for this class it's just `__class__`).
        """
        return type(self)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular object.
        """
        return object.__repr__(self)

    def _reset_cache(self, key: Optional[Hashable] = None) -> None:
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
    """
    __frozen: bool

    def _freeze(self) -> None:
        """
        Prevents setting additional attributes.
        """
        object.__setattr__(self, '__frozen', True)

    def __setattr__(self, key: str, value: Any) -> None:
        if getattr(self, '__frozen', False) and (not (key == '_cache' or key in type(self).__dict__ or getattr(self, key, None) is not None)):
            raise AttributeError(f"You cannot add any new attribute '{key}'")
        object.__setattr__(self, key, value)

class SelectionMixin(Generic[NDFrameT]):
    """
    mixin implementing the selection & aggregation interface on a group-like
    object sub-classes need to define: obj, exclusions
    """
    _selection: Optional[Union[Hashable, Sequence[Hashable]]] = None
    _internal_names: List[str] = ['_cache', '__setstate__']
    _internal_names_set: Set[str] = set(_internal_names)

    @final
    @property
    def _selection_list(self) -> List[Hashable]:
        if not isinstance(self._selection, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            return [self._selection]
        return list(self._selection)

    @cache_readonly
    def _selected_obj(self) -> NDFrameT:
        if self._selection is None or isinstance(self.obj, ABCSeries):
            return self.obj
        else:
            return self.obj[self._selection]

    @final
    @cache_readonly
    def ndim(self) -> int:
        return self._selected_obj.ndim

    @final
    @cache_readonly
    def _obj_with_exclusions(self) -> NDFrameT:
        if isinstance(self.obj, ABCSeries):
            return self.obj
        if self._selection is not None:
            return self.obj[self._selection_list]
        if len(self.exclusions) > 0:
            return self.obj._drop_axis(self.exclusions, axis=1, only_slice=True)
        else:
            return self.obj

    def __getitem__(self, key: Union[Hashable, Sequence[Hashable]]) -> Any:
        if self._selection is not None:
            raise IndexError(f'Column(s) {self._selection} already selected')
        if isinstance(key, (list, tuple, ABCSeries, ABCIndex, np.ndarray)):
            if len(self.obj.columns.intersection(key)) != len(set(key)):
                bad_keys = list(set(key).difference(self.obj.columns))
                raise KeyError(f'Columns not found: {str(bad_keys)[1:-1]}')
            return self._gotitem(list(key), ndim=2)
        else:
            if key not in self.obj:
                raise KeyError(f'Column not found: {key}')
            ndim = self.obj[key].ndim
            return self._gotitem(key, ndim=ndim)

    def _gotitem(self, key: Union[Hashable, Sequence[Hashable]], ndim: int, subset: Optional[NDFrameT] = None) -> Any:
        raise AbstractMethodError(self)

    @final
    def _infer_selection(self, key: Hashable, subset: NDFrameT) -> Optional[Hashable]:
        selection = None
        if subset.ndim == 2 and (lib.is_scalar(key) and key in subset or lib.is_list_like(key)):
            selection = key
        elif subset.ndim == 1 and lib.is_scalar(key) and (key == subset.name):
            selection = key
        return selection

    def aggregate(self, func: Union[str, Callable], *args: Any, **kwargs: Any) -> Any:
        raise AbstractMethodError(self)
    agg = aggregate

class IndexOpsMixin(OpsMixin):
    """
    Common ops mixin to support a unified interface / docs for Series / Index
    """
    __array_priority__: int = 1000
    _hidden_attrs: FrozenSet[str] = frozenset(['tolist'])

    @property
    def dtype(self) -> DtypeObj:
        raise AbstractMethodError(self)

    @property
    def _values(self) -> Union[ExtensionArray, np.ndarray]:
        raise AbstractMethodError(self)

    @final
    def transpose(self, *args: Any, **kwargs: Any) -> Self:
        nv.validate_transpose(args, kwargs)
        return self
    T = property(transpose)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._values.shape

    def __len__(self) -> int:
        raise AbstractMethodError(self)

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @final
    def item(self) -> Any:
        if len(self) == 1:
            return next(iter(self))
        raise ValueError('can only convert an array of size 1 to a Python scalar')

    @property
    def nbytes(self) -> int:
        return self._values.nbytes

    @property
    def size(self) -> int:
        return len(self._values)

    @property
    def array(self) -> ExtensionArray:
        raise AbstractMethodError(self)

    def to_numpy(
        self,
        dtype: Optional[Union[str, np.dtype]] = None,
        copy: bool = False,
        na_value: Any = lib.no_default,
        **kwargs: Any
    ) -> np.ndarray:
        if isinstance(self.dtype, ExtensionDtype):
            return self.array.to_numpy(dtype, copy=copy, na_value=na_value, **kwargs)
        elif kwargs:
            bad_keys = next(iter(kwargs.keys()))
            raise TypeError(f"to_numpy() got an unexpected keyword argument '{bad_keys}'")
        fillna = na_value is not lib.no_default and (not (na_value is np.nan and np.issubdtype(self.dtype, np.floating)))
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
        return not self.size

    @overload
    def argmax(self, axis: None = ..., skipna: bool = ..., *args: Any, **kwargs: Any) -> int:
        ...
    @overload
    def argmax(self, axis: int, skipna: bool = ..., *args: Any, **kwargs: Any) -> Union[int, np.ndarray]:
        ...
    def argmax(self, axis: Optional[int] = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Union[int, np.ndarray]:
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)
        if isinstance(delegate, ExtensionArray):
            return delegate.argmax(skipna=skipna)
        else:
            result = nanops.nanargmax(delegate, skipna=skipna)
            return result

    @overload
    def argmin(self, axis: None = ..., skipna: bool = ..., *args: Any, **kwargs: Any) -> int:
        ...
    @overload
    def argmin(self, axis: int, skipna: bool = ..., *args: Any, **kwargs: Any) -> Union[int, np.ndarray]:
        ...
    def argmin(self, axis: Optional[int] = None, skipna: bool = True, *args: Any, **kwargs: Any) -> Union[int, np.ndarray]:
        delegate = self._values
        nv.validate_minmax_axis(axis)
        skipna = nv.validate_argmax_with_skipna(skipna, args, kwargs)
        if isinstance(delegate, ExtensionArray):
            return delegate.argmin(skipna=skipna)
        else:
            result = nanops.nanargmin(delegate, skipna=skipna)
            return result

    def tolist(self) -> List[Any]:
        return self._values.tolist()
    to_list = tolist

    def __iter__(self) -> Iterator[Any]:
        if not isinstance(self._values, np.ndarray):
            return iter(self._values)
        else:
            return map(self._values.item, range(self._values.size))

    @cache_readonly
    def hasnans(self) -> bool:
        return bool(isna(self).any())

    @final
    def _map_values(
        self, 
        mapper: Union[Mapping, Callable], 
        na_action: Optional[Literal['ignore']] = None
    ) -> Union[Index, ABCMultiIndex]:
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.map(mapper, na_action=na_action)
        return algorithms.map_array(arr, mapper, na_action=na_action)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins: Optional[int] = None,
        dropna: bool = True
    ) -> ABCSeries:
        return algorithms.value_counts_internal(
            self, sort=sort, ascending=ascending, 
            normalize=normalize, bins=bins, dropna=dropna
        )

    def unique(self) -> Union[np.ndarray, ExtensionArray]:
        values = self._values
        if not isinstance(values, np.ndarray):
            result = values.unique()
        else:
            result = algorithms.unique1d(values)
        return result

    @final
    def nunique(self, dropna: bool = True) -> int:
        uniqs = self.unique()
        if dropna:
            uniqs = remove_na_arraylike(uniqs)
        return len(uniqs)

    @property
    def is_unique(self) -> bool:
        return self.nunique(dropna=False) == len(self)

    @property
    def is_monotonic_increasing(self) -> bool:
        from pandas import Index
        return Index(self).is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        from pandas import Index
        return Index(self).is_monotonic_decreasing

    @final
    def _memory_usage(self, deep: bool = False) -> int:
        if hasattr(self.array, 'memory_usage'):
            return self.array.memory_usage(deep=deep)
        v = self.array.nbytes
        if deep and is_object_dtype(self.dtype) and (not PYPY):
            values = cast(np.ndarray, self._values)
            v += lib.memory_usage_of_objects(values)
        return v

    def factorize(
        self, 
        sort: bool = False, 
        use_na_sentinel: bool = True
    ) -> Tuple[np.ndarray, Union[Index, ABCMultiIndex]]:
        codes, uniques = algorithms.factorize(
            self._values, sort=sort, use_na_sentinel=use_na_sentinel
        )
        if uniques.dtype == np.float16:
            uniques = uniques.astype(np.float32)
        if isinstance(self, ABCMultiIndex):
            uniques = self._constructor(uniques)
        else:
            from pandas import Index
            try:
                uniques = Index(uniques, dtype=self.dtype)
            except NotImplementedError:
                uniques = Index(uniques)
        return (codes, uniques)

    @overload
    def searchsorted(
        self, 
        value: ScalarLike_co, 
        side: Literal['left', 'right'] = ..., 
        sorter: Optional[NumpySorter] = ...
    ) -> int:
        ...
    @overload
    def searchsorted(
        self, 
        value: NumpyValueArrayLike, 
        side: Literal['left', 'right'] = ..., 
        sorter: Optional[NumpySorter] = ...
    ) -> np.ndarray:
        ...
    def searchsorted(
        self, 
        value: Union[ScalarLike_co, NumpyValueArrayLike], 
        side: Literal['left', 'right'] = 'left', 
        sorter: Optional[NumpySorter] = None
    ) -> Union[int, np.ndarray]:
        if isinstance(value, ABCDataFrame):
            msg = f'Value must be 1-D array-like or scalar, {type(value).__name__} is not supported'
            raise ValueError(msg)
        values = self._values
        if not isinstance(values, np.ndarray):
            return values.searchsorted(value, side=side, sorter=sorter)
        return algorithms.searchsorted(values, value, side=side, sorter=sorter)

    def drop_duplicates(self, *, keep: Literal['first', 'last', False] = 'first') -> Self:
        duplicated = self._duplicated(keep=keep)
        return self[~duplicated]

    @final
    def _duplicated(self, keep: Literal['first', 'last', False] = 'first') -> np.ndarray:
        arr = self._values
        if isinstance(arr, ExtensionArray):
            return arr.duplicated(keep=keep)
        return algorithms.duplicated(arr, keep=keep)

    def _arith_method(self, other: Any, op: Callable) -> Any:
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

    def _construct_result(self, result: Any, name: Optional[Hashable]) -> Any:
        raise AbstractMethodError(self)
