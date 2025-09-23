from __future__ import annotations
from contextlib import suppress
import sys
from typing import (
    Any,
    Optional,
    Sequence,
    Tuple,
    List,
    Union,
    Iterator,
    Hashable,
    TYPE_CHECKING,
    TypeVar,
    cast,
)
from typing_extensions import final
import warnings
import numpy as np
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import (
    AbstractMethodError,
    ChainedAssignmentError,
    IndexingError,
    InvalidIndexError,
    LossySetitemError,
)
from pandas.errors.cow import _chained_assignment_msg
from pandas.util._decorators import doc
from pandas.core.dtypes.cast import can_hold_element, maybe_promote
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_sequence,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import (
    construct_1d_array_from_inferred_fill_value,
    infer_fill_value,
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import array as pd_array, extract_array
from pandas.core.indexers import check_array_indexer, is_list_like_indexer, is_scalar_indexer, length_of_indexer
from pandas.core.indexes.api import Index, MultiIndex

if TYPE_CHECKING:
    from collections.abc import Iterator as CollectionsIterator
    from pandas._typing import Axis, AxisInt, npt
    from pandas import DataFrame, Series

T = TypeVar("T")
_NS = slice(None, None)
_one_ellipsis_message: str = "indexer may only contain one '...' entry"


class _IndexSlice:
    """
    Create an object to more easily perform multi-index slicing.

    See Also
    --------
    MultiIndex.remove_unused_levels : New MultiIndex with no unused levels.
    """
    def __getitem__(self, arg: Any) -> Any:
        return arg


IndexSlice: _IndexSlice = _IndexSlice()


class IndexingMixin:
    """
    Mixin for adding .loc/.iloc/.at/.iat to Dataframes and Series.
    """

    @property
    def iloc(self) -> _iLocIndexer:
        return _iLocIndexer("iloc", self)

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer("loc", self)

    @property
    def at(self) -> _AtIndexer:
        return _AtIndexer("at", self)

    @property
    def iat(self) -> _iAtIndexer:
        return _iAtIndexer("iat", self)


class _LocationIndexer(NDFrameIndexerBase):
    _valid_types: str
    axis: Optional[int] = None
    _takeable: bool

    @final
    def __call__(self, axis: Optional[AxisInt] = None) -> _LocationIndexer:
        new_self: _LocationIndexer = type(self)(self.name, self.obj)
        if axis is not None:
            axis_int_none: Optional[int] = self.obj._get_axis_number(axis)
        else:
            axis_int_none = axis
        new_self.axis = axis_int_none
        return new_self

    def _get_setitem_indexer(self, key: Any) -> Any:
        if self.name == "loc":
            self._ensure_listlike_indexer(key, axis=self.axis)
        if isinstance(key, tuple):
            for x in key:
                check_dict_or_set_indexers(x)
        if self.axis is not None:
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)
        ax = self.obj._get_axis(0)
        if isinstance(ax, MultiIndex) and self.name != "iloc" and is_hashable(key) and (not isinstance(key, slice)):
            with suppress(KeyError, InvalidIndexError):
                return ax.get_loc(key)
        if isinstance(key, tuple):
            with suppress(IndexingError):
                return self._convert_tuple(key)
        if isinstance(key, range):
            key = list(key)
        return self._convert_to_indexer(key, axis=0)

    @final
    def _maybe_mask_setitem_value(self, indexer: Any, value: Any) -> Tuple[Any, Any]:
        if isinstance(indexer, tuple) and len(indexer) == 2 and isinstance(value, (ABCSeries, ABCDataFrame)):
            (pi, icols) = indexer
            ndim: int = value.ndim
            if com.is_bool_indexer(pi) and len(value) == len(pi):
                newkey = pi.nonzero()[0]
                if is_scalar_indexer(icols, self.ndim - 1) and ndim == 1:
                    if len(newkey) == 0:
                        value = value.iloc[:0]
                    else:
                        value = self.obj.iloc._align_series(indexer, value)
                    indexer = (newkey, icols)
                elif isinstance(icols, np.ndarray) and icols.dtype.kind == "i" and (len(icols) == 1):
                    if ndim == 1:
                        value = self.obj.iloc._align_series(indexer, value)
                        indexer = (newkey, icols)
                    elif ndim == 2 and value.shape[1] == 1:
                        if len(newkey) == 0:
                            value = value.iloc[:0]
                        else:
                            value = self.obj.iloc._align_frame(indexer, value)
                        indexer = (newkey, icols)
        elif com.is_bool_indexer(indexer):
            indexer = indexer.nonzero()[0]
        return (indexer, value)

    @final
    def _ensure_listlike_indexer(self, key: Any, axis: Optional[int] = None, value: Any = None) -> None:
        column_axis: int = 1
        if self.ndim != 2:
            return
        if isinstance(key, tuple) and len(key) > 1:
            if axis is None:
                axis = column_axis
            key = key[axis]
        if axis == column_axis and (not isinstance(self.obj.columns, MultiIndex)) and is_list_like_indexer(key) and (not com.is_bool_indexer(key)) and all((is_hashable(k) for k in key)):
            keys = self.obj.columns.union(key, sort=False)
            diff = Index(key).difference(self.obj.columns, sort=False)
            if len(diff):
                indexer = np.arange(len(keys), dtype=np.intp)
                indexer[len(self.obj.columns):] = -1
                new_mgr = self.obj._mgr.reindex_indexer(keys, indexer=indexer, axis=0, only_slice=True, use_na_proxy=True)
                self.obj._mgr = new_mgr
                return
            self.obj._mgr = self.obj._mgr.reindex_axis(keys, axis=0, only_slice=True)

    @final
    def __setitem__(self, key: Any, value: Any) -> None:
        if not PYPY:
            if sys.getrefcount(self.obj) <= 2:
                warnings.warn(_chained_assignment_msg, ChainedAssignmentError, stacklevel=2)
        check_dict_or_set_indexers(key)
        if isinstance(key, tuple):
            key = tuple((list(x) if is_iterator(x) else x for x in key))
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
        else:
            maybe_callable: Any = com.apply_if_callable(key, self.obj)
            key = self._raise_callable_usage(key, maybe_callable)
        indexer = self._get_setitem_indexer(key)
        self._has_valid_setitem_indexer(key)
        iloc: _iLocIndexer = cast("_iLocIndexer", self) if self.name == "iloc" else self.obj.iloc
        iloc._setitem_with_indexer(indexer, value, self.name)

    def _validate_key(self, key: Any, axis: int) -> None:
        raise AbstractMethodError(self)

    @final
    def _expand_ellipsis(self, tup: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if any((x is Ellipsis for x in tup)):
            if tup.count(Ellipsis) > 1:
                raise IndexingError(_one_ellipsis_message)
            if len(tup) == self.ndim:
                i = tup.index(Ellipsis)
                new_key = tup[:i] + (_NS,) + tup[i + 1 :]
                return new_key
        return tup

    @final
    def _validate_tuple_indexer(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        key = self._validate_key_length(key)
        key = self._expand_ellipsis(key)
        for i, k in enumerate(key):
            try:
                self._validate_key(k, i)
            except ValueError as err:
                raise ValueError(f"Location based indexing can only have [{self._valid_types}] types") from err
        return key

    @final
    def _is_nested_tuple_indexer(self, tup: Tuple[Any, ...]) -> bool:
        if any((isinstance(ax, MultiIndex) for ax in self.obj.axes)):
            return any((is_nested_tuple(tup, ax) for ax in self.obj.axes))
        return False

    @final
    def _convert_tuple(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        self._validate_key_length(key)
        keyidx = [self._convert_to_indexer(k, axis=i) for i, k in enumerate(key)]
        return tuple(keyidx)

    @final
    def _validate_key_length(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if len(key) > self.ndim:
            if key[0] is Ellipsis:
                key = key[1:]
                if Ellipsis in key:
                    raise IndexingError(_one_ellipsis_message)
                return self._validate_key_length(key)
            raise IndexingError("Too many indexers")
        return key

    @final
    def _getitem_tuple_same_dim(self, tup: Tuple[Any, ...]) -> Any:
        retval: Any = self.obj
        start_val: int = self.ndim - len(tup) + 1
        for i_rev, key in enumerate(reversed(tup)):
            i_val = self.ndim - i_rev - start_val
            if com.is_null_slice(key):
                continue
            retval = getattr(retval, self.name)._getitem_axis(key, axis=i_val)
            assert retval.ndim == self.ndim
        if retval is self.obj:
            retval = retval.copy(deep=False)
        return retval

    @final
    def _getitem_lowerdim(self, tup: Tuple[Any, ...]) -> Any:
        if self.axis is not None:
            axis = self.obj._get_axis_number(self.axis)
            return self._getitem_axis(tup, axis=axis)
        if self._is_nested_tuple_indexer(tup):
            return self._getitem_nested_tuple(tup)
        ax0 = self.obj._get_axis(0)
        if isinstance(ax0, MultiIndex) and self.name != "iloc" and (not any((isinstance(x, slice) for x in tup))):
            with suppress(IndexingError):
                return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)
        tup = self._validate_key_length(tup)
        for i, key in enumerate(tup):
            if is_label_like(key):
                section = self._getitem_axis(key, axis=i)
                if section.ndim == self.ndim:
                    new_key = tup[:i] + (_NS,) + tup[i + 1 :]
                else:
                    new_key = tup[:i] + tup[i + 1 :]
                    if len(new_key) == 1:
                        new_key = new_key[0]
                if com.is_null_slice(new_key):
                    return section
                return getattr(section, self.name)[new_key]
        raise IndexingError("not applicable")

    @final
    def _getitem_nested_tuple(self, tup: Tuple[Any, ...]) -> Any:
        def _contains_slice(x: Any) -> bool:
            if isinstance(x, tuple):
                return any((isinstance(v, slice) for v in x))
            elif isinstance(x, slice):
                return True
            return False
        for key in tup:
            check_dict_or_set_indexers(key)
        if len(tup) > self.ndim:
            if self.name != "loc":
                raise ValueError("Too many indices")
            if all((is_hashable(x) and (not _contains_slice(x)) or com.is_null_slice(x) for x in tup)):
                with suppress(IndexingError):
                    return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)
            elif isinstance(self.obj, ABCSeries) and any((isinstance(k, tuple) for k in tup)):
                raise IndexingError("Too many indexers")
            axis: int = self.axis or 0
            return self._getitem_axis(tup, axis=axis)
        obj: Any = self.obj
        axis: int = len(tup) - 1
        for key in reversed(tup):
            if com.is_null_slice(key):
                axis -= 1
                continue
            obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
            axis -= 1
            if is_scalar(obj) or not hasattr(obj, "ndim"):
                break
        return obj

    def _convert_to_indexer(self, key: Any, axis: AxisInt) -> Any:
        raise AbstractMethodError(self)

    def _raise_callable_usage(self, key: Any, maybe_callable: T) -> T:
        if self.name == "iloc" and callable(key) and isinstance(maybe_callable, tuple):
            raise ValueError("Returning a tuple from a callable with iloc is not allowed.")
        return maybe_callable

    @final
    def __getitem__(self, key: Any) -> Any:
        check_dict_or_set_indexers(key)
        if type(key) is tuple:
            key = tuple((list(x) if is_iterator(x) else x for x in key))
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
            if self._is_scalar_access(key):
                return self.obj._get_value(*key, takeable=self._takeable)
            return self._getitem_tuple(key)
        else:
            axis: int = self.axis or 0
            maybe_callable: Any = com.apply_if_callable(key, self.obj)
            maybe_callable = self._raise_callable_usage(key, maybe_callable)
            return self._getitem_axis(maybe_callable, axis=axis)

    def _is_scalar_access(self, key: Tuple[Any, ...]) -> bool:
        raise NotImplementedError

    def _getitem_tuple(self, tup: Tuple[Any, ...]) -> Any:
        raise AbstractMethodError(self)

    def _getitem_axis(self, key: Any, axis: AxisInt) -> Any:
        raise NotImplementedError

    def _has_valid_setitem_indexer(self, indexer: Any) -> bool:
        raise AbstractMethodError(self)

    @final
    def _getbool_axis(self, key: Any, axis: AxisInt) -> Any:
        labels = self.obj._get_axis(axis)
        key = check_bool_indexer(labels, key)
        inds = key.nonzero()[0]
        return self.obj.take(inds, axis=axis)


@doc(IndexingMixin.loc)
class _LocIndexer(_LocationIndexer):
    _takeable: bool = False
    _valid_types: str = "labels (MUST BE IN THE INDEX), slices of labels (BOTH endpoints included! Can be slices of integers if the index is integers), listlike of labels, boolean"

    @doc(_LocationIndexer._validate_key)
    def _validate_key(self, key: Any, axis: int) -> None:
        ax = self.obj._get_axis(axis)
        if isinstance(key, bool) and (not (is_bool_dtype(ax.dtype) or ax.dtype.name == "boolean" or (isinstance(ax, MultiIndex) and is_bool_dtype(ax.get_level_values(0).dtype)))):
            raise KeyError(f"{key}: boolean label can not be used without a boolean index")
        if isinstance(key, slice) and (isinstance(key.start, bool) or isinstance(key.stop, bool)):
            raise TypeError(f"{key}: boolean values can not be used in a slice")

    def _has_valid_setitem_indexer(self, indexer: Any) -> bool:
        return True

    def _is_scalar_access(self, key: Tuple[Any, ...]) -> bool:
        if len(key) != self.ndim:
            return False
        for i, k in enumerate(key):
            if not is_scalar(k):
                return False
            ax = self.obj.axes[i]
            if isinstance(ax, MultiIndex):
                return False
            if isinstance(k, str) and ax._supports_partial_string_indexing:
                return False
            if not ax._index_as_unique:
                return False
        return True

    def _multi_take_opportunity(self, tup: Tuple[Any, ...]) -> bool:
        if not all((is_list_like_indexer(x) for x in tup)):
            return False
        return not any((com.is_bool_indexer(x) for x in tup))

    def _multi_take(self, tup: Tuple[Any, ...]) -> Any:
        d = {axis: self._get_listlike_indexer(key, axis) for key, axis in zip(tup, self.obj._AXIS_ORDERS)}
        return self.obj._reindex_with_indexers(d, allow_dups=True)

    def _getitem_iterable(self, key: Any, axis: AxisInt) -> Any:
        self._validate_key(key, axis)
        keyarr, indexer = self._get_listlike_indexer(key, axis)
        return self.obj._reindex_with_indexers({axis: [keyarr, indexer]}, allow_dups=True)

    def _getitem_tuple(self, tup: Tuple[Any, ...]) -> Any:
        with suppress(IndexingError):
            tup = self._expand_ellipsis(tup)
            return self._getitem_lowerdim(tup)
        tup = self._validate_tuple_indexer(tup)
        if self._multi_take_opportunity(tup):
            return self._multi_take(tup)
        return self._getitem_tuple_same_dim(tup)

    def _get_label(self, label: Any, axis: AxisInt) -> Any:
        return self.obj.xs(label, axis=axis)

    def _handle_lowerdim_multi_index_axis0(self, tup: Tuple[Any, ...]) -> Any:
        axis: int = self.axis or 0
        try:
            return self._get_label(tup, axis=axis)
        except KeyError as ek:
            if self.ndim < len(tup) <= self.obj.index.nlevels:
                raise ek
            raise IndexingError("No label returned") from ek

    def _getitem_axis(self, key: Any, axis: AxisInt) -> Any:
        key = item_from_zerodim(key)
        if is_iterator(key):
            key = list(key)
        if key is Ellipsis:
            key = slice(None)
        labels = self.obj._get_axis(axis)
        if isinstance(key, tuple) and isinstance(labels, MultiIndex):
            key = tuple(key)
        if isinstance(key, slice):
            self._validate_key(key, axis)
            return self._get_slice_axis(key, axis=axis)
        elif com.is_bool_indexer(key):
            return self._getbool_axis(key, axis=axis)
        elif is_list_like_indexer(key):
            if not (isinstance(key, tuple) and isinstance(labels, MultiIndex)):
                if hasattr(key, "ndim") and key.ndim > 1:
                    raise ValueError("Cannot index with multidimensional key")
                return self._getitem_iterable(key, axis=axis)
            if is_nested_tuple(key, labels):
                locs = labels.get_locs(key)
                indexer: List[Union[slice, np.ndarray]] = [slice(None)] * self.ndim
                indexer[axis] = locs
                return self.obj.iloc[tuple(indexer)]
        self._validate_key(key, axis)
        return self._get_label(key, axis=axis)

    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt) -> Any:
        obj = self.obj
        if not need_slice(slice_obj):
            return obj.copy(deep=False)
        labels = obj._get_axis(axis)
        indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop, slice_obj.step)
        if isinstance(indexer, slice):
            return self.obj._slice(indexer, axis=axis)
        else:
            return self.obj.take(indexer, axis=axis)

    def _convert_to_indexer(self, key: Any, axis: AxisInt) -> Any:
        labels = self.obj._get_axis(axis)
        if isinstance(key, slice):
            return labels._convert_slice_indexer(key, kind="loc")
        if isinstance(key, tuple) and (not isinstance(labels, MultiIndex)) and (self.ndim < 2) and (len(key) > 1):
            raise IndexingError("Too many indexers")
        contains_slice: bool = False
        if isinstance(key, tuple):
            contains_slice = any((isinstance(v, slice) for v in key))
        if is_scalar(key) or (isinstance(labels, MultiIndex) and is_hashable(key) and (not contains_slice)):
            try:
                return labels.get_loc(key)
            except LookupError:
                if isinstance(key, tuple) and isinstance(labels, MultiIndex):
                    if len(key) == labels.nlevels:
                        return {"key": key}
                    raise
            except InvalidIndexError:
                if not isinstance(labels, MultiIndex):
                    raise
            except ValueError:
                if not is_integer(key):
                    raise
                return {"key": key}
        if is_nested_tuple(key, labels):
            if self.ndim == 1 and any((isinstance(k, tuple) for k in key)):
                raise IndexingError("Too many indexers")
            return labels.get_locs(key)
        elif is_list_like_indexer(key):
            if is_iterator(key):
                key = list(key)
            if com.is_bool_indexer(key):
                key = check_bool_indexer(labels, key)
                return key
            else:
                return self._get_listlike_indexer(key, axis)[1]
        else:
            try:
                return labels.get_loc(key)
            except LookupError:
                if not is_list_like_indexer(key):
                    return {"key": key}
                raise

    def _get_listlike_indexer(self, key: Any, axis: AxisInt) -> Tuple[Any, Any]:
        ax = self.obj._get_axis(axis)
        axis_name: Any = self.obj._get_axis_name(axis)
        keyarr, indexer = ax._get_indexer_strict(key, axis_name)
        return (keyarr, indexer)


@doc(IndexingMixin.iloc)
class _iLocIndexer(_LocationIndexer):
    _valid_types: str = "integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array"
    _takeable: bool = True

    def _validate_key(self, key: Any, axis: AxisInt) -> None:
        if com.is_bool_indexer(key):
            if hasattr(key, "index") and isinstance(key.index, Index):
                if key.index.inferred_type == "integer":
                    raise NotImplementedError("iLocation based boolean indexing on an integer type is not available")
                raise ValueError("iLocation based boolean indexing cannot use an indexable as a mask")
            return
        if isinstance(key, slice):
            return
        elif is_integer(key):
            self._validate_integer(key, axis)
        elif isinstance(key, tuple):
            raise IndexingError("Too many indexers")
        elif is_list_like_indexer(key):
            if isinstance(key, ABCSeries):
                arr = key._values
            elif is_array_like(key):
                arr = key
            else:
                arr = np.array(key)
            len_axis: int = len(self.obj._get_axis(axis))
            if not is_numeric_dtype(arr.dtype):
                raise IndexError(f".iloc requires numeric indexers, got {arr}")
            if len(arr) and (arr.max() >= len_axis or arr.min() < -len_axis):
                raise IndexError("positional indexers are out-of-bounds")
        else:
            raise ValueError(f"Can only index by location with a [{self._valid_types}]")

    def _has_valid_setitem_indexer(self, indexer: Any) -> bool:
        if isinstance(indexer, dict):
            raise IndexError("iloc cannot enlarge its target object")
        if isinstance(indexer, ABCDataFrame):
            raise TypeError("DataFrame indexer for .iloc is not supported. Consider using .loc with a DataFrame indexer for automatic alignment.")
        if not isinstance(indexer, tuple):
            indexer = _tuplify(self.ndim, indexer)
        for ax, i in zip(self.obj.axes, indexer):
            if isinstance(i, slice):
                pass
            elif is_list_like_indexer(i):
                pass
            elif is_integer(i):
                if i >= len(ax):
                    raise IndexError("iloc cannot enlarge its target object")
            elif isinstance(i, dict):
                raise IndexError("iloc cannot enlarge its target object")
        return True

    def _is_scalar_access(self, key: Tuple[Any, ...]) -> bool:
        if len(key) != self.ndim:
            return False
        return all((is_integer(k) for k in key))

    def _validate_integer(self, key: Union[int, np.integer], axis: AxisInt) -> None:
        len_axis: int = len(self.obj._get_axis(axis))
        if key >= len_axis or key < -len_axis:
            raise IndexError("single positional indexer is out-of-bounds")

    def _getitem_tuple(self, tup: Tuple[Any, ...]) -> Any:
        tup = self._validate_tuple_indexer(tup)
        with suppress(IndexingError):
            return self._getitem_lowerdim(tup)
        return self._getitem_tuple_same_dim(tup)

    def _get_list_axis(self, key: Any, axis: AxisInt) -> Any:
        try:
            return self.obj.take(key, axis=axis)
        except IndexError as err:
            raise IndexError("positional indexers are out-of-bounds") from err

    def _getitem_axis(self, key: Any, axis: AxisInt) -> Any:
        if key is Ellipsis:
            key = slice(None)
        elif isinstance(key, ABCDataFrame):
            raise IndexError("DataFrame indexer is not allowed for .iloc\nConsider using .loc for automatic alignment.")
        if isinstance(key, slice):
            return self._get_slice_axis(key, axis=axis)
        if is_iterator(key):
            key = list(key)
        if isinstance(key, list):
            key = np.asarray(key)
        if com.is_bool_indexer(key):
            self._validate_key(key, axis)
            return self._getbool_axis(key, axis=axis)
        elif is_list_like_indexer(key):
            return self._get_list_axis(key, axis=axis)
        else:
            key = item_from_zerodim(key)
            if not is_integer(key):
                raise TypeError("Cannot index by location index with a non-integer key")
            self._validate_integer(key, axis)
            return self.obj._ixs(key, axis=axis)

    def _get_slice_axis(self, slice_obj: slice, axis: AxisInt) -> Any:
        obj = self.obj
        if not need_slice(slice_obj):
            return obj.copy(deep=False)
        labels = obj._get_axis(axis)
        labels._validate_positional_slice(slice_obj)
        return self.obj._slice(slice_obj, axis=axis)

    def _convert_to_indexer(self, key: T, axis: AxisInt) -> T:
        return key

    def _get_setitem_indexer(self, key: Any) -> Any:
        if is_iterator(key):
            key = list(key)
        if self.axis is not None:
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)
        return key

    def _setitem_with_indexer(self, indexer: Any, value: Any, name: str = "iloc") -> None:
        info_axis: int = self.obj._info_axis_number
        take_split_path: bool = not self.obj._mgr.is_single_block
        if not take_split_path and isinstance(value, ABCDataFrame):
            take_split_path = not value._mgr.is_single_block
        if not take_split_path and len(self.obj._mgr.blocks) and (self.ndim > 1):
            val = list(value.values()) if isinstance(value, dict) else value
            arr = self.obj._mgr.blocks[0].values
            take_split_path = not can_hold_element(arr, extract_array(val, extract_numpy=True))
        if isinstance(indexer, tuple) and len(indexer) == len(self.obj.axes):
            for i, ax in zip(indexer, self.obj.axes):
                if isinstance(ax, MultiIndex) and (not (is_integer(i) or com.is_null_slice(i))):
                    take_split_path = True
                    break
        if isinstance(indexer, tuple):
            nindexer: List[Any] = []
            for i, idx in enumerate(indexer):
                if isinstance(idx, dict):
                    key_val, _ = convert_missing_indexer(idx)
                    if self.ndim > 1 and i == info_axis:
                        if not len(self.obj):
                            if not is_list_like_indexer(value):
                                raise ValueError("cannot set a frame with no defined index and a scalar")
                            self.obj[key_val] = value
                            return
                        if com.is_null_slice(indexer[0]):
                            self.obj[key_val] = value
                            return
                        elif is_array_like(value):
                            arr = extract_array(value, extract_numpy=True)
                            taker = -1 * np.ones(len(self.obj), dtype=np.intp)
                            empty_value = algos.take_nd(arr, taker)
                            if not isinstance(value, ABCSeries):
                                if isinstance(arr, np.ndarray) and arr.ndim == 1 and (len(arr) == 1):
                                    arr = arr[0, ...]
                                empty_value[indexer[0]] = arr
                                self.obj[key_val] = empty_value
                                return
                            self.obj[key_val] = empty_value
                        elif not is_list_like(value):
                            self.obj[key_val] = construct_1d_array_from_inferred_fill_value(value, len(self.obj))
                        else:
                            self.obj[key_val] = infer_fill_value(value)
                        new_indexer = convert_from_missing_indexer_tuple(indexer, self.obj.axes)
                        self._setitem_with_indexer(new_indexer, value, name)
                        return
                    index = self.obj._get_axis(i)
                    labels = index.insert(len(index), key_val)
                    taker = np.arange(len(index) + 1, dtype=np.intp)
                    taker[-1] = -1
                    reindexers = {i: (labels, taker)}
                    new_obj = self.obj._reindex_with_indexers(reindexers, allow_dups=True)
                    self.obj._mgr = new_obj._mgr
                    nindexer.append(labels.get_loc(key_val))
                else:
                    nindexer.append(idx)
            indexer = tuple(nindexer)
        else:
            indexer, missing = convert_missing_indexer(indexer)
            if missing:
                self._setitem_with_indexer_missing(indexer, value)
                return
        if name == "loc":
            indexer, value = self._maybe_mask_setitem_value(indexer, value)
        if take_split_path:
            self._setitem_with_indexer_split_path(indexer, value, name)
        else:
            self._setitem_single_block(indexer, value, name)

    def _setitem_with_indexer_split_path(self, indexer: Any, value: Any, name: str) -> None:
        assert self.ndim == 2
        if not isinstance(indexer, tuple):
            indexer = _tuplify(self.ndim, indexer)
        if len(indexer) > self.ndim:
            raise IndexError("too many indices for array")
        if isinstance(indexer[0], np.ndarray) and indexer[0].ndim > 2:
            raise ValueError("Cannot set values with ndim > 2")
        if isinstance(value, ABCSeries) and name != "iloc" or isinstance(value, dict):
            from pandas import Series
            value = self._align_series(indexer, Series(value))
        info_axis: int = indexer[1]
        ilocs = self._ensure_iterable_column_indexer(info_axis)
        pi = indexer[0]
        lplane_indexer: int = length_of_indexer(pi, self.obj.index)
        if is_list_like_indexer(value) and getattr(value, "ndim", 1) > 0:
            if isinstance(value, ABCDataFrame):
                self._setitem_with_indexer_frame_value(indexer, value, name)
            elif np.ndim(value) == 2:
                self._setitem_with_indexer_2d_value(indexer, value)
            elif len(ilocs) == 1 and lplane_indexer == len(value) and (not is_scalar(pi)):
                self._setitem_single_column(ilocs[0], value, pi)
            elif len(ilocs) == 1 and 0 != lplane_indexer != len(value):
                if len(value) == 1 and (not is_integer(info_axis)):
                    return self._setitem_with_indexer((pi, info_axis[0]), value[0])
                raise ValueError("Must have equal len keys and value when setting with an iterable")
            elif lplane_indexer == 0 and len(value) == len(self.obj.index):
                pass
            elif self._is_scalar_access(indexer) and is_object_dtype(self.obj.dtypes._values[ilocs[0]]):
                self._setitem_single_column(indexer[1], value, pi)
            elif len(ilocs) == len(value):
                for loc, v in zip(ilocs, value):
                    self._setitem_single_column(loc, v, pi)
            elif len(ilocs) == 1 and com.is_null_slice(pi) and (len(self.obj) == 0):
                self._setitem_single_column(ilocs[0], value, pi)
            else:
                raise ValueError("Must have equal len keys and value when setting with an iterable")
        else:
            for loc in ilocs:
                self._setitem_single_column(loc, value, pi)

    def _setitem_with_indexer_2d_value(self, indexer: Any, value: Any) -> None:
        pi = indexer[0]
        ilocs = self._ensure_iterable_column_indexer(indexer[1])
        if not is_array_like(value):
            value = np.array(value, dtype=object)
        if len(ilocs) != value.shape[1]:
            raise ValueError("Must have equal len keys and value when setting with an ndarray")
        for i, loc in enumerate(ilocs):
            value_col = value[:, i]
            if is_object_dtype(value_col.dtype):
                value_col = value_col.tolist()
            self._setitem_single_column(loc, value_col, pi)

    def _setitem_with_indexer_frame_value(self, indexer: Any, value: DataFrame, name: str) -> None:
        ilocs = self._ensure_iterable_column_indexer(indexer[1])
        sub_indexer: List[Any] = list(indexer)
        pi = indexer[0]
        multiindex_indexer: bool = isinstance(self.obj.columns, MultiIndex)
        unique_cols: bool = value.columns.is_unique
        if name == "iloc":
            for i, loc in enumerate(ilocs):
                val = value.iloc[:, i]
                self._setitem_single_column(loc, val, pi)
        elif not unique_cols and value.columns.equals(self.obj.columns):
            for loc in ilocs:
                item = self.obj.columns[loc]
                if item in value:
                    sub_indexer[1] = item
                    val = self._align_series(tuple(sub_indexer), value.iloc[:, loc], multiindex_indexer)
                else:
                    val = np.nan
                self._setitem_single_column(loc, val, pi)
        elif not unique_cols:
            raise ValueError("Setting with non-unique columns is not allowed.")
        else:
            for loc in ilocs:
                item = self.obj.columns[loc]
                if item in value:
                    sub_indexer[1] = item
                    val = self._align_series(tuple(sub_indexer), value[item], multiindex_indexer, using_cow=True)
                else:
                    val = np.nan
                self._setitem_single_column(loc, val, pi)

    def _setitem_single_column(self, loc: int, value: Any, plane_indexer: Any) -> None:
        pi = plane_indexer
        is_full_setter: bool = com.is_null_slice(pi) or com.is_full_slice(pi, len(self.obj))
        is_null_setter: bool = com.is_empty_slice(pi) or (is_array_like(pi) and len(pi) == 0)
        if is_null_setter:
            return
        elif is_full_setter:
            try:
                self.obj._mgr.column_setitem(loc, plane_indexer, value, inplace_only=True)
            except (ValueError, TypeError, LossySetitemError) as exc:
                dtype = self.obj.dtypes.iloc[loc]
                if dtype not in (np.void, object) and (not self.obj.empty):
                    raise TypeError(f"Invalid value '{value}' for dtype '{dtype}'") from exc
                self.obj.isetitem(loc, value)
        else:
            dtype = self.obj.dtypes.iloc[loc]
            if dtype == np.void:
                self.obj.iloc[:, loc] = construct_1d_array_from_inferred_fill_value(value, len(self.obj))
            self.obj._mgr.column_setitem(loc, plane_indexer, value)

    def _setitem_single_block(self, indexer: Any, value: Any, name: str) -> None:
        from pandas import Series
        if isinstance(value, ABCSeries) and name != "iloc" or isinstance(value, dict):
            value = self._align_series(indexer, Series(value))
        info_axis: int = self.obj._info_axis_number
        item_labels = self.obj._get_axis(info_axis)
        if isinstance(indexer, tuple):
            if self.ndim == len(indexer) == 2 and is_integer(indexer[1]) and com.is_null_slice(indexer[0]):
                col = item_labels[indexer[info_axis]]
                if len(item_labels.get_indexer_for([col])) == 1:
                    loc = item_labels.get_loc(col)
                    self._setitem_single_column(loc, value, indexer[0])
                    return
            indexer = maybe_convert_ix(*indexer)
        if isinstance(value, ABCDataFrame) and name != "iloc":
            value = self._align_frame(indexer, value)._values
        self.obj._mgr = self.obj._mgr.setitem(indexer=indexer, value=value)

    def _setitem_with_indexer_missing(self, indexer: Any, value: Any) -> None:
        from pandas import Series
        if self.ndim == 1:
            index = self.obj.index
            new_index = index.insert(len(index), indexer)
            if index.is_unique:
                new_indexer = index.get_indexer(new_index[-1:])
                if (new_indexer != -1).any():
                    return self._setitem_with_indexer(new_indexer, value, "loc")
            if not is_scalar(value):
                new_dtype = None
            elif is_valid_na_for_dtype(value, self.obj.dtype):
                if not is_object_dtype(self.obj.dtype):
                    value = na_value_for_dtype(self.obj.dtype, compat=False)
                new_dtype = maybe_promote(self.obj.dtype, value)[0]
            elif isna(value):
                new_dtype = None
            elif not self.obj.empty and (not is_object_dtype(self.obj.dtype)):
                curr_dtype = self.obj.dtype
                curr_dtype = getattr(curr_dtype, "numpy_dtype", curr_dtype)
                new_dtype = maybe_promote(curr_dtype, value)[0]
            else:
                new_dtype = None
            new_values = Series([value], dtype=new_dtype)._values
            if len(self.obj._values):
                new_values = concat_compat([self.obj._values, new_values])
            self.obj._mgr = self.obj._constructor(new_values, index=new_index, name=self.obj.name)._mgr
        elif self.ndim == 2:
            if not len(self.obj.columns):
                raise ValueError("cannot set a frame with no defined columns")
            has_dtype = hasattr(value, "dtype")
            if isinstance(value, ABCSeries):
                value = value.reindex(index=self.obj.columns)
                value.name = indexer
            elif isinstance(value, dict):
                value = Series(value, index=self.obj.columns, name=indexer, dtype=object)
            else:
                if is_list_like_indexer(value):
                    if len(value) != len(self.obj.columns):
                        raise ValueError("cannot set a row with mismatched columns")
                value = Series(value, index=self.obj.columns, name=indexer)
            if not len(self.obj):
                df = value.to_frame().T
                idx = self.obj.index
                if isinstance(idx, MultiIndex):
                    name = idx.names
                else:
                    name = idx.name
                df.index = Index([indexer], name=name)
                if not has_dtype:
                    df = df.infer_objects()
                self.obj._mgr = df._mgr
            else:
                self.obj._mgr = self.obj._append(value)._mgr

    def _ensure_iterable_column_indexer(self, column_indexer: Any) -> Sequence[Union[int, np.integer]]:
        if is_integer(column_indexer):
            ilocs: List[int] = [column_indexer]
        elif isinstance(column_indexer, slice):
            ilocs = list(range(len(self.obj.columns)))[column_indexer]
        elif isinstance(column_indexer, np.ndarray) and column_indexer.dtype.kind == "b":
            ilocs = np.arange(len(column_indexer))[column_indexer].tolist()
        else:
            ilocs = column_indexer
        return ilocs

    def _align_series(self, indexer: Any, ser: Series, multiindex_indexer: bool = False, using_cow: bool = False) -> Any:
        if isinstance(indexer, (slice, np.ndarray, list, Index)):
            indexer = (indexer,)
        if isinstance(indexer, tuple):
            def ravel(i: Any) -> Any:
                return i.ravel() if isinstance(i, np.ndarray) else i
            indexer = tuple(map(ravel, indexer))
            aligners = [not com.is_null_slice(idx) for idx in indexer]
            sum_aligners = sum(aligners)
            single_aligner = sum_aligners == 1
            is_frame: bool = self.ndim == 2
            obj = self.obj
            if is_frame:
                single_aligner = single_aligner and aligners[0]
            if sum_aligners == self.ndim and all((is_sequence(_) for _ in indexer)):
                ser_values = ser.reindex(obj.axes[0][indexer[0]])._values
                if len(indexer) > 1 and (not multiindex_indexer):
                    len_indexer = len(indexer[1])
                    ser_values = np.tile(ser_values, len_indexer).reshape(len_indexer, -1).T
                return ser_values
            for i, idx in enumerate(indexer):
                ax = obj.axes[i]
                if is_sequence(idx) or isinstance(idx, slice):
                    if single_aligner and com.is_null_slice(idx):
                        continue
                    new_ix = ax[idx]
                    if not is_list_like_indexer(new_ix):
                        new_ix = Index([new_ix])
                    else:
                        new_ix = Index(new_ix)
                    if (not len(new_ix)) or ser.index.equals(new_ix):
                        if using_cow:
                            return ser
                        return ser._values.copy()
                    return ser.reindex(new_ix)._values
                elif single_aligner:
                    ax = self.obj.axes[1]
                    if ser.index.equals(ax) or not len(ax):
                        return ser._values.copy()
                    return ser.reindex(ax)._values
        elif is_integer(indexer) and self.ndim == 1:
            if is_object_dtype(self.obj.dtype):
                return ser
            ax = self.obj._get_axis(0)
            if ser.index.equals(ax):
                return ser._values.copy()
            return ser.reindex(ax)._values[indexer]
        elif is_integer(indexer):
            ax = self.obj._get_axis(1)
            if ser.index.equals(ax):
                return ser._values.copy()
            return ser.reindex(ax)._values
        raise ValueError("Incompatible indexer with Series")

    def _align_frame(self, indexer: Any, df: DataFrame) -> DataFrame:
        is_frame: bool = self.ndim == 2
        if isinstance(indexer, tuple):
            idx = None
            cols = None
            sindexers: List[int] = []
            for i, ix in enumerate(indexer):
                ax = self.obj.axes[i]
                if is_sequence(ix) or isinstance(ix, slice):
                    if isinstance(ix, np.ndarray):
                        ix = ix.reshape(-1)
                    if idx is None:
                        idx = ax[ix]
                    elif cols is None:
                        cols = ax[ix]
                    else:
                        break
                else:
                    sindexers.append(i)
            if idx is not None and cols is not None:
                if df.index.equals(idx) and df.columns.equals(cols):
                    val = df.copy()
                else:
                    val = df.reindex(idx, columns=cols)
                return val
        elif (isinstance(indexer, slice) or is_list_like_indexer(indexer)) and is_frame:
            ax = self.obj.index[indexer]
            if df.index.equals(ax):
                val = df.copy()
            else:
                if isinstance(ax, MultiIndex) and isinstance(df.index, MultiIndex) and (ax.nlevels != df.index.nlevels):
                    raise TypeError("cannot align on a multi-index with out specifying the join levels")
                val = df.reindex(index=ax)
            return val
        raise ValueError("Incompatible indexer with DataFrame")


class _ScalarAccessIndexer(NDFrameIndexerBase):
    _takeable: bool

    def _convert_key(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        raise AbstractMethodError(self)

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            if not is_list_like_indexer(key):
                key = (key,)
            else:
                raise ValueError("Invalid call for scalar access (getting)!")
        key = self._convert_key(key)
        return self.obj._get_value(*key, takeable=self._takeable)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, tuple):
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
        else:
            key = com.apply_if_callable(key, self.obj)
        if not isinstance(key, tuple):
            key = _tuplify(self.ndim, key)
        key = list(self._convert_key(key))
        if len(key) != self.ndim:
            raise ValueError("Not enough indexers for scalar access (setting)!")
        self.obj._set_value(*key, value=value, takeable=self._takeable)


@doc(IndexingMixin.at)
class _AtIndexer(_ScalarAccessIndexer):
    _takeable: bool = False

    def _convert_key(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        if self.ndim == 1 and len(key) > 1:
            key = (key,)
        return key

    @property
    def _axes_are_unique(self) -> bool:
        assert self.ndim == 2
        return self.obj.index.is_unique and self.obj.columns.is_unique

    def __getitem__(self, key: Any) -> Any:
        if self.ndim == 2 and (not self._axes_are_unique):
            if not isinstance(key, tuple) or not all((is_scalar(x) for x in key)):
                raise ValueError("Invalid call for scalar access (getting)!")
            return self.obj.loc[key]
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        if self.ndim == 2 and (not self._axes_are_unique):
            if not isinstance(key, tuple) or not all((is_scalar(x) for x in key)):
                raise ValueError("Invalid call for scalar access (setting)!")
            self.obj.loc[key] = value
            return
        return super().__setitem__(key, value)


@doc(IndexingMixin.iat)
class _iAtIndexer(_ScalarAccessIndexer):
    _takeable: bool = True

    def _convert_key(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        for i in key:
            if not is_integer(i):
                raise ValueError("iAt based indexing can only have integer indexers")
        return key


def _tuplify(ndim: int, loc: Any) -> Tuple[Union[Hashable, slice], ...]:
    _tup: List[Union[Hashable, slice]] = [slice(None, None) for _ in range(ndim)]
    _tup[0] = loc
    return tuple(_tup)


def _tupleize_axis_indexer(ndim: int, axis: int, key: Any) -> Tuple[Any, ...]:
    new_key: List[Any] = [slice(None)] * ndim
    new_key[axis] = key
    return tuple(new_key)


def check_bool_indexer(index: Index, key: Any) -> np.ndarray:
    result: Any = key
    if isinstance(key, ABCSeries) and (not key.index.equals(index)):
        indexer = result.index.get_indexer_for(index)
        if -1 in indexer:
            raise IndexingError("Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).")
        result = result.take(indexer)
        if not isinstance(result.dtype, ExtensionDtype):
            return result.astype(bool)._values
    if is_object_dtype(key):
        result = np.asarray(result, dtype=bool)
    elif not is_array_like(result):
        result = pd_array(result, dtype=bool)
    return check_array_indexer(index, result)


def convert_missing_indexer(indexer: Any) -> Tuple[Any, bool]:
    if isinstance(indexer, dict):
        indexer = indexer["key"]
        if isinstance(indexer, bool):
            raise KeyError("cannot use a single bool to index into setitem")
        return (indexer, True)
    return (indexer, False)


def convert_from_missing_indexer_tuple(indexer: Tuple[Any, ...], axes: Sequence[Any]) -> Tuple[Any, ...]:
    def get_indexer(_i: int, _idx: Any) -> Any:
        return axes[_i].get_loc(_idx["key"]) if isinstance(_idx, dict) else _idx
    return tuple((get_indexer(_i, _idx) for _i, _idx in enumerate(indexer)))


def maybe_convert_ix(*args: Any) -> Any:
    for arg in args:
        if not isinstance(arg, (np.ndarray, list, ABCSeries, Index)):
            return args
    return np.ix_(*args)


def is_nested_tuple(tup: Any, labels: Any) -> bool:
    if not isinstance(tup, tuple):
        return False
    for k in tup:
        if is_list_like(k) or isinstance(k, slice):
            return isinstance(labels, MultiIndex)
    return False


def is_label_like(key: Any) -> bool:
    return not isinstance(key, slice) and (not is_list_like_indexer(key)) and (key is not Ellipsis)


def need_slice(obj: slice) -> bool:
    return obj.start is not None or obj.stop is not None or (obj.step is not None and obj.step != 1)


def check_dict_or_set_indexers(key: Any) -> None:
    if isinstance(key, set) or (isinstance(key, tuple) and any((isinstance(x, set) for x in key))):
        raise TypeError("Passing a set as an indexer is not supported. Use a list instead.")
    if isinstance(key, dict) or (isinstance(key, tuple) and any((isinstance(x, dict) for x in key))):
        raise TypeError("Passing a dict as an indexer is not supported. Use a list instead.")