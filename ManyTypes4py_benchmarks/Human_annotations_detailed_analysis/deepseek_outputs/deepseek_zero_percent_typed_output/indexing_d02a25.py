from __future__ import annotations
from contextlib import suppress
import sys
from typing import TYPE_CHECKING, Any, TypeVar, cast, final, overload, Union, Optional, List, Tuple, Dict, Sequence, Hashable, Callable, Iterable
import warnings
import numpy as np
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import AbstractMethodError, ChainedAssignmentError, IndexingError, InvalidIndexError, LossySetitemError
from pandas.errors.cow import _chained_assignment_msg
from pandas.util._decorators import doc
from pandas.core.dtypes.cast import can_hold_element, maybe_promote
from pandas.core.dtypes.common import is_array_like, is_bool_dtype, is_hashable, is_integer, is_iterator, is_list_like, is_numeric_dtype, is_object_dtype, is_scalar, is_sequence
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import construct_1d_array_from_inferred_fill_value, infer_fill_value, is_valid_na_for_dtype, isna, na_value_for_dtype
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import array as pd_array, extract_array
from pandas.core.indexers import check_array_indexer, is_list_like_indexer, is_scalar_indexer, length_of_indexer
from pandas.core.indexes.api import Index, MultiIndex

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence
    from pandas._typing import Axis, AxisInt, Self, npt
    from pandas import DataFrame, Series

T = TypeVar('T')
_NS = slice(None, None)
_one_ellipsis_message = "indexer may only contain one '...' entry"

class _IndexSlice:
    """
    Create an object to more easily perform multi-index slicing.
    """
    def __getitem__(self, arg: Any) -> Any:
        return arg

IndexSlice = _IndexSlice()

class IndexingMixin:
    """
    Mixin for adding .loc/.iloc/.at/.iat to Dataframes and Series.
    """
    @property
    def iloc(self) -> _iLocIndexer:
        """
        Purely integer-location based indexing for selection by position.
        """
        return _iLocIndexer('iloc', self)

    @property
    def loc(self) -> _LocIndexer:
        """
        Access a group of rows and columns by label(s) or a boolean array.
        """
        return _LocIndexer('loc', self)

    @property
    def at(self) -> _AtIndexer:
        """
        Access a single value for a row/column label pair.
        """
        return _AtIndexer('at', self)

    @property
    def iat(self) -> _iAtIndexer:
        """
        Access a single value for a row/column pair by integer position.
        """
        return _iAtIndexer('iat', self)

class _LocationIndexer(NDFrameIndexerBase):
    axis: Optional[int] = None

    @final
    def __call__(self, axis: Optional[Any] = None) -> _LocationIndexer:
        new_self = type(self)(self.name, self.obj)
        if axis is not None:
            axis_int_none = self.obj._get_axis_number(axis)
        else:
            axis_int_none = axis
        new_self.axis = axis_int_none
        return new_self

    def _get_setitem_indexer(self, key: Any) -> Any:
        """
        Convert a potentially-label-based key into a positional indexer.
        """
        if self.name == 'loc':
            self._ensure_listlike_indexer(key, axis=self.axis)
        if isinstance(key, tuple):
            for x in key:
                check_dict_or_set_indexers(x)
        if self.axis is not None:
            key = _tupleize_axis_indexer(self.ndim, self.axis, key)
        ax = self.obj._get_axis(0)
        if isinstance(ax, MultiIndex) and self.name != 'iloc' and is_hashable(key) and (not isinstance(key, slice)):
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
        """
        If we have obj.iloc[mask] = series_or_frame and series_or_frame has the
        same length as obj, we treat this as obj.iloc[mask] = series_or_frame[mask],
        similar to Series.__setitem__.
        """
        if isinstance(indexer, tuple) and len(indexer) == 2 and isinstance(value, (ABCSeries, ABCDataFrame)):
            pi, icols = indexer
            ndim = value.ndim
            if com.is_bool_indexer(pi) and len(value) == len(pi):
                newkey = pi.nonzero()[0]
                if is_scalar_indexer(icols, self.ndim - 1) and ndim == 1:
                    if len(newkey) == 0:
                        value = value.iloc[:0]
                    else:
                        value = self.obj.iloc._align_series(indexer, value)
                    indexer = (newkey, icols)
                elif isinstance(icols, np.ndarray) and icols.dtype.kind == 'i' and (len(icols) == 1):
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
        """
        Ensure that a list-like of column labels are all present by adding them if
        they do not already exist.
        """
        column_axis = 1
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
            key = (list(x) if is_iterator(x) else x for x in key)
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
        else:
            maybe_callable = com.apply_if_callable(key, self.obj)
            key = self._raise_callable_usage(key, maybe_callable)
        indexer = self._get_setitem_indexer(key)
        self._has_valid_setitem_indexer(key)
        iloc = cast('_iLocIndexer', self) if self.name == 'iloc' else self.obj.iloc
        iloc._setitem_with_indexer(indexer, value, self.name)

    def _validate_key(self, key: Any, axis: int) -> None:
        """
        Ensure that key is valid for current indexer.
        """
        raise AbstractMethodError(self)

    @final
    def _expand_ellipsis(self, tup: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """
        If a tuple key includes an Ellipsis, replace it with an appropriate
        number of null slices.
        """
        if any((x is Ellipsis for x in tup)):
            if tup.count(Ellipsis) > 1:
                raise IndexingError(_one_ellipsis_message)
            if len(tup) == self.ndim:
                i = tup.index(Ellipsis)
                new_key = tup[:i] + (_NS,) + tup[i + 1:]
                return new_key
        return tup

    @final
    def _validate_tuple_indexer(self, key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """
        Check the key for valid keys across my indexer.
        """
        key = self._validate_key_length(key)
        key = self._expand_ellipsis(key)
        for i, k in enumerate(key):
            try:
                self._validate_key(k, i)
            except ValueError as err:
                raise ValueError(f'Location based indexing can only have [{self._valid_types}] types') from err
        return key

    @final
    def _is_nested_tuple_indexer(self, tup: Tuple[Any, ...]) -> bool:
        """
        Returns
        -------
        bool
        """
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
            raise IndexingError('Too many indexers')
        return key

    @final
    def _getitem_tuple_same_dim(self, tup: Tuple[Any, ...]) -> Any:
        """
        Index with indexers that should return an object of the same dimension
        as self.obj.
        """
        retval = self.obj
        start_val = self.ndim - len(tup) + 1
        for i, key in enumerate(reversed(tup)):
            i = self.ndim - i - start_val
            if com.is_null_slice(key):
                continue
            retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
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
        if isinstance(ax0, MultiIndex) and self.name != 'iloc' and (not any((isinstance(x, slice) for x in tup))):
            with suppress(IndexingError):
                return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)
        tup = self._validate_key_length(tup)
        for i, key in enumerate(tup):
            if is_label_like(key):
                section = self._getitem_axis(key, axis=i)
                if section.ndim == self.ndim:
                    new_key = tup[:i] + (_NS,) + tup[i + 1:]
                else:
                    new_key = tup[:i] + tup[i + 1:]
                    if len(new_key) == 1:
                        new_key = new_key[0]
                if com.is_null_slice(new_key):
                    return section
                return getattr(section, self.name)[new_key]
        raise IndexingError('not applicable')

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
            if self.name != 'loc':
                raise ValueError('Too many indices')
            if all((is_hashable(x) and (not _contains_slice(x)) or com.is_null_slice(x) for x in tup)):
                with suppress(IndexingError):
                    return cast(_LocIndexer, self)._handle_lowerdim_multi_index_axis0(tup)
            elif isinstance(self.obj, ABCSeries) and any((isinstance(k, tuple) for k in tup)):
                raise IndexingError('Too many indexers')
            axis = self.axis or 0
            return self._getitem_axis(tup, axis=axis)
        obj = self.obj
        axis = len(tup) - 1
        for key in reversed(tup):
            if com.is_null_slice(key):
                axis -= 1
                continue
            obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
            axis -= 1
            if is_scalar(obj) or not hasattr(obj, 'ndim'):
                break
        return obj

    def _convert_to_indexer(self, key: Any, axis: int) -> Any:
        raise AbstractMethodError(self)

    def _raise_callable_usage(self, key: Any, maybe_callable: Any) -> Any:
        if self.name == 'iloc' and callable(key) and isinstance(maybe_callable, tuple):
            raise ValueError('Returning a tuple from a callable with iloc is not allowed.')
        return maybe_callable

    @final
    def __getitem__(self, key: Any) -> Any:
        check_dict_or_set_indexers(key)
        if type(key) is tuple:
            key = (list(x) if is_iterator(x) else x for x in key)
            key = tuple((com.apply_if_callable(x, self.obj) for x in key))
            if self._is_scalar_access(key):
                return self.obj._get_value(*key, takeable=self._takeable)
            return self._getitem_tuple(key)
        else:
            axis = self.axis or 0
            maybe_callable = com.apply_if_callable(key, self.obj)
            maybe_callable = self._raise_callable_usage(key, maybe_callable)
            return self._getitem_axis(maybe_callable, axis=axis)

    def _is_scalar_access(self, key: Tuple[Any, ...]) -> bool:
        raise NotImplementedError

    def _getitem_tuple(self, tup: Tuple[Any, ...]) -> Any:
        raise AbstractMethodError(self)

    def _getitem_axis(self, key: Any, axis: int) -> Any:
        raise NotImplementedError

    def _has_valid_setitem_indexer(self, indexer: Any) -> bool:
        raise AbstractMethodError(self)

    @final
    def _getbool_axis(self, key: Any, axis: int) -> Any:
        labels = self.obj._get_axis(axis)
        key = check_bool_indexer(labels, key)
        inds = key.nonzero()[0]
        return self.obj.take(inds, axis=axis)

@doc(IndexingMixin.loc)
class _LocIndexer(_LocationIndexer):
    _takeable: bool = False
    _valid_types: str = 'labels (MUST BE IN THE INDEX), slices of labels (BOTH endpoints included! Can be slices of integers if the index is integers), listlike of labels, boolean'

    @doc(_LocationIndexer._validate_key)
    def _validate_key(self, key: Any, axis: int) -> None:
        ax = self.obj._get_axis(axis)
        if isinstance(key, bool) and (not (is_bool_dtype(ax.dtype) or ax.dtype.name == 'boolean' or (isinstance(ax, MultiIndex) and is_bool_dtype(ax.get_level_values(0).dtype))):
            raise KeyError(f'{key}: boolean label can not be used without a boolean index')
        if isinstance(key, slice) and (isinstance(key.start, bool) or isinstance(key.stop, bool)):
            raise TypeError(f'{key}: boolean values can not be used in a slice')

    def _has_valid_setitem_indexer(self, indexer: Any) -> bool:
