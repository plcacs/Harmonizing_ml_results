"""
SparseArray data structure
"""
from __future__ import annotations
from collections import abc
import numbers
import operator
from typing import (
    TYPE_CHECKING, Any, Literal, cast, overload, Union, Optional, Tuple, 
    List, Dict, TypeVar, Generic, Type, Callable, Sequence, Iterator, 
    Iterable, Set, FrozenSet, Mapping, MutableMapping, Deque, 
    NamedTuple, Protocol, runtime_checkable
)
import warnings
import numpy as np
from pandas._config.config import get_option
from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import BlockIndex, IntIndex, SparseIndex
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg, validate_insert_loc
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import find_common_type, maybe_box_datetimelike
from pandas.core.dtypes.common import (
    is_bool_dtype, is_integer, is_list_like, is_object_dtype, 
    is_scalar, is_string_dtype, pandas_dtype
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype, SparseDtype
from pandas.core.dtypes.generic import ABCIndex, ABCSeries
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna
from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike, extract_array, sanitize_array
)
from pandas.core.indexers import check_array_indexer, unpack_tuple_and_ellipses
from pandas.core.nanops import check_below_min_count
from pandas.io.formats import printing

if TYPE_CHECKING:
    from enum import Enum
    from collections.abc import Callable, Sequence
    from typing_extensions import Self

    class ellipsis(Enum):
        Ellipsis = '...'
    Ellipsis = ellipsis.Ellipsis
    from scipy.sparse import spmatrix
    from pandas._typing import NumpySorter
    SparseIndexKind = Literal['integer', 'block']
    from pandas._typing import (
        ArrayLike, AstypeArg, Axis, AxisInt, Dtype, NpDtype, 
        PositionalIndexer, Scalar, ScalarIndexer, SequenceIndexer, npt
    )
    from pandas import Series
else:
    ellipsis = type(Ellipsis)

_sparray_doc_kwargs = {'klass': 'SparseArray'}

T = TypeVar('T', bound='SparseArray')

def _get_fill(arr: 'SparseArray') -> np.ndarray:
    """
    Create a 0-dim ndarray containing the fill value
    """
    try:
        return np.asarray(arr.fill_value, dtype=arr.dtype.subtype)
    except ValueError:
        return np.asarray(arr.fill_value)

def _sparse_array_op(
    left: Union['SparseArray', np.ndarray],
    right: Union['SparseArray', np.ndarray],
    op: Callable,
    name: str
) -> 'SparseArray':
    if name.startswith('__'):
        name = name[2:-2]
    ltype = left.dtype.subtype
    rtype = right.dtype.subtype
    if ltype != rtype:
        subtype = find_common_type([ltype, rtype])
        ltype = SparseDtype(subtype, left.fill_value)
        rtype = SparseDtype(subtype, right.fill_value)
        left = left.astype(ltype, copy=False)
        right = right.astype(rtype, copy=False)
        dtype = ltype.subtype
    else:
        dtype = ltype
    result_dtype = None
    if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
        with np.errstate(all='ignore'):
            result = op(left.to_dense(), right.to_dense())
            fill = op(_get_fill(left), _get_fill(right))
        if left.sp_index.ngaps == 0:
            index = left.sp_index
        else:
            index = right.sp_index
    elif left.sp_index.equals(right.sp_index):
        with np.errstate(all='ignore'):
            result = op(left.sp_values, right.sp_values)
            fill = op(_get_fill(left), _get_fill(right))
        index = left.sp_index
    else:
        if name[0] == 'r':
            left, right = (right, left)
            name = name[1:]
        if name in ('and', 'or', 'xor') and dtype == 'bool':
            opname = f'sparse_{name}_uint8'
            left_sp_values = left.sp_values.view(np.uint8)
            right_sp_values = right.sp_values.view(np.uint8)
            result_dtype = bool
        else:
            opname = f'sparse_{name}_{dtype}'
            left_sp_values = left.sp_values
            right_sp_values = right.sp_values
        if name in ['floordiv', 'mod'] and (right == 0).any() and (left.dtype.kind in 'iu'):
            opname = f'sparse_{name}_float64'
            left_sp_values = left_sp_values.astype('float64')
            right_sp_values = right_sp_values.astype('float64')
        sparse_op = getattr(splib, opname)
        with np.errstate(all='ignore'):
            result, index, fill = sparse_op(left_sp_values, left.sp_index, left.fill_value, right_sp_values, right.sp_index, right.fill_value)
    if name == 'divmod':
        return (_wrap_result(name, result[0], index, fill[0], dtype=result_dtype), _wrap_result(name, result[1], index, fill[1], dtype=result_dtype))
    if result_dtype is None:
        result_dtype = result.dtype
    return _wrap_result(name, result, index, fill, dtype=result_dtype)

def _wrap_result(
    name: str,
    data: np.ndarray,
    sparse_index: SparseIndex,
    fill_value: Any,
    dtype: Optional[np.dtype] = None
) -> 'SparseArray':
    if name.startswith('__'):
        name = name[2:-2]
    if name in ('eq', 'ne', 'lt', 'gt', 'le', 'ge'):
        dtype = bool
    fill_value = lib.item_from_zerodim(fill_value)
    if is_bool_dtype(dtype):
        fill_value = bool(fill_value)
    return SparseArray(data, sparse_index=sparse_index, fill_value=fill_value, dtype=dtype)

class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    _subtyp = 'sparse_array'
    _hidden_attrs = PandasObject._hidden_attrs | frozenset([])

    def __init__(
        self,
        data: Any,
        sparse_index: Optional[SparseIndex] = None,
        fill_value: Optional[Any] = None,
        kind: str = 'integer',
        dtype: Optional[Union[np.dtype, SparseDtype]] = None,
        copy: bool = False
    ):
        if fill_value is None and isinstance(dtype, SparseDtype):
            fill_value = dtype.fill_value
        if isinstance(data, type(self)):
            if sparse_index is None:
                sparse_index = data.sp_index
            if fill_value is None:
                fill_value = data.fill_value
            if dtype is None:
                dtype = data.dtype
            data = data.sp_values
        if isinstance(dtype, str):
            try:
                dtype = SparseDtype.construct_from_string(dtype)
            except TypeError:
                dtype = pandas_dtype(dtype)
        if isinstance(dtype, SparseDtype):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype
        if is_scalar(data):
            raise TypeError(f'Cannot construct {type(self).__name__} from scalar data. Pass a sequence instead.')
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if data is None:
            data = np.array([], dtype=dtype)
        try:
            data = sanitize_array(data, index=None)
        except ValueError:
            if dtype is None:
                dtype = np.dtype(object)
                data = np.atleast_1d(np.asarray(data, dtype=dtype))
            else:
                raise
        if copy:
            data = data.copy()
        if fill_value is None:
            fill_value_dtype = data.dtype if dtype is None else dtype
            if fill_value_dtype is None:
                fill_value = np.nan
            else:
                fill_value = na_value_for_dtype(fill_value_dtype)
        if isinstance(data, type(self)) and sparse_index is None:
            sparse_index = data._sparse_index
            sparse_values = np.asarray(data.sp_values, dtype=dtype)
        elif sparse_index is None:
            data = extract_array(data, extract_numpy=True)
            if not isinstance(data, np.ndarray):
                if isinstance(data.dtype, DatetimeTZDtype):
                    warnings.warn(f'Creating SparseArray from {data.dtype} data loses timezone information. Cast to object before sparse to retain timezone information.', UserWarning, stacklevel=find_stack_level())
                    data = np.asarray(data, dtype='datetime64[ns]')
                    if fill_value is NaT:
                        fill_value = np.datetime64('NaT', 'ns')
                data = np.asarray(data)
            sparse_values, sparse_index, fill_value = _make_sparse(data, kind=kind, fill_value=fill_value, dtype=dtype)
        else:
            sparse_values = np.asarray(data, dtype=dtype)
            if len(sparse_values) != sparse_index.npoints:
                raise AssertionError(f'Non array-like type {type(sparse_values)} must have the same length as the index')
        self._sparse_index = sparse_index
        self._sparse_values = sparse_values
        self._dtype = SparseDtype(sparse_values.dtype, fill_value)

    @classmethod
    def _simple_new(
        cls,
        sparse_array: np.ndarray,
        sparse_index: SparseIndex,
        dtype: SparseDtype
    ) -> 'SparseArray':
        new = object.__new__(cls)
        new._sparse_index = sparse_index
        new._sparse_values = sparse_array
        new._dtype = dtype
        return new

    @classmethod
    def from_spmatrix(cls, data: 'spmatrix') -> 'SparseArray':
        length, ncol = data.shape
        if ncol != 1:
            raise ValueError(f"'data' must have a single column, not '{ncol}'")
        data = data.tocsc()
        data.sort_indices()
        arr = data.data
        idx = data.indices
        zero = np.array(0, dtype=arr.dtype).item()
        dtype = SparseDtype(arr.dtype, zero)
        index = IntIndex(length, idx)
        return cls._simple_new(arr, index, dtype)

    def __array__(self, dtype: Optional[np.dtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        if self.sp_index.ngaps == 0:
            if copy is True:
                return np.array(self.sp_values)
            else:
                return self.sp_values
        if copy is False:
            raise ValueError('Unable to avoid copy while creating an array as requested.')
        fill_value = self.fill_value
        if dtype is None:
            if self.sp_values.dtype.kind == 'M':
                if fill_value is NaT:
                    fill_value = np.datetime64('NaT')
            try:
                dtype = np.result_type(self.sp_values.dtype, type(fill_value))
            except TypeError:
                dtype = object
        out = np.full(self.shape, fill_value, dtype=dtype)
        out[self.sp_index.indices] = self.sp_values
        return out

    def __setitem__(self, key: Any, value: Any) -> None:
        msg = 'SparseArray does not support item assignment via setitem'
        raise TypeError(msg)

    @classmethod
    def _from_sequence(
        cls, 
        scalars: Sequence[Any], 
        *, 
        dtype: Optional[Dtype] = None, 
        copy: bool = False
    ) -> 'SparseArray':
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(
        cls, 
        values: np.ndarray, 
        original: 'SparseArray'
    ) -> 'SparseArray':
        return cls(values, dtype=original.dtype)

    @property
    def sp_index(self) -> SparseIndex:
        return self._sparse_index

    @property
    def sp_values(self) -> np.ndarray:
        return self._sparse_values

    @property
    def dtype(self) -> SparseDtype:
        return self._dtype

    @property
    def fill_value(self) -> Any:
        return self.dtype.fill_value

    @fill_value.setter
    def fill_value(self, value: Any) -> None:
        self._dtype = SparseDtype(self.dtype.subtype, value)

    @property
    def kind(self) -> str:
        if isinstance(self.sp_index, IntIndex):
            return 'integer'
        else:
            return 'block'

    @property
    def _valid_sp_values(self) -> np.ndarray:
        sp_vals = self.sp_values
        mask = notna(sp_vals)
        return sp_vals[mask]

    def __len__(self) -> int:
        return self.sp_index.length

    @property
    def _null_fill_value(self) -> bool:
        return self._dtype._is_na_fill_value

    @property
    def nbytes(self) -> int:
        return self.sp_values.nbytes + self.sp_index.nbytes

    @property
    def density(self) -> float:
        return self.sp_index.npoints / self.sp_index.length

    @property
    def npoints(self) -> int:
        return self.sp_index.npoints

    def isna(self) -> 'SparseArray':
        dtype = SparseDtype(bool, self._null_fill_value)
        if self._null_fill_value:
            return type(self)._simple_new(isna(self.sp_values), self.sp_index, dtype)
        mask = np.full(len(self), False, dtype=np.bool_)
        mask[self.sp_index.indices] = isna(self.sp_values)
        return type(self)(mask, fill_value=False, dtype=dtype)

    def fillna(
        self, 
        value: Any, 
        limit: Optional[int] = None, 
        copy: bool = True
    ) -> 'SparseArray':
        if limit is not None:
            raise ValueError('limit must be None')
        new_values = np.where(isna(self.sp_values), value, self.sp_values)
        if self._null_fill_value:
            new_dtype = SparseDtype(self.dtype.subtype, fill_value=value)
        else:
            new_dtype = self.dtype
        return self._simple_new(new_values, self._sparse_index, new_dtype)

    def shift(self, periods: int = 1, fill_value: Optional[Any] = None) -> 'SparseArray':
        if not len(self) or periods == 0:
            return self.copy()
        if isna(fill_value):
            fill_value = self.dtype.na_value
        subtype = np.result_type(fill_value, self.dtype.subtype)
        if subtype != self.dtype.subtype:
            arr = self.astype(SparseDtype(subtype, self.fill_value))
        else:
            arr = self
        empty = self._from_sequence([fill_value] * min(abs(periods), len(self)), dtype=arr.dtype)
        if periods > 0:
            a = empty
            b = arr[:-periods]
        else:
            a = arr[abs(periods):]
            b = empty
        return arr._concat_same_type([a, b])

    def _first_fill_value_loc(self) -> int:
        if len(self) == 0 or self.sp_index.npoints == len(self):
            return -1
        indices = self.sp_index.indices
        if not len(indices) or indices[0] > 0:
            return 0
        diff = np.r_[np.diff(indices), 2]
        return indices[(diff > 1).argmax()] + 1

    def duplicated(self, keep: str = 'first') -> np.ndarray:
        values = np.asarray(self)
        mask = np.asarray(self.isna())
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> 'SparseArray':
        uniques = algos.unique(self.sp_values)
        if len(self.sp_values) != len(self):
            fill_loc = self._first_fill_value_loc()
            insert_loc = len(algos.unique(self.sp_values[:fill_loc]))
            uniques = np.insert(uniques, insert_loc, self.fill_value)
        return type(self)._from_sequence(uniques, dtype=self.dtype)

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        return (np.asarray(self), self.fill_value)

    def factorize(
        self, 
        use_na_sentinel: bool = True
    ) -> Tuple[np.ndarray, 'SparseArray']:
        codes, uniques = algos.factorize(np.asarray(self), use_na_sentinel=use_na_sentinel)
        uniques_sp = SparseArray(uniques, dtype=self.dtype)
        return (codes, uniques_sp)

    def value_counts(self, dropna: bool = True) -> 'Series':
        from pandas import Index, Series
        keys, counts, _ = algos.value_counts_arraylike(self.sp_values, dropna=dropna)
        fcounts = self.sp_index