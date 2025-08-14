#!/usr/bin/env python3
"""
SparseArray data structure
"""

from __future__ import annotations

from collections import abc
import numbers
import operator
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    overload,
    Sequence,
    Tuple,
    Type,
    Union,
)
import warnings

import numpy as np

from pandas._config.config import get_option

from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
    BlockIndex,
    IntIndex,
    SparseIndex,
)
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_insert_loc,
)

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
    find_common_type,
    maybe_box_datetimelike,
)
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
)
from pandas.core.nanops import check_below_min_count

from pandas.io.formats import printing

if TYPE_CHECKING:
    from collections.abc import Callable as Cb, Sequence as Seq
    from enum import Enum

    class ellipsis(Enum):
        Ellipsis = "..."

    Ellipsis = ellipsis.Ellipsis

    from scipy.sparse import spmatrix

    from pandas._typing import NumpySorter

    SparseIndexKind = Literal["integer", "block"]

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        Axis,
        AxisInt,
        Dtype,
        NpDtype,
        PositionalIndexer,
        Scalar,
        ScalarIndexer,
        Self,
        SequenceIndexer,
        npt,
    )

    from pandas import Series
else:
    ellipsis = type(Ellipsis)

_sparray_doc_kwargs: dict[str, Any] = {"klass": "SparseArray"}


def _get_fill(arr: SparseArray) -> np.ndarray:
    """
    Create a 0-dim ndarray containing the fill value

    Parameters
    ----------
    arr : SparseArray

    Returns
    -------
    fill_value : ndarray
        0-dim ndarray with just the fill value.
    """
    try:
        return np.asarray(arr.fill_value, dtype=arr.dtype.subtype)
    except ValueError:
        return np.asarray(arr.fill_value)


def _sparse_array_op(
    left: SparseArray, right: SparseArray, op: Callable, name: str
) -> SparseArray:
    """
    Perform a binary operation between two arrays.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    op : Callable
        The binary operation to perform
    name : str
        Name of the callable.

    Returns
    -------
    SparseArray
    """
    if name.startswith("__"):
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

    result_dtype: Optional[Any] = None

    if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
        with np.errstate(all="ignore"):
            result = op(left.to_dense(), right.to_dense())
            fill = op(_get_fill(left), _get_fill(right))

        if left.sp_index.ngaps == 0:
            index = left.sp_index
        else:
            index = right.sp_index
    elif left.sp_index.equals(right.sp_index):
        with np.errstate(all="ignore"):
            result = op(left.sp_values, right.sp_values)
            fill = op(_get_fill(left), _get_fill(right))
        index = left.sp_index
    else:
        if name[0] == "r":
            left, right = right, left
            name = name[1:]

        if name in ("and", "or", "xor") and dtype == "bool":
            opname = f"sparse_{name}_uint8"
            left_sp_values = left.sp_values.view(np.uint8)
            right_sp_values = right.sp_values.view(np.uint8)
            result_dtype = bool
        else:
            opname = f"sparse_{name}_{dtype}"
            left_sp_values = left.sp_values
            right_sp_values = right.sp_values

        if (
            name in ["floordiv", "mod"]
            and (right == 0).any()
            and left.dtype.kind in "iu"
        ):
            opname = f"sparse_{name}_float64"
            left_sp_values = left_sp_values.astype("float64")
            right_sp_values = right_sp_values.astype("float64")

        sparse_op = getattr(splib, opname)

        with np.errstate(all="ignore"):
            result, index, fill = sparse_op(
                left_sp_values,
                left.sp_index,
                left.fill_value,
                right_sp_values,
                right.sp_index,
                right.fill_value,
            )

    if name == "divmod":
        return (
            _wrap_result(name, result[0], index, fill[0], dtype=result_dtype),
            _wrap_result(name, result[1], index, fill[1], dtype=result_dtype),
        )

    if result_dtype is None:
        result_dtype = result.dtype

    return _wrap_result(name, result, index, fill, dtype=result_dtype)


def _wrap_result(
    name: str, data, sparse_index: SparseIndex, fill_value, dtype: Optional[Any] = None
) -> SparseArray:
    """
    wrap op result to have correct dtype
    """
    if name.startswith("__"):
        name = name[2:-2]

    if name in ("eq", "ne", "lt", "gt", "le", "ge"):
        dtype = bool

    fill_value = lib.item_from_zerodim(fill_value)

    if is_bool_dtype(dtype):
        fill_value = bool(fill_value)
    return SparseArray(
        data, sparse_index=sparse_index, fill_value=fill_value, dtype=dtype
    )


class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    """
    An ExtensionArray for storing sparse data.
    """

    _subtyp: str = "sparse_array"
    _hidden_attrs: frozenset[str] = PandasObject._hidden_attrs | frozenset([])
    _sparse_index: SparseIndex
    _sparse_values: np.ndarray
    _dtype: SparseDtype

    def __init__(
        self,
        data: ArrayLike,
        sparse_index: Optional[SparseIndex] = None,
        fill_value: Any = None,
        kind: SparseIndexKind = "integer",
        dtype: Optional[Union[str, Dtype]] = None,
        copy: bool = False,
    ) -> None:
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
            raise TypeError(
                f"Cannot construct {type(self).__name__} from scalar data. "
                "Pass a sequence instead."
            )

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        if data is None:
            data = np.array([], dtype=dtype)  # type: ignore[arg-type]

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
            sparse_values = np.asarray(
                data.sp_values,
                dtype=dtype,  # type: ignore[arg-type]
            )
        elif sparse_index is None:
            data = extract_array(data, extract_numpy=True)
            if not isinstance(data, np.ndarray):
                if isinstance(data.dtype, DatetimeTZDtype):
                    warnings.warn(
                        f"Creating SparseArray from {data.dtype} data "
                        "loses timezone information. Cast to object before "
                        "sparse to retain timezone information.",
                        UserWarning,
                        stacklevel=find_stack_level(),
                    )
                    data = np.asarray(data, dtype="datetime64[ns]")
                    if fill_value is NaT:
                        fill_value = np.datetime64("NaT", "ns")
                data = np.asarray(data)
            sparse_values, sparse_index, fill_value = _make_sparse(
                data, kind=kind, fill_value=fill_value, dtype=dtype  # type: ignore[arg-type]
            )
        else:
            sparse_values = np.asarray(data, dtype=dtype)  # type: ignore[arg-type]
            if len(sparse_values) != sparse_index.npoints:
                raise AssertionError(
                    f"Non array-like type {type(sparse_values)} must "
                    "have the same length as the index"
                )
        self._sparse_index = sparse_index
        self._sparse_values = sparse_values
        self._dtype = SparseDtype(sparse_values.dtype, fill_value)

    @classmethod
    def _simple_new(cls: Type[SparseArray], sparse_array: np.ndarray, sparse_index: SparseIndex, dtype: SparseDtype) -> SparseArray:
        new = object.__new__(cls)
        new._sparse_index = sparse_index
        new._sparse_values = sparse_array
        new._dtype = dtype
        return new

    @classmethod
    def from_spmatrix(cls: Type[SparseArray], data: spmatrix) -> SparseArray:
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

    def __array__(self, dtype: Optional[NpDtype] = None, copy: Optional[bool] = None) -> np.ndarray:
        if self.sp_index.ngaps == 0:
            if copy is True:
                return np.array(self.sp_values)
            else:
                return self.sp_values

        if copy is False:
            raise ValueError(
                "Unable to avoid copy while creating an array as requested."
            )

        fill_value = self.fill_value

        if dtype is None:
            if self.sp_values.dtype.kind == "M":
                if fill_value is NaT:
                    fill_value = np.datetime64("NaT")
            try:
                dtype = np.result_type(self.sp_values.dtype, type(fill_value))
            except TypeError:
                dtype = object

        out = np.full(self.shape, fill_value, dtype=dtype)
        out[self.sp_index.indices] = self.sp_values
        return out

    def __setitem__(self, key, value) -> None:
        msg = "SparseArray does not support item assignment via setitem"
        raise TypeError(msg)

    @classmethod
    def _from_sequence(cls: Type[SparseArray], scalars: Sequence[Any], *, dtype: Optional[Dtype] = None, copy: bool = False) -> SparseArray:
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls: Type[SparseArray], values: np.ndarray, original: SparseArray) -> SparseArray:
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
    def kind(self) -> SparseIndexKind:
        if isinstance(self.sp_index, IntIndex):
            return "integer"
        else:
            return "block"

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

    def isna(self) -> SparseArray:
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
        copy: bool = True,
    ) -> SparseArray:
        if limit is not None:
            raise ValueError("limit must be None")
        new_values = np.where(isna(self.sp_values), value, self.sp_values)

        if self._null_fill_value:
            new_dtype = SparseDtype(self.dtype.subtype, fill_value=value)
        else:
            new_dtype = self.dtype

        return self._simple_new(new_values, self._sparse_index, new_dtype)

    def shift(self, periods: int = 1, fill_value: Any = None) -> SparseArray:
        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        subtype = np.result_type(fill_value, self.dtype.subtype)

        if subtype != self.dtype.subtype:
            arr = self.astype(SparseDtype(subtype, self.fill_value))
        else:
            arr = self

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=arr.dtype
        )

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

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        values = np.asarray(self)
        mask = np.asarray(self.isna())
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> SparseArray:
        uniques = algos.unique(self.sp_values)
        if len(self.sp_values) != len(self):
            fill_loc = self._first_fill_value_loc()
            insert_loc = len(algos.unique(self.sp_values[:fill_loc]))
            uniques = np.insert(uniques, insert_loc, self.fill_value)
        return type(self)._from_sequence(uniques, dtype=self.dtype)

    def _values_for_factorize(self) -> Tuple[np.ndarray, Any]:
        return np.asarray(self), self.fill_value

    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> Tuple[np.ndarray, SparseArray]:
        codes, uniques = algos.factorize(
            np.asarray(self), use_na_sentinel=use_na_sentinel
        )
        uniques_sp = SparseArray(uniques, dtype=self.dtype)
        return codes, uniques_sp

    def value_counts(self, dropna: bool = True) -> Series:
        from pandas import Index, Series

        keys, counts, _ = algos.value_counts_arraylike(self.sp_values, dropna=dropna)
        fcounts = self.sp_index.ngaps
        if fcounts > 0 and (not self._null_fill_value or not dropna):
            mask = isna(keys) if self._null_fill_value else keys == self.fill_value
            if mask.any():
                counts[mask] += fcounts
            else:
                keys = np.insert(keys, 0, self.fill_value)  # type: ignore[arg-type]
                counts = np.insert(counts, 0, fcounts)

        if not isinstance(keys, ABCIndex):
            index = Index(keys)
        else:
            index = keys
        return Series(counts, index=index, copy=False)

    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any: ...
    @overload
    def __getitem__(
        self,
        key: Union[SequenceIndexer, tuple[Union[int, ellipsis], ...]],
    ) -> SparseArray: ...

    def __getitem__(
        self,
        key: Union[PositionalIndexer, tuple[Union[int, ellipsis], ...]]
    ) -> Union[SparseArray, Any]:
        if isinstance(key, tuple):
            key = unpack_tuple_and_ellipses(key)
            if key is Ellipsis:
                raise ValueError("Cannot slice with Ellipsis")

        if is_integer(key):
            return self._get_val_at(key)
        elif isinstance(key, tuple):
            data_slice = self.to_dense()[key]  # type: ignore[index]
        elif isinstance(key, slice):
            if key.step is None or key.step == 1:
                start = 0 if key.start is None else key.start
                if start < 0:
                    start += len(self)

                end = len(self) if key.stop is None else key.stop
                if end < 0:
                    end += len(self)

                indices = self.sp_index.indices
                keep_inds = np.flatnonzero((indices >= start) & (indices < end))
                sp_vals = self.sp_values[keep_inds]

                sp_index = indices[keep_inds].copy()
                if start > 0:
                    sp_index -= start

                new_len = len(range(len(self))[key])
                new_sp_index = make_sparse_index(new_len, sp_index, self.kind)
                return type(self)._simple_new(sp_vals, new_sp_index, self.dtype)
            else:
                indices = np.arange(len(self), dtype=np.int32)[key]
                return self.take(indices)
        elif not is_list_like(key):
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        else:
            if isinstance(key, SparseArray):
                if is_bool_dtype(key):
                    if isna(key.fill_value):
                        return self.take(key.sp_index.indices[key.sp_values])
                    if not key.fill_value:
                        return self.take(key.sp_index.indices)
                    n = len(self)
                    mask = np.full(n, True, dtype=np.bool_)
                    mask[key.sp_index.indices] = False
                    return self.take(np.arange(n)[mask])
                else:
                    key = np.asarray(key)

            key = check_array_indexer(self, key)

            if com.is_bool_indexer(key):
                key = key  # type: ignore
                return self.take(np.arange(len(key), dtype=np.int32)[key])
            elif hasattr(key, "__len__"):
                return self.take(key)
            else:
                raise ValueError(f"Cannot slice with '{key}'")

        return type(self)(data_slice, kind=self.kind)

    def _get_val_at(self, loc: int) -> Any:
        loc = validate_insert_loc(loc, len(self))
        sp_loc = self.sp_index.lookup(loc)
        if sp_loc == -1:
            return self.fill_value
        else:
            val = self.sp_values[sp_loc]
            val = maybe_box_datetimelike(val, self.sp_values.dtype)
            return val

    def take(self, indices: ArrayLike, *, allow_fill: bool = False, fill_value: Any = None) -> SparseArray:
        if is_scalar(indices):
            raise ValueError(f"'indices' must be an array, not a scalar '{indices}'.")
        indices = np.asarray(indices, dtype=np.int32)

        dtype: Optional[Any] = None
        if indices.size == 0:
            result = np.array([], dtype="object")
            dtype = self.dtype
        elif allow_fill:
            result = self._take_with_fill(indices, fill_value=fill_value)
        else:
            return self._take_without_fill(indices)

        return type(self)(
            result, fill_value=self.fill_value, kind=self.kind, dtype=dtype
        )

    def _take_with_fill(self, indices: np.ndarray, fill_value: Any = None) -> np.ndarray:
        if fill_value is None:
            fill_value = self.dtype.na_value

        if indices.min() < -1:
            raise ValueError(
                "Invalid value in 'indices'. Must be between -1 "
                "and the length of the array."
            )

        if indices.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        if len(self) == 0:
            if (indices == -1).all():
                dtype_local = np.result_type(self.sp_values, type(fill_value))
                taken = np.empty_like(indices, dtype=dtype_local)
                taken.fill(fill_value)
                return taken
            else:
                raise IndexError("cannot do a non-empty take from an empty axes.")

        sp_indexer = self.sp_index.lookup_array(indices)
        new_fill_indices = indices == -1
        old_fill_indices = (sp_indexer == -1) & ~new_fill_indices

        if self.sp_index.npoints == 0 and old_fill_indices.all():
            taken = np.full(
                sp_indexer.shape, fill_value=self.fill_value, dtype=self.dtype.subtype
            )
        elif self.sp_index.npoints == 0:
            _dtype = np.result_type(self.dtype.subtype, type(fill_value))
            taken = np.full(sp_indexer.shape, fill_value=fill_value, dtype=_dtype)
            taken[old_fill_indices] = self.fill_value
        else:
            taken = self.sp_values.take(sp_indexer)
            m0 = sp_indexer[old_fill_indices] < 0
            m1 = sp_indexer[new_fill_indices] < 0

            result_type = taken.dtype

            if m0.any():
                result_type = np.result_type(result_type, type(self.fill_value))
                taken = taken.astype(result_type)
                taken[old_fill_indices] = self.fill_value

            if m1.any():
                result_type = np.result_type(result_type, type(fill_value))
                taken = taken.astype(result_type)
                taken[new_fill_indices] = fill_value

        return taken

    def _take_without_fill(self, indices: np.ndarray) -> SparseArray:
        to_shift = indices < 0
        n = len(self)

        if (indices.max() >= n) or (indices.min() < -n):
            if n == 0:
                raise IndexError("cannot do a non-empty take from an empty axes.")
            raise IndexError("out of bounds value in 'indices'.")

        if to_shift.any():
            indices = indices.copy()
            indices[to_shift] += n

        sp_indexer = self.sp_index.lookup_array(indices)
        value_mask = sp_indexer != -1
        new_sp_values = self.sp_values[sp_indexer[value_mask]]
        value_indices = np.flatnonzero(value_mask).astype(np.int32, copy=False)
        new_sp_index = make_sparse_index(len(indices), value_indices, kind=self.kind)
        return type(self)._simple_new(new_sp_values, new_sp_index, dtype=self.dtype)

    def searchsorted(
        self,
        v: Union[ArrayLike, object],
        side: Literal["left", "right"] = "left",
        sorter: Optional[NumpySorter] = None,
    ) -> Union[npt.NDArray[np.intp], np.intp]:
        if get_option("performance_warnings"):
            msg = "searchsorted requires high memory usage."
            warnings.warn(msg, PerformanceWarning, stacklevel=find_stack_level())
        v = np.asarray(v)
        return np.asarray(self, dtype=self.dtype.subtype).searchsorted(v, side, sorter)

    def copy(self) -> SparseArray:
        values = self.sp_values.copy()
        return self._simple_new(values, self.sp_index, self.dtype)

    @classmethod
    def _concat_same_type(cls: Type[SparseArray], to_concat: Sequence[SparseArray]) -> SparseArray:
        fill_value = to_concat[0].fill_value

        values: list[np.ndarray] = []
        length = 0

        if to_concat:
            sp_kind = to_concat[0].kind
        else:
            sp_kind = "integer"

        if sp_kind == "integer":
            indices: list[np.ndarray] = []

            for arr in to_concat:
                int_idx = arr.sp_index.indices.copy()
                int_idx += length
                length += arr.sp_index.length

                values.append(arr.sp_values)
                indices.append(int_idx)

            data = np.concatenate(values)
            indices_arr = np.concatenate(indices)
            sp_index = IntIndex(length, indices_arr)  # type: ignore[arg-type]
        else:
            blengths: list[np.ndarray] = []
            blocs: list[np.ndarray] = []

            for arr in to_concat:
                block_idx = arr.sp_index.to_block_index()

                values.append(arr.sp_values)
                blocs.append(block_idx.blocs.copy() + length)
                blengths.append(block_idx.blengths)
                length += arr.sp_index.length

            data = np.concatenate(values)
            blocs_arr = np.concatenate(blocs)
            blengths_arr = np.concatenate(blengths)

            sp_index = BlockIndex(length, blocs_arr, blengths_arr)

        return cls(data, sparse_index=sp_index, fill_value=fill_value)

    def astype(self, dtype: Optional[AstypeArg] = None, copy: bool = True):
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                return self.copy()

        future_dtype = pandas_dtype(dtype)
        if not isinstance(future_dtype, SparseDtype):
            values = np.asarray(self)
            values = ensure_wrapped_if_datetimelike(values)
            return astype_array(values, dtype=future_dtype, copy=False)

        dtype = self.dtype.update_dtype(dtype)
        subtype = pandas_dtype(dtype._subtype_with_str)
        subtype = cast(np.dtype, subtype)
        values = ensure_wrapped_if_datetimelike(self.sp_values)
        sp_values = astype_array(values, subtype, copy=copy)
        sp_values = np.asarray(sp_values)

        return self._simple_new(sp_values, self.sp_index, dtype)

    def map(self, mapper: Union[dict, "ABCSeries", Callable], na_action: Optional[Literal["ignore"]] = None) -> SparseArray:
        is_map = isinstance(mapper, (abc.Mapping, ABCSeries))

        fill_val = self.fill_value

        if na_action is None or notna(fill_val):
            fill_val = mapper.get(fill_val, fill_val) if is_map else mapper(fill_val)

        def func(sp_val: Any) -> Any:
            new_sp_val = mapper.get(sp_val, None) if is_map else mapper(sp_val)
            if new_sp_val is fill_val or new_sp_val == fill_val:
                msg = "fill value in the sparse values not supported"
                raise ValueError(msg)
            return new_sp_val

        sp_values = [func(x) for x in self.sp_values]

        return type(self)(sp_values, sparse_index=self.sp_index, fill_value=fill_val)

    def to_dense(self) -> np.ndarray:
        return np.asarray(self, dtype=self.sp_values.dtype)

    def _where(self, mask: np.ndarray, value: Any) -> SparseArray:
        naive_implementation = np.where(mask, self, value)
        dtype_local = SparseDtype(naive_implementation.dtype, fill_value=self.fill_value)
        result = type(self)._from_sequence(naive_implementation, dtype=dtype_local)
        return result

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple):
            nd_state, (fill_value, sp_index) = state
            sparse_values = np.array([])
            sparse_values.__setstate__(nd_state)

            self._sparse_values = sparse_values
            self._sparse_index = sp_index
            self._dtype = SparseDtype(sparse_values.dtype, fill_value)
        else:
            self.__dict__.update(state)

    def nonzero(self) -> Tuple[npt.NDArray[np.int32]]:
        if self.fill_value == 0:
            return (self.sp_index.indices,)
        else:
            return (self.sp_index.indices[self.sp_values != 0],)

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any
    ):
        method = getattr(self, name, None)

        if method is None:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if skipna:
            arr = self
        else:
            arr = self.dropna()

        result = getattr(arr, name)(**kwargs)

        if keepdims:
            return type(self)([result], dtype=self.dtype)
        else:
            return result

    def all(self, axis: Optional[Axis] = None, *args: Any, **kwargs: Any) -> bool:
        nv.validate_all(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and not np.all(self.fill_value):
            return False

        return bool(values.all())

    def any(self, axis: AxisInt = 0, *args: Any, **kwargs: Any) -> bool:
        nv.validate_any(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and np.any(self.fill_value):
            return True

        return bool(values.any().item())

    def sum(
        self,
        axis: AxisInt = 0,
        min_count: int = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Scalar:
        nv.validate_sum(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        has_na = self.sp_index.ngaps > 0 and not self._null_fill_value

        if has_na and not skipna:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

        if self._null_fill_value:
            if check_below_min_count(valid_vals.shape, None, min_count):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum
        else:
            nsparse = self.sp_index.ngaps
            if check_below_min_count(valid_vals.shape, None, min_count - nsparse):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum + self.fill_value * nsparse

    def cumsum(self, axis: AxisInt = 0, *args: Any, **kwargs: Any) -> SparseArray:
        nv.validate_cumsum(args, kwargs)

        if axis is not None and axis >= self.ndim:
            raise ValueError(f"axis(={axis}) out of bounds")

        if not self._null_fill_value:
            return SparseArray(self.to_dense()).cumsum()

        return SparseArray(
            self.sp_values.cumsum(),
            sparse_index=self.sp_index,
            fill_value=self.fill_value,
        )

    def mean(self, axis: Axis = 0, *args: Any, **kwargs: Any) -> float:
        nv.validate_mean(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        ct = len(valid_vals)

        if self._null_fill_value:
            return sp_sum / ct
        else:
            nsparse = self.sp_index.ngaps
            return (sp_sum + self.fill_value * nsparse) / (ct + nsparse)

    def max(self, *, axis: Optional[AxisInt] = None, skipna: bool = True) -> Scalar:
        nv.validate_minmax_axis(axis, self.ndim)
        return self._min_max("max", skipna=skipna)

    def min(self, *, axis: Optional[AxisInt] = None, skipna: bool = True) -> Scalar:
        nv.validate_minmax_axis(axis, self.ndim)
        return self._min_max("min", skipna=skipna)

    def _min_max(self, kind: Literal["min", "max"], skipna: bool) -> Scalar:
        valid_vals = self._valid_sp_values
        has_nonnull_fill_vals = not self._null_fill_value and self.sp_index.ngaps > 0

        if len(valid_vals) > 0:
            sp_min_max = getattr(valid_vals, kind)()

            if has_nonnull_fill_vals:
                func = max if kind == "max" else min
                return func(sp_min_max, self.fill_value)
            elif skipna:
                return sp_min_max
            elif self.sp_index.ngaps == 0:
                return sp_min_max
            else:
                return na_value_for_dtype(self.dtype.subtype, compat=False)
        elif has_nonnull_fill_vals:
            return self.fill_value
        else:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

    def _argmin_argmax(self, kind: Literal["argmin", "argmax"]) -> int:
        values = self._sparse_values
        index = self._sparse_index.indices
        mask = np.asarray(isna(values))
        func = np.argmax if kind == "argmax" else np.argmin

        idx = np.arange(values.shape[0])
        non_nans = values[~mask]
        non_nan_idx = idx[~mask]

        _candidate = non_nan_idx[func(non_nans)]
        candidate = index[_candidate]

        if isna(self.fill_value):
            return candidate
        if kind == "argmin" and self[candidate] < self.fill_value:
            return candidate
        if kind == "argmax" and self[candidate] > self.fill_value:
            return candidate
        _loc = self._first_fill_value_loc()
        if _loc == -1:
            return candidate
        else:
            return _loc

    def argmax(self, skipna: bool = True) -> int:
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return self._argmin_argmax("argmax")

    def argmin(self, skipna: bool = True) -> int:
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        return self._argmin_argmax("argmin")

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (SparseArray,)):
                return NotImplemented

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            res = arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )
            return res

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        if len(inputs) == 1:
            sp_values = getattr(ufunc, method)(self.sp_values, **kwargs)
            fill_value = getattr(ufunc, method)(self.fill_value, **kwargs)

            if ufunc.nout > 1:
                arrays = tuple(
                    self._simple_new(
                        sp_value, self.sp_index, SparseDtype(sp_value.dtype, fv)
                    )
                    for sp_value, fv in zip(sp_values, fill_value)
                )
                return arrays
            elif method == "reduce":
                return sp_values

            return self._simple_new(
                sp_values, self.sp_index, SparseDtype(sp_values.dtype, fill_value)
            )

        new_inputs = tuple(np.asarray(x) for x in inputs)
        result = getattr(ufunc, method)(*new_inputs, **kwargs)
        if out:
            if len(out) == 1:
                out = out[0]
            return out

        if ufunc.nout > 1:
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            return None
        else:
            return type(self)(result)

    def _arith_method(self, other: Any, op: Callable) -> SparseArray:
        op_name = op.__name__

        if isinstance(other, SparseArray):
            return _sparse_array_op(self, other, op, op_name)

        elif is_scalar(other):
            with np.errstate(all="ignore"):
                fill = op(_get_fill(self), np.asarray(other))
                result = op(self.sp_values, other)

            if op_name == "divmod":
                left, right = result
                lfill, rfill = fill
                return _wrap_result(op_name, left, self.sp_index, lfill)
            return _wrap_result(op_name, result, self.sp_index, fill)

        else:
            other = np.asarray(other)
            with np.errstate(all="ignore"):
                if len(self) != len(other):
                    raise AssertionError(
                        f"length mismatch: {len(self)} vs. {len(other)}"
                    )
                if not isinstance(other, SparseArray):
                    dtype = getattr(other, "dtype", None)
                    other = SparseArray(other, fill_value=self.fill_value, dtype=dtype)
                return _sparse_array_op(self, other, op, op_name)

    def _cmp_method(self, other: Any, op: Callable) -> SparseArray:
        if not is_scalar(other) and not isinstance(other, type(self)):
            other = np.asarray(other)

        if isinstance(other, np.ndarray):
            other = SparseArray(other, fill_value=self.fill_value)

        if isinstance(other, SparseArray):
            if len(self) != len(other):
                raise ValueError(
                    f"operands have mismatched length {len(self)} and {len(other)}"
                )

            op_name = op.__name__.strip("_")
            return _sparse_array_op(self, other, op, op_name)
        else:
            fill_value = op(self.fill_value, other)
            result = np.full(len(self), fill_value, dtype=np.bool_)
            result[self.sp_index.indices] = op(self.sp_values, other)

            return type(self)(
                result,
                fill_value=fill_value,
                dtype=np.bool_,
            )

    _logical_method = _cmp_method

    def _unary_method(self, op: Callable) -> SparseArray:
        fill_value = op(np.array(self.fill_value)).item()
        dtype_local = SparseDtype(self.dtype.subtype, fill_value)
        if isna(self.fill_value) or fill_value == self.fill_value:
            values = op(self.sp_values)
            return type(self)._simple_new(values, self.sp_index, self.dtype)
        return type(self)(op(self.to_dense()), dtype=dtype_local)

    def __pos__(self) -> SparseArray:
        return self._unary_method(operator.pos)

    def __neg__(self) -> SparseArray:
        return self._unary_method(operator.neg)

    def __invert__(self) -> SparseArray:
        return self._unary_method(operator.invert)

    def __abs__(self) -> SparseArray:
        return self._unary_method(operator.abs)

    def __repr__(self) -> str:
        pp_str = printing.pprint_thing(self)
        pp_fill = printing.pprint_thing(self.fill_value)
        pp_index = printing.pprint_thing(self.sp_index)
        return f"{pp_str}\nFill: {pp_fill}\n{pp_index}"

    def _formatter(self, boxed: bool = False) -> Optional[Callable[[Any], Union[str, None]]]:
        return None


def _make_sparse(
    arr: np.ndarray,
    kind: SparseIndexKind = "block",
    fill_value: Optional[Any] = None,
    dtype: Optional[np.dtype] = None,
) -> Tuple[np.ndarray, SparseIndex, Any]:
    assert isinstance(arr, np.ndarray)

    if arr.ndim > 1:
        raise TypeError("expected dimension <= 1 data")

    if fill_value is None:
        fill_value = na_value_for_dtype(arr.dtype)

    if isna(fill_value):
        mask = notna(arr)
    else:
        if is_string_dtype(arr.dtype):
            arr = arr.astype(object)

        if is_object_dtype(arr.dtype):
            mask = splib.make_mask_object_ndarray(arr, fill_value)
        else:
            mask = arr != fill_value

    length = len(arr)
    if length != len(mask):
        indices = mask.sp_index.indices
    else:
        indices = mask.nonzero()[0].astype(np.int32)

    index = make_sparse_index(length, indices, kind)
    sparsified_values = arr[mask]
    if dtype is not None:
        sparsified_values = ensure_wrapped_if_datetimelike(sparsified_values)
        sparsified_values = astype_array(sparsified_values, dtype=dtype)
        sparsified_values = np.asarray(sparsified_values)

    return sparsified_values, index, fill_value


@overload
def make_sparse_index(length: int, indices: Any, kind: Literal["block"]) -> BlockIndex: ...
@overload
def make_sparse_index(length: int, indices: Any, kind: Literal["integer"]) -> IntIndex: ...

def make_sparse_index(length: int, indices: Any, kind: SparseIndexKind) -> SparseIndex:
    if kind == "block":
        locs, lens = splib.get_blocks(indices)
        index: SparseIndex = BlockIndex(length, locs, lens)
    elif kind == "integer":
        index = IntIndex(length, indices)
    else:  # pragma: no cover
        raise ValueError("must be block or integer type")
    return index
