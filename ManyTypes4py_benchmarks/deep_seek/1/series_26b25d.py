from __future__ import annotations
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
import functools
import operator
import sys
from textwrap import dedent
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Union,
    cast,
    overload,
)
import warnings
import numpy as np
import pandas as pd
from pandas._libs import lib, properties, reshape
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import ChainedAssignmentError, InvalidIndexError
from pandas.errors.cow import _chained_assignment_method_msg, _chained_assignment_msg
from pandas.util._decorators import (
    Appender,
    Substitution,
    deprecate_nonkeyword_arguments,
    doc,
    set_module,
)
from pandas.util._validators import (
    validate_ascending,
    validate_bool_kwarg,
    validate_percentile,
)
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
    LossySetitemError,
    construct_1d_arraylike_from_scalar,
    find_common_type,
    infer_dtype_from,
    maybe_box_native,
    maybe_cast_pointwise_result,
)
from pandas.core.dtypes.common import (
    is_dict_like,
    is_float,
    is_integer,
    is_iterator,
    is_list_like,
    is_object_dtype,
    is_scalar,
    pandas_dtype,
    validate_all_hashable,
)
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype, SparseDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
    remove_na_arraylike,
)
from pandas.core import (
    algorithms,
    base,
    common as com,
    nanops,
    ops,
    roperator,
)
from pandas.core.accessor import Accessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import ListAccessor, StructAccessor
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
    array as pd_array,
    extract_array,
    sanitize_array,
)
from pandas.core.generic import NDFrame, make_doc
from pandas.core.indexers import disallow_ndim_indexing, unpack_1tuple
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    default_index,
    ensure_index,
    maybe_sequence_to_range,
)
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import check_bool_indexer, check_dict_or_set_indexers
from pandas.core.internals import SingleBlockManager
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import ensure_key_mapped, nargsort
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime
import pandas.io.formats.format as fmt
from pandas.io.formats.info import INFO_DOCSTRING, SeriesInfo, series_sub_kwargs
import pandas.plotting

if TYPE_CHECKING:
    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import (
        AggFuncType,
        AnyAll,
        AnyArrayLike,
        ArrayLike,
        Axis,
        AxisInt,
        CorrelationMethod,
        DropKeep,
        Dtype,
        DtypeObj,
        FilePath,
        Frequency,
        IgnoreRaise,
        IndexKeyFunc,
        IndexLabel,
        Level,
        ListLike,
        MutableMappingT,
        NaPosition,
        NumpySorter,
        NumpyValueArrayLike,
        QuantileInterpolation,
        ReindexMethod,
        Renamer,
        Scalar,
        Self,
        SortKind,
        StorageOptions,
        Suffixes,
        ValueKeyFunc,
        WriteBuffer,
        npt,
    )
    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import SeriesGroupBy

__all__ = ["Series"]

_shared_doc_kwargs = {
    "axes": "index",
    "klass": "Series",
    "axes_single_arg": "{0 or 'index'}",
    "axis": "axis : {0 or 'index'}\n        Unused. Parameter needed for compatibility with DataFrame.",
    "inplace": "inplace : bool, default False\n        If True, performs operation inplace and returns None.",
    "unique": "np.ndarray",
    "duplicated": "Series",
    "optional_by": "",
    "optional_reindex": "\nindex : array-like, optional\n    New labels for the index. Preferably an Index object to avoid\n    duplicating data.\naxis : int or str, optional\n    Unused.",
}

@set_module("pandas")
class Series(base.IndexOpsMixin, NDFrame):
    _typ = "series"
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)
    _metadata = ["_name"]
    _internal_names_set = {"index", "name"} | NDFrame._internal_names_set
    _accessors = {"dt", "cat", "str", "sparse"}
    _hidden_attrs = base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    __pandas_priority__ = 3000
    hasnans = property(base.IndexOpsMixin.hasnans.fget, doc=base.IndexOpsMixin.hasnans.__doc__)

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        dtype: Dtype | None = None,
        name: Hashable | None = None,
        copy: bool | None = None,
    ) -> None:
        allow_mgr = False
        if isinstance(data, SingleBlockManager) and index is None and (dtype is None) and (copy is False or copy is None):
            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            data = data.copy(deep=False)
            NDFrame.__init__(self, data)
            self.name = name
            return
        if isinstance(data, (ExtensionArray, np.ndarray)):
            if copy is not False:
                if dtype is None or astype_is_view(data.dtype, pandas_dtype(dtype)):
                    data = data.copy()
        if copy is None:
            copy = False
        if isinstance(data, SingleBlockManager) and (not copy):
            data = data.copy(deep=False)
            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        name = ibase.maybe_extract_name(name, data, type(self))
        if index is not None:
            index = ensure_index(index)
        if dtype is not None:
            dtype = self._validate_dtype(dtype)
        if data is None:
            index = index if index is not None else default_index(0)
            if len(index) or dtype is not None:
                data = na_value_for_dtype(pandas_dtype(dtype), compat=False)
            else:
                data = []
        if isinstance(data, MultiIndex):
            raise NotImplementedError("initializing a Series from a MultiIndex is not supported")
        refs = None
        if isinstance(data, Index):
            if dtype is not None:
                data = data.astype(dtype)
            refs = data._references
            copy = False
        elif isinstance(data, np.ndarray):
            if len(data.dtype):
                raise ValueError(
                    "Cannot construct a Series from an ndarray with compound dtype.  Use DataFrame instead."
                )
        elif isinstance(data, Series):
            if index is None:
                index = data.index
                data = data._mgr.copy(deep=False)
            else:
                data = data.reindex(index)
                copy = False
                data = data._mgr
        elif isinstance(data, Mapping):
            data, index = self._init_dict(data, index, dtype)
            dtype = None
            copy = False
        elif isinstance(data, SingleBlockManager):
            if index is None:
                index = data.index
            elif not data.index.equals(index) or copy:
                raise AssertionError(
                    "Cannot pass both SingleBlockManager `data` argument and a different `index` argument. `copy` must be False."
                )
            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                allow_mgr = True
        elif isinstance(data, ExtensionArray):
            pass
        else:
            data = com.maybe_iterable_to_list(data)
            if is_list_like(data) and (not len(data)) and (dtype is None):
                dtype = np.dtype(object)
        if index is None:
            if not is_list_like(data):
                data = [data]
            index = default_index(len(data))
        elif is_list_like(data):
            com.require_length_match(data, index)
        if isinstance(data, SingleBlockManager):
            if dtype is not None:
                data = data.astype(dtype=dtype)
            elif copy:
                data = data.copy()
        else:
            data = sanitize_array(data, index, dtype, copy)
            data = SingleBlockManager.from_array(data, index, refs=refs)
        NDFrame.__init__(self, data)
        self.name = name
        self._set_axis(0, index)

    def _init_dict(
        self, data: Mapping, index: Index | None = None, dtype: Dtype | None = None
    ) -> tuple[SingleBlockManager, Index]:
        if data:
            keys = maybe_sequence_to_range(tuple(data.keys()))
            values = list(data.values())
        elif index is not None:
            if len(index) or dtype is not None:
                values = na_value_for_dtype(pandas_dtype(dtype), compat=False)
            else:
                values = []
            keys = index
        else:
            keys, values = (default_index(0), [])
        s = Series(values, index=keys, dtype=dtype)
        if data and index is not None:
            s = s.reindex(index)
        return (s._mgr, s.index)

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        pa = import_optional_dependency("pyarrow", min_version="16.0.0")
        type = pa.DataType._import_from_c_capsule(requested_schema) if requested_schema is not None else None
        ca = pa.array(self, type=type)
        if not isinstance(ca, pa.ChunkedArray):
            ca = pa.chunked_array([ca])
        return ca.__arrow_c_stream__()

    @property
    def _constructor(self) -> type[Series]:
        return Series

    def _constructor_from_mgr(self, mgr: SingleBlockManager, axes: list[Index]) -> Series:
        ser = Series._from_mgr(mgr, axes=axes)
        ser._name = None
        if type(self) is Series:
            return ser
        return self._constructor(ser)

    @property
    def _constructor_expanddim(self) -> type[DataFrame]:
        from pandas.core.frame import DataFrame
        return DataFrame

    def _constructor_expanddim_from_mgr(self, mgr: SingleBlockManager, axes: list[Index]) -> DataFrame:
        from pandas.core.frame import DataFrame
        df = DataFrame._from_mgr(mgr, axes=mgr.axes)
        if type(self) is Series:
            return df
        return self._constructor_expanddim(df)

    @property
    def _can_hold_na(self) -> bool:
        return self._mgr._can_hold_na

    @property
    def dtype(self) -> DtypeObj:
        return self._mgr.dtype

    @property
    def dtypes(self) -> DtypeObj:
        return self.dtype

    @property
    def name(self) -> Hashable | None:
        return self._name

    @name.setter
    def name(self, value: Hashable | None) -> None:
        validate_all_hashable(value, error_name=f"{type(self).__name__}.name")
        object.__setattr__(self, "_name", value)

    @property
    def values(self) -> np.ndarray | ExtensionArray:
        return self._mgr.external_values()

    @property
    def _values(self) -> np.ndarray | ExtensionArray:
        return self._mgr.internal_values()

    @property
    def _references(self) -> BlockValuesRefs | None:
        return self._mgr._block.refs

    @property
    def array(self) -> ExtensionArray:
        return self._mgr.array_values()

    def __len__(self) -> int:
        return len(self._mgr)

    def __array__(self, dtype: npt.DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        values = self._values
        if copy is None:
            arr = np.asarray(values, dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype, copy=copy)
        if copy is True:
            return arr
        if copy is False or astype_is_view(values.dtype, arr.dtype):
            arr = arr.view()
            arr.flags.writeable = False
        return arr

    @property
    def axes(self) -> list[Index]:
        return [self.index]

    def _ixs(self, i: int, axis: int = 0) -> Any:
        return self._values[i]

    def _slice(self, slobj: slice, axis: int = 0) -> Series:
        mgr = self._mgr.get_slice(slobj, axis=axis)
        out = self._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self._name
        return out.__finalize__(self)

    def __getitem__(self, key: Any) -> Any:
        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)
        if key is Ellipsis:
            return self.copy(deep=False)
        key_is_scalar = is_scalar(key)
        if isinstance(key, (list, tuple)):
            key = unpack_1tuple(key)
        elif key_is_scalar:
            return self._get_value(key)
        if is_iterator(key):
            key = list(key)
        if is_hashable(key) and (not isinstance(key, slice)):
            try:
                result = self._get_value(key)
                return result
            except (KeyError, TypeError, InvalidIndexError):
                if isinstance(key, tuple) and isinstance(self.index, MultiIndex):
                    return self._get_values_tuple(key)
        if isinstance(key, slice):
            return self._getitem_slice(key)
        if com.is_bool_indexer(key):
            key = check_bool_indexer(self.index, key)
            key = np.asarray(key, dtype=bool)
            return self._get_rows_with_mask(key)
        return self._get_with(key)

    def _get_with(self, key: Any) -> Any:
        if isinstance(key, ABCDataFrame):
            raise TypeError(
                "Indexing a Series with DataFrame is not supported, use the appropriate DataFrame column"
            )
        elif isinstance(key, tuple):
            return self._get_values_tuple(key)
        return self.loc[key]

    def _get_values_tuple(self, key: tuple) -> Any:
        if com.any_none(*key):
            result = np.asarray(self._values[key])
            disallow_ndim_indexing(result)
            return result
        if not isinstance(self.index, MultiIndex):
            raise KeyError("key of type tuple not found and not a MultiIndex")
        indexer, new_index = self.index.get_loc_level(key)
        new_ser = self._constructor(
            self._values[indexer], index=new_index, copy=False
        )
        if isinstance(indexer, slice):
            new_ser._mgr.add_references(self._mgr)
        return new_ser.__finalize__(self)

    def _get_rows_with_mask(self, indexer: np.ndarray) -> Series:
        new_mgr = self._mgr.get_rows_with_mask(indexer)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    def _get_value(self, label: Hashable, takeable: bool = False) -> Any:
        if takeable:
            return self._values[label]
        loc = self.index.get_loc(label)
        if is_integer(loc):
            return self._values[loc]
        if isinstance(self.index, MultiIndex):
            mi = self.index
            new_values = self._values[loc]
            if len(new_values) == 1 and mi.nlevels == 1:
                return new_values[0]
            new_index = mi[loc]
            new_index = maybe_droplevels(new_index, label)
            new_ser = self._constructor(
                new_values, index=new_index, name=self.name, copy=False
            )
            if isinstance(loc, slice):
                new_ser._mgr.add_references(self._mgr)
            return new_ser.__finalize__(self)
        else:
            return self.iloc[loc]

    def __setitem__(self, key: Any,