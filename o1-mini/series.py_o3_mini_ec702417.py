"""
Data structure for 1-dimensional cross-sectional and time series data
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
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
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    properties,
    reshape,
)
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
    ChainedAssignmentError,
    InvalidIndexError,
)
from pandas.errors.cow import (
    _chained_assignment_method_msg,
    _chained_assignment_msg,
)
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
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
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
from pandas.core.arrays.arrow import (
    ListAccessor,
    StructAccessor,
)
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
    array as pd_array,
    extract_array,
    sanitize_array,
)
from pandas.core.generic import (
    NDFrame,
    make_doc,
)
from pandas.core.indexers import (
    disallow_ndim_indexing,
    unpack_1tuple,
)
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
from pandas.core.indexing import (
    check_bool_indexer,
    check_dict_or_set_indexers,
)
from pandas.core.internals import SingleBlockManager
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
    ensure_key_mapped,
    nargsort,
)
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime

import pandas.io.formats.format as fmt
from pandas.io.formats.info import (
    INFO_DOCSTRING,
    SeriesInfo,
    series_sub_kwargs,
)
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
    "axis": """axis : {0 or 'index'}
        Unused. Parameter needed for compatibility with DataFrame.""",
    "inplace": """inplace : bool, default False
        If True, performs operation inplace and returns None.""",
    "unique": "np.ndarray",
    "duplicated": "Series",
    "optional_by": "",
    "optional_reindex": """
index : array-like, optional
    New labels for the index. Preferably an Index object to avoid
    duplicating data.
axis : int or str, optional
    Unused.""",
}

# ----------------------------------------------------------------------
# Series class


@dataclass
@set_module("pandas")
class Series(base.IndexOpsMixin, NDFrame):
    """
    One-dimensional ndarray with axis labels (including time series).

    Labels need not be unique but must be a hashable type. The object
    supports both integer- and label-based indexing and provides a host of
    methods for performing operations involving the index. Statistical
    methods from ndarray have been overridden to automatically exclude
    missing data (currently represented as NaN).

    Operations between Series (+, -, /, *, **) align values based on their
    associated index values-- they need not be the same length. The result
    index will be the sorted union of the two indexes.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series. If data is a dict, argument order is
        maintained. Unordered sets are not supported.
    index : array-like or Index (1d), optional
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If data is dict-like
        and index is None, then the keys in the data are used as the index. If the
        index is not None, the resulting Series is reindexed with the index values.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Series. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.
        If ``data`` is Series then is ignored.
    name : Hashable, default None
        The name to give to the Series.
    copy : bool, default False
        Copy input data. Only affects Series or 1d ndarray input. See examples.

    See Also
    --------
    DataFrame : Two-dimensional, size-mutable, potentially heterogeneous tabular data.
    Index : Immutable sequence used for indexing and alignment.

    Notes
    -----
    Please reference the :ref:`User Guide <basics.series>` for more information.

    Examples
    --------
    Constructing Series from a dictionary with an Index specified

    >>> d = {"a": 1, "b": 2, "c": 3}
    >>> ser = pd.Series(data=d, index=["a", "b", "c"])
    >>> ser
    a   1
    b   2
    c   3
    dtype: int64

    The keys of the dictionary match with the Index values, hence the Index
    values have no effect.

    >>> d = {"a": 1, "b": 2, "c": 3}
    >>> ser = pd.Series(data=d, index=["x", "y", "z"])
    >>> ser
    x   NaN
    y   NaN
    z   NaN
    dtype: float64

    Note that the Index is first built with the keys from the dictionary.
    After this the Series is reindexed with the given Index values, hence we
    get all NaN as a result.

    Constructing Series from a list with `copy=False`.

    >>> r = [1, 2]
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    [1, 2]
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `copy` of
    the original data even though `copy=False`, so
    the data is unchanged.

    Constructing Series from a 1d ndarray with `copy=False`.

    >>> r = np.array([1, 2])
    >>> ser = pd.Series(r, copy=False)
    >>> ser.iloc[0] = 999
    >>> r
    array([999,   2])
    >>> ser
    0    999
    1      2
    dtype: int64

    Due to input data type the Series has a `view` on
    the original data, so
    the data is changed as well.
    """

    _typ: str = "series"
    _HANDLED_TYPES: tuple[type, ...] = (Index, ExtensionArray, np.ndarray)

    _name: Hashable = None
    _metadata: list[str] = field(default_factory=lambda: ["_name"])
    _internal_names_set: set[str] = field(
        default_factory=lambda: {"index", "name"} | NDFrame._internal_names_set
    )
    _accessors: set[str] = field(default_factory=lambda: {"dt", "cat", "str", "sparse"})
    _hidden_attrs: set[str] = field(
        default_factory=lambda: (
            base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
        )
    )

    __pandas_priority__: int = 3000

    hasnans: property[int]
    _mgr: SingleBlockManager

    def __init__(
        self,
        data: Any = None,
        index: Index | Sequence[Any] | None = None,
        dtype: Dtype | None = None,
        name: Hashable | None = None,
        copy: bool | None = None,
    ) -> None:
        allow_mgr = False
        if (
            isinstance(data, SingleBlockManager)
            and index is None
            and dtype is None
            and (copy is False or copy is None)
        ):
            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
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

        if isinstance(data, SingleBlockManager) and not copy:
            data = data.copy(deep=False)

            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
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
            raise NotImplementedError(
                "initializing a Series from a MultiIndex is not supported"
            )

        refs: BlockValuesRefs | None = None
        if isinstance(data, Index):
            if dtype is not None:
                data = data.astype(dtype)

            refs = data._references
            copy = False

        elif isinstance(data, np.ndarray):
            if len(data.dtype):
                raise ValueError(
                    "Cannot construct a Series from an ndarray with "
                    "compound dtype.  Use DataFrame instead."
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
                    "Cannot pass both SingleBlockManager "
                    "`data` argument and a different "
                    "`index` argument. `copy` must be False."
                )

            if not allow_mgr:
                warnings.warn(
                    f"Passing a {type(data).__name__} to {type(self).__name__} "
                    "is deprecated and will raise in a future version. "
                    "Use public APIs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                allow_mgr = True

        elif isinstance(data, ExtensionArray):
            pass
        else:
            data = com.maybe_iterable_to_list(data)
            if is_list_like(data) and not len(data) and dtype is None:
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
        self, data: Mapping, index: Index | None = None, dtype: DtypeObj | None = None
    ) -> tuple[BlockValuesRefs, Index]:
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
            keys, values = default_index(0), []

        s = Series(values, index=keys, dtype=dtype)

        if data and index is not None:
            s = s.reindex(index)
        return s._mgr, s.index

    def __arrow_c_stream__(self, requested_schema=None) -> Any:
        pa = import_optional_dependency("pyarrow", min_version="16.0.0")
        type_ = (
            pa.DataType._import_from_c_capsule(requested_schema)
            if requested_schema is not None
            else None
        )
        ca = pa.array(self, type=type_)
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
    def _constructor_expanddim(self) -> Callable[..., DataFrame]:
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
    def name(self) -> Hashable:
        return self._name

    @name.setter
    def name(self, value: Hashable) -> None:
        validate_all_hashable(value, error_name=f"{type(self).__name__}.name")
        object.__setattr__(self, "_name", value)

    @property
    def values(self) -> AnyArrayLike:
        return self._mgr.external_values()

    @property
    def _values(self) -> AnyArrayLike:
        return self._mgr.internal_values()

    @property
    def _references(self) -> BlockValuesRefs:
        return self._mgr._block.refs

    @classmethod
    def _from_mgr(cls, mgr: SingleBlockManager, axes: list[Index]) -> Series:
        obj = cls.__new__(cls)
        obj._mgr = mgr
        obj._set_axis(0, axes[0])
        obj.name = mgr.items[0]
        return obj

    @property
    def array(self) -> ExtensionArray:
        return self._mgr.array_values()

    def __len__(self) -> int:
        return len(self._mgr)

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
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

    def _ixs(self, i: int, axis: AxisInt = 0) -> Any:
        return self._values[i]

    def _slice(self, slobj: slice, axis: AxisInt = 0) -> Series:
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

        if is_hashable(key) and not isinstance(key, slice):
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
                "Indexing a Series with DataFrame is not "
                "supported, use the appropriate DataFrame column"
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
        new_ser = self._constructor(self._values[indexer], index=new_index, copy=False)
        if isinstance(indexer, slice):
            new_ser._mgr.add_references(self._mgr)
        return new_ser.__finalize__(self)

    def _get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Series:
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

    def __setitem__(self, key: Any, value: Any) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= 3:
                warnings.warn(
                    _chained_assignment_msg, ChainedAssignmentError, stacklevel=2
                )

        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)

        if key is Ellipsis:
            key = slice(None)

        if isinstance(key, slice):
            indexer = self.index._convert_slice_indexer(key, kind="getitem")
            return self._set_values(indexer, value)

        try:
            self._set_with_engine(key, value)
        except KeyError:
            self.loc[key] = value
        except (TypeError, ValueError, LossySetitemError):
            indexer = self.index.get_loc(key)
            self._set_values(indexer, value)
        except InvalidIndexError as err:
            if isinstance(key, tuple) and not isinstance(self.index, MultiIndex):
                raise KeyError(
                    "key of type tuple not found and not a MultiIndex"
                ) from err

            if com.is_bool_indexer(key):
                key = check_bool_indexer(self.index, key)
                key = np.asarray(key, dtype=bool)

                if (
                    is_list_like(value)
                    and len(value) != len(self)
                    and not isinstance(value, Series)
                    and not is_object_dtype(self.dtype)
                ):
                    indexer = key.nonzero()[0]
                    self._set_values(indexer, value)
                    return

                try:
                    self._where(~key, value, inplace=True)
                except InvalidIndexError:
                    self.iloc[key] = value
                return
            else:
                self._set_with(key, value)

    def _set_with_engine(self, key: Any, value: Any) -> None:
        loc = self.index.get_loc(key)
        self._mgr.setitem_inplace(loc, value)

    def _set_with(self, key: Any, value: Any) -> None:
        assert not isinstance(key, tuple)
        if is_iterator(key):
            key = list(key)
        self._set_labels(key, value)

    def _set_labels(self, key: Any, value: Any) -> None:
        key = com.asarray_tuplesafe(key)
        indexer: np.ndarray = self.index.get_indexer(key)
        mask = indexer == -1
        if mask.any():
            raise KeyError(f"{key[mask]} not in index")
        self._set_values(indexer, value)

    def _set_values(self, key: Any, value: Any) -> None:
        if isinstance(key, (Index, Series)):
            key = key._values
        self._mgr = self._mgr.setitem(indexer=key, value=value)

    def _set_value(
        self,
        label: Hashable,
        value: Any,
        takeable: bool = False,
    ) -> None:
        if not takeable:
            try:
                loc = self.index.get_loc(label)
            except KeyError:
                self.loc[label] = value
                return
        else:
            loc = label
        self._set_values(loc, value)

    def repeat(
        self, repeats: int | Sequence[int], axis: Literal[None] = None
    ) -> Series:
        nv.validate_repeat((), {"axis": axis})
        new_index = self.index.repeat(repeats)
        new_values = self._values.repeat(repeats)
        return self._constructor(new_values, index=new_index, copy=False).__finalize__(
            self, method="repeat"
        )

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[False] = ...,
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> DataFrame:
        ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[True],
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> Series:
        ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: bool = ...,
        name: Level = ...,
        inplace: Literal[True],
        allow_duplicates: bool = ...,
    ) -> None:
        ...

    def reset_index(
        self,
        level: IndexLabel | None = None,
        *,
        drop: bool = False,
        name: Level | lib.NoDefault = lib.no_default,
        inplace: bool = False,
        allow_duplicates: bool = False,
    ) -> DataFrame | Series | None:
        inplace = validate_bool_kwarg(inplace, "inplace")
        if drop:
            new_index = default_index(len(self)) if not level else None
            if level is not None:
                level_list: Sequence[Hashable]
                if not isinstance(level, (tuple, list)):
                    level_list = [level]
                else:
                    level_list = level
                level_list = [self.index._get_level_number(lev) for lev in level_list]
                if len(level_list) < self.index.nlevels:
                    new_index = self.index.droplevel(level_list)
            if inplace:
                self.index = new_index
            else:
                new_ser = self.copy(deep=False)
                new_ser.index = new_index
                return new_ser.__finalize__(self, method="reset_index")
        elif inplace:
            raise TypeError(
                "Cannot reset_index inplace on a Series to create a DataFrame"
            )
        else:
            if name is lib.no_default:
                name = 0 if self.name is None else self.name
            df = self.to_frame(name)
            return df.reset_index(
                level=level, drop=drop, allow_duplicates=allow_duplicates
            )
        return None

    def __repr__(self) -> str:
        repr_params = fmt.get_series_repr_params()
        return self.to_string(**repr_params)

    @overload
    def to_string(
        self,
        buf: None = ...,
        *,
        na_rep: str = ...,
        float_format: str | None = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype: bool = ...,
        name: bool = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> str:
        ...

    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        na_rep: str = ...,
        float_format: str | None = ...,
        header: bool = ...,
        index: bool = ...,
        length: bool = ...,
        dtype: bool = ...,
        name: bool = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> None:
        ...

    @deprecate_nonkeyword_arguments(
        version="4.0", allowed_args=["self", "buf"], name="to_string"
    )
    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        na_rep: str = "NaN",
        float_format: str | None = None,
        header: bool = True,
        index: bool = True,
        length: bool = False,
        dtype: bool = False,
        name: bool = False,
        max_rows: int | None = None,
        min_rows: int | None = None,
    ) -> str | None:
        formatter = fmt.SeriesFormatter(
            self,
            name=name,
            length=length,
            header=header,
            index=index,
            dtype=dtype,
            na_rep=na_rep,
            float_format=float_format,
            min_rows=min_rows,
            max_rows=max_rows,
        )
        result = formatter.to_string()

        if not isinstance(result, str):
            raise AssertionError(
                "result must be of type str, type "
                f"of result is {type(result).__name__!r}"
            )

        if buf is None:
            return result
        else:
            if hasattr(buf, "write"):
                buf.write(result)
            else:
                with open(buf, "w", encoding="utf-8") as f:
                    f.write(result)
        return None

    @overload
    def to_markdown(
        self,
        buf: None = ...,
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: StorageOptions | None = ...,
        **kwargs,
    ) -> str:
        ...

    @overload
    def to_markdown(
        self,
        buf: IO[str],
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: StorageOptions | None = ...,
        **kwargs,
    ) -> None:
        ...

    @overload
    def to_markdown(
        self,
        buf: IO[str] | None,
        *,
        mode: str = ...,
        index: bool = ...,
        storage_options: StorageOptions | None = ...,
        **kwargs,
    ) -> str | None:
        ...

    @doc(
        klass=_shared_doc_kwargs["klass"],
        storage_options=_shared_docs["storage_options"],
        examples=dedent(
            """Examples
            --------
            >>> s = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")
            >>> print(s.to_markdown())
            |    | animal   |
            |---:|:---------|
            |  0 | elk      |
            |  1 | pig      |
            |  2 | dog      |
            |  3 | quetzal  |

            Output markdown with a tabulate option.

            >>> print(s.to_markdown(tablefmt="grid"))
            +----+----------+
            |    | animal   |
            +====+==========+
            |  0 | elk      |
            +----+----------+
            |  1 | pig      |
            +----+----------+
            |  2 | dog      |
            +----+----------+
            |  3 | quetzal  |
            +----+----------+"""
        ),
    )
    @deprecate_nonkeyword_arguments(
        version="4.0", allowed_args=["self", "buf"], name="to_markdown"
    )
    def to_markdown(
        self,
        buf: IO[str] | None = None,
        mode: str = "wt",
        index: bool = True,
        storage_options: StorageOptions | None = None,
        **kwargs,
    ) -> str | None:
        return self.to_frame().to_markdown(
            buf, mode=mode, index=index, storage_options=storage_options, **kwargs
        )

    def items(self) -> Iterable[tuple[Hashable, Any]]:
        return zip(iter(self.index), iter(self))

    def keys(self) -> Index:
        return self.index

    @overload
    def to_dict(
        self, *, into: type[MutableMappingT] | MutableMappingT
    ) -> MutableMappingT:
        ...

    @overload
    def to_dict(self, *, into: type[dict] = ...) -> dict:
        ...

    def to_dict(
        self,
        *,
        into: type[MutableMappingT] | MutableMappingT = dict,
    ) -> MutableMappingT:
        into_c = com.standardize_mapping(into)

        if is_object_dtype(self.dtype) or isinstance(self.dtype, ExtensionDtype):
            return into_c((k, maybe_box_native(v)) for k, v in self.items())
        else:
            return into_c(self.items())

    def to_frame(self, name: Hashable | lib.NoDefault = lib.no_default) -> DataFrame:
        columns: Index
        if name is lib.no_default:
            name = self.name
            if name is None:
                columns = default_index(1)
            else:
                columns = Index([name])
        else:
            columns = Index([name])

        mgr = self._mgr.to_2d_mgr(columns)
        df = self._constructor_expanddim_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(self, method="to_frame")

    def _set_name(
        self, name: Hashable, inplace: bool = False, deep: bool | None = None
    ) -> Series:
        inplace = validate_bool_kwarg(inplace, "inplace")
        ser = self if inplace else self.copy(deep=False)
        ser.name = name
        return ser

    def groupby(
        self,
        by: Any | Sequence[Any] | None = None,
        level: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ) -> SeriesGroupBy:
        from pandas.core.groupby.generic import SeriesGroupBy

        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")

        return SeriesGroupBy(
            obj=self,
            keys=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    def count(self) -> int:
        return notna(self._values).sum().astype("int64")

    def mode(self, dropna: bool = True) -> Series:
        values = self._values
        if isinstance(values, np.ndarray):
            res_values = algorithms.mode(values, dropna=dropna)
        else:
            res_values = values._mode(dropna=dropna)

        return self._constructor(
            res_values,
            index=range(len(res_values)),
            name=self.name,
            copy=False,
            dtype=self.dtype,
        ).__finalize__(self, method="mode")

    def unique(self) -> ArrayLike:
        return super().unique()

    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[False] = ...,
        ignore_index: bool = ...,
    ) -> Series:
        ...

    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[True],
        ignore_index: bool = ...,
    ) -> None:
        ...

    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: bool = ...,
        ignore_index: bool = ...,
    ) -> Series | None:
        ...

    def drop_duplicates(
        self,
        *,
        keep: DropKeep = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> Series | None:
        inplace = validate_bool_kwarg(inplace, "inplace")
        result = super().drop_duplicates(keep=keep)

        if ignore_index:
            result.index = default_index(len(result))

        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, keep: DropKeep = "first") -> Series:
        res = self._duplicated(keep=keep)
        result = self._constructor(res, index=self.index, copy=False)
        return result.__finalize__(self, method="duplicated")

    def idxmin(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Hashable:
        axis = self._get_axis_number(axis)
        iloc = self.argmin(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def idxmax(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Hashable:
        axis = self._get_axis_number(axis)
        iloc = self.argmax(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def round(
        self,
        decimals: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        nv.validate_round(args, kwargs)
        new_mgr = self._mgr.round(decimals=decimals)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(
            self, method="round"
        )

    @overload
    def quantile(
        self,
        q: float = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float:
        ...

    @overload
    def quantile(
        self,
        q: Sequence[float] | AnyArrayLike,
        interpolation: QuantileInterpolation = ...,
    ) -> Series:
        ...

    @overload
    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float | Series:
        ...

    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = 0.5,
        interpolation: QuantileInterpolation = "linear",
    ) -> float | Series:
        validate_percentile(q)
        df = self.to_frame()
        result = df.quantile(q=q, interpolation=interpolation, numeric_only=False)
        if result.ndim == 2:
            result = result.iloc[:, 0]

        if is_list_like(q):
            result.name = self.name
            idx = Index(q, dtype=np.float64)
            return self._constructor(result, index=idx, name=self.name)
        else:
            return result.iloc[0]

    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: int | None = None,
    ) -> float:
        this, other = self.align(other, join="inner")
        if len(this) == 0:
            return np.nan

        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)

        if method in ["pearson", "spearman", "kendall"] or callable(method):
            return nanops.nancorr(
                this_values, other_values, method=method, min_periods=min_periods
            )

        raise ValueError(
            "method must be either 'pearson', "
            "'spearman', 'kendall', or a callable, "
            f"'{method}' was supplied"
        )

    def cov(
        self,
        other: Series,
        min_periods: int | None = None,
        ddof: int | None = 1,
    ) -> float:
        this, other = self.align(other, join="inner")
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)
        return nanops.nancov(
            this_values, other_values, min_periods=min_periods, ddof=ddof
        )

    @doc(
        _shared_docs["groupby"],
        klass=_shared_doc_kwargs["klass"],
        examples=_shared_doc_kwargs["examples"],
    )
    def groupby(
        self,
        by: Any | Sequence[Any] | None = None,
        level: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ) -> SeriesGroupBy:
        from pandas.core.groupby.generic import SeriesGroupBy

        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")

        return SeriesGroupBy(
            obj=self,
            keys=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    def __matmul__(self, other: AnyArrayLike | DataFrame) -> Series | np.ndarray:
        return self.dot(other)

    def __rmatmul__(self, other: AnyArrayLike | DataFrame) -> Series | np.ndarray:
        return self.dot(np.transpose(other))

    def _cmp_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        res_name = ops.get_op_result_name(self, other)

        if isinstance(other, Series) and not self._indexed_same(other):
            raise ValueError("Can only compare identically-labeled Series objects")

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        res_values = op(lvalues, rvalues)

        return self._construct_result(res_values, name=res_name)

    def _logical_method(
        self, other: Any, op: Callable[[Any, Any], Any]
    ) -> Series:
        res_name = ops.get_op_result_name(self, other)
        self, other = self._align_for_op(other, align_asobject=True)

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        res_values = op(lvalues, rvalues)
        return self._construct_result(res_values, name=res_name)

    def _arith_method(
        self, other: Any, op: Callable[[Any, Any], Any]
    ) -> Series:
        self, other = self._align_for_op(other)
        return base.IndexOpsMixin._arith_method(self, other, op)

    def _flex_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
        *,
        level: Level | None = None,
        fill_value: Any = None,
        axis: Axis = 0,
    ) -> Series:
        if axis is not None:
            self._get_axis_number(axis)

        res_name = ops.get_op_result_name(self, other)

        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError("Lengths must be equal")
            other = self._constructor(other, self.index, copy=False)
            result = self._binop(other, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                if isna(other):
                    return op(self, fill_value)
                self = self.fillna(fill_value)

            return op(self, other)

    def _construct_result(
        self, result: ArrayLike | tuple[ArrayLike, ArrayLike], name: Hashable
    ) -> Series | tuple[Series, Series]:
        if isinstance(result, tuple):
            res1 = self._construct_result(result[0], name=name)
            res2 = self._construct_result(result[1], name=name)
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)

        dtype = getattr(result, "dtype", None)
        out = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)

        out.name = name
        return out

    def _align_for_op(
        self, right: Any, align_asobject: bool = False
    ) -> tuple[Series, Any]:
        left = self

        if isinstance(right, Series):
            if not left.index.equals(right.index):
                if align_asobject:
                    if left.dtype not in (object, np.bool_) or right.dtype not in (
                        object,
                        np.bool_,
                    ):
                        pass
                    else:
                        left = left.astype(object)
                        right = right.astype(object)
                left, right = left.align(right)

        return left, right

    def _binop(
        self,
        other: Series,
        func: Callable[[Any, Any], Any],
        level: Level | None = None,
        fill_value: Any | None = None,
    ) -> Series:
        this = self

        if not self.index.equals(other.index):
            this, other = self.align(other, level=level, join="outer")

        this_vals, other_vals = ops.fill_binop(this._values, other._values, fill_value)

        with np.errstate(all="ignore"):
            result = func(this_vals, other_vals)

        name = ops.get_op_result_name(self, other)
        out = self._construct_result(result, name)
        return cast(Series, out)

    def _map_values(
        self,
        mapper: Callable[[Any], Any] | Mapping[Hashable, Any] | Series,
        na_action: Literal["ignore"] | None = None,
    ) -> ArrayLike:
        if isinstance(mapper, Series):
            mapper = mapper.reindex(self.index).to_dict()
        elif isinstance(mapper, Mapping):
            mapper = {k: v for k, v in mapper.items()}

        return algorithms.map_values(self._values, mapper, na_action=na_action)

    def _update_inplace(self, result: Series) -> None:
        self._mgr = result._mgr
        self.name = result.name
        self._clear_cache()

    def cummin(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    def cummax(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    def cumsum(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    def cumprod(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)

    def compare(
        self,
        other: Series,
        align_axis: Axis = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ("self", "other"),
    ) -> DataFrame | Series:
        return super().compare(
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
            result_names=result_names,
        )

    def combine(
        self,
        other: Series | Hashable,
        func: Callable[[Hashable, Hashable], Hashable],
        fill_value: Hashable | None = None,
    ) -> Series:
        if fill_value is None:
            fill_value = na_value_for_dtype(self.dtype, compat=False)

        if isinstance(other, Series):
            new_index = self.index.union(other.index)
            new_name = ops.get_op_result_name(self, other)
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all="ignore"):
                for i, idx in enumerate(new_index):
                    lv = self.get(idx, fill_value)
                    rv = other.get(idx, fill_value)
                    new_values[i] = func(lv, rv)
        else:
            new_index = self.index
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all="ignore"):
                new_values[:] = [func(lv, other) for lv in self._values]
            new_name = self.name

        npvalues = lib.maybe_convert_objects(new_values, try_float=False)
        same_dtype = isinstance(self.dtype, (StringDtype, CategoricalDtype))
        res_values = maybe_cast_pointwise_result(
            npvalues, self.dtype, same_dtype=same_dtype
        )
        return self._constructor(res_values, index=new_index, name=new_name, copy=False)

    def combine_first(self, other: Series) -> Series:
        from pandas.core.reshape.concat import concat

        if self.dtype == other.dtype:
            if self.index.equals(other.index):
                return self.mask(self.isna(), other)
            elif self._can_hold_na and not isinstance(self.dtype, SparseDtype):
                this, other = self.align(other, join="outer")
                return this.mask(this.isna(), other)

        new_index = self.index.union(other.index)

        this = self
        keep_other = other.index.difference(this.index[notna(this)])
        keep_this = this.index.difference(keep_other)

        this = this.reindex(keep_this)
        other = other.reindex(keep_other)

        if this.dtype.kind == "M" and other.dtype.kind != "M":
            other = to_datetime(other)
        combined = concat([this, other])
        combined = combined.reindex(new_index)
        return combined.__finalize__(self, method="combine_first")

    def update(self, other: Series | Sequence | Mapping) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(
                    _chained_assignment_method_msg,
                    ChainedAssignmentError,
                    stacklevel=2,
                )

        if not isinstance(other, Series):
            other = Series(other)

        other = other.reindex_like(self)
        mask = notna(other)

        self._mgr = self._mgr.putmask(mask=mask, new=other)

    def _reindex_indexer(
        self,
        new_index: Index | None,
        indexer: np.ndarray | None,
    ) -> Series:
        if indexer is None and (
            new_index is None or new_index.names == self.index.names
        ):
            return self.copy(deep=False)

        new_values = algorithms.take_nd(
            self._values, indexer, allow_fill=True, fill_value=None
        )
        return self._constructor(new_values, index=new_index, copy=False)

    def _needs_reindex_multi(self, axes: list[Index], method: str, level: Level | None) -> bool:
        return False

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: Literal[bool, lib.NoDefault] = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None:
        ...

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: Literal[bool, lib.NoDefault] = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series:
        ...

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: Literal[bool, lib.NoDefault] = ...,
        inplace: bool = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series | None:
        ...

    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool | lib.NoDefault = ...,
        inplace: bool = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series | None:
        ...

    def rename(
        self,
        index: Renamer | Hashable | None = None,
        *,
        axis: Axis | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
        inplace: bool = False,
        level: Level | None = None,
        errors: IgnoreRaise = "ignore",
    ) -> Series | None:
        self._check_copy_deprecation(copy)
        if axis is not None:
            axis = self._get_axis_number(axis)

        if callable(index) or is_dict_like(index):
            return super()._rename(
                index=index,
                inplace=inplace,
                level=level,
                errors=errors,
            )
        else:
            return self._set_name(index, inplace=inplace)

    def to_markdown(
        self,
        buf: IO[str] | None = None,
        mode: str = "wt",
        index: bool = True,
        storage_options: StorageOptions | None = None,
        **kwargs: Any,
    ) -> str | None:
        return self.to_frame().to_markdown(
            buf, mode=mode, index=index, storage_options=storage_options, **kwargs
        )

    def items(self) -> Iterable[tuple[Hashable, Any]]:
        return zip(iter(self.index), iter(self))

    def keys(self) -> Index:
        return self.index

    @overload
    def to_dict(
        self, *, into: type[MutableMappingT] | MutableMappingT
    ) -> MutableMappingT:
        ...

    @overload
    def to_dict(self, *, into: type[dict] = ...) -> dict:
        ...

    def to_dict(
        self,
        *,
        into: type[MutableMappingT] | MutableMappingT = dict,
    ) -> MutableMappingT:
        into_c = com.standardize_mapping(into)

        if is_object_dtype(self.dtype) or isinstance(self.dtype, ExtensionDtype):
            return into_c((k, maybe_box_native(v)) for k, v in self.items())
        else:
            return into_c(self.items())

    def to_frame(self, name: Hashable | lib.NoDefault = lib.no_default) -> DataFrame:
        columns: Index
        if name is lib.no_default:
            name = self.name
            if name is None:
                columns = default_index(1)
            else:
                columns = Index([name])
        else:
            columns = Index([name])

        mgr = self._mgr.to_2d_mgr(columns)
        df = self._constructor_expanddim_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(self, method="to_frame")

    def _set_name(
        self, name: Hashable, inplace: bool = False, deep: bool | None = None
    ) -> Series:
        inplace = validate_bool_kwarg(inplace, "inplace")
        ser = self if inplace else self.copy(deep=False)
        ser.name = name
        return ser

    def groupby(
        self,
        by: Any | Sequence[Any] | None = None,
        level: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ) -> SeriesGroupBy:
        from pandas.core.groupby.generic import SeriesGroupBy

        if level is None and by is not None:
            if callable(by):
                mapped_keys = by(self)
                other = mapped_keys
            elif is_dict_like(by):
                other = [by.get(k, k) for k in self.index]
            else:
                other = by
        elif level is not None:
            other = None
        else:
            raise TypeError("You have to supply one of 'by' and 'level'")

        return SeriesGroupBy(
            obj=self,
            keys=other,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    def count(self) -> int:
        return notna(self._values).sum().astype("int64")

    def mode(self, dropna: bool = True) -> Series:
        values = self._values
        if isinstance(values, np.ndarray):
            res_values = algorithms.mode(values, dropna=dropna)
        else:
            res_values = values._mode(dropna=dropna)

        return self._constructor(
            res_values,
            index=range(len(res_values)),
            name=self.name,
            copy=False,
            dtype=self.dtype,
        ).__finalize__(self, method="mode")

    def unique(self) -> ArrayLike:
        return super().unique()

    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[False] = ...,
        ignore_index: bool = ...,
    ) -> Series:
        ...

    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[True],
        ignore_index: bool = ...,
    ) -> None:
        ...

    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: bool = ...,
        ignore_index: bool = ...,
    ) -> Series | None:
        ...

    def drop_duplicates(
        self,
        *,
        keep: DropKeep = "first",
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> Series | None:
        inplace = validate_bool_kwarg(inplace, "inplace")
        result = super().drop_duplicates(keep=keep)

        if ignore_index:
            result.index = default_index(len(result))

        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, keep: DropKeep = "first") -> Series:
        res = self._duplicated(keep=keep)
        result = self._constructor(res, index=self.index, copy=False)
        return result.__finalize__(self, method="duplicated")

    def idxmin(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Hashable:
        axis = self._get_axis_number(axis)
        iloc = self.argmin(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def idxmax(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Hashable:
        axis = self._get_axis_number(axis)
        iloc = self.argmax(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def round(
        self,
        decimals: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> Series:
        nv.validate_round(args, kwargs)
        new_mgr = self._mgr.round(decimals=decimals)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(
            self, method="round"
        )

    @overload
    def quantile(
        self,
        q: float = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float:
        ...

    @overload
    def quantile(
        self,
        q: Sequence[float] | AnyArrayLike,
        interpolation: QuantileInterpolation = ...,
    ) -> Series:
        ...

    @overload
    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float | Series:
        ...

    def quantile(
        self,
        q: float | Sequence[float] | AnyArrayLike = 0.5,
        interpolation: QuantileInterpolation = "linear",
    ) -> float | Series:
        validate_percentile(q)
        df = self.to_frame()
        result = df.quantile(q=q, interpolation=interpolation, numeric_only=False)
        if result.ndim == 2:
            result = result.iloc[:, 0]

        if is_list_like(q):
            result.name = self.name
            idx = Index(q, dtype=np.float64)
            return self._constructor(result, index=idx, name=self.name)
        else:
            return result.iloc[0]

    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: int | None = None,
    ) -> float:
        this, other = self.align(other, join="inner")
        if len(this) == 0:
            return np.nan

        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)

        if method in ["pearson", "spearman", "kendall"] or callable(method):
            return nanops.nancorr(
                this_values, other_values, method=method, min_periods=min_periods
            )

        raise ValueError(
            "method must be either 'pearson', "
            "'spearman', 'kendall', or a callable, "
            f"'{method}' was supplied"
        )

    def cov(
        self,
        other: Series,
        min_periods: int | None = None,
        ddof: int | None = 1,
    ) -> float:
        this, other = self.align(other, join="inner")
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)
        return nanops.nancov(
            this_values, other_values, min_periods=min_periods, ddof=ddof
        )

    def min(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.min(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def max(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.max(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def sum(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.sum(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def prod(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.prod(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def mean(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.mean(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def median(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.median(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def sem(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.sem(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def var(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.var(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def std(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.std(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def skew(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.skew(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def kurt(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.kurt(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def argsort(
        self,
        axis: Axis = 0,
        kind: SortKind = "quicksort",
        order: None = None,
        stable: None = None,
    ) -> Series:
        if axis != -1:
            self._get_axis_number(axis)

        result = self.array.argsort(kind=kind)

        res = self._constructor(
            result, index=self.index, name=self.name, dtype=np.intp, copy=False
        )
        return res.__finalize__(self, method="argsort")

    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nsmallest()

    def swaplevel(
        self, i: Level = -2, j: Level = -1, copy: bool | lib.NoDefault = lib.no_default
    ) -> Series:
        self._check_copy_deprecation(copy)
        assert isinstance(self.index, MultiIndex)
        result = self.copy(deep=False)
        result.index = self.index.swaplevel(i, j)
        return result

    def reorder_levels(self, order: Sequence[Level]) -> Series:
        if not isinstance(self.index, MultiIndex):
            raise Exception("Can only reorder levels on a hierarchical axis.")

        result = self.copy(deep=False)
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result

    def explode(self, ignore_index: bool = False) -> Series:
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result = self.copy()
            return result.reset_index(drop=True) if ignore_index else result

        new_index = default_index(len(values)) if ignore_index else self.index.repeat(counts)
        return self._constructor(values, index=new_index, name=self.name, copy=False)

    def to_timestamp(
        self,
        freq: Frequency | None = None,
        how: Literal["s", "e", "start", "end"] = "start",
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        new_obj = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, "index", new_index)
        return new_obj

    def to_period(
        self,
        freq: str | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        new_obj = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, "index", new_index)
        return new_obj

    def _update_inplace(self, result: Series) -> None:
        self._mgr = result._mgr
        self.name = result.name
        self._clear_cache()

    def map(
        self,
        arg: Callable[[Any], Any] | Mapping[Hashable, Any] | Series,
        na_action: Literal["ignore"] | None = None,
        **kwargs: Any,
    ) -> Series:
        if callable(arg):
            arg = functools.partial(arg, **kwargs)
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(
            self, method="map"
        )

    def _gotitem(self, key: Any, ndim: int, subset: Any = None) -> Self:
        return self

    def execute_numba(
        self, engine: Any, task: Any, i: int, j: int, name: Any
    ) -> None:
        pass

    def apply(
        self,
        func: AggFuncType,
        args: tuple[Any, ...] = (),
        *,
        by_row: Literal[False, "compat"] = "compat",
        **kwargs: Any,
    ) -> DataFrame | Series:
        return SeriesApply(
            self,
            func,
            by_row=by_row,
            args=args,
            kwargs=kwargs,
        ).apply()

    def isin(self, values: AnyArrayLike | Callable[..., Any], /) -> Series:
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(
            self, method="isin"
        )

    def between(
        self,
        left: Any,
        right: Any,
        inclusive: Literal["both", "neither", "left", "right"] = "both",
    ) -> Series:
        if inclusive == "both":
            lmask = self >= left
            rmask = self <= right
        elif inclusive == "left":
            lmask = self >= left
            rmask = self < right
        elif inclusive == "right":
            lmask = self > left
            rmask = self <= right
        elif inclusive == "neither":
            lmask = self > left
            rmask = self < right
        else:
            raise ValueError(
                "Inclusive has to be either string of 'both',"
                "'left', 'right', or 'neither'."
            )

        return lmask & rmask

    def case_when(
        self,
        caselist: list[
            tuple[
                ArrayLike | Callable[[Series], Series | np.ndarray | Sequence[bool]],
                ArrayLike | Scalar | Callable[[Series], Series | np.ndarray],
            ],
        ],
    ) -> Series:
        if not isinstance(caselist, list):
            raise TypeError(
                f"The caselist argument should be a list; instead got {type(caselist)}"
            )

        if not caselist:
            raise ValueError(
                "provide at least one boolean condition, "
                "with a corresponding replacement."
            )

        for num, entry in enumerate(caselist):
            if not isinstance(entry, tuple):
                raise TypeError(
                    f"Argument {num} must be a tuple; instead got {type(entry)}."
                )
            if len(entry) != 2:
                raise ValueError(
                    f"Argument {num} must have length 2; "
                    "a condition and replacement; "
                    f"instead got length {len(entry)}."
                )
        caselist = [
            (
                com.apply_if_callable(condition, self),
                com.apply_if_callable(replacement, self),
            )
            for condition, replacement in caselist
        ]
        default = self.copy(deep=False)
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(
                        value=replacement, length=len(condition), dtype=common_dtype
                    )
                elif isinstance(replacement, ABCSeries):
                    replacement = replacement.astype(common_dtype)
                else:
                    replacement = pd_array(replacement, dtype=common_dtype)
                updated_replacements.append(replacement)
            replacements = updated_replacements
            default = default.astype(common_dtype)

        counter = range(len(conditions) - 1, -1, -1)
        for position, condition, replacement in zip(
            counter, reversed(conditions), reversed(replacements)
        ):
            try:
                default = default.mask(
                    condition, other=replacement, axis=0, inplace=False, level=None
                )
            except Exception as error:
                raise ValueError(
                    f"Failed to apply condition{position} and replacement{position}."
                ) from error
        return default

    def any(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        nv.validate_logical_func((), kwargs, fname="any")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanany,
            name="any",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    def all(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        nv.validate_logical_func((), kwargs, fname="all")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanall,
            name="all",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    def any_any(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        return self.any(axis=axis, bool_only=bool_only, skipna=skipna, **kwargs)

    def all_all(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        return self.all(axis=axis, bool_only=bool_only, skipna=skipna, **kwargs)

    def min(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.min(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def max(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.max(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def argsort(
        self,
        axis: Axis = 0,
        kind: SortKind = "quicksort",
        order: None = None,
        stable: None = None,
    ) -> Series:
        if axis != -1:
            self._get_axis_number(axis)

        result = self.array.argsort(kind=kind)

        res = self._constructor(
            result, index=self.index, name=self.name, dtype=np.intp, copy=False
        )
        return res.__finalize__(self, method="argsort")

    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nsmallest()

    def swaplevel(
        self, i: Level = -2, j: Level = -1, copy: bool | lib.NoDefault = lib.no_default
    ) -> Series:
        self._check_copy_deprecation(copy)
        assert isinstance(self.index, MultiIndex)
        result = self.copy(deep=False)
        result.index = self.index.swaplevel(i, j)
        return result

    def reorder_levels(self, order: Sequence[Level]) -> Series:
        if not isinstance(self.index, MultiIndex):
            raise Exception("Can only reorder levels on a hierarchical axis.")

        result = self.copy(deep=False)
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result

    def explode(self, ignore_index: bool = False) -> Series:
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result = self.copy()
            return result.reset_index(drop=True) if ignore_index else result

        new_index = default_index(len(values)) if ignore_index else self.index.repeat(counts)
        return self._constructor(values, index=new_index, name=self.name, copy=False)

    def to_timestamp(
        self,
        freq: Frequency | None = None,
        how: Literal["s", "e", "start", "end"] = "start",
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        new_obj = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, "index", new_index)
        return new_obj

    def to_period(
        self,
        freq: str | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        new_obj = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, "index", new_index)
        return new_obj

    def update(self, other: Series | Sequence | Mapping) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(
                    _chained_assignment_method_msg,
                    ChainedAssignmentError,
                    stacklevel=2,
                )

        if not isinstance(other, Series):
            other = Series(other)

        other = other.reindex_like(self)
        mask = notna(other)

        self._mgr = self._mgr.putmask(mask=mask, new=other)

    def __repr__(self) -> str:
        repr_params = fmt.get_series_repr_params()
        return self.to_string(**repr_params)

    def groupby(
        self,
        by: Any | Sequence[Any] | None = None,
        level: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ) -> SeriesGroupBy:
        from pandas.core.groupby.generic import SeriesGroupBy

        if level is None and by is not None:
            if callable(by):
                mapped_keys = by(self)
                other = mapped_keys
            elif is_dict_like(by):
                other = [by.get(k, k) for k in self.index]
            else:
                other = by
        elif level is not None:
            other = None
        else:
            raise TypeError("You have to supply one of 'by' and 'level'")

        return SeriesGroupBy(
            obj=self,
            keys=other,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    def any(self, *, axis: Axis = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> bool:
        nv.validate_logical_func((), kwargs, fname="any")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanany,
            name="any",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    def all(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        nv.validate_logical_func((), kwargs, fname="all")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanall,
            name="all",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    def any_any(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        return self.any(axis=axis, bool_only=bool_only, skipna=skipna, **kwargs)

    def all_all(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        return self.all(axis=axis, bool_only=bool_only, skipna=skipna, **kwargs)

    def sum(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.sum(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def prod(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.prod(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def mean(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.mean(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def median(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.median(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def sem(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.sem(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def var(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.var(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def std(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.std(
            self,
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def skew(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.skew(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def kurt(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.kurt(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def any(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        nv.validate_logical_func((), kwargs, fname="any")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanany,
            name="any",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    def all(
        self,
        *,
        axis: Axis = 0,
        bool_only: bool = False,
        skipna: bool = True,
        **kwargs: Any,
    ) -> bool:
        nv.validate_logical_func((), kwargs, fname="all")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(
            nanops.nanall,
            name="all",
            axis=axis,
            numeric_only=bool_only,
            skipna=skipna,
            filter_type="bool",
        )

    def min(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.min(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def max(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.max(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def prod(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.prod(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def sum(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.sum(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def skew(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.skew(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def kurt(
        self,
        axis: Axis | None = 0,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        return NDFrame.kurt(
            self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    def _cmp_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        res_name = ops.get_op_result_name(self, other)

        if isinstance(other, Series) and not self._indexed_same(other):
            raise ValueError("Can only compare identically-labeled Series objects")

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        res_values = op(lvalues, rvalues)

        return self._construct_result(res_values, name=res_name)

    def _logical_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Series:
        res_name = ops.get_op_result_name(self, other)
        self, other = self._align_for_op(other, align_asobject=True)

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        res_values = op(lvalues, rvalues)
        return self._construct_result(res_values, name=res_name)

    def _arith_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Series:
        self, other = self._align_for_op(other)
        return base.IndexOpsMixin._arith_method(self, other, op)

    def _flex_method(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
        *,
        level: Level | None = None,
        fill_value: Any | None = None,
        axis: Axis = 0,
    ) -> Series:
        if axis is not None:
            self._get_axis_number(axis)

        res_name = ops.get_op_result_name(self, other)

        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError("Lengths must be equal")
            other = self._constructor(other, self.index, copy=False)
            result = self._binop(other, op, level=level, fill_value=fill_value)
            result._name = res_name
            return result
        else:
            if fill_value is not None:
                if isna(other):
                    return op(self, fill_value)
                self = self.fillna(fill_value)

            return op(self, other)

    def _construct_result(
        self, result: ArrayLike | tuple[ArrayLike, ArrayLike], name: Hashable
    ) -> Series | tuple[Series, Series]:
        if isinstance(result, tuple):
            res1 = self._construct_result(result[0], name=name)
            res2 = self._construct_result(result[1], name=name)
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)

        dtype = getattr(result, "dtype", None)
        out = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)

        out.name = name
        return out

    def _align_for_op(
        self, right: Any, align_asobject: bool = False
    ) -> tuple[Series, Any]:
        left = self

        if isinstance(right, Series):
            if not left.index.equals(right.index):
                if align_asobject:
                    if left.dtype not in (object, np.bool_) or right.dtype not in (
                        object,
                        np.bool_,
                    ):
                        pass
                    else:
                        left = left.astype(object)
                        right = right.astype(object)
                left, right = left.align(right)

        return left, right

    def _binop(
        self,
        other: Series,
        func: Callable[[Any, Any], Any],
        level: Level | None = None,
        fill_value: Any | None = None,
    ) -> Series:
        this = self

        if not self.index.equals(other.index):
            this, other = self.align(other, level=level, join="outer")

        this_vals, other_vals = ops.fill_binop(this._values, other._values, fill_value)

        with np.errstate(all="ignore"):
            result = func(this_vals, other_vals)

        name = ops.get_op_result_name(self, other)
        out = self._construct_result(result, name)
        return cast(Series, out)

    def _map_values(
        self,
        mapper: Callable[[Any], Any] | Mapping[Hashable, Any] | Series,
        na_action: Literal["ignore"] | None = None,
    ) -> ArrayLike:
        if isinstance(mapper, Series):
            mapper = mapper.reindex(self.index).to_dict()
        elif isinstance(mapper, Mapping):
            mapper = {k: v for k, v in mapper.items()}

        return algorithms.map_values(self._values, mapper, na_action=na_action)

    def _update_inplace(self, result: Series) -> None:
        self._mgr = result._mgr
        self.name = result.name
        self._clear_cache()

    def explode(self, ignore_index: bool = False) -> Series:
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result = self.copy()
            return result.reset_index(drop=True) if ignore_index else result

        new_index = default_index(len(values)) if ignore_index else self.index.repeat(counts)
        return self._constructor(values, index=new_index, name=self.name, copy=False)

    def to_timestamp(
        self,
        freq: Frequency | None = None,
        how: Literal["s", "e", "start", "end"] = "start",
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        new_obj = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, "index", new_index)
        return new_obj

    def to_period(
        self,
        freq: str | None = None,
        copy: bool | lib.NoDefault = lib.no_default,
    ) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")

        new_obj = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, "index", new_index)
        return new_obj

    def update(self, other: Series | Sequence | Mapping) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(
                    _chained_assignment_method_msg,
                    ChainedAssignmentError,
                    stacklevel=2,
                )

        if not isinstance(other, Series):
            other = Series(other)

        other = other.reindex_like(self)
        mask = notna(other)

        self._mgr = self._mgr.putmask(mask=mask, new=other)

    def groupby(
        self,
        by: Any | Sequence[Any] | None = None,
        level: IndexLabel | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
    ) -> SeriesGroupBy:
        from pandas.core.groupby.generic import SeriesGroupBy

        if level is None and by is not None:
            if callable(by):
                mapped_keys = by(self)
                other = mapped_keys
            elif is_dict_like(by):
                other = [by.get(k, k) for k in self.index]
            else:
                other = by
        elif level is not None:
            other = None
        else:
            raise TypeError("You have to supply one of 'by' and 'level'")

        return SeriesGroupBy(
            obj=self,
            keys=other,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    def map(
        self,
        arg: Callable[[Any], Any] | Mapping[Hashable, Any] | Series,
        na_action: Literal["ignore"] | None = None,
        **kwargs: Any,
    ) -> Series:
        if callable(arg):
            arg = functools.partial(arg, **kwargs)
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(
            self, method="map"
        )

    def apply(
        self,
        func: AggFuncType,
        args: tuple[Any, ...] = (),
        *,
        by_row: Literal[False, "compat"] = "compat",
        **kwargs: Any,
    ) -> DataFrame | Series:
        return SeriesApply(
            self,
            func,
            by_row=by_row,
            args=args,
            kwargs=kwargs,
        ).apply()

    def isin(self, values: AnyArrayLike | Callable[..., Any], /) -> Series:
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(
            self, method="isin"
        )

    def between(
        self,
        left: Any,
        right: Any,
        inclusive: Literal["both", "neither", "left", "right"] = "both",
    ) -> Series:
        if inclusive == "both":
            lmask = self >= left
            rmask = self <= right
        elif inclusive == "left":
            lmask = self >= left
            rmask = self < right
        elif inclusive == "right":
            lmask = self > left
            rmask = self <= right
        elif inclusive == "neither":
            lmask = self > left
            rmask = self < right
        else:
            raise ValueError(
                "Inclusive has to be either string of 'both',"
                "'left', 'right', or 'neither'."
            )

        return lmask & rmask

    def case_when(
        self,
        caselist: list[
            tuple[
                ArrayLike | Callable[[Series], Series | np.ndarray | Sequence[bool]],
                ArrayLike | Scalar | Callable[[Series], Series | np.ndarray],
            ],
        ],
    ) -> Series:
        if not isinstance(caselist, list):
            raise TypeError(
                f"The caselist argument should be a list; instead got {type(caselist)}"
            )

        if not caselist:
            raise ValueError(
                "provide at least one boolean condition, "
                "with a corresponding replacement."
            )

        for num, entry in enumerate(caselist):
            if not isinstance(entry, tuple):
                raise TypeError(
                    f"Argument {num} must be a tuple; instead got {type(entry)}."
                )
            if len(entry) != 2:
                raise ValueError(
                    f"Argument {num} must have length 2; "
                    "a condition and replacement; "
                    f"instead got length {len(entry)}."
                )
        caselist = [
            (
                com.apply_if_callable(condition, self),
                com.apply_if_callable(replacement, self),
            )
            for condition, replacement in caselist
        ]
        default = self.copy(deep=False)
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(
                        value=replacement, length=len(condition), dtype=common_dtype
                    )
                elif isinstance(replacement, ABCSeries):
                    replacement = replacement.astype(common_dtype)
                else:
                    replacement = pd_array(replacement, dtype=common_dtype)
                updated_replacements.append(replacement)
            replacements = updated_replacements
            default = default.astype(common_dtype)

        counter = range(len(conditions) - 1, -1, -1)
        for position, condition, replacement in zip(
            counter, reversed(conditions), reversed(replacements)
        ):
            try:
                default = default.mask(
                    condition, other=replacement, axis=0, inplace=False, level=None
                )
            except Exception as error:
                raise ValueError(
                    f"Failed to apply condition{position} and replacement{position}."
                ) from error
        return default
