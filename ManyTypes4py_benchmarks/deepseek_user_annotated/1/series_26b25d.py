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
    Union,
    Optional,
    List,
    Tuple,
    Dict,
    TypeVar,
    Generic,
    Type,
    Collection,
    MutableMapping,
    Set,
    Iterator,
)
import warnings

import numpy as np
from numpy.typing import NDArray, DTypeLike

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

# Type variables
T = TypeVar('T')
S = TypeVar('S')
MutableMappingT = TypeVar('MutableMappingT', bound=MutableMapping[Any, Any])

@set_module("pandas")
class Series(base.IndexOpsMixin, NDFrame):
    _typ = "series"
    _HANDLED_TYPES = (Index, ExtensionArray, np.ndarray)

    _name: Hashable
    _metadata: List[str] = ["_name"]
    _internal_names_set = {"index", "name"} | NDFrame._internal_names_set
    _accessors = {"dt", "cat", "str", "sparse"}
    _hidden_attrs = (
        base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    )

    __pandas_priority__ = 3000

    @property
    def _constructor(self) -> Type[Series]:
        return Series

    def _constructor_from_mgr(self, mgr, axes) -> Series:
        ser = Series._from_mgr(mgr, axes=axes)
        ser._name = None
        return ser

    @property
    def _constructor_expanddim(self) -> Callable[..., DataFrame]:
        from pandas.core.frame import DataFrame
        return DataFrame

    def _constructor_expanddim_from_mgr(self, mgr, axes) -> DataFrame:
        from pandas.core.frame import DataFrame
        df = DataFrame._from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(self, method="to_frame")

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
    def values(self) -> np.ndarray:
        return self._mgr.external_values()

    @property
    def _values(self) -> ArrayLike:
        return self._mgr.internal_values()

    @property
    def _references(self) -> BlockValuesRefs:
        return self._mgr._block.refs

    @property
    def array(self) -> ExtensionArray:
        return self._mgr.array_values()

    def __len__(self) -> int:
        return len(self._mgr)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
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
    def axes(self) -> List[Index]:
        return [self.index]

    def _ixs(self, i: int, axis: AxisInt = 0) -> Any:
        return self._values[i]

    def _slice(self, slobj: slice, axis: AxisInt = 0) -> Series:
        mgr = self._mgr.get_slice(slobj, axis=axis)
        out = self._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self._name
        return out.__finalize__(self, method="_slice")

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
                return self._get_value(key)
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

    def _get_values_tuple(self, key: Tuple) -> Any:
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

    def _get_rows_with_mask(self, indexer: NDArray[np.bool_]) -> Series:
        new_mgr = self._mgr.get_rows_with_mask(indexer)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(
            self, method="_get_rows_with_mask"
        )

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

    def _set_with_engine(self, key: Hashable, value: Any) -> None:
        loc = self.index.get_loc(key)
        self._mgr.setitem_inplace(loc, value)

    def _set_with(self, key: Any, value: Any) -> None:
        assert not isinstance(key, tuple)
        if is_iterator(key):
            key = list(key)
        self._set_labels(key, value)

    def _set_labels(self, key: Any, value: Any) -> None:
        key = com.asarray_tuplesafe(key)
        indexer: NDArray[np.intp] = self.index.get_indexer(key)
        mask = indexer == -1
        if mask.any():
            raise KeyError(f"{key[mask]} not in index")
        self._set_values(indexer, value)

    def _set_values(self, key: Any, value: Any) -> None:
        if isinstance(key, (Index, Series)):
            key = key._values
        self._mgr = self._mgr.setitem(indexer=key, value=value)

    def _set_value(self, label: Hashable, value: Any, takeable: bool = False) -> None:
        if not takeable:
            try:
                loc = self.index.get_loc(label)
            except KeyError:
                self.loc[label] = value
                return
        else:
            loc = label
        self._set_values(loc, value)

    def repeat(self, repeats: int | Sequence[int], axis: None = None) -> Series:
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
    ) -> DataFrame: ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: Literal[True],
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> Series: ...

    @overload
    def reset_index(
        self,
        level: IndexLabel = ...,
        *,
        drop: bool = ...,
        name: Level = ...,
        inplace: Literal[True],
        allow_duplicates: bool = ...,
    ) -> None: ...

    def reset_index(
        self,
        level: IndexLabel | None = None,
        *,
        drop: bool = False,
        name: Level = lib.no_default,
        inplace: bool = False,
        allow_duplicates: bool = False,
    ) -> DataFrame | Series | None:
        inplace = validate_bool_kwarg(inplace, "inplace")
        if drop:
            new_index = default_index(len(self))
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
                new_ser.index