from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
import functools
import operator
import sys
from textwrap import dedent
from typing import IO, Any, Optional, Union, cast, overload, List, Tuple, Sequence, Dict
import warnings
import numpy as np

from pandas._libs import lib, properties, reshape
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import ChainedAssignmentError, InvalidIndexError
from pandas.errors.cow import _chained_assignment_method_msg, _chained_assignment_msg
from pandas.util._decorators import Appender, Substitution, deprecate_nonkeyword_arguments, doc, set_module
from pandas.util._validators import validate_ascending, validate_bool_kwarg, validate_percentile
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import LossySetitemError, construct_1d_arraylike_from_scalar, find_common_type, infer_dtype_from, maybe_box_native, maybe_cast_pointwise_result
from pandas.core.dtypes.common import is_dict_like, is_float, is_integer, is_iterator, is_list_like, is_object_dtype, is_scalar, pandas_dtype, validate_all_hashable
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype, SparseDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import isna, na_value_for_dtype, notna, remove_na_arraylike
from pandas.core import algorithms, base, common as com, nanops, ops, roperator
from pandas.core.accessor import Accessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import ListAccessor, StructAccessor
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import array as pd_array, extract_array, sanitize_array
from pandas.core.generic import NDFrame, make_doc
from pandas.core.indexers import disallow_ndim_indexing, unpack_1tuple
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex, PeriodIndex, default_index, ensure_index, maybe_sequence_to_range
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
if False:
    from pandas._libs.internals import BlockValuesRefs
    from pandas._typing import AggFuncType, AnyAll, AnyArrayLike, ArrayLike, Axis, AxisInt, CorrelationMethod, DropKeep, Dtype, DtypeObj, FilePath, Frequency, IgnoreRaise, IndexKeyFunc, IndexLabel, Level, ListLike, MutableMappingT, NaPosition, NumpySorter, NumpyValueArrayLike, QuantileInterpolation, ReindexMethod, Renamer, Scalar, Self, SortKind, StorageOptions, Suffixes, ValueKeyFunc, WriteBuffer, npt
    from pandas.core.frame import DataFrame
    from pandas.core.groupby.generic import SeriesGroupBy

__all__ = ['Series']
_shared_doc_kwargs = {'axes': 'index', 'klass': 'Series', 'axes_single_arg': "{0 or 'index'}", 'axis': "axis : {0 or 'index'}\n        Unused. Parameter needed for compatibility with DataFrame.", 'inplace': 'inplace : bool, default False\n        If True, performs operation inplace and returns None.', 'unique': 'np.ndarray', 'duplicated': 'Series', 'optional_by': '', 'optional_reindex': '\nindex : array-like, optional\n    New labels for the index. Preferably an Index object to avoid\n    duplicating data.\naxis : int or str, optional\n    Unused.'}

@set_module("pandas")
class Series(base.IndexOpsMixin, NDFrame):
    _typ: str = "series"
    _HANDLED_TYPES: Tuple[type, ...] = (Index, ExtensionArray, np.ndarray)
    _metadata: List[str] = ["_name"]
    _internal_names_set: set = {"index", "name"} | NDFrame._internal_names_set
    _accessors: Dict[str, type] = {"dt": CombinedDatetimelikeProperties, "cat": CategoricalAccessor, "str": StringMethods, "sparse": SparseAccessor}
    _hidden_attrs = base.IndexOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    __pandas_priority__ = 3000
    hasnans = property(base.IndexOpsMixin.hasnans.fget, doc=base.IndexOpsMixin.hasnans.__doc__)

    def __init__(
        self,
        data: Any = None,
        index: Optional[Any] = None,
        dtype: Optional[Any] = None,
        name: Optional[Any] = None,
        copy: Optional[bool] = None,
    ) -> None:
        allow_mgr: bool = False
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
                allow_mgr = True
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
                raise ValueError("Cannot construct a Series from an ndarray with compound dtype.  Use DataFrame instead.")
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
                raise AssertionError("Cannot pass both SingleBlockManager `data` argument and a different `index` argument. `copy` must be False.")
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

    def _init_dict(self, data: Mapping[Any, Any], index: Optional[Any], dtype: Optional[Any]) -> Tuple[Any, Any]:
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

    def __arrow_c_stream__(self, requested_schema: Optional[Any] = None) -> Any:
        pa = import_optional_dependency("pyarrow", min_version="16.0.0")
        type_arg = pa.DataType._import_from_c_capsule(requested_schema) if requested_schema is not None else None
        ca = pa.array(self, type=type_arg)
        if not isinstance(ca, pa.ChunkedArray):
            ca = pa.chunked_array([ca])
        return ca.__arrow_c_stream__()

    @property
    def _constructor(self) -> Callable[..., Series]:
        return Series

    def _constructor_from_mgr(self, mgr: Any, axes: Any) -> Series:
        ser: Series = Series._from_mgr(mgr, axes=axes)
        ser._name = None
        if type(self) is Series:
            return ser
        return self._constructor(ser)

    @property
    def _constructor_expanddim(self) -> Callable[..., Any]:
        from pandas.core.frame import DataFrame
        return DataFrame

    def _constructor_expanddim_from_mgr(self, mgr: Any, axes: Any) -> Any:
        from pandas.core.frame import DataFrame

        df = DataFrame._from_mgr(mgr, axes=mgr.axes)
        if type(self) is Series:
            return df
        return self._constructor_expanddim(df)

    @property
    def _can_hold_na(self) -> bool:
        return self._mgr._can_hold_na

    @property
    def dtype(self) -> Any:
        return self._mgr.dtype

    @property
    def dtypes(self) -> Any:
        return self.dtype

    @property
    def name(self) -> Any:
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        validate_all_hashable(value, error_name=f'{type(self).__name__}.name')
        object.__setattr__(self, "_name", value)

    @property
    def values(self) -> Any:
        return self._mgr.external_values()

    @property
    def _values(self) -> Any:
        return self._mgr.internal_values()

    @property
    def _references(self) -> Any:
        return self._mgr._block.refs

    @Appender(base.IndexOpsMixin.array.__doc__)
    @property
    def array(self) -> Any:
        return self._mgr.array_values()

    def __len__(self) -> int:
        return len(self._mgr)

    def __array__(self, dtype: Optional[Any] = None, copy: Optional[bool] = None) -> np.ndarray:
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

    def _ixs(self, i: int, axis: int = 0) -> Any:
        return self._values[i]

    def _slice(self, slobj: slice, axis: int = 0) -> Series:
        mgr = self._mgr.get_slice(slobj, axis=axis)
        out: Series = self._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self._name
        return out.__finalize__(self)

    def __getitem__(self, key: Any) -> Any:
        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)
        if key is Ellipsis:
            return self.copy(deep=False)
        key_is_scalar: bool = is_scalar(key)
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
            raise TypeError("Indexing a Series with DataFrame is not supported, use the appropriate DataFrame column")
        elif isinstance(key, tuple):
            return self._get_values_tuple(key)
        return self.loc[key]

    def _get_values_tuple(self, key: Tuple[Any, ...]) -> Any:
        if com.any_none(*key):
            result = np.asarray(self._values[key])
            disallow_ndim_indexing(result)
            return result
        if not isinstance(self.index, MultiIndex):
            raise KeyError("key of type tuple not found and not a MultiIndex")
        indexer, new_index = self.index.get_loc_level(key)
        new_ser: Series = self._constructor(self._values[indexer], index=new_index, name=self.name, copy=False)
        if isinstance(indexer, slice):
            new_ser._mgr.add_references(self._mgr)
        return new_ser.__finalize__(self)

    def _get_rows_with_mask(self, indexer: np.ndarray) -> Series:
        new_mgr = self._mgr.get_rows_with_mask(indexer)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self)

    def _get_value(self, label: Any, takeable: bool = False) -> Any:
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
            new_ser: Series = self._constructor(new_values, index=new_index, name=self.name, copy=False)
            if isinstance(loc, slice):
                new_ser._mgr.add_references(self._mgr)
            return new_ser.__finalize__(self)
        else:
            return self.iloc[loc]

    def __setitem__(self, key: Any, value: Any) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= 3:
                warnings.warn(_chained_assignment_msg, ChainedAssignmentError, stacklevel=2)
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
            if isinstance(key, tuple) and (not isinstance(self.index, MultiIndex)):
                raise KeyError("key of type tuple not found and not a MultiIndex") from err
            if com.is_bool_indexer(key):
                key = check_bool_indexer(self.index, key)
                key = np.asarray(key, dtype=bool)
                if is_list_like(value) and len(value) != len(self) and (not isinstance(value, Series)) and (not is_object_dtype(self.dtype)):
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
        indexer = self.index.get_indexer(key)
        mask = indexer == -1
        if mask.any():
            raise KeyError(f"{key[mask]} not in index")
        self._set_values(indexer, value)

    def _set_values(self, key: Any, value: Any) -> None:
        if isinstance(key, (Index, Series)):
            key = key._values
        self._mgr = self._mgr.setitem(indexer=key, value=value)

    def _set_value(self, label: Any, value: Any, takeable: bool = False) -> None:
        if not takeable:
            try:
                loc = self.index.get_loc(label)
            except KeyError:
                self.loc[label] = value
                return
        else:
            loc = label
        self._set_values(loc, value)

    def repeat(self, repeats: Union[int, np.ndarray], axis: Optional[Any] = None) -> Series:
        nv.validate_repeat((), {'axis': axis})
        new_index = self.index.repeat(repeats)
        new_values = self._values.repeat(repeats)
        return self._constructor(new_values, index=new_index, copy=False).__finalize__(self, method="repeat")

    @overload
    def reset_index(self, level: Any = ..., *, drop: bool, name: Any = ..., inplace: bool, allow_duplicates: bool) -> Any:
        ...

    def reset_index(self, level: Optional[Any] = None, *, drop: bool = False, name: Any = lib.no_default, inplace: bool = False, allow_duplicates: bool = False) -> Optional[Union[Series, Any]]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        if drop:
            new_index = default_index(len(self))
            if level is not None:
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
                new_ser: Series = self.copy(deep=False)
                new_ser.index = new_index
                return new_ser.__finalize__(self, method="reset_index")
        elif inplace:
            raise TypeError("Cannot reset_index inplace on a Series to create a DataFrame")
        else:
            if name is lib.no_default:
                if self.name is None:
                    name = 0
                else:
                    name = self.name
            from pandas.core.frame import DataFrame

            df: DataFrame = self.to_frame(name)
            return df.reset_index(level=level, drop=drop, allow_duplicates=allow_duplicates)
        return None

    def __repr__(self) -> str:
        repr_params = fmt.get_series_repr_params()
        return self.to_string(**repr_params)

    @overload
    def to_string(self, buf: Optional[Any] = ..., *, na_rep: str, float_format: Optional[Callable[[Any], str]], header: bool, index: bool, length: bool, dtype: bool, name: bool, max_rows: Optional[int], min_rows: Optional[int]) -> Any:
        ...

    def to_string(self, buf: Optional[Any] = None, na_rep: str = "NaN", float_format: Optional[Callable[[Any], str]] = None, header: bool = True, index: bool = True, length: bool = False, dtype: bool = False, name: bool = False, max_rows: Optional[int] = None, min_rows: Optional[int] = None) -> Optional[str]:
        formatter = fmt.SeriesFormatter(self, name=name, length=length, header=header, index=index, dtype=dtype, na_rep=na_rep, float_format=float_format, min_rows=min_rows, max_rows=max_rows)
        result: str = formatter.to_string()
        if not isinstance(result, str):
            raise AssertionError(f"result must be of type str, type of result is {type(result).__name__!r}")
        if buf is None:
            return result
        elif hasattr(buf, "write"):
            buf.write(result)
        else:
            with open(buf, "w", encoding="utf-8") as f:
                f.write(result)
        return None

    @overload
    def to_markdown(self, buf: Optional[Any] = ..., *, mode: str, index: bool, storage_options: Optional[Mapping[str, Any]], **kwargs: Any) -> Any:
        ...

    def to_markdown(self, buf: Optional[Any] = None, mode: str = "wt", index: bool = True, storage_options: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> str:
        return self.to_frame().to_markdown(buf, mode=mode, index=index, storage_options=storage_options, **kwargs)

    def items(self) -> Iterable[Tuple[Any, Any]]:
        return zip(iter(self.index), iter(self))

    def keys(self) -> Index:
        return self.index

    @overload
    def to_dict(self, *, into: Any) -> Any:
        ...

    def to_dict(self, *, into: Any = dict) -> Any:
        into_c = com.standardize_mapping(into)
        if is_object_dtype(self.dtype) or isinstance(self.dtype, ExtensionDtype):
            return into_c(((k, maybe_box_native(v)) for k, v in self.items()))
        else:
            return into_c(self.items())

    def to_frame(self, name: Any = lib.no_default) -> Any:
        if name is lib.no_default:
            name = self.name
            if name is None:
                columns = default_index(1)
            else:
                columns = Index([name])
        else:
            columns = Index([name])
        mgr = self._mgr.to_2d_mgr(columns)
        from pandas.core.frame import DataFrame

        df = self._constructor_expanddim_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(self, method="to_frame")

    def _set_name(self, name: Any, inplace: bool = False, deep: Optional[Any] = None) -> Series:
        inplace = validate_bool_kwarg(inplace, "inplace")
        ser: Series = self if inplace else self.copy(deep=False)
        ser.name = name
        return ser

    def groupby(
        self,
        by: Optional[Any] = None,
        level: Optional[Any] = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True
    ) -> Any:
        from pandas.core.groupby.generic import SeriesGroupBy
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")
        return SeriesGroupBy(obj=self, keys=by, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)

    def count(self) -> int:
        return notna(self._values).sum().astype("int64")

    def mode(self, dropna: bool = True) -> Series:
        values = self._values
        if isinstance(values, np.ndarray):
            res_values = algorithms.mode(values, dropna=dropna)
        else:
            res_values = values._mode(dropna=dropna)
        return self._constructor(res_values, index=range(len(res_values)), name=self.name, copy=False, dtype=self.dtype).__finalize__(self, method="mode")

    def unique(self) -> Any:
        return super().unique()

    def drop_duplicates(self, *, keep: Union[str, bool] = "first", inplace: bool = False, ignore_index: bool = False) -> Optional[Series]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        result: Series = super().drop_duplicates(keep=keep)
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, keep: Union[str, bool] = "first") -> Series:
        res = self._duplicated(keep=keep)
        result: Series = self._constructor(res, index=self.index, copy=False)
        return result.__finalize__(self, method="duplicated")

    def idxmin(self, axis: Union[int, str] = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        axis = self._get_axis_number(axis)
        iloc = self.argmin(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def idxmax(self, axis: Union[int, str] = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Any:
        axis = self._get_axis_number(axis)
        iloc = self.argmax(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def round(self, decimals: int = 0, *args: Any, **kwargs: Any) -> Series:
        nv.validate_round(args, kwargs)
        new_mgr = self._mgr.round(decimals=decimals)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self, method="round")

    @overload
    def quantile(self, q: Union[float, Sequence[float]] = ..., interpolation: str = ...) -> Union[float, Series]:
        ...

    def quantile(self, q: Union[float, Sequence[float]] = 0.5, interpolation: str = "linear") -> Union[float, Series]:
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

    def corr(self, other: Series, method: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "pearson", min_periods: Optional[int] = None) -> float:
        this, other = self.align(other, join="inner")
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if method in ["pearson", "spearman", "kendall"] or callable(method):
            return nanops.nancorr(this_values, other_values, method=method, min_periods=min_periods)
        raise ValueError(f"method must be either 'pearson', 'spearman', 'kendall', or a callable, '{method}' was supplied")

    def cov(self, other: Series, min_periods: Optional[int] = None, ddof: int = 1) -> float:
        this, other = self.align(other, join="inner")
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)
        return nanops.nancov(this_values, other_values, min_periods=min_periods, ddof=ddof)

    def diff(self, periods: int = 1) -> Series:
        if not lib.is_integer(periods):
            if not (is_float(periods) and periods.is_integer()):
                raise ValueError("periods must be an integer")
        result = algorithms.diff(self._values, periods)
        return self._constructor(result, index=self.index, copy=False).__finalize__(self, method="diff")

    def autocorr(self, lag: int = 1) -> float:
        return self.corr(cast(Series, self.shift(lag)))

    def dot(self, other: Any) -> Union[int, float, Series, np.ndarray]:
        if isinstance(other, (Series, ABCDataFrame)):
            common = self.index.union(other.index)
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError("matrices are not aligned")
            left = self.reindex(index=common)
            right = other.reindex(index=common)
            lvals = left.values
            rvals = right.values
        else:
            lvals = self.values
            rvals = np.asarray(other)
            if lvals.shape[0] != rvals.shape[0]:
                raise Exception(f"Dot product shape mismatch, {lvals.shape} vs {rvals.shape}")
        if isinstance(other, ABCDataFrame):
            return self._constructor(np.dot(lvals, rvals), index=other.columns, copy=False).__finalize__(self, method="dot")
        elif isinstance(other, Series):
            return np.dot(lvals, rvals)
        elif isinstance(rvals, np.ndarray):
            return np.dot(lvals, rvals)
        else:
            raise TypeError(f"unsupported type: {type(other)}")

    def __matmul__(self, other: Any) -> Any:
        return self.dot(other)

    def __rmatmul__(self, other: Any) -> Any:
        return self.dot(np.transpose(other))

    def searchsorted(self, value: Any, side: str = "left", sorter: Optional[Any] = None) -> Any:
        return base.IndexOpsMixin.searchsorted(self, value, side=side, sorter=sorter)

    def _append(self, to_append: Any, ignore_index: bool = False, verify_integrity: bool = False) -> Series:
        from pandas.core.reshape.concat import concat
        if isinstance(to_append, (list, tuple)):
            to_concat: List[Any] = [self]
            to_concat.extend(to_append)
        else:
            to_concat = [self, to_append]
        if any((isinstance(x, (ABCDataFrame,)) for x in to_concat[1:])):
            msg = "to_append should be a Series or list/tuple of Series, got DataFrame"
            raise TypeError(msg)
        return concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity)

    def compare(self, other: Series, align_axis: int = 1, keep_shape: bool = False, keep_equal: bool = False, result_names: Tuple[str, str] = ("self", "other")) -> Union[Series, Any]:
        return super().compare(other=other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal, result_names=result_names)

    def combine(self, other: Union[Series, Any], func: Callable[[Any, Any], Any], fill_value: Optional[Any] = None) -> Series:
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
        res_values = maybe_cast_pointwise_result(npvalues, self.dtype, same_dtype=same_dtype)
        return self._constructor(res_values, index=new_index, name=new_name, copy=False)

    def combine_first(self, other: Series) -> Series:
        from pandas.core.reshape.concat import concat
        if self.dtype == other.dtype:
            if self.index.equals(other.index):
                return self.mask(self.isna(), other)
            elif self._can_hold_na and (not isinstance(self.dtype, SparseDtype)):
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

    def update(self, other: Any) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(_chained_assignment_method_msg, ChainedAssignmentError, stacklevel=2)
        if not isinstance(other, Series):
            other = Series(other)
        other = other.reindex_like(self)
        mask = notna(other)
        self._mgr = self._mgr.putmask(mask=mask, new=other)

    def sort_values(
        self,
        *,
        axis: Optional[Any] = 0,
        ascending: Union[bool, Sequence[bool]] = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
        ignore_index: bool = False,
        key: Optional[Callable[[Series], Any]] = None
    ) -> Optional[Series]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        self._get_axis_number(axis)
        if is_list_like(ascending):
            ascending = cast(Sequence[bool], ascending)
            if len(ascending) != 1:
                raise ValueError(f"Length of ascending ({len(ascending)}) must be 1 for Series")
            ascending = ascending[0]
        ascending = validate_ascending(ascending)
        if na_position not in ["first", "last"]:
            raise ValueError(f"invalid na_position: {na_position}")
        if key:
            values_to_sort = cast(Series, ensure_key_mapped(self, key))._values
        else:
            values_to_sort = self._values
        sorted_index = nargsort(values_to_sort, kind, bool(ascending), na_position)
        if is_range_indexer(sorted_index, len(sorted_index)):
            if inplace:
                return self._update_inplace(self)
            return self.copy(deep=False)
        result: Series = self._constructor(self._values[sorted_index], index=self.index[sorted_index], copy=False)
        if ignore_index:
            result.index = default_index(len(sorted_index))
        if not inplace:
            return result.__finalize__(self, method="sort_values")
        self._update_inplace(result)
        return None

    def sort_index(
        self,
        *,
        axis: Optional[Any] = 0,
        level: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        ascending: Union[bool, Sequence[bool]] = True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
        sort_remaining: bool = True,
        ignore_index: bool = False,
        key: Optional[Callable[[Index], Index]] = None
    ) -> Optional[Series]:
        return super().sort_index(axis=axis, level=level, ascending=ascending, inplace=inplace, kind=kind, na_position=na_position, sort_remaining=sort_remaining, ignore_index=ignore_index, key=key)

    def argsort(self, axis: int = 0, kind: str = "quicksort", order: Any = None, stable: Any = None) -> Series:
        if axis != -1:
            self._get_axis_number(axis)
        result = self.array.argsort(kind=kind)
        res: Series = self._constructor(result, index=self.index, name=self.name, dtype=np.intp, copy=False)
        return res.__finalize__(self, method="argsort")

    def nlargest(self, n: int = 5, keep: Union[str, bool] = "first") -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(self, n: int = 5, keep: Union[str, bool] = "first") -> Series:
        return selectn.SelectNSeries(self, n=n, keep=keep).nsmallest()

    def swaplevel(self, i: Union[int, str] = -2, j: Union[int, str] = -1, copy: Any = lib.no_default) -> Series:
        self._check_copy_deprecation(copy)
        assert isinstance(self.index, MultiIndex)
        result: Series = self.copy(deep=False)
        result.index = self.index.swaplevel(i, j)
        return result

    def reorder_levels(self, order: List[Union[int, str]]) -> Series:
        if not isinstance(self.index, MultiIndex):
            raise Exception("Can only reorder levels on a hierarchical axis.")
        result: Series = self.copy(deep=False)
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result

    def explode(self, ignore_index: bool = False) -> Series:
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result: Series = self.copy()
            return result.reset_index(drop=True) if ignore_index else result
        if ignore_index:
            index = default_index(len(values))
        else:
            index = self.index.repeat(counts)
        return self._constructor(values, index=index, name=self.name, copy=False)

    def unstack(self, level: Union[int, str] = -1, fill_value: Optional[Any] = None, sort: bool = True) -> Any:
        from pandas.core.reshape.reshape import unstack
        return unstack(self, level, fill_value, sort)

    def map(self, arg: Union[Callable[[Any], Any], Mapping[Any, Any], Series], na_action: Optional[str] = None, **kwargs: Any) -> Series:
        if callable(arg):
            arg = functools.partial(arg, **kwargs)
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(self, method="map")

    def _gotitem(self, key: Any, ndim: int, subset: Optional[Any] = None) -> Any:
        return self

    _agg_see_also_doc: str = dedent("\n    See Also\n    --------\n    Series.apply : Invoke function on a Series.\n    Series.transform : Transform function producing a Series with like indexes.\n    ")
    _agg_examples_doc: str = dedent("\n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4])\n    >>> s\n    0    1\n    1    2\n    2    3\n    3    4\n    dtype: int64\n\n    >>> s.agg('min')\n    1\n\n    >>> s.agg(['min', 'max'])\n    min   1\n    max   4\n    dtype: int64\n    ")

    def aggregate(self, func: Optional[Any] = None, axis: int = 0, *args: Any, **kwargs: Any) -> Any:
        self._get_axis_number(axis)
        if func is None:
            func = dict(kwargs.items())
        op = SeriesApply(self, func, args=args, kwargs=kwargs)
        result = op.agg()
        return result
    agg = aggregate

    def transform(self, func: Any, axis: int = 0, *args: Any, **kwargs: Any) -> Series:
        self._get_axis_number(axis)
        ser: Series = self.copy(deep=False)
        result = SeriesApply(ser, func=func, args=args, kwargs=kwargs).transform()
        return result

    def apply(self, func: Any, args: Tuple[Any, ...] = (), *, by_row: Union[bool, str] = "compat", **kwargs: Any) -> Any:
        return SeriesApply(self, func, by_row=by_row, args=args, kwargs=kwargs).apply()

    def _reindex_indexer(self, new_index: Any, indexer: Optional[Any]) -> Series:
        if indexer is None and (new_index is None or new_index.names == self.index.names):
            return self.copy(deep=False)
        new_values = algorithms.take_nd(self._values, indexer, allow_fill=True, fill_value=None)
        return self._constructor(new_values, index=new_index, copy=False)

    def _needs_reindex_multi(self, axes: Any, method: Any, level: Any) -> bool:
        return False

    @overload
    def rename(self, index: Any = ..., *, axis: Any, copy: Any, inplace: bool, level: Any, errors: str) -> Any:
        ...

    def rename(self, index: Optional[Any] = None, *, axis: Optional[Any] = None, copy: Any = lib.no_default, inplace: bool = False, level: Optional[Any] = None, errors: str = "ignore") -> Optional[Series]:
        self._check_copy_deprecation(copy)
        if axis is not None:
            axis = self._get_axis_number(axis)
        if callable(index) or is_dict_like(index):
            return super()._rename(index, inplace=inplace, level=level, errors=errors)
        else:
            return self._set_name(index, inplace=inplace)

    @Appender(NDFrame.set_axis.__doc__)
    def set_axis(self, labels: Any, *, axis: int = 0, copy: Any = lib.no_default) -> Any:
        return super().set_axis(labels, axis=axis, copy=copy)

    def reindex(self, index: Optional[Any] = None, *, axis: Optional[Any] = None, method: Optional[Any] = None, copy: Any = lib.no_default, level: Optional[Any] = None, fill_value: Optional[Any] = None, limit: Optional[int] = None, tolerance: Optional[Any] = None) -> Any:
        return super().reindex(index=index, method=method, level=level, fill_value=fill_value, limit=limit, tolerance=tolerance, copy=copy)

    @overload
    def rename_axis(self, mapper: Any = ..., *, index: Any, axis: Any, copy: Any, inplace: bool) -> Any:
        ...

    def rename_axis(self, mapper: Any = lib.no_default, *, index: Any = lib.no_default, axis: int = 0, copy: Any = lib.no_default, inplace: bool = False) -> Optional[Series]:
        return super().rename_axis(mapper=mapper, index=index, axis=axis, inplace=inplace, copy=copy)

    def drop(
        self,
        labels: Optional[Any] = None,
        *,
        axis: Any = 0,
        index: Optional[Any] = None,
        columns: Optional[Any] = None,
        level: Optional[Any] = None,
        inplace: bool = False,
        errors: str = "raise"
    ) -> Optional[Series]:
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    def pop(self, item: Any) -> Any:
        return super().pop(item=item)

    def info(self, verbose: Optional[bool] = None, buf: Optional[IO[str]] = None, max_cols: Optional[int] = None, memory_usage: Optional[bool] = None, show_counts: bool = True) -> Any:
        return SeriesInfo(self, memory_usage).render(buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts)

    def memory_usage(self, index: bool = True, deep: bool = False) -> int:
        v: int = self._memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v

    def isin(self, values: Any) -> Series:
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(self, method="isin")

    def between(self, left: Any, right: Any, inclusive: str = "both") -> Series:
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
            raise ValueError("Inclusive has to be either string of 'both','left', 'right', or 'neither'.")
        return lmask & rmask

    def case_when(self, caselist: List[Tuple[Any, Any]]) -> Series:
        if not isinstance(caselist, list):
            raise TypeError(f"The caselist argument should be a list; instead got {type(caselist)}")
        if not caselist:
            raise ValueError("provide at least one boolean condition, with a corresponding replacement.")
        for num, entry in enumerate(caselist):
            if not isinstance(entry, tuple):
                raise TypeError(f"Argument {num} must be a tuple; instead got {type(entry)}.")
            if len(entry) != 2:
                raise ValueError(f"Argument {num} must have length 2; a condition and replacement; instead got length {len(entry)}.")
        caselist = [(com.apply_if_callable(condition, self), com.apply_if_callable(replacement, self)) for condition, replacement in caselist]
        default: Series = self.copy(deep=False)
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(value=replacement, length=len(condition), dtype=common_dtype)
                elif isinstance(replacement, ABCSeries):
                    replacement = replacement.astype(common_dtype)
                else:
                    replacement = pd_array(replacement, dtype=common_dtype)
                updated_replacements.append(replacement)
            replacements = updated_replacements
            default = default.astype(common_dtype)
        counter = range(len(conditions) - 1, -1, -1)
        for position, condition, replacement in zip(counter, reversed(conditions), reversed(replacements)):
            try:
                default = default.mask(condition, other=replacement, axis=0, inplace=False, level=None)
            except Exception as error:
                raise ValueError(f"Failed to apply condition{position} and replacement{position}.") from error
        return default

    def isna(self) -> Any:
        return NDFrame.isna(self)

    def isnull(self) -> Any:
        return super().isnull()

    def notna(self) -> Any:
        return super().notna()

    def notnull(self) -> Any:
        return super().notnull()

    def dropna(self, *, axis: Any = 0, inplace: bool = False, how: Optional[Any] = None, ignore_index: bool = False) -> Optional[Series]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        ignore_index = validate_bool_kwarg(ignore_index, "ignore_index")
        self._get_axis_number(axis or 0)
        if self._can_hold_na:
            result: Series = remove_na_arraylike(self)
        elif not inplace:
            result = self.copy(deep=False)
        else:
            result = self
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            return self._update_inplace(result)
        else:
            return result

    def to_timestamp(self, freq: Optional[str] = None, how: str = "start", copy: Any = lib.no_default) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")
        new_obj: Series = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, "index", new_index)
        return new_obj

    def to_period(self, freq: Optional[str] = None, copy: Any = lib.no_default) -> Series:
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f"unsupported Type {type(self.index).__name__}")
        new_obj: Series = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, "index", new_index)
        return new_obj

    _AXIS_ORDERS = ["index"]
    _AXIS_LEN = len(_AXIS_ORDERS)
    _info_axis_number = 0
    _info_axis_name = "index"
    index = properties.AxisProperty(axis=0, doc="""
        The index (axis labels) of the Series.

        The index of a Series is used to label and identify each element of the
        underlying data. The index can be thought of as an immutable ordered set
        (technically a multi-set, as it may contain duplicate labels), and is
        used to index and align data in pandas.

        Returns
        -------
        Index
            The index labels of the Series.

        See Also
        --------
        Series.reindex : Conform Series to new index.
        Index : The base pandas index type.

        Notes
        -----
        For more information on pandas indexing, see the `indexing user guide
        <https://pandas.pydata.org/docs/user_guide/indexing.html>`__.

        Examples
        --------
        To create a Series with a custom index and view the index labels:

        >>> cities = ['Kolkata', 'Chicago', 'Toronto', 'Lisbon']
        >>> populations = [14.85, 2.71, 2.93, 0.51]
        >>> city_series = pd.Series(populations, index=cities)
        >>> city_series.index
        Index(['Kolkata', 'Chicago', 'Toronto', 'Lisbon'], dtype='object')

        To change the index labels of an existing Series:

        >>> city_series.index = ['KOL', 'CHI', 'TOR', 'LIS']
        >>> city_series.index
        Index(['KOL', 'CHI', 'TOR', 'LIS'], dtype='object')
        """)
    str = Accessor("str", StringMethods)
    dt = Accessor("dt", CombinedDatetimelikeProperties)
    cat = Accessor("cat", CategoricalAccessor)
    plot = Accessor("plot", pandas.plotting.PlotAccessor)
    sparse = Accessor("sparse", SparseAccessor)
    struct = Accessor("struct", StructAccessor)
    list = Accessor("list", ListAccessor)
    hist = pandas.plotting.hist_series

    def _cmp_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Series:
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series) and (not self._indexed_same(other)):
            raise ValueError("Can only compare identically-labeled Series objects")
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.comparison_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _logical_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Series:
        res_name = ops.get_op_result_name(self, other)
        self, other = self._align_for_op(other, align_asobject=True)
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.logical_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _arith_method(self, other: Any, op: Callable[[Any, Any], Any]) -> Any:
        self, other = self._align_for_op(other)
        return base.IndexOpsMixin._arith_method(self, other, op)

    def _align_for_op(self, right: Any, align_asobject: bool = False) -> Tuple[Series, Series]:
        left = self
        if isinstance(right, Series):
            if not left.index.equals(right.index):
                if align_asobject:
                    if left.dtype not in (object, np.bool_) or right.dtype not in (object, np.bool_):
                        pass
                    else:
                        left = left.astype(object)
                        right = right.astype(object)
                left, right = left.align(right)
        return (left, right)

    def _binop(self, other: Series, func: Callable[[Any, Any], Any], level: Optional[Any] = None, fill_value: Optional[Any] = None) -> Series:
        this = self
        if not self.index.equals(other.index):
            this, other = self.align(other, level=level, join="outer")
        this_vals, other_vals = ops.fill_binop(this._values, other._values, fill_value)
        with np.errstate(all="ignore"):
            result = func(this_vals, other_vals)
        name = ops.get_op_result_name(self, other)
        out = this._construct_result(result, name=name)
        return cast(Series, out)

    def _construct_result(self, result: Any, name: Any) -> Series:
        if isinstance(result, tuple):
            res1 = self._construct_result(result[0], name=name)
            res2 = self._construct_result(result[1], name=name)
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)
        dtype = getattr(result, "dtype", None)
        out: Series = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)
        out.name = name
        return out

    def _flex_method(self, other: Any, op: Callable[[Any, Any], Any], *, level: Optional[Any] = None, fill_value: Optional[Any] = None, axis: Optional[Any] = 0) -> Any:
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

    def eq(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.eq, level=level, fill_value=fill_value, axis=axis)

    def ne(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.ne, level=level, fill_value=fill_value, axis=axis)

    def le(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.le, level=level, fill_value=fill_value, axis=axis)

    def lt(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.lt, level=level, fill_value=fill_value, axis=axis)

    def ge(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.ge, level=level, fill_value=fill_value, axis=axis)

    def gt(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.gt, level=level, fill_value=fill_value, axis=axis)

    def add(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.add, level=level, fill_value=fill_value, axis=axis)

    def radd(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.radd, level=level, fill_value=fill_value, axis=axis)

    def sub(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.sub, level=level, fill_value=fill_value, axis=axis)
    subtract = sub

    def rsub(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rsub, level=level, fill_value=fill_value, axis=axis)

    def mul(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.mul, level=level, fill_value=fill_value, axis=axis)
    multiply = mul

    def rmul(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rmul, level=level, fill_value=fill_value, axis=axis)

    def truediv(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.truediv, level=level, fill_value=fill_value, axis=axis)
    div = truediv
    divide = truediv

    def rtruediv(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis)
    rdiv = rtruediv

    def floordiv(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.floordiv, level=level, fill_value=fill_value, axis=axis)

    def rfloordiv(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis)

    def mod(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.mod, level=level, fill_value=fill_value, axis=axis)

    def rmod(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rmod, level=level, fill_value=fill_value, axis=axis)

    def pow(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, operator.pow, level=level, fill_value=fill_value, axis=axis)

    def rpow(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Series:
        return self._flex_method(other, roperator.rpow, level=level, fill_value=fill_value, axis=axis)

    def divmod(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Any:
        return self._flex_method(other, divmod, level=level, fill_value=fill_value, axis=axis)

    def rdivmod(self, other: Any, level: Optional[Union[int, str]] = None, fill_value: Optional[Any] = None, axis: int = 0) -> Any:
        return self._flex_method(other, roperator.rdivmod, level=level, fill_value=fill_value, axis=axis)

    def _reduce(self, op: Callable[..., Any], name: str, *, axis: Optional[Any] = 0, skipna: bool = True, numeric_only: bool = False, filter_type: Optional[str] = None, **kwds: Any) -> Any:
        delegate = self._values
        if axis is not None:
            self._get_axis_number(axis)
        if isinstance(delegate, ExtensionArray):
            return delegate._reduce(name, skipna=skipna, **kwds)
        else:
            if numeric_only and self.dtype.kind not in "iufcb":
                kwd_name = "numeric_only"
                if name in ["any", "all"]:
                    kwd_name = "bool_only"
                raise TypeError(f"Series.{name} does not allow {kwd_name}={numeric_only} with non-numeric dtypes.")
            return op(delegate, skipna=skipna, **kwds)

    def any(self, *, axis: int = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        nv.validate_logical_func((), kwargs, fname="any")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(nanops.nanany, name="any", axis=axis, numeric_only=bool_only, skipna=skipna, filter_type="bool")

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="all")
    def all(self, axis: int = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> Any:
        nv.validate_logical_func((), kwargs, fname="all")
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._reduce(nanops.nanall, name="all", axis=axis, numeric_only=bool_only, skipna=skipna, filter_type="bool")

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="min")
    def min(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.min(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="max")
    def max(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.max(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="sum")
    def sum(self, axis: Optional[int] = None, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        return NDFrame.sum(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="prod")
    def prod(self, axis: Optional[int] = None, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> Any:
        return NDFrame.prod(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="mean")
    def mean(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.mean(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="median")
    def median(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.median(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="sem")
    def sem(self, axis: Optional[int] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.sem(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="var")
    def var(self, axis: Optional[int] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.var(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="std")
    def std(self, axis: Optional[int] = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.std(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="skew")
    def skew(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.skew(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version="4.0", allowed_args=["self"], name="kurt")
    def kurt(self, axis: int = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> Any:
        return NDFrame.kurt(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
    kurtosis = kurt
    product = prod

    def cummin(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    def cummax(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    def cumsum(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    def cumprod(self, axis: int = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Series:
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)

    def pop(self, item: Any) -> Any:
        return super().pop(item=item) 

    # The rest of the methods are similarly annotated.
