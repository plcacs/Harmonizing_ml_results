from typing import Any, Callable, Hashable, Iterable, List, Optional, Sequence, Tuple, Union, TypeVar, overload
from pandas._libs import lib, properties, reshape
from pandas._typing import AggFuncType, AnyAll, AnyArrayLike, ArrayLike, Axis, AxisInt, CorrelationMethod, DropKeep, Dtype, DtypeObj, FilePath, Frequency, IgnoreRaise, IndexKeyFunc, IndexLabel, Level, ListLike, MutableMappingT, NaPosition, NumpySorter, NumpyValueArrayLike, QuantileInterpolation, ReindexMethod, Renamer, Scalar, Self, SortKind, StorageOptions, Suffixes, ValueKeyFunc, WriteBuffer, npt
from pandas.core.dtypes.common import is_bool, is_float, is_integer, is_object_dtype
from pandas.core.dtypes.missing import isna
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex
from pandas.core.indexes.base import IndexOpsMixin
from pandas.core.internals import SingleBlockManager
from pandas.core.ops import (
    arithmetic_op,
    comparison_op,
    logical_op,
    make_flex_doc,
    ops,
    reduce_op,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.base import BaseOpsMixin
from pandas.core.generic import NDFrame
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex
from pandas.core.indexes.base import IndexOpsMixin
from pandas.core.internals import SingleBlockManager
from pandas.core.ops import (
    arithmetic_op,
    comparison_op,
    logical_op,
    make_flex_doc,
    ops,
    reduce_op,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.base import BaseOpsMixin
from pandas.core.frame import DataFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.internals import SingleBlockManager
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex
from pandas.core.indexes.base import IndexOpsMixin
from pandas.core.internals import SingleBlockManager
from pandas.core.ops import (
    arithmetic_op,
    comparison_op,
    logical_op,
    make_flex_doc,
    ops,
    reduce_op,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.base import BaseOpsMixin
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex
from pandas.core.indexes.base import IndexOpsMixin
from pandas.core.internals import SingleBlockManager
from pandas.core.ops import (
    arithmetic_op,
    comparison_op,
    logical_op,
    make_flex_doc,
    ops,
    reduce_op,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.base import BaseOpsMixin
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex
from pandas.core.indexes.base import IndexOpsMixin
from pandas.core.internals import SingleBlockManager
from pandas.core.ops import (
    arithmetic_op,
    comparison_op,
    logical_op,
    make_flex_doc,
    ops,
    reduce_op,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.base import BaseOpsMixin
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import SeriesGroupBy

T = TypeVar('T')
T_Dict = TypeVar('T_Dict', bound=Mapping)
T_Scalar = TypeVar('T_Scalar', bound=Scalar)
T_ArrayLike = TypeVar('T_ArrayLike', bound=ArrayLike)
T_IndexLike = TypeVar('T_IndexLike', bound=IndexLike)
T_Dtype = TypeVar('T_Dtype', bound=Dtype)

class Series(BaseOpsMixin, NDFrame):
    _typ: str = 'series'
    _HANDLED_TYPES: Tuple[Type[SingleBlockManager], Type[DatetimeIndex], Type[IntervalArray], Type[PeriodArray]] = (SingleBlockManager, DatetimeIndex, IntervalArray, PeriodArray)
    _metadata: Tuple[str] = ('_name',)
    _internal_names_set: FrozenSet[str] = frozenset(['index', 'name']) | NDFrame._internal_names_set
    _accessors: FrozenSet[str] = frozenset(['dt', 'cat', 'str', 'sparse'])
    _hidden_attrs: FrozenSet[str] = BaseOpsMixin._hidden_attrs | NDFrame._hidden_attrs | frozenset([])
    __pandas_priority__: int = 3000

    @overload
    def __init__(self, data: T_ArrayLike, index: Optional[T_IndexLike] = ..., dtype: Optional[T_Dtype] = ..., name: Hashable = ..., copy: bool = ...) -> 'Series[T_Scalar]':
        ...

    @overload
    def __init__(self, data: T_Dict, index: Optional[T_IndexLike] = ..., dtype: Optional[T_Dtype] = ..., name: Hashable = ..., copy: bool = ...) -> 'Series[T_Scalar]':
        ...

    def __init__(self, data: T_ArrayLike, index: Optional[T_IndexLike] = None, dtype: Optional[T_Dtype] = None, name: Hashable = ..., copy: bool = False) -> 'Series[T_Scalar]':
        if isinstance(data, SingleBlockManager) and index is None and (dtype is None) and (copy is False or copy is None):
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
                warnings.warn(f'Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.', DeprecationWarning, stacklevel=2)
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
            raise NotImplementedError('initializing a Series from a MultiIndex is not supported')
        refs = None
        if isinstance(data, Index):
            if dtype is not None:
                data = data.astype(dtype)
            refs = data._references
            copy = False
        elif isinstance(data, np.ndarray):
            if len(data.dtype):
                raise ValueError('Cannot construct a Series from an ndarray with compound dtype.  Use DataFrame instead.')
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
                raise AssertionError('Cannot pass both SingleBlockManager `data` argument and a different `index` argument. `copy` must be False.')
            if not allow_mgr:
                warnings.warn(f'Passing a {type(data).__name__} to {type(self).__name__} is deprecated and will raise in a future version. Use public APIs instead.', DeprecationWarning, stacklevel=2)
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

    @overload
    def _init_dict(self, data: T_Dict, index: Optional[T_IndexLike] = ..., dtype: Optional[T_Dtype] = ..., ) -> Tuple[SingleBlockManager, Index]:
        ...

    @overload
    def _init_dict(self, data: T_Dict, index: Optional[T_IndexLike] = ..., dtype: Optional[T_Dtype] = ..., ) -> Tuple[SingleBlockManager, Index]:
        ...

    def _init_dict(self, data: T_Dict, index: Optional[T_IndexLike] = None, dtype: Optional[T_Dtype] = None) -> Tuple[SingleBlockManager, Index]:
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

    def __arrow_c_stream__(self, requested_schema: Optional[PyCapsule] = ...) -> PyCapsule:
        pa = import_optional_dependency('pyarrow', min_version='16.0.0')
        type = pa.DataType._import_from_c_capsule(requested_schema) if requested_schema is not None else None
        ca = pa.array(self, type=type)
        if not isinstance(ca, pa.ChunkedArray):
            ca = pa.chunked_array([ca])
        return ca.__arrow_c_stream__()

    @property
    def _constructor(self) -> Type['Series[T_Scalar]']:
        return Series

    def _constructor_from_mgr(self, mgr: SingleBlockManager, axes: Tuple[Axis, ...]) -> 'Series[T_Scalar]':
        ser = Series._from_mgr(mgr, axes=axes)
        ser._name = None
        if type(self) is Series:
            return ser
        return self._constructor(ser)

    @property
    def _constructor_expanddim(self) -> Type['DataFrame[T_Scalar]']:
        from pandas.core.frame import DataFrame
        return DataFrame

    def _constructor_expanddim_from_mgr(self, mgr: SingleBlockManager, axes: Tuple[Axis, ...]) -> 'DataFrame[T_Scalar]':
        from pandas.core.frame import DataFrame
        df = DataFrame._from_mgr(mgr, axes=mgr.axes)
        if type(self) is Series:
            return df
        return self._constructor_expanddim(df)

    @property
    def _can_hold_na(self) -> bool:
        return self._mgr._can_hold_na

    @property
    def dtype(self) -> Dtype:
        return self._mgr.dtype

    @property
    def dtypes(self) -> Dtype:
        return self.dtype

    @property
    def name(self) -> Hashable:
        return self._name

    @name.setter
    def name(self, value: Hashable) -> None:
        validate_all_hashable(value, error_name=f'{type(self).__name__}.name')
        object.__setattr__(self, '_name', value)

    @property
    def values(self) -> npt.ArrayLike:
        return self._mgr.external_values()

    @property
    def _values(self) -> npt.ArrayLike:
        return self._mgr.internal_values()

    @property
    def _references(self) -> Tuple[int, ...]:
        return self._mgr._block.refs

    @Appender(base.IndexOpsMixin.array.__doc__)
    @property
    def array(self) -> npt.ArrayLike:
        return self._mgr.array_values()

    def __len__(self) -> int:
        return len(self._mgr)

    def __array__(self, dtype: Optional[T_Dtype] = ..., copy: Optional[bool] = ...) -> npt.ArrayLike:
        if dtype is None:
            dtype = self.dtype
        if copy is None:
            copy = False
        return self._mgr.array_values(dtype=dtype, copy=copy)

    def _ixs(self, i: int, axis: AxisInt = 0) -> T_Scalar:
        return self._values[i]

    def _slice(self, slobj: Union[slice, Index], axis: AxisInt = 0) -> 'Series[T_Scalar]':
        mgr = self._mgr.get_slice(slobj, axis=axis)
        out = self._constructor_from_mgr(mgr, axes=mgr.axes)
        out._name = self._name
        return out.__finalize__(self)

    def __getitem__(self, key: Union[int, Index, Tuple[Hashable, ...], List[Hashable], slice, Ellipsis]) -> 'Series[T_Scalar]':
        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            indexer = self.index._convert_slice_indexer(key, kind='getitem')
            return self._get_rows_with_mask(indexer)
        try:
            return self._get_with_engine(key)
        except KeyError:
            return self.loc[key]
        except (TypeError, ValueError, LossySetitemError):
            return self._get_with(key)
        except InvalidIndexError as err:
            if isinstance(key, Tuple[Hashable, ...]) and (not isinstance(self.index, MultiIndex)):
                raise KeyError('key of type tuple not found and not a MultiIndex') from err
            if com.is_bool_indexer(key):
                key = check_bool_indexer(self.index, key)
                key = np.asarray(key, dtype=bool)
                if is_list_like(key) and len(key) != len(self) and (not isinstance(key, Series)) and (not is_object_dtype(self.dtype)):
                    indexer = key.nonzero()[0]
                    return self._set_values(indexer, key)
                try:
                    self._where(~key, key, inplace=True)
                except InvalidIndexError:
                    return self.iloc[key]
                return self._get_rows_with_mask(key)
            else:
                return self._get_with(key)

    def _get_with(self, key: Union[int, Index, Tuple[Hashable, ...], List[Hashable], slice]) -> 'Series[T_Scalar]':
        if isinstance(key, ABCDataFrame):
            raise TypeError('Indexing a Series with DataFrame is not supported, use the appropriate DataFrame column')
        elif isinstance(key, Tuple[Hashable, ...]):
            return self._get_values_tuple(key)
        return self.loc[key]

    def _get_values_tuple(self, key: Tuple[Hashable, ...]) -> 'Series[T_Scalar]':
        if com.any_none(*key):
            result = np.asarray(self._values[key])
            disallow_ndim_indexing(result)
            return result
        if not isinstance(self.index, MultiIndex):
            raise KeyError('key of type tuple not found and not a MultiIndex')
        indexer, new_index = self.index.get_loc_level(key)
        new_ser = self._constructor(self._values[indexer], index=new_index, copy=False)
        if isinstance(indexer, slice):
            new_ser._mgr.add_references(self._mgr)
        return new_ser.__finalize__(self)

    def _get_rows_with_mask(self, indexer: npt.ArrayLike) -> 'Series[T_Scalar]':
        new_mgr = self._mgr.get_rows_with_mask(indexer)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self, method='get_rows_with_mask')

    def _get_value(self, label: Hashable, takeable: bool = False) -> T_Scalar:
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
            new_ser = self._constructor(new_values, index=new_index, name=self.name, copy=False)
            if isinstance(loc, slice):
                new_ser._mgr.add_references(self._mgr)
            return new_ser.__finalize__(self)
        else:
            return self.iloc[loc]

    def __setitem__(self, key: Union[int, Index, Tuple[Hashable, ...], List[Hashable], slice], value: T_Scalar) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= 3:
                warnings.warn(_chained_assignment_msg, ChainedAssignmentError, stacklevel=2)
        check_dict_or_set_indexers(key)
        key = com.apply_if_callable(key, self)
        if key is Ellipsis:
            key = slice(None)
        if isinstance(key, slice):
            indexer = self.index._convert_slice_indexer(key, kind='getitem')
            return self._set_values(indexer, value)
        try:
            self._set_with_engine(key, value)
        except KeyError:
            self.loc[key] = value
        except (TypeError, ValueError, LossySetitemError):
            indexer = self.index.get_loc(key)
            self._set_values(indexer, value)
        except InvalidIndexError as err:
            if isinstance(key, Tuple[Hashable, ...]) and (not isinstance(self.index, MultiIndex)):
                raise KeyError('key of type tuple not found and not a MultiIndex') from err
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

    def _set_with_engine(self, key: Union[int, Index, Tuple[Hashable, ...], List[Hashable], slice], value: T_Scalar) -> None:
        loc = self.index.get_loc(key)
        self._mgr.setitem_inplace(loc, value)

    def _set_with(self, key: Union[int, Index, Tuple[Hashable, ...], List[Hashable], slice], value: T_Scalar) -> None:
        assert not isinstance(key, Tuple[Hashable, ...])
        if is_iterator(key):
            key = list(key)
        self._set_labels(key, value)

    def _set_labels(self, key: Union[int, Index, Tuple[Hashable, ...], List[Hashable], slice], value: T_Scalar) -> None:
        key = com.asarray_tuplesafe(key)
        indexer = self.index.get_indexer(key)
        mask = indexer == -1
        if mask.any():
            raise KeyError(f'{key[mask]} not in index')
        self._set_values(indexer, value)

    def _set_values(self, key: npt.ArrayLike, value: T_Scalar) -> None:
        if isinstance(key, (Index, Series)):
            key = key._values
        self._mgr = self._mgr.setitem(indexer=key, value=value)

    def _set_value(self, label: Hashable, value: T_Scalar, takeable: bool = False) -> None:
        if not takeable:
            try:
                loc = self.index.get_loc(label)
            except KeyError:
                self.loc[label] = value
                return
        else:
            loc = label
        self._set_values(loc, value)

    def repeat(self, repeats: Union[int, npt.ArrayLike], axis: AxisInt = 0) -> 'Series[T_Scalar]':
        nv.validate_repeat((), {'axis': axis})
        new_index = self.index.repeat(repeats)
        new_values = self._values.repeat(repeats)
        return self._constructor(new_values, index=new_index, copy=False).__finalize__(self, method='repeat')

    def reset_index(self, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., drop: bool = ..., name: Optional[Hashable] = ..., inplace: bool = ..., allow_duplicates: bool = ...) -> 'Series[T_Scalar]':
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if drop:
            new_index = default_index(len(self))
            if level is not None:
                if not isinstance(level, (Tuple[List[Union[int, str]], ...], List[Union[int, str]])):
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
                return new_ser.__finalize__(self, method='reset_index')
        elif inplace:
            raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')
        else:
            if name is lib.no_default:
                if self.name is None:
                    name = 0
                else:
                    name = self.name
            df = self.to_frame(name)
            return df.reset_index(level=level, drop=drop, allow_duplicates=allow_duplicates)
        return None

    def __repr__(self) -> str:
        return self.to_string()

    @overload
    def to_string(self, buf: Optional[Union[str, Path, StringIO]] = ..., *, na_rep: str = ..., float_format: Optional[str] = ..., header: bool = ..., index: bool = ..., length: bool = ..., dtype: bool = ..., name: bool = ..., max_rows: Optional[int] = ..., min_rows: Optional[int] = ...) -> str:
        ...

    @overload
    def to_string(self, buf: Union[str, Path, StringIO], *, na_rep: str = ..., float_format: Optional[str] = ..., header: bool = ..., index: bool = ..., length: bool = ..., dtype: bool = ..., name: bool = ..., max_rows: Optional[int] = ..., min_rows: Optional[int] = ...) -> None:
        ...

    def to_string(self, buf: Optional[Union[str, Path, StringIO]] = ..., na_rep: str = 'NaN', float_format: Optional[str] = ..., header: bool = True, index: bool = True, length: bool = False, dtype: bool = False, name: bool = False, max_rows: Optional[int] = ..., min_rows: Optional[int] = ...) -> str:
        formatter = fmt.SeriesFormatter(self, name=name, length=length, header=header, index=index, dtype=dtype, na_rep=na_rep, float_format=float_format, min_rows=min_rows, max_rows=max_rows)
        result = formatter.to_string()
        if not isinstance(result, str):
            raise AssertionError(f'result must be of type str, type of result is {type(result).__name__!r}')
        if buf is None:
            return result
        elif hasattr(buf, 'write'):
            buf.write(result)
        else:
            with open(buf, 'w', encoding='utf-8') as f:
                f.write(result)
        return None

    @overload
    def to_markdown(self, buf: Optional[Union[str, Path, StringIO]] = ..., *, mode: str = ..., index: bool = ..., storage_options: Optional[StorageOptions] = ..., **kwargs: Any) -> str:
        ...

    @overload
    def to_markdown(self, buf: Union[str, Path, StringIO], *, mode: str = ..., index: bool = ..., storage_options: Optional[StorageOptions] = ..., **kwargs: Any) -> None:
        ...

    def to_markdown(self, buf: Optional[Union[str, Path, StringIO]] = ..., mode: str = 'wt', index: bool = True, storage_options: Optional[StorageOptions] = ...) -> str:
        return self.to_frame().to_markdown(buf, mode=mode, index=index, storage_options=storage_options, **kwargs)

    def items(self) -> Iterable[Tuple[Hashable, T_Scalar]]:
        return zip(iter(self.index), iter(self))

    def keys(self) -> Index:
        return self.index

    @overload
    def to_dict(self, *, into: Type[T_Dict] = ...) -> T_Dict:
        ...

    @overload
    def to_dict(self, *, into: Type[T_Dict]) -> T_Dict:
        ...

    def to_dict(self, *, into: Type[T_Dict] = dict) -> T_Dict:
        into_c = com.standardize_mapping(into)
        if is_object_dtype(self.dtype) or isinstance(self.dtype, ExtensionDtype):
            return into_c(((k, maybe_box_native(v)) for k, v in self.items()))
        else:
            return into_c(self.items())

    def to_frame(self, name: Optional[Hashable] = ...) -> 'DataFrame[T_Scalar]':
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
        return df.__finalize__(self, method='to_frame')

    def _set_name(self, name: Hashable, inplace: bool = False, deep: Optional[bool] = ...) -> 'Series[T_Scalar]':
        inplace = validate_bool_kwarg(inplace, 'inplace')
        ser = self if inplace else self.copy(deep=False)
        ser.name = name
        return ser

    def groupby(self, by: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., level: Optional[Union[int, str, List[Union[int, str]]]] = ..., as_index: bool = ..., sort: bool = ..., group_keys: bool = ..., observed: bool = ..., dropna: bool = ...) -> SeriesGroupBy:
        from pandas.core.groupby.generic import SeriesGroupBy
        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        if not as_index:
            raise TypeError('as_index=False only valid with DataFrame')
        return SeriesGroupBy(obj=self, keys=by, level=level, as_index=as_index, sort=sort, group_keys=group_keys, observed=observed, dropna=dropna)

    def count(self) -> int:
        return notna(self._values).sum().astype('int64')

    def mode(self, dropna: bool = True) -> 'Series[T_Scalar]':
        values = self._values
        if isinstance(values, np.ndarray):
            res_values = algorithms.mode(values, dropna=dropna)
        else:
            res_values = values._mode(dropna=dropna)
        return self._constructor(res_values, index=range(len(res_values)), name=self.name, copy=False, dtype=self.dtype).__finalize__(self, method='mode')

    def unique(self) -> npt.ArrayLike:
        return super().unique()

    def drop_duplicates(self, keep: str = 'first', inplace: bool = False, ignore_index: bool = ...) -> 'Series[T_Scalar]':
        inplace = validate_bool_kwarg(inplace, 'inplace')
        result = super().drop_duplicates(keep=keep)
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def duplicated(self, keep: str = 'first') -> 'Series[bool]':
        res = self._duplicated(keep=keep)
        result = self._constructor(res, index=self.index, copy=False)
        return result.__finalize__(self, method='duplicated')

    def idxmin(self, axis: AxisInt = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Index:
        axis = self._get_axis_number(axis)
        iloc = self.argmin(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def idxmax(self, axis: AxisInt = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> Index:
        axis = self._get_axis_number(axis)
        iloc = self.argmax(axis, skipna, *args, **kwargs)
        return self.index[iloc]

    def round(self, decimals: int = 0, *args: Any, **kwargs: Any) -> 'Series[T_Scalar]':
        nv.validate_round(args, kwargs)
        new_mgr = self._mgr.round(decimals=decimals)
        return self._constructor_from_mgr(new_mgr, axes=new_mgr.axes).__finalize__(self, method='round')

    def quantile(self, q: Union[float, npt.ArrayLike] = 0.5, interpolation: str = 'linear') -> Union[float, 'Series[T_Scalar]']:
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

    def corr(self, other: Series, method: str = 'pearson', min_periods: Optional[int] = ...) -> float:
        this, other = self.align(other, join='inner')
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)
        if method in ['pearson', 'spearman', 'kendall'] or callable(method):
            return nanops.nancorr(this_values, other_values, method=method, min_periods=min_periods)
        raise ValueError(f"method must be either 'pearson', 'spearman', 'kendall', or a callable, '{method}' was supplied")

    def cov(self, other: Series, min_periods: Optional[int] = ..., ddof: int = 1) -> float:
        this, other = self.align(other, join='inner')
        if len(this) == 0:
            return np.nan
        this_values = this.to_numpy(dtype=float, na_value=np.nan, copy=False)
        other_values = other.to_numpy(dtype=float, na_value=np.nan, copy=False)
        return nanops.nancov(this_values, other_values, min_periods=min_periods, ddof=ddof)

    def diff(self, periods: int = 1) -> 'Series[T_Scalar]':
        if not lib.is_integer(periods):
            if not (is_float(periods) and periods.is_integer()):
                raise ValueError('periods must be an integer')
        result = algorithms.diff(self._values, periods)
        return self._constructor(result, index=self.index, copy=False).__finalize__(self, method='diff')

    def autocorr(self, lag: int = 1) -> float:
        return self.corr(cast(Series, self.shift(lag)))

    def dot(self, other: Series) -> float:
        if isinstance(other, Series):
            common = self.index.union(other.index)
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError('matrices are not aligned')
            left = self.reindex(index=common)
            right = other.reindex(index=common)
            lvals = left.values
            rvals = right.values
        else:
            lvals = self.values
            rvals = np.asarray(other)
            if lvals.shape[0] != rvals.shape[0]:
                raise Exception(f'Dot product shape mismatch, {lvals.shape} vs {rvals.shape}')
        if isinstance(other, Series):
            return self._constructor(np.dot(lvals, rvals), index=other.columns, copy=False).__finalize__(self, method='dot')
        elif isinstance(rvals, np.ndarray):
            return np.dot(lvals, rvals)
        else:
            raise TypeError(f'unsupported type: {type(other)}')

    def __matmul__(self, other: Series) -> float:
        return self.dot(other)

    def __rmatmul__(self, other: Series) -> float:
        return self.dot(np.transpose(other))

    def searchsorted(self, value: Union[Hashable, npt.ArrayLike], side: str = 'left', sorter: Optional[npt.ArrayLike] = ...) -> npt.ArrayLike:
        return base.IndexOpsMixin.searchsorted(self, value, side=side, sorter=sorter)

    def _append(self, to_append: Union[Series, List[Series]], ignore_index: bool = ..., verify_integrity: bool = ...) -> 'Series[T_Scalar]':
        from pandas.core.reshape.concat import concat
        if isinstance(to_append, (List, Tuple)):
            to_concat = [self]
            to_concat.extend(to_append)
        else:
            to_concat = [self, to_append]
        if any((isinstance(x, DataFrame) for x in to_concat[1:])):
            msg = 'to_append should be a Series or list/tuple of Series, got DataFrame'
            raise TypeError(msg)
        return concat(to_concat, ignore_index=ignore_index, verify_integrity=verify_integrity)

    def compare(self, other: Series, align_axis: int = 1, keep_shape: bool = ..., keep_equal: bool = ..., result_names: Tuple[str, str] = ...) -> DataFrame:
        return super().compare(other=other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal, result_names=result_names)

    def combine(self, other: Union[Series, T_Scalar], func: Callable[[T_Scalar, T_Scalar], T_Scalar], fill_value: Optional[T_Scalar] = ...) -> 'Series[T_Scalar]':
        if fill_value is None:
            fill_value = na_value_for_dtype(self.dtype, compat=False)
        if isinstance(other, Series):
            new_index = self.index.union(other.index)
            new_name = ops.get_op_result_name(self, other)
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all='ignore'):
                for i, idx in enumerate(new_index):
                    lv = self.get(idx, fill_value)
                    rv = other.get(idx, fill_value)
                    new_values[i] = func(lv, rv)
        else:
            new_index = self.index
            new_values = np.empty(len(new_index), dtype=object)
            with np.errstate(all='ignore'):
                new_values[:] = [func(lv, other) for lv in self._values]
            new_name = self.name
        npvalues = lib.maybe_convert_objects(new_values, try_float=False)
        same_dtype = isinstance(self.dtype, (StringDtype, CategoricalDtype))
        res_values = maybe_cast_pointwise_result(npvalues, self.dtype, same_dtype=same_dtype)
        return self._constructor(res_values, index=new_index, name=new_name, copy=False)

    def combine_first(self, other: Series) -> 'Series[T_Scalar]':
        from pandas.core.reshape.concat import concat
        if self.dtype == other.dtype:
            if self.index.equals(other.index):
                return self.mask(self.isna(), other)
            elif self._can_hold_na and (not isinstance(self.dtype, SparseDtype)):
                this, other = self.align(other, join='outer')
                return this.mask(this.isna(), other)
        new_index = self.index.union(other.index)
        this = self
        keep_other = other.index.difference(this.index[notna(this)])
        keep_this = this.index.difference(keep_other)
        this = this.reindex(keep_this)
        other = other.reindex(keep_other)
        if this.dtype.kind == 'M' and other.dtype.kind != 'M':
            other = to_datetime(other)
        combined = concat([this, other])
        combined = combined.reindex(new_index)
        return combined.__finalize__(self, method='combine_first')

    def update(self, other: Union[Series, T_Dict, List[T_Scalar]]) -> None:
        if not PYPY:
            if sys.getrefcount(self) <= REF_COUNT:
                warnings.warn(_chained_assignment_method_msg, ChainedAssignmentError, stacklevel=2)
        if not isinstance(other, Series):
            other = Series(other)
        other = other.reindex_like(self)
        mask = notna(other)
        self._mgr = self._mgr.putmask(mask=mask, new=other)

    def sort_values(self, axis: AxisInt = 0, ascending: bool = ..., inplace: bool = ..., kind: str = ..., na_position: str = ..., ignore_index: bool = ..., key: Optional[Callable[[T_Scalar], T_Scalar]] = ...) -> 'Series[T_Scalar]':
        inplace = validate_bool_kwarg(inplace, 'inplace')
        self._get_axis_number(axis)
        if is_list_like(ascending):
            ascending = cast(Sequence[bool], ascending)
            if len(ascending) != 1:
                raise ValueError(f'Length of ascending ({len(ascending)}) must be 1 for Series')
            ascending = ascending[0]
        ascending = validate_ascending(ascending)
        if na_position not in ['first', 'last']:
            raise ValueError(f'invalid na_position: {na_position}')
        if key:
            values_to_sort = cast(Series, ensure_key_mapped(self, key))._values
        else:
            values_to_sort = self._values
        sorted_index = nargsort(values_to_sort, kind, bool(ascending), na_position)
        if is_range_indexer(sorted_index, len(sorted_index)):
            if inplace:
                return self._update_inplace(self)
            return self.copy(deep=False)
        result = self._constructor(self._values[sorted_index], index=self.index[sorted_index], copy=False)
        if ignore_index:
            result.index = default_index(len(sorted_index))
        if not inplace:
            return result.__finalize__(self, method='sort_values')
        self._update_inplace(result)
        return None

    def sort_index(self, axis: AxisInt = 0, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., ascending: bool = ..., inplace: bool = ..., kind: str = ..., na_position: str = ..., sort_remaining: bool = ..., ignore_index: bool = ..., key: Optional[Callable[[Index], Index]] = ...) -> 'Series[T_Scalar]':
        return super().sort_index(axis=axis, level=level, ascending=ascending, inplace=inplace, kind=kind, na_position=na_position, sort_remaining=sort_remaining, ignore_index=ignore_index, key=key)

    def argsort(self, axis: AxisInt = 0, kind: str = 'quicksort', order: Optional[Tuple[str, ...]] = ..., stable: Optional[bool] = ...) -> npt.ArrayLike:
        if axis != -1:
            self._get_axis_number(axis)
        result = self.array.argsort(kind=kind)
        res = self._constructor(result, index=self.index, name=self.name, dtype=np.intp, copy=False)
        return res.__finalize__(self, method='argsort')

    def nlargest(self, n: int = 5, keep: str = 'first') -> 'Series[T_Scalar]':
        return selectn.SelectNSeries(self, n=n, keep=keep).nlargest()

    def nsmallest(self, n: int = 5, keep: str = 'first') -> 'Series[T_Scalar]':
        return selectn.SelectNSeries(self, n=n, keep=keep).nsmallest()

    def swaplevel(self, i: Union[int, str, List[Union[int, str]]] = -2, j: Union[int, str, List[Union[int, str]]] = -1, copy: bool = lib.no_default) -> 'Series[T_Scalar]':
        self._check_copy_deprecation(copy)
        assert isinstance(self.index, MultiIndex)
        result = self.copy(deep=False)
        result.index = self.index.swaplevel(i, j)
        return result

    def reorder_levels(self, order: List[Union[int, str]]) -> 'Series[T_Scalar]':
        if not isinstance(self.index, MultiIndex):
            raise Exception('Can only reorder levels on a hierarchical axis.')
        result = self.copy(deep=False)
        assert isinstance(result.index, MultiIndex)
        result.index = result.index.reorder_levels(order)
        return result

    def explode(self, ignore_index: bool = False) -> 'Series[T_Scalar]':
        if isinstance(self.dtype, ExtensionDtype):
            values, counts = self._values._explode()
        elif len(self) and is_object_dtype(self.dtype):
            values, counts = reshape.explode(np.asarray(self._values))
        else:
            result = self.copy()
            return result.reset_index(drop=True) if ignore_index else result
        if ignore_index:
            index = default_index(len(values))
        else:
            index = self.index.repeat(counts)
        return self._constructor(values, index=index, name=self.name, copy=False)

    def unstack(self, level: Union[int, str, List[Union[int, str]]] = -1, fill_value: Optional[T_Scalar] = ..., sort: bool = True) -> DataFrame:
        from pandas.core.reshape.reshape import unstack
        return unstack(self, level, fill_value, sort)

    def map(self, arg: Union[Callable[[T_Scalar], T_Scalar], T_Dict, Series], na_action: Optional[str] = ..., **kwargs: Any) -> 'Series[T_Scalar]':
        if callable(arg):
            arg = functools.partial(arg, **kwargs)
        new_values = self._map_values(arg, na_action=na_action)
        return self._constructor(new_values, index=self.index, copy=False).__finalize__(self, method='map')

    def _gotitem(self, key: Union[str, List[str]], ndim: int, subset: Optional[Any] = ...) -> 'Series[T_Scalar]':
        return self

    def _reindex_indexer(self, new_index: Index, indexer: Optional[npt.ArrayLike] = ...) -> 'Series[T_Scalar]':
        if indexer is None and (new_index is None or new_index.names == self.index.names):
            return self.copy(deep=False)
        new_values = algorithms.take_nd(self._values, indexer, allow_fill=True, fill_value=None)
        return self._constructor(new_values, index=new_index, copy=False)

    def _needs_reindex_multi(self, axes: Tuple[Axis, ...], method: str, level: Union[int, str]) -> bool:
        return False

    @overload
    def rename(self, index: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., *, axis: AxisInt = ..., copy: bool = ..., inplace: bool = ..., level: Optional[Union[int, str, List[Union[int, str]]]] = ..., errors: str = ...) -> 'Series[T_Scalar]':
        ...

    @overload
    def rename(self, index: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., *, axis: AxisInt = ..., copy: bool = ..., inplace: bool = ..., level: Optional[Union[int, str, List[Union[int, str]]]] = ..., errors: str = ...) -> 'Series[T_Scalar]':
        ...

    @overload
    def rename(self, index: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., *, axis: AxisInt = ..., copy: bool = ..., inplace: bool = ..., level: Optional[Union[int, str, List[Union[int, str]]]] = ..., errors: str = ...) -> 'Series[T_Scalar]':
        ...

    def rename(self, index: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = None, *, axis: AxisInt = 0, copy: bool = lib.no_default, inplace: bool = False, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., errors: str = 'ignore') -> 'Series[T_Scalar]':
        self._check_copy_deprecation(copy)
        if axis is not None:
            self._get_axis_number(axis)
        if callable(index) or is_dict_like(index):
            return super()._rename(index, inplace=inplace, level=level, errors=errors)
        else:
            return self._set_name(index, inplace=inplace)

    def set_axis(self, labels: Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice], *, axis: AxisInt = 0, copy: bool = lib.no_default) -> 'Series[T_Scalar]':
        return super().set_axis(labels, axis=axis, copy=copy)

    def reindex(self, index: Optional[Index] = ..., *, method: str = ..., copy: bool = lib.no_default, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., limit: Optional[int] = ..., tolerance: Optional[float] = ..., **kwargs: Any) -> 'Series[T_Scalar]':
        return super().reindex(index=index, method=method, level=level, fill_value=fill_value, limit=limit, tolerance=tolerance, copy=copy)

    def rename_axis(self, mapper: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., *, index: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., axis: AxisInt = 0, copy: bool = lib.no_default, inplace: bool = False) -> 'Series[T_Scalar]':
        return super().rename_axis(mapper=mapper, index=index, axis=axis, inplace=inplace, copy=copy)

    def drop(self, labels: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., *, axis: AxisInt = 0, index: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., columns: Optional[Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]] = ..., level: Optional[Union[int, str, List[Union[int, str]]]] = ..., inplace: bool = False, errors: str = 'raise') -> 'Series[T_Scalar]':
        return super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    def pop(self, item: Hashable) -> T_Scalar:
        return super().pop(item=item)

    def info(self, verbose: bool = ..., buf: Optional[Union[str, Path, StringIO]] = ..., max_cols: Optional[int] = ..., memory_usage: bool = ..., show_counts: bool = ...) -> None:
        return SeriesInfo(self, memory_usage).render(buf=buf, max_cols=max_cols, verbose=verbose, show_counts=show_counts)

    def memory_usage(self, index: bool = True, deep: bool = ...) -> int:
        v = self._memory_usage(deep=deep)
        if index:
            v += self.index.memory_usage(deep=deep)
        return v

    def isin(self, values: Union[Hashable, Index, Tuple[Hashable, ...], List[Hashable], slice]) -> 'Series[bool]':
        result = algorithms.isin(self._values, values)
        return self._constructor(result, index=self.index, copy=False).__finalize__(self, method='isin')

    def between(self, left: Union[Hashable, npt.ArrayLike], right: Union[Hashable, npt.ArrayLike], inclusive: str = 'both') -> 'Series[bool]':
        if inclusive == 'both':
            lmask = self >= left
            rmask = self <= right
        elif inclusive == 'left':
            lmask = self >= left
            rmask = self < right
        elif inclusive == 'right':
            lmask = self > left
            rmask = self <= right
        elif inclusive == 'neither':
            lmask = self > left
            rmask = self < right
        else:
            raise ValueError("Inclusive has to be either string of 'both','left', 'right', or 'neither'.")
        return lmask & rmask

    def case_when(self, caselist: List[Tuple[Callable[[T_Scalar], bool], Callable[[T_Scalar], T_Scalar]]]) -> 'Series[T_Scalar]':
        if not isinstance(caselist, List):
            raise TypeError(f'The caselist argument should be a list; instead got {type(caselist)}')
        if not caselist:
            raise ValueError('provide at least one boolean condition, with a corresponding replacement.')
        for num, entry in enumerate(caselist):
            if not isinstance(entry, Tuple):
                raise TypeError(f'Argument {num} must be a tuple; instead got {type(entry)}.')
            if len(entry) != 2:
                raise ValueError(f'Argument {num} must have length 2; a condition and replacement; instead got length {len(entry)}.')
        caselist = [(com.apply_if_callable(condition, self), com.apply_if_callable(replacement, self)) for condition, replacement in caselist]
        default = self.copy(deep=False)
        conditions, replacements = zip(*caselist)
        common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
        if len(set(common_dtypes)) > 1:
            common_dtype = find_common_type(common_dtypes)
            updated_replacements = []
            for condition, replacement in zip(conditions, replacements):
                if is_scalar(replacement):
                    replacement = construct_1d_arraylike_from_scalar(value=replacement, length=len(condition), dtype=common_dtype)
                elif isinstance(replacement, Series):
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
                raise ValueError(f'Failed to apply condition{position} and replacement{position}.') from error
        return default

    def isna(self) -> 'Series[bool]':
        return NDFrame.isna(self)

    def isnull(self) -> 'Series[bool]':
        """
        Series.isnull is an alias for Series.isna.
        """
        return super().isnull()

    def notna(self) -> 'Series[bool]':
        return NDFrame.notna(self)

    def notnull(self) -> 'Series[bool]':
        """
        Series.notnull is an alias for Series.notna.
        """
        return super().notnull()

    def dropna(self, axis: AxisInt = 0, inplace: bool = False, how: str = ..., ignore_index: bool = ...) -> 'Series[T_Scalar]':
        inplace = validate_bool_kwarg(inplace, 'inplace')
        ignore_index = validate_bool_kwarg(ignore_index, 'ignore_index')
        self._get_axis_number(axis or 0)
        if self._can_hold_na:
            result = remove_na_arraylike(self)
        elif not inplace:
            result = self.copy(deep=False)
        else:
            result = self
        if ignore_index:
            result.index = default_index(len(result))
        if inplace:
            self._update_inplace(result)
            return None
        else:
            return result

    def to_timestamp(self, freq: str = ..., how: str = ..., copy: bool = lib.no_default) -> 'Series[T_Scalar]':
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, PeriodIndex):
            raise TypeError(f'unsupported Type {type(self.index).__name__}')
        new_obj = self.copy(deep=False)
        new_index = self.index.to_timestamp(freq=freq, how=how)
        setattr(new_obj, 'index', new_index)
        return new_obj

    def to_period(self, freq: str = ..., copy: bool = lib.no_default) -> 'Series[T_Scalar]':
        self._check_copy_deprecation(copy)
        if not isinstance(self.index, DatetimeIndex):
            raise TypeError(f'unsupported Type {type(self.index).__name__}')
        new_obj = self.copy(deep=False)
        new_index = self.index.to_period(freq=freq)
        setattr(new_obj, 'index', new_index)
        return new_obj

    @property
    def index(self) -> Index:
        self._check_axis_number(0)
        return IndexOpsMixin._get_axis(self, 0)

    str = Accessor('str', StringMethods)
    dt = Accessor('dt', CombinedDatetimelikeProperties)
    cat = Accessor('cat', CategoricalAccessor)
    plot = Accessor('plot', pandas.plotting.PlotAccessor)
    sparse = Accessor('sparse', SparseAccessor)
    struct = Accessor('struct', StructAccessor)
    list = Accessor('list', ListAccessor)
    hist = pandas.plotting.hist_series

    def _cmp_method(self, other: Series, op: Callable[[T_Scalar, T_Scalar], bool]) -> 'Series[bool]':
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series) and (not self._indexed_same(other)):
            raise ValueError('Can only compare identically-labeled Series objects')
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.comparison_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _logical_method(self, other: Series, op: Callable[[T_Scalar, T_Scalar], bool]) -> 'Series[bool]':
        res_name = ops.get_op_result_name(self, other)
        self, other = self._align_for_op(other, align_asobject=True)
        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        res_values = ops.logical_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    def _arith_method(self, other: Series, op: Callable[[T_Scalar, T_Scalar], T_Scalar]) -> 'Series[T_Scalar]':
        self, other = self._align_for_op(other)
        return base.IndexOpsMixin._arith_method(self, other, op)

    def _align_for_op(self, right: Series, align_asobject: bool = False) -> Tuple['Series[T_Scalar]', Series]:
        """align lhs and rhs Series"""
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

    def _binop(self, other: Series, func: Callable[[T_Scalar, T_Scalar], T_Scalar], level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        this = self
        if not self.index.equals(other.index):
            this, other = self.align(other, level=level, join='outer')
        this_vals, other_vals = ops.fill_binop(this._values, other._values, fill_value)
        with np.errstate(all='ignore'):
            result = func(this_vals, other_vals)
        name = ops.get_op_result_name(self, other)
        out = this._construct_result(result, name)
        return cast(Series, out)

    def _construct_result(self, result: npt.ArrayLike, name: Hashable) -> 'Series[T_Scalar]':
        if isinstance(result, tuple):
            res1 = self._construct_result(result[0], name=name)
            res2 = self._construct_result(result[1], name=name)
            assert isinstance(res1, Series)
            assert isinstance(res2, Series)
            return (res1, res2)
        dtype = getattr(result, 'dtype', None)
        out = self._constructor(result, index=self.index, dtype=dtype, copy=False)
        out = out.__finalize__(self)
        out.name = name
        return out

    def _flex_method(self, other: Series, op: Callable[[T_Scalar, T_Scalar], T_Scalar], *, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        if axis is not None:
            self._get_axis_number(axis)
        res_name = ops.get_op_result_name(self, other)
        if isinstance(other, Series):
            return self._binop(other, op, level=level, fill_value=fill_value)
        elif isinstance(other, (np.ndarray, list, tuple)):
            if len(other) != len(self):
                raise ValueError('Lengths must be equal')
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

    def eq(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[bool]':
        return self._flex_method(other, operator.eq, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('ne', 'series'))
    def ne(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[bool]':
        return self._flex_method(other, operator.ne, level=level, fill_value=fill_value, axis=axis)

    def le(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[bool]':
        return self._flex_method(other, operator.le, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('lt', 'series'))
    def lt(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[bool]':
        return self._flex_method(other, operator.lt, level=level, fill_value=fill_value, axis=axis)

    def ge(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[bool]':
        return self._flex_method(other, operator.ge, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('gt', 'series'))
    def gt(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[bool]':
        return self._flex_method(other, operator.gt, level=level, fill_value=fill_value, axis=axis)

    def add(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, operator.add, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('radd', 'series'))
    def radd(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, roperator.radd, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('sub', 'series'))
    def sub(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, operator.sub, level=level, fill_value=fill_value, axis=axis)
    subtract = sub

    @Appender(ops.make_flex_doc('rsub', 'series'))
    def rsub(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, roperator.rsub, level=level, fill_value=fill_value, axis=axis)

    def mul(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, operator.mul, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rmul', 'series'))
    def rmul(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, roperator.rmul, level=level, fill_value=fill_value, axis=axis)

    def truediv(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, operator.truediv, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rtruediv', 'series'))
    def rtruediv(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, roperator.rtruediv, level=level, fill_value=fill_value, axis=axis)
    rdiv = rtruediv

    def floordiv(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, operator.floordiv, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rfloordiv', 'series'))
    def rfloordiv(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, roperator.rfloordiv, level=level, fill_value=fill_value, axis=axis)

    def mod(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, operator.mod, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rmod', 'series'))
    def rmod(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, roperator.rmod, level=level, fill_value=fill_value, axis=axis)

    def pow(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, operator.pow, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rpow', 'series'))
    def rpow(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> 'Series[T_Scalar]':
        return self._flex_method(other, roperator.rpow, level=level, fill_value=fill_value, axis=axis)

    def divmod(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> Tuple['Series[T_Scalar]', 'Series[T_Scalar]']:
        return self._flex_method(other, divmod, level=level, fill_value=fill_value, axis=axis)

    @Appender(ops.make_flex_doc('rdivmod', 'series'))
    def rdivmod(self, other: Series, level: Optional[Union[int, str, List[Union[int, str]]]] = ..., fill_value: Optional[T_Scalar] = ..., axis: AxisInt = 0) -> Tuple['Series[T_Scalar]', 'Series[T_Scalar]']:
        return self._flex_method(other, roperator.rdivmod, level=level, fill_value=fill_value, axis=axis)

    def _reduce(self, op: Callable[[npt.ArrayLike, bool], npt.ArrayLike], name: str, *, axis: AxisInt = 0, skipna: bool = True, numeric_only: bool = False, filter_type: Optional[str] = ..., **kwargs: Any) -> npt.ArrayLike:
        delegate = self._values
        if axis is not None:
            self._get_axis_number(axis)
        if isinstance(delegate, ExtensionArray):
            return delegate._reduce(name, skipna=skipna, **kwargs)
        else:
            if numeric_only and self.dtype.kind not in 'iufcb':
                kwd_name = 'numeric_only'
                if name in ['any', 'all']:
                    kwd_name = 'bool_only'
                raise TypeError(f'Series.{name} does not allow {kwd_name}={numeric_only} with non-numeric dtypes.')
            return op(delegate, skipna=skipna, **kwargs)

    @Appender(make_doc('any', ndim=1))
    def any(self, *, axis: AxisInt = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> bool:
        nv.validate_logical_func((), kwargs, fname='any')
        validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(nanops.nanany, name='any', axis=axis, numeric_only=bool_only, skipna=skipna, filter_type='bool')

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='all')
    @Appender(make_doc('all', ndim=1))
    def all(self, axis: AxisInt = 0, bool_only: bool = False, skipna: bool = True, **kwargs: Any) -> bool:
        nv.validate_logical_func((), kwargs, fname='all')
        validate_bool_kwarg(skipna, 'skipna', none_allowed=False)
        return self._reduce(nanops.nanall, name='all', axis=axis, numeric_only=bool_only, skipna=skipna, filter_type='bool')

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='min')
    def min(self, axis: AxisInt = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.min(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='max')
    def max(self, axis: AxisInt = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.max(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sum')
    def sum(self, axis: AxisInt = None, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> T_Scalar:
        return NDFrame.sum(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='prod')
    @doc(make_doc('prod', ndim=1))
    def prod(self, axis: AxisInt = None, skipna: bool = True, numeric_only: bool = False, min_count: int = 0, **kwargs: Any) -> T_Scalar:
        return NDFrame.prod(self, axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='mean')
    def mean(self, axis: AxisInt = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.mean(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='median')
    def median(self, axis: AxisInt = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.median(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='sem')
    @doc(make_doc('sem', ndim=1))
    def sem(self, axis: AxisInt = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.sem(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='var')
    def var(self, axis: AxisInt = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.var(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='std')
    @doc(make_doc('std', ndim=1))
    def std(self, axis: AxisInt = None, skipna: bool = True, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.std(self, axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='skew')
    @doc(make_doc('skew', ndim=1))
    def skew(self, axis: AxisInt = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.skew(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    @deprecate_nonkeyword_arguments(version='4.0', allowed_args=['self'], name='kurt')
    def kurt(self, axis: AxisInt = 0, skipna: bool = True, numeric_only: bool = False, **kwargs: Any) -> T_Scalar:
        return NDFrame.kurt(self, axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)
    kurtosis = kurt
    product = prod

    @doc(make_doc('cummin', ndim=1))
    def cummin(self, axis: AxisInt = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> 'Series[T_Scalar]':
        return NDFrame.cummin(self, axis, skipna, *args, **kwargs)

    @doc(make_doc('cummax', ndim=1))
    def cummax(self, axis: AxisInt = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> 'Series[T_Scalar]':
        return NDFrame.cummax(self, axis, skipna, *args, **kwargs)

    @doc(make_doc('cumsum', ndim=1))
    def cumsum(self, axis: AxisInt = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> 'Series[T_Scalar]':
        return NDFrame.cumsum(self, axis, skipna, *args, **kwargs)

    @doc(make_doc('cumprod', ndim=1))
    def cumprod(self, axis: AxisInt = 0, skipna: bool = True, *args: Any, **kwargs: Any) -> 'Series[T_Scalar]':
        return NDFrame.cumprod(self, axis, skipna, *args, **kwargs)
