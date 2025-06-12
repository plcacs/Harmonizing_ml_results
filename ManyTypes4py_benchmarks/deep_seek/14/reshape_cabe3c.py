from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, cast, overload, Any, Optional, Union, List, Tuple, Dict, Set
import warnings
import numpy as np
from pandas._config.config import get_option
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import find_common_type, maybe_promote
from pandas.core.dtypes.common import ensure_platform_int, is_1d_only_ea_dtype, is_integer, needs_i8_conversion
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import factorize, unique
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import Index, MultiIndex, default_index
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import compress_group_index, decons_obs_group_ids, get_compressed_ids, get_group_index, get_group_index_sorter

if TYPE_CHECKING:
    from pandas._typing import ArrayLike, Level, npt
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.frozen import FrozenList

class _Unstacker:
    def __init__(
        self,
        index: MultiIndex,
        level: Union[int, str],
        constructor: Any,
        sort: bool = True
    ) -> None:
        self.constructor = constructor
        self.sort = sort
        self.index = index.remove_unused_levels()
        self.level = self.index._get_level_number(level)
        self.lift = 1 if -1 in self.index.codes[self.level] else 0
        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)
        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]
        if not self.sort:
            unique_codes = unique(self.index.codes[self.level])
            self.removed_level = self.removed_level.take(unique_codes)
            self.removed_level_full = self.removed_level_full.take(unique_codes)
        if get_option('performance_warnings'):
            num_rows = max((index_level.size for index_level in self.new_index_levels))
            num_columns = self.removed_level.size
            num_cells = num_rows * num_columns
            if num_cells > np.iinfo(np.int32).max:
                warnings.warn(f'The following operation may generate {num_cells} cells in the resulting pandas object.', PerformanceWarning, stacklevel=find_stack_level())
        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        v = self.level
        codes = list(self.index.codes)
        if not self.sort:
            codes = [factorize(code)[0] for code in codes]
        levs = list(self.index.levels)
        to_sort = codes[:v] + codes[v + 1:] + [codes[v]]
        sizes = tuple((len(x) for x in levs[:v] + levs[v + 1:] + [levs[v]]))
        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)
        indexer = get_group_index_sorter(comp_index, ngroups)
        return (indexer, to_sort)

    @cache_readonly
    def sorted_labels(self) -> List[np.ndarray]:
        indexer, to_sort = self._indexer_and_to_sort
        if self.sort:
            return [line.take(indexer) for line in to_sort]
        return to_sort

    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray:
        indexer, _ = self._indexer_and_to_sort
        sorted_values = algos.take_nd(values, indexer, axis=0)
        return sorted_values

    def _make_selectors(self) -> None:
        new_levels = self.new_index_levels
        remaining_labels = self.sorted_labels[:-1]
        level_sizes = tuple((len(x) for x in new_levels))
        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)
        comp_index = ensure_platform_int(comp_index)
        stride = self.index.levshape[self.level] + self.lift
        self.full_shape = (ngroups, stride)
        selector = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)
        if mask.sum() < len(self.index):
            raise ValueError('Index contains duplicate entries, cannot reshape')
        self.group_index = comp_index
        self.mask = mask
        if self.sort:
            self.compressor = comp_index.searchsorted(np.arange(ngroups))
        else:
            self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])

    @cache_readonly
    def mask_all(self) -> bool:
        return bool(self.mask.all())

    @cache_readonly
    def arange_result(self) -> Tuple[np.ndarray, np.ndarray]:
        dummy_arr = np.arange(len(self.index), dtype=np.intp)
        new_values, mask = self.get_new_values(dummy_arr, fill_value=-1)
        return (new_values, mask.any(0))

    def get_result(
        self,
        obj: Union[Series, DataFrame],
        value_columns: Optional[Any],
        fill_value: Optional[Any]
    ) -> DataFrame:
        values = obj._values
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if value_columns is None and values.shape[1] != 1:
            raise ValueError('must pass column labels for multi-column data')
        new_values, _ = self.get_new_values(values, fill_value)
        columns = self.get_new_columns(value_columns)
        index = self.new_index
        result = self.constructor(new_values, index=index, columns=columns, dtype=new_values.dtype, copy=False)
        if isinstance(values, np.ndarray):
            base, new_base = (values.base, new_values.base)
        elif isinstance(values, NDArrayBackedExtensionArray):
            base, new_base = (values._ndarray.base, new_values._ndarray.base)
        else:
            base, new_base = (1, 2)
        if base is new_base:
            result._mgr.add_references(obj._mgr)
        return result

    def get_new_values(
        self,
        values: np.ndarray,
        fill_value: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if values.ndim == 1:
            values = values[:, np.newaxis]
        sorted_values = self._make_sorted_values(values)
        length, width = self.full_shape
        stride = values.shape[1]
        result_width = width * stride
        result_shape = (length, result_width)
        mask = self.mask
        mask_all = self.mask_all
        if mask_all and len(values):
            new_values = sorted_values.reshape(length, width, stride).swapaxes(1, 2).reshape(result_shape)
            new_mask = np.ones(result_shape, dtype=bool)
            return (new_values, new_mask)
        dtype = values.dtype
        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            new_values = cls._empty(result_shape, dtype=dtype)
            if not mask_all:
                new_values[:] = fill_value
        else:
            if not mask_all:
                dtype, fill_value = maybe_promote(dtype, fill_value)
            new_values = np.empty(result_shape, dtype=dtype)
            if not mask_all:
                new_values.fill(fill_value)
        name = dtype.name
        new_mask = np.zeros(result_shape, dtype=bool)
        if needs_i8_conversion(values.dtype):
            sorted_values = sorted_values.view('i8')
            new_values = new_values.view('i8')
        else:
            sorted_values = sorted_values.astype(name, copy=False)
        libreshape.unstack(sorted_values, mask.view('u1'), stride, length, width, new_values, new_mask.view('u1'))
        if needs_i8_conversion(values.dtype):
            new_values = new_values.view('M8[ns]')
            new_values = ensure_wrapped_if_datetimelike(new_values)
            new_values = new_values.view(values.dtype)
        return (new_values, new_mask)

    def get_new_columns(self, value_columns: Optional[Any]) -> Union[Index, MultiIndex]:
        if value_columns is None:
            if self.lift == 0:
                return self.removed_level._rename(name=self.removed_name)
            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)
        stride = len(self.removed_level) + self.lift
        width = len(value_columns)
        propagator = np.repeat(np.arange(width), stride)
        if isinstance(value_columns, MultiIndex):
            new_levels = value_columns.levels + (self.removed_level_full,)
            new_names = value_columns.names + (self.removed_name,)
            new_codes = [lab.take(propagator) for lab in value_columns.codes]
        else:
            new_levels = [value_columns, self.removed_level_full]
            new_names = [value_columns.name, self.removed_name]
            new_codes = [propagator]
        repeater = self._repeater
        new_codes.append(np.tile(repeater, width))
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    @cache_readonly
    def _repeater(self) -> np.ndarray:
        if len(self.removed_level_full) != len(self.removed_level):
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            if self.lift:
                repeater = np.insert(repeater, 0, -1)
        else:
            stride = len(self.removed_level) + self.lift
            repeater = np.arange(stride) - self.lift
        return repeater

    @cache_readonly
    def new_index(self) -> Union[Index, MultiIndex]:
        if self.sort:
            labels = self.sorted_labels[:-1]
        else:
            v = self.level
            codes = list(self.index.codes)
            labels = codes[:v] + codes[v + 1:]
        result_codes = [lab.take(self.compressor) for lab in labels]
        if len(self.new_index_levels) == 1:
            level, level_codes = (self.new_index_levels[0], result_codes[0])
            if (level_codes == -1).any():
                level = level.insert(len(level), level._na_value)
            return level.take(level_codes).rename(self.new_index_names[0])
        return MultiIndex(levels=self.new_index_levels, codes=result_codes, names=self.new_index_names, verify_integrity=False)

def _unstack_multiple(
    data: Union[Series, DataFrame],
    clocs: Union[int, List[int]],
    fill_value: Optional[Any] = None,
    sort: bool = True
) -> Union[Series, DataFrame]:
    if len(clocs) == 0:
        return data
    index = data.index
    index = cast(MultiIndex, index)
    if clocs in index.names:
        clocs = [clocs]
    clocs = [index._get_level_number(i) for i in clocs]
    rlocs = [i for i in range(index.nlevels) if i not in clocs]
    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]
    shape = tuple((len(x) for x in clevels))
    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)
    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)
    if not rlocs:
        dummy_index = Index(obs_ids, name='__placeholder__')
    else:
        dummy_index = MultiIndex(levels=rlevels + [obs_ids], codes=rcodes + [comp_ids], names=rnames + ['__placeholder__'], verify_integrity=False)
    if isinstance(data, Series):
        dummy = data.copy(deep=False)
        dummy.index = dummy_index
        unstacked = dummy.unstack('__placeholder__', fill_value=fill_value, sort=sort)
        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            while clocs:
                val = clocs.pop(0)
                result = result.unstack(val, fill_value=fill_value, sort=sort)
                clocs = [v if v < val else v - 1 for v in clocs]
            return result
        dummy_df = data.copy(deep=False)
        dummy_df.index = dummy_index
        unstacked = dummy_df.unstack('__placeholder__', fill_value=fill_value, sort=sort)
        if isinstance(unstacked, Series):
            unstcols = unstacked.index
        else:
            unstcols = unstacked.columns
        assert isinstance(unstcols, MultiIndex)
        new_levels = [unstcols.levels[0]] + clevels
        new_names = [data.columns.name] + cnames
        new_codes = [unstcols.codes[0]]
        new_codes.extend((rec.take(unstcols.codes[-1]) for rec in recons_codes))
    new_columns = MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns
    return unstacked

@overload
def unstack(obj: Series, level: Union[int, str, List[Union[int, str]] = ..., fill_value: Any = ..., sort: bool = ...) -> DataFrame: ...

@overload
def unstack(obj: DataFrame, level: Union[int, str, List[Union[int, str]]] = ..., fill_value: Any = ..., sort: bool = ...) -> Union[DataFrame, Series]: ...

def unstack(
    obj: Union[Series, DataFrame],
    level: Union[int, str, List[Union[int, str]] = -1,
    fill_value: Optional[Any] = None,
    sort: bool = True
) -> Union[DataFrame, Series]:
    if isinstance(level, (tuple, list)):
        if len(level) != 1:
            return _unstack_multiple(obj, level, fill_value=fill_value, sort=sort)
        else:
            level = level[0]
    if not is_integer(level) and (not level == '__placeholder__'):
        obj.index._get_level_number(level)
    if isinstance(obj, DataFrame):
        if isinstance(obj.index, MultiIndex):
            return _unstack_frame(obj, level, fill_value=fill_value, sort=sort)
        else:
            return obj.T.stack()
    elif not isinstance(obj.index, MultiIndex):
        raise ValueError(f'index must be a MultiIndex to unstack, {type(obj.index)} was passed')
    else:
        if is_1d_only_ea_dtype(obj.dtype):
            return _unstack_extension_series(obj, level, fill_value, sort=sort)
        unstacker = _Unstacker(obj.index, level=level, constructor=obj._constructor_expanddim, sort=sort)
        return unstacker.get_result(obj, value_columns=None, fill_value=fill_value)

def _unstack_frame(
    obj: DataFrame,
    level: Union[int, str],
    fill_value: Optional[Any] = None,
    sort: bool = True
) -> Union[DataFrame, Series]:
    assert isinstance(obj.index, MultiIndex)
    unstacker = _Unstacker(obj.index, level=level, constructor=obj._constructor, sort=sort)
    if not obj._can_fast_transpose:
        mgr = obj._mgr.unstack(unstacker, fill_value=fill_value)
        return obj._constructor_from_mgr(mgr, axes=mgr.axes)
    else:
        return unstacker.get_result(obj, value_columns=obj.columns, fill_value=fill_value)

def _unstack_extension_series(
    series: Series,
    level: Union[int, str],
    fill_value: Any,
    sort: bool
) -> DataFrame:
    df = series.to_frame()
    result = df.unstack(level=level, fill_value=fill_value, sort=sort)
    result.columns = result.columns._drop_level_numbers([0])
    return result

def stack(
    frame: DataFrame,
    level: int = -1,
    dropna: bool = True,
    sort: bool = True
) -> Union[Series, DataFrame]:
    def stack_factorize(index: Index) -> Tuple[Index, np.ndarray]:
        if index.is_unique:
            return (index, np.arange(len(index)))
        codes, categories = factorize_from_iterable(index)
        return (categories, codes)
    N, K = frame.shape
