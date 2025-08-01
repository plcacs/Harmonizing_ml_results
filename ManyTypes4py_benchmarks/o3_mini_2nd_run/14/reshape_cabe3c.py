from __future__ import annotations
import itertools
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Set, overload, cast
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
    """
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    index : MultiIndex
    level : int or str, default last level
        Level to "unstack". Accepts a name for the level.
    fill_value : scalar, optional
        Default value to fill in missing values if subgroups do not have the
        same set of labels. By default, missing values will be replaced with
        the default fill value for that data type, NaN for float, NaT for
        datetimelike, etc. For integer types, by default data will converted to
        float and missing values will be set to NaN.
    constructor : object
        Pandas ``DataFrame`` or subclass used to create unstacked
        response.  If None, DataFrame will be used.

    Examples
    --------
    >>> index = pd.MultiIndex.from_tuples(
    ...     [("one", "a"), ("one", "b"), ("two", "a"), ("two", "b")]
    ... )
    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)
    >>> s
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1)
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0)
       one  two
    a    1    3
    b    2    4

    Returns
    -------
    unstacked : DataFrame
    """

    def __init__(self, index: MultiIndex, level: Union[int, str], constructor: Any, sort: bool = True) -> None:
        self.constructor: Any = constructor
        self.sort: bool = sort
        self.index: MultiIndex = index.remove_unused_levels()
        self.level: int = self.index._get_level_number(level)
        self.lift: int = 1 if -1 in self.index.codes[self.level] else 0
        self.new_index_levels: List[Index] = list(self.index.levels)
        self.new_index_names: List[Optional[str]] = list(self.index.names)
        self.removed_name: Optional[str] = self.new_index_names.pop(self.level)
        self.removed_level: Index = self.new_index_levels.pop(self.level)
        self.removed_level_full: Index = index.levels[self.level]
        if not self.sort:
            unique_codes = unique(self.index.codes[self.level])
            self.removed_level = self.removed_level.take(unique_codes)
            self.removed_level_full = self.removed_level_full.take(unique_codes)
        if get_option('performance_warnings'):
            num_rows = max((index_level.size for index_level in self.new_index_levels))
            num_columns = self.removed_level.size
            num_cells = num_rows * num_columns
            if num_cells > np.iinfo(np.int32).max:
                warnings.warn(f'The following operation may generate {num_cells} cells in the resulting pandas object.',
                              PerformanceWarning, stacklevel=find_stack_level())
        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        v = self.level
        codes: List[np.ndarray] = list(self.index.codes)
        if not self.sort:
            codes = [factorize(code)[0] for code in codes]
        levs: List[Index] = list(self.index.levels)
        to_sort: List[np.ndarray] = codes[:v] + codes[v + 1:] + [codes[v]]
        sizes: Tuple[int, ...] = tuple((len(x) for x in levs[:v] + levs[v + 1:] + [levs[v]]))
        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups: int = len(obs_ids)
        indexer: np.ndarray = get_group_index_sorter(comp_index, ngroups)
        return (indexer, to_sort)

    @cache_readonly
    def sorted_labels(self) -> List[np.ndarray]:
        indexer, to_sort = self._indexer_and_to_sort
        if self.sort:
            return [line.take(indexer) for line in to_sort]
        return to_sort

    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray:
        indexer, _ = self._indexer_and_to_sort
        sorted_values: np.ndarray = algos.take_nd(values, indexer, axis=0)
        return sorted_values

    def _make_selectors(self) -> None:
        new_levels: List[Index] = self.new_index_levels
        remaining_labels: List[np.ndarray] = self.sorted_labels[:-1]
        level_sizes: Tuple[int, ...] = tuple((len(x) for x in new_levels))
        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups: int = len(obs_ids)
        comp_index = ensure_platform_int(comp_index)
        stride: int = self.index.levshape[self.level] + self.lift
        self.full_shape: Tuple[int, int] = (ngroups, stride)
        selector: np.ndarray = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask: np.ndarray = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)
        if mask.sum() < len(self.index):
            raise ValueError('Index contains duplicate entries, cannot reshape')
        self.group_index: np.ndarray = comp_index
        self.mask: np.ndarray = mask
        if self.sort:
            self.compressor: np.ndarray = comp_index.searchsorted(np.arange(ngroups))
        else:
            self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])

    @cache_readonly
    def mask_all(self) -> bool:
        return bool(self.mask.all())

    @cache_readonly
    def arange_result(self) -> Tuple[np.ndarray, np.ndarray]:
        dummy_arr: np.ndarray = np.arange(len(self.index), dtype=np.intp)
        new_values, mask = self.get_new_values(dummy_arr, fill_value=-1)
        return (new_values, mask.any(0))

    def get_result(self, obj: Union[Series, DataFrame], value_columns: Optional[Any], fill_value: Any) -> DataFrame:
        values: np.ndarray = obj._values
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if value_columns is None and values.shape[1] != 1:
            raise ValueError('must pass column labels for multi-column data')
        new_values, _ = self.get_new_values(values, fill_value)
        columns: MultiIndex = self.get_new_columns(value_columns)
        index: Union[Index, MultiIndex] = self.new_index
        result: DataFrame = self.constructor(new_values, index=index, columns=columns, dtype=new_values.dtype, copy=False)
        if isinstance(values, np.ndarray):
            base, new_base = (values.base, new_values.base)
        elif isinstance(values, NDArrayBackedExtensionArray):
            base, new_base = (values._ndarray.base, new_values._ndarray.base)
        else:
            base, new_base = (1, 2)
        if base is new_base:
            result._mgr.add_references(obj._mgr)
        return result

    def get_new_values(self, values: np.ndarray, fill_value: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        if values.ndim == 1:
            values = values[:, np.newaxis]
        sorted_values: np.ndarray = self._make_sorted_values(values)
        length, width = self.full_shape
        stride: int = values.shape[1]
        result_width: int = width * stride
        result_shape: Tuple[int, int] = (length, result_width)
        mask: np.ndarray = self.mask
        mask_all: bool = self.mask_all
        if mask_all and len(values):
            new_values: np.ndarray = sorted_values.reshape(length, width, stride).swapaxes(1, 2).reshape(result_shape)
            new_mask: np.ndarray = np.ones(result_shape, dtype=bool)
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
        new_mask: np.ndarray = np.zeros(result_shape, dtype=bool)
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

    def get_new_columns(self, value_columns: Optional[Any]) -> MultiIndex:
        if value_columns is None:
            if self.lift == 0:
                return self.removed_level._rename(name=self.removed_name)
            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)
        stride: int = len(self.removed_level) + self.lift
        width: int = len(value_columns)
        propagator: np.ndarray = np.repeat(np.arange(width), stride)
        if isinstance(value_columns, MultiIndex):
            new_levels: List[Index] = value_columns.levels + (self.removed_level_full,)
            new_names: List[Optional[str]] = value_columns.names + (self.removed_name,)
            new_codes: List[np.ndarray] = [lab.take(propagator) for lab in value_columns.codes]
        else:
            new_levels = [value_columns, self.removed_level_full]
            new_names = [value_columns.name, self.removed_name]
            new_codes = [propagator]
        repeater: np.ndarray = self._repeater
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
            labels: List[np.ndarray] = self.sorted_labels[:-1]
        else:
            v = self.level
            codes: List[np.ndarray] = list(self.index.codes)
            labels = codes[:v] + codes[v + 1:]
        result_codes = [lab.take(self.compressor) for lab in labels]
        if len(self.new_index_levels) == 1:
            level: Index = self.new_index_levels[0]
            level_codes: np.ndarray = result_codes[0]
            if (level_codes == -1).any():
                level = level.insert(len(level), level._na_value)
            return level.take(level_codes).rename(self.new_index_names[0])
        return MultiIndex(levels=self.new_index_levels, codes=result_codes, names=self.new_index_names, verify_integrity=False)


def _unstack_multiple(data: Union[DataFrame, Series],
                      clocs: Union[str, int, List[Union[str, int]]],
                      fill_value: Any = None,
                      sort: bool = True) -> Union[DataFrame, Series]:
    index: MultiIndex = data.index  # type: ignore
    index = cast(MultiIndex, index)
    if clocs in index.names:
        clocs = [clocs]  # type: ignore
    clocs = [index._get_level_number(i) for i in cast(Sequence[Union[str, int]], clocs)]
    rlocs: List[int] = [i for i in range(index.nlevels) if i not in clocs]
    clevels: List[Index] = [index.levels[i] for i in clocs]
    ccodes: List[np.ndarray] = [index.codes[i] for i in clocs]
    cnames: List[Optional[str]] = [index.names[i] for i in clocs]
    rlevels: List[Index] = [index.levels[i] for i in rlocs]
    rcodes: List[np.ndarray] = [index.codes[i] for i in rlocs]
    rnames: List[Optional[str]] = [index.names[i] for i in rlocs]
    shape: Tuple[int, ...] = tuple((len(x) for x in clevels))
    group_index: np.ndarray = get_group_index(ccodes, shape, sort=False, xnull=False)
    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    recons_codes: List[np.ndarray] = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)
    if not rlocs:
        dummy_index: Index = Index(obs_ids, name='__placeholder__')
    else:
        dummy_index = MultiIndex(levels=rlevels + [obs_ids], codes=rcodes + [comp_ids], names=rnames + ['__placeholder__'], verify_integrity=False)
    if isinstance(data, Series):
        dummy: Series = data.copy(deep=False)
        dummy.index = dummy_index
        unstacked: Union[DataFrame, Series] = dummy.unstack('__placeholder__', fill_value=fill_value, sort=sort)
        new_levels: List[Index] = clevels
        new_names: List[Optional[str]] = cnames
        new_codes: List[np.ndarray] = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            while clocs:
                val = clocs.pop(0)
                result = result.unstack(val, fill_value=fill_value, sort=sort)
                clocs = [v if v < val else v - 1 for v in clocs]
            return result
        dummy_df: DataFrame = data.copy(deep=False)
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
    new_columns: MultiIndex = MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns
    return unstacked


@overload
def unstack(obj: Union[DataFrame, Series], level: Union[int, str, List[Union[int, str]], Tuple[Union[int, str], ...]],
            fill_value: Any = ..., sort: bool = ...) -> Union[DataFrame, Series]:
    ...


@overload
def unstack(obj: Union[DataFrame, Series], level: Union[int, str, List[Union[int, str]], Tuple[Union[int, str], ...]],
            fill_value: Any = ..., sort: bool = ...) -> Union[DataFrame, Series]:
    ...


def unstack(obj: Union[DataFrame, Series],
            level: Union[int, str, List[Union[int, str]], Tuple[Union[int, str], ...]],
            fill_value: Any = None,
            sort: bool = True) -> Union[DataFrame, Series]:
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


def _unstack_frame(obj: DataFrame, level: Union[int, str], fill_value: Any = None, sort: bool = True) -> DataFrame:
    assert isinstance(obj.index, MultiIndex)
    unstacker = _Unstacker(obj.index, level=level, constructor=obj._constructor, sort=sort)
    if not obj._can_fast_transpose:
        mgr = obj._mgr.unstack(unstacker, fill_value=fill_value)
        return obj._constructor_from_mgr(mgr, axes=mgr.axes)
    else:
        return unstacker.get_result(obj, value_columns=obj.columns, fill_value=fill_value)


def _unstack_extension_series(series: Series, level: Union[int, str], fill_value: Any, sort: bool) -> DataFrame:
    """
    Unstack an ExtensionArray-backed Series.

    The ExtensionDtype is preserved.

    Parameters
    ----------
    series : Series
        A Series with an ExtensionArray for values
    level : Any
        The level name or number.
    fill_value : Any
        The user-level (not physical storage) fill value to use for
        missing values introduced by the reshape. Passed to
        ``series.values.take``.
    sort : bool
        Whether to sort the resulting MuliIndex levels

    Returns
    -------
    DataFrame
        Each column of the DataFrame will have the same dtype as
        the input Series.
    """
    df: DataFrame = series.to_frame()
    result: DataFrame = df.unstack(level=level, fill_value=fill_value, sort=sort)
    result.columns = result.columns._drop_level_numbers([0])
    return result


def stack(frame: DataFrame, level: Union[int, str] = -1, dropna: bool = True, sort: bool = True) -> Union[Series, DataFrame]:
    """
    Convert DataFrame to Series with multi-level Index. Columns become the
    second level of the resulting hierarchical index

    Returns
    -------
    stacked : Series or DataFrame
    """
    def stack_factorize(index: Index) -> Tuple[Union[Index, np.ndarray], np.ndarray]:
        if index.is_unique:
            return (index, np.arange(len(index)))
        codes, categories = factorize_from_iterable(index)
        return (categories, codes)
    N: int
    K: int
    N, K = frame.shape
    level_num: int = frame.columns._get_level_number(level)
    if isinstance(frame.columns, MultiIndex):
        return _stack_multi_columns(frame, level_num=level_num, dropna=dropna, sort=sort)
    elif isinstance(frame.index, MultiIndex):
        new_levels: List[Index] = list(frame.index.levels)
        new_codes: List[np.ndarray] = [lab.repeat(K) for lab in frame.index.codes]
        clev, clab = stack_factorize(frame.columns)
        new_levels.append(clev)
        new_codes.append(np.tile(clab, N).ravel())
        new_names: List[Optional[str]] = list(frame.index.names)
        new_names.append(frame.columns.name)
        new_index: MultiIndex = MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
    else:
        levels_codes = [stack_factorize(x) for x in (frame.index, frame.columns)]
        levels = [levels_codes[0][0], levels_codes[1][0]]
        codes = (levels_codes[0][1].repeat(K), np.tile(levels_codes[1][1], N).ravel())
        new_index = MultiIndex(levels=levels, codes=codes, names=[frame.index.name, frame.columns.name], verify_integrity=False)
    if not frame.empty and frame._is_homogeneous_type:
        dtypes = list(frame.dtypes._values)
        dtype = dtypes[0]
        if isinstance(dtype, ExtensionDtype):
            arr = dtype.construct_array_type()
            new_values = arr._concat_same_type([col._values for _, col in frame.items()])
            new_values = _reorder_for_extension_array_stack(new_values, N, K)
        else:
            new_values = frame._values.ravel()
    else:
        new_values = frame._values.ravel()
    if dropna:
        mask = notna(new_values)
        new_values = new_values[mask]
        new_index = new_index[mask]
    return frame._constructor_sliced(new_values, index=new_index)


def stack_multiple(frame: DataFrame, level: List[Union[int, str]], dropna: bool = True, sort: bool = True) -> Union[Series, DataFrame]:
    if all((lev in frame.columns.names for lev in level)):
        result = frame
        for lev in level:
            result = stack(result, lev, dropna=dropna, sort=sort)
    elif all((isinstance(lev, int) for lev in level)):
        result = frame
        level = [frame.columns._get_level_number(lev) for lev in level]
        while level:
            lev = level.pop(0)
            result = stack(result, lev, dropna=dropna, sort=sort)
            level = [v if v <= lev else v - 1 for v in level]
    else:
        raise ValueError('level should contain all level names or all level numbers, not a mixture of the two.')
    return result


def _stack_multi_column_index(columns: MultiIndex) -> MultiIndex:
    """Creates a MultiIndex from the first N-1 levels of this MultiIndex."""
    if len(columns.levels) <= 2:
        return columns.levels[0]._rename(name=columns.names[0])
    levs = ([lev[c] if c >= 0 else None for c in codes] for lev, codes in zip(columns.levels[:-1], columns.codes[:-1]))
    tuples = zip(*levs)
    unique_tuples = (key for key, _ in itertools.groupby(tuples))
    new_levs = zip(*unique_tuples)
    return MultiIndex.from_arrays([Index(new_lev, dtype=lev.dtype) if None not in new_lev else new_lev for new_lev, lev in zip(new_levs, columns.levels)], names=columns.names[:-1])


def _stack_multi_columns(frame: DataFrame, level_num: int = -1, dropna: bool = True, sort: bool = True) -> DataFrame:
    def _convert_level_number(level_num: int, columns: MultiIndex) -> Union[int, str]:
        """
        Logic for converting the level number to something we can safely pass
        to swaplevel.

        If `level_num` matches a column name return the name from
        position `level_num`, otherwise return `level_num`.
        """
        if level_num in columns.names:
            return columns.names[level_num]
        return level_num
    this: DataFrame = frame.copy(deep=False)
    mi_cols: MultiIndex = this.columns  # type: ignore
    assert isinstance(mi_cols, MultiIndex)
    if level_num != mi_cols.nlevels - 1:
        roll_columns: MultiIndex = mi_cols
        for i in range(level_num, mi_cols.nlevels - 1):
            lev1 = _convert_level_number(i, roll_columns)
            lev2 = _convert_level_number(i + 1, roll_columns)
            roll_columns = roll_columns.swaplevel(lev1, lev2)
        this.columns = mi_cols = roll_columns
    if not mi_cols._is_lexsorted() and sort:
        level_to_sort = _convert_level_number(0, mi_cols)
        this = this.sort_index(level=level_to_sort, axis=1)
        mi_cols = this.columns  # type: ignore
    mi_cols = cast(MultiIndex, mi_cols)
    new_columns: MultiIndex = _stack_multi_column_index(mi_cols)
    new_data: dict[Any, Any] = {}
    level_vals: Index = mi_cols.levels[-1]
    level_codes: np.ndarray = unique(mi_cols.codes[-1])
    if sort:
        level_codes = np.sort(level_codes)
    level_vals_nan: Index = level_vals.insert(len(level_vals), None)
    level_vals_used: np.ndarray = np.take(level_vals_nan, level_codes)
    levsize: int = len(level_codes)
    drop_cols: List[Any] = []
    for key in new_columns:
        try:
            loc = this.columns.get_loc(key)
        except KeyError:
            drop_cols.append(key)
            continue
        if not isinstance(loc, slice):
            slice_len = len(loc)
        else:
            slice_len = loc.stop - loc.start
        if slice_len != levsize:
            chunk = this.loc[:, this.columns[loc]]
            chunk.columns = level_vals_nan.take(chunk.columns.codes[-1])
            value_slice = chunk.reindex(columns=level_vals_used).values
        else:
            subset = this.iloc[:, loc]
            dtype = find_common_type(subset.dtypes.tolist())
            if isinstance(dtype, ExtensionDtype):
                value_slice = dtype.construct_array_type()._concat_same_type([x._values.astype(dtype, copy=False) for _, x in subset.items()])
                N, K = subset.shape
                idx = np.arange(N * K).reshape(K, N).T.reshape(-1)
                value_slice = value_slice.take(idx)
            else:
                value_slice = subset.values
        if value_slice.ndim > 1:
            value_slice = value_slice.ravel()
        new_data[key] = value_slice
    if len(drop_cols) > 0:
        new_columns = new_columns.difference(drop_cols)
    N: int = len(this)
    if isinstance(this.index, MultiIndex):
        new_levels = list(this.index.levels)
        new_names = list(this.index.names)
        new_codes = [lab.repeat(levsize) for lab in this.index.codes]
    else:
        old_codes, old_levels = factorize_from_iterable(this.index)
        new_levels = [old_levels]
        new_codes = [old_codes.repeat(levsize)]
        new_names = [this.index.name]
    new_levels.append(level_vals)
    new_codes.append(np.tile(level_codes, N))
    new_names.append(frame.columns.names[level_num])
    new_index: MultiIndex = MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)
    result: DataFrame = frame._constructor(new_data, index=new_index, columns=new_columns)
    if frame.columns.nlevels > 1:
        desired_columns = frame.columns._drop_level_numbers([level_num]).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]
    if dropna:
        result = result.dropna(axis=0, how='all')
    return result


def _reorder_for_extension_array_stack(arr: Any, n_rows: int, n_columns: int) -> Any:
    """
    Re-orders the values when stacking multiple extension-arrays.

    The indirect stacking method used for EAs requires a followup
    take to get the order correct.

    Parameters
    ----------
    arr : ExtensionArray
    n_rows, n_columns : int
        The number of rows and columns in the original DataFrame.

    Returns
    -------
    taken : ExtensionArray
        The original `arr` with elements re-ordered appropriately

    Examples
    --------
    >>> arr = np.array(["a", "b", "c", "d", "e", "f"])
    >>> _reorder_for_extension_array_stack(arr, 2, 3)
    array(['a', 'c', 'e', 'b', 'd', 'f'], dtype='<U1')

    >>> _reorder_for_extension_array_stack(arr, 3, 2)
    array(['a', 'd', 'b', 'e', 'c', 'f'], dtype='<U1')
    """
    idx: np.ndarray = np.arange(n_rows * n_columns).reshape(n_columns, n_rows).T.reshape(-1)
    return arr.take(idx)


def stack_v3(frame: DataFrame, level: List[int]) -> Union[Series, DataFrame]:
    if frame.columns.nunique() != len(frame.columns):
        raise ValueError('Columns with duplicate values are not supported in stack')
    if not len(level):
        return frame
    set_levels: Set[int] = set(level)
    stack_cols: Index = frame.columns._drop_level_numbers([k for k in range(frame.columns.nlevels - 1, -1, -1) if k not in set_levels])
    result: Union[DataFrame, Series] = stack_reshape(frame, level, set_levels, stack_cols)
    ratio: int = 0 if frame.empty else len(result) // len(frame)
    if isinstance(frame.index, MultiIndex):
        index_levels = frame.index.levels
        index_codes = list(np.tile(frame.index.codes, (1, ratio)))
    else:
        codes, uniques = factorize(frame.index, use_na_sentinel=False)
        index_levels = [uniques]
        index_codes = list(np.tile(codes, (1, ratio)))
    if len(level) > 1:
        sorter = np.argsort(level)
        assert isinstance(stack_cols, MultiIndex)
        ordered_stack_cols = stack_cols._reorder_ilevels(sorter)
    else:
        ordered_stack_cols = stack_cols
    ordered_stack_cols_unique = ordered_stack_cols.unique()
    if isinstance(ordered_stack_cols, MultiIndex):
        column_levels = ordered_stack_cols.levels
        column_codes = ordered_stack_cols.drop_duplicates().codes
    else:
        column_levels = [ordered_stack_cols_unique]
        column_codes = [factorize(ordered_stack_cols_unique, use_na_sentinel=False)[0]]
    column_codes = [np.repeat(codes, len(frame)) for codes in column_codes]
    result.index = MultiIndex(levels=index_levels + column_levels, codes=index_codes + column_codes, names=frame.index.names + list(ordered_stack_cols.names), verify_integrity=False)
    len_df: int = len(frame)
    n_uniques: int = len(ordered_stack_cols_unique)
    indexer: np.ndarray = np.arange(n_uniques)
    idxs: np.ndarray = np.tile(len_df * indexer, len_df) + np.repeat(np.arange(len_df), n_uniques)
    result = result.take(idxs)
    if result.ndim == 2 and frame.columns.nlevels == len(level):
        if len(result.columns) == 0:
            result = Series(index=result.index)
        else:
            result = result.iloc[:, 0]
    if result.ndim == 1:
        result.name = None
    return result


def stack_reshape(frame: DataFrame, level: List[int], set_levels: Set[int], stack_cols: Index) -> DataFrame:
    """Reshape the data of a frame for stack.

    This function takes care of most of the work that stack needs to do. Caller
    will sort the result once the appropriate index is set.

    Parameters
    ----------
    frame: DataFrame
        DataFrame that is to be stacked.
    level: list of ints.
        Levels of the columns to stack.
    set_levels: set of ints.
        Same as level, but as a set.
    stack_cols: Index.
        Columns of the result when the DataFrame is stacked.

    Returns
    -------
    The data of behind the stacked DataFrame.
    """
    drop_levnums: List[int] = sorted(level, reverse=True)
    buf: List[DataFrame] = []
    for idx in stack_cols.unique():
        if len(frame.columns) == 1:
            data = frame.copy(deep=False)
        else:
            if not isinstance(frame.columns, MultiIndex) and (not isinstance(idx, tuple)):
                column_indexer = idx
            else:
                if len(level) == 1:
                    idx = (idx,)
                gen = iter(idx)
                column_indexer = tuple((next(gen) if k in set_levels else slice(None) for k in range(frame.columns.nlevels)))
            data = frame.loc[:, column_indexer]
        if len(level) < frame.columns.nlevels:
            data.columns = data.columns._drop_level_numbers(drop_levnums)
        elif stack_cols.nlevels == 1:
            if data.ndim == 1:
                data.name = 0
            else:
                data.columns = default_index(len(data.columns))
        buf.append(data)
    if len(buf) > 0 and (not frame.empty):
        result = concat(buf, ignore_index=True)
    else:
        if len(level) < frame.columns.nlevels:
            new_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        else:
            new_columns = [0]
        result = DataFrame(columns=new_columns, dtype=frame._values.dtype)
    if len(level) < frame.columns.nlevels:
        desired_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]
    return result
