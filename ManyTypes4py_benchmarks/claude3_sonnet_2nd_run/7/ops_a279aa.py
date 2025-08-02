"""
Provide classes to perform the groupby aggregate operations.

These are not exposed to the user and provide implementations of the grouping
operations, primarily in cython. These classes (BaseGrouper and BinGrouper)
are contained *in* the SeriesGroupBy and DataFrameGroupBy objects.
"""
from __future__ import annotations
import collections
import functools
from typing import TYPE_CHECKING, Generic, final, Any, Callable, Dict, List, Optional, Tuple, Union, cast
import numpy as np
from pandas._libs import NaT, lib
import pandas._libs.groupby as libgroupby
from pandas._typing import ArrayLike, AxisInt, NDFrameT, Shape, npt
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import maybe_cast_pointwise_result, maybe_downcast_to_dtype
from pandas.core.dtypes.common import ensure_float64, ensure_int64, ensure_platform_int, ensure_uint64, is_1d_only_ea_dtype
from pandas.core.dtypes.missing import isna, maybe_fill
from pandas.core.arrays import Categorical
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import CategoricalIndex, Index, MultiIndex, ensure_index
from pandas.core.series import Series
from pandas.core.sorting import compress_group_index, decons_obs_group_ids, get_group_index, get_group_index_sorter, get_indexer_dict
if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable, Iterator
    from pandas.core.generic import NDFrame

def check_result_array(obj: Any, dtype: np.dtype) -> None:
    if isinstance(obj, np.ndarray):
        if dtype != object:
            raise ValueError('Must produce aggregated value')

def extract_result(res: Any) -> Any:
    """
    Extract the result object, it might be a 0-dim ndarray
    or a len-1 0-dim, or a scalar
    """
    if hasattr(res, '_values'):
        res = res._values
        if res.ndim == 1 and len(res) == 1:
            res = res[0]
    return res

class WrappedCythonOp:
    """
    Dispatch logic for functions defined in _libs.groupby

    Parameters
    ----------
    kind: str
        Whether the operation is an aggregate or transform.
    how: str
        Operation name, e.g. "mean".
    has_dropped_na: bool
        True precisely when dropna=True and the grouper contains a null value.
    """
    cast_blocklist = frozenset(['any', 'all', 'rank', 'count', 'size', 'idxmin', 'idxmax'])

    def __init__(self, kind: str, how: str, has_dropped_na: bool):
        self.kind = kind
        self.how = how
        self.has_dropped_na = has_dropped_na
    _CYTHON_FUNCTIONS: Dict[str, Dict[str, Union[str, Callable]]] = {'aggregate': {'any': functools.partial(libgroupby.group_any_all, val_test='any'), 'all': functools.partial(libgroupby.group_any_all, val_test='all'), 'sum': 'group_sum', 'prod': 'group_prod', 'idxmin': functools.partial(libgroupby.group_idxmin_idxmax, name='idxmin'), 'idxmax': functools.partial(libgroupby.group_idxmin_idxmax, name='idxmax'), 'min': 'group_min', 'max': 'group_max', 'mean': 'group_mean', 'median': 'group_median_float64', 'var': 'group_var', 'std': functools.partial(libgroupby.group_var, name='std'), 'sem': functools.partial(libgroupby.group_var, name='sem'), 'skew': 'group_skew', 'kurt': 'group_kurt', 'first': 'group_nth', 'last': 'group_last', 'ohlc': 'group_ohlc'}, 'transform': {'cumprod': 'group_cumprod', 'cumsum': 'group_cumsum', 'cummin': 'group_cummin', 'cummax': 'group_cummax', 'rank': 'group_rank'}}
    _cython_arity: Dict[str, int] = {'ohlc': 4}

    @classmethod
    def get_kind_from_how(cls, how: str) -> str:
        if how in cls._CYTHON_FUNCTIONS['aggregate']:
            return 'aggregate'
        return 'transform'

    @classmethod
    @functools.cache
    def _get_cython_function(cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool) -> Callable:
        dtype_str = dtype.name
        ftype = cls._CYTHON_FUNCTIONS[kind][how]
        if callable(ftype):
            f = ftype
        else:
            f = getattr(libgroupby, ftype)
        if is_numeric:
            return f
        elif dtype == np.dtype(object):
            if how in ['median', 'cumprod']:
                raise NotImplementedError(f'function is not implemented for this dtype: [how->{how},dtype->{dtype_str}]')
            elif how in ['std', 'sem', 'idxmin', 'idxmax']:
                return f
            elif how in ['skew', 'kurt']:
                pass
            elif 'object' not in f.__signatures__:
                raise NotImplementedError(f'function is not implemented for this dtype: [how->{how},dtype->{dtype_str}]')
            return f
        else:
            raise NotImplementedError('This should not be reached. Please report a bug at github.com/pandas-dev/pandas/', dtype)

    def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
        """
        Cast numeric dtypes to float64 for functions that only support that.

        Parameters
        ----------
        values : np.ndarray

        Returns
        -------
        values : np.ndarray
        """
        how = self.how
        if how in ['median', 'std', 'sem', 'skew', 'kurt']:
            values = ensure_float64(values)
        elif values.dtype.kind in 'iu':
            if how in ['var', 'mean'] or (self.kind == 'transform' and self.has_dropped_na):
                values = ensure_float64(values)
            elif how in ['sum', 'ohlc', 'prod', 'cumsum', 'cumprod']:
                if values.dtype.kind == 'i':
                    values = ensure_int64(values)
                else:
                    values = ensure_uint64(values)
        return values

    def _get_output_shape(self, ngroups: int, values: np.ndarray) -> Tuple[int, ...]:
        how = self.how
        kind = self.kind
        arity = self._cython_arity.get(how, 1)
        if how == 'ohlc':
            out_shape = (ngroups, arity)
        elif arity > 1:
            raise NotImplementedError("arity of more than 1 is not supported for the 'how' argument")
        elif kind == 'transform':
            out_shape = values.shape
        else:
            out_shape = (ngroups,) + values.shape[1:]
        return out_shape

    def _get_out_dtype(self, dtype: np.dtype) -> np.dtype:
        how = self.how
        if how == 'rank':
            out_dtype = 'float64'
        elif how in ['idxmin', 'idxmax']:
            out_dtype = 'intp'
        elif dtype.kind in 'iufcb':
            out_dtype = f'{dtype.kind}{dtype.itemsize}'
        else:
            out_dtype = 'object'
        return np.dtype(out_dtype)

    def _get_result_dtype(self, dtype: np.dtype) -> np.dtype:
        """
        Get the desired dtype of a result based on the
        input dtype and how it was computed.

        Parameters
        ----------
        dtype : np.dtype

        Returns
        -------
        np.dtype
            The desired dtype of the result.
        """
        how = self.how
        if how in ['sum', 'cumsum', 'sum', 'prod', 'cumprod']:
            if dtype == np.dtype(bool):
                return np.dtype(np.int64)
        elif how in ['mean', 'median', 'var', 'std', 'sem']:
            if dtype.kind in 'fc':
                return dtype
            elif dtype.kind in 'iub':
                return np.dtype(np.float64)
        return dtype

    @final
    def _cython_op_ndim_compat(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: Optional[np.ndarray] = None, result_mask: Optional[np.ndarray] = None, **kwargs: Any) -> np.ndarray:
        if values.ndim == 1:
            values2d = values[None, :]
            if mask is not None:
                mask = mask[None, :]
            if result_mask is not None:
                result_mask = result_mask[None, :]
            res = self._call_cython_op(values2d, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
            if res.shape[0] == 1:
                return res[0]
            return res.T
        return self._call_cython_op(values, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=mask, result_mask=result_mask, **kwargs)

    @final
    def _call_cython_op(self, values: np.ndarray, *, min_count: int, ngroups: int, comp_ids: np.ndarray, mask: Optional[np.ndarray], result_mask: Optional[np.ndarray], **kwargs: Any) -> np.ndarray:
        orig_values = values
        dtype = values.dtype
        is_numeric = dtype.kind in 'iufcb'
        is_datetimelike = dtype.kind in 'mM'
        if self.how in ['any', 'all']:
            if mask is None:
                mask = isna(values)
        if is_datetimelike:
            values = values.view('int64')
            is_numeric = True
        elif dtype.kind == 'b':
            values = values.view('uint8')
        if values.dtype == 'float16':
            values = values.astype(np.float32)
        if self.how in ['any', 'all']:
            if dtype == object:
                if kwargs['skipna']:
                    if mask is not None and mask.any():
                        values = values.copy()
                        values[mask] = True
            values = values.astype(bool, copy=False).view(np.int8)
            is_numeric = True
        values = values.T
        if mask is not None:
            mask = mask.T
            if result_mask is not None:
                result_mask = result_mask.T
        out_shape = self._get_output_shape(ngroups, values)
        func = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
        values = self._get_cython_vals(values)
        out_dtype = self._get_out_dtype(values.dtype)
        result = maybe_fill(np.empty(out_shape, dtype=out_dtype))
        if self.kind == 'aggregate':
            counts = np.zeros(ngroups, dtype=np.int64)
            if self.how in ['idxmin', 'idxmax', 'min', 'max', 'mean', 'last', 'first', 'sum', 'median']:
                func(out=result, counts=counts, values=values, labels=comp_ids, min_count=min_count, mask=mask, result_mask=result_mask, is_datetimelike=is_datetimelike, **kwargs)
            elif self.how in ['sem', 'std', 'var', 'ohlc', 'prod']:
                if self.how in ['std', 'sem']:
                    kwargs['is_datetimelike'] = is_datetimelike
                func(result, counts, values, comp_ids, min_count=min_count, mask=mask, result_mask=result_mask, **kwargs)
            elif self.how in ['any', 'all']:
                func(out=result, values=values, labels=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
                result = result.astype(bool, copy=False)
            elif self.how in ['skew', 'kurt']:
                func(out=result, counts=counts, values=values, labels=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
                if dtype == object:
                    result = result.astype(object)
            else:
                raise NotImplementedError(f'{self.how} is not implemented')
        else:
            if self.how != 'rank':
                kwargs['result_mask'] = result_mask
            func(out=result, values=values, labels=comp_ids, ngroups=ngroups, is_datetimelike=is_datetimelike, mask=mask, **kwargs)
        if self.kind == 'aggregate' and self.how not in ['idxmin', 'idxmax']:
            if result.dtype.kind in 'iu' and (not is_datetimelike):
                cutoff = max(0 if self.how in ['sum', 'prod'] else 1, min_count)
                empty_groups = counts < cutoff
                if empty_groups.any():
                    if result_mask is not None:
                        assert result_mask[empty_groups].all()
                    else:
                        result = result.astype('float64')
                        result[empty_groups] = np.nan
        result = result.T
        if self.how not in self.cast_blocklist:
            res_dtype = self._get_result_dtype(orig_values.dtype)
            op_result = maybe_downcast_to_dtype(result, res_dtype)
        else:
            op_result = result
        return op_result

    @final
    def _validate_axis(self, axis: AxisInt, values: np.ndarray) -> None:
        if values.ndim > 2:
            raise NotImplementedError('number of dimensions is currently limited to 2')
        if values.ndim == 2:
            assert axis == 1, axis
        elif not is_1d_only_ea_dtype(values.dtype):
            assert axis == 0

    @final
    def cython_operation(self, *, values: np.ndarray, axis: AxisInt, min_count: int = -1, comp_ids: np.ndarray, ngroups: int, **kwargs: Any) -> np.ndarray:
        """
        Call our cython function, with appropriate pre- and post- processing.
        """
        self._validate_axis(axis, values)
        if not isinstance(values, np.ndarray):
            return values._groupby_op(how=self.how, has_dropped_na=self.has_dropped_na, min_count=min_count, ngroups=ngroups, ids=comp_ids, **kwargs)
        return self._cython_op_ndim_compat(values, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=None, **kwargs)

class BaseGrouper:
    """
    This is an internal Grouper class, which actually holds
    the generated groups

    Parameters
    ----------
    axis : Index
    groupings : Sequence[Grouping]
        all the grouping instances to handle in this grouper
        for example for grouper list to groupby, need to pass the list
    sort : bool, default True
        whether this grouper will give sorted result or not

    """

    def __init__(self, axis: Index, groupings: List[grouper.Grouping], sort: bool = True, dropna: bool = True):
        assert isinstance(axis, Index), axis
        self.axis = axis
        self._groupings = groupings
        self._sort = sort
        self.dropna = dropna

    @property
    def groupings(self) -> List[grouper.Grouping]:
        return self._groupings

    def __iter__(self) -> Iterator[Tuple[Any, List[int]]]:
        return iter(self.indices)

    @property
    def nkeys(self) -> int:
        return len(self.groupings)

    def get_iterator(self, data: NDFrame) -> Generator[Tuple[Any, NDFrame], None, None]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        splitter = self._get_splitter(data)
        keys = self.result_index
        yield from zip(keys, splitter)

    @final
    def _get_splitter(self, data: NDFrame) -> Union[SeriesSplitter, FrameSplitter]:
        """
        Returns
        -------
        Generator yielding subsetted objects
        """
        if isinstance(data, Series):
            klass = SeriesSplitter
        else:
            klass = FrameSplitter
        return klass(data, self.ngroups, sorted_ids=self._sorted_ids, sort_idx=self.result_ilocs)

    @cache_readonly
    def indices(self) -> Dict[Any, List[int]]:
        """dict {group name -> group indices}"""
        if len(self.groupings) == 1 and isinstance(self.result_index, CategoricalIndex):
            return self.groupings[0].indices
        codes_list = [ping.codes for ping in self.groupings]
        return get_indexer_dict(codes_list, self.levels)

    @final
    @cache_readonly
    def result_ilocs(self) -> np.ndarray:
        """
        Get the original integer locations of result_index in the input.
        """
        ids = self.ids
        if self.has_dropped_na:
            mask = np.where(ids >= 0)
            null_gaps = np.cumsum(ids == -1)[mask]
            ids = ids[mask]
        result = get_group_index_sorter(ids, self.ngroups)
        if self.has_dropped_na:
            result += np.take(null_gaps, result)
        return result

    @property
    def codes(self) -> List[np.ndarray]:
        return [ping.codes for ping in self.groupings]

    @property
    def levels(self) -> List[Index]:
        if len(self.groupings) > 1:
            return list(self.result_index.levels)
        else:
            return [self.result_index]

    @property
    def names(self) -> List[Optional[Hashable]]:
        return [ping.name for ping in self.groupings]

    @final
    def size(self) -> Series:
        """
        Compute group sizes.
        """
        ids = self.ids
        ngroups = self.ngroups
        if ngroups:
            out = np.bincount(ids[ids != -1], minlength=ngroups)
        else:
            out = []
        return Series(out, index=self.result_index, dtype='int64', copy=False)

    @cache_readonly
    def groups(self) -> Dict[Any, np.ndarray]:
        """dict {group name -> group labels}"""
        if len(self.groupings) == 1:
            return self.groupings[0].groups
        result_index, ids = self.result_index_and_ids
        values = result_index._values
        categories = Categorical(ids, categories=range(len(result_index)))
        result = {values[group]: self.axis.take(axis_ilocs) for group, axis_ilocs in categories._reverse_indexer().items()}
        return result

    @final
    @cache_readonly
    def is_monotonic(self) -> bool:
        return Index(self.ids).is_monotonic_increasing

    @final
    @cache_readonly
    def has_dropped_na(self) -> bool:
        """
        Whether grouper has null value(s) that are dropped.
        """
        return bool((self.ids < 0).any())

    @cache_readonly
    def codes_info(self) -> np.ndarray:
        return self.ids

    @final
    @cache_readonly
    def ngroups(self) -> int:
        return len(self.result_index)

    @property
    def result_index(self) -> Index:
        return self.result_index_and_ids[0]

    @property
    def ids(self) -> np.ndarray:
        return self.result_index_and_ids[1]

    @cache_readonly
    def result_index_and_ids(self) -> Tuple[Index, np.ndarray]:
        levels = [Index._with_infer(ping.uniques) for ping in self.groupings]
        obs = [ping._observed or not ping._passed_categorical for ping in self.groupings]
        sorts = [ping._sort for ping in self.groupings]
        for k, (ping, level) in enumerate(zip(self.groupings, levels)):
            if ping._passed_categorical:
                levels[k] = level.set_categories(ping._orig_cats)
        if len(self.groupings) == 1:
            result_index = levels[0]
            result_index.name = self.names[0]
            ids = ensure_platform_int(self.codes[0])
        elif all(obs):
            result_index, ids = self._ob_index_and_ids(levels, self.codes, self.names, sorts)
        elif not any(obs):
            result_index, ids = self._unob_index_and_ids(levels, self.codes, self.names)
        else:
            names = self.names
            codes = [ping.codes for ping in self.groupings]
            ob_indices = [idx for idx, ob in enumerate(obs) if ob]
            unob_indices = [idx for idx, ob in enumerate(obs) if not ob]
            ob_index, ob_ids = self._ob_index_and_ids(levels=[levels[idx] for idx in ob_indices], codes=[codes[idx] for idx in ob_indices], names=[names[idx] for idx in ob_indices], sorts=[sorts[idx] for idx in ob_indices])
            unob_index, unob_ids = self._unob_index_and_ids(levels=[levels[idx] for idx in unob_indices], codes=[codes[idx] for idx in unob_indices], names=[names[idx] for idx in unob_indices])
            result_index_codes = np.concatenate([np.tile(unob_index.codes, len(ob_index)), np.repeat(ob_index.codes, len(unob_index), axis=1)], axis=0)
            _, index = np.unique(unob_indices + ob_indices, return_index=True)
            result_index = MultiIndex(levels=list(unob_index.levels) + list(ob_index.levels), codes=result_index_codes, names=list(unob_index.names) + list(ob_index.names)).reorder_levels(index)
            ids = len(unob_index) * ob_ids + unob_ids
            if any(sorts):
                n_levels = len(sorts)
                drop_levels = [n_levels - idx for idx, sort in enumerate(reversed(sorts), 1) if not sort]
                if len(drop_levels) > 0:
                    sorter = result_index._drop_level_numbers(drop_levels).argsort()
                else:
                    sorter = result_index.argsort()
                result_index = result_index.take(sorter)
                _, index = np.unique(sorter, return_index=True)
                ids = ensure_platform_int(ids)
                ids = index.take(ids)
            else:
                ids, uniques = compress_group_index(ids, sort=False)
                ids = ensure_platform_int(ids)
                taker = np.concatenate([uniques, np.delete(np.arange(len(result_index)), uniques)])
                result_index = result_index.take(taker)
        return (result_index, ids)

    @property
    def observed_grouper(self) -> BaseGrouper:
        if all((ping._observed for ping in self.groupings)):
            return self
        return self._observed_grouper

    @cache_readonly
    def _observed_grouper(self) -> BaseGrouper:
        groupings = [ping.observed_grouping for ping in self.groupings]
        grouper = BaseGrouper(self.axis, groupings, sort=self._sort, dropna=self.dropna)
        return grouper

    def _ob_index_and_ids(self, levels: List[Index], codes: List[np.ndarray], names: List[Optional[Hashable]], sorts: List[bool]) -> Tuple[MultiIndex, np.ndarray]:
        consistent_sorting = all((sorts[0] == sort for sort in sorts[1:]))
        sort_in_compress = sorts[0] if consistent_sorting else False
        shape = tuple((len(level) for level in levels))
        group_index = get_group_index(codes, shape, sort=True, xnull=True)
        ob_ids, obs_group_ids = compress_group_index(group_index, sort=sort_in_compress)
        ob_ids = ensure_platform_int(ob_ids)
        ob_index_codes = decons_obs_group_ids(ob_ids, obs_group_ids, shape, codes, xnull=True)
        ob_index = MultiIndex(levels=levels, codes=ob_index_codes, names=names, verify_integrity=False)
        if not consistent_sorting and len(ob_index) > 0:
            n_levels = len(sorts)
            drop_levels = [n_levels - idx for idx, sort in enumerate(reversed(sorts), 1) if not sort]
            if len(drop_levels) > 0:
                sorter = ob_index._drop_level_numbers(drop_levels).argsort()
            else:
                sorter = ob_index.argsort()
            ob_index = ob_index.take(sorter)
            _, index = np.unique(sorter, return_index=True)
            ob_ids = np.where(ob_ids == -1, -1, index.take(ob_ids))
        ob_ids = ensure_platform_int(ob_ids)
        return (ob_index, ob_ids)

    def _unob_index_and_ids(self, levels: List[Index], codes: List[np.ndarray], names: List[Optional[Hashable]]) -> Tuple[MultiIndex, np.ndarray]:
        shape = tuple((len(level) for level in levels))
        unob_ids = get_group_index(codes, shape, sort=True, xnull=True)
        unob_index = MultiIndex.from_product(levels, names=names)
        unob_ids = ensure_platform_int(unob_ids)
        return (unob_index, unob_ids)

    @final
    def get_group_levels(self) -> Generator[Index, None, None]:
        result_index = self.result_index
        if len(self.groupings) == 1:
            yield result_index
        else:
            for level in range(result_index.nlevels - 1, -1, -1):
                yield result_index.get_level_values(level)

    @final
    def _cython_operation(self, kind: str, values: np.ndarray, how: str, axis: AxisInt, min_count: int = -1, **kwargs: Any) -> np.ndarray:
        """
        Returns the values of a cython operation.
        """
        assert kind in ['transform', 'aggregate']
        cy_op = WrappedCythonOp(kind=kind, how=how, has_dropped_na=self.has_dropped_na)
        return cy_op.cython_operation(values=values, axis=axis, min_count=min_count, comp_ids=self.ids, ngroups=self.ngroups, **kwargs)

    @final
    def agg_series(self, obj: Series, func: Callable[[Series], Any], preserve_dtype: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        obj : Series
        func : function taking a Series and returning a scalar-like
        preserve_dtype : bool
            Whether the aggregation is known to be dtype-preserving.

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if not isinstance(obj._values, np.ndarray):
            preserve_dtype = True
        result = self._aggregate_series_pure_python(obj, func)
        npvalues = lib.maybe_convert_objects(result, try_float=False)
        if preserve_dtype:
            out = maybe_cast_pointwise_result(npvalues, obj.dtype, numeric_only=True)
        else:
            out = npvalues
        return out

    @final
    def _aggregate_series_pure_python(self, obj: Series, func: Callable[[Series], Any]) -> np.ndarray:
        result = np.empty(self.ngroups, dtype='O')
        initialized = False
        splitter = self._get_splitter(obj)
        for i, group in enumerate(splitter):
            res = func(group)
            res = extract_result(res)
            if not initialized:
                check_result_array(res, group.dtype)
                initialized = True
            result[i] = res
        return result

    @final
    def apply_groupwise(self, f: Callable[[NDFrame], Any], data: NDFrame) -> Tuple[List[Any], bool]:
        mutated = False
        splitter = self._get_splitter(data)
        group_keys = self.result_index
        result_values = []
        zipped = zip(group_keys, splitter)
        for key, group in zipped:
            object.__setattr__(group, 'name', key)
            group_axes = group.axes
            res = f(group)
            if not mutated and (not _is_indexed_like(res, group_axes)):
                mutated = True
            result_values.append(res)
        if len(group_keys) == 0 and getattr(f, '__name__', None) in ['skew', 'kurt', 'sum', 'prod']:
            f(data.iloc[:0])
        return (result_values, mutated)

    @final
    @cache_readonly
    def _sorted_ids(self) -> np.ndarray:
        result = self.ids.take(self.result_ilocs)
        if getattr(self, 'dropna', True):
            result = result[result >= 0]
        return result

class BinGrouper(BaseGrouper):
    """
    This is an internal Grouper class

    Parameters
    ----------
    bins : the split index of binlabels to group the item of axis
    binlabels : the label list
    indexer : np.ndarray[np.intp], optional
        the indexer created by Grouper
        some groupers (TimeGrouper) will sort its axis and its
        group_info is also sorted, so need the indexer to reorder

    Examples
    --------
    bins: [2, 4, 6, 8, 10]
    binlabels: DatetimeIndex(['2005-01-01', '2005-01-03',
        '2005-01-05', '2005-01-07', '2005-01-09'],
        dtype='datetime64[ns]', freq='2D')

    the group_info, which contains the label of each item in grouped
    axis, the index of label in label list, group number, is

    (array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), array([0, 1, 2, 3, 4]), 5)

    means that, the grouped axis has 10 items, can be grouped into 5
    labels, the first and second items belong to the first label, the
    third and forth items belong to the second label, and so on

    """

    def __init__(self, bins: np.ndarray, binlabels: Index, indexer: Optional[np.ndarray] = None):
        self.bins = ensure_int64(bins)
        self.binlabels = ensure_index(binlabels)
        self.indexer = indexer
        assert len(self.binlabels) == len(self.bins)

    @cache_readonly
    def groups(self) -> Dict[Any, np.ndarray]:
        """dict {group name -> group labels}"""
        result = {key: value for key, value in zip(self.binlabels, self.bins) if key is not NaT}
        return result

    @property
    def nkeys(self) -> int:
        return 1

    @cache_readonly
    def codes_info(self) -> np.ndarray:
        ids = self.ids
        if self.indexer is not None:
            sorter = np.lexsort((ids, self.indexer))
            ids = ids[sorter]
        return ids

    def get_iterator(self, data: NDFrame) -> Generator[Tuple[Any, NDFrame], None, None]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        slicer = lambda start, edge: data.iloc[start:edge]
        start = 0
        for edge, label in zip(self.bins, self.binlabels):
            if label is not NaT:
                yield (label, slicer(start, edge))
            start = edge
        if start < len(data):
            yield (self.binlabels[-1], slicer(start, None))

    @cache_readonly
    def indices(self) -> Dict[Any, List[int]]:
        indices: Dict[Any, List[int]] = collections.defaultdict(list)
        i = 0
        for label, bin in zip(self.binlabels, self.bins):
            if i < bin:
                if label is not NaT:
                    indices[label] = list(range(i, bin))
                i = bin
        return indices

    @cache_readonly
    def codes(self) -> List[np.ndarray]:
        return [self.ids]

    @cache_readonly
    def result_index_and_ids(self) -> Tuple[Index, np.ndarray]:
        result_index = self.binlabels
        if len(self.binlabels) != 0 and isna(self.binlabels[0]):
            result_index = result_index[1:]
        ngroups = len(result_index)
        rep = np.diff(np.r_[0, self.bins])
        rep = ensure_platform_int(rep)
        if ngroups == len(self.bins):
            ids = np.repeat(np.arange(ngroups), rep)
        else:
            ids = np.repeat(np.r_[-1, np.arange(ngroups)], rep)
        ids = ensure_platform_int(ids)
        return (result_index, ids)

    @property
    def levels(self) -> List[Index]:
        return [self.binlabels]

    @property
    def names(self) -> List[Optional[Hashable]]:
        return [self.binlabels.name]

    @property
    def groupings(self) -> List[grouper.Grouping]:
        lev = self.binlabels
        codes = self.ids
        labels = lev.take(codes)
        ping = grouper.Grouping(labels, labels, in_axis=False, level=None, uniques=lev._values)
        return [ping]

    @property
    def observed_grouper(self) -> BinGrouper:
        return self

def _is_indexed_like(obj: Any, axes: List[Index]) -> bool:
    if isinstance(obj, Series):
        if len(axes) > 1:
            return False
        return obj.index.equals(axes[0])
    elif isinstance(obj, DataFrame):
        return obj.index.equals(axes[0])
    return False

class DataSplitter(Generic[NDFrameT]):

    def __init__(self, data: NDFrameT, ngroups: int, *, sort_idx: np.ndarray, sorted_ids: np.ndarray):
        self.data = data
        self.ngroups = ngroups
        self._slabels = sorted_ids
        self._sort_idx = sort_idx

    def __iter__(self) -> Iterator[NDFrameT]:
        if self.ngroups == 0:
            return
        starts, ends = lib.generate_slices(self._slabels, self.ngroups)
        sdata = self._sorted_data
        for start, end in zip(starts, ends):
            yield self._chop(sdata, slice(start, end))

    @cache_readonly
    def _sorted_data(self) -> NDFrameT:
        return self.data.take(self._sort_idx, axis=0)

    def _chop(self, sdata: NDFrameT, slice_obj: slice) -> NDFrameT:
        raise AbstractMethodError(self)

class SeriesSplitter(DataSplitter[Series]):

    def _chop(self, sdata: Series, slice_obj: slice) -> Series:
        mgr = sdata._mgr.get_slice(slice_obj)
        ser = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        ser._name = sdata.name
        return ser.__finalize__(sdata, method='groupby')

class FrameSplitter(DataSplitter[DataFrame]):

    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame:
        mgr = sdata._mgr.get_slice(slice_obj, axis=1)
        df = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(sdata, method='groupby')
