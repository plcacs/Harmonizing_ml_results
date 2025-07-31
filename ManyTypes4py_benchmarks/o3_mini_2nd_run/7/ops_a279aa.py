#!/usr/bin/env python3
"""
Provide classes to perform the groupby aggregate operations.

These are not exposed to the user and provide implementations of the grouping
operations, primarily in cython. These classes (BaseGrouper and BinGrouper)
are contained *in* the SeriesGroupBy and DataFrameGroupBy objects.
"""
from __future__ import annotations
import collections
import functools
from typing import Any, Callable, Dict, Generator, Iterator, List, Sequence, Tuple, TypeVar, Union
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

if False:
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
    _CYTHON_FUNCTIONS: Dict[str, Dict[str, Any]] = {
        'aggregate': {
            'any': functools.partial(libgroupby.group_any_all, val_test='any'),
            'all': functools.partial(libgroupby.group_any_all, val_test='all'),
            'sum': 'group_sum',
            'prod': 'group_prod',
            'idxmin': functools.partial(libgroupby.group_idxmin_idxmax, name='idxmin'),
            'idxmax': functools.partial(libgroupby.group_idxmin_idxmax, name='idxmax'),
            'min': 'group_min',
            'max': 'group_max',
            'mean': 'group_mean',
            'median': 'group_median_float64',
            'var': 'group_var',
            'std': functools.partial(libgroupby.group_var, name='std'),
            'sem': functools.partial(libgroupby.group_var, name='sem'),
            'skew': 'group_skew',
            'kurt': 'group_kurt',
            'first': 'group_nth',
            'last': 'group_last',
            'ohlc': 'group_ohlc'
        },
        'transform': {
            'cumprod': 'group_cumprod',
            'cumsum': 'group_cumsum',
            'cummin': 'group_cummin',
            'cummax': 'group_cummax',
            'rank': 'group_rank'
        }
    }
    _cython_arity: Dict[str, int] = {'ohlc': 4}

    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None:
        self.kind = kind
        self.how = how
        self.has_dropped_na = has_dropped_na

    @classmethod
    def get_kind_from_how(cls, how: str) -> str:
        if how in cls._CYTHON_FUNCTIONS['aggregate']:
            return 'aggregate'
        return 'transform'

    @classmethod
    @functools.cache
    def _get_cython_function(cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool) -> Callable:
        dtype_str: str = dtype.name
        ftype: Any = cls._CYTHON_FUNCTIONS[kind][how]
        if callable(ftype):
            f: Any = ftype
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
        how: str = self.how
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
        how: str = self.how
        kind: str = self.kind
        arity: int = self._cython_arity.get(how, 1)
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
        how: str = self.how
        if how == 'rank':
            out_dtype = np.dtype('float64')
        elif how in ['idxmin', 'idxmax']:
            out_dtype = np.dtype('intp')
        elif dtype.kind in 'iufcb':
            out_dtype = np.dtype(f'{dtype.kind}{dtype.itemsize}')
        else:
            out_dtype = np.dtype('object')
        return out_dtype

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
        how: str = self.how
        if how in ['sum', 'cumsum', 'prod', 'cumprod']:
            if dtype == np.dtype(bool):
                return np.dtype(np.int64)
        elif how in ['mean', 'median', 'var', 'std', 'sem']:
            if dtype.kind in 'fc':
                return dtype
            elif dtype.kind in 'iub':
                return np.dtype(np.float64)
        return dtype

    def _cython_op_ndim_compat(
        self,
        values: np.ndarray,
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: Union[np.ndarray, None] = None,
        result_mask: Union[np.ndarray, None] = None,
        **kwargs: Any
    ) -> np.ndarray:
        if values.ndim == 1:
            values2d: np.ndarray = values[None, :]
            if mask is not None:
                mask = mask[None, :]
            if result_mask is not None:
                result_mask = result_mask[None, :]
            res: np.ndarray = self._call_cython_op(values2d, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=mask, result_mask=result_mask, **kwargs)
            if res.shape[0] == 1:
                return res[0]
            return res.T
        return self._call_cython_op(values, min_count=min_count, ngroups=ngroups, comp_ids=comp_ids, mask=mask, result_mask=result_mask, **kwargs)

    def _call_cython_op(
        self,
        values: np.ndarray,
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: Union[np.ndarray, None],
        result_mask: Union[np.ndarray, None],
        **kwargs: Any
    ) -> np.ndarray:
        orig_values: np.ndarray = values
        dtype: np.dtype = values.dtype
        is_numeric: bool = dtype.kind in 'iufcb'
        is_datetimelike: bool = dtype.kind in 'mM'
        if self.how in ['any', 'all']:
            if mask is None:
                mask = isna(values)
        if is_datetimelike:
            values = values.view('int64')
            is_numeric = True
        elif dtype.kind == 'b':
            values = values.view('uint8')
        if values.dtype == np.dtype('float16'):
            values = values.astype(np.float32)
        if self.how in ['any', 'all']:
            if dtype == np.dtype(object):
                if kwargs.get('skipna', False):
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
        out_shape: Tuple[int, ...] = self._get_output_shape(ngroups, values)
        func: Callable = self._get_cython_function(self.kind, self.how, values.dtype, is_numeric)
        values = self._get_cython_vals(values)
        out_dtype: np.dtype = self._get_out_dtype(values.dtype)
        result: np.ndarray = maybe_fill(np.empty(out_shape, dtype=out_dtype))
        if self.kind == 'aggregate':
            counts: np.ndarray = np.zeros(ngroups, dtype=np.int64)
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
                if dtype == np.dtype(object):
                    result = result.astype(object)
            else:
                raise NotImplementedError(f'{self.how} is not implemented')
        else:
            if self.how != 'rank':
                kwargs['result_mask'] = result_mask
            func(out=result, values=values, labels=comp_ids, ngroups=ngroups, is_datetimelike=is_datetimelike, mask=mask, **kwargs)
        if self.kind == 'aggregate' and self.how not in ['idxmin', 'idxmax']:
            if result.dtype.kind in 'iu' and (not is_datetimelike):
                cutoff: int = max(0 if self.how in ['sum', 'prod'] else 1, min_count)
                empty_groups: np.ndarray = counts < cutoff
                if empty_groups.any():
                    if result_mask is not None:
                        assert result_mask[empty_groups].all()
                    else:
                        result = result.astype('float64')
                        result[empty_groups] = np.nan
        result = result.T
        if self.how not in self.cast_blocklist:
            res_dtype: np.dtype = self._get_result_dtype(orig_values.dtype)
            op_result: np.ndarray = maybe_downcast_to_dtype(result, res_dtype)
        else:
            op_result = result
        return op_result

    def _validate_axis(self, axis: AxisInt, values: np.ndarray) -> None:
        if values.ndim > 2:
            raise NotImplementedError('number of dimensions is currently limited to 2')
        if values.ndim == 2:
            assert axis == 1, axis
        elif not is_1d_only_ea_dtype(values.dtype):
            assert axis == 0

    def cython_operation(
        self,
        *,
        values: np.ndarray,
        axis: AxisInt,
        min_count: int = -1,
        comp_ids: np.ndarray,
        ngroups: int,
        **kwargs: Any
    ) -> Any:
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
    groupings : Sequence[Any]
        all the grouping instances to handle in this grouper
        for example for grouper list to groupby, need to pass the list
    sort : bool, default True
        whether this grouper will give sorted result or not

    """
    def __init__(self, axis: Index, groupings: Sequence[Any], sort: bool = True, dropna: bool = True) -> None:
        assert isinstance(axis, Index), axis
        self.axis: Index = axis
        self._groupings: Sequence[Any] = groupings
        self._sort: bool = sort
        self.dropna: bool = dropna

    @property
    def groupings(self) -> Sequence[Any]:
        return self._groupings

    def __iter__(self) -> Iterator[Any]:
        return iter(self.indices)

    @property
    def nkeys(self) -> int:
        return len(self.groupings)

    def get_iterator(self, data: NDFrameT) -> Generator[Tuple[Any, NDFrameT], None, None]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        splitter = self._get_splitter(data)
        keys: Index = self.result_index
        yield from zip(keys, splitter)

    def _get_splitter(self, data: NDFrameT) -> Generator[NDFrameT, None, None]:
        """
        Returns
        -------
        Generator yielding subsetted objects
        """
        if isinstance(data, Series):
            klass: type[DataSplitter[Series]] = SeriesSplitter  # type: ignore
        else:
            klass = FrameSplitter
        return klass(data, self.ngroups, sorted_ids=self._sorted_ids, sort_idx=self.result_ilocs)

    @cache_readonly
    def indices(self) -> Dict[Any, Any]:
        """dict {group name -> group indices}"""
        if len(self.groupings) == 1 and isinstance(self.result_index, CategoricalIndex):
            return self.groupings[0].indices
        codes_list: List[np.ndarray] = [ping.codes for ping in self.groupings]  # type: ignore
        return get_indexer_dict(codes_list, self.levels)

    @cache_readonly
    def result_ilocs(self) -> np.ndarray:
        """
        Get the original integer locations of result_index in the input.
        """
        ids: np.ndarray = self.ids
        if self.has_dropped_na:
            mask = np.where(ids >= 0)
            null_gaps = np.cumsum(ids == -1)[mask]
            ids = ids[mask]
        result: np.ndarray = get_group_index_sorter(ids, self.ngroups)
        if self.has_dropped_na:
            result += np.take(null_gaps, result)
        return result

    @property
    def codes(self) -> List[np.ndarray]:
        return [ping.codes for ping in self.groupings]  # type: ignore

    @property
    def levels(self) -> List[Index]:
        if len(self.groupings) > 1:
            return list(self.result_index.levels)  # type: ignore
        else:
            return [self.result_index]

    @property
    def names(self) -> List[str]:
        return [ping.name for ping in self.groupings]  # type: ignore

    @cache_readonly
    def is_monotonic(self) -> bool:
        return Index(self.ids).is_monotonic_increasing

    @cache_readonly
    def groups(self) -> Dict[Any, Any]:
        """dict {group name -> group labels}"""
        if len(self.groupings) == 1:
            return self.groupings[0].groups  # type: ignore
        result_index, ids = self.result_index_and_ids
        values = result_index._values
        categories = Categorical(ids, categories=range(len(result_index)))
        result: Dict[Any, Any] = {values[group]: self.axis.take(axis_ilocs) for group, axis_ilocs in categories._reverse_indexer().items()}
        return result

    @cache_readonly
    def codes_info(self) -> np.ndarray:
        return self.ids

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
        levels: List[Index] = [Index._with_infer(ping.uniques) for ping in self.groupings]  # type: ignore
        obs: List[bool] = [ping._observed or not ping._passed_categorical for ping in self.groupings]  # type: ignore
        sorts: List[bool] = [ping._sort for ping in self.groupings]  # type: ignore
        for k, (ping, level) in enumerate(zip(self.groupings, levels)):
            if ping._passed_categorical:  # type: ignore
                levels[k] = level.set_categories(ping._orig_cats)  # type: ignore
        if len(self.groupings) == 1:
            result_index: Index = levels[0]
            result_index.name = self.names[0]
            ids: np.ndarray = ensure_platform_int(self.codes[0])
        elif all(obs):
            result_index, ids = self._ob_index_and_ids(levels, self.codes, self.names, sorts)
        elif not any(obs):
            result_index, ids = self._unob_index_and_ids(levels, self.codes, self.names)
        else:
            names = self.names
            codes: List[np.ndarray] = [ping.codes for ping in self.groupings]  # type: ignore
            ob_indices: List[int] = [idx for idx, ob in enumerate(obs) if ob]
            unob_indices: List[int] = [idx for idx, ob in enumerate(obs) if not ob]
            ob_index, ob_ids = self._ob_index_and_ids(
                levels=[levels[idx] for idx in ob_indices],
                codes=[codes[idx] for idx in ob_indices],
                names=[names[idx] for idx in ob_indices],
                sorts=[sorts[idx] for idx in ob_indices]
            )
            unob_index, unob_ids = self._unob_index_and_ids(
                levels=[levels[idx] for idx in unob_indices],
                codes=[codes[idx] for idx in unob_indices],
                names=[names[idx] for idx in unob_indices]
            )
            result_index_codes = np.concatenate([np.tile(unob_index.codes, len(ob_index)), np.repeat(ob_index.codes, len(unob_index), axis=1)], axis=0)  # type: ignore
            _, index = np.unique(unob_indices + ob_indices, return_index=True)
            result_index = MultiIndex(levels=list(unob_index.levels) + list(ob_index.levels), codes=result_index_codes, names=list(unob_index.names) + list(ob_index.names)).reorder_levels(index)  # type: ignore
            ids = len(unob_index) * ob_ids + unob_ids  # type: ignore
            if any(sorts):
                n_levels: int = len(sorts)
                drop_levels: List[int] = [n_levels - idx for idx, sort in enumerate(reversed(sorts), 1) if not sort]
                if len(drop_levels) > 0:
                    sorter = result_index._drop_level_numbers(drop_levels).argsort()  # type: ignore
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
        if all((ping._observed for ping in self.groupings)):  # type: ignore
            return self
        return self._observed_grouper

    @cache_readonly
    def _observed_grouper(self) -> BaseGrouper:
        groupings: List[Any] = [ping.observed_grouping for ping in self.groupings]  # type: ignore
        grouper = BaseGrouper(self.axis, groupings, sort=self._sort, dropna=self.dropna)
        return grouper

    def _ob_index_and_ids(
        self,
        levels: List[Index],
        codes: List[np.ndarray],
        names: List[str],
        sorts: List[bool]
    ) -> Tuple[MultiIndex, np.ndarray]:
        consistent_sorting: bool = all((sorts[0] == sort for sort in sorts[1:]))
        sort_in_compress: bool = sorts[0] if consistent_sorting else False
        shape: Tuple[int, ...] = tuple((len(level) for level in levels))
        group_index: np.ndarray = get_group_index(codes, shape, sort=True, xnull=True)
        ob_ids, obs_group_ids = compress_group_index(group_index, sort=sort_in_compress)
        ob_ids = ensure_platform_int(ob_ids)
        ob_index_codes = decons_obs_group_ids(ob_ids, obs_group_ids, shape, codes, xnull=True)
        ob_index: MultiIndex = MultiIndex(levels=levels, codes=ob_index_codes, names=names, verify_integrity=False)
        if not consistent_sorting and len(ob_index) > 0:
            n_levels: int = len(sorts)
            drop_levels: List[int] = [n_levels - idx for idx, sort in enumerate(reversed(sorts), 1) if not sort]
            if len(drop_levels) > 0:
                sorter = ob_index._drop_level_numbers(drop_levels).argsort()  # type: ignore
            else:
                sorter = ob_index.argsort()
            ob_index = ob_index.take(sorter)
            _, index = np.unique(sorter, return_index=True)
            ob_ids = np.where(ob_ids == -1, -1, index.take(ob_ids))
        ob_ids = ensure_platform_int(ob_ids)
        return (ob_index, ob_ids)

    def _unob_index_and_ids(
        self,
        levels: List[Index],
        codes: List[np.ndarray],
        names: List[str]
    ) -> Tuple[MultiIndex, np.ndarray]:
        shape: Tuple[int, ...] = tuple((len(level) for level in levels))
        unob_ids: np.ndarray = get_group_index(codes, shape, sort=True, xnull=True)
        unob_index: MultiIndex = MultiIndex.from_product(levels, names=names)
        unob_ids = ensure_platform_int(unob_ids)
        return (unob_index, unob_ids)

    def get_group_levels(self) -> Generator[Index, None, None]:
        result_index: Index = self.result_index
        if len(self.groupings) == 1:
            yield result_index
        else:
            for level in range(result_index.nlevels - 1, -1, -1):
                yield result_index.get_level_values(level)

    def _cython_operation(
        self,
        kind: str,
        values: np.ndarray,
        how: str,
        axis: AxisInt,
        min_count: int = -1,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Returns the values of a cython operation.
        """
        assert kind in ['transform', 'aggregate']
        cy_op: WrappedCythonOp = WrappedCythonOp(kind=kind, how=how, has_dropped_na=self.has_dropped_na)
        return cy_op.cython_operation(values=values, axis=axis, min_count=min_count, comp_ids=self.ids, ngroups=self.ngroups, **kwargs)

    def agg_series(self, obj: Series, func: Callable[[Series], Any], preserve_dtype: bool = False) -> Union[np.ndarray, ArrayLike]:
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
        result: np.ndarray = self._aggregate_series_pure_python(obj, func)
        npvalues: Any = lib.maybe_convert_objects(result, try_float=False)
        if preserve_dtype:
            out: Any = maybe_cast_pointwise_result(npvalues, obj.dtype, numeric_only=True)
        else:
            out = npvalues
        return out

    def _aggregate_series_pure_python(self, obj: Series, func: Callable[[Series], Any]) -> np.ndarray:
        result: np.ndarray = np.empty(self.ngroups, dtype='O')
        initialized: bool = False
        splitter: Generator[Series, None, None] = self._get_splitter(obj)
        for i, group in enumerate(splitter):
            res: Any = func(group)
            res = extract_result(res)
            if not initialized:
                check_result_array(res, group.dtype)
                initialized = True
            result[i] = res
        return result

    def apply_groupwise(self, f: Callable[[NDFrameT], Any], data: NDFrameT) -> Tuple[List[Any], bool]:
        mutated: bool = False
        splitter = self._get_splitter(data)
        group_keys: Index = self.result_index
        result_values: List[Any] = []
        for key, group in zip(group_keys, splitter):
            object.__setattr__(group, 'name', key)
            group_axes = group.axes
            res: Any = f(group)
            if not mutated and (not _is_indexed_like(res, group_axes)):
                mutated = True
            result_values.append(res)
        if len(group_keys) == 0 and getattr(f, '__name__', None) in ['skew', 'kurt', 'sum', 'prod']:
            f(data.iloc[:0])
        return (result_values, mutated)

    @cache_readonly
    def _sorted_ids(self) -> np.ndarray:
        result: np.ndarray = self.ids.take(self.result_ilocs)
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
    indexer : Optional[np.ndarray]
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
    def __init__(self, bins: ArrayLike, binlabels: ArrayLike, indexer: Union[np.ndarray, None] = None) -> None:
        self.bins: np.ndarray = ensure_int64(bins)
        self.binlabels: Index = ensure_index(binlabels)
        self.indexer: Union[np.ndarray, None] = indexer
        assert len(self.binlabels) == len(self.bins)

    @cache_readonly
    def groups(self) -> Dict[Any, Any]:
        """dict {group name -> group labels}"""
        result: Dict[Any, Any] = {key: value for key, value in zip(self.binlabels, self.bins) if key is not NaT}
        return result

    @property
    def nkeys(self) -> int:
        return 1

    @cache_readonly
    def codes_info(self) -> np.ndarray:
        ids: np.ndarray = self.ids
        if self.indexer is not None:
            sorter: np.ndarray = np.lexsort((ids, self.indexer))
            ids = ids[sorter]
        return ids

    def get_iterator(self, data: DataFrame | Series) -> Generator[Tuple[Any, DataFrame | Series], None, None]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        slicer: Callable[[int, Union[int, None]], DataFrame | Series] = lambda start, edge: data.iloc[start:edge]
        start: int = 0
        for edge, label in zip(self.bins, self.binlabels):
            if label is not NaT:
                yield (label, slicer(start, edge))
            start = edge
        if start < len(data):
            yield (self.binlabels[-1], slicer(start, None))

    @cache_readonly
    def indices(self) -> Dict[Any, List[int]]:
        indices: Dict[Any, List[int]] = collections.defaultdict(list)
        i: int = 0
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
        result_index: Index = self.binlabels
        if len(self.binlabels) != 0 and isna(self.binlabels[0]):
            result_index = result_index[1:]
        ngroups: int = len(result_index)
        rep: np.ndarray = np.diff(np.r_[0, self.bins])
        rep = ensure_platform_int(rep)
        if ngroups == len(self.bins):
            ids: np.ndarray = np.repeat(np.arange(ngroups), rep)
        else:
            ids = np.repeat(np.r_[-1, np.arange(ngroups)], rep)
        ids = ensure_platform_int(ids)
        return (result_index, ids)

    @property
    def levels(self) -> List[Index]:
        return [self.binlabels]

    @property
    def names(self) -> List[str]:
        return [self.binlabels.name]

    @property
    def groupings(self) -> List[Any]:
        lev: Index = self.binlabels
        codes: np.ndarray = self.ids
        labels: Index = lev.take(codes)
        ping: Any = grouper.Grouping(labels, labels, in_axis=False, level=None, uniques=lev._values)
        return [ping]

    @property
    def observed_grouper(self) -> BinGrouper:
        return self

def _is_indexed_like(obj: Any, axes: Sequence[Any]) -> bool:
    if isinstance(obj, Series):
        if len(axes) > 1:
            return False
        return obj.index.equals(axes[0])
    elif isinstance(obj, DataFrame):
        return obj.index.equals(axes[0])
    return False

T = TypeVar("T", bound=NDFrameT)

class DataSplitter(Generic[T]):
    def __init__(self, data: T, ngroups: int, *, sort_idx: np.ndarray, sorted_ids: np.ndarray) -> None:
        self.data: T = data
        self.ngroups: int = ngroups
        self._slabels: np.ndarray = sorted_ids
        self._sort_idx: np.ndarray = sort_idx

    def __iter__(self) -> Iterator[T]:
        if self.ngroups == 0:
            return
        starts, ends = lib.generate_slices(self._slabels, self.ngroups)
        sdata: T = self._sorted_data
        for start, end in zip(starts, ends):
            yield self._chop(sdata, slice(start, end))

    @cache_readonly
    def _sorted_data(self) -> T:
        return self.data.take(self._sort_idx, axis=0)

    def _chop(self, sdata: T, slice_obj: slice) -> T:
        raise AbstractMethodError(self)

class SeriesSplitter(DataSplitter[Series]):
    def _chop(self, sdata: Series, slice_obj: slice) -> Series:
        mgr = sdata._mgr.get_slice(slice_obj)
        ser: Series = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        ser._name = sdata.name
        return ser.__finalize__(sdata, method='groupby')

class FrameSplitter(DataSplitter[DataFrame]):
    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame:
        mgr = sdata._mgr.get_slice(slice_obj, axis=1)
        df: DataFrame = sdata._constructor_from_mgr(mgr, axes=mgr.axes)
        return df.__finalize__(sdata, method='groupby')