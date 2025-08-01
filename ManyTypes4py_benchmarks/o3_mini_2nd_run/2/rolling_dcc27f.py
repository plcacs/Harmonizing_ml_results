#!/usr/bin/env python3
"""
Provide a generic structure to support window functions,
similar to how we have a Groupby object.
"""

from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union, overload
from typing import Literal
import numpy as np
from pandas._libs.tslibs import BaseOffset, Timedelta, to_offset
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import Appender, Substitution, doc
from pandas.core.dtypes.common import ensure_float64, is_bool, is_integer, is_numeric_dtype, needs_i8_conversion
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply, reconstruct_func
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import BaseIndexer, FixedWindowIndexer, GroupbyIndexer, VariableWindowIndexer
from pandas.core.indexes.api import DatetimeIndex, Index, MultiIndex, PeriodIndex, TimedeltaIndex
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import get_jit_arguments, maybe_use_numba, prepare_function_arguments
from pandas.core.window.common import flex_binary_moment, zsqrt
from pandas.core.window.doc import _shared_docs, create_section_header, kwargs_numeric_only, kwargs_scipy, numba_notes, template_header, template_pipe, template_returns, template_see_also, window_agg_numba_parameters, window_apply_parameters
from pandas.core.window.numba_ import generate_manual_numpy_nan_agg_with_axis, generate_numba_apply_func, generate_numba_table_func
if TYPE_CHECKING:
    from collections.abc import Hashable, Sized
    from pandas._typing import ArrayLike, NDFrameT, QuantileInterpolation, WindowingRankType
    from pandas import DataFrame, Series
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.ops import BaseGrouper
from pandas.core.arrays.datetimelike import dtype_to_unit


class BaseWindow(SelectionMixin):
    """Provides utilities for performing windowing operations."""
    _attributes: List[str] = []
    exclusions = frozenset()

    def __init__(self, 
                 obj: Union[ABCSeries, ABCDataFrame],
                 window: Optional[Union[int, BaseIndexer, timedelta, str, BaseOffset]] = None,
                 min_periods: Optional[int] = None, 
                 center: bool = False, 
                 win_type: Optional[str] = None, 
                 on: Optional[Union[str, Index]] = None, 
                 closed: Optional[Literal['right', 'both', 'left', 'neither']] = None, 
                 step: Optional[int] = None, 
                 method: Literal['single', 'table'] = 'single', 
                 *, 
                 selection: Optional[Any] = None) -> None:
        self.obj: Union[ABCSeries, ABCDataFrame] = obj
        self.on: Optional[Union[str, Index]] = on
        self.closed: Optional[Literal['right', 'both', 'left', 'neither']] = closed
        self.step: Optional[int] = step
        self.window: Optional[Union[int, BaseIndexer, timedelta, str, BaseOffset]] = window
        self.min_periods: Optional[int] = min_periods
        self.center: bool = center
        self.win_type: Optional[str] = win_type
        self.method: Literal['single', 'table'] = method
        self._win_freq_i8: Optional[Union[int, float]] = None
        if self.on is None:
            self._on: Index = self.obj.index  # type: ignore
        elif isinstance(self.on, Index):
            self._on = self.on
        elif isinstance(self.obj, ABCDataFrame) and self.on in self.obj.columns:
            from pandas import Index as pdIndex  # local import
            self._on = pdIndex(self.obj[self.on])
        else:
            raise ValueError(f'invalid on specified as {self.on}, must be a column (of DataFrame), an Index or None')
        self._selection: Optional[Any] = selection
        self._validate()

    def _validate(self) -> None:
        if self.center is not None and (not is_bool(self.center)):
            raise ValueError('center must be a boolean')
        if self.min_periods is not None:
            if not is_integer(self.min_periods):
                raise ValueError('min_periods must be an integer')
            if self.min_periods < 0:
                raise ValueError('min_periods must be >= 0')
            if is_integer(self.window) and self.min_periods > self.window:  # type: ignore
                raise ValueError(f'min_periods {self.min_periods} must be <= window {self.window}')
        if self.closed is not None and self.closed not in ['right', 'both', 'left', 'neither']:
            raise ValueError("closed must be 'right', 'left', 'both' or 'neither'")
        if not isinstance(self.obj, (ABCSeries, ABCDataFrame)):
            raise TypeError(f'invalid type: {type(self.obj)}')
        if isinstance(self.window, BaseIndexer):
            get_window_bounds_signature = inspect.signature(self.window.get_window_bounds).parameters.keys()  # type: ignore
            expected_signature = inspect.signature(BaseIndexer().get_window_bounds).parameters.keys()
            if get_window_bounds_signature != expected_signature:
                raise ValueError(f'{type(self.window).__name__} does not implement the correct signature for get_window_bounds')
        if self.method not in ['table', 'single']:
            raise ValueError("method must be 'table' or 'single")
        if self.step is not None:
            if not is_integer(self.step):
                raise ValueError('step must be an integer')
            if self.step < 0:
                raise ValueError('step must be >= 0')

    def _check_window_bounds(self, start: Sequence[int], end: Sequence[int], num_vals: int) -> None:
        if len(start) != len(end):
            raise ValueError(f'start ({len(start)}) and end ({len(end)}) bounds must be the same length')
        expected_length = (num_vals + (self.step or 1) - 1) // (self.step or 1)
        if len(start) != expected_length:
            raise ValueError(f'start and end bounds ({len(start)}) must be the same length as the object ({num_vals}) divided by the step ({self.step}) if given and rounded up')

    def _slice_axis_for_step(self, index: Sequence[Any], result: Optional[Sequence[Any]] = None) -> Sequence[Any]:
        """
        Slices the index for a given result and the preset step.
        """
        if result is None or len(result) == len(index):
            return index
        else:
            return index[::self.step]  # type: ignore

    def _validate_numeric_only(self, name: str, numeric_only: bool) -> None:
        """
        Validate numeric_only argument, raising if invalid for the input.
        """
        if getattr(self, '_selected_obj', self.obj).ndim == 1 and numeric_only and (not is_numeric_dtype(getattr(self, '_selected_obj', self.obj).dtype)):
            raise NotImplementedError(f'{type(self).__name__}.{name} does not implement numeric_only')

    def _make_numeric_only(self, obj: DataFrame) -> DataFrame:
        """Subset DataFrame to numeric columns.
        """
        result = obj.select_dtypes(include=['number'], exclude=['timedelta'])
        return result

    def _create_data(self, obj: Union[ABCSeries, DataFrame], numeric_only: bool = False) -> Union[ABCSeries, DataFrame]:
        """
        Split data into blocks & return conformed data.
        """
        if self.on is not None and (not isinstance(self.on, Index)) and (obj.ndim == 2):
            obj = obj.reindex(columns=obj.columns.difference([self.on], sort=False))
        if obj.ndim > 1 and numeric_only:
            obj = self._make_numeric_only(obj)  # type: ignore
        return obj

    def _gotitem(self, key: Union[str, List[Any]], ndim: int, subset: Optional[Any] = None) -> BaseWindow:
        """
        Sub-classes to define. Return a sliced object.
        """
        if subset is None:
            subset = self.obj
        kwargs: Dict[str, Any] = {attr: getattr(self, attr) for attr in self._attributes}
        selection = self._infer_selection(key, subset)  # type: ignore
        new_win = type(self)(subset, selection=selection, **kwargs)
        return new_win

    def __getattr__(self, attr: str) -> Any:
        if attr in self._internal_names_set:  # type: ignore
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def _dir_additions(self) -> List[str]:
        return self.obj._dir_additions()  # type: ignore

    def __repr__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
        attrs_list = (f'{attr_name}={getattr(self, attr_name)}' 
                      for attr_name in self._attributes 
                      if getattr(self, attr_name, None) is not None and attr_name[0] != '_')
        attrs = ','.join(attrs_list)
        return f'{type(self).__name__} [{attrs}]'

    def __iter__(self) -> Iterator[Any]:
        obj = self._selected_obj.set_axis(self._on)  # type: ignore
        obj = self._create_data(obj)
        indexer = self._get_window_indexer()
        start, end = indexer.get_window_bounds(num_values=len(obj), min_periods=self.min_periods, center=self.center, closed=self.closed, step=self.step)  # type: ignore
        self._check_window_bounds(start, end, len(obj))
        for s, e in zip(start, end):
            result = obj.iloc[s:e]
            yield result

    def _prep_values(self, values: np.ndarray) -> np.ndarray:
        """Convert input to numpy arrays for Cython routines"""
        if needs_i8_conversion(values.dtype):
            raise NotImplementedError(f'ops for {type(self).__name__} for this dtype {values.dtype} are not implemented')
        try:
            if isinstance(values, ExtensionArray):
                values = values.to_numpy(np.float64, na_value=np.nan)
            else:
                values = ensure_float64(values)
        except (ValueError, TypeError) as err:
            raise TypeError(f'cannot handle this type -> {values.dtype}') from err
        inf = np.isinf(values)
        if inf.any():
            values = np.where(inf, np.nan, values)
        return values

    def _insert_on_column(self, result: DataFrame, obj: DataFrame) -> None:
        from pandas import Series
        if self.on is not None and (not self._on.equals(obj.index)):
            name = self._on.name
            extra_col = Series(self._on, index=self.obj.index, name=name, copy=False)
            if name in result.columns:
                result[name] = extra_col
            elif name in result.index.names:
                pass
            elif name in getattr(self, '_selected_obj', obj).columns:
                old_cols = getattr(self, '_selected_obj', obj).columns
                new_cols = result.columns
                old_loc = old_cols.get_loc(name)
                overlap = new_cols.intersection(old_cols[:old_loc])
                new_loc = len(overlap)
                result.insert(new_loc, name, extra_col)
            else:
                result[name] = extra_col

    @property
    def _index_array(self) -> Optional[np.ndarray]:
        if isinstance(self._on, (PeriodIndex, DatetimeIndex, TimedeltaIndex)):
            return self._on.asi8
        elif isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in 'mM':
            return self._on.to_numpy(dtype=np.int64)
        return None

    def _resolve_output(self, out: DataFrame, obj: DataFrame) -> DataFrame:
        """Validate and finalize result."""
        if out.shape[1] == 0 and obj.shape[1] > 0:
            raise DataError('No numeric types to aggregate')
        if out.shape[1] == 0:
            return obj.astype('float64')
        self._insert_on_column(out, obj)
        return out

    def _get_window_indexer(self) -> Union[FixedWindowIndexer, VariableWindowIndexer, BaseIndexer]:
        """
        Return an indexer class that will compute the window start and end bounds
        """
        if isinstance(self.window, BaseIndexer):
            return self.window
        if self._win_freq_i8 is not None:
            return VariableWindowIndexer(index_array=self._index_array, window_size=self._win_freq_i8, center=self.center)
        return FixedWindowIndexer(window_size=self.window)  # type: ignore

    def _apply_series(self, homogeneous_func: Callable[[np.ndarray], np.ndarray], name: Optional[str] = None) -> Any:
        obj = self._create_data(getattr(self, '_selected_obj', self.obj))
        if name == 'count':
            obj = notna(obj).astype(int)
        try:
            values = self._prep_values(obj._values)  # type: ignore
        except (TypeError, NotImplementedError) as err:
            raise DataError('No numeric types to aggregate') from err
        result = homogeneous_func(values)
        index = self._slice_axis_for_step(obj.index, result)
        return obj._constructor(result, index=index, name=obj.name)  # type: ignore

    def _apply_columnwise(self, homogeneous_func: Callable[[np.ndarray], np.ndarray], name: str, numeric_only: bool = False) -> Any:
        self._validate_numeric_only(name, numeric_only)
        if getattr(self, '_selected_obj', self.obj).ndim == 1:
            return self._apply_series(homogeneous_func, name)
        obj = self._create_data(getattr(self, '_selected_obj', self.obj), numeric_only)
        if name == 'count':
            obj = notna(obj).astype(int)
            obj._mgr = obj._mgr.consolidate()  # type: ignore
        taker: List[int] = []
        res_values: List[np.ndarray] = []
        for i, arr in enumerate(obj._iter_column_arrays()):  # type: ignore
            try:
                arr = self._prep_values(arr)
            except (TypeError, NotImplementedError) as err:
                raise DataError(f'Cannot aggregate non-numeric type: {arr.dtype}') from err
            res = homogeneous_func(arr)
            res_values.append(res)
            taker.append(i)
        index = self._slice_axis_for_step(obj.index, res_values[0] if len(res_values) > 0 else None)
        df = type(obj)._from_arrays(res_values, index=index, columns=obj.columns.take(taker), verify_integrity=False)  # type: ignore
        return self._resolve_output(df, obj)

    def _apply_tablewise(self, homogeneous_func: Callable[[np.ndarray], np.ndarray], name: Optional[str] = None, numeric_only: bool = False) -> Any:
        if getattr(self, '_selected_obj', self.obj).ndim == 1:
            raise ValueError("method='table' not applicable for Series objects.")
        obj = self._create_data(getattr(self, '_selected_obj', self.obj), numeric_only)
        values = self._prep_values(obj.to_numpy())
        result = homogeneous_func(values)
        index = self._slice_axis_for_step(obj.index, result)
        columns = obj.columns if result.shape[1] == len(obj.columns) else obj.columns[::self.step]  # type: ignore
        out = obj._constructor(result, index=index, columns=columns)  # type: ignore
        return self._resolve_output(out, obj)

    def _apply_pairwise(self, target: Union[ABCSeries, DataFrame], other: Optional[Union[ABCSeries, DataFrame]], pairwise: Optional[bool], func: Callable, numeric_only: bool) -> Any:
        target = self._create_data(target, numeric_only)
        if other is None:
            other = target
            pairwise = True if pairwise is None else pairwise
        elif not isinstance(other, (ABCDataFrame, ABCSeries)):
            raise ValueError('other must be a DataFrame or Series')
        elif other.ndim == 2 and numeric_only:
            other = self._make_numeric_only(other)  # type: ignore
        return flex_binary_moment(target, other, func, pairwise=bool(pairwise))

    def _apply(self, func: Callable, name: str, numeric_only: bool = False, numba_args: Tuple[Any, ...] = (), **kwargs: Any) -> Any:
        """
        Rolling statistical measure using supplied function.
        """
        window_indexer = self._get_window_indexer()
        min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size  # type: ignore

        def homogeneous_func(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return values.copy()
            def calc(x: np.ndarray) -> np.ndarray:
                start, end = window_indexer.get_window_bounds(num_values=len(x), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)  # type: ignore
                self._check_window_bounds(start, end, len(x))
                return func(x, start, end, min_periods, *numba_args)
            with np.errstate(all='ignore'):
                result = calc(values)
            return result
        if self.method == 'single':
            return self._apply_columnwise(homogeneous_func, name, numeric_only)
        else:
            return self._apply_tablewise(homogeneous_func, name, numeric_only)

    def _numba_apply(self, func: Callable, engine_kwargs: Optional[Dict[str, Any]] = None, **func_kwargs: Any) -> Any:
        window_indexer = self._get_window_indexer()
        min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size  # type: ignore
        obj = self._create_data(getattr(self, '_selected_obj', self.obj))
        values = self._prep_values(obj.to_numpy())
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        start, end = window_indexer.get_window_bounds(num_values=len(values), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)  # type: ignore
        self._check_window_bounds(start, end, len(values))
        dtype_mapping = executor.float_dtype_mapping
        aggregator = executor.generate_shared_aggregator(func, dtype_mapping, is_grouped_kernel=False, **get_jit_arguments(engine_kwargs))
        result = aggregator(values.T, start=start, end=end, min_periods=min_periods, **func_kwargs).T
        index = self._slice_axis_for_step(obj.index, result)
        if obj.ndim == 1:
            result = result.squeeze()
            out = obj._constructor(result, index=index, name=obj.name)  # type: ignore
            return out
        else:
            columns = self._slice_axis_for_step(obj.columns, result.T)
            out = obj._constructor(result, index=index, columns=columns)  # type: ignore
            return self._resolve_output(out, obj)

    def aggregate(self, func: Optional[Callable] = None, *args: Any, **kwargs: Any) -> Any:
        relabeling, func, columns, order = reconstruct_func(func, **kwargs)
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if isinstance(result, ABCDataFrame) and relabeling:
            result = result.iloc[:, order]
            result.columns = columns
        if result is None:
            return self.apply(func, raw=False, args=args, kwargs=kwargs)
        return result
    agg = aggregate


class BaseWindowGroupby(BaseWindow):
    """
    Provide the groupby windowing facilities.
    """
    _attributes: List[str] = ['_grouper']

    def __init__(self, 
                 obj: Union[ABCSeries, ABCDataFrame], 
                 *args: Any, 
                 _grouper: Any, 
                 _as_index: bool = True, 
                 **kwargs: Any) -> None:
        from pandas.core.groupby.ops import BaseGrouper
        if not isinstance(_grouper, BaseGrouper):
            raise ValueError('Must pass a BaseGrouper object.')
        self._grouper = _grouper
        self._as_index = _as_index
        obj = obj.drop(columns=self._grouper.names, errors='ignore')
        if kwargs.get('step') is not None:
            raise NotImplementedError('step not implemented for groupby')
        super().__init__(obj, *args, **kwargs)

    def _apply(self, func: Callable, name: str, numeric_only: bool = False, numba_args: Tuple[Any, ...] = (), **kwargs: Any) -> Any:
        result = super()._apply(func, name, numeric_only, numba_args, **kwargs)
        grouped_object_index = self.obj.index
        grouped_index_name = [*grouped_object_index.names]
        groupby_keys = copy.copy(self._grouper.names)
        result_index_names = groupby_keys + grouped_index_name
        drop_columns = [key for key in self._grouper.names if key not in self.obj.index.names or key is None]
        if len(drop_columns) != len(groupby_keys):
            result = result.drop(columns=drop_columns, errors='ignore')
        codes = self._grouper.codes
        levels = copy.copy(self._grouper.levels)
        group_indices = self._grouper.indices.values()
        if group_indices:
            indexer = np.concatenate(list(group_indices))
        else:
            indexer = np.array([], dtype=np.intp)
        codes = [c.take(indexer) for c in codes]
        if grouped_object_index is not None:
            idx = grouped_object_index.take(indexer)
            if not isinstance(idx, MultiIndex):
                idx = MultiIndex.from_arrays([idx])
            codes.extend(list(idx.codes))
            levels.extend(list(idx.levels))
        result_index = MultiIndex(levels, codes, names=result_index_names, verify_integrity=False)
        result.index = result_index
        if not self._as_index:
            result = result.reset_index(level=list(range(len(groupby_keys))))
        return result

    def _apply_pairwise(self, target: Union[ABCSeries, DataFrame], other: Optional[Union[ABCSeries, DataFrame]], pairwise: Optional[bool], func: Callable, numeric_only: bool) -> Any:
        target = target.drop(columns=self._grouper.names, errors='ignore')
        result = super()._apply_pairwise(target, other, pairwise, func, numeric_only)
        if other is not None and (not all((len(group) == len(other) for group in self._grouper.indices.values()))):
            old_result_len = len(result)
            result = concat([result.take(gb_indices).reindex(result.index) for gb_indices in self._grouper.indices.values()])
            gb_pairs = (com.maybe_make_list(pair) for pair in self._grouper.indices.keys())
            groupby_codes: List[np.ndarray] = []
            groupby_levels: List[np.ndarray] = []
            for gb_level_pair in map(list, zip(*gb_pairs)):
                labels = np.repeat(np.array(gb_level_pair), old_result_len)
                codes, levels = factorize(labels)
                groupby_codes.append(codes)
                groupby_levels.append(levels)
        else:
            groupby_codes = self._grouper.codes
            groupby_levels = self._grouper.levels
            group_indices = self._grouper.indices.values()
            if group_indices:
                indexer = np.concatenate(list(group_indices))
            else:
                indexer = np.array([], dtype=np.intp)
            repeat_by = 1 if target.ndim == 1 else len(target.columns)
            groupby_codes = [np.repeat(c.take(indexer), repeat_by) for c in groupby_codes]
        if isinstance(result.index, MultiIndex):
            result_codes = list(result.index.codes)
            result_levels = list(result.index.levels)
            result_names = list(result.index.names)
        else:
            idx_codes, idx_levels = factorize(result.index)
            result_codes = [idx_codes]
            result_levels = [idx_levels]
            result_names = [result.index.name]
        result_codes = groupby_codes + result_codes
        result_levels = groupby_levels + result_levels
        result_names = self._grouper.names + result_names
        result_index = MultiIndex(result_levels, result_codes, names=result_names, verify_integrity=False)
        result.index = result_index
        return result

    def _create_data(self, obj: Union[ABCSeries, DataFrame], numeric_only: bool = False) -> Union[ABCSeries, DataFrame]:
        if not obj.empty:
            groupby_order = np.concatenate(list(self._grouper.indices.values())).astype(np.int64)
            obj = obj.take(groupby_order)
        return super()._create_data(obj, numeric_only)

    def _gotitem(self, key: Union[str, List[Any]], ndim: int, subset: Optional[Any] = None) -> BaseWindow:
        if self.on is not None:
            subset = self.obj.set_index(self._on)
        return super()._gotitem(key, ndim, subset=subset)


class Window(BaseWindow):
    """
    Provide rolling window calculations.
    """
    _attributes: List[str] = ['window', 'min_periods', 'center', 'win_type', 'on', 'closed', 'step', 'method']

    def _validate(self) -> None:
        super()._validate()
        if not isinstance(self.win_type, str):
            raise ValueError(f'Invalid win_type {self.win_type}')
        signal = import_optional_dependency('scipy.signal.windows', extra='Scipy is required to generate window weight.')
        self._scipy_weight_generator = getattr(signal, self.win_type, None)
        if self._scipy_weight_generator is None:
            raise ValueError(f'Invalid win_type {self.win_type}')
        if isinstance(self.window, BaseIndexer):
            raise NotImplementedError('BaseIndexer subclasses not implemented with win_types.')
        if not is_integer(self.window) or self.window < 0:  # type: ignore
            raise ValueError('window must be an integer 0 or greater')
        if self.method != 'single':
            raise NotImplementedError("'single' is the only supported method type.")

    def _center_window(self, result: np.ndarray, offset: int) -> np.ndarray:
        """
        Center the result in the window for weighted rolling aggregations.
        """
        if offset > 0:
            lead_indexer = slice(offset, None)
            result = np.copy(result[lead_indexer])
        return result

    def _apply(self, func: Callable, name: str, numeric_only: bool = False, numba_args: Tuple[Any, ...] = (), **kwargs: Any) -> Any:
        window = self._scipy_weight_generator(self.window, **kwargs)
        offset = (len(window) - 1) // 2 if self.center else 0

        def homogeneous_func(values: np.ndarray) -> np.ndarray:
            if values.size == 0:
                return values.copy()
            def calc(x: np.ndarray) -> np.ndarray:
                additional_nans = np.full(offset, np.nan)
                x = np.concatenate((x, additional_nans))
                return func(x, window, self.min_periods if self.min_periods is not None else len(window))
            with np.errstate(all='ignore'):
                result = np.asarray(calc(values))
            if self.center:
                result = self._center_window(result, offset)
            return result
        return self._apply_columnwise(homogeneous_func, name, numeric_only)[::self.step]  # type: ignore

    @doc(_shared_docs['aggregate'], see_also=dedent('\n        See Also\n        --------\n        DataFrame.aggregate : Similar DataFrame method.\n        Series.aggregate : Similar Series method.\n        '), examples=dedent('\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n        >>> df\n           A  B  C\n        0  1  4  7\n        1  2  5  8\n        2  3  6  9\n\n        >>> df.rolling(2, win_type="boxcar").agg("mean")\n             A    B    C\n        0  NaN  NaN  NaN\n        1  1.5  4.5  7.5\n        2  2.5  5.5  8.5\n        '), klass='Series/DataFrame', axis='')
    def aggregate(self, func: Optional[Callable] = None, *args: Any, **kwargs: Any) -> Any:
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            result = func(self)
        return result
    agg = aggregate

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, kwargs_scipy, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent("        >>> ser = pd.Series([0, 1, 5, 2, 8])\n\n        To get an instance of :class:`~pandas.core.window.rolling.Window` we need\n        to pass the parameter `win_type`.\n\n        >>> type(ser.rolling(2, win_type='gaussian'))\n        <class 'pandas.core.window.rolling.Window'>\n\n        In order to use the `SciPy` Gaussian window we need to provide the parameters\n        `M` and `std`. The parameter `M` corresponds to 2 in our example.\n        We pass the second parameter `std` as a parameter of the following method\n        (`sum` in this case):\n\n        >>> ser.rolling(2, win_type='gaussian').sum(std=3)\n        0         NaN\n        1    0.986207\n        2    5.917243\n        3    6.903450\n        4    9.862071\n        dtype: float64\n        "), window_method='rolling', aggregation_description='weighted window sum', agg_method='sum')
    def sum(self, numeric_only: bool = False, **kwargs: Any) -> Any:
        window_func = window_aggregations.roll_weighted_sum
        return self._apply(window_func, name='sum', numeric_only=numeric_only, **kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, kwargs_scipy, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent("        >>> ser = pd.Series([0, 1, 5, 2, 8])\n\n        To get an instance of :class:`~pandas.core.window.rolling.Window` we need\n        to pass the parameter `win_type`.\n\n        >>> type(ser.rolling(2, win_type='gaussian'))\n        <class 'pandas.core.window.rolling.Window'>\n\n        In order to use the `SciPy` Gaussian window we need to provide the parameters\n        `M` and `std`. The parameter `M` corresponds to 2 in our example.\n        We pass the second parameter `std` as a parameter of the following method:\n\n        >>> ser.rolling(2, win_type='gaussian').mean(std=3)\n        0    NaN\n        1    0.5\n        2    3.0\n        3    3.5\n        4    5.0\n        dtype: float64\n        "), window_method='rolling', aggregation_description='weighted window mean', agg_method='mean')
    def mean(self, numeric_only: bool = False, **kwargs: Any) -> Any:
        window_func = window_aggregations.roll_weighted_mean
        return self._apply(window_func, name='mean', numeric_only=numeric_only, **kwargs)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n        ').replace('\n', '', 1), kwargs_numeric_only, kwargs_scipy, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent("        >>> ser = pd.Series([0, 1, 5, 2, 8])\n\n        To get an instance of :class:`~pandas.core.window.rolling.Window` we need\n        to pass the parameter `win_type`.\n\n        >>> type(ser.rolling(2, win_type='gaussian'))\n        <class 'pandas.core.window.rolling.Window'>\n\n        In order to use the `SciPy` Gaussian window we need to provide the parameters\n        `M` and `std`. The parameter `M` corresponds to 2 in our example.\n        We pass the second parameter `std` as a parameter of the following method:\n\n        >>> ser.rolling(2, win_type='gaussian').var(std=3)\n        0     NaN\n        1     0.5\n        2     8.0\n        3     4.5\n        4    18.0\n        dtype: float64\n        "), window_method='rolling', aggregation_description='weighted window variance', agg_method='var')
    def var(self, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        window_func = partial(window_aggregations.roll_weighted_var, ddof=ddof)
        kwargs.pop('name', None)
        return self._apply(window_func, name='var', numeric_only=numeric_only, **kwargs)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n        ').replace('\n', '', 1), kwargs_numeric_only, kwargs_scipy, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent("        >>> ser = pd.Series([0, 1, 5, 2, 8])\n\n        To get an instance of :class:`~pandas.core.window.rolling.Window` we need\n        to pass the parameter `win_type`.\n\n        >>> type(ser.rolling(2, win_type='gaussian'))\n        <class 'pandas.core.window.rolling.Window'>\n\n        In order to use the `SciPy` Gaussian window we need to provide the parameters\n        `M` and `std`. The parameter `M` corresponds to 2 in our example.\n        We pass the second parameter `std` as a parameter of the following method:\n\n        >>> ser.rolling(2, win_type='gaussian').std(std=3)\n        0         NaN\n        1    0.707107\n        2    2.828427\n        3    2.121320\n        4    4.242641\n        dtype: float64\n        "), window_method='rolling', aggregation_description='weighted window standard deviation', agg_method='std')
    def std(self, ddof: int = 1, numeric_only: bool = False, **kwargs: Any) -> Any:
        return zsqrt(self.var(ddof=ddof, name='std', numeric_only=numeric_only, **kwargs))


class RollingAndExpandingMixin(BaseWindow):

    def count(self, numeric_only: bool = False) -> Any:
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name='count', numeric_only=numeric_only)

    def apply(self, func: Callable, raw: bool, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None, args: Optional[Sequence[Any]] = None, kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if not is_bool(raw):
            raise ValueError('raw parameter must be `True` or `False`')
        numba_args: Tuple[Any, ...] = ()
        if maybe_use_numba(engine):
            if raw is False:
                raise ValueError('raw must be `True` when using the numba engine')
            numba_args, kwargs = prepare_function_arguments(func, args, kwargs, num_required_args=1)
            if self.method == 'single':
                apply_func = generate_numba_apply_func(func, **get_jit_arguments(engine_kwargs))
            else:
                apply_func = generate_numba_table_func(func, **get_jit_arguments(engine_kwargs))
        elif engine in ('cython', None):
            if engine_kwargs is not None:
                raise ValueError('cython engine does not accept engine_kwargs')
            apply_func = self._generate_cython_apply_func(args, kwargs, raw, func)
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")
        return self._apply(apply_func, name='apply', numba_args=numba_args)

    def _generate_cython_apply_func(self, args: Sequence[Any], kwargs: Dict[str, Any], raw: bool, function: Callable) -> Callable:
        from pandas import Series
        window_func = partial(window_aggregations.roll_apply, args=args, kwargs=kwargs, raw=bool(raw), function=function)

        def apply_func(values: np.ndarray, begin: Sequence[int], end: Sequence[int], min_periods: int, raw: bool = raw) -> Any:
            if not raw:
                from pandas import Series
                values = Series(values, index=self._on, copy=False)  # type: ignore
            return window_func(values, begin, end, min_periods)
        return apply_func

    @overload
    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        ...

    @overload
    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        ...

    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        return com.pipe(self, func, *args, **kwargs)

    def sum(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nansum)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_sum
                return self._numba_apply(sliding_sum, engine_kwargs)
        window_func = window_aggregations.roll_sum
        return self._apply(window_func, name='sum', numeric_only=numeric_only)

    def max(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmax)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_min_max
                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=True)
        window_func = window_aggregations.roll_max
        return self._apply(window_func, name='max', numeric_only=numeric_only)

    def min(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmin)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_min_max
                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=False)
        window_func = window_aggregations.roll_min
        return self._apply(window_func, name='min', numeric_only=numeric_only)

    def mean(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmean)
                return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
            else:
                from pandas.core._numba.kernels import sliding_mean
                return self._numba_apply(sliding_mean, engine_kwargs)
        window_func = window_aggregations.roll_mean
        return self._apply(window_func, name='mean', numeric_only=numeric_only)

    def median(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if maybe_use_numba(engine):
            if self.method == 'table':
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmedian)
            else:
                func = np.nanmedian
            return self.apply(func, raw=True, engine=engine, engine_kwargs=engine_kwargs)
        window_func = window_aggregations.roll_median_c
        return self._apply(window_func, name='median', numeric_only=numeric_only)

    def std(self, ddof: int = 1, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if maybe_use_numba(engine):
            if self.method == 'table':
                raise NotImplementedError("std not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var
            return zsqrt(self._numba_apply(sliding_var, engine_kwargs, ddof=ddof))
        window_func = window_aggregations.roll_var
        def zsqrt_func(values: np.ndarray, begin: Sequence[int], end: Sequence[int], min_periods: int) -> np.ndarray:
            return zsqrt(window_func(values, begin, end, min_periods, ddof=ddof))
        return self._apply(zsqrt_func, name='std', numeric_only=numeric_only)

    def var(self, ddof: int = 1, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        if maybe_use_numba(engine):
            if self.method == 'table':
                raise NotImplementedError("var not supported with method='table'")
            from pandas.core._numba.kernels import sliding_var
            return self._numba_apply(sliding_var, engine_kwargs, ddof=ddof)
        window_func = partial(window_aggregations.roll_var, ddof=ddof)
        return self._apply(window_func, name='var', numeric_only=numeric_only)

    def skew(self, numeric_only: bool = False) -> Any:
        window_func = window_aggregations.roll_skew
        return self._apply(window_func, name='skew', numeric_only=numeric_only)

    def sem(self, ddof: int = 1, numeric_only: bool = False) -> Any:
        self._validate_numeric_only('sem', numeric_only)
        return self.std(numeric_only=numeric_only) / (self.count(numeric_only=numeric_only) - ddof).pow(0.5)

    def kurt(self, numeric_only: bool = False) -> Any:
        window_func = window_aggregations.roll_kurt
        return self._apply(window_func, name='kurt', numeric_only=numeric_only)

    def first(self, numeric_only: bool = False) -> Any:
        window_func = window_aggregations.roll_first
        return self._apply(window_func, name='first', numeric_only=numeric_only)

    def last(self, numeric_only: bool = False) -> Any:
        window_func = window_aggregations.roll_last
        return self._apply(window_func, name='last', numeric_only=numeric_only)

    def quantile(self, q: float, interpolation: QuantileInterpolation = 'linear', numeric_only: bool = False) -> Any:
        if q == 1.0:
            window_func = window_aggregations.roll_max
        elif q == 0.0:
            window_func = window_aggregations.roll_min
        else:
            window_func = partial(window_aggregations.roll_quantile, quantile=q, interpolation=interpolation)
        return self._apply(window_func, name='quantile', numeric_only=numeric_only)

    def rank(self, method: str = 'average', ascending: bool = True, pct: bool = False, numeric_only: bool = False) -> Any:
        window_func = partial(window_aggregations.roll_rank, method=method, ascending=ascending, percentile=pct)
        return self._apply(window_func, name='rank', numeric_only=numeric_only)

    def cov(self, other: Optional[Union[ABCSeries, DataFrame]] = None, pairwise: Optional[bool] = None, ddof: int = 1, numeric_only: bool = False) -> Any:
        if self.step is not None:
            raise NotImplementedError('step not implemented for cov')
        self._validate_numeric_only('cov', numeric_only)
        from pandas import Series
        def cov_func(x: Series, y: Series) -> Series:
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size  # type: ignore
            start, end = window_indexer.get_window_bounds(num_values=len(x_array), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)  # type: ignore
            self._check_window_bounds(start, end, len(x_array))
            with np.errstate(all='ignore'):
                mean_x_y = window_aggregations.roll_mean(x_array * y_array, start, end, min_periods)
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(notna(x_array + y_array).astype(np.float64), start, end, 0)
                result = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
            return Series(result, index=x.index, name=x.name, copy=False)
        return self._apply_pairwise(getattr(self, '_selected_obj', self.obj), other, pairwise, cov_func, numeric_only)

    def corr(self, other: Optional[Union[ABCSeries, DataFrame]] = None, pairwise: Optional[bool] = None, ddof: int = 1, numeric_only: bool = False) -> Any:
        if self.step is not None:
            raise NotImplementedError('step not implemented for corr')
        self._validate_numeric_only('corr', numeric_only)
        from pandas import Series
        def corr_func(x: Series, y: Series) -> Series:
            x_array = self._prep_values(x)
            y_array = self._prep_values(y)
            window_indexer = self._get_window_indexer()
            min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size  # type: ignore
            start, end = window_indexer.get_window_bounds(num_values=len(x_array), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)  # type: ignore
            self._check_window_bounds(start, end, len(x_array))
            with np.errstate(all='ignore'):
                mean_x_y = window_aggregations.roll_mean(x_array * y_array, start, end, min_periods)
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                count_x_y = window_aggregations.roll_sum(notna(x_array + y_array).astype(np.float64), start, end, 0)
                x_var = window_aggregations.roll_var(x_array, start, end, min_periods, ddof)
                y_var = window_aggregations.roll_var(y_array, start, end, min_periods, ddof)
                numerator = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
                denominator = (x_var * y_var) ** 0.5
                result = numerator / denominator
            return Series(result, index=x.index, name=x.name, copy=False)
        return self._apply_pairwise(getattr(self, '_selected_obj', self.obj), other, pairwise, corr_func, numeric_only)


Rolling.__doc__ = Window.__doc__


class Rolling(RollingAndExpandingMixin):
    _attributes: List[str] = ['window', 'min_periods', 'center', 'win_type', 'on', 'closed', 'step', 'method']

    def _validate(self) -> None:
        super()._validate()
        if (self.obj.empty or isinstance(self._on, (DatetimeIndex, TimedeltaIndex, PeriodIndex)) or (isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in 'mM')) and isinstance(self.window, (str, BaseOffset, timedelta)):
            self._validate_datetimelike_monotonic()
            try:
                freq = to_offset(self.window)
            except (TypeError, ValueError) as err:
                raise ValueError(f'passed window {self.window} is not compatible with a datetimelike index') from err
            if isinstance(self._on, PeriodIndex):
                self._win_freq_i8 = freq.nanos / (self._on.freq.nanos / self._on.freq.n)
            else:
                try:
                    unit = dtype_to_unit(self._on.dtype)
                except TypeError:
                    unit = 'ns'
                self._win_freq_i8 = Timedelta(freq.nanos).as_unit(unit)._value
            if self.min_periods is None:
                self.min_periods = 1
            if self.step is not None:
                raise NotImplementedError('step is not supported with frequency windows')
        elif isinstance(self.window, BaseIndexer):
            pass
        elif not is_integer(self.window) or self.window < 0:  # type: ignore
            raise ValueError('window must be an integer 0 or greater')

    def _validate_datetimelike_monotonic(self) -> None:
        """
        Validate self._on is monotonic (increasing or decreasing) and has
        no NaT values for frequency windows.
        """
        if self._on.hasnans:
            self._raise_monotonic_error('values must not have NaT')
        if not (self._on.is_monotonic_increasing or self._on.is_monotonic_decreasing):
            self._raise_monotonic_error('values must be monotonic')

    def _raise_monotonic_error(self, msg: str) -> None:
        on_val = self.on
        if on_val is None:
            on_val = 'index'
        raise ValueError(f'{on_val} {msg}')

    @doc(_shared_docs['aggregate'], see_also=dedent('\n        See Also\n        --------\n        Series.rolling : Calling object with Series data.\n        DataFrame.rolling : Calling object with DataFrame data.\n        '), examples=dedent('\n        Examples\n        --------\n        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})\n        >>> df\n           A  B  C\n        0  1  4  7\n        1  2  5  8\n        2  3  6  9\n\n        >>> df.rolling(2).sum()\n             A     B     C\n        0  NaN   NaN   NaN\n        1  3.0   9.0  15.0\n        2  5.0  11.0  17.0\n\n        >>> df.rolling(2).agg({"A": "sum", "B": "min"})\n             A    B\n        0  NaN  NaN\n        1  3.0  4.0\n        2  5.0  5.0\n        '), klass='Series/Dataframe', axis='')
    def aggregate(self, func: Optional[Callable] = None, *args: Any, **kwargs: Any) -> Any:
        return super().aggregate(func, *args, **kwargs)
    agg = aggregate

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('\n        >>> s = pd.Series([2, 3, np.nan, 10])\n        >>> s.rolling(2).count()\n        0    NaN\n        1    2.0\n        2    1.0\n        3    1.0\n        dtype: float64\n        >>> s.rolling(3).count()\n        0    NaN\n        1    NaN\n        2    2.0\n        3    2.0\n        dtype: float64\n        >>> s.rolling(4).count()\n        0    NaN\n        1    NaN\n        2    NaN\n        3    3.0\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='count of non NaN observations', agg_method='count')
    def count(self, numeric_only: bool = False) -> Any:
        return super().count(numeric_only)

    @doc(template_header, create_section_header('Parameters'), window_apply_parameters, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('        >>> ser = pd.Series([1, 6, 5, 4])\n        >>> ser.rolling(2).apply(lambda s: s.sum() - s.min())\n        0    NaN\n        1    6.0\n        2    6.0\n        3    5.0\n        dtype: float64\n        '), window_method='rolling', aggregation_description='custom aggregation function', agg_method='apply')
    def apply(self, func: Callable, raw: bool, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None, args: Optional[Sequence[Any]] = None, kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return super().apply(func, raw=raw, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)

    @overload
    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        ...

    @overload
    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        ...

    @final
    @Substitution(klass='Rolling', examples="\n    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},\n    ...                   index=pd.date_range('2012-08-02', periods=4))\n    >>> df\n                A\n    2012-08-02  1\n    2012-08-03  2\n    2012-08-04  3\n    2012-08-05  4\n\n    To get the difference between each rolling 2-day window's maximum and minimum\n    value in one pass, you can do\n\n    >>> df.rolling('2D').pipe(lambda x: x.max() - x.min())\n                  A\n    2012-08-02  0.0\n    2012-08-03  1.0\n    2012-08-04  1.0\n    2012-08-05  1.0")
    @Appender(template_pipe)
    def pipe(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        return super().pipe(func, *args, **kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent('\n        >>> s = pd.Series([1, 2, 3, 4, 5])\n        >>> s\n        0    1\n        1    2\n        2    3\n        3    4\n        4    5\n        dtype: int64\n\n        >>> s.rolling(3).sum()\n        0     NaN\n        1     NaN\n        2     6.0\n        3     9.0\n        4    12.0\n        dtype: float64\n\n        >>> s.rolling(3, center=True).sum()\n        0     NaN\n        1     6.0\n        2     9.0\n        3    12.0\n        4     NaN\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='sum', agg_method='sum')
    def sum(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return super().sum(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), numba_notes, create_section_header('Examples'), dedent('\n        >>> ser = pd.Series([1, 2, 3, 4])\n        >>> ser.rolling(2).max()\n        0    NaN\n        1    2.0\n        2    3.0\n        3    4.0\n        dtype: float64\n        '), window_method='rolling', aggregation_description='maximum', agg_method='max')
    def max(self, numeric_only: bool = False, *args: Any, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        return super().max(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section('Notes'), numba_notes, create_section_header('Examples'), dedent('\n        Performing a rolling minimum with a window size of 3.\n\n        >>> s = pd.Series([4, 3, 5, 2, 6])\n        >>> s.rolling(3).min()\n        0    NaN\n        1    NaN\n        2    3.0\n        3    2.0\n        4    2.0\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='minimum', agg_method='min')
    def min(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return super().min(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, window_agg_numba_parameters(), create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section('Notes'), numba_notes, create_section_header('Examples'), dedent('\n        The below examples will show rolling mean calculations with window sizes of\n        two and three, respectively.\n\n        >>> s = pd.Series([1, 2, 3, 4])\n        >>> s.rolling(2).mean()\n        0    NaN\n        1    1.5\n        2    2.5\n        3    3.5\n        dtype: float64\n\n        >>> s.rolling(3).mean()\n        0    NaN\n        1    NaN\n        2    2.0\n        3    3.0\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='mean', agg_method='mean')
    def mean(self, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return super().mean(numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n        ').replace('\n', '', 1), kwargs_numeric_only, window_agg_numba_parameters('1.4'), create_section_header('Returns'), template_returns, create_section_header('See Also'), 'numpy.std : Equivalent method for NumPy array.\n', template_see_also, create_section_header('Notes'), dedent('\n        The default ``ddof`` of 1 used in :meth:`Series.std` is different\n        than the default ``ddof`` of 0 in :func:`numpy.std`.\n\n        A minimum of one period is required for the rolling calculation.\n\n        ').replace('\n', '', 1), create_section_header('Examples'), dedent('\n        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])\n        >>> s.rolling(3).std()\n        0         NaN\n        1         NaN\n        2    0.577350\n        3    1.000000\n        4    1.000000\n        5    1.154701\n        6    0.000000\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='standard deviation', agg_method='std')
    def std(self, ddof: int = 1, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return super().std(ddof=ddof, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n        ').replace('\n', '', 1), kwargs_numeric_only, window_agg_numba_parameters('1.4'), create_section_header('Returns'), template_returns, create_section_header('See Also'), 'numpy.var : Equivalent method for NumPy array.\n', template_see_also, create_section_header('Notes'), dedent('\n        The default ``ddof`` of 1 used in :meth:`Series.var` is different\n        than the default ``ddof`` of 0 in :func:`numpy.var`.\n\n        A minimum of one period is required for the rolling calculation.\n\n        ').replace('\n', '', 1), create_section_header('Examples'), dedent('\n        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])\n        >>> s.rolling(3).var()\n        0         NaN\n        1         NaN\n        2    0.333333\n        3    1.000000\n        4    1.000000\n        5    1.333333\n        6    0.000000\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='variance', agg_method='var')
    def var(self, ddof: int = 1, numeric_only: bool = False, engine: Optional[str] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> Any:
        return super().var(ddof=ddof, numeric_only=numeric_only, engine=engine, engine_kwargs=engine_kwargs)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), dedent('\n        scipy.stats.skew : Third moment of a probability density.\n        '), template_see_also, create_section_header('Notes'), dedent('\n        A minimum of three periods is required for the rolling calculation.\n\n        '), create_section_header('Examples'), dedent('\n        >>> ser = pd.Series([1, 5, 2, 7, 15, 6])\n        >>> ser.rolling(3).skew().round(6)\n        0         NaN\n        1         NaN\n        2    1.293343\n        3   -0.585583\n        4    0.670284\n        5    1.652317\n        dtype: float64\n        '), window_method='rolling', aggregation_description='unbiased skewness', agg_method='skew')
    def skew(self, numeric_only: bool = False) -> Any:
        return super().skew(numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n        ').replace('\n', '', 1), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Notes'), 'A minimum of one period is required for the calculation.\n\n', create_section_header('Examples'), dedent('\n        >>> s = pd.Series([0, 1, 2, 3])\n        >>> s.rolling(2, min_periods=1).sem()\n        0         NaN\n        1    0.707107\n        2    0.707107\n        3    0.707107\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='standard error of mean', agg_method='sem')
    def sem(self, ddof: int = 1, numeric_only: bool = False) -> Any:
        self._validate_numeric_only('sem', numeric_only)
        return self.std(numeric_only=numeric_only) / (self.count(numeric_only) - ddof).pow(0.5)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), dedent('\n        scipy.stats.kurtosis : Reference SciPy method.\n        '), template_see_also, create_section_header('Notes'), dedent('\n        A minimum of four periods is required for the calculation.\n\n        '), create_section_header('Examples'), dedent('\n        The example below will show a rolling calculation with a window size of\n        four matching the equivalent function call using `scipy.stats`.\n\n        >>> arr = [1, 2, 3, 4, 999]\n        >>> import scipy.stats\n        >>> print(f"{{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}}")\n        -1.200000\n        >>> print(f"{{scipy.stats.kurtosis(arr[1:], bias=False):.6f}}")\n        3.999946\n        >>> s = pd.Series(arr)\n        >>> s.rolling(4).kurt()\n        0         NaN\n        1         NaN\n        2         NaN\n        3   -1.200000\n        4    3.999946\n        dtype: float64\n        '), window_method='rolling', aggregation_description="Fisher's definition of kurtosis without bias", agg_method='kurt')
    def kurt(self, numeric_only: bool = False) -> Any:
        return super().kurt(numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), dedent('\n        GroupBy.first : Similar method for GroupBy objects.\n        Rolling.last : Method to get the last element in each window.\n\n        '), create_section_header('Examples'), dedent('\n        The example below will show a rolling calculation with a window size of\n        three.\n\n        >>> s = pd.Series(range(5))\n        >>> s.rolling(3).first()\n        0         NaN\n        1         NaN\n        2         0.0\n        3         1.0\n        4         2.0\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='First (left-most) element of the window', agg_method='first')
    def first(self, numeric_only: bool = False) -> Any:
        return super().first(numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), dedent('\n        GroupBy.last : Similar method for GroupBy objects.\n        Rolling.first : Method to get the first element in each window.\n\n        '), create_section_header('Examples'), dedent('\n        The example below will show a rolling calculation with a window size of\n        three.\n\n        >>> s = pd.Series(range(5))\n        >>> s.rolling(3).last()\n        0         NaN\n        1         NaN\n        2         2.0\n        3         3.0\n        4         4.0\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='Last (right-most) element of the window', agg_method='last')
    def last(self, numeric_only: bool = False) -> Any:
        return super().last(numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), dedent("\n        q : float\n            Quantile to compute. 0 <= quantile <= 1.\n\n            .. deprecated:: 2.1.0\n                This was renamed from 'quantile' to 'q' in version 2.1.0.\n        interpolation : {{'linear', 'lower', 'higher', 'midpoint', 'nearest'}}\n            This optional parameter specifies the interpolation method to use,\n            when the desired quantile lies between two data points `i` and `j`:\n\n                * linear: `i + (j - i) * fraction`, where `fraction` is the\n                  fractional part of the index surrounded by `i` and `j`.\n                * lower: `i`.\n                * higher: `j`.\n                * nearest: `i` or `j` whichever is nearest.\n                * midpoint: (`i` + `j`) / 2.\n        ").replace('\n', '', 1), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent("\n        >>> s = pd.Series([1, 2, 3, 4])\n        >>> s.rolling(2).quantile(.4, interpolation='lower')\n        0    NaN\n        1    1.0\n        2    2.0\n        3    3.0\n        dtype: float64\n\n        >>> s.rolling(2).quantile(.4, interpolation='midpoint')\n        0    NaN\n        1    1.5\n        2    2.5\n        3    3.5\n        dtype: float64\n        ").replace('\n', '', 1), window_method='rolling', aggregation_description='quantile', agg_method='quantile')
    def quantile(self, q: float, interpolation: QuantileInterpolation = 'linear', numeric_only: bool = False) -> Any:
        return super().quantile(q=q, interpolation=interpolation, numeric_only=numeric_only)

    @doc(template_header, '.. versionadded:: 1.4.0 \n\n', create_section_header('Parameters'), dedent("\n        method : {{'average', 'min', 'max'}}, default 'average'\n            How to rank the group of records that have the same value (i.e. ties):\n\n            * average: average rank of the group\n            * min: lowest rank in the group\n            * max: highest rank in the group\n\n        ascending : bool, default True\n            Whether or not the elements should be ranked in ascending order.\n        pct : bool, default False\n            Whether or not to display the returned rankings in percentile\n            form.\n        ").replace('\n', '', 1), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), template_see_also, create_section_header('Examples'), dedent('\n        >>> s = pd.Series([1, 4, 2, 3, 5, 3])\n        >>> s.rolling(3).rank()\n        0    NaN\n        1    2.0\n        2    2.0\n        3    2.0\n        4    3.0\n        5    1.5\n        dtype: float64\n\n        >>> s.rolling(3).rank(method="max")\n        0    NaN\n        1    2.0\n        2    2.0\n        3    2.0\n        4    3.0\n        5    2.0\n        dtype: float64\n\n        >>> s.rolling(3).rank(method="min")\n        0    NaN\n        1    2.0\n        2    2.0\n        3    2.0\n        4    3.0\n        5    1.0\n        dtype: float64\n        ').replace('\n', '', 1), window_method='rolling', aggregation_description='rank', agg_method='rank')
    def rank(self, method: str = 'average', ascending: bool = True, pct: bool = False, numeric_only: bool = False) -> Any:
        return super().rank(method=method, ascending=ascending, pct=pct, numeric_only=numeric_only)

    @doc(template_header, create_section_header('Parameters'), dedent('\n        other : Series or DataFrame, optional\n            If not supplied then will default to self and produce pairwise\n            output.\n        pairwise : bool, default None\n            If False then only matching columns between self and other will be\n            used and the output will be a DataFrame.\n            If True then all pairwise combinations will be calculated and the\n            output will be a MultiIndexed DataFrame in the case of DataFrame\n            inputs. In the case of missing elements, only complete pairwise\n            observations will be used.\n        ddof : int, default 1\n            Delta Degrees of Freedom.  The divisor used in calculations\n            is ``N - ddof``, where ``N`` represents the number of elements.\n        ").replace('\n', '', 1), kwargs_numeric_only, create_section_header('Returns'), template_returns, create_section_header('See Also'), dedent("\n        cov : Similar method to calculate covariance.\n        numpy.corrcoef : NumPy Pearson's correlation calculation.\n        ").replace('\n', '', 1), template_see_also, create_section('Notes'), dedent("\n        This function uses Pearson's definition of correlation\n        (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).\n\n        When `other` is not specified, the output will be self correlation (e.g.\n        all 1's), except for :class:`~pandas.DataFrame` inputs with `pairwise`\n        set to `True`.\n\n        Function will return ``NaN`` for correlations of equal valued sequences;\n        this is the result of a 0/0 division error.\n\n        When `pairwise` is set to `False`, only matching columns between `self` and\n        `other` will be used.\n\n        When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame\n        with the original index on the first level, and the `other` DataFrame\n        columns on the second level.\n\n        In the case of missing elements, only complete pairwise observations\n        will be used.\n\n        ").replace('\n', '', 1), create_section('Examples'), dedent("\n        The below example shows a rolling calculation with a window size of\n        four matching the equivalent function call using :meth:`numpy.corrcoef`.\n\n        >>> v1 = [3, 3, 3, 5, 8]\n        >>> v2 = [3, 4, 4, 4, 8]\n        >>> np.corrcoef(v1[:-1], v2[:-1])\n        array([[1.        , 0.33333333],\n               [0.33333333, 1.        ]])\n        >>> np.corrcoef(v1[1:], v2[1:])\n        array([[1.       , 0.9169493],\n               [0.9169493, 1.       ]])\n        >>> s1 = pd.Series(v1)\n        >>> s2 = pd.Series(v2)\n        >>> s1.rolling(4).corr(s2)\n        0         NaN\n        1         NaN\n        2         NaN\n        3    0.333333\n        4    0.916949\n        dtype: float64\n\n        The below example shows a similar rolling calculation on a\n        DataFrame using the pairwise option.\n\n        >>> matrix = np.array([[51., 35.],\n        ...                    [49., 30.],\n        ...                    [47., 32.],\n        ...                    [46., 31.],\n        ...                    [50., 36.]])\n        >>> np.corrcoef(matrix[:-1, 0], matrix[:-1, 1])\n        array([[1.       , 0.6263001],\n               [0.6263001, 1.       ]])\n        >>> np.corrcoef(matrix[1:, 0], matrix[1:, 1])\n        array([[1.        , 0.55536811],\n               [0.55536811, 1.        ]])\n        >>> df = pd.DataFrame(matrix, columns=['X', 'Y'])\n        >>> df\n              X     Y\n        0  51.0  35.0\n        1  49.0  30.0\n        2  47.0  32.0\n        3  46.0  31.0\n        4  50.0  36.0\n        >>> df.rolling(4).corr(pairwise=True)\n                    X         Y\n        0 X       NaN       NaN\n          Y       NaN       NaN\n        1 X       NaN       NaN\n          Y       NaN       NaN\n        2 X       NaN       NaN\n          Y       NaN       NaN\n        3 X  1.000000  0.626300\n          Y  0.626300  1.000000\n        4 X  1.000000  0.555368\n          Y  0.555368  1.000000\n        ").replace('\n', '', 1), window_method='rolling', aggregation_description='correlation', agg_method='corr')
    def corr(self, other: Optional[Union[ABCSeries, DataFrame]] = None, pairwise: Optional[bool] = None, ddof: int = 1, numeric_only: bool = False) -> Any:
        return super().corr(other=other, pairwise=pairwise, ddof=ddof, numeric_only=numeric_only)


class RollingGroupby(BaseWindowGroupby, Rolling):
    """
    Provide a rolling groupby implementation.
    """
    _attributes: List[str] = Rolling._attributes + BaseWindowGroupby._attributes

    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
        """
        indexer_kwargs: Optional[Dict[str, Any]] = None
        index_array = self._index_array
        if isinstance(self.window, BaseIndexer):
            rolling_indexer = type(self.window)
            indexer_kwargs = self.window.__dict__.copy()  # type: ignore
            assert isinstance(indexer_kwargs, dict)
            indexer_kwargs.pop('index_array', None)
            window_val = self.window
        elif self._win_freq_i8 is not None:
            rolling_indexer = VariableWindowIndexer
            window_val = self._win_freq_i8
        else:
            rolling_indexer = FixedWindowIndexer
            window_val = self.window  # type: ignore
        window_indexer = GroupbyIndexer(index_array=index_array,
                                        window_size=window_val,
                                        groupby_indices=self._grouper.indices,
                                        window_indexer=rolling_indexer,
                                        indexer_kwargs=indexer_kwargs)
        return window_indexer

    def _validate_datetimelike_monotonic(self) -> None:
        """
        Validate that each group in self._on is monotonic
        """
        if self._on.hasnans:
            self._raise_monotonic_error('values must not have NaT')
        for group_indices in self._grouper.indices.values():
            group_on = self._on.take(group_indices)
            if not (group_on.is_monotonic_increasing or group_on.is_monotonic_decreasing):
                on_val = 'index' if self.on is None else self.on
                raise ValueError(f'Each group within {on_val} must be monotonic. Sort the values in {on_val} first.')
