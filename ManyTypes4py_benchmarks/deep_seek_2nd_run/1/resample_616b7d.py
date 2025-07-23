from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
    TYPE_CHECKING, 
    Literal, 
    cast, 
    final, 
    no_type_check, 
    overload, 
    Any, 
    Callable, 
    Hashable, 
    Optional, 
    Union, 
    Tuple, 
    List, 
    Dict, 
    FrozenSet, 
    Set, 
    Type, 
    TypeVar, 
    Generic
)
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
    BaseOffset, 
    IncompatibleFrequency, 
    NaT, 
    Period, 
    Timedelta, 
    Timestamp, 
    to_offset
)
from pandas._typing import (
    NDFrameT, 
    Axis, 
    Frequency, 
    IndexLabel, 
    InterpolateOptions, 
    TimeGrouperOrigin, 
    TimedeltaConvertibleTypes, 
    TimestampConvertibleTypes, 
    npt
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import Appender, Substitution, doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.dtypes import ArrowDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
import pandas.core.algorithms as algos
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject, SelectionMixin
from pandas.core.generic import NDFrame, _shared_docs
from pandas.core.groupby.groupby import BaseGroupBy, GroupBy, _pipe_template, get_groupby
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex, date_range
from pandas.core.indexes.period import PeriodIndex, period_range
from pandas.core.indexes.timedeltas import TimedeltaIndex, timedelta_range
from pandas.core.reshape.concat import concat
from pandas.tseries.frequencies import is_subperiod, is_superperiod
from pandas.tseries.offsets import Day, Tick

if TYPE_CHECKING:
    from collections.abc import Callable as CallableT, Hashable as HashableT
    from pandas import DataFrame, Series
    from pandas._typing import (
        Any as AnyT, 
        AnyArrayLike, 
        Concatenate, 
        FreqIndexT, 
        P, 
        Self, 
        T
    )

_shared_docs_kwargs: Dict[str, Any] = {}

T = TypeVar('T', bound='Resampler')

class Resampler(BaseGroupBy, PandasObject):
    exclusions: FrozenSet[Hashable] = frozenset()
    _internal_names_set: Set[str] = {'obj', 'ax', '_indexer'}
    _attributes: List[str] = ['freq', 'closed', 'label', 'convention', 'origin', 'offset']

    def __init__(
        self, 
        obj: Union[Series, DataFrame], 
        timegrouper: TimeGrouper, 
        *, 
        gpr_index: Index, 
        group_keys: bool = False, 
        selection: Optional[IndexLabel] = None, 
        include_groups: bool = False
    ) -> None:
        if include_groups:
            raise ValueError('include_groups=True is no longer allowed.')
        self._timegrouper: TimeGrouper = timegrouper
        self.keys: Optional[List[Hashable]] = None
        self.sort: bool = True
        self.group_keys: bool = group_keys
        self.as_index: bool = True
        self.obj: Union[Series, DataFrame]
        self.ax: Index
        self._indexer: Optional[npt.NDArray[np.intp]]
        self.obj, self.ax, self._indexer = self._timegrouper._set_grouper(
            self._convert_obj(obj), 
            sort=True, 
            gpr_index=gpr_index
        )
        self.binner: Index
        self._grouper: BinGrouper
        self.binner, self._grouper = self._get_binner()
        self._selection: Optional[IndexLabel] = selection
        if self._timegrouper.key is not None:
            self.exclusions = frozenset([self._timegrouper.key])
        else:
            self.exclusions = frozenset()

    @final
    def __str__(self) -> str:
        attrs = (
            f'{k}={getattr(self._timegrouper, k)}' 
            for k in self._attributes 
            if getattr(self._timegrouper, k, None) is not None
        )
        return f'{type(self).__name__} [{", ".join(attrs)}]'

    @final
    def __getattr__(self, attr: str) -> Any:
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self._attributes:
            return getattr(self._timegrouper, attr)
        if attr in self.obj:
            return self[attr]
        return object.__getattribute__(self, attr)

    @final
    @property
    def _from_selection(self) -> bool:
        return self._timegrouper is not None and (
            self._timegrouper.key is not None or 
            self._timegrouper.level is not None
        )

    def _convert_obj(self, obj: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
        return obj._consolidate()

    def _get_binner_for_time(self) -> Tuple[Index, npt.NDArray[np.intp], Index]:
        raise AbstractMethodError(self)

    @final
    def _get_binner(self) -> Tuple[Index, BinGrouper]:
        binner: Index
        bins: npt.NDArray[np.intp]
        binlabels: Index
        binner, bins, binlabels = self._get_binner_for_time()
        assert len(bins) == len(binlabels)
        bin_grouper = BinGrouper(bins, binlabels, indexer=self._indexer)
        return (binner, bin_grouper)

    @overload
    def pipe(
        self: T, 
        func: Callable[..., T], 
        *args: Any, 
        **kwargs: Any
    ) -> Any: ...

    @overload
    def pipe(
        self, 
        func: Callable[..., Any], 
        *args: Any, 
        **kwargs: Any
    ) -> Any: ...

    @final
    @Substitution(
        klass='Resampler', 
        examples="\n    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},\n    ...                   index=pd.date_range('2012-08-02', periods=4))\n    >>> df\n                A\n    2012-08-02  1\n    2012-08-03  2\n    2012-08-04  3\n    2012-08-05  4\n\n    To get the difference between each 2-day period's maximum and minimum\n    value in one pass, you can do\n\n    >>> df.resample('2D').pipe(lambda x: x.max() - x.min())\n                A\n    2012-08-02  1\n    2012-08-04  1"
    )
    @Appender(_pipe_template)
    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return super().pipe(func, *args, **kwargs)

    _agg_see_also_doc = dedent('''
    See Also
    --------
    DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
        or list of string/callables.
    DataFrame.resample.transform : Transforms the Series on each group
        based on the given function.
    DataFrame.aggregate: Aggregate using one or more
        operations over the specified axis.
    ''')
    
    _agg_examples_doc = dedent('''
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5],
    ...               index=pd.date_range('20130101', periods=5, freq='s'))
    >>> s
    2013-01-01 00:00:00    1
    2013-01-01 00:00:01    2
    2013-01-01 00:00:02    3
    2013-01-01 00:00:03    4
    2013-01-01 00:00:04    5
    Freq: s, dtype: int64

    >>> r = s.resample('2s')

    >>> r.agg("sum")
    2013-01-01 00:00:00    3
    2013-01-01 00:00:02    7
    2013-01-01 00:00:04    5
    Freq: 2s, dtype: int64

    >>> r.agg(['sum', 'mean', 'max'])
                         sum  mean  max
    2013-01-01 00:00:00    3   1.5    2
    2013-01-01 00:00:02    7   3.5    4
    2013-01-01 00:00:04    5   5.0    5

    >>> r.agg({'result': lambda x: x.mean() / x.std(),
    ...        'total': "sum"})
                           result  total
    2013-01-01 00:00:00  2.121320      3
    2013-01-01 00:00:02  4.949747      7
    2013-01-01 00:00:04       NaN      5

    >>> r.agg(average="mean", total="sum")
                             average  total
    2013-01-01 00:00:00      1.5      3
    2013-01-01 00:00:02      3.5      7
    2013-01-01 00:00:04      5.0      5
    ''')

    @final
    @doc(
        _shared_docs['aggregate'], 
        see_also=_agg_see_also_doc, 
        examples=_agg_examples_doc, 
        klass='DataFrame', 
        axis=''
    )
    def aggregate(
        self, 
        func: Optional[Union[str, Callable[..., Any], List[Union[str, Callable[..., Any]]], Dict[str, Union[str, Callable[..., Any]]]]] = None, 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            how = func
            result = self._groupby_and_aggregate(how, *args, **kwargs)
        return result
    
    agg = aggregate
    apply = aggregate

    @final
    def transform(
        self, 
        arg: Callable[..., Any], 
        *args: Any, 
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        return self._selected_obj.groupby(self._timegrouper).transform(arg, *args, **kwargs)

    def _downsample(self, how: str, **kwargs: Any) -> Union[Series, DataFrame]:
        raise AbstractMethodError(self)

    def _upsample(
        self, 
        f: str, 
        limit: Optional[int] = None, 
        fill_value: Any = None
    ) -> Union[Series, DataFrame]:
        raise AbstractMethodError(self)

    def _gotitem(
        self, 
        key: Union[str, List[str]], 
        ndim: int, 
        subset: Optional[Union[Series, DataFrame]] = None
    ) -> Any:
        grouper = self._grouper
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                assert subset.ndim == 1
        if ndim == 1:
            assert subset.ndim == 1
        grouped = get_groupby(subset, by=None, grouper=grouper, group_keys=self.group_keys)
        return grouped

    def _groupby_and_aggregate(
        self, 
        how: Union[str, Callable[..., Any]], 
        *args: Any, 
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        grouper = self._grouper
        obj = self._obj_with_exclusions
        grouped = get_groupby(obj, by=None, grouper=grouper, group_keys=self.group_keys)
        try:
            if callable(how):
                func = lambda x: how(x, *args, **kwargs)
                result = grouped.aggregate(func)
            else:
                result = grouped.aggregate(how, *args, **kwargs)
        except (AttributeError, KeyError):
            result = grouped.apply(how, *args, **kwargs)
        except ValueError as err:
            if 'Must produce aggregated value' in str(err):
                pass
            else:
                raise
            result = grouped.apply(how, *args, **kwargs)
        return self._wrap_result(result)

    @final
    def _get_resampler_for_grouping(
        self, 
        groupby: GroupBy, 
        key: Optional[Hashable]
    ) -> Resampler:
        return self._resampler_for_grouping(groupby=groupby, key=key, parent=self)

    def _wrap_result(self, result: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
        obj = self.obj
        if isinstance(result, ABCDataFrame) and len(result) == 0 and (not isinstance(result.index, PeriodIndex)):
            result = result.set_index(_asfreq_compat(obj.index[:0], freq=self.freq), append=True)
        if isinstance(result, ABCSeries) and self._selection is not None:
            result.name = self._selection
        if isinstance(result, ABCSeries) and result.empty:
            result.index = _asfreq_compat(obj.index[:0], freq=self.freq)
            result.name = getattr(obj, 'name', None)
        if self._timegrouper._arrow_dtype is not None:
            result.index = result.index.astype(self._timegrouper._arrow_dtype)
        return result

    @final
    def ffill(self, limit: Optional[int] = None) -> Union[Series, DataFrame]:
        return self._upsample('ffill', limit=limit)

    @final
    def nearest(self, limit: Optional[int] = None) -> Union[Series, DataFrame]:
        return self._upsample('nearest', limit=limit)

    @final
    def bfill(self, limit: Optional[int] = None) -> Union[Series, DataFrame]:
        return self._upsample('bfill', limit=limit)

    @final
    def interpolate(
        self, 
        method: str = 'linear', 
        *, 
        axis: Axis = 0, 
        limit: Optional[int] = None, 
        inplace: bool = False, 
        limit_direction: Literal['forward', 'backward', 'both'] = 'forward', 
        limit_area: Optional[Literal['inside', 'outside']] = None, 
        downcast: Optional[Literal['infer']] = lib.no_default, 
        **kwargs: Any
    ) -> Union[Series, DataFrame]:
        assert downcast is lib.no_default
        result = self._upsample('asfreq')
        obj = self._selected_obj
        is_period_index = isinstance(obj.index, PeriodIndex)
        if not is_period_index:
            final_index = result.index
            if isinstance(final_index, MultiIndex):
                raise NotImplementedError(
                    'Direct interpolation of MultiIndex data frames is not supported. '
                    'If you tried to resample and interpolate on a grouped data frame, '
                    'please use:\n`df.groupby(...).apply(lambda x: x.resample(...).interpolate(...))`\n'
                    'instead, as resampling and interpolation has to be performed for each group independently.'
                )
            missing_data_points_index = obj.index.difference(final_index)
            if len(missing_data_points_index) > 0:
                result = concat([result, obj.loc[missing_data_points_index]]).sort_index()
        result_interpolated = result.interpolate(
            method=method, 
            axis=axis, 
            limit=limit, 
            inplace=inplace, 
            limit_direction=limit_direction, 
            limit_area=limit_area, 
            downcast=downcast, 
            **kwargs
        )
        if is_period_index:
            return result_interpolated
        result_interpolated = result_interpolated.loc[final_index]
        result_interpolated.index = final_index
        return result_interpolated

    @final
    def asfreq(self, fill_value: Any = None) -> Union[Series, DataFrame]:
        return self._upsample('asfreq', fill_value=fill_value)

    @final
    def sum(
        self, 
        numeric_only: bool = False, 
        min_count: int = 0
    ) -> Union[Series, DataFrame]:
        return self._downsample('sum', numeric_only=numeric_only, min_count=min_count)

    @final
    def prod(
        self, 
        numeric_only: bool = False, 
        min_count: int = 0
    ) -> Union[Series, DataFrame]:
        return self._downsample('prod', numeric_only=numeric_only, min_count=min_count)

    @final
    def min(
        self, 
        numeric_only: bool = False, 
        min_count: int = 0
    ) -> Union[Series, DataFrame]:
        return self._downsample('min', numeric_only=numeric_only, min_count=min_count)

    @final
   