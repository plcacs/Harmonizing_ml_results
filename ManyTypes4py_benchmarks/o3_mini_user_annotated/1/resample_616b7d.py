#!/usr/bin/env python3
"""Annotated version of the Pandas resampling module with type annotations added."""
from __future__ import annotations

import copy
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Concatenate,
    List,
    Literal,
    NoReturn,
    overload,
    Tuple,
    Union,
    cast,
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
    to_offset,
)
from pandas._typing import NDFrameT, AnyArrayLike, Axis, FreqIndexT, Frequency, IndexLabel, InterpolateOptions, P, Self, T, TimedeltaConvertibleTypes, TimestampConvertibleTypes, npt
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
    from collections.abc import Hashable


_shared_docs_kwargs: dict[str, str] = {}


class Resampler(BaseGroupBy, PandasObject):
    """
    Class for resampling datetimelike data, a groupby-like operation.
    See aggregate, transform, and apply functions on this object.
    It's easiest to use obj.resample(...) to use Resampler.
    """

    _grouper: BinGrouper
    _timegrouper: TimeGrouper
    binner: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex]  # depends on subclass
    exclusions: frozenset[Hashable] = frozenset()  # for SelectionMixin compat
    _internal_names_set = {"obj", "ax", "_indexer"}
    _attributes = [
        "freq",
        "closed",
        "label",
        "convention",
        "origin",
        "offset",
    ]

    def __init__(
        self,
        obj: NDFrameT,
        timegrouper: TimeGrouper,
        *,
        gpr_index: Index,
        group_keys: bool = False,
        selection: Any = None,
        include_groups: bool = False,
    ) -> None:
        if include_groups:
            raise ValueError("include_groups=True is no longer allowed.")
        self._timegrouper = timegrouper
        self.keys = None
        self.sort = True
        self.group_keys = group_keys
        self.as_index = True

        self.obj, self.ax, self._indexer = self._timegrouper._set_grouper(
            self._convert_obj(obj), sort=True, gpr_index=gpr_index
        )
        self.binner, self._grouper = self._get_binner()
        self._selection = selection
        if self._timegrouper.key is not None:
            self.exclusions = frozenset([self._timegrouper.key])
        else:
            self.exclusions = frozenset()

    @final
    def __str__(self) -> str:
        attrs = (
            f"{k}={getattr(self._timegrouper, k)}"
            for k in self._attributes
            if getattr(self._timegrouper, k, None) is not None
        )
        return f"{type(self).__name__} [{', '.join(attrs)}]"

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
            self._timegrouper.key is not None or self._timegrouper.level is not None
        )

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        return obj._consolidate()

    def _get_binner_for_time(self) -> Tuple[
        Union[DatetimeIndex, TimedeltaIndex, PeriodIndex], List[int], Union[DatetimeIndex, TimedeltaIndex, PeriodIndex]
    ]:
        raise AbstractMethodError(self)

    @final
    def _get_binner(self) -> Tuple[
        Union[DatetimeIndex, TimedeltaIndex, PeriodIndex], BinGrouper
    ]:
        binner, bins, binlabels = self._get_binner_for_time()
        assert len(bins) == len(binlabels)
        bin_grouper = BinGrouper(bins, binlabels, indexer=self._indexer)
        return binner, bin_grouper

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...
    @overload
    def pipe(
        self,
        func: tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T: ...

    @final
    @Substitution(
        klass="Resampler",
        examples="""
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4]},
    ...                   index=pd.date_range('2012-08-02', periods=4))
    >>> df
                A
    2012-08-02  1
    2012-08-03  2
    2012-08-04  3
    2012-08-05  4

    To get the difference between each 2-day period's maximum and minimum
    value in one pass, you can do

    >>> df.resample('2D').pipe(lambda x: x.max() - x.min())
                A
    2012-08-02  1
    2012-08-04  1""",
    )
    @Appender(_pipe_template)
    def pipe(
        self,
        func: Callable[Concatenate[Self, P], T] | tuple[Callable[..., T], str],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        return super().pipe(func, *args, **kwargs)

    _agg_see_also_doc = dedent(
        """
    See Also
    --------
    DataFrame.groupby.aggregate : Aggregate using callable, string, dict,
        or list of string/callables.
    DataFrame.resample.transform : Transforms the Series on each group
        based on the given function.
    DataFrame.aggregate: Aggregate using one or more
        operations over the specified axis.
    """
    )

    _agg_examples_doc = dedent(
        """
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
    """
    )

    @final
    @doc(
        _shared_docs["aggregate"],
        see_also=_agg_see_also_doc,
        examples=_agg_examples_doc,
        klass="DataFrame",
        axis="",
    )
    def aggregate(self, func: Any = None, *args: Any, **kwargs: Any) -> Any:
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            how = func
            result = self._groupby_and_aggregate(how, *args, **kwargs)
        return result

    agg = aggregate
    apply = aggregate

    @final
    def transform(self, arg: Any, *args: Any, **kwargs: Any) -> Any:
        return self._selected_obj.groupby(self._timegrouper).transform(
            arg, *args, **kwargs
        )

    def _downsample(self, how: Any, **kwargs: Any) -> Any:
        raise AbstractMethodError(self)

    def _upsample(self, f: Any, limit: int | None = None, fill_value: Any = None) -> Any:
        raise AbstractMethodError(self)

    def _gotitem(self, key: Any, ndim: int, subset: Any = None) -> Any:
        grouper = self._grouper
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                assert subset.ndim == 1
        if ndim == 1:
            assert subset.ndim == 1
        grouped = get_groupby(
            subset, by=None, grouper=grouper, group_keys=self.group_keys
        )
        return grouped

    def _groupby_and_aggregate(self, how: Any, *args: Any, **kwargs: Any) -> Any:
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
            if "Must produce aggregated value" in str(err):
                pass
            else:
                raise
            result = grouped.apply(how, *args, **kwargs)
        return self._wrap_result(result)

    @final
    def _get_resampler_for_grouping(
        self, groupby: GroupBy, key: Any
    ) -> Any:
        return self._resampler_for_grouping(
            groupby=groupby,
            key=key,
            parent=self,
        )

    def _wrap_result(self, result: Any) -> Any:
        obj = self.obj
        if (
            isinstance(result, ABCDataFrame)
            and len(result) == 0
            and not isinstance(result.index, PeriodIndex)
        ):
            result = result.set_index(
                _asfreq_compat(obj.index[:0], freq=self.freq), append=True
            )
        if isinstance(result, ABCSeries) and self._selection is not None:
            result.name = self._selection
        if isinstance(result, ABCSeries) and result.empty:
            result.index = _asfreq_compat(obj.index[:0], freq=self.freq)
            result.name = getattr(obj, "name", None)
        if self._timegrouper._arrow_dtype is not None:
            result.index = result.index.astype(self._timegrouper._arrow_dtype)
        return result

    @final
    def ffill(self, limit: int | None = None) -> Any:
        return self._upsample("ffill", limit=limit)

    @final
    def nearest(self, limit: int | None = None) -> Any:
        return self._upsample("nearest", limit=limit)

    @final
    def bfill(self, limit: int | None = None) -> Any:
        return self._upsample("bfill", limit=limit)

    @final
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: int | None = None,
        inplace: bool = False,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area: Any = None,
        downcast: Any = lib.no_default,
        **kwargs: Any,
    ) -> Any:
        assert downcast is lib.no_default
        result = self._upsample("asfreq")
        obj = self._selected_obj
        is_period_index = isinstance(obj.index, PeriodIndex)
        if not is_period_index:
            final_index = result.index
            if isinstance(final_index, MultiIndex):
                raise NotImplementedError(
                    "Direct interpolation of MultiIndex data frames is not "
                    "supported. If you tried to resample and interpolate on a "
                    "grouped data frame, please use:\n"
                    "`df.groupby(...).apply(lambda x: x.resample(...)."
                    "interpolate(...))`"
                    "\ninstead, as resampling and interpolation has to be "
                    "performed for each group independently."
                )
            missing_data_points_index = obj.index.difference(final_index)
            if len(missing_data_points_index) > 0:
                result = concat(
                    [result, obj.loc[missing_data_points_index]]
                ).sort_index()
        result_interpolated = result.interpolate(
            method=method,
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs,
        )
        if is_period_index:
            return result_interpolated
        result_interpolated = result_interpolated.loc[final_index]
        result_interpolated.index = final_index
        return result_interpolated

    @final
    def asfreq(self, fill_value: Any = None) -> Any:
        return self._upsample("asfreq", fill_value=fill_value)

    @final
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> Any:
        return self._downsample("sum", numeric_only=numeric_only, min_count=min_count)

    @final
    def prod(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> Any:
        return self._downsample("prod", numeric_only=numeric_only, min_count=min_count)

    @final
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> Any:
        return self._downsample("min", numeric_only=numeric_only, min_count=min_count)

    @final
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> Any:
        return self._downsample("max", numeric_only=numeric_only, min_count=min_count)

    @final
    @doc(GroupBy.first)
    def first(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
    ) -> Any:
        return self._downsample("first", numeric_only=numeric_only, min_count=min_count, skipna=skipna)

    @final
    @doc(GroupBy.last)
    def last(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
    ) -> Any:
        return self._downsample("last", numeric_only=numeric_only, min_count=min_count, skipna=skipna)

    @final
    def median(self, numeric_only: bool = False) -> Any:
        return self._downsample("median", numeric_only=numeric_only)

    @final
    def mean(
        self,
        numeric_only: bool = False,
    ) -> Any:
        return self._downsample("mean", numeric_only=numeric_only)

    @final
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Any:
        return self._downsample("std", ddof=ddof, numeric_only=numeric_only)

    @final
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Any:
        return self._downsample("var", ddof=ddof, numeric_only=numeric_only)

    @final
    def sem(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Any:
        return self._downsample("sem", ddof=ddof, numeric_only=numeric_only)

    @final
    @doc(GroupBy.ohlc)
    def ohlc(self) -> Any:
        ax = self.ax
        obj = self._obj_with_exclusions
        if len(ax) == 0:
            obj = obj.copy()
            obj.index = _asfreq_compat(obj.index, self.freq)
            if obj.ndim == 1:
                obj = obj.to_frame()
                obj = obj.reindex(["open", "high", "low", "close"], axis=1)
            else:
                mi = MultiIndex.from_product([obj.columns, ["open", "high", "low", "close"]])
                obj = obj.reindex(mi, axis=1)
            return obj
        return self._downsample("ohlc")

    @final
    def nunique(self) -> Any:
        return self._downsample("nunique")

    @final
    @doc(GroupBy.size)
    def size(self) -> Any:
        result = self._downsample("size")
        if isinstance(result, ABCDataFrame) and not result.empty:
            result = result.stack()
        if not len(self.ax):
            from pandas import Series
            if self._selected_obj.ndim == 1:
                name = self._selected_obj.name
            else:
                name = None
            result = Series([], index=result.index, dtype="int64", name=name)
        return result

    @final
    @doc(GroupBy.count)
    def count(self) -> Any:
        result = self._downsample("count")
        if not len(self.ax):
            if self._selected_obj.ndim == 1:
                result = type(self._selected_obj)(
                    [], index=result.index, dtype="int64", name=self._selected_obj.name
                )
            else:
                from pandas import DataFrame
                result = DataFrame(
                    [], index=result.index, columns=result.columns, dtype="int64"
                )
        return result

    @final
    def quantile(self, q: float | list[float] | AnyArrayLike = 0.5, **kwargs: Any) -> Any:
        return self._downsample("quantile", q=q, **kwargs)


class _GroupByMixin(PandasObject, SelectionMixin):
    _attributes: List[str]
    _selection: IndexLabel | None = None
    _groupby: GroupBy
    _timegrouper: TimeGrouper

    def __init__(
        self,
        *,
        parent: Resampler,
        groupby: GroupBy,
        key: Any = None,
        selection: IndexLabel | None = None,
    ) -> None:
        assert isinstance(groupby, GroupBy), type(groupby)
        assert isinstance(parent, Resampler), type(parent)
        for attr in self._attributes:
            setattr(self, attr, getattr(parent, attr))
        self._selection = selection
        self.binner = parent.binner
        self.key = key
        self._groupby = groupby
        self._timegrouper = copy.copy(parent._timegrouper)
        self.ax = parent.ax
        self.obj = parent.obj

    def _apply(self, f: Any, *args: Any, **kwargs: Any) -> Any:
        def func(x: Any) -> Any:
            x = self._resampler_cls(x, timegrouper=self._timegrouper, gpr_index=self.ax)
            if isinstance(f, str):
                return getattr(x, f)(**kwargs)
            return x.apply(f, *args, **kwargs)
        result = self._groupby.apply(func)
        return self._wrap_result(result)
    _upsample = _apply
    _downsample = _apply
    _groupby_and_aggregate = _apply

    @final
    def _gotitem(self, key: Any, ndim: int, subset: Any = None) -> _GroupByMixin:
        if subset is None:
            subset = self.obj
            if key is not None:
                subset = subset[key]
            else:
                assert subset.ndim == 1
        try:
            if isinstance(key, list) and self.key not in key and self.key is not None:
                key.append(self.key)
            groupby = self._groupby[key]
        except IndexError:
            groupby = self._groupby
        selection = self._infer_selection(key, subset)
        new_rs = type(self)(
            groupby=groupby,
            parent=cast(Resampler, self),
            selection=selection,
        )
        return new_rs


class DatetimeIndexResampler(Resampler):
    ax: DatetimeIndex

    @property
    def _resampler_for_grouping(self) -> type[DatetimeIndexResamplerGroupby]:
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[DatetimeIndex, List[int], DatetimeIndex]:
        if isinstance(self.ax, PeriodIndex):
            return self._timegrouper._get_time_period_bins(self.ax)
        return self._timegrouper._get_time_bins(self.ax)

    def _downsample(self, how: Any, **kwargs: Any) -> Any:
        ax = self.ax
        obj = self._obj_with_exclusions
        if not len(ax):
            obj = obj.copy()
            obj.index = obj.index._with_freq(self.freq)
            assert obj.index.freq == self.freq, (obj.index.freq, self.freq)
            return obj
        if (
            (ax.freq is not None or ax.inferred_freq is not None)
            and len(self._grouper.binlabels) > len(ax)
            and how is None
        ):
            return self.asfreq()
        result = obj.groupby(self._grouper).aggregate(how, **kwargs)
        return self._wrap_result(result)

    def _adjust_binner_for_upsample(self, binner: Union[DatetimeIndex, Any]) -> Union[DatetimeIndex, Any]:
        if self.closed == "right":
            binner = binner[1:]
        else:
            binner = binner[:-1]
        return binner

    def _upsample(self, method: Any, limit: int | None = None, fill_value: Any = None) -> Any:
        if self._from_selection:
            raise ValueError(
                "Upsampling from level= or on= selection "
                "is not supported, use .set_index(...) "
                "to explicitly set index to datetime-like"
            )
        ax = self.ax
        obj = self._selected_obj
        binner = self.binner
        res_index = self._adjust_binner_for_upsample(binner)
        if (
            limit is None
            and to_offset(ax.inferred_freq) == self.freq
            and len(obj) == len(res_index)
        ):
            result = obj.copy()
            result.index = res_index
        else:
            if method == "asfreq":
                method = None
            result = obj.reindex(res_index, method=method, limit=limit, fill_value=fill_value)
        return self._wrap_result(result)

    def _wrap_result(self, result: Any) -> Any:
        result = super()._wrap_result(result)
        if isinstance(self.ax, PeriodIndex) and not isinstance(result.index, PeriodIndex):
            if isinstance(result.index, MultiIndex):
                if not isinstance(result.index.levels[-1], PeriodIndex):
                    new_level = result.index.levels[-1].to_period(self.freq)
                    result.index = result.index.set_levels(new_level, level=-1)
            else:
                result.index = result.index.to_period(self.freq)
        return result


class DatetimeIndexResamplerGroupby(_GroupByMixin, DatetimeIndexResampler):
    @property
    def _resampler_cls(self) -> type[DatetimeIndexResampler]:
        return DatetimeIndexResampler


class PeriodIndexResampler(DatetimeIndexResampler):
    ax: PeriodIndex

    @property
    def _resampler_for_grouping(self) -> type[PeriodIndexResamplerGroupby]:
        warnings.warn(
            "Resampling a groupby with a PeriodIndex is deprecated. "
            "Cast to DatetimeIndex before resampling instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[Union[DatetimeIndex, PeriodIndex], List[int], Union[DatetimeIndex, PeriodIndex]]:
        if isinstance(self.ax, DatetimeIndex):
            return super()._get_binner_for_time()
        return self._timegrouper._get_period_bins(self.ax)

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        obj = super()._convert_obj(obj)
        if self._from_selection:
            msg = (
                "Resampling from level= or on= selection "
                "with a PeriodIndex is not currently supported, "
                "use .set_index(...) to explicitly set index"
            )
            raise NotImplementedError(msg)
        if isinstance(obj, DatetimeIndex):
            obj = obj.to_timestamp(how=self.convention)
        return obj

    def _downsample(self, how: Any, **kwargs: Any) -> Any:
        if is_subperiod(self.ax.freq, self.freq):
            return self._groupby_and_aggregate(how, **kwargs)
        elif is_superperiod(self.ax.freq, self.freq):
            if how == "ohlc":
                return self._groupby_and_aggregate(how)
            return self.asfreq()
        elif self.ax.freq == self.freq:
            return self.asfreq()
        raise IncompatibleFrequency(
            f"Frequency {self.ax.freq} cannot be resampled to {self.freq}, "
            "as they are not sub or super periods"
        )

    def _upsample(self, method: Any, limit: int | None = None, fill_value: Any = None) -> Any:
        if isinstance(self.ax, DatetimeIndex):
            return super()._upsample(method, limit=limit, fill_value=fill_value)
        ax = self.ax
        obj = self.obj
        new_index = self.binner
        memb = ax.asfreq(self.freq, how=self.convention)
        if method == "asfreq":
            method = None
        indexer = memb.get_indexer(new_index, method=method, limit=limit)
        new_obj = _take_new_index(obj, indexer, new_index)
        return self._wrap_result(new_obj)


class PeriodIndexResamplerGroupby(_GroupByMixin, PeriodIndexResampler):
    @property
    def _resampler_cls(self) -> type[PeriodIndexResampler]:
        return PeriodIndexResampler


class TimedeltaIndexResampler(DatetimeIndexResampler):
    ax: TimedeltaIndex

    @property
    def _resampler_for_grouping(self) -> type[TimedeltaIndexResamplerGroupby]:
        return TimedeltaIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[TimedeltaIndex, List[int], TimedeltaIndex]:
        return self._timegrouper._get_time_delta_bins(self.ax)

    def _adjust_binner_for_upsample(self, binner: Any) -> Any:
        return binner


class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    @property
    def _resampler_cls(self) -> type[TimedeltaIndexResampler]:
        return TimedeltaIndexResampler


def get_resampler(obj: Union[ABCSeries, ABCDataFrame], **kwds: Any) -> Resampler:
    tg = TimeGrouper(obj, **kwds)  # type: ignore[arg-type]
    return tg._get_resampler(obj)


get_resampler.__doc__ = Resampler.__doc__


def get_resampler_for_grouping(
    groupby: GroupBy,
    rule: Any,
    how: Any = None,
    fill_method: Any = None,
    limit: int | None = None,
    on: Any = None,
    **kwargs: Any,
) -> Resampler:
    tg = TimeGrouper(freq=rule, key=on, **kwargs)
    resampler = tg._get_resampler(groupby.obj)
    return resampler._get_resampler_for_grouping(groupby=groupby, key=tg.key)


class TimeGrouper(Grouper):
    _attributes = Grouper._attributes + (
        "closed",
        "label",
        "how",
        "convention",
        "origin",
        "offset",
    )

    origin: Union[str, Timestamp]
    closed: Literal["left", "right"]
    label: Literal["left", "right"]
    convention: Literal["start", "end", "e", "s"]
    how: str
    fill_method: Any
    limit: int | None
    group_keys: bool
    _arrow_dtype: ArrowDtype | None

    def __init__(
        self,
        obj: Grouper | None = None,
        freq: Frequency = "Min",
        key: str | None = None,
        closed: Literal["left", "right"] | None = None,
        label: Literal["left", "right"] | None = None,
        how: str = "mean",
        fill_method: Any = None,
        limit: int | None = None,
        convention: Literal["start", "end", "e", "s"] | None = None,
        origin: Union[Literal["epoch", "start", "start_day", "end", "end_day"], TimestampConvertibleTypes] = "start_day",
        offset: TimedeltaConvertibleTypes | None = None,
        group_keys: bool = False,
        **kwargs: Any,
    ) -> None:
        if label not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {label} for `label`")
        if closed not in {None, "left", "right"}:
            raise ValueError(f"Unsupported value {closed} for `closed`")
        if convention not in {None, "start", "end", "e", "s"}:
            raise ValueError(f"Unsupported value {convention} for `convention`")
        if (
            (key is None and obj is not None and isinstance(obj.index, PeriodIndex))
            or (
                key is not None
                and obj is not None
                and getattr(obj[key], "dtype", None) == "period"
            )
        ):
            freq = to_offset(freq, is_period=True)
        else:
            freq = to_offset(freq)
        end_types = {"ME", "YE", "QE", "BME", "BYE", "BQE", "W"}
        rule = freq.rule_code
        if rule in end_types or ("-" in rule and rule[: rule.find("-")] in end_types):
            if closed is None:
                closed = "right"
            if label is None:
                label = "right"
        else:
            if origin in ["end", "end_day"]:
                if closed is None:
                    closed = "right"
                if label is None:
                    label = "right"
            else:
                if closed is None:
                    closed = "left"
                if label is None:
                    label = "left"
        self.closed = closed
        self.label = label
        self.convention = convention if convention is not None else "e"
        self.how = how
        self.fill_method = fill_method
        self.limit = limit
        self.group_keys = group_keys
        self._arrow_dtype = None
        if origin in ("epoch", "start", "start_day", "end", "end_day"):
            self.origin = origin  # type: ignore[assignment]
        else:
            try:
                self.origin = Timestamp(origin)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    "'origin' should be equal to 'epoch', 'start', 'start_day', "
                    "'end', 'end_day' or "
                    f"should be a Timestamp convertible type. Got '{origin}' instead."
                ) from err
        try:
            self.offset = Timedelta(offset) if offset is not None else None
        except (ValueError, TypeError) as err:
            raise ValueError(
                "'offset' should be a Timedelta convertible type. "
                f"Got '{offset}' instead."
            ) from err
        kwargs["sort"] = True
        super().__init__(freq=freq, key=key, **kwargs)

    def _get_resampler(self, obj: NDFrame) -> Resampler:
        _, ax, _ = self._set_grouper(obj, gpr_index=None)
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        elif isinstance(ax, PeriodIndex):
            if isinstance(ax, PeriodIndex):
                warnings.warn(
                    "Resampling with a PeriodIndex is deprecated. "
                    "Cast index to DatetimeIndex before resampling instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            return PeriodIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(
                obj,
                timegrouper=self,
                group_keys=self.group_keys,
                gpr_index=ax,
            )
        raise TypeError(
            "Only valid with DatetimeIndex, "
            "TimedeltaIndex or PeriodIndex, "
            f"but got an instance of '{type(ax).__name__}'"
        )

    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True
    ) -> Tuple[BinGrouper, NDFrameT]:
        r = self._get_resampler(obj)
        return r._grouper, cast(NDFrameT, r.obj)

    def _get_time_bins(self, ax: DatetimeIndex) -> Tuple[DatetimeIndex, List[int], DatetimeIndex]:
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )
        if len(ax) == 0:
            binner = labels = DatetimeIndex(
                data=[], freq=self.freq, name=ax.name, dtype=ax.dtype
            )
            return binner, [], labels
        first, last = _get_timestamp_range_edges(
            ax.min(),
            ax.max(),
            self.freq,
            unit=ax.unit,
            closed=self.closed,
            origin=self.origin,
            offset=self.offset,
        )
        binner = labels = date_range(
            freq=self.freq,
            start=first,
            end=last,
            tz=ax.tz,
            name=ax.name,
            ambiguous=True,
            nonexistent="shift_forward",
            unit=ax.unit,
        )
        ax_values = ax.asi8
        binner, bin_edges = self._adjust_bin_edges(binner, ax_values)
        bins = lib.generate_bins_dt64(
            ax_values, bin_edges, self.closed, hasnans=ax.hasnans
        )
        if self.closed == "right":
            labels = binner
            if self.label == "right":
                labels = labels[1:]
        elif self.label == "right":
            labels = labels[1:]
        if ax.hasnans:
            binner = binner.insert(0, NaT)
            labels = labels.insert(0, NaT)
        if len(bins) < len(labels):
            labels = labels[: len(bins)]
        return binner, bins, labels

    def _adjust_bin_edges(
        self, binner: DatetimeIndex, ax_values: npt.NDArray[np.int64]
    ) -> Tuple[DatetimeIndex, npt.NDArray[np.int64]]:
        if self.freq.name in ("BME", "ME", "W") or self.freq.name.split("-")[0] in (
            "BQE",
            "BYE",
            "QE",
            "YE",
            "W",
        ):
            if self.closed == "right":
                edges_dti = binner.tz_localize(None)
                edges_dti = (
                    edges_dti
                    + Timedelta(days=1, unit=edges_dti.unit).as_unit(edges_dti.unit)
                    - Timedelta(1, unit=edges_dti.unit).as_unit(edges_dti.unit)
                )
                bin_edges = edges_dti.tz_localize(binner.tz).asi8
            else:
                bin_edges = binner.asi8
            if bin_edges[-2] > ax_values.max():
                bin_edges = bin_edges[:-1]
                binner = binner[:-1]
        else:
            bin_edges = binner.asi8
        return binner, bin_edges

    def _get_time_delta_bins(self, ax: TimedeltaIndex) -> Tuple[TimedeltaIndex, List[int], TimedeltaIndex]:
        if not isinstance(ax, TimedeltaIndex):
            raise TypeError(
                "axis must be a TimedeltaIndex, but got "
                f"an instance of {type(ax).__name__}"
            )
        if not isinstance(self.freq, Tick):
            raise ValueError(
                "Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
                f"e.g. '24h' or '3D', not {self.freq}"
            )
        if not len(ax):
            binner = labels = TimedeltaIndex(data=[], freq=self.freq, name=ax.name)
            return binner, [], labels
        start, end = ax.min(), ax.max()
        if self.closed == "right":
            end += self.freq
        labels = binner = timedelta_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )
        end_stamps = labels
        if self.closed == "left":
            end_stamps += self.freq
        bins = ax.searchsorted(end_stamps, side=self.closed)
        if self.offset:
            labels += self.offset
        return binner, bins, labels

    def _get_time_period_bins(self, ax: DatetimeIndex) -> Tuple[PeriodIndex, List[int], PeriodIndex]:
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                "axis must be a DatetimeIndex, but got "
                f"an instance of {type(ax).__name__}"
            )
        freq = self.freq
        if len(ax) == 0:
            binner = labels = PeriodIndex(data=[], freq=freq, name=ax.name, dtype=ax.dtype)
            return binner, [], labels
        labels = binner = period_range(start=ax[0], end=ax[-1], freq=freq, name=ax.name)
        end_stamps = (labels + freq).asfreq(freq, "s").to_timestamp()
        if ax.tz:
            end_stamps = end_stamps.tz_localize(ax.tz)
        bins = ax.searchsorted(end_stamps, side="left")
        return binner, bins, labels

    def _get_period_bins(self, ax: PeriodIndex) -> Tuple[PeriodIndex, np.ndarray, PeriodIndex]:
        if not isinstance(ax, PeriodIndex):
            raise TypeError(
                "axis must be a PeriodIndex, but got "
                f"an instance of {type(ax).__name__}"
            )
        memb = ax.asfreq(self.freq, how=self.convention)
        nat_count = 0
        if memb.hasnans:
            nat_count = int(np.sum(memb._isnan))  # type: ignore[assignment]
            memb = memb[~memb._isnan]
        if not len(memb):
            bins = np.array([], dtype=np.int64)
            binner = labels = PeriodIndex(data=[], freq=self.freq, name=ax.name)
            if len(ax) > 0:
                binner, bins, labels = _insert_nat_bin(binner, bins, labels, len(ax))
            return binner, bins, labels
        freq_mult = self.freq.n
        start = ax.min().asfreq(self.freq, how=self.convention)
        end = ax.max().asfreq(self.freq, how="end")
        bin_shift = 0
        if isinstance(self.freq, Tick):
            p_start, end = _get_period_range_edges(
                start,
                end,
                self.freq,
                closed=self.closed,
                origin=self.origin,
                offset=self.offset,
            )
            start_offset = Period(start, self.freq) - Period(p_start, self.freq)
            bin_shift = start_offset.n % freq_mult  # type: ignore[union-attr]
            start = p_start
        labels = binner = period_range(
            start=start, end=end, freq=self.freq, name=ax.name
        )
        i8 = memb.asi8
        expected_bins_count = len(binner) * freq_mult
        i8_extend = expected_bins_count - (i8[-1] - i8[0])
        rng = np.arange(i8[0], i8[-1] + i8_extend, freq_mult)
        rng += freq_mult
        rng -= bin_shift
        prng = type(memb._data)(rng, dtype=memb.dtype)
        bins = memb.searchsorted(prng, side="left")
        if nat_count > 0:
            binner, bins, labels = _insert_nat_bin(binner, bins, labels, nat_count)
        return binner, bins, labels

    def _set_grouper(
        self, obj: NDFrameT, sort: bool = False, *, gpr_index: Index | None = None
    ) -> Tuple[NDFrameT, Index, npt.NDArray[np.intp] | None]:
        obj, ax, indexer = super()._set_grouper(obj, sort, gpr_index=gpr_index)
        if isinstance(ax.dtype, ArrowDtype) and ax.dtype.kind in "Mm":
            self._arrow_dtype = ax.dtype
            ax = Index(
                cast(ArrowExtensionArray, ax.array)._maybe_convert_datelike_array()
            )
        return obj, ax, indexer


@overload
def _take_new_index(
    obj: ABCDataFrame, indexer: npt.NDArray[np.intp], new_index: Index
) -> ABCDataFrame: ...
@overload
def _take_new_index(
    obj: ABCSeries, indexer: npt.NDArray[np.intp], new_index: Index
) -> ABCSeries: ...

def _take_new_index(
    obj: Union[ABCDataFrame, ABCSeries],
    indexer: npt.NDArray[np.intp],
    new_index: Index,
) -> Union[ABCDataFrame, ABCSeries]:
    from pandas import Series, DataFrame
    if isinstance(obj, ABCSeries):
        new_values = algos.take_nd(obj._values, indexer)
        return obj._constructor(new_values, index=new_index, name=obj.name)
    elif isinstance(obj, ABCDataFrame):
        new_mgr = obj._mgr.reindex_indexer(new_axis=new_index, indexer=indexer, axis=1)
        return obj._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
    else:
        raise ValueError("'obj' should be either a Series or a DataFrame")


def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    unit: str,
    closed: Literal["right", "left"] = "left",
    origin: Union[str, Timestamp],
    offset: Timedelta | None = None,
) -> Tuple[Timestamp, Timestamp]:
    if isinstance(freq, Tick):
        index_tz = first.tz
        if isinstance(origin, Timestamp) and (origin.tz is None) != (index_tz is None):
            raise ValueError("The origin must have the same timezone as the index.")
        if origin == "epoch":
            origin = Timestamp("1970-01-01", tz=index_tz)
        if isinstance(freq, Day):
            first = first.tz_localize(None)
            last = last.tz_localize(None)
            if isinstance(origin, Timestamp):
                origin = origin.tz_localize(None)
        first, last = _adjust_dates_anchored(
            first, last, freq, closed=closed, origin=origin, offset=offset, unit=unit
        )
        if isinstance(freq, Day):
            first = first.tz_localize(index_tz)
            last = last.tz_localize(index_tz, nonexistent="shift_forward")
    else:
        first = first.normalize()
        last = last.normalize()
        if closed == "left":
            first = Timestamp(freq.rollback(first))
        else:
            first = Timestamp(first - freq)
        last = Timestamp(last + freq)
    return first, last


def _get_period_range_edges(
    first: Period,
    last: Period,
    freq: BaseOffset,
    closed: Literal["right", "left"] = "left",
    origin: Union[str, Timestamp] = "start_day",
    offset: Timedelta | None = None,
) -> Tuple[Period, Period]:
    if not all(isinstance(obj, Period) for obj in [first, last]):
        raise TypeError("'first' and 'last' must be instances of type Period")
    first_ts = first.to_timestamp()
    last_ts = last.to_timestamp()
    adjust_first = not freq.is_on_offset(first_ts)
    adjust_last = freq.is_on_offset(last_ts)
    first_ts, last_ts = _get_timestamp_range_edges(
        first_ts, last_ts, freq, unit="ns", closed=closed, origin=origin, offset=offset
    )
    first = (first_ts + int(adjust_first) * freq).to_period(freq)
    last = (last_ts - int(adjust_last) * freq).to_period(freq)
    return first, last


def _insert_nat_bin(
    binner: PeriodIndex, bins: np.ndarray, labels: PeriodIndex, nat_count: int
) -> Tuple[PeriodIndex, np.ndarray, PeriodIndex]:
    assert nat_count > 0
    bins += nat_count
    bins = np.insert(bins, 0, nat_count)
    binner = binner.insert(0, NaT)  # type: ignore[assignment]
    labels = labels.insert(0, NaT)  # type: ignore[assignment]
    return binner, bins, labels


def _adjust_dates_anchored(
    first: Timestamp,
    last: Timestamp,
    freq: Tick,
    closed: Literal["right", "left"] = "right",
    origin: Union[str, Timestamp] = "start_day",
    offset: Timedelta | None = None,
    unit: str = "ns",
) -> Tuple[Timestamp, Timestamp]:
    first = first.as_unit(unit)
    last = last.as_unit(unit)
    if offset is not None:
        offset = offset.as_unit(unit)
    freq_value = Timedelta(freq).as_unit(unit)._value
    origin_timestamp = 0
    if origin == "start_day":
        origin_timestamp = first.normalize()._value
    elif origin == "start":
        origin_timestamp = first._value
    elif isinstance(origin, Timestamp):
        origin_timestamp = origin.as_unit(unit)._value
    elif origin in ["end", "end_day"]:
        origin_last = last if origin == "end" else last.ceil("D")
        sub_freq_times = (origin_last._value - first._value) // freq_value
        if closed == "left":
            sub_freq_times += 1
        first = origin_last - sub_freq_times * freq
        origin_timestamp = first._value
    origin_timestamp += offset._value if offset else 0
    first_tzinfo = first.tzinfo
    last_tzinfo = last.tzinfo
    if first_tzinfo is not None:
        first = first.tz_convert("UTC")
    if last_tzinfo is not None:
        last = last.tz_convert("UTC")
    foffset = (first._value - origin_timestamp) % freq_value
    loffset = (last._value - origin_timestamp) % freq_value
    if closed == "right":
        if foffset > 0:
            fresult_int = first._value - foffset
        else:
            fresult_int = first._value - freq_value
        if loffset > 0:
            lresult_int = last._value + (freq_value - loffset)
        else:
            lresult_int = last._value
    else:
        if foffset > 0:
            fresult_int = first._value - foffset
        else:
            fresult_int = first._value
        if loffset > 0:
            lresult_int = last._value + (freq_value - loffset)
        else:
            lresult_int = last._value + freq_value
    fresult = Timestamp(fresult_int, unit=unit)
    lresult = Timestamp(lresult_int, unit=unit)
    if first_tzinfo is not None:
        fresult = fresult.tz_localize("UTC").tz_convert(first_tzinfo)
    if last_tzinfo is not None:
        lresult = lresult.tz_localize("UTC").tz_convert(last_tzinfo)
    return fresult, lresult


def asfreq(
    obj: NDFrameT,
    freq: Any,
    method: Any = None,
    how: Any = None,
    normalize: bool = False,
    fill_value: Any = None,
) -> NDFrameT:
    if isinstance(obj.index, PeriodIndex):
        if method is not None:
            raise NotImplementedError("'method' argument is not supported")
        if how is None:
            how = "E"
        if isinstance(freq, BaseOffset):
            if hasattr(freq, "_period_dtype_code"):
                freq = PeriodDtype(freq)._freqstr
        new_obj = obj.copy()
        new_obj.index = obj.index.asfreq(freq, how=how)
    elif len(obj.index) == 0:
        new_obj = obj.copy()
        new_obj.index = _asfreq_compat(obj.index, freq)
    else:
        unit = None
        if isinstance(obj.index, DatetimeIndex):
            unit = obj.index.unit
        dti = date_range(obj.index.min(), obj.index.max(), freq=freq, unit=unit)
        dti.name = obj.index.name
        new_obj = obj.reindex(dti, method=method, fill_value=fill_value)
        if normalize:
            new_obj.index = new_obj.index.normalize()
    return new_obj


def _asfreq_compat(index: FreqIndexT, freq: Any) -> FreqIndexT:
    if len(index) != 0:
        raise ValueError(
            "Can only set arbitrary freq for empty DatetimeIndex or TimedeltaIndex"
        )
    if isinstance(index, PeriodIndex):
        new_index = index.asfreq(freq=freq)
    elif isinstance(index, DatetimeIndex):
        new_index = DatetimeIndex([], dtype=index.dtype, freq=freq, name=index.name)
    elif isinstance(index, TimedeltaIndex):
        new_index = TimedeltaIndex([], dtype=index.dtype, freq=freq, name=index.name)
    else:
        raise TypeError(type(index))
    return new_index
