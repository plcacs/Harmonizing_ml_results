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
    Dict,
    List,
    Tuple,
    Type,
    TypeVar,
    Generic,
    Sequence,
)
import warnings

import numpy as np
import numpy.typing as npt

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
from pandas._typing import NDFrameT, Axis, IndexLabel, AnyArrayLike, InterpolateOptions, TimedeltaConvertibleTypes, TimestampConvertibleTypes, Frequency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

import pandas.core.algorithms as algos
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import (
    PandasObject,
    SelectionMixin,
)
from pandas.core.generic import (
    NDFrame,
    _shared_docs,
)
from pandas.core.groupby.groupby import (
    BaseGroupBy,
    GroupBy,
    _pipe_template,
    get_groupby,
)
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
from pandas.core.indexes.api import MultiIndex
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    date_range,
)
from pandas.core.indexes.period import (
    PeriodIndex,
    period_range,
)
from pandas.core.indexes.timedeltas import (
    TimedeltaIndex,
    timedelta_range,
)
from pandas.core.reshape.concat import concat

from pandas.tseries.frequencies import (
    is_subperiod,
    is_superperiod,
)
from pandas.tseries.offsets import (
    Day,
    Tick,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
    )

    from pandas._typing import (
        Any,
        AnyArrayLike,
        Axis,
        Concatenate,
        FreqIndexT,
        Frequency,
        IndexLabel,
        InterpolateOptions,
        P,
        Self,
        T,
        TimedeltaConvertibleTypes,
        TimeGrouperOrigin,
        TimestampConvertibleTypes,
        npt,
    )

    from pandas import (
        DataFrame,
        Series,
    )

_shared_docs_kwargs: Dict[str, str] = {}


class Resampler(BaseGroupBy, PandasObject):
    _grouper: BinGrouper
    _timegrouper: TimeGrouper
    binner: Union[DatetimeIndex, TimedeltaIndex, PeriodIndex]  # depends on subclass
    exclusions: frozenset[Hashable] = frozenset()  # for SelectionMixin compat
    _internal_names_set: set[str] = set({"obj", "ax", "_indexer"})

    _attributes: List[str] = [
        "freq",
        "closed",
        "label",
        "convention",
        "origin",
        "offset",
    ]

    def __init__(
        self,
        obj: NDFrame,
        timegrouper: TimeGrouper,
        *,
        gpr_index: Index,
        group_keys: bool = False,
        selection: Optional[IndexLabel] = None,
        include_groups: bool = False,
    ) -> None:
        if include_groups:
            raise ValueError("include_groups=True is no longer allowed.")
        self._timegrouper = timegrouper
        self.keys: Optional[Any] = None
        self.sort: bool = True
        self.group_keys: bool = group_keys
        self.as_index: bool = True

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

    def _get_binner_for_time(self) -> Tuple[Index, npt.NDArray[np.intp], Index]:
        raise AbstractMethodError(self)

    @final
    def _get_binner(self) -> Tuple[Index, BinGrouper]:
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
        func: Tuple[Callable[..., T], str],
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
        func: Union[Callable[Concatenate[Self, P], T], Tuple[Callable[..., T], str]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        return super().pipe(func, *args, **kwargs)

    _agg_see_also_doc: str = dedent(
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

    _agg_examples_doc: str = dedent(
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
    def aggregate(self, func: Optional[Union[str, Callable[..., Any]]] = None, *args: Any, **kwargs: Any) -> Any:
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            how = func
            result = self._groupby_and_aggregate(how, *args, **kwargs)

        return result

    agg = aggregate
    apply = aggregate

    @final
    def transform(self, arg: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return self._selected_obj.groupby(self._timegrouper).transform(
            arg, *args, **kwargs
        )

    def _downsample(self, how: str, **kwargs: Any) -> Any:
        raise AbstractMethodError(self)

    def _upsample(self, f: str, limit: Optional[int] = None, fill_value: Optional[Any] = None) -> Any:
        raise AbstractMethodError(self)

    def _gotitem(self, key: Optional[IndexLabel], ndim: int, subset: Optional[Any] = None) -> Any:
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

    def _groupby_and_aggregate(self, how: str, *args: Any, **kwargs: Any) -> Any:
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
        self,
        groupby: GroupBy,
        key: Optional[Any],
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
    def ffill(self, limit: Optional[int] = None) -> Any:
        return self._upsample("ffill", limit=limit)

    @final
    def nearest(self, limit: Optional[int] = None) -> Any:
        return self._upsample("nearest", limit=limit)

    @final
    def bfill(self, limit: Optional[int] = None) -> Any:
        return self._upsample("bfill", limit=limit)

    @final
    def interpolate(
        self,
        method: InterpolateOptions = "linear",
        *,
        axis: Axis = 0,
        limit: Optional[int] = None,
        inplace: bool = False,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area: Optional[Any] = None,
        downcast: Optional[Any] = lib.no_default,
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
    def asfreq(self, fill_value: Optional[Any] = None) -> Any:
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
        return self._downsample(
            "first", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

    @final
    @doc(GroupBy.last)
    def last(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
    ) -> Any:
        return self._downsample(
            "last", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

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
                mi = MultiIndex.from_product(
                    [obj.columns, ["open", "high", "low", "close"]]
                )
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
    def quantile(self, q: Union[float, List[float], AnyArrayLike] = 0.5, **kwargs: Any) -> Any:
        return self._downsample("quantile", q=q, **kwargs)


class _GroupByMixin(PandasObject, SelectionMixin):
    _attributes: List[str]
    _selection: Optional[IndexLabel] = None
    _groupby: GroupBy
    _timegrouper: TimeGrouper

    def __init__(
        self,
        *,
        parent: Resampler,
        groupby: GroupBy,
        key: Optional[Any] = None,
        selection: Optional[IndexLabel] = None,
    ) -> None:
        for attr in self._attributes:
            setattr(self, attr, getattr(parent, attr))
        self._selection = selection

        self.binner = parent.binner
        self.key = key

        self._groupby = groupby
        self._timegrouper = copy.copy(parent._timegrouper)

        self.ax = parent.ax
        self.obj = parent.obj

    @no_type_check
    def _apply(self, f: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
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
    def _gotitem(self, key: Optional[IndexLabel], ndim: int, subset: Optional[Any] = None) -> Any:
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
    def _resampler_for_grouping(self) -> Type[DatetimeIndexResamplerGroupby]:
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[Index, npt.NDArray[np.intp], Index]:
        if isinstance(self.ax, PeriodIndex):
            return self._timegrouper._get_time_period_bins(self.ax)
        return self._timegrouper._get_time_bins(self.ax)

    def _downsample(self, how: str, **kwargs: Any) -> Any:
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

    def _adjust_binner_for_upsample(self, binner: DatetimeIndex) -> DatetimeIndex:
        if self.closed == "right":
            binner = binner[1:]
        else:
            binner = binner[:-1]
        return binner

    def _upsample(self, method: str, limit: Optional[int] = None, fill_value: Optional[Any] = None) -> Any:
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
            result = obj.reindex(
                res_index, method=method, limit=limit, fill_value=fill_value
            )

        return self._wrap_result(result)

    def _wrap_result(self, result: Any) -> Any:
        result = super()._wrap_result(result)

        if isinstance(self.ax, PeriodIndex) and not isinstance(
            result.index, PeriodIndex
        ):
            if isinstance(result.index, MultiIndex):
                if not isinstance(result.index.levels[-1], PeriodIndex):
                    new_level = result.index.levels[-1].to_period(self.freq)
                    result.index = result.index.set_levels(new_level, level=-1)
            else:
                result.index = result.index.to_period(self.freq)
        return result


class DatetimeIndexResamplerGroupby(_GroupByMixin, DatetimeIndexResampler):
    @property
    def _resampler_cls(self) -> Type[DatetimeIndexResampler]:
        return DatetimeIndexResampler


class PeriodIndexResampler(DatetimeIndexResampler):
    ax: PeriodIndex

    @property
    def _resampler_for_grouping(self) -> Type[PeriodIndexResamplerGroupby]:
        warnings.warn(
            "Resampling a groupby with a PeriodIndex is deprecated. "
            "Cast to DatetimeIndex before resampling instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[Index, npt.NDArray[np.intp], Index]:
        if isinstance(self.ax, DatetimeIndex):
            return super()._get_binner_for_time()
        return self._timegrouper._get_period_bins(self.ax)

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        obj = super()._convert_obj(obj)

        if self._from_selection:
            raise NotImplementedError(
                "Resampling from level= or on= selection "
                "with a PeriodIndex is not currently supported, "
                "use .set_index(...) to explicitly set index"
            )

        if isinstance(obj, DatetimeIndex):
            obj = obj.to_timestamp(how=self.convention)

        return obj

    def _downsample(self, how: str, **kwargs: Any) -> Any:
        if isinstance(self.ax, DatetimeIndex):
            return super()._downsample(how, **kwargs)

        ax = self.ax

        if is_subperiod(ax.freq, self.freq):
            return self._groupby_and_aggregate(how, **kwargs)
        elif is_superperiod(ax.freq, self.freq):
            if how == "ohlc":
                return self._groupby_and_aggregate(how)
            return self.asfreq()
        elif ax.freq == self.freq:
            return self.asfreq()

        raise IncompatibleFrequency(
            f"Frequency {ax.freq} cannot be resampled to {self.freq}, "
            "as they are not sub or super periods"
        )

    def _upsample(self, method: str, limit: Optional[int] = None, fill_value: Optional[Any] = None) -> Any:
        if isinstance(self.ax, DatetimeIndex):
            return super()._upsample(method, limit=limit, fill_value=fill_value)

        ax = self.ax
        obj = self.obj
        new_index = self.binner

        memb = ax.asfreq(self.freq, how=self.convention)

        if method == "asfreq":
            method = None
        indexer = memb.get_indexer(new_index, method=method, limit=limit)
        new_obj = _take_new_index(
            obj,
            indexer,
            new_index,
        )
        return self._wrap_result(new_obj)


class PeriodIndexResamplerGroupby(_GroupByMixin, PeriodIndexResampler):
    @property
    def _resampler_cls(self) -> Type[PeriodIndexResampler]:
        return PeriodIndexResampler


class TimedeltaIndexResampler(DatetimeIndexResampler):
    ax: TimedeltaIndex

    @property
    def _resampler_for_grouping(self) -> Type[TimedeltaIndexResamplerGroupby]:
        return TimedeltaIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[Index, npt.NDArray[np.intp], Index]:
        return self._timegrouper._get_time_delta_bins(self.ax)

    def _adjust_binner_for_upsample(self, binner: TimedeltaIndex) -> TimedeltaIndex:
        return binner


class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    @property
    def _resampler_cls(self) -> Type[TimedeltaIndexResampler]:
        return TimedeltaIndexResampler


def get_resampler(obj: Union[Series, DataFrame], **kwds: Any) -> Resampler:
    tg = TimeGrouper(obj, **kwds)
    return tg._get_resampler(obj)


get_resampler.__doc__ = Resampler.__doc__


def get_resampler_for_grouping(
    groupby: GroupBy,
    rule: Any,
    how: Optional[str] = None,
    fill_method: Optional[str] = None,
    limit: Optional[int] = None,
    on: Optional[str] = None,
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

    origin: TimeGrouperOrigin

    def __init__(
        self,
        obj: Optional[Grouper] = None,
        freq: Frequency = "Min",
        key: Optional[str] = None,
        closed: Optional[Literal["left", "right"]] = None,
        label: Optional[Literal["left", "right"]] = None,
        how: str = "mean",
        fill_method: Optional[str] = None,
        limit: Optional[int] = None,
        convention: Optional[Literal["start", "end", "e", "s"]] = None,
        origin: Union[Literal["epoch", "start", "start_day", "end", "end_day"], TimestampConvertibleTypes] = "start_day",
        offset: Optional[TimedeltaConvertibleTypes] = None,
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
        self._arrow_dtype: Optional[ArrowDtype] = None

        if origin in ("epoch", "start", "start_day", "end", "end_day"):
            self.origin = origin
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
            self.offset = Timed