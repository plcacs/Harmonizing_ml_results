from __future__ import annotations
import copy
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    AnyArrayLike,
    Callable,
    Hashable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
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
from pandas._typing import NDFrameT
from pandas import DataFrame, Series
from pandas.errors import AbstractMethodError
from pandas.util._decorators import Appender, Substitution, doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.dtypes import ArrowDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
import pandas.core.algorithms as algos
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.base import PandasObject, SelectionMixin
from pandas.core.generic import NDFrame
from pandas.core.groupby.groupby import BaseGroupBy, GroupBy, get_groupby
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
    from collections.abc import Mapping

_shared_docs_kwargs = {}


class Resampler(BaseGroupBy, PandasObject):
    """
    Class for resampling datetimelike data, a groupby-like operation.
    See aggregate, transform, and apply functions on this object.

    It's easiest to use obj.resample(...) to use Resampler.

    Parameters
    ----------
    obj : Series or DataFrame
    groupby : TimeGrouper

    Returns
    -------
    a Resampler of the appropriate type

    Notes
    -----
    After resampling, see aggregate, apply, and transform functions.
    """

    exclusions: frozenset = frozenset()
    _internal_names_set: set = set({"obj", "ax", "_indexer"})
    _attributes: list[str] = [
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
        gpr_index: Optional[Index] = None,
        group_keys: bool = False,
        selection: Optional[Hashable] = None,
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
        self._selection: Optional[Hashable] = selection
        if self._timegrouper.key is not None:
            self.exclusions = frozenset([self._timegrouper.key])
        else:
            self.exclusions = frozenset()

    @final
    def __str__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
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
        if isinstance(self.obj, (ABCSeries, ABCDataFrame)) and attr in self.obj:
            return self[attr]
        return object.__getattribute__(self, attr)

    @final
    @property
    def _from_selection(self) -> bool:
        """
        Is the resampling from a DataFrame column or MultiIndex level.
        """
        return (
            self._timegrouper is not None
            and (self._timegrouper.key is not None or self._timegrouper.level is not None)
        )

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        """
        Provide any conversions for the object in order to correctly handle.

        Parameters
        ----------
        obj : Series or DataFrame

        Returns
        -------
        Series or DataFrame
        """
        return obj._consolidate()

    def _get_binner_for_time(self) -> Tuple[Index, np.ndarray, Index]:
        raise AbstractMethodError(self)

    @final
    def _get_binner(self) -> Tuple[Index, BinGrouper]:
        """
        Create the BinGrouper, assume that self.set_grouper(obj)
        has already been called.
        """
        binner, bins, binlabels = self._get_binner_for_time()
        assert len(bins) == len(binlabels)
        bin_grouper = BinGrouper(bins, binlabels, indexer=self._indexer)
        return (binner, bin_grouper)

    @overload
    def pipe(
        self,
        func: Callable[[Resampler], Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...

    @overload
    def pipe(
        self,
        func: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...

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
    2012-08-04  1"""
    )
    @Appender(_pipe_template)
    def pipe(
        self,
        func: Union[Callable[[Resampler], Any], str],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
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
    def aggregate(
        self,
        func: Optional[Union[Callable[..., Any], str, Sequence[Union[str, Callable[..., Any]]], Mapping[str, Union[str, Callable[..., Any]]]]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]:
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
        **kwargs: Any,
    ) -> Series:
        """
        Call function producing a like-indexed Series on each group.

        Return a Series with the transformed values.

        Parameters
        ----------
        arg : function
            To apply to each group. Should return a Series with the same index.
        *args, **kwargs
            Additional arguments and keywords.

        Returns
        -------
        Series
            A Series with the transformed values, maintaining the same index as
            the original object.

        See Also
        --------
        core.resample.Resampler.apply : Apply a function along each group.
        core.resample.Resampler.aggregate : Aggregate using one or more operations
            over the specified axis.

        Examples
        --------
        >>> s = pd.Series([1, 2], index=pd.date_range("20180101", periods=2, freq="1h"))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> resampled = s.resample("15min")
        >>> resampled.transform(lambda x: (x - x.mean()) / x.std())
        2018-01-01 00:00:00   NaN
        2018-01-01 01:00:00   NaN
        Freq: h, dtype: float64
        """
        return self._selected_obj.groupby(self._timegrouper).transform(
            arg, *args, **kwargs
        )

    def _downsample(self, how: Any, **kwargs: Any) -> Union[DataFrame, Series]:
        raise AbstractMethodError(self)

    def _upsample(
        self, f: str, limit: Optional[int] = None, fill_value: Optional[Any] = None
    ) -> Union[DataFrame, Series]:
        raise AbstractMethodError(self)

    def _gotitem(
        self,
        key: Any,
        ndim: int,
        subset: Optional[Any] = None,
    ) -> Union[GroupBy, Resampler]:
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
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

    def _groupby_and_aggregate(
        self, how: Any, *args: Any, **kwargs: Any
    ) -> Union[DataFrame, Series]:
        """
        Re-evaluate the obj with a groupby aggregation.
        """
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
        self, groupby: GroupBy, key: Optional[Any] = None
    ) -> Resampler:
        """
        Return the correct class for resampling with groupby.
        """
        return self._resampler_for_grouping(
            groupby=groupby, key=key, parent=self
        )

    def _wrap_result(self, result: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
        """
        Potentially wrap any results.
        """
        obj = self.obj
        if isinstance(result, ABCDataFrame) and len(result) == 0 and not isinstance(
            result.index, PeriodIndex
        ):
            result = result.set_index(
                _asfreq_compat(obj.index[:0], freq=self.freq), append=True
            )
        if isinstance(result, ABCSeries) and self._selection is not None:
            result.name = self._selection
        if isinstance(result, ABCSeries) and result.empty:
            result.index = _asfreq_compat(obj.index[:0], freq=self.freq)
            result.name = getattr(obj, "name", None)
        if (
            hasattr(self._timegrouper, "_arrow_dtype")
            and self._timegrouper._arrow_dtype is not None
        ):
            result.index = result.index.astype(self._timegrouper._arrow_dtype)
        return result

    @final
    def ffill(self, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        """
        Forward fill the values.

        This method fills missing values by propagating the last valid
        observation forward, up to the next valid observation. It is commonly
        used in time series analysis when resampling data to a higher frequency
        (upsampling) and filling gaps in the resampled output.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series
            The resampled data with missing values filled forward.

        See Also
        --------
        Series.fillna: Fill NA/NaN values using the specified method.
        DataFrame.fillna: Fill NA/NaN values using the specified method.

        Examples
        --------
        Here we only create a ``Series``.

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64

        Example for ``ffill`` with downsampling (we have fewer dates after resampling):

        >>> ser.resample("MS").ffill()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64

        Example for ``ffill`` with upsampling (fill the new dates with
        the previous value):

        >>> ser.resample("W").ffill()
        2023-01-01    1
        2023-01-08    1
        2023-01-15    2
        2023-01-22    2
        2023-01-29    2
        2023-02-05    3
        2023-02-12    3
        2023-02-19    4
        Freq: W-SUN, dtype: int64

        With upsampling and limiting (only fill the first new date with the
        previous value):

        >>> ser.resample("W").ffill(limit=1)
        2023-01-01    1.0
        2023-01-08    1.0
        2023-01-15    2.0
        2023-01-22    2.0
        2023-01-29    NaN
        2023-02-05    3.0
        2023-02-12    NaN
        2023-02-19    4.0
        Freq: W-SUN, dtype: float64
        """
        return self._upsample("ffill", limit=limit)

    @final
    def nearest(self, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        """
        Resample by using the nearest value.

        When resampling data, missing values may appear (e.g., when the
        resampling frequency is higher than the original frequency).
        The `nearest` method will replace ``NaN`` values that appeared in
        the resampled data with the value from the nearest member of the
        sequence, based on the index value.
        Missing values that existed in the original data will not be modified.
        If `limit` is given, fill only this many values in each direction for
        each of the original values.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series or DataFrame
            An upsampled Series or DataFrame with ``NaN`` values filled with
            their nearest value.

        See Also
        --------
        bfill : Backward fill the new missing values in the resampled data.
        ffill : Forward fill ``NaN`` values.

        Examples
        --------
        >>> s = pd.Series([1, 2], index=pd.date_range("20180101", periods=2, freq="1h"))
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        Freq: h, dtype: int64

        >>> s.resample("15min").nearest()
        2018-01-01 00:00:00    1
        2018-01-01 00:15:00    1
        2018-01-01 00:30:00    2
        2018-01-01 00:45:00    2
        2018-01-01 01:00:00    2
        Freq: 15min, dtype: int64

        Limit the number of upsampled values imputed by the nearest:

        >>> s.resample("15min").nearest(limit=1)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    1.0
        2018-01-01 00:30:00    NaN
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        Freq: 15min, dtype: float64
        """
        return self._upsample("nearest", limit=limit)

    @final
    def bfill(self, limit: Optional[int] = None) -> Union[DataFrame, Series]:
        """
        Backward fill the new missing values in the resampled data.

        In statistics, imputation is the process of replacing missing data with
        substituted values [1]_. When resampling data, missing values may
        appear (e.g., when the resampling frequency is higher than the original
        frequency). The backward fill will replace NaN values that appeared in
        the resampled data with the next value in the original sequence.
        Missing values that existed in the original data will not be modified.

        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.

        Returns
        -------
        Series, DataFrame
            An upsampled Series or DataFrame with backward filled NaN values.

        See Also
        --------
        nearest : Fill NaN values with nearest neighbor starting from center.
        ffill : Forward fill NaN values.
        Series.fillna : Fill NaN values in the Series using the
            specified method, which can be 'backfill'.
        DataFrame.fillna : Fill NaN values in the DataFrame using the
            specified method, which can be 'backfill'.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Imputation_%28statistics%29

        Examples
        --------
        Resampling a Series:

        >>> s = pd.Series(
        ...     [1, 2, 3], index=pd.date_range("20180101", periods=3, freq="h")
        ... )
        >>> s
        2018-01-01 00:00:00    1
        2018-01-01 01:00:00    2
        2018-01-01 02:00:00    3
        Freq: h, dtype: int64

        >>> s.resample("30min").bfill()
        2018-01-01 00:00:00    1
        2018-01-01 00:30:00    2
        2018-01-01 01:00:00    2
        2018-01-01 01:30:00    3
        2018-01-01 02:00:00    3
        Freq: 30min, dtype: int64

        >>> s.resample("15min").bfill(limit=2)
        2018-01-01 00:00:00    1.0
        2018-01-01 00:15:00    NaN
        2018-01-01 00:30:00    2.0
        2018-01-01 00:45:00    2.0
        2018-01-01 01:00:00    2.0
        2018-01-01 01:15:00    NaN
        2018-01-01 01:30:00    3.0
        2018-01-01 01:45:00    3.0
        2018-01-01 02:00:00    3.0
        Freq: 15min, dtype: float64

        Resampling a DataFrame that has missing values:

        >>> df = pd.DataFrame(
        ...     {"a": [2, np.nan, 6], "b": [1, 3, 5]},
        ...     index=pd.date_range("20180101", periods=3, freq="h"),
        ... )
        >>> df
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 01:00:00  NaN  3
        2018-01-01 02:00:00  6.0  5

        >>> df.resample("30min").bfill()
                               a  b
        2018-01-01 00:00:00  2.0  1
        2018-01-01 00:30:00  NaN  3
        2018-01-01 01:00:00  NaN  3
        2018-01-01 01:30:00  6.0  5
        2018-01-01 02:00:00  6.0  5

        >>> df.resample("15min").bfill(limit=2)
                               a    b
        2018-01-01 00:00:00  2.0  1.0
        2018-01-01 00:15:00  NaN  NaN
        2018-01-01 00:30:00  NaN  3.0
        2018-01-01 00:45:00  NaN  3.0
        2018-01-01 01:00:00  NaN  3.0
        2018-01-01 01:15:00  NaN  NaN
        2018-01-01 01:30:00  6.0  5.0
        2018-01-01 01:45:00  6.0  5.0
        2018-01-01 02:00:00  6.0  5.0
        Freq: 15min, dtype: float64
        """
        return self._upsample("bfill", limit=limit)

    @final
    def interpolate(
        self,
        method: str = "linear",
        *,
        axis: Literal[0, "index", 1, "columns"] = 0,
        limit: Optional[int] = None,
        inplace: bool = False,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area: Optional[Literal["inside", "outside"]] = None,
        downcast: Union[Literal["infer"], type(None)] = lib.no_default,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]:
        """
        Interpolate values between target timestamps according to different methods.

        The original index is first reindexed to target timestamps
        (see :meth:`core.resample.Resampler.asfreq`),
        then the interpolation of ``NaN`` values via :meth:`DataFrame.interpolate`
        happens.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. One of:

            * 'linear': Ignore the index and treat the values as equally
              spaced. This is the only method supported on MultiIndexes.
            * 'time': Works on daily and higher resolution data to interpolate
              given length of interval.
            * 'index', 'values': use the actual numerical values of the index.
            * 'pad': Fill in NaNs using existing values.
            * 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
              'barycentric', 'polynomial': Passed to
              `scipy.interpolate.interp1d`, whereas 'spline' is passed to
              `scipy.interpolate.UnivariateSpline`. These methods use the numerical
              values of the index.  Both 'polynomial' and 'spline' require that
              you also specify an `order` (int), e.g.
              ``df.interpolate(method='polynomial', order=5)``. Note that,
              `slinear` method in Pandas refers to the Scipy first order `spline`
              instead of Pandas first order `spline`.
            * 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
              'cubicspline': Wrappers around the SciPy interpolation methods of
              similar names. See `Notes`.
            * 'from_derivatives': Refers to
              `scipy.interpolate.BPoly.from_derivatives`.

        axis : {0 or 'index', 1 or 'columns', None}, default None
            Axis to interpolate along. For `Series` this parameter is unused
            and defaults to 0.
        limit : int, optional
            Maximum number of consecutive NaNs to fill. Must be greater than
            0.
        inplace : bool, default False
            Update the data in place if possible.
        limit_direction : {'forward', 'backward', 'both'}, Optional
            Consecutive NaNs will be filled in this direction.

        limit_area : {None, 'inside', 'outside'}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * 'inside': Only fill NaNs surrounded by valid values
              (interpolate).
            * 'outside': Only fill NaNs outside valid values (extrapolate).

        downcast : optional, 'infer' or None, defaults to None
            Downcast dtypes if possible.

            .. deprecated:: 2.1.0

        **kwargs : optional
            Keyword arguments to pass on to the interpolating function.

        Returns
        -------
        DataFrame or Series
            Interpolated values at the specified freq.

        See Also
        --------
        core.resample.Resampler.asfreq: Return the values at the new freq,
            essentially a reindex.
        DataFrame.interpolate: Fill NaN values using an interpolation method.
        DataFrame.bfill : Backward fill NaN values in the resampled data.
        DataFrame.ffill : Forward fill NA/NaN values.

        Notes
        -----
        For high-frequent or non-equidistant time-series with timestamps
        the reindexing followed by interpolation may lead to information loss
        as shown in the last example.

        Examples
        --------

        >>> start = "2023-03-01T07:00:00"
        >>> timesteps = pd.date_range(start, periods=5, freq="s")
        >>> series = pd.Series(data=[1, -1, 2, 1, 3], index=timesteps)
        >>> series
        2023-03-01 07:00:00    1
        2023-03-01 07:00:01   -1
        2023-03-01 07:00:02    2
        2023-03-01 07:00:03    1
        2023-03-01 07:00:04    3
        Freq: s, dtype: int64

        Downsample the dataframe to 0.5Hz by providing the period time of 2s.

        >>> series.resample("2s").interpolate("linear")
        2023-03-01 07:00:00    1
        2023-03-01 07:00:02    2
        2023-03-01 07:00:04    3
        Freq: 2s, dtype: int64

        Upsample the dataframe to 2Hz by providing the period time of 500ms.

        >>> series.resample("500ms").interpolate("linear")
        2023-03-01 07:00:00.000    1.0
        2023-03-01 07:00:00.500    0.0
        2023-03-01 07:00:01.000   -1.0
        2023-03-01 07:00:01.500    0.5
        2023-03-01 07:00:02.000    2.0
        2023-03-01 07:00:02.500    1.5
        2023-03-01 07:00:03.000    1.0
        2023-03-01 07:00:03.500    2.0
        2023-03-01 07:00:04.000    3.0
        Freq: 500ms, dtype: float64

        Internal reindexing with ``asfreq()`` prior to interpolation leads to
        an interpolated timeseries on the basis of the reindexed timestamps
        (anchors). It is assured that all available datapoints from original
        series become anchors, so it also works for resampling-cases that lead
        to non-aligned timestamps, as in the following example:

        >>> series.resample("400ms").interpolate("linear")
        2023-03-01 07:00:00.000    1.0
        2023-03-01 07:00:00.400    0.2
        2023-03-01 07:00:00.800   -0.6
        2023-03-01 07:00:01.200   -0.4
        2023-03-01 07:00:01.600    0.8
        2023-03-01 07:00:02.000    2.0
        2023-03-01 07:00:02.400    1.6
        2023-03-01 07:00:02.800    1.2
        2023-03-01 07:00:03.200    1.4
        2023-03-01 07:00:03.600    2.2
        2023-03-01 07:00:04.000    3.0
        Freq: 400ms, dtype: float64

        Note that the series correctly decreases between two anchors
        ``07:00:00`` and ``07:00:02``.
        """
        assert downcast is lib.no_default
        result = self._upsample("asfreq")
        obj = self._selected_obj
        is_period_index = isinstance(obj.index, PeriodIndex)
        if not is_period_index:
            final_index = result.index
            if isinstance(final_index, MultiIndex):
                raise NotImplementedError(
                    "Direct interpolation of MultiIndex data frames is not supported. "
                    "If you tried to resample and interpolate on a grouped data frame, please use:\n"
                    "`df.groupby(...).apply(lambda x: x.resample(...).interpolate(...))`\n"
                    "instead, as resampling and interpolation has to be performed for each group independently."
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
    def asfreq(
        self,
        fill_value: Optional[Any] = None,
    ) -> Union[DataFrame, Series]:
        """
        Return the values at the new freq, essentially a reindex.

        Parameters
        ----------
        fill_value : scalar, optional
            Value to use for missing values, applied during upsampling (note
            this does not fill NaNs that already were present).

        Returns
        -------
        DataFrame or Series
            Values at the specified freq.

        See Also
        --------
        Series.asfreq: Convert TimeSeries to specified frequency.
        DataFrame.asfreq: Convert TimeSeries to specified frequency.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-31", "2023-02-01", "2023-02-28"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-31    2
        2023-02-01    3
        2023-02-28    4
        dtype: int64
        >>> ser.resample("MS").asfreq()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._upsample("asfreq", fill_value=fill_value)

    @final
    def sum(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> Union[DataFrame, Series]:
        """
        Compute sum of group values.

        This method provides a simple way to compute the sum of values within each
        resampled group, particularly useful for aggregating time-based data into
        daily, monthly, or yearly sums.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed sum of values within each group.

        See Also
        --------
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.count : Compute count of group, excluding missing
            values.
        DataFrame.resample : Resample time-series data.
        Series.sum : Return the sum of the values over the requested axis.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").sum()
        2023-01-01    3
        2023-02-01    7
        Freq: MS, dtype: int64
        """
        return self._downsample("sum", numeric_only=numeric_only, min_count=min_count)

    @final
    def prod(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> Union[DataFrame, Series]:
        """
        Compute prod of group values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computed prod of values within each group.

        See Also
        --------
        core.resample.Resampler.sum : Compute sum of groups, excluding missing values.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").prod()
        2023-01-01    2
        2023-02-01   12
        Freq: MS, dtype: int64
        """
        return self._downsample("prod", numeric_only=numeric_only, min_count=min_count)

    @final
    def min(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> Union[DataFrame, Series]:
        """
        Compute min value of group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Compute the minimum value in the given Series or DataFrame.

        See Also
        --------
        core.resample.Resampler.max : Compute max value of group.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").min()
        2023-01-01    1
        2023-02-01    3
        Freq: MS, dtype: int64
        """
        return self._downsample("min", numeric_only=numeric_only, min_count=min_count)

    @final
    def max(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
    ) -> Union[DataFrame, Series]:
        """
        Compute max value of group.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None``.

        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.

        Returns
        -------
        Series or DataFrame
            Computes the maximum value in the given Series or Dataframe.

        See Also
        --------
        core.resample.Resampler.min : Compute min value of group.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").max()
        2023-01-01    2
        2023-02-01    4
        Freq: MS, dtype: int64
        """
        return self._downsample("max", numeric_only=numeric_only, min_count=min_count)

    @final
    @doc(GroupBy.first)
    def first(
        self,
        numeric_only: bool = False,
        min_count: int = 0,
        skipna: bool = True,
    ) -> Union[DataFrame, Series]:
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
    ) -> Union[DataFrame, Series]:
        return self._downsample(
            "last", numeric_only=numeric_only, min_count=min_count, skipna=skipna
        )

    @final
    def median(
        self,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]:
        """
        Compute median of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex

        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.

            .. versionchanged:: 2.0.0

                numeric_only no longer accepts ``None`` and defaults to False.

        Returns
        -------
        Series or DataFrame
            Median of values within each group.

        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 2, 3, 3, 4, 5],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").median()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64
        """
        return self._downsample("median", numeric_only=numeric_only)

    @final
    def mean(
        self,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]:
        """
        Compute mean of groups, excluding missing values.

        Parameters
        ----------
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Mean of values within each group.

        See Also
        --------
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.
        core.resample.Resampler.sum : Compute sum of groups, excluding missing values.
        core.resample.Resampler.std : Compute standard deviation of groups, excluding
            missing values.
        core.resample.Resampler.var : Compute variance of groups, excluding missing
            values.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample("MS").mean()
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
        return self._downsample("mean", numeric_only=numeric_only)

    @final
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]:
        """
        Compute standard deviation of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Standard deviation of values within each group.

        See Also
        --------
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.
        core.resample.Resampler.var : Compute variance of groups, excluding missing
            values.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").std()
        2023-01-01    1.000000
        2023-02-01    2.645751
        Freq: MS, dtype: float64
        """
        return self._downsample(
            "std", ddof=ddof, numeric_only=numeric_only
        )

    @final
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]:
        """
        Compute variance of groups, excluding missing values.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        DataFrame or Series
            Variance of values within each group.

        See Also
        --------
        core.resample.Resampler.std : Compute standard deviation of groups, excluding
            missing values.
        core.resample.Resampler.mean : Compute mean of groups, excluding missing values.
        core.resample.Resampler.median : Compute median of groups, excluding missing
            values.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").var()
        2023-01-01    1.0
        2023-02-01    7.0
        Freq: MS, dtype: float64

        >>> ser.resample("MS").var(ddof=0)
        2023-01-01    0.666667
        2023-02-01    4.666667
        Freq: MS, dtype: float64
        """
        return self._downsample(
            "var", ddof=ddof, numeric_only=numeric_only
        )

    @final
    def sem(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
    ) -> Union[DataFrame, Series]:
        """
        Compute standard error of the mean of groups, excluding missing values.

        For multiple groupings, the result index will be a MultiIndex.

        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0

                numeric_only now defaults to ``False``.

        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.

        See Also
        --------
        DataFrame.sem : Return unbiased standard error of the mean over requested axis.
        Series.sem : Return unbiased standard error of the mean over requested axis.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").sem()
        2023-01-01    0.577350
        2023-02-01    1.527525
        Freq: MS, dtype: float64
        """
        return self._downsample(
            "sem", ddof=ddof, numeric_only=numeric_only
        )

    @final
    @doc(GroupBy.ohlc)
    def ohlc(self) -> Union[DataFrame, Series]:
        ax = self.ax
        obj = self._obj_with_exclusions
        if len(ax) == 0:
            obj = obj.copy()
            obj.index = _asfreq_compat(obj.index, freq=self.freq)
            if obj.ndim == 1:
                obj = obj.to_frame()
                obj = obj.reindex(["open", "high", "low", "close"], axis=1)
            else:
                mi = MultiIndex.from_product([obj.columns, ["open", "high", "low", "close"]])
                obj = obj.reindex(mi, axis=1)
            return obj
        return self._downsample("ohlc")

    @final
    def nunique(self) -> Union[DataFrame, Series]:
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.

        See Also
        --------
        core.groupby.SeriesGroupBy.nunique : Method nunique for SeriesGroupBy.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 3],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    3
        dtype: int64
        >>> ser.resample("MS").nunique()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        return self._downsample("nunique")

    @final
    @doc(GroupBy.size)
    def size(self) -> Union[DataFrame, Series]:
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
    def count(self) -> Union[DataFrame, Series]:
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
    def quantile(
        self,
        q: Union[float, Sequence[float]] = 0.5,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]:
        """
        Return value at the given quantile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)

        Returns
        -------
        DataFrame or Series
            Quantile of values within each group.

        See Also
        --------
        Series.quantile
            Return a series, where the index is q and the values are the quantiles.
        DataFrame.quantile
            Return a DataFrame, where the columns are the columns of self,
            and the values are the quantiles.
        DataFrameGroupBy.quantile
            Return a DataFrame, where the columns are groupby columns,
            and the values are its quantiles.

        Examples
        --------

        >>> ser = pd.Series(
        ...     [1, 3, 2, 4, 3, 8],
        ...     index=pd.DatetimeIndex(
        ...         [
        ...             "2023-01-01",
        ...             "2023-01-10",
        ...             "2023-01-15",
        ...             "2023-02-01",
        ...             "2023-02-10",
        ...             "2023-02-15",
        ...         ]
        ...     ),
        ... )
        >>> ser.resample("MS").quantile()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64

        >>> ser.resample("MS").quantile(0.25)
        2023-01-01    1.5
        2023-02-01    3.5
        Freq: MS, dtype: float64
        """
        return self._downsample("quantile", q=q, **kwargs)


class _GroupByMixin(PandasObject, SelectionMixin):
    """
    Provide the groupby facilities.
    """

    _selection: Optional[Hashable] = None

    def __init__(
        self,
        *,
        parent: Resampler,
        groupby: GroupBy,
        key: Optional[Hashable] = None,
        selection: Optional[Hashable] = None,
    ) -> None:
        assert isinstance(groupby, GroupBy), type(groupby)
        assert isinstance(parent, Resampler), type(parent)
        for attr in parent._attributes:
            setattr(self, attr, getattr(parent, attr))
        self._selection = selection
        self.binner = parent.binner
        self.key = key
        self._groupby = groupby
        self._timegrouper = copy.copy(parent._timegrouper)
        self.ax = parent.ax
        self.obj = parent.obj

    @no_type_check
    def _apply(
        self,
        f: Union[str, Callable[..., Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]:
        """
        Dispatch to _upsample; we are stripping all of the _upsample kwargs and
        performing the original function call on the grouped object.
        """

        def func(x: NDFrameT) -> Any:
            x = self._resampler_cls(
                x, timegrouper=self._timegrouper, gpr_index=self.ax
            )
            if isinstance(f, str):
                return getattr(x, f)(**kwargs)
            return x.apply(f, *args, **kwargs)

        result = self._groupby.apply(func)
        return self._wrap_result(result)

    _upsample: Callable[..., Union[DataFrame, Series]] = _apply
    _downsample: Callable[..., Union[DataFrame, Series]] = _apply
    _groupby_and_aggregate: Callable[..., Union[DataFrame, Series]] = _apply

    @final
    def _gotitem(
        self,
        key: Any,
        ndim: int,
        subset: Optional[Any] = None,
    ) -> Resampler:
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
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
            groupby=groupby, parent=cast(Resampler, self), selection=selection
        )
        return new_rs


class DatetimeIndexResampler(Resampler):
    @property
    def _resampler_for_grouping(self) -> Type[DatetimeIndexResamplerGroupby]:
        return DatetimeIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[Index, np.ndarray, Index]:
        if isinstance(self.ax, PeriodIndex):
            return self._timegrouper._get_time_period_bins(self.ax)
        return self._timegrouper._get_time_bins(self.ax)

    def _downsample(
        self,
        how: Any,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]:
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
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

    def _adjust_binner_for_upsample(self, binner: Index) -> Index:
        """
        Adjust our binner when upsampling.

        The range of a new index should not be outside specified range
        """
        if self.closed == "right":
            binner = cast(Index, binner[1:])
        else:
            binner = cast(Index, binner[:-1])
        return binner

    def _upsample(
        self,
        method: str,
        limit: Optional[int] = None,
        fill_value: Optional[Any] = None,
    ) -> Union[DataFrame, Series]:
        """
        Parameters
        ----------
        method : string {'backfill', 'bfill', 'pad',
            'ffill', 'asfreq'} method for upsampling
        limit : int, default None
            Maximum size gap to fill when reindexing
        fill_value : scalar, default None
            Value to use for missing values
        """
        if self._from_selection:
            raise ValueError(
                "Upsampling from level= or on= selection is not supported, use .set_index(...) to explicitly set index to datetime-like"
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

    def _wrap_result(
        self, result: Union[DataFrame, Series]
    ) -> Union[DataFrame, Series]:
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
    """
    Provides a resample of a groupby implementation
    """

    @property
    def _resampler_cls(self) -> Type[DatetimeIndexResampler]:
        return DatetimeIndexResampler


class PeriodIndexResampler(DatetimeIndexResampler):
    @property
    def _resampler_for_grouping(self) -> Type[PeriodIndexResamplerGroupby]:
        warnings.warn(
            "Resampling a groupby with a PeriodIndex is deprecated. Cast to DatetimeIndex before resampling instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return PeriodIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[Index, np.ndarray, Index]:
        if isinstance(self.ax, DatetimeIndex):
            return super()._get_binner_for_time()
        return self._timegrouper._get_period_bins(self.ax)

    def _convert_obj(self, obj: NDFrameT) -> NDFrameT:
        obj = super()._convert_obj(obj)
        if self._from_selection:
            msg = (
                "Resampling from level= or on= selection with a PeriodIndex is not currently supported, "
                "use .set_index(...) to explicitly set index"
            )
            raise NotImplementedError(msg)
        if isinstance(obj, DatetimeIndex):
            obj = obj.to_timestamp(how=self.convention)
        return obj

    def _downsample(
        self,
        how: Any,
        **kwargs: Any,
    ) -> Union[DataFrame, Series]:
        """
        Downsample the cython defined function.

        Parameters
        ----------
        how : string / cython mapped function
        **kwargs : kw args passed to how function
        """
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
            f"Frequency {ax.freq} cannot be resampled to {self.freq}, as they are not sub or super periods"
        )

    def _upsample(
        self,
        method: str,
        limit: Optional[int] = None,
        fill_value: Optional[Any] = None,
    ) -> Union[DataFrame, Series]:
        """
        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method for upsampling.
        limit : int, default None
            Maximum size gap to fill when reindexing.
        fill_value : scalar, default None
            Value to use for missing values.
        """
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
    """
    Provides a resample of a groupby implementation.
    """

    @property
    def _resampler_cls(self) -> Type[PeriodIndexResampler]:
        return PeriodIndexResampler


class TimedeltaIndexResampler(DatetimeIndexResampler):
    @property
    def _resampler_for_grouping(self) -> Type[TimedeltaIndexResamplerGroupby]:
        return TimedeltaIndexResamplerGroupby

    def _get_binner_for_time(self) -> Tuple[Index, np.ndarray, Index]:
        return self._timegrouper._get_time_delta_bins(self.ax)

    def _adjust_binner_for_upsample(self, binner: Index) -> Index:
        """
        Adjust our binner when upsampling.

        The range of a new index is allowed to be greater than original range
        so we don't need to change the length of a binner, GH 13022
        """
        return binner


class TimedeltaIndexResamplerGroupby(_GroupByMixin, TimedeltaIndexResampler):
    """
    Provides a resample of a groupby implementation.
    """

    @property
    def _resampler_cls(self) -> Type[TimedeltaIndexResampler]:
        return TimedeltaIndexResampler


def get_resampler(obj: NDFrameT, **kwds: Any) -> Resampler:
    """
    Create a TimeGrouper and return our resampler.
    """
    tg = TimeGrouper(obj=obj, **kwds)
    return tg._get_resampler(obj)


get_resampler.__doc__ = Resampler.__doc__


def get_resampler_for_grouping(
    groupby: GroupBy,
    rule: BaseOffset,
    how: Optional[Any] = None,
    fill_method: Optional[str] = None,
    limit: Optional[int] = None,
    on: Optional[Any] = None,
    **kwargs: Any,
) -> Resampler:
    """
    Return our appropriate resampler when grouping as well.
    """
    tg = TimeGrouper(freq=rule, key=on, **kwargs)
    resampler = tg._get_resampler(groupby.obj)
    return resampler._get_resampler_for_grouping(groupby=groupby, key=tg.key)


class TimeGrouper(Grouper):
    """
    Custom groupby class for time-interval grouping.

    Parameters
    ----------
    freq : pandas date offset or offset alias for identifying bin edges
    closed : closed end of interval; 'left' or 'right'
    label : interval boundary to use for labeling; 'left' or 'right'
    convention : {'start', 'end', 'e', 's'}
        If axis is PeriodIndex
    """

    _attributes: Tuple[str, ...] = Grouper._attributes + (
        "closed",
        "label",
        "how",
        "convention",
        "origin",
        "offset",
    )

    def __init__(
        self,
        obj: Optional[NDFrameT] = None,
        freq: Union[str, BaseOffset] = "Min",
        key: Optional[Hashable] = None,
        closed: Optional[Literal["left", "right"]] = None,
        label: Optional[Literal["left", "right"]] = None,
        how: str = "mean",
        fill_method: Optional[str] = None,
        limit: Optional[int] = None,
        convention: Optional[Literal["start", "end", "e", "s"]] = None,
        origin: Union[
            Literal["epoch", "start", "start_day", "end", "end_day"], Timestamp
        ] = "start_day",
        offset: Optional[Union[str, Timedelta]] = None,
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
            key is None
            and obj is not None
            and isinstance(obj.index, PeriodIndex)
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
        if (
            rule in end_types
            or ("-" in rule and rule[: rule.find("-")] in end_types)
        ):
            if closed is None:
                closed = "right"
            if label is None:
                label = "right"
        elif origin in ["end", "end_day"]:
            if closed is None:
                closed = "right"
            if label is None:
                label = "right"
        else:
            if closed is None:
                closed = "left"
            if label is None:
                label = "left"

        self.closed: Literal["left", "right"]
        self.label: Literal["left", "right"]
        self.closed = closed  # type: ignore
        self.label = label  # type: ignore
        self.convention: str = convention if convention is not None else "e"
        self.how: str = how
        self.fill_method: Optional[str] = fill_method
        self.limit: Optional[int] = limit
        self.group_keys: bool = group_keys
        self._arrow_dtype: Optional[ArrowDtype] = None
        if origin in ("epoch", "start", "start_day", "end", "end_day"):
            self.origin = origin
        else:
            try:
                self.origin = Timestamp(origin)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"'origin' should be equal to 'epoch', 'start', 'start_day', 'end', 'end_day' or should be a Timestamp convertible type. Got '{origin}' instead."
                ) from err
        try:
            self.offset = Timedelta(offset) if offset is not None else None
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"'offset' should be a Timedelta convertible type. Got '{offset}' instead."
            ) from err
        kwargs["sort"] = True
        super().__init__(freq=freq, key=key, **kwargs)

    def _get_resampler(
        self, obj: NDFrameT
    ) -> Resampler:
        """
        Return my resampler or raise if we have an invalid axis.

        Parameters
        ----------
        obj : Series or DataFrame

        Returns
        -------
        Resampler

        Raises
        ------
        TypeError if incompatible axis
        """
        _, ax, _ = self._set_grouper(obj, sort=False, gpr_index=None)
        if isinstance(ax, DatetimeIndex):
            return DatetimeIndexResampler(
                obj, timegrouper=self, group_keys=self.group_keys, gpr_index=ax
            )
        elif isinstance(ax, PeriodIndex):
            if isinstance(ax, PeriodIndex):
                warnings.warn(
                    "Resampling with a PeriodIndex is deprecated. Cast index to DatetimeIndex before resampling instead.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            return PeriodIndexResampler(
                obj, timegrouper=self, group_keys=self.group_keys, gpr_index=ax
            )
        elif isinstance(ax, TimedeltaIndex):
            return TimedeltaIndexResampler(
                obj, timegrouper=self, group_keys=self.group_keys, gpr_index=ax
            )
        raise TypeError(
            f"Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of '{type(ax).__name__}'"
        )

    def _get_grouper(
        self, obj: NDFrameT, validate: bool = True
    ) -> Tuple[Any, NDFrameT]:
        r = self._get_resampler(obj)
        return (r._grouper, cast(NDFrameT, r.obj))

    def _get_time_bins(
        self, ax: DatetimeIndex
    ) -> Tuple[Index, np.ndarray, Index]:
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                f"axis must be a DatetimeIndex, but got an instance of {type(ax).__name__}"
            )
        if len(ax) == 0:
            binner = labels = DatetimeIndex(
                data=[], freq=self.freq, name=ax.name, dtype=ax.dtype
            )
            return (binner, np.array([]), labels)
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
            binner = cast(Index, binner.insert(0, NaT))
            labels = cast(Index, labels.insert(0, NaT))
        if len(bins) < len(labels):
            labels = labels[: len(bins)]
        return (binner, bins, labels)

    def _adjust_bin_edges(
        self, binner: Index, ax_values: np.ndarray
    ) -> Tuple[Index, np.ndarray]:
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
        return (binner, bin_edges)

    def _get_time_delta_bins(
        self, ax: TimedeltaIndex
    ) -> Tuple[Index, np.ndarray, Index]:
        if not isinstance(ax, TimedeltaIndex):
            raise TypeError(
                f"axis must be a TimedeltaIndex, but got an instance of {type(ax).__name__}"
            )
        if not isinstance(self.freq, Tick):
            raise ValueError(
                f"Resampling on a TimedeltaIndex requires fixed-duration `freq`, "
                f"e.g. '24h' or '3D', not {self.freq}"
            )
        if not len(ax):
            binner = labels = TimedeltaIndex(
                data=[], freq=self.freq, name=ax.name
            )
            return (binner, np.array([]), labels)
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
        return (binner, bins, labels)

    def _get_time_period_bins(
        self, ax: DatetimeIndex
    ) -> Tuple[Index, np.ndarray, Index]:
        if not isinstance(ax, DatetimeIndex):
            raise TypeError(
                f"axis must be a DatetimeIndex, but got an instance of {type(ax).__name__}"
            )
        freq = self.freq
        if len(ax) == 0:
            binner = labels = PeriodIndex(
                data=[], freq=freq, name=ax.name, dtype=ax.dtype
            )
            return (binner, np.array([]), labels)
        labels = binner = period_range(
            start=ax[0], end=ax[-1], freq=freq, name=ax.name
        )
        end_stamps = (labels + freq).asfreq(freq, "s").to_timestamp()
        if ax.tz:
            end_stamps = end_stamps.tz_localize(ax.tz)
        bins = ax.searchsorted(end_stamps, side="left")
        return (binner, bins, labels)

    def _get_period_bins(
        self, ax: PeriodIndex
    ) -> Tuple[Index, np.ndarray, Index]:
        if not isinstance(ax, PeriodIndex):
            raise TypeError(
                f"axis must be a PeriodIndex, but got an instance of {type(ax).__name__}"
            )
        memb = ax.asfreq(self.freq, how=self.convention)
        nat_count = 0
        if memb.hasnans:
            nat_count = np.sum(memb._isnan)
            memb = memb[~memb._isnan]
        if not len(memb):
            bins = np.array([], dtype=np.int64)
            binner = labels = PeriodIndex(
                data=[], freq=self.freq, name=ax.name
            )
            if len(ax) > 0:
                binner, bins, labels = _insert_nat_bin(
                    binner, bins, labels, len(ax)
                )
            return (binner, bins, labels)
        freq_mult = self.freq.n
        start = ax.min().asfreq(self.freq, how=self.convention)
        end = ax.max().asfreq(self.freq, how="end")
        bin_shift = 0
        if isinstance(self.freq, Tick):
            p_start, end = _get_period_range_edges(
                start, end, self.freq, closed=self.closed, origin=self.origin, offset=self.offset
            )
            start_offset = Period(start, self.freq) - Period(p_start, self.freq)
            bin_shift = start_offset.n % freq_mult
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
            binner, bins, labels = _insert_nat_bin(
                binner, bins, labels, nat_count
            )
        return (binner, bins, labels)

    def _set_grouper(
        self, obj: NDFrameT, sort: bool = False, *, gpr_index: Optional[Index] = None
    ) -> Tuple[NDFrameT, Index, np.ndarray]:
        obj, ax, indexer = super()._set_grouper(obj, sort, gpr_index=gpr_index)
        if isinstance(ax.dtype, ArrowDtype) and ax.dtype.kind in "Mm":
            self._arrow_dtype = cast(ArrowDtype, ax.dtype)
            ax = cast(Index, Index(cast(ArrowExtensionArray, ax.array)._maybe_convert_datelike_array()))
        return (obj, ax, indexer)


@overload
def _take_new_index(
    obj: Series,
    indexer: np.ndarray,
    new_index: Index,
) -> Series:
    ...


@overload
def _take_new_index(
    obj: DataFrame,
    indexer: np.ndarray,
    new_index: Index,
) -> DataFrame:
    ...


def _take_new_index(
    obj: Union[Series, DataFrame],
    indexer: np.ndarray,
    new_index: Index,
) -> Union[Series, DataFrame]:
    if isinstance(obj, ABCSeries):
        new_values = algos.take_nd(obj._values, indexer)
        return obj._constructor(new_values, index=new_index, name=obj.name)
    elif isinstance(obj, ABCDataFrame):
        new_mgr = obj._mgr.reindex_indexer(
            new_axis=new_index, indexer=indexer, axis=1
        )
        return obj._constructor_from_mgr(new_mgr, axes=new_mgr.axes)
    else:
        raise ValueError("'obj' should be either a Series or a DataFrame")


def _get_timestamp_range_edges(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    unit: str,
    closed: Literal["left", "right"] = "left",
    origin: Union[
        Literal["epoch", "start", "start_day", "end", "end_day"], Timestamp
    ] = "start_day",
    offset: Optional[Timedelta] = None,
) -> Tuple[Timestamp, Timestamp]:
    """
    Adjust the `first` Timestamp to the preceding Timestamp that resides on
    the provided offset. Adjust the `last` Timestamp to the following
    Timestamp that resides on the provided offset. Input Timestamps that
    already reside on the offset will be adjusted depending on the type of
    offset and the `closed` parameter.

    Parameters
    ----------
    first : pd.Timestamp
        The beginning Timestamp of the range to be adjusted.
    last : pd.Timestamp
        The ending Timestamp of the range to be adjusted.
    freq : pd.DateOffset
        The dateoffset to which the Timestamps will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'} or Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Timestamp objects.
    """
    if isinstance(freq, Tick):
        index_tz = first.tz
        if isinstance(origin, Timestamp) and (origin.tz is None) != (
            index_tz is None
        ):
            raise ValueError("The origin must have the same timezone as the index.")
        if origin == "epoch":
            origin = Timestamp("1970-01-01", tz=index_tz)
        if isinstance(freq, Day):
            first = first.tz_localize(None)
            last = last.tz_localize(None)
            if isinstance(origin, Timestamp):
                origin = origin.tz_localize(None)
        first, last = _adjust_dates_anchored(
            first,
            last,
            freq,
            closed=closed,
            origin=origin,
            offset=offset,
            unit=unit,
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
    return (first, last)


def _get_period_range_edges(
    first: Period,
    last: Period,
    freq: BaseOffset,
    closed: Literal["left", "right"] = "left",
    origin: Union[
        Literal["epoch", "start", "start_day"], Timestamp
    ] = "start_day",
    offset: Optional[Timedelta] = None,
) -> Tuple[Period, Period]:
    """
    Adjust the provided `first` and `last` Periods to the respective Period of
    the given offset that encompasses them.

    Parameters
    ----------
    first : pd.Period
        The beginning Period of the range to be adjusted.
    last : pd.Period
        The ending Period of the range to be adjusted.
    freq : pd.DateOffset
        The freq to which the Periods will be adjusted.
    closed : {'right', 'left'}, default "left"
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'}, Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin must
        match the timezone of the index.
        If a timestamp is not used, these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Period objects.
    """
    if not all((isinstance(obj, Period) for obj in [first, last])):
        raise TypeError("'first' and 'last' must be instances of type Period")
    first_ts = first.to_timestamp()
    last_ts = last.to_timestamp()
    adjust_first = not freq.is_on_offset(first_ts)
    adjust_last = freq.is_on_offset(last_ts)
    first_ts, last_ts = _get_timestamp_range_edges(
        first_ts,
        last_ts,
        freq,
        unit="ns",
        closed=closed,
        origin=origin,
        offset=offset,
    )
    first = (first_ts + int(adjust_first) * freq).to_period(freq)
    last = (last_ts - int(adjust_last) * freq).to_period(freq)
    return (first, last)


def _insert_nat_bin(
    binner: Index, bins: np.ndarray, labels: Index, nat_count: int
) -> Tuple[Index, np.ndarray, Index]:
    assert nat_count > 0
    bins = np.insert(bins, 0, nat_count)
    binner = cast(Index, binner.insert(0, NaT))
    labels = cast(Index, labels.insert(0, NaT))
    return (binner, bins, labels)


def _adjust_dates_anchored(
    first: Timestamp,
    last: Timestamp,
    freq: BaseOffset,
    closed: Literal["left", "right"] = "right",
    origin: Union[
        Literal["epoch", "start", "start_day", "end", "end_day"], Timestamp
    ] = "start_day",
    offset: Optional[Timedelta] = None,
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
    return (fresult, lresult)


def asfreq(
    obj: NDFrameT,
    freq: Union[str, BaseOffset],
    method: Optional[str] = None,
    how: Optional[Literal["E", "S"]] = None,
    normalize: bool = False,
    fill_value: Optional[Any] = None,
) -> Union[DataFrame, Series]:
    """
    Utility frequency conversion method for Series/DataFrame.

    See :meth:`pandas.NDFrame.asfreq` for full documentation.
    """
    if isinstance(obj.index, PeriodIndex):
        if method is not None:
            raise NotImplementedError("'method' argument is not supported")
        if how is None:
            how = "E"
        if isinstance(freq, BaseOffset):
            if hasattr(freq, "_period_dtype_code"):
                freq = PeriodDtype(freq)._freqstr
        new_obj = obj.copy()
        new_obj.index = obj.index.asfreq(freq=freq, how=how)
    elif len(obj.index) == 0:
        new_obj = obj.copy()
        new_obj.index = _asfreq_compat(obj.index, freq)
    else:
        unit: Optional[str] = None
        if isinstance(obj.index, DatetimeIndex):
            unit = obj.index.unit
        dti = date_range(
            obj.index.min(),
            obj.index.max(),
            freq=freq,
            unit=unit,
        )
        dti.name = obj.index.name
        new_obj = obj.reindex(dti, method=method, fill_value=fill_value)
        if normalize:
            new_obj.index = new_obj.index.normalize()
    return new_obj


def _asfreq_compat(index: Index, freq: Union[str, BaseOffset]) -> Index:
    """
    Helper to mimic asfreq on (empty) DatetimeIndex and TimedeltaIndex.

    Parameters
    ----------
    index : PeriodIndex, DatetimeIndex, or TimedeltaIndex
    freq : DateOffset

    Returns
    -------
    same type as index
    """
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
