import datetime
from functools import partial
from typing import Any, Optional, Union, Callable
import pandas as pd
from pandas.api.types import is_hashable
from pyspark._globals import _NoValue
from databricks import koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeDatetimeIndex
from databricks.koalas.series import Series, first_series
from databricks.koalas.utils import verify_temp_column_name


class DatetimeIndex(Index):
    """
    Immutable ndarray-like of datetime64 data.
    """

    def __new__(cls, 
                data: Optional[Any] = None, 
                freq: Any = _NoValue, 
                normalize: bool = False, 
                closed: Optional[str] = None,
                ambiguous: Union[str, bool, Any] = 'raise', 
                dayfirst: bool = False, 
                yearfirst: bool = False, 
                dtype: Optional[Union[str, Any]] = None,
                copy: bool = False, 
                name: Any = None) -> "DatetimeIndex":
        if not is_hashable(name):
            raise TypeError('Index.name must be a hashable type')
        if isinstance(data, (Series, Index)):
            if dtype is None:
                dtype = 'datetime64[ns]'
            return Index(data, dtype=dtype, copy=copy, name=name)
        kwargs: dict[str, Any] = dict(data=data, normalize=normalize, closed=closed,
                                      ambiguous=ambiguous, dayfirst=dayfirst, yearfirst=yearfirst,
                                      dtype=dtype, copy=copy, name=name)
        if freq is not _NoValue:
            kwargs['freq'] = freq
        return ks.from_pandas(pd.DatetimeIndex(**kwargs))

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeDatetimeIndex, item):
            property_or_func: Any = getattr(MissingPandasLikeDatetimeIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError("'DatetimeIndex' object has no attribute '{}'"
                             .format(item))

    @property
    def func_dxhjmnow(self) -> Index:
        """
        The year of the datetime.
        """
        return Index(self.to_series().dt.year)

    @property
    def func_u73ctdkl(self) -> Index:
        """
        The month of the timestamp as January = 1 December = 12.
        """
        return Index(self.to_series().dt.month)

    @property
    def func_asfsx3bm(self) -> Index:
        """
        The days of the datetime.
        """
        return Index(self.to_series().dt.day)

    @property
    def func_y935kqjb(self) -> Index:
        """
        The hours of the datetime.
        """
        return Index(self.to_series().dt.hour)

    @property
    def func_tpe7f6k2(self) -> Index:
        """
        The minutes of the datetime.
        """
        return Index(self.to_series().dt.minute)

    @property
    def func_m2gfo0x9(self) -> Index:
        """
        The seconds of the datetime.
        """
        return Index(self.to_series().dt.second)

    @property
    def func_s7weu1cf(self) -> Index:
        """
        The microseconds of the datetime.
        """
        return Index(self.to_series().dt.microsecond)

    @property
    def func_pdkun9zu(self) -> Index:
        """
        The week ordinal of the year.
        """
        return Index(self.to_series().dt.week)

    @property
    def func_ee3sblaz(self) -> Index:
        return Index(self.to_series().dt.weekofyear)
    weekofyear.__doc__ = week.__doc__  # type: ignore

    @property
    def func_u2flg5fa(self) -> Index:
        """
        The day of the week with Monday=0, Sunday=6.
        Return the day of the week. It is assumed the week starts on
        Monday, which is denoted by 0 and ends on Sunday which is denoted
        by 6. This method is available on both Series with datetime
        values (using the `dt` accessor) or DatetimeIndex.

        Returns
        -------
        Series or Index
            Containing integers indicating the day number.

        See Also
        --------
        Series.dt.dayofweek : Alias.
        Series.dt.weekday : Alias.
        Series.dt.day_name : Returns the name of the day of the week.

        Examples
        --------
        >>> idx = ks.date_range('2016-12-31', '2017-01-08', freq='D')
        >>> idx.dayofweek
        Int64Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int64')
        """
        return Index(self.to_series().dt.dayofweek)

    @property
    def func_s0kclwpo(self) -> Index:
        return self.dayofweek
    day_of_week.__doc__ = dayofweek.__doc__  # type: ignore

    @property
    def func_afnb00k5(self) -> Index:
        return Index(self.to_series().dt.weekday)
    weekday.__doc__ = dayofweek.__doc__  # type: ignore

    @property
    def func_o7movi1e(self) -> Index:
        """
        The ordinal day of the year.
        """
        return Index(self.to_series().dt.dayofyear)

    @property
    def func_z4uov256(self) -> Index:
        return self.dayofyear
    day_of_year.__doc__ = dayofyear.__doc__  # type: ignore

    @property
    def func_j8v9bunc(self) -> Index:
        """
        The quarter of the date.
        """
        return Index(self.to_series().dt.quarter)

    @property
    def func_67k6j1xl(self) -> Index:
        """
        Indicates whether the date is the first day of the month.

        Returns
        -------
        Index
            Returns an Index with boolean values

        See Also
        --------
        is_month_end : Return a boolean indicating whether the date
            is the last day of the month.

        Examples
        --------
        >>> idx = ks.date_range("2018-02-27", periods=3)
        >>> idx.is_month_start
        Index([False, False, True], dtype='object')
        """
        return Index(self.to_series().dt.is_month_start)

    @property
    def func_i5uskw1u(self) -> Index:
        """
        Indicates whether the date is the last day of the month.

        Returns
        -------
        Index
            Returns an Index with boolean values.

        See Also
        --------
        is_month_start : Return a boolean indicating whether the date
            is the first day of the month.

        Examples
        --------
        >>> idx = ks.date_range("2018-02-27", periods=3)
        >>> idx.is_month_end
        Index([False, True, False], dtype='object')
        """
        return Index(self.to_series().dt.is_month_end)

    @property
    def func_y8sy4jzk(self) -> Index:
        """
        Indicator for whether the date is the first day of a quarter.

        Returns
        -------
        is_quarter_start : Index
            Returns an Index with boolean values.

        See Also
        --------
        quarter : Return the quarter of the date.
        is_quarter_end : Similar property for indicating the quarter start.

        Examples
        --------
        >>> idx = ks.date_range('2017-03-30', periods=4)
        >>> idx.is_quarter_start
        Index([False, False, True, False], dtype='object')
        """
        return Index(self.to_series().dt.is_quarter_start)

    @property
    def func_znggbzdi(self) -> Index:
        """
        Indicator for whether the date is the last day of a quarter.

        Returns
        -------
        is_quarter_end : Index
            Returns an Index with boolean values.

        See Also
        --------
        quarter : Return the quarter of the date.
        is_quarter_start : Similar property indicating the quarter start.

        Examples
        --------
        >>> idx = ks.date_range('2017-03-30', periods=4)
        >>> idx.is_quarter_end
        Index([False, True, False, False], dtype='object')
        """
        return Index(self.to_series().dt.is_quarter_end)

    @property
    def func_7byofjx9(self) -> Index:
        """
        Indicate whether the date is the first day of a year.

        Returns
        -------
        Index
            Returns an Index with boolean values.

        See Also
        --------
        is_year_end : Similar property indicating the last day of the year.

        Examples
        --------
        >>> idx = ks.date_range("2017-12-30", periods=3)
        >>> idx.is_year_start
        Index([False, False, True], dtype='object')
        """
        return Index(self.to_series().dt.is_year_start)

    @property
    def func_dwt65kdl(self) -> Index:
        """
        Indicate whether the date is the last day of the year.

        Returns
        -------
        Index
            Returns an Index with boolean values.

        See Also
        --------
        is_year_start : Similar property indicating the start of the year.

        Examples
        --------
        >>> idx = ks.date_range("2017-12-30", periods=3)
        >>> idx.is_year_end
        Index([False, True, False], dtype='object')
        """
        return Index(self.to_series().dt.is_year_end)

    @property
    def func_dqfgmzl1(self) -> Index:
        """
        Boolean indicator if the date belongs to a leap year.

        A leap year is a year, which has 366 days (instead of 365) including
        29th of February as an intercalary day.
        Leap years are years which are multiples of four with the exception
        of years divisible by 100 but not by 400.

        Returns
        -------
        Index
             Booleans indicating if dates belong to a leap year.

        Examples
        --------
        >>> idx = ks.date_range("2012-01-01", "2015-01-01", freq="Y")
        >>> idx.is_leap_year
        Index([True, False, False], dtype='object')
        """
        return Index(self.to_series().dt.is_leap_year)

    @property
    def func_bi84zbp1(self) -> Index:
        """
        The number of days in the month.
        """
        return Index(self.to_series().dt.daysinmonth)

    @property
    def func_pyw0rzf4(self) -> Index:
        return Index(self.to_series().dt.days_in_month)
    days_in_month.__doc__ = daysinmonth.__doc__  # type: ignore

    def func_htfqjh9p(self, freq: Union[str, Any], *args: Any, **kwargs: Any) -> "DatetimeIndex":
        """
        Perform ceil operation on the data to the specified freq.
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.ceil(freq, *args, **kwargs))

    def func_0u5avkfz(self, freq: Union[str, Any], *args: Any, **kwargs: Any) -> "DatetimeIndex":
        """
        Perform floor operation on the data to the specified freq.
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.floor(freq, *args, **kwargs))

    def func_9r1ltatw(self, freq: Union[str, Any], *args: Any, **kwargs: Any) -> "DatetimeIndex":
        """
        Perform round operation on the data to the specified freq.
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.round(freq, *args, **kwargs))

    def func_tm369h9u(self, locale: Optional[str] = None) -> Index:
        """
        Return the month names of the DatetimeIndex with specified locale.
        """
        return Index(self.to_series().dt.month_name(locale))

    def func_b34tl3ju(self, locale: Optional[str] = None) -> Index:
        """
        Return the day names of the series with specified locale.
        """
        return Index(self.to_series().dt.day_name(locale))

    def func_53798iej(self) -> "DatetimeIndex":
        """
        Convert times to midnight.
        """
        return DatetimeIndex(self.to_series().dt.normalize())

    def func_ks7hld6a(self, date_format: str) -> Index:
        """
        Convert to a string Index using specified date_format.
        """
        return Index(self.to_series().dt.strftime(date_format))

    def func_9f8nl5eu(self, start_time: Union[datetime.time, str], 
                      end_time: Union[datetime.time, str], 
                      include_start: bool = True,
                      include_end: bool = True) -> Index:
        """
        Return index locations of values between particular times of day.
        """

        def func_4oko4swm(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.between_time(start_time, end_time, include_start, include_end)

        kdf = self.to_frame()[[]]
        id_column_name: str = verify_temp_column_name(kdf, '__id_column__')
        kdf = kdf.koalas.attach_id_column('distributed-sequence', id_column_name)
        with ks.option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(func_4oko4swm)
        return ks.Index(first_series(kdf).rename(self.name))

    def func_ujf83p74(self, time: Union[datetime.time, str], asof: bool = False) -> Index:
        """
        Return index locations of values at particular time of day.
        """
        if asof:
            raise NotImplementedError("'asof' argument is not supported")

        def func_kwmpae79(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.at_time(time, asof)

        kdf = self.to_frame()[[]]
        id_column_name: str = verify_temp_column_name(kdf, '__id_column__')
        kdf = kdf.koalas.attach_id_column('distributed-sequence', id_column_name)
        with ks.option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(func_kwmpae79)
        return ks.Index(first_series(kdf).rename(self.name))


def disallow_nanoseconds(freq: Union[str, Any]) -> None:
    if freq in ['N', 'ns']:
        raise ValueError('nanoseconds is not supported')


def func_mdxpfl7i(freq: Union[str, Any]) -> None:
    if freq in ['N', 'ns']:
        raise ValueError('nanoseconds is not supported')