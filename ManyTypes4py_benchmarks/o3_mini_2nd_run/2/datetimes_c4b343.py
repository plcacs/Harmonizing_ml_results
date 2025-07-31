import datetime
from functools import partial
from typing import Any, Optional, Union, Type
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

    def __new__(cls: Type["DatetimeIndex"],
                data: Optional[Any] = None,
                freq: Any = _NoValue,
                normalize: bool = False,
                closed: Optional[str] = None,
                ambiguous: Union[str, bool, Any] = 'raise',
                dayfirst: bool = False,
                yearfirst: bool = False,
                dtype: Optional[Any] = None,
                copy: bool = False,
                name: Optional[Any] = None) -> "DatetimeIndex":
        if not is_hashable(name):
            raise TypeError('Index.name must be a hashable type')
        if isinstance(data, (Series, Index)):
            if dtype is None:
                dtype = 'datetime64[ns]'
            return Index(data, dtype=dtype, copy=copy, name=name)
        kwargs: dict[str, Any] = dict(data=data, normalize=normalize, closed=closed,
                                      ambiguous=ambiguous, dayfirst=dayfirst,
                                      yearfirst=yearfirst, dtype=dtype,
                                      copy=copy, name=name)
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
        raise AttributeError("'DatetimeIndex' object has no attribute '{}'".format(item))

    @property
    def year(self) -> Index:
        """
        The year of the datetime.
        """
        return Index(self.to_series().dt.year)

    @property
    def month(self) -> Index:
        """
        The month of the timestamp as January = 1 December = 12.
        """
        return Index(self.to_series().dt.month)

    @property
    def day(self) -> Index:
        """
        The days of the datetime.
        """
        return Index(self.to_series().dt.day)

    @property
    def hour(self) -> Index:
        """
        The hours of the datetime.
        """
        return Index(self.to_series().dt.hour)

    @property
    def minute(self) -> Index:
        """
        The minutes of the datetime.
        """
        return Index(self.to_series().dt.minute)

    @property
    def second(self) -> Index:
        """
        The seconds of the datetime.
        """
        return Index(self.to_series().dt.second)

    @property
    def microsecond(self) -> Index:
        """
        The microseconds of the datetime.
        """
        return Index(self.to_series().dt.microsecond)

    @property
    def week(self) -> Index:
        """
        The week ordinal of the year.
        """
        return Index(self.to_series().dt.week)

    @property
    def weekofyear(self) -> Index:
        return Index(self.to_series().dt.weekofyear)
    weekofyear.__doc__ = week.__doc__

    @property
    def dayofweek(self) -> Index:
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
        """
        return Index(self.to_series().dt.dayofweek)

    @property
    def day_of_week(self) -> Index:
        return self.dayofweek
    day_of_week.__doc__ = dayofweek.__doc__

    @property
    def weekday(self) -> Index:
        return Index(self.to_series().dt.weekday)
    weekday.__doc__ = dayofweek.__doc__

    @property
    def dayofyear(self) -> Index:
        """
        The ordinal day of the year.
        """
        return Index(self.to_series().dt.dayofyear)

    @property
    def day_of_year(self) -> Index:
        return self.dayofyear
    day_of_year.__doc__ = dayofyear.__doc__

    @property
    def quarter(self) -> Index:
        """
        The quarter of the date.
        """
        return Index(self.to_series().dt.quarter)

    @property
    def is_month_start(self) -> Index:
        """
        Indicates whether the date is the first day of the month.

        Returns
        -------
        Index
            Returns an Index with boolean values
        """
        return Index(self.to_series().dt.is_month_start)

    @property
    def is_month_end(self) -> Index:
        """
        Indicates whether the date is the last day of the month.

        Returns
        -------
        Index
            Returns an Index with boolean values.
        """
        return Index(self.to_series().dt.is_month_end)

    @property
    def is_quarter_start(self) -> Index:
        """
        Indicator for whether the date is the first day of a quarter.

        Returns
        -------
        is_quarter_start : Index
            Returns an Index with boolean values.
        """
        return Index(self.to_series().dt.is_quarter_start)

    @property
    def is_quarter_end(self) -> Index:
        """
        Indicator for whether the date is the last day of a quarter.

        Returns
        -------
        is_quarter_end : Index
            Returns an Index with boolean values.
        """
        return Index(self.to_series().dt.is_quarter_end)

    @property
    def is_year_start(self) -> Index:
        """
        Indicate whether the date is the first day of a year.

        Returns
        -------
        Index
            Returns an Index with boolean values.
        """
        return Index(self.to_series().dt.is_year_start)

    @property
    def is_year_end(self) -> Index:
        """
        Indicate whether the date is the last day of the year.

        Returns
        -------
        Index
            Returns an Index with boolean values.
        """
        return Index(self.to_series().dt.is_year_end)

    @property
    def is_leap_year(self) -> Index:
        """
        Boolean indicator if the date belongs to a leap year.
        """
        return Index(self.to_series().dt.is_leap_year)

    @property
    def daysinmonth(self) -> Index:
        """
        The number of days in the month.
        """
        return Index(self.to_series().dt.daysinmonth)

    @property
    def days_in_month(self) -> Index:
        return Index(self.to_series().dt.days_in_month)
    days_in_month.__doc__ = daysinmonth.__doc__

    def ceil(self, freq: Any, *args: Any, **kwargs: Any) -> "DatetimeIndex":
        """
        Perform ceil operation on the data to the specified freq.
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.ceil(freq, *args, **kwargs))

    def floor(self, freq: Any, *args: Any, **kwargs: Any) -> "DatetimeIndex":
        """
        Perform floor operation on the data to the specified freq.
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.floor(freq, *args, **kwargs))

    def round(self, freq: Any, *args: Any, **kwargs: Any) -> "DatetimeIndex":
        """
        Perform round operation on the data to the specified freq.
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.round(freq, *args, **kwargs))

    def month_name(self, locale: Optional[str] = None) -> Index:
        """
        Return the month names of the DatetimeIndex with specified locale.
        """
        return Index(self.to_series().dt.month_name(locale))

    def day_name(self, locale: Optional[str] = None) -> Index:
        """
        Return the day names of the series with specified locale.
        """
        return Index(self.to_series().dt.day_name(locale))

    def normalize(self) -> "DatetimeIndex":
        """
        Convert times to midnight.
        """
        return DatetimeIndex(self.to_series().dt.normalize())

    def strftime(self, date_format: str) -> Index:
        """
        Convert to a string Index using specified date_format.
        """
        return Index(self.to_series().dt.strftime(date_format))

    def indexer_between_time(self,
                             start_time: Union[datetime.time, str],
                             end_time: Union[datetime.time, str],
                             include_start: bool = True,
                             include_end: bool = True) -> Index:
        """
        Return index locations of values between particular times of day.
        """
        def pandas_between_time(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.between_time(start_time, end_time, include_start, include_end)
        kdf: ks.DataFrame = self.to_frame()[[]]
        id_column_name: str = verify_temp_column_name(kdf, '__id_column__')
        kdf = kdf.koalas.attach_id_column('distributed-sequence', id_column_name)
        with ks.option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_between_time)
        return ks.Index(first_series(kdf).rename(self.name))

    def indexer_at_time(self,
                        time: Union[datetime.time, str],
                        asof: bool = False) -> Index:
        """
        Return index locations of values at particular time of day.
        """
        if asof:
            raise NotImplementedError("'asof' argument is not supported")

        def pandas_at_time(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.at_time(time, asof)
        kdf: ks.DataFrame = self.to_frame()[[]]
        id_column_name: str = verify_temp_column_name(kdf, '__id_column__')
        kdf = kdf.koalas.attach_id_column('distributed-sequence', id_column_name)
        with ks.option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_at_time)
        return ks.Index(first_series(kdf).rename(self.name))

def disallow_nanoseconds(freq: Any) -> None:
    if freq in ['N', 'ns']:
        raise ValueError('nanoseconds is not supported')