import datetime
from functools import partial
from typing import Any, Optional, Union
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

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional datetime-like data to construct index with.
    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        'infer' can be passed in order to set the frequency of the index as the
        inferred frequency upon creation.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    closed : {'left', 'right'}, optional
        Set whether to include `start` and `end` that are on the
        boundary. The default includes boundary points on either end.
    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from 03:00
        DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
        and at 01:30:00 UTC. In such a situation, the `ambiguous` parameter
        dictates how ambiguous times should be handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False signifies a
          non-DST time (note that this flag is only applicable for ambiguous
          times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous times.
    dayfirst : bool, default False
        If True, parse dates in `data` with the day first order.
    yearfirst : bool, default False
        If True parse dates in `data` with the year first order.
    dtype : numpy.dtype or str, default None
        Note that the only NumPy dtype allowed is ‘datetime64[ns]’.
    copy : bool, default False
        Make a copy of input ndarray.
    name : label, default None
        Name to be stored in the index.

    See Also
    --------
    Index : The base pandas Index type.
    to_datetime : Convert argument to datetime.

    Examples
    --------
    >>> ks.DatetimeIndex(['1970-01-01', '1970-01-01', '1970-01-01'])
    DatetimeIndex(['1970-01-01', '1970-01-01', '1970-01-01'], dtype='datetime64[ns]', freq=None)

    From a Series:

    >>> from datetime import datetime
    >>> s = ks.Series([datetime(2021, 3, 1), datetime(2021, 3, 2)], index=[10, 20])
    >>> ks.DatetimeIndex(s)
    DatetimeIndex(['2021-03-01', '2021-03-02'], dtype='datetime64[ns]', freq=None)

    From an Index:

    >>> idx = ks.DatetimeIndex(['1970-01-01', '1970-01-01', '1970-01-01'])
    >>> ks.DatetimeIndex(idx)
    DatetimeIndex(['1970-01-01', '1970-01-01', '1970-01-01'], dtype='datetime64[ns]', freq=None)
    """

    def __new__(cls, data: Optional[Union[pd.Series, Index]] = None, 
               freq: Optional[str] = _NoValue, 
               normalize: bool = False, 
               closed: Optional[str] = None, 
               ambiguous: str = 'raise', 
               dayfirst: bool = False, 
               yearfirst: bool = False, 
               dtype: Optional[str] = None, 
               copy: bool = False, 
               name: Optional[str] = None) -> 'DatetimeIndex':
        if not is_hashable(name):
            raise TypeError('Index.name must be a hashable type')
        if isinstance(data, (Series, Index)):
            if dtype is None:
                dtype = 'datetime64[ns]'
            return Index(data, dtype=dtype, copy=copy, name=name)
        kwargs = dict(data=data, normalize=normalize, closed=closed, ambiguous=ambiguous, dayfirst=dayfirst, yearfirst=yearfirst, dtype=dtype, copy=copy, name=name)
        if freq is not _NoValue:
            kwargs['freq'] = freq
        return ks.from_pandas(pd.DatetimeIndex(**kwargs))

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeDatetimeIndex, item):
            property_or_func = getattr(MissingPandasLikeDatetimeIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)
            else:
                return partial(property_or_func, self)
        raise AttributeError(f"'DatetimeIndex' object has no attribute '{item}'")

    @property
    def year(self) -> 'DatetimeIndex':
        """
        The year of the datetime.
        """
        return Index(self.to_series().dt.year)

    @property
    def month(self) -> 'DatetimeIndex':
        """
        The month of the timestamp as January = 1 December = 12.
        """
        return Index(self.to_series().dt.month)

    @property
    def day(self) -> 'DatetimeIndex':
        """
        The days of the datetime.
        """
        return Index(self.to_series().dt.day)

    @property
    def hour(self) -> 'DatetimeIndex':
        """
        The hours of the datetime.
        """
        return Index(self.to_series().dt.hour)

    @property
    def minute(self) -> 'DatetimeIndex':
        """
        The minutes of the datetime.
        """
        return Index(self.to_series().dt.minute)

    @property
    def second(self) -> 'DatetimeIndex':
        """
        The seconds of the datetime.
        """
        return Index(self.to_series().dt.second)

    @property
    def microsecond(self) -> 'DatetimeIndex':
        """
        The microseconds of the datetime.
        """
        return Index(self.to_series().dt.microsecond)

    @property
    def week(self) -> 'DatetimeIndex':
        """
        The week ordinal of the year.
        """
        return Index(self.to_series().dt.week)

    @property
    def weekofyear(self) -> 'DatetimeIndex':
        return Index(self.to_series().dt.weekofyear)

    @property
    def dayofweek(self) -> 'DatetimeIndex':
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
    def daysinmonth(self) -> 'DatetimeIndex':
        """
        The number of days in the month.
        """
        return Index(self.to_series().dt.daysinmonth)

    @property
    def days_in_month(self) -> 'DatetimeIndex':
        return Index(self.to_series().dt.days_in_month)

    def ceil(self, freq: str, *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        """
        Perform ceil operation on the data to the specified freq.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to ceil the index to. Must be a fixed
            frequency like 'S' (second) not 'ME' (month end).

        Returns
        -------
        DatetimeIndex

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        Examples
        --------
        >>> rng = ks.date_range('1/1/2018 11:59:00', periods=3, freq='min')
        >>> rng.ceil('H')  # doctest: +NORMALIZE_WHITESPACE
        DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 13:00:00'],
                      dtype='datetime64[ns]', freq=None)
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.ceil(freq, *args, **kwargs))

    def floor(self, freq: str, *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        """
        Perform floor operation on the data to the specified freq.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to floor the index to. Must be a fixed
            frequency like 'S' (second) not 'ME' (month end).

        Returns
        -------
        DatetimeIndex

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        Examples
        --------
        >>> rng = ks.date_range('1/1/2018 11:59:00', periods=3, freq='min')
        >>> rng.floor("H")  # doctest: +NORMALIZE_WHITESPACE
        DatetimeIndex(['2018-01-01 11:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:00:00'],
                      dtype='datetime64[ns]', freq=None)
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.floor(freq, *args, **kwargs))

    def round(self, freq: str, *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        """
        Perform round operation on the data to the specified freq.

        Parameters
        ----------
        freq : str or Offset
            The frequency level to round the index to. Must be a fixed
            frequency like 'S' (second) not 'ME' (month end).

        Returns
        -------
        DatetimeIndex

        Raises
        ------
        ValueError if the `freq` cannot be converted.

        Examples
        --------
        >>> rng = ks.date_range('1/1/2018 11:59:00', periods=3, freq='min')
        >>> rng.round("H")  # doctest: +NORMALIZE_WHITESPACE
        DatetimeIndex(['2018-01-01 12:00:00', '2018-01-01 12:00:00',
                       '2018-01-01 12:00:00'],
                      dtype='datetime64[ns]', freq=None)
        """
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.round(freq, *args, **kwargs))

    def month_name(self, locale: Optional[str] = None) -> 'DatetimeIndex':
        """
        Return the month names of the DatetimeIndex with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the month name.
            Default is English locale.

        Returns
        -------
        Index
            Index of month names.

        See Also
        --------
        normalize : Return series with times to midnight.
        round : Round the series to the specified freq.
        floor : Floor the series to the specified freq.

        Examples
        --------
        >>> idx = ks.date_range(start='2018-01', freq='M', periods=3)
        >>> idx.month_name()
        Index(['January', 'February', 'March'], dtype='object')
        """
        return Index(self.to_series().dt.month_name(locale))

    def day_name(self, locale: Optional[str] = None) -> 'DatetimeIndex':
        """
        Return the day names of the series with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the day name.
            Default is English locale.

        Returns
        -------
        Index
            Index of day names.

        See Also
        --------
        normalize : Return series with times to midnight.
        round : Round the series to the specified freq.
        floor : Floor the series to the specified freq.

        Examples
        --------
        >>> idx = ks.date_range(start='2018-01-01', freq='D', periods=3)
        >>> idx.day_name()
        Index(['Monday', 'Tuesday', 'Wednesday'], dtype='object')
        """
        return Index(self.to