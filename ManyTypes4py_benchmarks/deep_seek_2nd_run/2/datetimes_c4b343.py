import datetime
from functools import partial
from typing import Any, Optional, Union, cast, Dict, List, Tuple, TypeVar
import pandas as pd
from pandas.api.types import is_hashable
from pyspark._globals import _NoValue
from databricks import koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeDatetimeIndex
from databricks.koalas.series import Series, first_series
from databricks.koalas.utils import verify_temp_column_name

T = TypeVar('T')

class DatetimeIndex(Index):
    def __new__(
        cls,
        data: Optional[Union[pd.Series, pd.Index, List[Any], Tuple[Any, ...]] = None,
        freq: Any = _NoValue,
        normalize: bool = False,
        closed: Optional[str] = None,
        ambiguous: str = 'raise',
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Optional[Union[str, Tuple[str, ...]] = None
    ) -> 'DatetimeIndex':
        if not is_hashable(name):
            raise TypeError('Index.name must be a hashable type')
        if isinstance(data, (Series, Index)):
            if dtype is None:
                dtype = 'datetime64[ns]'
            return Index(data, dtype=dtype, copy=copy, name=name)
        kwargs: Dict[str, Any] = dict(
            data=data, normalize=normalize, closed=closed, ambiguous=ambiguous,
            dayfirst=dayfirst, yearfirst=yearfirst, dtype=dtype, copy=copy, name=name
        )
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
        raise AttributeError("'DatetimeIndex' object has no attribute '{}'".format(item))

    @property
    def year(self) -> Index:
        return Index(self.to_series().dt.year)

    @property
    def month(self) -> Index:
        return Index(self.to_series().dt.month)

    @property
    def day(self) -> Index:
        return Index(self.to_series().dt.day)

    @property
    def hour(self) -> Index:
        return Index(self.to_series().dt.hour)

    @property
    def minute(self) -> Index:
        return Index(self.to_series().dt.minute)

    @property
    def second(self) -> Index:
        return Index(self.to_series().dt.second)

    @property
    def microsecond(self) -> Index:
        return Index(self.to_series().dt.microsecond)

    @property
    def week(self) -> Index:
        return Index(self.to_series().dt.week)

    @property
    def weekofyear(self) -> Index:
        return Index(self.to_series().dt.weekofyear)

    @property
    def dayofweek(self) -> Index:
        return Index(self.to_series().dt.dayofweek)

    @property
    def day_of_week(self) -> Index:
        return self.dayofweek

    @property
    def weekday(self) -> Index:
        return Index(self.to_series().dt.weekday)

    @property
    def dayofyear(self) -> Index:
        return Index(self.to_series().dt.dayofyear)

    @property
    def day_of_year(self) -> Index:
        return self.dayofyear

    @property
    def quarter(self) -> Index:
        return Index(self.to_series().dt.quarter)

    @property
    def is_month_start(self) -> Index:
        return Index(self.to_series().dt.is_month_start)

    @property
    def is_month_end(self) -> Index:
        return Index(self.to_series().dt.is_month_end)

    @property
    def is_quarter_start(self) -> Index:
        return Index(self.to_series().dt.is_quarter_start)

    @property
    def is_quarter_end(self) -> Index:
        return Index(self.to_series().dt.is_quarter_end)

    @property
    def is_year_start(self) -> Index:
        return Index(self.to_series().dt.is_year_start)

    @property
    def is_year_end(self) -> Index:
        return Index(self.to_series().dt.is_year_end)

    @property
    def is_leap_year(self) -> Index:
        return Index(self.to_series().dt.is_leap_year)

    @property
    def daysinmonth(self) -> Index:
        return Index(self.to_series().dt.daysinmonth)

    @property
    def days_in_month(self) -> Index:
        return Index(self.to_series().dt.days_in_month)

    def ceil(self, freq: str, *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.ceil(freq, *args, **kwargs))

    def floor(self, freq: str, *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.floor(freq, *args, **kwargs))

    def round(self, freq: str, *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        disallow_nanoseconds(freq)
        return DatetimeIndex(self.to_series().dt.round(freq, *args, **kwargs))

    def month_name(self, locale: Optional[str] = None) -> Index:
        return Index(self.to_series().dt.month_name(locale))

    def day_name(self, locale: Optional[str] = None) -> Index:
        return Index(self.to_series().dt.day_name(locale))

    def normalize(self) -> 'DatetimeIndex':
        return DatetimeIndex(self.to_series().dt.normalize())

    def strftime(self, date_format: str) -> Index:
        return Index(self.to_series().dt.strftime(date_format))

    def indexer_between_time(
        self,
        start_time: Union[datetime.time, str],
        end_time: Union[datetime.time, str],
        include_start: bool = True,
        include_end: bool = True
    ) -> Index:
        def pandas_between_time(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.between_time(start_time, end_time, include_start, include_end)
        kdf = self.to_frame()[[]]
        id_column_name = verify_temp_column_name(kdf, '__id_column__')
        kdf = kdf.koalas.attach_id_column('distributed-sequence', id_column_name)
        with ks.option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_between_time)
        return ks.Index(first_series(kdf).rename(self.name))

    def indexer_at_time(self, time: Union[datetime.time, str], asof: bool = False) -> Index:
        if asof:
            raise NotImplementedError("'asof' argument is not supported")

        def pandas_at_time(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.at_time(time, asof)
        kdf = self.to_frame()[[]]
        id_column_name = verify_temp_column_name(kdf, '__id_column__')
        kdf = kdf.koalas.attach_id_column('distributed-sequence', id_column_name)
        with ks.option_context('compute.default_index_type', 'distributed'):
            kdf = kdf.koalas.apply_batch(pandas_at_time)
        return ks.Index(first_series(kdf).rename(self.name))

def disallow_nanoseconds(freq: str) -> None:
    if freq in ['N', 'ns']:
        raise ValueError('nanoseconds is not supported')
