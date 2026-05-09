import datetime
from typing import Any, Optional, Union, List, Property, bool, int, str
from pandas.api.types import is_hashable
from pyspark._globals import _NoValue
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeDatetimeIndex
from databricks.koalas.series import Series
from pandas.tseries.frequencies import BaseOffset

class DatetimeIndex(Index):
    """
    Immutable ndarray-like of datetime64 data.
    """
    def __new__(cls, data: Any = None, freq: Union[str, BaseOffset] = _NoValue, normalize: bool = False, closed: Optional[str] = None, ambiguous: Union[str, List[bool]] = 'raise', dayfirst: bool = False, yearfirst: bool = False, dtype: Optional[Any] = None, copy: bool = False, name: Optional[Any] = None) -> 'DatetimeIndex':
        ...

    def __getattr__(self, item: str) -> Any:
        ...

    @property
    def year(self) -> Property[Index[int], None]:
        ...

    @property
    def month(self) -> Property[Index[int], None]:
        ...

    @property
    def day(self) -> Property[Index[int], None]:
        ...

    @property
    def hour(self) -> Property[Index[int], None]:
        ...

    @property
    def minute(self) -> Property[Index[int], None]:
        ...

    @property
    def second(self) -> Property[Index[int], None]:
        ...

    @property
    def microsecond(self) -> Property[Index[int], None]:
        ...

    @property
    def week(self) -> Property[Index[int], None]:
        ...

    @property
    def weekofyear(self) -> Property[Index[int], None]:
        ...

    @property
    def dayofweek(self) -> Property[Index[int], None]:
        ...

    @property
    def day_of_week(self) -> Property[Index[int], None]:
        ...

    @property
    def weekday(self) -> Property[Index[int], None]:
        ...

    @property
    def dayofyear(self) -> Property[Index[int], None]:
        ...

    @property
    def day_of_year(self) -> Property[Index[int], None]:
        ...

    @property
    def quarter(self) -> Property[Index[int], None]:
        ...

    @property
    def is_month_start(self) -> Property[Index[bool], None]:
        ...

    @property
    def is_month_end(self) -> Property[Index[bool], None]:
        ...

    @property
    def is_quarter_start(self) -> Property[Index[bool], None]:
        ...

    @property
    def is_quarter_end(self) -> Property[Index[bool], None]:
        ...

    @property
    def is_year_start(self) -> Property[Index[bool], None]:
        ...

    @property
    def is_year_end(self) -> Property[Index[bool], None]:
        ...

    @property
    def is_leap_year(self) -> Property[Index[bool], None]:
        ...

    @property
    def daysinmonth(self) -> Property[Index[int], None]:
        ...

    @property
    def days_in_month(self) -> Property[Index[int], None]:
        ...

    def ceil(self, freq: Union[str, BaseOffset], *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        ...

    def floor(self, freq: Union[str, BaseOffset], *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        ...

    def round(self, freq: Union[str, BaseOffset], *args: Any, **kwargs: Any) -> 'DatetimeIndex':
        ...

    def month_name(self, locale: Optional[str] = None) -> Index[str]:
        ...

    def day_name(self, locale: Optional[str] = None) -> Index[str]:
        ...

    def normalize(self) -> 'DatetimeIndex':
        ...

    def strftime(self, date_format: str) -> Index[str]:
        ...

    def indexer_between_time(self, start_time: Union[datetime.time, str], end_time: Union[datetime.time, str], include_start: bool = True, include_end: bool = True) -> Index[int]:
        ...

    def indexer_at_time(self, time: Union[datetime.time, str], asof: bool = False) -> Index[int]:
        ...

def disallow_nanoseconds(freq: Union[str, BaseOffset]) -> None:
    ...