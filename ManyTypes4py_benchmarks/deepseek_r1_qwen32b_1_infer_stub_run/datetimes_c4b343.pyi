from datetime import datetime, time
from typing import Any, Optional, Union, List, Tuple, Dict, Set, FrozenSet, TypeVar, Protocol, overload, Literal, Final, ClassVar, NewType, get_type_hints, Type, Callable, Iterator, Iterable, Generator, AnyStr, overload, NoReturn, TYPE_CHECKING
from pandas._libs.tslibs.offsets import BaseOffset
from pyspark._globals import _NoValue
from databricks.koalas.indexes.base import Index
from databricks.koalas.series import Series
from databricks.koalas.utils import first_series

class DatetimeIndex(Index):
    """
    Immutable ndarray-like of datetime64 data.
    """
    def __new__(cls, data: Optional[Any] = ..., freq: Union[_NoValue, Any] = ..., normalize: bool = ..., closed: Optional[Any] = ..., ambiguous: Union[str, Any] = ..., dayfirst: bool = ..., yearfirst: bool = ..., dtype: Optional[Any] = ..., copy: bool = ..., name: Optional[Any] = ...) -> Any:
        ...

    def __getattr__(self: DatetimeIndex, item: str) -> Any:
        ...

    @property
    def year(self: DatetimeIndex) -> Index:
        ...

    @property
    def month(self: DatetimeIndex) -> Index:
        ...

    @property
    def day(self: DatetimeIndex) -> Index:
        ...

    @property
    def hour(self: DatetimeIndex) -> Index:
        ...

    @property
    def minute(self: DatetimeIndex) -> Index:
        ...

    @property
    def second(self: DatetimeIndex) -> Index:
        ...

    @property
    def microsecond(self: DatetimeIndex) -> Index:
        ...

    @property
    def week(self: DatetimeIndex) -> Index:
        ...

    @property
    def weekofyear(self: DatetimeIndex) -> Index:
        ...

    @property
    def dayofweek(self: DatetimeIndex) -> Index:
        ...

    @property
    def day_of_week(self: DatetimeIndex) -> Index:
        ...

    @property
    def weekday(self: DatetimeIndex) -> Index:
        ...

    @property
    def dayofyear(self: DatetimeIndex) -> Index:
        ...

    @property
    def day_of_year(self: DatetimeIndex) -> Index:
        ...

    @property
    def quarter(self: DatetimeIndex) -> Index:
        ...

    @property
    def is_month_start(self: DatetimeIndex) -> Index:
        ...

    @property
    def is_month_end(self: DatetimeIndex) -> Index:
        ...

    @property
    def is_quarter_start(self: DatetimeIndex) -> Index:
        ...

    @property
    def is_quarter_end(self: DatetimeIndex) -> Index:
        ...

    @property
    def is_year_start(self: DatetimeIndex) -> Index:
        ...

    @property
    def is_year_end(self: DatetimeIndex) -> Index:
        ...

    @property
    def is_leap_year(self: DatetimeIndex) -> Index:
        ...

    @property
    def daysinmonth(self: DatetimeIndex) -> Index:
        ...

    @property
    def days_in_month(self: DatetimeIndex) -> Index:
        ...

    def ceil(self: DatetimeIndex, freq: Union[str, BaseOffset], *args: Any, **kwargs: Any) -> DatetimeIndex:
        ...

    def floor(self: DatetimeIndex, freq: Union[str, BaseOffset], *args: Any, **kwargs: Any) -> DatetimeIndex:
        ...

    def round(self: DatetimeIndex, freq: Union[str, BaseOffset], *args: Any, **kwargs: Any) -> DatetimeIndex:
        ...

    def month_name(self: DatetimeIndex, locale: Optional[str] = ...) -> Index:
        ...

    def day_name(self: DatetimeIndex, locale: Optional[str] = ...) -> Index:
        ...

    def normalize(self: DatetimeIndex) -> DatetimeIndex:
        ...

    def strftime(self: DatetimeIndex, date_format: str) -> Index:
        ...

    def indexer_between_time(self: DatetimeIndex, start_time: Union[time, str], end_time: Union[time, str], include_start: bool = ..., include_end: bool = ...) -> Index:
        ...

    def indexer_at_time(self: DatetimeIndex, time: Union[time, str], asof: bool = ...) -> Index:
        ...

def disallow_nanoseconds(freq: str) -> None:
    ...