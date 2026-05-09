from typing import Any, Optional, Union, List
from pandas.api.types import _NoValue
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeDatetimeIndex
from databricks.koalas.series import Series
from databricks.koalas.utils import verify_temp_column_name

class DatetimeIndex(Index):
    def __new__(
        cls,
        data: Optional[Any] = None,
        freq: Union[str, _NoValue] = _NoValue,
        normalize: bool = False,
        closed: Optional[str] = None,
        ambiguous: Union[str, List[bool]] = 'raise',
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Optional[Any] = None,
    ) -> 'DatetimeIndex':
        ...

    def __getattr__(self, item: str) -> Any:
        ...

    @property
    def year(self) -> Index[int]:
        ...

    @property
    def month(self) -> Index[int]:
        ...

    @property
    def day(self) -> Index[int]:
        ...

    @property
    def hour(self) -> Index[int]:
        ...

    @property
    def minute(self) -> Index[int]:
        ...

    @property
    def second(self) -> Index[int]:
        ...

    @property
    def microsecond(self) -> Index[int]:
        ...

    @property
    def week(self) -> Index[int]:
        ...

    @property
    def weekofyear(self) -> Index[int]:
        ...

    @property
    def dayofweek(self) -> Index[int]:
        ...

    @property
    def weekday(self) -> Index[int]:
        ...

    @property
    def dayofyear(self) -> Index[int]:
        ...

    @property
    def day_of_year(self) -> Index[int]:
        ...

    @property
    def quarter(self) -> Index[int]:
        ...

    @property
    def is_month_start(self) -> Index[bool]:
        ...

    @property
    def is_month_end(self) -> Index[bool]:
        ...

    @property
    def is_quarter_start(self) -> Index[bool]:
        ...

    @property
    def is_quarter_end(self) -> Index[bool]:
        ...

    @property
    def is_year_start(self) -> Index[bool]:
        ...

    @property
    def is_year_end(self) -> Index[bool]:
        ...

    @property
    def is_leap_year(self) -> Index[bool]:
        ...

    @property
    def daysinmonth(self) -> Index[int]:
        ...

    @property
    def days_in_month(self) -> Index[int]:
        ...

    def ceil(self, freq: Union[str, Any]) -> 'DatetimeIndex':
        ...

    def floor(self, freq: Union[str, Any]) -> 'DatetimeIndex':
        ...

    def round(self, freq: Union[str, Any]) -> 'DatetimeIndex':
        ...

    def month_name(self, locale: Optional[str] = None) -> Index[str]:
        ...

    def day_name(self, locale: Optional[str] = None) -> Index[str]:
        ...

    def normalize(self) -> 'DatetimeIndex':
        ...

    def strftime(self, date_format: str) -> Index[str]:
        ...

    def indexer_between_time(
        self,
        start_time: Union[datetime.time, str],
        end_time: Union[datetime.time, str],
        include_start: bool = True,
        include_end: bool = True,
    ) -> Index[int]:
        ...

    def indexer_at_time(
        self,
        time: Union[datetime.time, str],
        asof: bool = False,
    ) -> Index[int]:
        ...

def disallow_nanoseconds(freq: Union[str, Any]) -> None:
    ...