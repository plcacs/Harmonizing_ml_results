from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import numpy as np

from pandas.core.accessor import PandasDelegate
from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.base import NoNewAttributesMixin, PandasObject
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING:
    from pandas import DataFrame, Series


class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _hidden_attrs: frozenset[str]
    _parent: Series
    orig: Series | None
    name: str | None

    def __init__(self, data: Series, orig: Series | None) -> None: ...
    def _get_values(self) -> DatetimeIndex | TimedeltaIndex | PeriodArray: ...
    def _delegate_property_get(self, name: str) -> Series | object: ...
    def _delegate_property_set(self, name: str, value: object, *args: object, **kwargs: object) -> NoReturn: ...
    def _delegate_method(self, name: str, *args: object, **kwargs: object) -> Series | object: ...


class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _parent: Series
    _orig: Series | None

    def __init__(self, data: Series, orig: Series | None) -> None: ...
    def _delegate_property_get(self, name: str) -> Series | object: ...
    def _delegate_method(self, name: str, *args: object, **kwargs: object) -> Series: ...
    def to_pytimedelta(self) -> np.ndarray: ...
    def to_pydatetime(self) -> Series: ...
    def isocalendar(self) -> DataFrame: ...
    @property
    def components(self) -> DataFrame: ...


class DatetimeProperties(Properties):
    def to_pydatetime(self) -> Series: ...
    @property
    def freq(self) -> str | None: ...
    def isocalendar(self) -> DataFrame: ...
    # Delegated properties from DatetimeArray
    date: Series
    time: Series
    timetz: Series
    year: Series
    month: Series
    day: Series
    hour: Series
    minute: Series
    second: Series
    microsecond: Series
    nanosecond: Series
    dayofweek: Series
    day_of_week: Series
    weekday: Series
    dayofyear: Series
    day_of_year: Series
    quarter: Series
    is_month_start: Series
    is_month_end: Series
    is_quarter_start: Series
    is_quarter_end: Series
    is_year_start: Series
    is_year_end: Series
    is_leap_year: Series
    tz: object
    unit: str
    # Delegated methods from DatetimeArray
    def to_period(self, *args: object, **kwargs: object) -> Series: ...
    def tz_localize(self, *args: object, **kwargs: object) -> Series: ...
    def tz_convert(self, *args: object, **kwargs: object) -> Series: ...
    def normalize(self, *args: object, **kwargs: object) -> Series: ...
    def strftime(self, *args: object, **kwargs: object) -> Series: ...
    def round(self, *args: object, **kwargs: object) -> Series: ...
    def floor(self, *args: object, **kwargs: object) -> Series: ...
    def ceil(self, *args: object, **kwargs: object) -> Series: ...
    def month_name(self, *args: object, **kwargs: object) -> Series: ...
    def day_name(self, *args: object, **kwargs: object) -> Series: ...
    def as_unit(self, *args: object, **kwargs: object) -> Series: ...


class TimedeltaProperties(Properties):
    def to_pytimedelta(self) -> np.ndarray: ...
    @property
    def components(self) -> DataFrame: ...
    @property
    def freq(self) -> str | None: ...
    # Delegated properties from TimedeltaArray
    days: Series
    seconds: Series
    microseconds: Series
    nanoseconds: Series
    # Delegated methods from TimedeltaArray
    def total_seconds(self, *args: object, **kwargs: object) -> Series: ...


class PeriodProperties(Properties):
    # Delegated properties from PeriodArray
    start_time: Series
    end_time: Series
    # Delegated methods from PeriodArray
    def strftime(self, *args: object, **kwargs: object) -> Series: ...


class CombinedDatetimelikeProperties(DatetimeProperties, TimedeltaProperties, PeriodProperties):
    def __new__(cls, data: Series) -> CombinedDatetimelikeProperties | ArrowTemporalProperties | DatetimeProperties | TimedeltaProperties | PeriodProperties: ...  # type: ignore[misc]