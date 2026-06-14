from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.core.accessor import PandasDelegate, delegate_names
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
    def _delegate_property_get(self, name: str) -> Series: ...
    def _delegate_property_set(self, name: str, value: object, *args: object, **kwargs: object) -> None: ...
    def _delegate_method(self, name: str, *args: object, **kwargs: object) -> Series: ...


class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _parent: Series
    _orig: Series | None

    def __init__(self, data: Series, orig: Series | None) -> None: ...
    def _delegate_property_get(self, name: str) -> Series: ...
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


class TimedeltaProperties(Properties):
    def to_pytimedelta(self) -> np.ndarray: ...
    @property
    def components(self) -> DataFrame: ...
    @property
    def freq(self) -> str | None: ...


class PeriodProperties(Properties): ...


class CombinedDatetimelikeProperties(
    DatetimeProperties, TimedeltaProperties, PeriodProperties
):
    def __new__(
        cls, data: Series
    ) -> DatetimeProperties | TimedeltaProperties | PeriodProperties | ArrowTemporalProperties: ...