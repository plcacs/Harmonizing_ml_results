from __future__ import annotations
from typing import TYPE_CHECKING, NoReturn, cast
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_integer_dtype, is_list_like
from pandas.core.dtypes.dtypes import ArrowDtype, CategoricalDtype, DatetimeTZDtype, PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.accessor import PandasDelegate, delegate_names
from pandas.core.arrays import DatetimeArray, PeriodArray, TimedeltaArray
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.base import NoNewAttributesMixin, PandasObject
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
if TYPE_CHECKING:
    from pandas import DataFrame, Series

class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _hidden_attrs: set[str] = PandasObject._hidden_attrs | {'orig', 'name'}

    def __init__(self, data: ABCSeries, orig: ABCSeries) -> None:
        ...

    def _get_values(self) -> Union[DatetimeIndex, TimedeltaIndex, PeriodArray]:
        ...

    def _delegate_property_get(self, name: str) -> Series:
        ...

    def _delegate_property_set(self, name: str, value, *args, **kwargs) -> NoReturn:
        ...

    def _delegate_method(self, name: str, *args, **kwargs) -> Series:
        ...

@delegate_names(delegate=ArrowExtensionArray, accessors=TimedeltaArray._datetimelike_ops, typ='property', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=TimedeltaArray._datetimelike_methods, typ='method', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=DatetimeArray._datetimelike_ops, typ='property', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=DatetimeArray._datetimelike_methods, typ='method', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):

    def __init__(self, data: ABCSeries, orig: ABCSeries) -> None:
        ...

    def _delegate_property_get(self, name: str) -> Series:
        ...

    def _delegate_method(self, name: str, *args, **kwargs) -> Series:
        ...

    def to_pytimedelta(self) -> ArrowExtensionArray:
        ...

    def to_pydatetime(self) -> ArrowExtensionArray:
        ...

    def isocalendar(self) -> DataFrame:
        ...

    @property
    def components(self) -> DataFrame:
        ...

@delegate_names(delegate=DatetimeArray, accessors=DatetimeArray._datetimelike_ops + ['unit'], typ='property')
@delegate_names(delegate=DatetimeArray, accessors=DatetimeArray._datetimelike_methods + ['as_unit'], typ='method')
class DatetimeProperties(Properties):
    ...

    def to_pydatetime(self) -> Series:
        ...

    @property
    def freq(self) -> str:
        ...

    def isocalendar(self) -> DataFrame:
        ...

@delegate_names(delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ='property')
@delegate_names(delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_methods, typ='method')
class TimedeltaProperties(Properties):
    ...

    def to_pytimedelta(self) -> np.ndarray:
        ...

    @property
    def components(self) -> DataFrame:
        ...

    @property
    def freq(self) -> str:
        ...

@delegate_names(delegate=PeriodArray, accessors=PeriodArray._datetimelike_ops, typ='property')
@delegate_names(delegate=PeriodArray, accessors=PeriodArray._datetimelike_methods, typ='method')
class PeriodProperties(Properties):
    ...

class CombinedDatetimelikeProperties(DatetimeProperties, TimedeltaProperties, PeriodProperties):
    ...
