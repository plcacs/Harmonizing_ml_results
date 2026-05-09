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

class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    ...

@delegate_names(delegate=ArrowExtensionArray, accessors=TimedeltaArray._datetimelike_ops, typ='property', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=TimedeltaArray._datetimelike_methods, typ='method', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    ...

class DatetimeProperties(Properties):
    ...

class TimedeltaProperties(Properties):
    ...

class PeriodProperties(Properties):
    ...

class CombinedDatetimelikeProperties(DatetimeProperties, TimedeltaProperties, PeriodProperties):
    ...

def __new__(cls: type[CombinedDatetimelikeProperties], data: ABCSeries) -> CombinedDatetimelikeProperties:
    ...
