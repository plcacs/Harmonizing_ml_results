"""
datetimelike delegation
"""
from __future__ import annotations
from typing import TYPE_CHECKING, NoReturn, cast, Any, Union, Optional, Dict, List, Tuple
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
    from numpy import ndarray
    from datetime import datetime, timedelta

class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _hidden_attrs: set[str] = PandasObject._hidden_attrs | {'orig', 'name'}

    def __init__(self, data: ABCSeries, orig: Optional[ABCSeries]) -> None:
        if not isinstance(data, ABCSeries):
            raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')
        self._parent: ABCSeries = data
        self.orig: Optional[ABCSeries] = orig
        self.name: Optional[str] = getattr(data, 'name', None)
        self._freeze()

    def _get_values(self) -> Union[DatetimeIndex, TimedeltaIndex, PeriodArray]:
        data = self._parent
        if lib.is_np_dtype(data.dtype, 'M'):
            return DatetimeIndex(data, copy=False, name=self.name)
        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeIndex(data, copy=False, name=self.name)
        elif lib.is_np_dtype(data.dtype, 'm'):
            return TimedeltaIndex(data, copy=False, name=self.name)
        elif isinstance(data.dtype, PeriodDtype):
            return PeriodArray(data, copy=False)
        raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')

    def _delegate_property_get(self, name: str) -> Any:
        from pandas import Series
        values = self._get_values()
        result = getattr(values, name)
        if isinstance(result, np.ndarray):
            if is_integer_dtype(result):
                result = result.astype('int64')
        elif not is_list_like(result):
            return result
        result = np.asarray(result)
        if self.orig is not None:
            index = self.orig.index
        else:
            index = self._parent.index
        return Series(result, index=index, name=self.name).__finalize__(self._parent)

    def _delegate_property_set(self, name: str, value: Any, *args: Any, **kwargs: Any) -> NoReturn:
        raise ValueError('modifications to a property of a datetimelike object are not supported. Change values on the original.')

    def _delegate_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        from pandas import Series
        values = self._get_values()
        method = getattr(values, name)
        result = method(*args, **kwargs)
        if not is_list_like(result):
            return result
        return Series(result, index=self._parent.index, name=self.name).__finalize__(self._parent)

@delegate_names(delegate=ArrowExtensionArray, accessors=TimedeltaArray._datetimelike_ops, typ='property', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=TimedeltaArray._datetimelike_methods, typ='method', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=DatetimeArray._datetimelike_ops, typ='property', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=DatetimeArray._datetimelike_methods, typ='method', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    def __init__(self, data: ABCSeries, orig: Optional[ABCSeries]) -> None:
        if not isinstance(data, ABCSeries):
            raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')
        self._parent: ABCSeries = data
        self._orig: Optional[ABCSeries] = orig
        self._freeze()

    def _delegate_property_get(self, name: str) -> Any:
        if not hasattr(self._parent.array, f'_dt_{name}'):
            raise NotImplementedError(f'dt.{name} is not supported for {self._parent.dtype}')
        result = getattr(self._parent.array, f'_dt_{name}')
        if not is_list_like(result):
            return result
        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        result = type(self._parent)(result, index=index, name=self._parent.name).__finalize__(self._parent)
        return result

    def _delegate_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self._parent.array, f'_dt_{name}'):
            raise NotImplementedError(f'dt.{name} is not supported for {self._parent.dtype}')
        result = getattr(self._parent.array, f'_dt_{name}')(*args, **kwargs)
        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        result = type(self._parent)(result, index=index, name=self._parent.name).__finalize__(self._parent)
        return result

    def to_pytimedelta(self) -> ndarray:
        warnings.warn(f'The behavior of {type(self).__name__}.to_pytimedelta is deprecated, in a future version this will return a Series containing python datetime.timedelta objects instead of an ndarray. To retain the old behavior, call `np.array` on the result', FutureWarning, stacklevel=find_stack_level())
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pytimedelta()

    def to_pydatetime(self) -> ndarray:
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pydatetime()

    def isocalendar(self) -> DataFrame:
        from pandas import DataFrame
        result = cast(ArrowExtensionArray, self._parent.array)._dt_isocalendar()._pa_array.combine_chunks()
        iso_calendar_df = DataFrame({col: type(self._parent.array)(result.field(i)) for i, col in enumerate(['year', 'week', 'day'])})
        return iso_calendar_df

    @property
    def components(self) -> DataFrame:
        from pandas import DataFrame
        components_df = DataFrame({col: getattr(self._parent.array, f'_dt_{col}') for col in ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds']})
        return components_df

@delegate_names(delegate=DatetimeArray, accessors=DatetimeArray._datetimelike_ops + ['unit'], typ='property')
@delegate_names(delegate=DatetimeArray, accessors=DatetimeArray._datetimelike_methods + ['as_unit'], typ='method')
class DatetimeProperties(Properties):
    def to_pydatetime(self) -> Series:
        from pandas import Series
        return Series(self._get_values().to_pydatetime(), dtype=object)

    @property
    def freq(self) -> Optional[str]:
        return self._get_values().inferred_freq

    def isocalendar(self) -> DataFrame:
        return self._get_values().isocalendar().set_index(self._parent.index)

@delegate_names(delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ='property')
@delegate_names(delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_methods, typ='method')
class TimedeltaProperties(Properties):
    def to_pytimedelta(self) -> ndarray:
        warnings.warn(f'The behavior of {type(self).__name__}.to_pytimedelta is deprecated, in a future version this will return a Series containing python datetime.timedelta objects instead of an ndarray. To retain the old behavior, call `np.array` on the result', FutureWarning, stacklevel=find_stack_level())
        return self._get_values().to_pytimedelta()

    @property
    def components(self) -> DataFrame:
        return self._get_values().components.set_index(self._parent.index).__finalize__(self._parent)

    @property
    def freq(self) -> Optional[str]:
        return self._get_values().inferred_freq

@delegate_names(delegate=PeriodArray, accessors=PeriodArray._datetimelike_ops, typ='property')
@delegate_names(delegate=PeriodArray, accessors=PeriodArray._datetimelike_methods, typ='method')
class PeriodProperties(Properties):
    pass

class CombinedDatetimelikeProperties(DatetimeProperties, TimedeltaProperties, PeriodProperties):
    def __new__(cls, data: ABCSeries) -> Union[ArrowTemporalProperties, DatetimeProperties, TimedeltaProperties, PeriodProperties]:
        if not isinstance(data, ABCSeries):
            raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')
        orig = data if isinstance(data.dtype, CategoricalDtype) else None
        if orig is not None:
            data = data._constructor(orig.array, name=orig.name, copy=False, dtype=orig._values.categories.dtype, index=orig.index)
        if isinstance(data.dtype, ArrowDtype) and data.dtype.kind in 'Mm':
            return ArrowTemporalProperties(data, orig)
        if lib.is_np_dtype(data.dtype, 'M'):
            return DatetimeProperties(data, orig)
        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeProperties(data, orig)
        elif lib.is_np_dtype(data.dtype, 'm'):
            return TimedeltaProperties(data, orig)
        elif isinstance(data.dtype, PeriodDtype):
            return PeriodProperties(data, orig)
        raise AttributeError('Can only use .dt accessor with datetimelike values')
