"""implement the TimedeltaIndex"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union, cast
from pandas._libs import index as libindex, lib
from pandas._libs.tslibs import Resolution, Timedelta, to_offset
from pandas.util._decorators import set_module
from pandas.core.dtypes.common import is_scalar, pandas_dtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.indexes.base import Index, maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
if TYPE_CHECKING:
    from pandas._libs import NaTType
    from pandas._typing import DtypeObj, Frequency, NpDtype

@inherit_names(['__neg__', '__pos__', '__abs__', 'total_seconds', 'round', 'floor', 'ceil'] + TimedeltaArray._field_ops, TimedeltaArray, wrap=True)
@inherit_names(['components', 'to_pytimedelta', 'sum', 'std', 'median'], TimedeltaArray)
@set_module('pandas')
class TimedeltaIndex(DatetimeTimedeltaMixin):
    _typ: str = 'timedeltaindex'
    _data_cls: type[TimedeltaArray] = TimedeltaArray

    @property
    def _engine_type(self) -> type[libindex.TimedeltaEngine]:
        return libindex.TimedeltaEngine
    _get_string_slice: Any = Index._get_string_slice

    @property
    def _resolution_obj(self) -> Resolution:
        return self._data._resolution_obj

    def __new__(
        cls,
        data: Optional[Union[Any, TimedeltaArray, TimedeltaIndex]] = None,
        freq: Union[Frequency, lib.NoDefault] = lib.no_default,
        dtype: Optional[Union[DtypeObj, NpDtype]] = None,
        copy: bool = False,
        name: Optional[Any] = None
    ) -> TimedeltaIndex:
        name = maybe_extract_name(name, data, cls)
        if is_scalar(data):
            cls._raise_scalar_data_error(data)
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if isinstance(data, TimedeltaArray) and freq is lib.no_default and (dtype is None or dtype == data.dtype):
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)
        if isinstance(data, TimedeltaIndex) and freq is lib.no_default and (name is None) and (dtype is None or dtype == data.dtype):
            if copy:
                return data.copy()
            else:
                return data._view()
        tdarr = TimedeltaArray._from_sequence_not_strict(data, freq=freq, unit=None, dtype=dtype, copy=copy)
        refs = None
        if not copy and isinstance(data, (ABCSeries, Index)):
            refs = data._references
        return cls._simple_new(tdarr, name=name, refs=refs)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        return lib.is_np_dtype(dtype, 'm')

    def get_loc(self, key: Any) -> Union[int, slice, Any]:
        """
        Get integer location for requested label

        Returns
        -------
        loc : int, slice, or ndarray[int]
        """
        self._check_indexing_error(key)
        try:
            key = self._data._validate_scalar(key, unbox=False)
        except TypeError as err:
            raise KeyError(key) from err
        return Index.get_loc(self, key)

    def _parse_with_reso(self, label: str) -> tuple[Timedelta, None]:
        parsed = Timedelta(label)
        return (parsed, None)

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: Timedelta) -> tuple[Timedelta, Timedelta]:
        lbound = parsed.round(parsed.resolution_string)
        rbound = lbound + to_offset(parsed.resolution_string) - Timedelta(1, 'ns')
        return (lbound, rbound)

    @property
    def inferred_type(self) -> str:
        return 'timedelta64'

@set_module('pandas')
def timedelta_range(
    start: Optional[Union[str, Timedelta]] = None,
    end: Optional[Union[str, Timedelta]] = None,
    periods: Optional[int] = None,
    freq: Optional[Union[str, Timedelta]] = None,
    name: Optional[str] = None,
    closed: Optional[str] = None,
    *,
    unit: Optional[str] = None
) -> TimedeltaIndex:
    if freq is None and com.any_none(periods, start, end):
        freq = 'D'
    freq = to_offset(freq)
    tdarr = TimedeltaArray._generate_range(start, end, periods, freq, closed=closed, unit=unit)
    return TimedeltaIndex._simple_new(tdarr, name=name)
