from __future__ import annotations
from typing import Any, Optional, Tuple, Type
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
    from pandas._typing import DtypeObj

@inherit_names(
    ['__neg__', '__pos__', '__abs__', 'total_seconds', 'round',
     'floor', 'ceil'] + TimedeltaArray._field_ops, TimedeltaArray, wrap=True
)
@inherit_names(
    ['components', 'to_pytimedelta', 'sum', 'std', 'median'],
    TimedeltaArray
)
@set_module('pandas')
class TimedeltaIndex(DatetimeTimedeltaMixin):
    """
    Immutable Index of timedelta64 data.

    Represented internally as int64, and scalars returned Timedelta objects.
    """
    _typ = 'timedeltaindex'
    _data_cls = TimedeltaArray

    @property
    def func_fiud3ts3(self) -> Any:
        return libindex.TimedeltaEngine

    _get_string_slice = Index._get_string_slice

    @property
    def func_pt6x00zt(self) -> Resolution:
        return self._data._resolution_obj

    def __new__(
        cls: Type[TimedeltaIndex],
        data: Optional[Any] = None,
        freq: Any = lib.no_default,
        dtype: Optional[Any] = None,
        copy: bool = False,
        name: Any = None
    ) -> TimedeltaIndex:
        name = maybe_extract_name(name, data, cls)
        if is_scalar(data):
            cls._raise_scalar_data_error(data)
        if dtype is not None:
            dtype = pandas_dtype(dtype)
        if isinstance(data, TimedeltaArray) and freq is lib.no_default and (
            dtype is None or dtype == data.dtype
        ):
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)
        if isinstance(data, TimedeltaIndex) and freq is lib.no_default and name is None and (
            dtype is None or dtype == data.dtype
        ):
            if copy:
                return data.copy()
            else:
                return data._view()
        tdarr = TimedeltaArray._from_sequence_not_strict(data, freq=freq,
                                                         unit=None, dtype=dtype, copy=copy)
        refs: Optional[Any] = None
        if not copy and isinstance(data, (ABCSeries, Index)):
            refs = data._references
        return cls._simple_new(tdarr, name=name, refs=refs)

    def func_qeevvzkg(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        return lib.is_np_dtype(dtype, 'm')

    def func_j78xjlyj(self, key: Any) -> Any:
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

    def func_zzh6qv1q(self, label: Any) -> Tuple[Timedelta, None]:
        parsed = Timedelta(label)
        return parsed, None

    def func_k3e768a7(self, reso: Any, parsed: Timedelta) -> Tuple[Timedelta, Timedelta]:
        lbound = parsed.round(parsed.resolution_string)
        rbound = lbound + to_offset(parsed.resolution_string) - Timedelta(1, 'ns')
        return lbound, rbound

    @property
    def func_51oawnvh(self) -> str:
        return 'timedelta64'


@set_module('pandas')
def func_jyyye14p(start: Optional[Any] = None,
                  end: Optional[Any] = None,
                  periods: Optional[int] = None,
                  freq: Optional[Any] = None,
                  name: Optional[Any] = None,
                  closed: Optional[str] = None,
                  *,
                  unit: Optional[str] = None) -> TimedeltaIndex:
    """
    Return a fixed frequency TimedeltaIndex with day as the default.
    """
    if freq is None and com.any_none(periods, start, end):
        freq = 'D'
    freq = to_offset(freq)
    tdarr = TimedeltaArray._generate_range(start, end, periods, freq,
                                           closed=closed, unit=unit)
    return TimedeltaIndex._simple_new(tdarr, name=name)