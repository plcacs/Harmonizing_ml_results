import numpy as np
from typing import Any, ClassVar, List, Optional, Literal
from numpy.typing import NDArray
from pandas._libs.tslibs.fields import get_date_field, get_start_end_field, get_timedelta_field
from .tslib import _sizes


class TimeGetTimedeltaField:
    params: ClassVar[List[Any]] = [_sizes, ['seconds', 'microseconds', 'nanoseconds']]
    param_names: ClassVar[List[str]] = ['size', 'field']

    i8data: NDArray[np.int64]
    i8data_negative: NDArray[np.int64]

    def setup(self, size: int, field: Literal['seconds', 'microseconds', 'nanoseconds']) -> None:
        arr: NDArray[np.int64] = np.random.randint(0, 10, size=size, dtype='i8')
        self.i8data = arr
        arr = np.random.randint(-86400 * 1000000000, 0, size=size, dtype='i8')
        self.i8data_negative = arr

    def time_get_timedelta_field(self, size: int, field: Literal['seconds', 'microseconds', 'nanoseconds']) -> None:
        get_timedelta_field(self.i8data, field)

    def time_get_timedelta_field_negative_td(self, size: int, field: Literal['seconds', 'microseconds', 'nanoseconds']) -> None:
        get_timedelta_field(self.i8data_negative, field)


class TimeGetDateField:
    params: ClassVar[List[Any]] = [_sizes, ['Y', 'M', 'D', 'h', 'm', 's', 'us', 'ns', 'doy', 'dow', 'woy', 'q', 'dim', 'is_leap_year']]
    param_names: ClassVar[List[str]] = ['size', 'field']

    i8data: NDArray[np.int64]

    def setup(self, size: int, field: Literal['Y', 'M', 'D', 'h', 'm', 's', 'us', 'ns', 'doy', 'dow', 'woy', 'q', 'dim', 'is_leap_year']) -> None:
        arr: NDArray[np.int64] = np.random.randint(0, 10, size=size, dtype='i8')
        self.i8data = arr

    def time_get_date_field(self, size: int, field: Literal['Y', 'M', 'D', 'h', 'm', 's', 'us', 'ns', 'doy', 'dow', 'woy', 'q', 'dim', 'is_leap_year']) -> None:
        get_date_field(self.i8data, field)


class TimeGetStartEndField:
    params: ClassVar[List[Any]] = [_sizes, ['start', 'end'], ['month', 'quarter', 'year'], ['B', None, 'QS'], [12, 3, 5]]
    param_names: ClassVar[List[str]] = ['size', 'side', 'period', 'freqstr', 'month_kw']

    i8data: NDArray[np.int64]
    attrname: str

    def setup(
        self,
        size: int,
        side: Literal['start', 'end'],
        period: Literal['month', 'quarter', 'year'],
        freqstr: Optional[Literal['B', 'QS']],
        month_kw: int
    ) -> None:
        arr: NDArray[np.int64] = np.random.randint(0, 10, size=size, dtype='i8')
        self.i8data = arr
        self.attrname = f'is_{period}_{side}'

    def time_get_start_end_field(
        self,
        size: int,
        side: Literal['start', 'end'],
        period: Literal['month', 'quarter', 'year'],
        freqstr: Optional[Literal['B', 'QS']],
        month_kw: int
    ) -> None:
        get_start_end_field(self.i8data, self.attrname, freqstr, month_kw=month_kw)


from ..pandas_vb_common import setup