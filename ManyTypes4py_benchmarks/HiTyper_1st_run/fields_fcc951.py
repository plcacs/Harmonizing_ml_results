import numpy as np
from pandas._libs.tslibs.fields import get_date_field, get_start_end_field, get_timedelta_field
from .tslib import _sizes

class TimeGetTimedeltaField:
    params = [_sizes, ['seconds', 'microseconds', 'nanoseconds']]
    param_names = ['size', 'field']

    def setup(self, size: Union[int, float], field: Union[int, list[str], None, str]) -> None:
        arr = np.random.randint(0, 10, size=size, dtype='i8')
        self.i8data = arr
        arr = np.random.randint(-86400 * 1000000000, 0, size=size, dtype='i8')
        self.i8data_negative = arr

    def time_get_timedelta_field(self, size: Union[int, list[str], None, str], field: Union[int, T, str]) -> None:
        get_timedelta_field(self.i8data, field)

    def time_get_timedelta_field_negative_td(self, size: Union[int, list[str], None], field: Union[int, None, str, numpy.ndarray]) -> None:
        get_timedelta_field(self.i8data_negative, field)

class TimeGetDateField:
    params = [_sizes, ['Y', 'M', 'D', 'h', 'm', 's', 'us', 'ns', 'doy', 'dow', 'woy', 'q', 'dim', 'is_leap_year']]
    param_names = ['size', 'field']

    def setup(self, size: Union[int, float], field: Union[int, list[str], None, str]) -> None:
        arr = np.random.randint(0, 10, size=size, dtype='i8')
        self.i8data = arr

    def time_get_date_field(self, size: Union[int, typing.Sequence[str], float], field: Union[int, T, dict, None]) -> None:
        get_date_field(self.i8data, field)

class TimeGetStartEndField:
    params = [_sizes, ['start', 'end'], ['month', 'quarter', 'year'], ['B', None, 'QS'], [12, 3, 5]]
    param_names = ['size', 'side', 'period', 'freqstr', 'month_kw']

    def setup(self, size: Union[int, float], side, period, freqstr, month_kw) -> None:
        arr = np.random.randint(0, 10, size=size, dtype='i8')
        self.i8data = arr
        self.attrname = f'is_{period}_{side}'

    def time_get_start_end_field(self, size: Union[int, str], side: Union[int, str], period: Union[int, str], freqstr: Union[str, int, None], month_kw: Union[str, int, None]) -> None:
        get_start_end_field(self.i8data, self.attrname, freqstr, month_kw=month_kw)
from ..pandas_vb_common import setup