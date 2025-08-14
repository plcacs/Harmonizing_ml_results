import numpy as np
from typing import Optional

from pandas._libs.tslibs.fields import (
    get_date_field,
    get_start_end_field,
    get_timedelta_field,
)

from .tslib import _sizes


class TimeGetTimedeltaField:
    params = [
        _sizes,
        ["seconds", "microseconds", "nanoseconds"],
    ]
    param_names = ["size", "field"]

    def setup(self, size: int, field: str) -> None:
        arr: np.ndarray = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data: np.ndarray = arr
        arr = np.random.randint(-86400 * 1_000_000_000, 0, size=size, dtype="i8")
        self.i8data_negative: np.ndarray = arr

    def time_get_timedelta_field(self, size: int, field: str) -> None:
        get_timedelta_field(self.i8data, field)

    def time_get_timedelta_field_negative_td(self, size: int, field: str) -> None:
        get_timedelta_field(self.i8data_negative, field)


class TimeGetDateField:
    params = [
        _sizes,
        [
            "Y",
            "M",
            "D",
            "h",
            "m",
            "s",
            "us",
            "ns",
            "doy",
            "dow",
            "woy",
            "q",
            "dim",
            "is_leap_year",
        ],
    ]
    param_names = ["size", "field"]

    def setup(self, size: int, field: str) -> None:
        arr: np.ndarray = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data: np.ndarray = arr

    def time_get_date_field(self, size: int, field: str) -> None:
        get_date_field(self.i8data, field)


class TimeGetStartEndField:
    params = [
        _sizes,
        ["start", "end"],
        ["month", "quarter", "year"],
        ["B", None, "QS"],
        [12, 3, 5],
    ]
    param_names = ["size", "side", "period", "freqstr", "month_kw"]

    def setup(self, size: int, side: str, period: str, freqstr: Optional[str], month_kw: int) -> None:
        arr: np.ndarray = np.random.randint(0, 10, size=size, dtype="i8")
        self.i8data: np.ndarray = arr
        self.attrname: str = f"is_{period}_{side}"

    def time_get_start_end_field(self, size: int, side: str, period: str, freqstr: Optional[str], month_kw: int) -> None:
        get_start_end_field(self.i8data, self.attrname, freqstr, month_kw=month_kw)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip