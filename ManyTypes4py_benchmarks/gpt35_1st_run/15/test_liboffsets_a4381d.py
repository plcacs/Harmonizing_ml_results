from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import get_firstbday, get_lastbday
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp

def day_opt(request: pytest.FixtureRequest) -> str:
    return request.param

def test_get_last_bday(dt: datetime, exp_week_day: int, exp_last_day: int) -> None:
def test_get_first_bday(dt: datetime, exp_week_day: int, exp_first_day: int) -> None:
def test_shift_month_dt(months: int, day_opt: str, expected: datetime) -> None:
def test_shift_month_ts(months: int, day_opt: str, expected: Timestamp) -> None:
def test_shift_month_error() -> None:
def test_roll_qtrday_year(other: datetime, expected: dict, n: int) -> None:
def test_roll_qtrday_year2(other: datetime, expected: dict, n: int) -> None:
def test_get_day_of_month_error() -> None:
def test_roll_qtr_day_not_mod_unequal(day_opt: str, month: int, n: int) -> None:
def test_roll_qtr_day_mod_equal(other: datetime, month: int, exp_dict: dict, n: int, day_opt: str) -> None:
def test_roll_convention(n: int, expected: dict, compare: int) -> None:
