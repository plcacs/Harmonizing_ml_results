from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import get_firstbday, get_lastbday
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp

def day_opt(request: pytest.FixtureRequest) -> str:
    return request.param

def test_get_last_bday(dt: datetime, exp_week_day: int, exp_last_day: int) -> None:
    assert dt.weekday() == exp_week_day
    assert get_lastbday(dt.year, dt.month) == exp_last_day

def test_get_first_bday(dt: datetime, exp_week_day: int, exp_first_day: int) -> None:
    assert dt.weekday() == exp_week_day
    assert get_firstbday(dt.year, dt.month) == exp_first_day

def test_shift_month_dt(months: int, day_opt: int, expected: datetime) -> None:
    dt = datetime(2017, 11, 30)
    assert liboffsets.shift_month(dt, months, day_opt=day_opt) == expected

def test_shift_month_ts(months: int, day_opt: str, expected: Timestamp) -> None:
    ts = Timestamp('1929-05-05')
    assert liboffsets.shift_month(ts, months, day_opt=day_opt) == expected

def test_shift_month_error() -> None:
    dt = datetime(2017, 11, 15)
    day_opt = 'this should raise'
    with pytest.raises(ValueError, match=day_opt):
        liboffsets.shift_month(dt, 3, day_opt=day_opt)

def test_roll_qtrday_year(other: datetime, expected: dict, n: int) -> None:
    month = 3
    day_opt = 'start'
    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]

def test_roll_qtrday_year2(other: datetime, expected: dict, n: int) -> None:
    month = 6
    day_opt = 'end'
    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]

def test_get_day_of_month_error() -> None:
    dt = datetime(2017, 11, 15)
    day_opt = 'foo'
    with pytest.raises(ValueError, match=day_opt):
        roll_qtrday(dt, n=3, month=11, day_opt=day_opt, modby=12)

def test_roll_qtr_day_not_mod_unequal(day_opt: str, month: int, n: int) -> None:
    expected = {3: {-3: -2, 4: 4}, 5: {-3: -3, 4: 3}}
    other = Timestamp(2072, 10, 1, 6, 17, 18)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected[month][n]

def test_roll_qtr_day_mod_equal(other: datetime, month: int, exp_dict: dict, n: int, day_opt: str) -> None:
    expected = exp_dict.get(n, {}).get(day_opt, n)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected

def test_roll_convention(n: int, expected: dict, compare: int) -> None:
    assert liboffsets.roll_convention(29, n, compare) == expected[compare]
