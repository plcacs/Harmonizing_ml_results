"""
Tests for helper functions in the cython tslibs.offsets
"""
from datetime import datetime
from typing import Any, Dict, Union, Tuple
import pytest
from pandas._libs.tslibs.ccalendar import get_firstbday, get_lastbday
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp

@pytest.fixture(params=['start', 'end', 'business_start', 'business_end'])
def day_opt(request: pytest.FixtureRequest) -> str:
    return request.param

@pytest.mark.parametrize('dt,exp_week_day,exp_last_day', [(datetime(2017, 11, 30), 3, 30), (datetime(1993, 10, 31), 6, 29)])
def test_get_last_bday(dt: datetime, exp_week_day: int, exp_last_day: int) -> None:
    assert dt.weekday() == exp_week_day
    assert get_lastbday(dt.year, dt.month) == exp_last_day

@pytest.mark.parametrize('dt,exp_week_day,exp_first_day', [(datetime(2017, 4, 1), 5, 3), (datetime(1993, 10, 1), 4, 1)])
def test_get_first_bday(dt: datetime, exp_week_day: int, exp_first_day: int) -> None:
    assert dt.weekday() == exp_week_day
    assert get_firstbday(dt.year, dt.month) == exp_first_day

@pytest.mark.parametrize('months,day_opt,expected', [(0, 15, datetime(2017, 11, 15)), (0, None, datetime(2017, 11, 30)), (1, 'start', datetime(2017, 12, 1)), (-145, 'end', datetime(2005, 10, 31)), (0, 'business_end', datetime(2017, 11, 30)), (0, 'business_start', datetime(2017, 11, 1))])
def test_shift_month_dt(months: int, day_opt: Union[int, str, None], expected: datetime) -> None:
    dt = datetime(2017, 11, 30)
    assert liboffsets.shift_month(dt, months, day_opt=day_opt) == expected

@pytest.mark.parametrize('months,day_opt,expected', [(1, 'start', Timestamp('1929-06-01')), (-3, 'end', Timestamp('1929-02-28')), (25, None, Timestamp('1931-06-5')), (-1, 31, Timestamp('1929-04-30'))])
def test_shift_month_ts(months: int, day_opt: Union[int, str, None], expected: Timestamp) -> None:
    ts = Timestamp('1929-05-05')
    assert liboffsets.shift_month(ts, months, day_opt=day_opt) == expected

def test_shift_month_error() -> None:
    dt = datetime(2017, 11, 15)
    day_opt = 'this should raise'
    with pytest.raises(ValueError, match=day_opt):
        liboffsets.shift_month(dt, 3, day_opt=day_opt)

@pytest.mark.parametrize('other,expected', [(datetime(2017, 2, 10), {2: 1, -7: -7, 0: 0}), (Timestamp('2014-03-15', tz='US/Eastern'), {2: 2, -7: -6, 0: 1})])
@pytest.mark.parametrize('n', [2, -7, 0])
def test_roll_qtrday_year(other: Union[datetime, Timestamp], expected: Dict[int, int], n: int) -> None:
    month = 3
    day_opt = 'start'
    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]

@pytest.mark.parametrize('other,expected', [(datetime(1999, 6, 29), {5: 4, -7: -7, 0: 0}), (Timestamp(2072, 8, 24, 6, 17, 18), {5: 5, -7: -6, 0: 1})])
@pytest.mark.parametrize('n', [5, -7, 0])
def test_roll_qtrday_year2(other: Union[datetime, Timestamp], expected: Dict[int, int], n: int) -> None:
    month = 6
    day_opt = 'end'
    assert roll_qtrday(other, n, month, day_opt, modby=12) == expected[n]

def test_get_day_of_month_error() -> None:
    dt = datetime(2017, 11, 15)
    day_opt = 'foo'
    with pytest.raises(ValueError, match=day_opt):
        roll_qtrday(dt, n=3, month=11, day_opt=day_opt, modby=12)

@pytest.mark.parametrize('month', [3, 5])
@pytest.mark.parametrize('n', [4, -3])
def test_roll_qtr_day_not_mod_unequal(day_opt: str, month: int, n: int) -> None:
    expected = {3: {-3: -2, 4: 4}, 5: {-3: -3, 4: 3}}
    other = Timestamp(2072, 10, 1, 6, 17, 18)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected[month][n]

@pytest.mark.parametrize('other,month,exp_dict', [(datetime(1999, 5, 31), 2, {-1: {'start': 0, 'business_start': 0}}), (Timestamp(2072, 10, 1, 6, 17, 18), 4, {2: {'end': 1, 'business_end': 1, 'business_start': 1}}), (Timestamp(2072, 10, 3, 6, 17, 18), 4, {2: {'end': 1, 'business_end': 1}, -1: {'start': 0}})])
@pytest.mark.parametrize('n', [2, -1])
def test_roll_qtr_day_mod_equal(other: Union[datetime, Timestamp], month: int, exp_dict: Dict[int, Dict[str, int]], n: int, day_opt: str) -> None:
    expected = exp_dict.get(n, {}).get(day_opt, n)
    assert roll_qtrday(other, n, month, day_opt, modby=3) == expected

@pytest.mark.parametrize('n,expected', [(42, {29: 42, 1: 42, 31: 41}), (-4, {29: -4, 1: -3, 31: -4})])
@pytest.mark.parametrize('compare', [29, 1, 31])
def test_roll_convention(n: int, expected: Dict[int, int], compare: int) -> None:
    assert liboffsets.roll_convention(29, n, compare) == expected[compare]
