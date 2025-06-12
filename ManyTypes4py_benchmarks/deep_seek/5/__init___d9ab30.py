"""Concrete date/time and related types.

See http://www.iana.org/time-zones/repository/tz-link.html for
time zone and DST data sources.
"""
import time as _time
import math as _math
from typing import Optional, Tuple, Union, Any, List, Dict, Callable, TypeVar, overload
from org.transcrypt.stubs.browser import __envir__

T = TypeVar('T')

def zfill(s: Union[str, int], c: int) -> str:
    s = str(s)
    if len(s) < c:
        return '0' * (c - len(s)) + s
    else:
        return s

def rjust(s: Union[str, int], c: int) -> str:
    s = str(s)
    if len(s) < c:
        return ' ' * (c - len(s)) + s
    else:
        return s

def _cmp(x: Any, y: Any) -> int:
    return 0 if x == y else 1 if x > y else -1

MINYEAR: int = 1
MAXYEAR: int = 9999
_MAXORDINAL: int = 3652059
_DAYS_IN_MONTH: List[int] = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_BEFORE_MONTH: List[int] = [-1]
dbm: int = 0
for dim in _DAYS_IN_MONTH[1:]:
    _DAYS_BEFORE_MONTH.append(dbm)
    dbm += dim
del dbm, dim

def _is_leap(year: int) -> int:
    """year -> 1 if leap year, else 0."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def _days_before_year(year: int) -> int:
    """year -> number of days before January 1st of year."""
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400

def _days_in_month(year: int, month: int) -> int:
    """year, month -> number of days in that month in that year."""
    assert 1 <= month <= 12, month
    if month == 2 and _is_leap(year):
        return 29
    return _DAYS_IN_MONTH[month]

def _days_before_month(year: int, month: int) -> int:
    """year, month -> number of days in year preceding first day of month."""
    assert 1 <= month <= 12, 'month must be in 1..12'
    return _DAYS_BEFORE_MONTH[month] + (month > 2 and _is_leap(year))

def _ymd2ord(year: int, month: int, day: int) -> int:
    """year, month, day -> ordinal, considering 01-Jan-0001 as day 1."""
    assert 1 <= month <= 12, 'month must be in 1..12'
    dim = _days_in_month(year, month)
    assert 1 <= day <= dim, 'day must be in 1..%d' % dim
    return _days_before_year(year) + _days_before_month(year, month) + day

_DI400Y: int = _days_before_year(401)
_DI100Y: int = _days_before_year(101)
_DI4Y: int = _days_before_year(5)
assert _DI4Y == 4 * 365 + 1
assert _DI400Y == 4 * _DI100Y + 1
assert _DI100Y == 25 * _DI4Y - 1

def _ord2ymd(n: int) -> Tuple[int, int, int]:
    """ordinal -> (year, month, day), considering 01-Jan-0001 as day 1."""
    n -= 1
    n400, n = divmod(n, _DI400Y)
    year = n400 * 400 + 1
    n100, n = divmod(n, _DI100Y)
    n4, n = divmod(n, _DI4Y)
    n1, n = divmod(n, 365)
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        assert n == 0
        return (year - 1, 12, 31)
    leapyear = n1 == 3 and (n4 != 24 or n100 == 3)
    assert leapyear == _is_leap(year)
    month = n + 50 >> 5
    preceding = _DAYS_BEFORE_MONTH[month] + (month > 2 and leapyear)
    if preceding > n:
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + (month == 2 and leapyear)
    n -= preceding
    assert 0 <= n < _days_in_month(year, month)
    return (year, month, n + 1)

_MONTHNAMES: List[Optional[str]] = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_DAYNAMES: List[Optional[str]] = [None, 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def _build_struct_time(y: int, m: int, d: int, hh: int, mm: int, ss: int, dstflag: int) -> Tuple[int, int, int, int, int, int, int, int, int]:
    wday = (_ymd2ord(y, m, d) + 6) % 7
    dnum = _days_before_month(y, m) + d
    return (y, m, d, hh, mm, ss, wday, dnum, dstflag)

def _format_time(hh: int, mm: int, ss: int, us: int) -> str:
    result = '{}:{}:{}'.format(zfill(hh, 2), zfill(mm, 2), zfill(ss, 2))
    if us:
        result += '.{}'.format(zfill(us, 6))
    return result

def _wrap_strftime(object: Any, format: str, timetuple: Tuple[int, ...]) -> str:
    freplace: Optional[str] = None
    zreplace: Optional[str] = None
    Zreplace: Optional[str] = None
    newformat: List[str] = []
    i, n = (0, len(format))
    while i < n:
        ch = format[i]
        i += 1
        if ch == '%':
            if i < n:
                ch = format[i]
                i += 1
                if ch == 'f':
                    if freplace is None:
                        freplace = '{}'.format(zfill(getattr(object, 'microsecond', 0), 6))
                    newformat.append(freplace)
                elif ch == 'z':
                    if zreplace is None:
                        zreplace = ''
                        if hasattr(object, 'utcoffset'):
                            offset = object.utcoffset()
                            if offset is not None:
                                sign = '+'
                                if offset.days < 0:
                                    offset = -offset
                                    sign = '-'
                                h, m = divmod(offset, timedelta(hours=1))
                                assert not m % timedelta(minutes=1), 'whole minute'
                                m //= timedelta(minutes=1)
                                zreplace = '{}{}{}'.format(sign, zfill(h, 2), zfill(m, 2))
                    assert '%' not in zreplace
                    newformat.append(zreplace)
                elif ch == 'Z':
                    if Zreplace is None:
                        Zreplace = ''
                        if hasattr(object, 'tzname'):
                            s = object.tzname()
                            if s is not None:
                                Zreplace = s.replace('%', '%%')
                    newformat.append(Zreplace)
                else:
                    newformat.append('%')
                    newformat.append(ch)
            else:
                newformat.append('%')
        else:
            newformat.append(ch)
    newformat = ''.join(newformat)
    return _time.strftime(newformat, timetuple)

def _check_tzname(name: Optional[str]) -> None:
    if name is not None and (not isinstance(name, str)):
        raise TypeError("tzinfo.tzname() must return None or string, not '{}'".format(type(name)))

def _check_utc_offset(name: str, offset: Optional['timedelta']) -> None:
    assert name in ('utcoffset', 'dst')
    if offset is None:
        return
    if not isinstance(offset, timedelta):
        raise TypeError("tzinfo.{}() must return None or timedelta, not '{}'".format(name, type(offset)))
    if offset.__mod__(timedelta(minutes=1)).microseconds or offset.microseconds:
        raise ValueError('tzinfo.{}() must return a whole number of minutes, got {}'.format(name, offset))
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError('{}()={}, must be must be strictly between -timedelta(hours=24) and timedelta(hours=24)'.format(name, offset))

def _check_int_field(value: Union[int, float, Any]) -> int:
    _type = type(value)
    if _type == int:
        return value
    if not _type == float:
        try:
            value = value.__int__()
        except AttributeError:
            pass
        else:
            if type(value) == int:
                return value
            raise TypeError('__int__ returned non-int (type {})'.format(type(value).__name__))
        raise TypeError('an integer is required (got type {})'.format(type(value).__name__))
    raise TypeError('integer argument expected, got float')

def _check_date_fields(year: Union[int, float, Any], month: Union[int, float, Any], day: Union[int, float, Any]) -> Tuple[int, int, int]:
    year = _check_int_field(year)
    month = _check_int_field(month)
    day = _check_int_field(day)
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError('year must be in {}..{}'.format(MINYEAR, MAXYEAR), year)
    if not 1 <= month <= 12:
        raise ValueError('month must be in 1..12', month)
    dim = _days_in_month(year, month)
    if not 1 <= day <= dim:
        raise ValueError('day must be in 1..{}'.format(dim), day)
    return (year, month, day)

def _check_time_fields(hour: Union[int, float, Any], minute: Union[int, float, Any], second: Union[int, float, Any], microsecond: Union[int, float, Any]) -> Tuple[int, int, int, int]:
    hour = _check_int_field(hour)
    minute = _check_int_field(minute)
    second = _check_int_field(second)
    microsecond = _check_int_field(microsecond)
    if not 0 <= hour <= 23:
        raise ValueError('hour must be in 0..23', hour)
    if not 0 <= minute <= 59:
        raise ValueError('minute must be in 0..59', minute)
    if not 0 <= second <= 59:
        raise ValueError('second must be in 0..59', second)
    if not 0 <= microsecond <= 999999:
        raise ValueError('microsecond must be in 0..999999', microsecond)
    return (hour, minute, second, microsecond)

def _check_tzinfo_arg(tz: Optional['tzinfo']) -> None:
    if tz is not None and (not isinstance(tz, tzinfo)):
        raise TypeError('tzinfo argument must be None or of a tzinfo subclass')

def _cmperror(x: Any, y: Any) -> None:
    raise TypeError("can't compare '{}' to '{}'".format(type(x).__name__, type(y).__name__))

def _divide_and_round(a: int, b: int) -> int:
    """divide a by b and round result to the nearest integer

    When the ratio is exactly half-way between two integers,
    the even integer is returned.
    """
    q, r = divmod(a, b)
    r *= 2
    greater_than_half = r > b if b > 0 else r < b
    if greater_than_half or (r == b and q % 2 == 1):
        q += 1
    return q

class timedelta:
    """Represent the difference between two datetime objects."""
    
    def __init__(self, days: Union[int, float] = 0, seconds: Union[int, float] = 0, microseconds: Union[int, float] = 0, 
                 milliseconds: Union[int, float] = 0, minutes: Union[int, float] = 0, hours: Union[int, float] = 0, 
                 weeks: Union[int, float] = 0) -> None:
        d = s = us = 0
        days += weeks * 7
        seconds += minutes * 60 + hours * 3600
        microseconds += milliseconds * 1000
        if isinstance(days, float):
            dayfrac, days = _math.modf(days)
            daysecondsfrac, daysecondswhole = _math.modf(dayfrac * (24.0 * 3600.0))
            assert daysecondswhole == int(daysecondswhole)
            s = int(daysecondswhole)
            assert days == int(days)
            d = int(days)
        else:
            daysecondsfrac = 0.0
            d = days
        assert isinstance(daysecondsfrac, (float, int))
        assert abs(daysecondsfrac) <= 1.0
        assert isinstance(d, int)
        assert abs(s) <= 24 * 3600
        if isinstance(seconds, float):
            secondsfrac, seconds = _math.modf(seconds)
            assert seconds == int(seconds)
            seconds = int(seconds)
            secondsfrac += daysecondsfrac
            assert abs(secondsfrac) <= 2.0
        else:
            secondsfrac = daysecondsfrac
        assert isinstance(secondsfrac, (float, int))
        assert abs(secondsfrac) <= 2.0
        assert isinstance(seconds, int)
        days, seconds = divmod(seconds, 24 * 3600)
        d += days
        s += int(seconds)
        assert isinstance(s, int)
        assert abs(s) <= 2 * 24 * 3600
        usdouble = secondsfrac * 1000000.0
        assert abs(usdouble) < 2100000.0
        if isinstance(microseconds, float):
            microseconds = round(microseconds + usdouble)
            seconds, microseconds = divmod(microseconds, 1000000)
            days, seconds = divmod(seconds, 24 * 3600)
            d += days
            s += seconds
        else:
            microseconds = int(microseconds)
            seconds, microseconds = divmod(microseconds, 1000000)
            days, seconds = divmod(seconds, 24 * 3600)
            d += days
            s += seconds
            microseconds = round(microseconds + usdouble)
        assert isinstance(s, int)
        assert isinstance(microseconds, int)
        assert abs(s) <= 3 * 24 * 3600
        assert abs(microseconds) < 3100000.0
        seconds, us = divmod(microseconds, 1000000)
        s += seconds
        days, s = divmod(s, 24 * 3600)
        d += days
        assert isinstance(d, int)
        assert isinstance(s, int) and 0 <= s < 24 * 3600
        assert isinstance(us, int) and 0 <= us < 1000000
        if abs(d) > 999999999:
            raise OverflowError('timedelta # of days is too large: %d' % d)
        self._days = d
        self._seconds = s
        self._microseconds = us

    def __repr__(self) -> str:
        if self._microseconds:
            return 'datetime.timedelta(days={}, seconds={}, microseconds={})'.format(self._days, self._seconds, self._microseconds)
        if self._seconds:
            return 'datetime.timedelta(days={}, seconds={})'.format(self._days, self._seconds)
        return 'datetime.timedelta(days={})'.format(self._days)

    def __str__(self) -> str:
        mm, ss = divmod(self._seconds, 60)
        hh, mm = divmod(mm, 60)
        s = '{}:{}:{}'.format(hh, zfill(mm, 2), zfill(ss, 2))
        if self._days:
            def plural(n: int) -> Tuple[int, str]:
                return (n, abs(n) != 1 and 's' or '')
            s = '{} day{}, '.format(plural(self._days)) + s
        if self._microseconds:
            s = s + '.{}'.format(zfill(self._microseconds, 6))
        return s

    def total_seconds(self) -> float:
        """Total seconds in the duration."""
        return ((self.days * 86400 + self.seconds) * 10 ** 6 + self.microseconds) / 10 ** 6

    @property
    def days(self) -> int:
        """days"""
        return self._days

    @property
    def seconds(self) -> int:
        """seconds"""
        return self._seconds

    @property
    def microseconds(self) -> int:
        """microseconds"""
        return self._microseconds

    def __add__(self, other: 'timedelta') -> 'timedelta':
        if isinstance(other, timedelta):
            return timedelta(self._days + other._days, self._seconds + other