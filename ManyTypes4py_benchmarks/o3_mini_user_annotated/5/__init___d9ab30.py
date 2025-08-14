from typing import Any, Tuple, Optional, Union, NoReturn
import time as _time
import math as _math
from org.transcrypt.stubs.browser import __envir__

def zfill(s: Any, c: int) -> str:
    s = str(s)
    if len(s) < c:
        # __pragma__('opov')
        return '0' * (c - len(s)) + s
        # __pragma__('noopov')
    else:
        return s

def rjust(s: Any, c: int) -> str:
    s = str(s)
    if len(s) < c:
        # __pragma__('opov')
        return ' ' * (c - len(s)) + s
        # __pragma__('noopov')
    else:
        return s

def _cmp(x: Any, y: Any) -> int:
    return 0 if x == y else 1 if x > y else -1

MINYEAR: int = 1
MAXYEAR: int = 9999
_MAXORDINAL: int = 3652059  # date.max.toordinal()

_DAYS_IN_MONTH: list[int] = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_DAYS_BEFORE_MONTH: list[int] = [-1]  # -1 is a placeholder for indexing purposes.
dbm: int = 0
for dim in _DAYS_IN_MONTH[1:]:
    _DAYS_BEFORE_MONTH.append(dbm)
    dbm += dim
del dbm, dim

def _is_leap(year: int) -> bool:
    """year -> True if leap year, else False."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def _days_before_year(year: int) -> int:
    y: int = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400

def _days_in_month(year: int, month: int) -> int:
    assert 1 <= month <= 12, month
    if month == 2 and _is_leap(year):
        return 29
    return _DAYS_IN_MONTH[month]

def _days_before_month(year: int, month: int) -> int:
    assert 1 <= month <= 12, 'month must be in 1..12'
    return _DAYS_BEFORE_MONTH[month] + (month > 2 and _is_leap(year))

def _ymd2ord(year: int, month: int, day: int) -> int:
    assert 1 <= month <= 12, 'month must be in 1..12'
    dim: int = _days_in_month(year, month)
    assert 1 <= day <= dim, ('day must be in 1..%d' % dim)
    return (_days_before_year(year) +
            _days_before_month(year, month) +
            day)

_DI400Y: int = _days_before_year(401)
_DI100Y: int = _days_before_year(101)
_DI4Y: int = _days_before_year(5)
assert _DI4Y == 4 * 365 + 1
assert _DI400Y == 4 * _DI100Y + 1
assert _DI100Y == 25 * _DI4Y - 1

def _ord2ymd(n: int) -> Tuple[int, int, int]:
    n -= 1
    n400, n = divmod(n, _DI400Y)
    year: int = n400 * 400 + 1
    n100, n = divmod(n, _DI100Y)
    n4, n = divmod(n, _DI4Y)
    n1, n = divmod(n, 365)
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        assert n == 0
        return year - 1, 12, 31
    leapyear: bool = n1 == 3 and (n4 != 24 or n100 == 3)
    assert leapyear == _is_leap(year)
    month: int = (n + 50) >> 5
    preceding: int = _DAYS_BEFORE_MONTH[month] + (month > 2 and leapyear)
    if preceding > n:
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + (month == 2 and leapyear)
    n -= preceding
    assert 0 <= n < _days_in_month(year, month)
    return year, month, n + 1

_MONTHNAMES: list[Optional[str]] = [None, "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DAYNAMES: list[Optional[str]] = [None, "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def _build_struct_time(y: int, m: int, d: int, hh: int, mm: int, ss: int, dstflag: int) -> Tuple[int, int, int, int, int, int, int, int, int]:
    wday: int = (_ymd2ord(y, m, d) + 6) % 7
    dnum: int = _days_before_month(y, m) + d
    return (y, m, d, hh, mm, ss, wday, dnum, dstflag)

def _format_time(hh: int, mm: int, ss: int, us: int) -> str:
    result: str = "{}:{}:{}".format(zfill(hh, 2), zfill(mm, 2), zfill(ss, 2))
    if us:
        result += ".{}".format(zfill(us, 6))
    return result

def _wrap_strftime(object: Any, format: str, timetuple: Tuple[int, ...]) -> str:
    freplace: Optional[str] = None
    zreplace: Optional[str] = None
    Zreplace: Optional[str] = None
    newformat_parts: list[str] = []
    i: int = 0
    n: int = len(format)
    while i < n:
        ch: str = format[i]
        i += 1
        if ch == '%':
            if i < n:
                ch = format[i]
                i += 1
                if ch == 'f':
                    if freplace is None:
                        freplace = '{}'.format(zfill(getattr(object, 'microsecond', 0), 6))
                    newformat_parts.append(freplace)
                elif ch == 'z':
                    if zreplace is None:
                        zreplace = ""
                        if hasattr(object, "utcoffset"):
                            offset = object.utcoffset()
                            if offset is not None:
                                sign: str = '+'
                                if offset.days < 0:
                                    offset = -offset
                                    sign = '-'
                                h, m = divmod(offset, timedelta(minutes=1))
                                assert not m % timedelta(minutes=1), "whole minute"
                                m //= timedelta(minutes=1)
                                zreplace = '{}{}{}'.format(sign, zfill(h, 2), zfill(m, 2))
                    newformat_parts.append(zreplace)
                elif ch == 'Z':
                    if Zreplace is None:
                        Zreplace = ""
                        if hasattr(object, "tzname"):
                            s = object.tzname()
                            if s is not None:
                                Zreplace = s.replace('%', '%%')
                    newformat_parts.append(Zreplace)
                else:
                    newformat_parts.append('%')
                    newformat_parts.append(ch)
            else:
                newformat_parts.append('%')
        else:
            newformat_parts.append(ch)
    newformat: str = "".join(newformat_parts)
    return _time.strftime(newformat, timetuple)

def _check_tzname(name: Optional[Any]) -> None:
    if name is not None and not isinstance(name, str):
        raise TypeError("tzinfo.tzname() must return None or string, "
                        "not '{}'".format(type(name)))

def _check_utc_offset(name: str, offset: Any) -> None:
    assert name in ("utcoffset", "dst")
    if offset is None:
        return
    from datetime import timedelta  # for type checking in this function
    if not isinstance(offset, timedelta):
        raise TypeError("tzinfo.{}() must return None "
                        "or timedelta, not '{}'".format(name, type(offset)))
    if offset.__mod__(timedelta(minutes=1)).microseconds or offset.microseconds:
        raise ValueError("tzinfo.{}() must return a whole number "
                         "of minutes, got {}".format(name, offset))
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError("{}()={}, must be must be strictly between "
                         "-timedelta(hours=24) and timedelta(hours=24)".format(name, offset))

def _check_int_field(value: Any) -> int:
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

def _check_date_fields(year: Any, month: Any, day: Any) -> Tuple[int, int, int]:
    year_int: int = _check_int_field(year)
    month_int: int = _check_int_field(month)
    day_int: int = _check_int_field(day)
    if not MINYEAR <= year_int <= MAXYEAR:
        raise ValueError('year must be in {}..{}'.format(MINYEAR, MAXYEAR), year_int)
    if not 1 <= month_int <= 12:
        raise ValueError('month must be in 1..12', month_int)
    dim: int = _days_in_month(year_int, month_int)
    if not 1 <= day_int <= dim:
        raise ValueError('day must be in 1..{}'.format(dim), day_int)
    return year_int, month_int, day_int

def _check_time_fields(hour: Any, minute: Any, second: Any, microsecond: Any) -> Tuple[int, int, int, int]:
    hour_int: int = _check_int_field(hour)
    minute_int: int = _check_int_field(minute)
    second_int: int = _check_int_field(second)
    microsecond_int: int = _check_int_field(microsecond)
    if not 0 <= hour_int <= 23:
        raise ValueError('hour must be in 0..23', hour_int)
    if not 0 <= minute_int <= 59:
        raise ValueError('minute must be in 0..59', minute_int)
    if not 0 <= second_int <= 59:
        raise ValueError('second must be in 0..59', second_int)
    if not 0 <= microsecond_int <= 999999:
        raise ValueError('microsecond must be in 0..999999', microsecond_int)
    return hour_int, minute_int, second_int, microsecond_int

def _check_tzinfo_arg(tz: Any) -> None:
    from datetime import tzinfo  # for type checking
    if tz is not None and not isinstance(tz, tzinfo):
        raise TypeError("tzinfo argument must be None or of a tzinfo subclass")

def _cmperror(x: Any, y: Any) -> NoReturn:
    raise TypeError("can't compare '{}' to '{}'".format(
        type(x).__name__, type(y).__name__))

def _divide_and_round(a: int, b: int) -> int:
    q, r = divmod(a, b)
    r *= 2
    greater_than_half: bool = r > b if b > 0 else r < b
    if greater_than_half or (r == b and q % 2 == 1):
        q += 1
    return q

class timedelta:
    def __init__(self, *, days: Union[int, float] = 0, seconds: Union[int, float] = 0, microseconds: Union[int, float] = 0,
                 milliseconds: Union[int, float] = 0, minutes: Union[int, float] = 0, hours: Union[int, float] = 0, weeks: Union[int, float] = 0) -> None:
        d: Union[int, float] = days
        s: Union[int, float] = seconds
        us: Union[int, float] = microseconds
        d += weeks * 7
        s += minutes * 60 + hours * 3600
        us += milliseconds * 1000
        if isinstance(days, float):
            dayfrac, days_val = _math.modf(days)
            daysecondsfrac, daysecondswhole = _math.modf(dayfrac * (24. * 3600.))
            assert daysecondswhole == int(daysecondswhole)
            s = int(daysecondswhole)
            assert days_val == int(days_val)
            d = int(days_val)
        else:
            daysecondsfrac = 0.0
            d = days
        if isinstance(seconds, float):
            secondsfrac, seconds_val = _math.modf(seconds)
            assert seconds_val == int(seconds_val)
            seconds_val = int(seconds_val)
            secondsfrac += daysecondsfrac
            s = seconds_val
        else:
            secondsfrac = daysecondsfrac
        days_from_seconds, s = divmod(s, 24 * 3600)
        d += days_from_seconds
        s = int(s)
        usdouble: float = secondsfrac * 1e6
        if isinstance(microseconds, float):
            microseconds_val = round(microseconds + usdouble)
            seconds_part, microseconds_val = divmod(microseconds_val, 1000000)
            days_from_sec, seconds_part = divmod(seconds_part, 24 * 3600)
            d += days_from_sec
            s += seconds_part
        else:
            microseconds_val = int(microseconds)
            seconds_part, microseconds_val = divmod(microseconds_val, 1000000)
            days_from_sec, seconds_part = divmod(seconds_part, 24 * 3600)
            d += days_from_sec
            s += seconds_part
            microseconds_val = round(microseconds_val + usdouble)
        seconds_extra, us_final = divmod(microseconds_val, 1000000)
        s += seconds_extra
        days_from_s, s = divmod(s, 24 * 3600)
        d += days_from_s
        if abs(d) > 999999999:
            raise OverflowError("timedelta # of days is too large: %d" % d)
        self._days: int = int(d)
        self._seconds: int = int(s)
        self._microseconds: int = int(us_final)

    def __repr__(self) -> str:
        if self._microseconds:
            return "datetime.timedelta(days={}, seconds={}, microseconds={})".format(
                                          self._days,
                                          self._seconds,
                                          self._microseconds)
        if self._seconds:
            return "datetime.timedelta(days={}, seconds={})".format(
                                      self._days,
                                      self._seconds)
        return "datetime.timedelta(days={})".format(self._days)

    def __str__(self) -> str:
        mm, ss = divmod(self._seconds, 60)
        hh, mm = divmod(mm, 60)
        s: str = "{}:{}:{}".format(hh, zfill(mm, 2), zfill(ss, 2))
        if self._days:
            def plural(n: int) -> Tuple[int, str]:
                return n, "s" if abs(n) != 1 else ""
            s = ("{} day{}, ".format(*plural(self._days))) + s
        if self._microseconds:
            s = s + ".{}".format(zfill(self._microseconds, 6))
        return s

    def total_seconds(self) -> float:
        return ((self.days * 86400 + self.seconds) * 10 ** 6 +
                self.microseconds) / 10 ** 6

    @property
    def days(self) -> int:
        return self._days

    @property
    def seconds(self) -> int:
        return self._seconds

    @property
    def microseconds(self) -> int:
        return self._microseconds

    def __add__(self, other: Any) -> Union[timedelta, Any]:
        if isinstance(other, timedelta):
            return timedelta(days=self._days + other._days,
                             seconds=self._seconds + other._seconds,
                             microseconds=self._microseconds + other._microseconds)
        return NotImplemented

    def __radd__(self, other: Any) -> Union[timedelta, Any]:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Union[timedelta, Any]:
        if isinstance(other, timedelta):
            return timedelta(days=self._days - other._days,
                             seconds=self._seconds - other._seconds,
                             microseconds=self._microseconds - other._microseconds)
        return NotImplemented

    def __rsub__(self, other: Any) -> Union[timedelta, Any]:
        if isinstance(other, timedelta):
            return -self + other
        return NotImplemented

    def __neg__(self) -> timedelta:
        return timedelta(days=-self._days,
                         seconds=-self._seconds,
                         microseconds=-self._microseconds)

    def __pos__(self) -> timedelta:
        return self

    def __abs__(self) -> timedelta:
        if self._days < 0:
            return -self
        else:
            return self

    def __mul__(self, other: Any) -> Union[timedelta, Any]:
        if isinstance(other, int):
            return timedelta(days=self._days * other,
                             seconds=self._seconds * other,
                             microseconds=self._microseconds * other)
        if isinstance(other, float):
            usec: int = self._to_microseconds()
            a, b = other.as_integer_ratio()
            return timedelta(microseconds=_divide_and_round(usec * a, b))
        return NotImplemented

    def __rmul__(self, other: Any) -> Union[timedelta, Any]:
        return self.__mul__(other)

    def _to_microseconds(self) -> int:
        return ((self._days * (24 * 3600) + self._seconds) * 1000000 +
                self._microseconds)

    def __floordiv__(self, other: Any) -> Union[timedelta, int, Any]:
        usec: int = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec // other._to_microseconds()
        if isinstance(other, int):
            return timedelta(microseconds=usec // other)
        return NotImplemented

    def __truediv__(self, other: Any) -> Union[timedelta, float, Any]:
        usec: int = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec / other._to_microseconds()
        if isinstance(other, int):
            return timedelta(microseconds=_divide_and_round(usec, other))
        if isinstance(other, float):
            a, b = other.as_integer_ratio()
            return timedelta(microseconds=_divide_and_round(b * usec, a))
        return NotImplemented

    def __mod__(self, other: Any) -> Union[timedelta, Any]:
        if isinstance(other, timedelta):
            r: int = self._to_microseconds() % other._to_microseconds()
            return timedelta(microseconds=r)
        return NotImplemented

    def __divmod__(self, other: Any) -> Union[Tuple[int, timedelta], Any]:
        if isinstance(other, timedelta):
            q, r = divmod(self._to_microseconds(), other._to_microseconds())
            return q, timedelta(microseconds=r)
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) == 0
        else:
            return False

    def __le__(self, other: Any) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) <= 0
        else:
            _cmperror(self, other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) < 0
        else:
            _cmperror(self, other)

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) >= 0
        else:
            _cmperror(self, other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) > 0
        else:
            _cmperror(self, other)

    def _cmp(self, other: timedelta) -> int:
        assert isinstance(other, timedelta)
        return _cmp(self._to_microseconds(), other._to_microseconds())

    def __bool__(self) -> bool:
        return (self._days != 0 or
                self._seconds != 0 or
                self._microseconds != 0)

_td_min: timedelta = timedelta(days=-999999999)
_td_max: timedelta = timedelta(days=999999999, hours=23, minutes=59, seconds=59,
                          microseconds=999999)
_td_resolution: timedelta = timedelta(microseconds=1)
# __pragma__ ('js', '{}', '''Object.defineProperty (timedelta, 'min', {get: function () {return _td_min;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (timedelta, 'max', {get: function () {return _td_max;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (timedelta, 'resolution', {get: function () {return _td_resolution;}})''')

class date:
    def __init__(self, year: int, month: int, day: int) -> None:
        year, month, day = _check_date_fields(year, month, day)
        self._year: int = year
        self._month: int = month
        self._day: int = day

    @classmethod
    def fromtimestamp(cls, t: float) -> date:
        y, m, d, hh, mm, ss, weekday, jday, dst = _time.localtime(t)
        return cls(y, m, d)

    @classmethod
    def today(cls) -> date:
        t: float = _time.time()
        return cls.fromtimestamp(t)

    @classmethod
    def fromordinal(cls, n: int) -> date:
        y, m, d = _ord2ymd(n)
        return cls(y, m, d)

    def __repr__(self) -> str:
        return "datetime.date({}, {}, {})".format(
                                      self._year,
                                      self._month,
                                      self._day)

    def ctime(self) -> str:
        weekday: int = self.toordinal() % 7 or 7
        return "{} {} {} 00:00:00 {}".format(
            _DAYNAMES[weekday],
            _MONTHNAMES[self._month],
            rjust(self._day, 2), zfill(self._year, 4))

    def strftime(self, fmt: str) -> str:
        return _wrap_strftime(self, fmt, self.timetuple())

    def __format__(self, fmt: str) -> str:
        if not isinstance(fmt, str):
            raise TypeError("must be str, not {}".format(type(fmt).__name__))
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    def isoformat(self) -> str:
        return "{}-{}-{}".format(zfill(self._year, 4), zfill(self._month, 2), zfill(self._day, 2))

    def __str__(self) -> str:
        return self.isoformat()

    @property
    def year(self) -> int:
        return self._year

    @property
    def month(self) -> int:
        return self._month

    @property
    def day(self) -> int:
        return self._day

    def timetuple(self) -> Tuple[int, int, int, int, int, int, int, int, int]:
        return _build_struct_time(self._year, self._month, self._day,
                                  0, 0, 0, -1)

    def toordinal(self) -> int:
        return _ymd2ord(self._year, self._month, self._day)

    def replace(self, year: Optional[int] = None, month: Optional[int] = None, day: Optional[int] = None) -> date:
        if year is None:
            year = self._year
        if month is None:
            month = self._month
        if day is None:
            day = self._day
        return date(year, month, day)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, date):
            return self._cmp(other) == 0
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, date):
            return self._cmp(other) <= 0
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, date):
            return self._cmp(other) < 0
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, date):
            return self._cmp(other) >= 0
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, date):
            return self._cmp(other) > 0
        return NotImplemented

    def _cmp(self, other: date) -> int:
        y, m, d = self._year, self._month, self._day
        y2, m2, d2 = other._year, other._month, other._day
        return _cmp('{}{}{}'.format(zfill(y, 4), zfill(m, 2), zfill(d, 2)),
                    '{}{}{}'.format(zfill(y2, 4), zfill(m2, 2), zfill(d2, 2)))

    def __add__(self, other: Any) -> Union[date, Any]:
        if isinstance(other, timedelta):
            o: int = self.toordinal() + other.days
            if 0 < o <= _MAXORDINAL:
                return date.fromordinal(o)
            raise OverflowError("result out of range")
        return NotImplemented

    def __radd__(self, other: Any) -> Union[date, Any]:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Union[timedelta, Any]:
        if isinstance(other, timedelta):
            return self + timedelta(days=-other.days)
        if isinstance(other, date):
            days1: int = self.toordinal()
            days2: int = other.toordinal()
            return timedelta(days=days1 - days2)
        return NotImplemented

    def weekday(self) -> int:
        return (self.toordinal() + 6) % 7

    def isoweekday(self) -> int:
        return self.toordinal() % 7 or 7

    def isocalendar(self) -> Tuple[int, int, int]:
        year: int = self._year
        week1monday: int = _isoweek1monday(year)
        today: int = _ymd2ord(self._year, self._month, self._day)
        week, day = divmod(today - week1monday, 7)
        if week < 0:
            year -= 1
            week1monday = _isoweek1monday(year)
            week, day = divmod(today - week1monday, 7)
        elif week >= 52:
            if today >= _isoweek1monday(year + 1):
                year += 1
                week = 0
        return year, week + 1, day + 1

    resolution: timedelta = timedelta(days=1)

_date_class = date
_d_min: date = date(1, 1, 1)
_d_max: date = date(9999, 12, 31)

class tzinfo:
    def tzname(self, dt: Optional[Any]) -> Optional[str]:
        raise NotImplementedError("tzinfo subclass must override tzname()")

    def utcoffset(self, dt: Optional[Any]) -> Optional['timedelta']:
        raise NotImplementedError("tzinfo subclass must override utcoffset()")

    def dst(self, dt: Optional[Any]) -> Optional['timedelta']:
        raise NotImplementedError("tzinfo subclass must override dst()")

    def fromutc(self, dt: Any) -> Any:
        if not isinstance(dt, datetime):
            raise TypeError("fromutc() requires a datetime argument")
        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")
        dtoff = dt.utcoffset()
        if dtoff is None:
            raise ValueError("fromutc() requires a non-None utcoffset() result")
        dtdst = dt.dst()
        if dtdst is None:
            raise ValueError("fromutc() requires a non-None dst() result")
        delta = dtoff - dtdst
        if delta:
            dt += delta
            dtdst = dt.dst()
            if dtdst is None:
                raise ValueError("fromutc(): dt.dst gave inconsistent results; cannot convert")
        return dt + dtdst

_tzinfo_class = tzinfo

class time:
    def __init__(self, *, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0, tzinfo: Optional[tzinfo] = None) -> None:
        hour, minute, second, microsecond = _check_time_fields(hour, minute, second, microsecond)
        _check_tzinfo_arg(tzinfo)
        self._hour: int = hour
        self._minute: int = minute
        self._second: int = second
        self._microsecond: int = microsecond
        self._tzinfo: Optional[tzinfo] = tzinfo

    @property
    def hour(self) -> int:
        return self._hour

    @property
    def minute(self) -> int:
        return self._minute

    @property
    def second(self) -> int:
        return self._second

    @property
    def microsecond(self) -> int:
        return self._microsecond

    @property
    def tzinfo(self) -> Optional[tzinfo]:
        return self._tzinfo

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=True) == 0
        else:
            return False

    def __le__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=False) <= 0
        else:
            _cmperror(self, other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=False) < 0
        else:
            _cmperror(self, other)

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=False) >= 0
        else:
            _cmperror(self, other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=False) > 0
        else:
            _cmperror(self, other)

    def _cmp(self, other: time, allow_mixed: bool = False) -> int:
        mytz: Optional[tzinfo] = self._tzinfo
        ottz: Optional[tzinfo] = other._tzinfo
        myoff: Optional[timedelta] = None
        otoff: Optional[timedelta] = None
        if mytz is ottz:
            base_compare: bool = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            base_compare = myoff == otoff
        if base_compare:
            return _cmp((self._hour, self._minute, self._second, self._microsecond),
                        (other._hour, other._minute, other._second, other._microsecond))
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2
            else:
                raise TypeError("cannot compare naive and aware times")
        myhhmm: int = self._hour * 60 + self._minute - myoff // timedelta(minutes=1)
        othhmm: int = other._hour * 60 + other._minute - otoff // timedelta(minutes=1)
        return _cmp((myhhmm, self._second, self._microsecond),
                    (othhmm, other._second, other._microsecond))

    def _tzstr(self, sep: str = ":") -> Optional[str]:
        off: Optional[timedelta] = self.utcoffset()
        if off is not None:
            if off.days < 0:
                sign: str = "-"
                off = -off
            else:
                sign = "+"
            hh, rem = divmod(off, timedelta(hours=1))
            assert not rem % timedelta(minutes=1), "whole minute"
            mm: int = rem // timedelta(minutes=1)
            off_str: str = "{}{}{}{}".format(sign, zfill(hh, 2), sep, zfill(mm, 2))
            return off_str
        return None

    def __repr__(self) -> str:
        if self._microsecond != 0:
            s: str = ", {}, {}".format(self._second, self._microsecond)
        elif self._second != 0:
            s = ", {}".format(self._second)
        else:
            s = ""
        s = "datetime.time({}, {}{})".format(self._hour, self._minute, s)
        if self._tzinfo is not None:
            s = s[:-1] + ", tzinfo={}".format(self._tzinfo.__repr__()) + ")"
        return s

    def isoformat(self) -> str:
        s: str = _format_time(self._hour, self._minute, self._second, self._microsecond)
        tz: Optional[str] = self._tzstr()
        if tz:
            s += tz
        return s

    def __str__(self) -> str:
        return self.isoformat()

    def strftime(self, fmt: str) -> str:
        timetuple: Tuple[int, ...] = (1900, 1, 1,
                     self._hour, self._minute, self._second,
                     0, 1, -1)
        return _wrap_strftime(self, fmt, timetuple)

    def __format__(self, fmt: str) -> str:
        if not isinstance(fmt, str):
            raise TypeError("must be str, not %s" % type(fmt).__name__)
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    def utcoffset(self) -> Optional[timedelta]:
        if self._tzinfo is None:
            return None
        offset: Optional[timedelta] = self._tzinfo.utcoffset(None)
        _check_utc_offset("utcoffset", offset)
        return offset

    def tzname(self) -> Optional[str]:
        if self._tzinfo is None:
            return None
        name: Optional[str] = self._tzinfo.tzname(None)
        _check_tzname(name)
        return name

    def dst(self) -> Optional[timedelta]:
        if self._tzinfo is None:
            return None
        offset: Optional[timedelta] = self._tzinfo.dst(None)
        _check_utc_offset("dst", offset)
        return offset

    def replace(self, *, hour: Optional[int] = None, minute: Optional[int] = None, second: Optional[int] = None, microsecond: Optional[int] = None,
                tzinfo: Any = True) -> time:
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        return time(hour=hour, minute=minute, second=second, microsecond=microsecond, tzinfo=tzinfo)

    resolution: timedelta = timedelta(microseconds=1)

_time_class = time
_tm_min: time = time(hour=0, minute=0, second=0)
_tm_max: time = time(hour=23, minute=59, second=59, microsecond=999999)

class datetime(date):
    def __init__(self, year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0,
                 microsecond: int = 0, tzinfo: Optional[tzinfo] = None) -> None:
        year, month, day = _check_date_fields(year, month, day)
        hour, minute, second, microsecond = _check_time_fields(hour, minute, second, microsecond)
        _check_tzinfo_arg(tzinfo)
        self._year: int = year
        self._month: int = month
        self._day: int = day
        self._hour: int = hour
        self._minute: int = minute
        self._second: int = second
        self._microsecond: int = microsecond
        self._tzinfo: Optional[tzinfo] = tzinfo

    @property
    def hour(self) -> int:
        return self._hour

    @property
    def minute(self) -> int:
        return self._minute

    @property
    def second(self) -> int:
        return self._second

    @property
    def microsecond(self) -> int:
        return self._microsecond

    @property
    def tzinfo(self) -> Optional[tzinfo]:
        return self._tzinfo

    @classmethod
    def _fromtimestamp(cls, t: float, utc: bool, tz: Optional[tzinfo]) -> datetime:
        frac, t_int = _math.modf(t)
        us: int = round(frac * 1e6)
        if us >= 1000000:
            t_int += 1
            us -= 1000000
        elif us < 0:
            t_int -= 1
            us += 1000000
        converter = _time.gmtime if utc else _time.localtime
        y, m, d, hh, mm, ss, weekday, jday, dst = converter(t_int)
        ss = min(ss, 59)
        return cls(y, m, d, hh, mm, ss, us, tz)

    @classmethod
    def fromtimestamp(cls, t: float, tz: Optional[tzinfo] = None) -> datetime:
        _check_tzinfo_arg(tz)
        result: datetime = cls._fromtimestamp(t, utc=(tz is not None), tz=tz)
        if tz is not None:
            result = tz.fromutc(result)
        return result

    @classmethod
    def utcfromtimestamp(cls, t: float) -> datetime:
        return cls._fromtimestamp(t, True, None)

    @classmethod
    def now(cls, tz: Optional[tzinfo] = None) -> datetime:
        t: float = _time.time()
        return cls.fromtimestamp(t, tz)

    @classmethod
    def utcnow(cls) -> datetime:
        t: float = _time.time()
        return cls.utcfromtimestamp(t)

    @classmethod
    def combine(cls, date_obj: date, time_obj: time) -> datetime:
        if not isinstance(date_obj, _date_class):
            raise TypeError("date argument must be a date instance")
        if not isinstance(time_obj, _time_class):
            raise TypeError("time argument must be a time instance")
        return cls(date_obj.year, date_obj.month, date_obj.day,
                   time_obj.hour, time_obj.minute, time_obj.second, time_obj.microsecond,
                   time_obj.tzinfo)

    def timetuple(self) -> Tuple[int, ...]:
        dst: int
        if self.dst() is None:
            dst = -1
        elif self.dst():
            dst = 1
        else:
            dst = 0
        return _build_struct_time(self.year, self.month, self.day,
                                  self.hour, self.minute, self.second,
                                  dst)

    def timestamp(self) -> float:
        if self._tzinfo is None:
            return _time.mktime((self.year, self.month, self.day,
                                 self.hour, self.minute, self.second,
                                 -1, -1, -1)) + self.microsecond / 1e6
        else:
            return (self - _EPOCH).total_seconds()

    def utctimetuple(self) -> Tuple[int, ...]:
        offset: Optional[timedelta] = self.utcoffset()
        if offset:
            self_minus_offset: datetime = self - offset
            y, m, d = self_minus_offset.year, self_minus_offset.month, self_minus_offset.day
            hh, mm, ss = self_minus_offset.hour, self_minus_offset.minute, self_minus_offset.second
        else:
            y, m, d = self.year, self.month, self.day
            hh, mm, ss = self.hour, self.minute, self.second
        return _build_struct_time(y, m, d, hh, mm, ss, 0)

    def date(self) -> date:
        return date(self._year, self._month, self._day)

    def time(self) -> time:
        return time(hour=self.hour, minute=self.minute, second=self.second, microsecond=self.microsecond)

    def timetz(self) -> time:
        return time(hour=self.hour, minute=self.minute, second=self.second, microsecond=self.microsecond,
                    tzinfo=self._tzinfo)

    def replace(self, *, year: Optional[int] = None, month: Optional[int] = None, day: Optional[int] = None, hour: Optional[int] = None,
                minute: Optional[int] = None, second: Optional[int] = None, microsecond: Optional[int] = None, tzinfo: Any = True) -> datetime:
        if year is None:
            year = self.year
        if month is None:
            month = self.month
        if day is None:
            day = self.day
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        return datetime(year, month, day, hour, minute, second, microsecond,
                        tzinfo)

    def astimezone(self, tz: Optional[tzinfo] = None) -> datetime:
        if tz is None:
            if self.tzinfo is None:
                raise ValueError("astimezone() requires an aware datetime")
            ts: int = (self - _EPOCH) // timedelta(seconds=1)  # type: ignore
            localtm = _time.localtime(ts)
            local = datetime(*(localtm[:6]))
            if len(localtm) > 9:
                gmtoff = localtm[10]
                zone = localtm[9]
                tz = timezone._create(timedelta(seconds=gmtoff), zone)
            else:
                delta = local - datetime(*(_time.gmtime(ts)[:6]))
                dst_flag = _time.daylight and localtm[8] > 0
                gmtoff = -(_time.altzone if dst_flag else _time.timezone)
                if delta == timedelta(seconds=gmtoff):
                    tz = timezone._create(delta, _time.tzname[dst_flag])
                else:
                    tz = timezone._create(delta)
        elif not isinstance(tz, tzinfo):
            raise TypeError("tz argument must be an instance of tzinfo")
        mytz: Optional[tzinfo] = self.tzinfo
        if mytz is None:
            raise ValueError("astimezone() requires an aware datetime")
        if tz is mytz:
            return self
        myoff: Optional[timedelta] = self.utcoffset()
        if myoff is None:
            raise ValueError("astimezone() requires an aware datetime")
        utc: datetime = (self - myoff).replace(tzinfo=tz)
        return tz.fromutc(utc)

    def ctime(self) -> str:
        weekday: int = self.toordinal() % 7 or 7
        return "{} {} {} {}:{}:{} {}".format(
            _DAYNAMES[weekday],
            _MONTHNAMES[self._month],
            zfill(self._day, 2),
            zfill(self._hour, 2), zfill(self._minute, 2), zfill(self._second, 2),
            zfill(self._year, 4))

    def isoformat(self, sep: str = 'T') -> str:
        s: str = ("{}-{}-{}{}".format(zfill(self._year, 4), zfill(self._month, 2), zfill(self._day, 2), sep) +
             _format_time(self._hour, self._minute, self._second, self._microsecond))
        off: Optional[timedelta] = self.utcoffset()
        if off is not None:
            if off.days < 0:
                sign: str = "-"
                off = -off
            else:
                sign = "+"
            hh, rem = divmod(off, timedelta(hours=1))
            assert not rem % timedelta(minutes=1), "whole minute"
            mm: int = rem // timedelta(minutes=1)
            s += "{}{}:{}".format(sign, zfill(hh, 2), zfill(mm, 2))
        return s

    def __repr__(self) -> str:
        L: list[int] = [self._year, self._month, self._day,
             self._hour, self._minute, self._second, self._microsecond]
        if L[-1] == 0:
            L.pop()
        if L and L[-1] == 0:
            L.pop()
        s: str = "datetime.datetime({})".format(", ".join(map(str, L)))
        if self._tzinfo is not None:
            s = s[:-1] + ", tzinfo={}".format(self._tzinfo.__repr__()) + ")"
        return s

    def __str__(self) -> str:
        return self.isoformat(sep=' ')

    @classmethod
    def strptime(cls, date_string: str, format: str) -> datetime:
        return cls(*(_time.strptime(date_string, format)[:6]))

    def utcoffset(self) -> Optional[timedelta]:
        if self._tzinfo is None:
            return None
        offset: Optional[timedelta] = self._tzinfo.utcoffset(self)
        _check_utc_offset("utcoffset", offset)
        return offset

    def tzname(self) -> Optional[str]:
        if self._tzinfo is None:
            return None
        name: Optional[str] = self._tzinfo.tzname(self)
        _check_tzname(name)
        return name

    def dst(self) -> Optional[timedelta]:
        if self._tzinfo is None:
            return None
        offset: Optional[timedelta] = self._tzinfo.dst(self)
        _check_utc_offset("dst", offset)
        return offset

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=True) == 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            return False

    def __le__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=False) <= 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=False) < 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=False) >= 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=False) > 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def _cmp(self, other: datetime, allow_mixed: bool = False) -> int:
        mytz: Optional[tzinfo] = self._tzinfo
        ottz: Optional[tzinfo] = other._tzinfo
        if mytz is ottz:
            s1: str = '{}{}{}{}{}{}{}'.format(
                zfill(self._year, 4), zfill(self._month, 2),
                zfill(self._day, 2), zfill(self._hour, 2),
                zfill(self._minute, 2), zfill(self._second, 2), zfill(self._microsecond, 6)
            )
            s2: str = '{}{}{}{}{}{}{}'.format(
                zfill(other._year, 4), zfill(other._month, 2),
                zfill(other._day, 2), zfill(other._hour, 2),
                zfill(other._minute, 2), zfill(other._second, 2), zfill(other._microsecond, 6)
            )
            return _cmp(s1, s2)
        myoff: Optional[timedelta] = self.utcoffset()
        otoff: Optional[timedelta] = other.utcoffset()
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2
            else:
                raise TypeError("cannot compare naive and aware datetimes")
        diff: timedelta = self - other  # offsets taken into account
        if diff.days < 0:
            return -1
        return 1 if diff else 0

    def __add__(self, other: Any) -> Union[datetime, Any]:
        if not isinstance(other, timedelta):
            return NotImplemented
        delta: timedelta = timedelta(days=self.toordinal(),
                          seconds=self._second + self._minute * 60 + self._hour * 3600,
                          microseconds=self._microsecond)
        delta += other
        hour, rem = divmod(delta.seconds, 3600)
        minute, second = divmod(rem, 60)
        if 0 < delta.days <= _MAXORDINAL:
            return datetime.combine(date.fromordinal(delta.days),
                                    time(hour=hour, minute=minute, second=second,
                                         microsecond=delta.microseconds,
                                         tzinfo=self._tzinfo))
        raise OverflowError("result out of range")

    def __radd__(self, other: Any) -> Union[datetime, Any]:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Union[timedelta, Any]:
        if not isinstance(other, datetime):
            if isinstance(other, timedelta):
                return self + -other
            return NotImplemented
        days1: int = self.toordinal()
        days2: int = other.toordinal()
        secs1: int = self._second + self._minute * 60 + self._hour * 3600
        secs2: int = other._second + other._minute * 60 + other._hour * 3600
        base: timedelta = timedelta(days=days1 - days2,
                         seconds=secs1 - secs2,
                         microseconds=self._microsecond - other._microsecond)
        if self._tzinfo is other._tzinfo:
            return base
        myoff: Optional[timedelta] = self.utcoffset()
        otoff: Optional[timedelta] = other.utcoffset()
        if myoff == otoff:
            return base
        if myoff is None or otoff is None:
            raise TypeError("cannot mix naive and timezone-aware time")
        return base + otoff - myoff

    resolution: timedelta = timedelta(microseconds=1)

_dt_min: datetime = datetime(1, 1, 1)
_dt_max: datetime = datetime(9999, 12, 31, 23, 59, 59, 999999)
# __pragma__ ('js', '{}', '''Object.defineProperty (datetime, 'min', {get: function () {return _dt_min;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (datetime, 'max', {get: function () {return _dt_max;}})''')

def _isoweek1monday(year: int) -> int:
    THURSDAY: int = 3
    firstday: int = _ymd2ord(year, 1, 1)
    firstweekday: int = (firstday + 6) % 7
    week1monday: int = firstday - firstweekday
    if firstweekday > THURSDAY:
        week1monday += 7
    return week1monday

_Omitted: str = '@#$^&$^'

class timezone(tzinfo):
    def __init__(self, offset: timedelta, name: Union[str, Any] = _Omitted) -> None:
        if not isinstance(offset, timedelta):
            raise TypeError("offset must be a timedelta")
        if name is _Omitted:
            if not offset:
                offset = self.utc  # type: ignore
            name = None
        elif not isinstance(name, str):
            raise TypeError("name must be a string")
        if not (self._minoffset <= offset <= self._maxoffset):
            raise ValueError("offset must be a timedelta strictly between -timedelta(hours=24) and timedelta(hours=24).")
        if (offset.microseconds != 0 or offset.seconds % 60 != 0):
            raise ValueError("offset must be a timedelta representing a whole number of minutes")
        self._offset: timedelta = offset
        self._name: Optional[str] = name

    @classmethod
    def _create(cls, offset: timedelta, name: Union[str, Any] = _Omitted) -> timezone:
        return cls(offset, name)

    def __eq__(self, other: Any) -> bool:
        if type(other) != timezone:
            return False
        return self._offset == other._offset

    def __repr__(self) -> str:
        if self is self.utc:
            return 'datetime.timezone.utc'
        if self._name is None:
            return "datetime.timezone({})".format(self._offset.__repr__())
        return "datetime.timezone({}, {})".format(self._offset.__repr__(), self._name.__repr__())

    def __str__(self) -> str:
        return self.tzname(None)  # type: ignore

    def utcoffset(self, dt: Optional[datetime]) -> timedelta:
        if isinstance(dt, datetime) or dt is None:
            return self._offset
        raise TypeError("utcoffset() argument must be a datetime instance or None")

    def tzname(self, dt: Optional[datetime]) -> Optional[str]:
        if isinstance(dt, datetime) or dt is None:
            if self._name is None:
                return self._name_from_offset(self._offset)
            return self._name
        raise TypeError("tzname() argument must be a datetime instance or None")

    def dst(self, dt: Optional[datetime]) -> None:
        if isinstance(dt, datetime) or dt is None:
            return None
        raise TypeError("dst() argument must be a datetime instance or None")

    def fromutc(self, dt: datetime) -> datetime:
        if isinstance(dt, datetime):
            if dt.tzinfo is not self:
                raise ValueError("fromutc: dt.tzinfo is not self")
            return dt + self._offset  # type: ignore
        raise TypeError("fromutc() argument must be a datetime instance or None")

    _maxoffset: timedelta = timedelta(hours=23, minutes=59)
    _minoffset: timedelta = -timedelta(hours=23, minutes=59)

    @staticmethod
    def _name_from_offset(delta: timedelta) -> str:
        if delta < timedelta(0):
            sign: str = '-'
            delta = -delta
        else:
            sign = '+'
        hours, rest = divmod(delta, timedelta(hours=1))
        minutes: int = rest // timedelta(minutes=1)
        return 'UTC{}{}:{}'.format(sign, zfill(hours, 2), zfill(minutes, 2))

_tz_utc: timezone = timezone._create(timedelta(0))
_tz_min: timezone = timezone._create(timezone._minoffset)
_tz_max: timezone = timezone._create(timezone._maxoffset)
# __pragma__ ('js', '{}', '''Object.defineProperty (timezone, 'utc', {get: function () {return _tz_utc;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (timezone, 'min', {get: function () {return _tz_min;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (timezone, 'max', {get: function () {return _tz_max;}})''')

_EPOCH: datetime = datetime(1970, 1, 1, tzinfo=timezone._create(timedelta(0)))