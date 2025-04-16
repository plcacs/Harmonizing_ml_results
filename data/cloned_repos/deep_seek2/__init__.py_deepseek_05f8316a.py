from typing import Optional, Tuple, Union, List, Any, Dict, overload

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
_MAXORDINAL: int = 3652059  # date.max.toordinal()

_DAYS_IN_MONTH: List[int] = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_DAYS_BEFORE_MONTH: List[int] = [-1]  # -1 is a placeholder for indexing purposes.
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
    assert 1 <= day <= dim, ('day must be in 1..%d' % dim)
    return (_days_before_year(year) + _days_before_month(year, month) + day)

_DI400Y: int = _days_before_year(401)  # number of days in 400 years
_DI100Y: int = _days_before_year(101)  # "    "   "   " 100   "
_DI4Y: int = _days_before_year(5)  # "    "   "   "   4   "

assert _DI4Y == 4 * 365 + 1
assert _DI400Y == 4 * _DI100Y + 1
assert _DI100Y == 25 * _DI4Y - 1

def _ord2ymd(n: int) -> Tuple[int, int, int]:
    """ordinal -> (year, month, day), considering 01-Jan-0001 as day 1."""
    n -= 1
    n400, n = divmod(n, _DI400Y)
    year = n400 * 400 + 1  # ..., -399, 1, 401, ...
    n100, n = divmod(n, _DI100Y)
    n4, n = divmod(n, _DI4Y)
    n1, n = divmod(n, 365)
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        assert n == 0
        return year - 1, 12, 31
    leapyear = n1 == 3 and (n4 != 24 or n100 == 3)
    assert leapyear == _is_leap(year)
    month = (n + 50) >> 5
    preceding = _DAYS_BEFORE_MONTH[month] + (month > 2 and leapyear)
    if preceding > n:  # estimate is too large
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + (month == 2 and leapyear)
    n -= preceding
    assert 0 <= n < _days_in_month(year, month)
    return year, month, n + 1

_MONTHNAMES: List[Optional[str]] = [None, "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DAYNAMES: List[Optional[str]] = [None, "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def _build_struct_time(y: int, m: int, d: int, hh: int, mm: int, ss: int, dstflag: int) -> Tuple[int, int, int, int, int, int, int, int, int]:
    wday = (_ymd2ord(y, m, d) + 6) % 7
    dnum = _days_before_month(y, m) + d
    return (y, m, d, hh, mm, ss, wday, dnum, dstflag)

def _format_time(hh: int, mm: int, ss: int, us: int) -> str:
    result = "{}:{}:{}".format(zfill(hh, 2), zfill(mm, 2), zfill(ss, 2))
    if us:
        result += ".{}".format(zfill(us, 6))
    return result

def _wrap_strftime(object: Any, format: str, timetuple: Tuple[int, int, int, int, int, int, int, int, int]) -> str:
    freplace: Optional[str] = None
    zreplace: Optional[str] = None
    Zreplace: Optional[str] = None
    newformat: List[str] = []
    i, n = 0, len(format)
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
                        zreplace = ""
                        if hasattr(object, "utcoffset"):
                            offset = object.utcoffset()
                            if offset is not None:
                                sign = '+'
                                if offset.days < 0:
                                    offset = -offset
                                    sign = '-'
                                h, m = divmod(offset, timedelta(hours=1))
                                assert not m % timedelta(minutes=1), "whole minute"
                                m //= timedelta(minutes=1)
                                zreplace = '{}{}{}'.format(sign, zfill(h, 2), zfill(m, 2))
                    assert '%' not in zreplace
                    newformat.append(zreplace)
                elif ch == 'Z':
                    if Zreplace is None:
                        Zreplace = ""
                        if hasattr(object, "tzname"):
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
    newformat = "".join(newformat)
    return _time.strftime(newformat, timetuple)

def _check_tzname(name: Optional[str]) -> None:
    if name is not None and not isinstance(name, str):
        raise TypeError("tzinfo.tzname() must return None or string, not '{}'".format(type(name)))

def _check_utc_offset(name: str, offset: Optional[timedelta]) -> None:
    assert name in ("utcoffset", "dst")
    if offset is None:
        return
    if not isinstance(offset, timedelta):
        raise TypeError("tzinfo.{}() must return None or timedelta, not '{}'".format(name, type(offset)))
    if offset.__mod__(timedelta(minutes=1)).microseconds or offset.microseconds:
        raise ValueError("tzinfo.{}() must return a whole number of minutes, got {}".format(name, offset))
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError("{}()={}, must be must be strictly between -timedelta(hours=24) and timedelta(hours=24)".format(name, offset))

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
    return year, month, day

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
    return hour, minute, second, microsecond

def _check_tzinfo_arg(tz: Optional[Any]) -> None:
    if tz is not None and not isinstance(tz, tzinfo):
        raise TypeError("tzinfo argument must be None or of a tzinfo subclass")

def _cmperror(x: Any, y: Any) -> None:
    raise TypeError("can't compare '{}' to '{}'".format(type(x).__name__, type(y).__name__))

def _divide_and_round(a: int, b: int) -> int:
    """divide a by b and round result to the nearest integer"""
    q, r = divmod(a, b)
    r *= 2
    greater_than_half = r > b if b > 0 else r < b
    if greater_than_half or r == b and q % 2 == 1:
        q += 1
    return q

class timedelta:
    def __init__(self, days: int = 0, seconds: int = 0, microseconds: int = 0, milliseconds: int = 0, minutes: int = 0, hours: int = 0, weeks: int = 0) -> None:
        d = s = us = 0
        days += weeks * 7
        seconds += minutes * 60 + hours * 3600
        microseconds += milliseconds * 1000
        if isinstance(days, float):
            dayfrac, days = _math.modf(days)
            daysecondsfrac, daysecondswhole = _math.modf(dayfrac * (24. * 3600.))
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
        usdouble = secondsfrac * 1e6
        assert abs(usdouble) < 2.1e6
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
        assert abs(microseconds) < 3.1e6
        seconds, us = divmod(microseconds, 1000000)
        s += seconds
        days, s = divmod(s, 24 * 3600)
        d += days
        assert isinstance(d, int)
        assert isinstance(s, int) and 0 <= s < 24 * 3600
        assert isinstance(us, int) and 0 <= us < 1000000
        if abs(d) > 999999999:
            raise OverflowError("timedelta # of days is too large: %d" % d)
        self._days = d
        self._seconds = s
        self._microseconds = us

    def __repr__(self) -> str:
        if self._microseconds:
            return "datetime.timedelta(days={}, seconds={}, microseconds={})".format(self._days, self._seconds, self._microseconds)
        if self._seconds:
            return "datetime.timedelta(days={}, seconds={})".format(self._days, self._seconds)
        return "datetime.timedelta(days={})".format(self._days)

    def __str__(self) -> str:
        mm, ss = divmod(self._seconds, 60)
        hh, mm = divmod(mm, 60)
        s = "{}:{}:{}".format(hh, zfill(mm, 2), zfill(ss, 2))
        if self._days:
            def plural(n: int) -> Tuple[int, str]:
                return n, abs(n) != 1 and "s" or ""
            s = ("{} day{}, ".format(plural(self._days))) + s
        if self._microseconds:
            s = s + ".{}".format(zfill(self._microseconds, 6))
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
            return timedelta(self._days + other._days, self._seconds + other._seconds, self._microseconds + other._microseconds)
        return NotImplemented

    def __radd__(self, other: 'timedelta') -> 'timedelta':
        return self.__add__(other)

    def __sub__(self, other: 'timedelta') -> 'timedelta':
        if isinstance(other, timedelta):
            return timedelta(self._days - other._days, self._seconds - other._seconds, self._microseconds - other._microseconds)
        return NotImplemented

    def __rsub__(self, other: 'timedelta') -> 'timedelta':
        if isinstance(other, timedelta):
            return -self + other
        return NotImplemented

    def __neg__(self) -> 'timedelta':
        return timedelta(-self._days, -self._seconds, -self._microseconds)

    def __pos__(self) -> 'timedelta':
        return self

    def __abs__(self) -> 'timedelta':
        if self._days < 0:
            return -self
        else:
            return self

    def __mul__(self, other: Union[int, float]) -> 'timedelta':
        if isinstance(other, int):
            return timedelta(self._days * other, self._seconds * other, self._microseconds * other)
        if isinstance(other, float):
            usec = self._to_microseconds()
            a, b = other.as_integer_ratio()
            return timedelta(0, 0, _divide_and_round(usec * a, b))
        return NotImplemented

    def __rmul__(self, other: Union[int, float]) -> 'timedelta':
        return self.__mul__(other)

    def _to_microseconds(self) -> int:
        return ((self._days * (24 * 3600) + self._seconds) * 1000000 + self._microseconds)

    def __floordiv__(self, other: Union[int, 'timedelta']) -> Union[int, 'timedelta']:
        if not isinstance(other, (int, timedelta)):
            return NotImplemented
        usec = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec // other._to_microseconds()
        if isinstance(other, int):
            return timedelta(0, 0, usec // other)

    def __truediv__(self, other: Union[int, float, 'timedelta']) -> Union[float, 'timedelta']:
        if not isinstance(other, (int, float, timedelta)):
            return NotImplemented
        usec = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec / other._to_microseconds()
        if isinstance(other, int):
            return timedelta(0, 0, _divide_and_round(usec, other))
        if isinstance(other, float):
            a, b = other.as_integer_ratio()
            return timedelta(0, 0, _divide_and_round(b * usec, a))

    def __mod__(self, other: 'timedelta') -> 'timedelta':
        if isinstance(other, timedelta):
            r = self._to_microseconds() % other._to_microseconds()
            return timedelta(0, 0, r)
        return NotImplemented

    def __divmod__(self, other: 'timedelta') -> Tuple[int, 'timedelta']:
        if isinstance(other, timedelta):
            q, r = divmod(self._to_microseconds(), other._to_microseconds())
            return q, timedelta(0, 0, r)
        return NotImplemented

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) == 0
        else:
            return False

    def __le__(self, other: 'timedelta') -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) <= 0
        else:
            _cmperror(self, other)

    def __lt__(self, other: 'timedelta') -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) < 0
        else:
            _cmperror(self, other)

    def __ge__(self, other: 'timedelta') -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) >= 0
        else:
            _cmperror(self, other)

    def __gt__(self, other: 'timedelta') -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) > 0
        else:
            _cmperror(self, other)

    def _cmp(self, other: 'timedelta') -> int:
        assert isinstance(other, timedelta)
        return _cmp(self._to_microseconds(), other._to_microseconds())

    def __bool__(self) -> bool:
        return (self._days != 0 or self._seconds != 0 or self._microseconds != 0)

_td_min = timedelta(-999999999)
_td_max = timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999)
_td_resolution = timedelta(microseconds=1)

class date:
    def __init__(self, year: int, month: Optional[int] = None, day: Optional[int] = None) -> None:
        year, month, day = _check_date_fields(year, month, day)
        self._year = year
        self._month = month
        self._day = day

    @classmethod
    def fromtimestamp(cls, t: float) -> 'date':
        y, m, d, hh, mm, ss, weekday, jday, dst = _time.localtime(t)
        return cls(y, m, d)

    @classmethod
    def today(cls) -> 'date':
        t = _time.time()
        return cls.fromtimestamp(t)

    @classmethod
    def fromordinal(cls, n: int) -> 'date':
        y, m, d = _ord2ymd(n)
        return cls(y, m, d)

    def __repr__(self) -> str:
        return "datetime.date({}, {}, {})".format(self._year, self._month, self._day)

    def ctime(self) -> str:
        weekday = self.toordinal() % 7 or 7
        return "{} {} {} 00:00:00 {}".format(_DAYNAMES[weekday], _MONTHNAMES[self._month], rjust(self._day, 2), zfill(self._year, 4))

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
        return _build_struct_time(self._year, self._month, self._day, 0, 0, 0, -1)

    def toordinal(self) -> int:
        return _ymd2ord(self._year, self._month, self._day)

    def replace(self, year: Optional[int] = None, month: Optional[int] = None, day: Optional[int] = None) -> 'date':
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

    def __le__(self, other: 'date') -> bool:
        if isinstance(other, date):
            return self._cmp(other) <= 0
        return NotImplemented

    def __lt__(self, other: 'date') -> bool:
        if isinstance(other, date):
            return self._cmp(other) < 0
        return NotImplemented

    def __ge__(self, other: 'date') -> bool:
        if isinstance(other, date):
            return self._cmp(other) >= 0
        return NotImplemented

    def __gt__(self, other: 'date') -> bool:
        if isinstance(other, date):
            return self._cmp(other) > 0
        return NotImplemented

    def _cmp(self, other: 'date') -> int:
        assert isinstance(other, date)
        y, m, d = self._year, self._month, self._day
        y2, m2, d2 = other._year, other._month, other._day
        return _cmp('{}{}{}'.format(zfill(y, 4), zfill(m, 2), zfill(d, 2)), '{}{}{}'.format(zfill(y2, 4), zfill(m2, 2), zfill(d2, 2)))

    def __add__(self, other: 'timedelta') -> 'date':
        if isinstance(other, timedelta):
            o = self.toordinal() + other.days
            if 0 < o <= _MAXORDINAL:
                return date.fromordinal(o)
            raise OverflowError("result out of range")
        return NotImplemented

    def __radd__(self, other: 'timedelta') -> 'date':
        return self.__add__(other)

    def __sub__(self, other: Union['date', 'timedelta']) -> Union['timedelta', 'date']:
        if isinstance(other, timedelta):
            return self + timedelta(-other.days)
        if isinstance(other, date):
            days1 = self.toordinal()
            days2 = other.toordinal()
            return timedelta(days1 - days2)
        return NotImplemented

    def weekday(self) -> int:
        return (self.toordinal() + 6) % 7

    def isoweekday(self) -> int:
        return self.toordinal() % 7 or 7

    def isocalendar(self) -> Tuple[int, int, int]:
        year = self._year
        week1monday = _isoweek1monday(year)
        today = _ymd2ord(self._year, self._month, self._day)
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

    resolution = timedelta(days=1)

_date_class = date

_d_min = date(1, 1, 1)
_d_max = date(9999, 12, 31)

class tzinfo:
    def tzname(self, dt: Optional['datetime']) -> Optional[str]:
        raise NotImplementedError("tzinfo subclass must override tzname()")

    def utcoffset(self, dt: Optional['datetime']) -> Optional[timedelta]:
        raise NotImplementedError("tzinfo subclass must override utcoffset()")

    def dst(self, dt: Optional['datetime']) -> Optional[timedelta]:
        raise NotImplementedError("tzinfo subclass must override dst()")

    def fromutc(self, dt: 'datetime') -> 'datetime':
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
    def __init__(self, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0, tzinfo: Optional[tzinfo] = None) -> None:
        hour, minute, second, microsecond = _check_time_fields(hour, minute, second, microsecond)
        _check_tzinfo_arg(tzinfo)
        self._hour = hour
        self._minute = minute
        self._second = second
        self._microsecond = microsecond
        self._tzinfo = tzinfo

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

    def __le__(self, other: 'time') -> bool:
        if isinstance(other, time):
            return self._cmp(other) <= 0
        else:
            _cmperror(self, other)

    def __lt__(self, other: 'time') -> bool:
        if isinstance(other, time):
            return self._cmp(other) < 0
        else:
            _cmperror(self, other)

    def __ge__(self, other: 'time') -> bool:
        if isinstance(other, time):
            return self._cmp(other) >= 0
        else:
            _cmperror(self, other)

    def __gt__(self, other: 'time') -> bool:
        if isinstance(other, time):
            return self._cmp(other) > 0
        else:
            _cmperror(self, other)

    def _cmp(self, other: 'time', allow_mixed: bool = False) -> int:
        assert isinstance(other, time)
        mytz = self._tzinfo
        ottz = other._tzinfo
        myoff = otoff = None
        if mytz is ottz:
            base_compare = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            base_compare = myoff == otoff
        if base_compare:
            return _cmp((self._hour, self._minute, self._second, self._microsecond), (other._hour, other._minute, other._second, other._microsecond))
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2  # arbitrary non-zero value
            else:
                raise TypeError("cannot compare naive and aware times")
        myhhmm = self._hour * 60 + self._minute - myoff // timedelta(minutes=1)
        othhmm = other._hour * 60 + other._minute - otoff // timedelta(minutes=1)
        return _cmp((myhhmm, self._second, self._microsecond), (othhmm, other._second, other._microsecond))

    def _tzstr(self, sep: str = ":") -> Optional[str]:
        off = self.utcoffset()
        if off is not None:
            if off.days < 0:
                sign = "-"
                off = -off
            else:
                sign = "+"
            hh, mm = divmod(off, timedelta(hours=1))
            assert not mm % timedelta(minutes=1), "whole minute"
            mm //= timedelta(minutes=1)
            assert 0 <= hh < 24
            off = "{}{}{}{}".format(sign, zfill(hh, 2), sep, zfill(mm, 2))
        return off

    def __repr__(self) -> str:
        if self._microsecond != 0:
            s = ", {}, {}".format(self._second, self._microsecond)
        elif self._second != 0:
            s = ", {}".format(self._second)
        else:
            s = ""
        s = "datetime.time({}, {}{})".format(self._hour, self._minute, s)
        if self._tzinfo is not None:
            assert s[-1:] == ")"
            s = s[:len(s)-1] + ", tzinfo={}".format(self._tzinfo.__repr__()) + ")"
        return s

    def isoformat(self) -> str:
        s = _format_time(self._hour, self._minute, self._second, self._microsecond)
        tz = self._tzstr()
        if t