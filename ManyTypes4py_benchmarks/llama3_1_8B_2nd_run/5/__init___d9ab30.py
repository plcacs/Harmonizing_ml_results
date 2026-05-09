"""Concrete date/time and related types.

See http://www.iana.org/time-zones/repository/tz-link.html for
time zone and DST data sources.
"""
import time as _time
import math as _math
from org.transcrypt.stubs.browser import __envir__

def zfill(s: int, c: int) -> str:
    s = str(s)
    if len(s) < c:
        return '0' * (c - len(s)) + s
    else:
        return s

def rjust(s: int, c: int) -> str:
    s = str(s)
    if len(s) < c:
        return ' ' * (c - len(s)) + s
    else:
        return s

def _cmp(x: int, y: int) -> int:
    return 0 if x == y else 1 if x > y else -1

MINYEAR: int = 1
MAXYEAR: int = 9999
_MAXORDINAL: int = 3652059
_DAYS_IN_MONTH: list[int] = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_BEFORE_MONTH: list[int] = [-1]
dbm: int = 0
for dim in _DAYS_IN_MONTH[1:]:
    _DAYS_BEFORE_MONTH.append(dbm)
    dbm += dim
del dbm, dim

def _is_leap(year: int) -> bool:
    """year -> 1 if leap year, else 0."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def _days_before_year(year: int) -> int:
    """year -> number of days before January 1st of year."""
    y: int = year - 1
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
    dim: int = _days_in_month(year, month)
    assert 1 <= day <= dim, 'day must be in 1..%d' % dim
    return _days_before_year(year) + _days_before_month(year, month) + day

_DI400Y: int = _days_before_year(401)
_DI100Y: int = _days_before_year(101)
_DI4Y: int = _days_before_year(5)
assert _DI4Y == 4 * 365 + 1
assert _DI400Y == 4 * _DI100Y + 1
assert _DI100Y == 25 * _DI4Y - 1

def _ord2ymd(n: int) -> tuple[int, int, int]:
    """ordinal -> (year, month, day), considering 01-Jan-0001 as day 1."""
    n -= 1
    n400, n: tuple[int, int] = divmod(n, _DI400Y)
    year: int = n400 * 400 + 1
    n100, n: tuple[int, int] = divmod(n, _DI100Y)
    n4, n: tuple[int, int] = divmod(n, _DI4Y)
    n1, n: tuple[int, int] = divmod(n, 365)
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        assert n == 0
        return (year - 1, 12, 31)
    leapyear: bool = n1 == 3 and (n4 != 24 or n100 == 3)
    assert leapyear == _is_leap(year)
    month: int = n + 50 >> 5
    preceding: int = _DAYS_BEFORE_MONTH[month] + (month > 2 and leapyear)
    if preceding > n:
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + (month == 2 and leapyear)
    n -= preceding
    assert 0 <= n < _days_in_month(year, month)
    return (year, month, n + 1)

_MONTHNAMES: list[str] = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_DAYNAMES: list[str] = [None, 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def _build_struct_time(y: int, m: int, d: int, hh: int, mm: int, ss: int, dstflag: int) -> tuple[int, int, int, int, int, int, int, int, int]:
    wday: int = (_ymd2ord(y, m, d) + 6) % 7
    dnum: int = _days_before_month(y, m) + d
    return (y, m, d, hh, mm, ss, wday, dnum, dstflag)

def _format_time(hh: int, mm: int, ss: int, us: int) -> str:
    result: str = '{}:{}:{}'.format(zfill(hh, 2), zfill(mm, 2), zfill(ss, 2))
    if us:
        result += '.{}'.format(zfill(us, 6))
    return result

def _wrap_strftime(object, format: str, timetuple: tuple[int, int, int, int, int, int, int, int, int]) -> str:
    freplace: str = None
    zreplace: str = None
    Zreplace: str = None
    newformat: list[str] = []
    i, n: tuple[int, int] = (0, len(format))
    while i < n:
        ch: str = format[i]
        i += 1
        if ch == '%':
            if i < n:
                ch: str = format[i]
                i += 1
                if ch == 'f':
                    if freplace is None:
                        freplace = '{}'.format(zfill(getattr(object, 'microsecond', 0), 6))
                    newformat.append(freplace)
                elif ch == 'z':
                    if zreplace is None:
                        zreplace = ''
                        if hasattr(object, 'utcoffset'):
                            offset: timedelta = object.utcoffset()
                            if offset is not None:
                                sign: str = '+'
                                if offset.days < 0:
                                    offset = -offset
                                    sign = '-'
                                h, m: tuple[int, int] = divmod(offset, timedelta(hours=1))
                                assert not m % timedelta(minutes=1), 'whole minute'
                                m //= timedelta(minutes=1)
                                zreplace = '{}{}{}'.format(sign, zfill(h, 2), zfill(m, 2))
                    assert '%' not in zreplace
                    newformat.append(zreplace)
                elif ch == 'Z':
                    if Zreplace is None:
                        Zreplace = ''
                        if hasattr(object, 'tzname'):
                            s: str = object.tzname()
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
    newformat: str = ''.join(newformat)
    return _time.strftime(newformat, timetuple)

def _check_tzname(name: str) -> None:
    if name is not None and (not isinstance(name, str)):
        raise TypeError("tzinfo.tzname() must return None or string, not '{}'".format(type(name)))

def _check_utc_offset(name: str, offset: timedelta) -> None:
    assert name in ('utcoffset', 'dst')
    if offset is None:
        return
    if not isinstance(offset, timedelta):
        raise TypeError("tzinfo.{}() must return None or timedelta, not '{}'".format(name, type(offset)))
    if offset.__mod__(timedelta(minutes=1)).microseconds or offset.microseconds:
        raise ValueError('tzinfo.{}() must return a whole number of minutes, got {}'.format(name, offset))
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError('{}()={}, must be must be strictly between -timedelta(hours=24) and timedelta(hours=24)'.format(name, offset))

def _check_int_field(value: int) -> int:
    _type: type = type(value)
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

def _check_date_fields(year: int, month: int, day: int) -> tuple[int, int, int]:
    year: int = _check_int_field(year)
    month: int = _check_int_field(month)
    day: int = _check_int_field(day)
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError('year must be in {}..{}'.format(MINYEAR, MAXYEAR), year)
    if not 1 <= month <= 12:
        raise ValueError('month must be in 1..12', month)
    dim: int = _days_in_month(year, month)
    if not 1 <= day <= dim:
        raise ValueError('day must be in 1..{}'.format(dim), day)
    return (year, month, day)

def _check_time_fields(hour: int, minute: int, second: int, microsecond: int) -> tuple[int, int, int, int]:
    hour: int = _check_int_field(hour)
    minute: int = _check_int_field(minute)
    second: int = _check_int_field(second)
    microsecond: int = _check_int_field(microsecond)
    if not 0 <= hour <= 23:
        raise ValueError('hour must be in 0..23', hour)
    if not 0 <= minute <= 59:
        raise ValueError('minute must be in 0..59', minute)
    if not 0 <= second <= 59:
        raise ValueError('second must be in 0..59', second)
    if not 0 <= microsecond <= 999999:
        raise ValueError('microsecond must be in 0..999999', microsecond)
    return (hour, minute, second, microsecond)

def _check_tzinfo_arg(tz: tzinfo) -> None:
    if tz is not None and (not isinstance(tz, tzinfo)):
        raise TypeError('tzinfo argument must be None or of a tzinfo subclass')

def _cmperror(x: object, y: object) -> TypeError:
    raise TypeError("can't compare '{}' to '{}'".format(type(x).__name__, type(y).__name__))

def _divide_and_round(a: int, b: int) -> int:
    """divide a by b and round result to the nearest integer

    When the ratio is exactly half-way between two integers,
    the even integer is returned.
    """
    q, r: tuple[int, int] = divmod(a, b)
    r *= 2
    greater_than_half: bool = r > b if b > 0 else r < b
    if greater_than_half or (r == b and q % 2 == 1):
        q += 1
    return q

class timedelta:
    """Represent the difference between two datetime objects.

    Supported operators:

    - add, subtract timedelta
    - unary plus, minus, abs
    - compare to timedelta
    - multiply, divide by int

    In addition, datetime supports subtraction of two datetime objects
    returning a timedelta, and addition or subtraction of a datetime
    and a timedelta giving a datetime.

    Representation: (days, seconds, microseconds).  Why?  Because I
    felt like it.
    """

    def __init__(self, days: int = 0, seconds: int = 0, microseconds: int = 0, milliseconds: int = 0, minutes: int = 0, hours: int = 0, weeks: int = 0) -> None:
        d: int = s: int = us: int = 0
        days += weeks * 7
        seconds += minutes * 60 + hours * 3600
        microseconds += milliseconds * 1000
        if isinstance(days, float):
            dayfrac, days: tuple[float, int] = _math.modf(days)
            daysecondsfrac, daysecondswhole: tuple[float, int] = _math.modf(dayfrac * (24.0 * 3600.0))
            assert daysecondswhole == int(daysecondswhole)
            s = int(daysecondswhole)
            assert days == int(days)
            d = int(days)
        else:
            daysecondsfrac: float = 0.0
            d = days
        assert isinstance(daysecondsfrac, (float, int))
        assert abs(daysecondsfrac) <= 1.0
        assert isinstance(d, int)
        assert abs(s) <= 24 * 3600
        if isinstance(seconds, float):
            secondsfrac, seconds: tuple[float, int] = _math.modf(seconds)
            assert seconds == int(seconds)
            seconds = int(seconds)
            secondsfrac += daysecondsfrac
            assert abs(secondsfrac) <= 2.0
        else:
            secondsfrac: float = daysecondsfrac
        assert isinstance(secondsfrac, (float, int))
        assert abs(secondsfrac) <= 2.0
        assert isinstance(seconds, int)
        days, seconds: tuple[int, int] = divmod(seconds, 24 * 3600)
        d += days
        s += int(seconds)
        assert isinstance(s, int)
        assert abs(s) <= 2 * 24 * 3600
        usdouble: float = secondsfrac * 1000000.0
        assert abs(usdouble) < 2100000.0
        if isinstance(microseconds, float):
            microseconds: int = round(microseconds + usdouble)
            seconds, microseconds: tuple[int, int] = divmod(microseconds, 1000000)
            days, seconds: tuple[int, int] = divmod(seconds, 24 * 3600)
            d += days
            s += seconds
        else:
            microseconds: int = int(microseconds)
            seconds, microseconds: tuple[int, int] = divmod(microseconds, 1000000)
            days, seconds: tuple[int, int] = divmod(seconds, 24 * 3600)
            d += days
            s += seconds
            microseconds: int = round(microseconds + usdouble)
        assert isinstance(s, int)
        assert isinstance(microseconds, int)
        assert abs(s) <= 3 * 24 * 3600
        assert abs(microseconds) < 3100000.0
        seconds, us: tuple[int, int] = divmod(microseconds, 1000000)
        s += seconds
        days, s: tuple[int, int] = divmod(s, 24 * 3600)
        d += days
        assert isinstance(d, int)
        assert isinstance(s, int) and 0 <= s < 24 * 3600
        assert isinstance(us, int) and 0 <= us < 1000000
        if abs(d) > 999999999:
            raise OverflowError('timedelta # of days is too large: %d' % d)
        self._days: int = d
        self._seconds: int = s
        self._microseconds: int = us

    def __repr__(self) -> str:
        if self._microseconds:
            return 'datetime.timedelta(days={}, seconds={}, microseconds={})'.format(self._days, self._seconds, self._microseconds)
        if self._seconds:
            return 'datetime.timedelta(days={}, seconds={})'.format(self._days, self._seconds)
        return 'datetime.timedelta(days={})'.format(self._days)

    def __str__(self) -> str:
        mm, ss: tuple[int, int] = divmod(self._seconds, 60)
        hh, mm: tuple[int, int] = divmod(mm, 60)
        s: str = '{}:{}:{}'.format(hh, zfill(mm, 2), zfill(ss, 2))
        if self._days:

            def plural(n: int) -> tuple[int, str]:
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

    def __add__(self, other: timedelta) -> timedelta:
        if isinstance(other, timedelta):
            return timedelta(self._days + other._days, self._seconds + other._seconds, self._microseconds + other._microseconds)
        return NotImplemented

    def __radd__(self, other: timedelta) -> timedelta:
        return self.__add__(other)

    def __sub__(self, other: timedelta) -> timedelta:
        if isinstance(other, timedelta):
            return timedelta(self._days - other._days, self._seconds - other._seconds, self._microseconds - other._microseconds)
        return NotImplemented

    def __rsub__(self, other: timedelta) -> timedelta:
        if isinstance(other, timedelta):
            return -self + other
        return NotImplemented

    def __neg__(self) -> timedelta:
        return timedelta(-self._days, -self._seconds, -self._microseconds)

    def __pos__(self) -> timedelta:
        return self

    def __abs__(self) -> timedelta:
        if self._days < 0:
            return -self
        else:
            return self

    def __mul__(self, other: int) -> timedelta:
        if isinstance(other, int):
            return timedelta(self._days * other, self._seconds * other, self._microseconds * other)
        if isinstance(other, float):
            usec: int = self._to_microseconds()
            a, b: tuple[int, int] = other.as_integer_ratio()
            return timedelta(0, 0, _divide_and_round(usec * a, b))
        return NotImplemented

    def __rmul__(self, other: int) -> timedelta:
        return self.__mul__(other)

    def _to_microseconds(self) -> int:
        return (self._days * (24 * 3600) + self._seconds) * 1000000 + self._microseconds

    def __floordiv__(self, other: int) -> int:
        if not isinstance(other, (int, timedelta)):
            return NotImplemented
        usec: int = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec // other._to_microseconds()
        if isinstance(other, int):
            return timedelta(0, 0, usec // other)

    def __truediv__(self, other: int) -> float:
        if not isinstance(other, (int, float, timedelta)):
            return NotImplemented
        usec: int = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec / other._to_microseconds()
        if isinstance(other, int):
            return timedelta(0, 0, _divide_and_round(usec, other))
        if isinstance(other, float):
            a, b: tuple[int, int] = other.as_integer_ratio()
            return timedelta(0, 0, _divide_and_round(b * usec, a))

    def __mod__(self, other: timedelta) -> timedelta:
        if isinstance(other, timedelta):
            r: int = self._to_microseconds() % other._to_microseconds()
            return timedelta(0, 0, r)
        return NotImplemented

    def __divmod__(self, other: timedelta) -> tuple[int, timedelta]:
        if isinstance(other, timedelta):
            q, r: tuple[int, timedelta] = divmod(self._to_microseconds(), other._to_microseconds())
            return (q, timedelta(0, 0, r))
        return NotImplemented

    def __eq__(self, other: timedelta) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) == 0
        else:
            return False

    def __le__(self, other: timedelta) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) <= 0
        else:
            _cmperror(self, other)

    def __lt__(self, other: timedelta) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) < 0
        else:
            _cmperror(self, other)

    def __ge__(self, other: timedelta) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) >= 0
        else:
            _cmperror(self, other)

    def __gt__(self, other: timedelta) -> bool:
        if isinstance(other, timedelta):
            return self._cmp(other) > 0
        else:
            _cmperror(self, other)

    def _cmp(self, other: timedelta) -> int:
        assert isinstance(other, timedelta)
        return _cmp(self._to_microseconds(), other._to_microseconds())

    def __bool__(self) -> bool:
        return self._days != 0 or self._seconds != 0 or self._microseconds != 0
_td_min: timedelta = timedelta(-999999999)
_td_max: timedelta = timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999)
_td_resolution: timedelta = timedelta(microseconds=1)

class date:
    """Concrete date type.

    Constructors:

    __new__()
    fromtimestamp()
    today()
    fromordinal()

    Operators:

    __repr__, __str__
    __eq__, __le__, __lt__, __ge__, __gt__,
    __add__, __radd__, __sub__ (add/radd only with timedelta arg)

    Methods:

    timetuple()
    toordinal()
    weekday()
    isoweekday(), isocalendar(), isoformat()
    ctime()
    strftime()

    Properties (readonly):
    year, month, day
    """

    def __init__(self, year: int, month: int = None, day: int = None) -> None:
        """Constructor.

        Arguments:

        year, month, day (required, base 1)
        """
        year, month, day = _check_date_fields(year, month, day)
        self._year: int = year
        self._month: int = month
        self._day: int = day

    @classmethod
    def fromtimestamp(cls, t: float) -> 'date':
        """Construct a date from a POSIX timestamp (like time.time())."""
        y, m, d, hh, mm, ss, weekday, jday, dst = _time.localtime(t)
        return cls(y, m, d)

    @classmethod
    def today(cls) -> 'date':
        """Construct a date from time.time()."""
        t: float = _time.time()
        return cls.fromtimestamp(t)

    @classmethod
    def fromordinal(cls, n: int) -> 'date':
        """Contruct a date from a proleptic Gregorian ordinal.

        January 1 of year 1 is day 1.  Only the year, month and day are
        non-zero in the result.
        """
        y, m, d = _ord2ymd(n)
        return cls(y, m, d)

    def __repr__(self) -> str:
        """Convert to formal string, for repr().

        >>> dt = datetime(2010, 1, 1)
        >>> repr(dt)
        'datetime.datetime(2010, 1, 1, 0, 0)'

        >>> dt = datetime(2010, 1, 1, tzinfo=timezone.utc)
        >>> repr(dt)
        'datetime.datetime(2010, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)'
        """
        return 'datetime.date({}, {}, {})'.format(self._year, self._month, self._day)

    def ctime(self) -> str:
        """Return ctime() style string."""
        weekday: int = self.toordinal() % 7 or 7
        return '{} {} {} 00:00:00 {}'.format(_DAYNAMES[weekday], _MONTHNAMES[self._month], rjust(self._day, 2), zfill(self._year, 4))

    def strftime(self, fmt: str) -> str:
        """Format using strftime()."""
        return _wrap_strftime(self, fmt, self.timetuple())

    def __format__(self, fmt: str) -> str:
        if not isinstance(fmt, str):
            raise TypeError('must be str, not {}'.format(type(fmt).__name__))
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    def isoformat(self) -> str:
        """Return the date formatted according to ISO.

        This is 'YYYY-MM-DD'.

        References:
        - http://www.w3.org/TR/NOTE-datetime
        - http://www.cl.cam.ac.uk/~mgk25/iso-time.html
        """
        return '{}-{}-{}'.format(zfill(self._year, 4), zfill(self._month, 2), zfill(self._day, 2))

    def __str__(self) -> str:
        return self.isoformat()

    @property
    def year(self) -> int:
        """year (1-9999)"""
        return self._year

    @property
    def month(self) -> int:
        """month (1-12)"""
        return self._month

    @property
    def day(self) -> int:
        """day (1-31)"""
        return self._day

    def timetuple(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        """Return local time tuple compatible with time.localtime()."""
        return _build_struct_time(self._year, self._month, self._day, 0, 0, 0, -1)

    def toordinal(self) -> int:
        """Return proleptic Gregorian ordinal for the year, month and day.

        January 1 of year 1 is day 1.  Only the year, month and day values
        contribute to the result.
        """
        return _ymd2ord(self._year, self._month, self._day)

    def replace(self, year: int = None, month: int = None, day: int = None) -> 'date':
        """Return a new date with new values for the specified fields."""
        if year is None:
            year = self._year
        if month is None:
            month = self._month
        if day is None:
            day = self._day
        return date(year, month, day)

    def __eq__(self, other: date) -> bool:
        if isinstance(other, date):
            return self._cmp(other) == 0
        return NotImplemented

    def __le__(self, other: date) -> bool:
        if isinstance(other, date):
            return self._cmp(other) <= 0
        return NotImplemented

    def __lt__(self, other: date) -> bool:
        if isinstance(other, date):
            return self._cmp(other) < 0
        return NotImplemented

    def __ge__(self, other: date) -> bool:
        if isinstance(other, date):
            return self._cmp(other) >= 0
        return NotImplemented

    def __gt__(self, other: date) -> bool:
        if isinstance(other, date):
            return self._cmp(other) > 0
        return NotImplemented

    def _cmp(self, other: date) -> int:
        assert isinstance(other, date)
        y, m, d = (self._year, self._month, self._day)
        y2, m2, d2 = (other._year, other._month, other._day)
        return _cmp('{}{}{}'.format(zfill(y, 4), zfill(m, 2), zfill(d, 2)), '{}{}{}'.format(zfill(y2, 4), zfill(m2, 2), zfill(d2, 2)))

    def __add__(self, other: timedelta) -> 'date':
        """Add a date to a timedelta."""
        if isinstance(other, timedelta):
            o: int = self.toordinal() + other.days
            if 0 < o <= _MAXORDINAL:
                return date.fromordinal(o)
            raise OverflowError('result out of range')
        return NotImplemented

    def __radd__(self, other: timedelta) -> 'date':
        return self.__add__(other)

    def __sub__(self, other: date) -> timedelta:
        """Subtract two dates, or a date and a timedelta."""
        if isinstance(other, timedelta):
            return self + timedelta(-other.days)
        if isinstance(other, date):
            days1: int = self.toordinal()
            days2: int = other.toordinal()
            return timedelta(days1 - days2)
        return NotImplemented

    def weekday(self) -> int:
        """Return day of the week, where Monday == 0 ... Sunday == 6."""
        return (self.toordinal() + 6) % 7

    def isoweekday(self) -> int:
        """Return day of the week, where Monday == 1 ... Sunday == 7."""
        return self.toordinal() % 7 or 7

    def isocalendar(self) -> tuple[int, int, int]:
        """Return a 3-tuple containing ISO year, week number, and weekday.

        The first ISO week of the year is the (Mon-Sun) week
        containing the year's first Thursday; everything else derives
        from that.

        The first week is 1; Monday is 1 ... Sunday is 7.

        ISO calendar algorithm taken from
        http://www.phys.uu.nl/~vgent/calendar/isocalendar.htm
        (used with permission)
        """
        year: int = self._year
        week1monday: int = _isoweek1monday(year)
        today: int = _ymd2ord(self._year, self._month, self._day)
        week, day: tuple[int, int] = divmod(today - week1monday, 7)
        if week < 0:
            year -= 1
            week1monday = _isoweek1monday(year)
            week, day = divmod(today - week1monday, 7)
        elif week >= 52:
            if today >= _isoweek1monday(year + 1):
                year += 1
                week = 0
        return (year, week + 1, day + 1)
    resolution: timedelta = timedelta(days=1)
_date_class: type = date
_d_min: date = date(1, 1, 1)
_d_max: date = date(9999, 12, 31)

class tzinfo:
    """Abstract base class for time zone info classes.

    Subclasses must override the name(), utcoffset() and dst() methods.
    """

    def tzname(self, dt: datetime) -> str:
        """datetime -> string name of time zone."""
        raise NotImplementedError('tzinfo subclass must override tzname()')

    def utcoffset(self, dt: datetime) -> timedelta:
        """datetime -> minutes east of UTC (negative for west of UTC)"""
        raise NotImplementedError('tzinfo subclass must override utcoffset()')

    def dst(self, dt: datetime) -> timedelta:
        """datetime -> DST offset in minutes east of UTC.

        Return 0 if DST not in effect.  utcoffset() must include the DST
        offset.
        """
        raise NotImplementedError('tzinfo subclass must override dst()')

    def fromutc(self, dt: datetime) -> datetime:
        """datetime in UTC -> datetime in local time."""
        if not isinstance(dt, datetime):
            raise TypeError('fromutc() requires a datetime argument')
        if dt.tzinfo is not self:
            raise ValueError('dt.tzinfo is not self')
        dtoff: timedelta = dt.utcoffset()
        if dtoff is None:
            raise ValueError('fromutc() requires a non-None utoffset() result')
        dtdst: timedelta = dt.dst()
        if dtdst is None:
            raise ValueError('fromutc() requires a non-None dst() result')
        delta: timedelta = dtoff - dtdst
        if delta:
            dt += delta
            dtdst = dt.dst()
            if dtdst is None:
                raise ValueError('fromutc(): dt.dst gave inconsistent results; cannot convert')
        return dt + dtdst
_tzinfo_class: type = tzinfo

class time:
    """Time with time zone.

    Constructors:

    __new__()

    Operators:

    __repr__, __str__
    __eq__, __le__, __lt__, __ge__, __gt__,

    Methods:

    strftime()
    isoformat()
    utcoffset()
    tzname()
    dst()

    Properties (readonly):
    hour, minute, second, microsecond, tzinfo
    """

    def __init__(self, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0, tzinfo: tzinfo = None) -> None:
        """Constructor.

        Arguments:

        hour, minute (required)
        second, microsecond (default to zero)
        tzinfo (default to None)
        """
        hour, minute, second, microsecond = _check_time_fields(hour, minute, second, microsecond)
        _check_tzinfo_arg(tzinfo)
        self._hour: int = hour
        self._minute: int = minute
        self._second: int = second
        self._microsecond: int = microsecond
        self._tzinfo: tzinfo = tzinfo

    @property
    def hour(self) -> int:
        """hour (0-23)"""
        return self._hour

    @property
    def minute(self) -> int:
        """minute (0-59)"""
        return self._minute

    @property
    def second(self) -> int:
        """second (0-59)"""
        return self._second

    @property
    def microsecond(self) -> int:
        """microsecond (0-999999)"""
        return self._microsecond

    @property
    def tzinfo(self) -> tzinfo:
        """timezone info object"""
        return self._tzinfo

    def __eq__(self, other: time) -> bool:
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=True) == 0
        else:
            return False

    def __le__(self, other: time) -> bool:
        if isinstance(other, time):
            return self._cmp(other) <= 0
        else:
            _cmperror(self, other)

    def __lt__(self, other: time) -> bool:
        if isinstance(other, time):
            return self._cmp(other) < 0
        else:
            _cmperror(self, other)

    def __ge__(self, other: time) -> bool:
        if isinstance(other, time):
            return self._cmp(other) >= 0
        else:
            _cmperror(self, other)

    def __gt__(self, other: time) -> bool:
        if isinstance(other, time):
            return self._cmp(other) > 0
        else:
            _cmperror(self, other)

    def _cmp(self, other: time, allow_mixed: bool = False) -> int:
        assert isinstance(other, time)
        mytz: tzinfo = self._tzinfo
        ottz: tzinfo = other._tzinfo
        myoff: timedelta = otoff: timedelta = None
        if mytz is ottz:
            base_compare: bool = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            base_compare: bool = myoff == otoff
        if base_compare:
            return _cmp((self._hour, self._minute, self._second, self._microsecond), (other._hour, other._minute, other._second, other._microsecond))
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2
            else:
                raise TypeError('cannot compare naive and aware times')
        myhhmm: int = self._hour * 60 + self._minute - myoff // timedelta(minutes=1)
        othhmm: int = other._hour * 60 + other._minute - otoff // timedelta(minutes=1)
        return _cmp((myhhmm, self._second, self._microsecond), (othhmm, other._second, other._microsecond))

    def _tzstr(self, sep: str = ':') -> str:
        """Return formatted timezone offset (+xx:xx) or None."""
        off: timedelta = self.utcoffset()
        if off is not None:
            if off.days < 0:
                sign: str = '-'
                off = -off
            else:
                sign: str = '+'
            hh, mm: tuple[int, int] = divmod(off, timedelta(hours=1))
            assert not mm % timedelta(minutes=1), 'whole minute'
            mm //= timedelta(minutes=1)
            assert 0 <= hh < 24
            off = '{}{}{}{}'.format(sign, zfill(hh, 2), sep, zfill(mm, 2))
        return off

    def __repr__(self) -> str:
        """Convert to formal string, for repr().

        >>> tz = timezone.utc
        >>> repr(tz)
        'datetime.timezone.utc'
        >>> tz = timezone(timedelta(hours=-5), 'EST')
        >>> repr(tz)
        "datetime.timezone(datetime.timedelta(-1, 68400), 'EST')"
        """
        if self._tzinfo is None:
            return 'datetime.time({}, {}, {}, {}, {})'.format(self._hour, self._minute, self._second, self._microsecond)
        return 'datetime.time({}, {}, {}, {}, tzinfo={})'.format(self._hour, self._minute, self._second, self._microsecond, self._tzinfo.__repr__())

    def isoformat(self) -> str:
        """Return the time formatted according to ISO.

        This is 'HH:MM:SS.mmmmmm+zz:zz', or 'HH:MM:SS+zz:zz' if
        self.microsecond == 0.
        """
        s: str = _format_time(self._hour, self._minute, self._second, self._microsecond)
        tz: str = self._tzstr()
        if tz:
            s += tz
        return s

    def __str__(self) -> str:
        return self.isoformat()

    def strftime(self, fmt: str) -> str:
        """Format using strftime().  The date part of the timestamp passed
        to underlying strftime should not be used.
        """
        timetuple: tuple[int, int, int, int, int, int, int, int, int] = (1900, 1, 1, self._hour, self._minute, self._second, 0, 1, -1)
        return _wrap_strftime(self, fmt, timetuple)

    def __format__(self, fmt: str) -> str:
        if not isinstance(fmt, str):
            raise TypeError('must be str, not {}'.format(type(fmt).__name__))
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    def utcoffset(self) -> timedelta:
        """Return the timezone offset in minutes east of UTC (negative west of
        UTC)."""
        if self._tzinfo is None:
            return None
        offset: timedelta = self._tzinfo.utcoffset(None)
        _check_utc_offset('utcoffset', offset)
        return offset

    def tzname(self) -> str:
        """Return the timezone name.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        if self._tzinfo is None:
            return None
        name: str = self._tzinfo.tzname(None)
        _check_tzname(name)
        return name

    def dst(self) -> timedelta:
        """Return 0 if DST is not in effect, or the DST offset (in minutes
        eastward) if DST is in effect.

        This is purely informational; the DST offset has already been added to
        the UTC offset returned by utcoffset() if applicable, so there's no
        need to consult dst() unless you're interested in displaying the DST
        info.
        """
        if self._tzinfo is None:
            return None
        offset: timedelta = self._tzinfo.dst(None)
        _check_utc_offset('dst', offset)
        return offset

    def replace(self, hour: int = None, minute: int = None, second: int = None, microsecond: int = None, tzinfo: bool = True) -> time:
        """Return a new time with new values for the specified fields."""
        if hour is None:
            hour = self._hour
        if minute is None:
            minute = self._minute
        if second is None:
            second = self._second
        if microsecond is None:
            microsecond = self._microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        return time(hour, minute, second, microsecond, tzinfo)
    resolution: timedelta = timedelta(microseconds=1)
_time_class: type = time
_tm_min: time = time(0, 0, 0)
_tm_max: time = time(23, 59, 59, 999999)

class datetime(date):
    """datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])

    The year, month and day arguments are required. tzinfo may be None, or an
    instance of a tzinfo subclass. The remaining arguments may be ints.
    """

    def __init__(self, year: int, month: int = None, day: int = None, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0, tzinfo: tzinfo = None) -> None:
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
        self._tzinfo: tzinfo = tzinfo

    @property
    def hour(self) -> int:
        """hour (0-23)"""
        return self._hour

    @property
    def minute(self) -> int:
        """minute (0-59)"""
        return self._minute

    @property
    def second(self) -> int:
        """second (0-59)"""
        return self._second

    @property
    def microsecond(self) -> int:
        """microsecond (0-999999)"""
        return self._microsecond

    @property
    def tzinfo(self) -> tzinfo:
        """timezone info object"""
        return self._tzinfo

    @classmethod
    def _fromtimestamp(cls, t: float, utc: bool, tz: tzinfo) -> 'datetime':
        """Construct a datetime from a POSIX timestamp (like time.time()).

        A timezone info object may be passed in as well.
        """
        frac, t: tuple[float, float] = _math.modf(t)
        us: int = round(frac * 1000000.0)
        if us >= 1000000:
            t += 1
            us -= 1000000
        elif us < 0:
            t -= 1
            us += 1000000
        converter: callable = _time.gmtime if utc else _time.localtime
        y, m, d, hh, mm, ss, weekday, jday, dst = converter(t)
        ss = min(ss, 59)
        return cls(y, m, d, hh, mm, ss, us, tz)

    @classmethod
    def fromtimestamp(cls, t: float, tz: tzinfo = None) -> 'datetime':
        """Construct a datetime from a POSIX timestamp (like time.time()).

        A timezone info object may be passed in as well.
        """
        _check_tzinfo_arg(tz)
        result: 'datetime' = cls._fromtimestamp(t, tz is not None, tz)
        if tz is not None:
            result = tz.fromutc(result)
        return result

    @classmethod
    def utcfromtimestamp(cls, t: float) -> 'datetime':
        """Construct a naive UTC datetime from a POSIX timestamp."""
        return cls._fromtimestamp(t, True, None)

    @classmethod
    def now(cls, tz: tzinfo = None) -> 'datetime':
        """Construct a datetime from time.time() and optional time zone info."""
        t: float = _time.time()
        return cls.fromtimestamp(t, tz)

    @classmethod
    def utcnow(cls) -> 'datetime':
        """Construct a UTC datetime from time.time()."""
        t: float = _time.time()
        return cls.utcfromtimestamp(t)

    @classmethod
    def combine(cls, date: date, time: time) -> 'datetime':
        """Construct a datetime from a given date and a given time."""
        if not isinstance(date, _date_class):
            raise TypeError('date argument must be a date instance')
        if not isinstance(time, _time_class):
            raise TypeError('time argument must be a time instance')
        return cls(date.year, date.month, date.day, time.hour, time.minute, time.second, time.microsecond, time.tzinfo)

    def timetuple(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        """Return local time tuple compatible with time.localtime()."""
        dst: int = self.dst()
        if dst is None:
            dst = -1
        elif dst:
            dst = 1
        else:
            dst = 0
        return _build_struct_time(self.year, self.month, self.day, self.hour, self.minute, self.second, dst)

    def timestamp(self) -> float:
        """Return POSIX timestamp as float"""
        if self._tzinfo is None:
            return _time.mktime((self.year, self.month, self.day, self.hour, self.minute, self.second, -1, -1, -1)) + self.microsecond / 1000000.0
        else:
            return (self - _EPOCH).total_seconds()

    def utctimetuple(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        """Return UTC time tuple compatible with time.gmtime()."""
        offset: timedelta = self.utcoffset()
        if offset:
            self -= offset
        y, m, d = (self.year, self.month, self.day)
        hh, mm, ss = (self.hour, self.minute, self.second)
        return _build_struct_time(y, m, d, hh, mm, ss, 0)

    def date(self) -> date:
        """Return the date part."""
        return date(self._year, self._month, self._day)

    def time(self) -> time:
        """Return the time part, with tzinfo None."""
        return time(self.hour, self.minute, self.second, self.microsecond)

    def timetz(self) -> time:
        """Return the time part, with same tzinfo."""
        return time(self.hour, self.minute, self.second, self.microsecond, self._tzinfo)

    def replace(self, year: int = None, month: int = None, day: int = None, hour: int = None, minute: int = None, second: int = None, microsecond: int = None, tzinfo: bool = True) -> 'datetime':
        """Return a new datetime with new values for the specified fields."""
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
        return datetime(year, month, day, hour, minute, second, microsecond, tzinfo)

    def astimezone(self, tz: tzinfo = None) -> 'datetime':
        if tz is None:
            if self.tzinfo is None:
                raise ValueError('astimezone() requires an aware datetime')
            ts: float = (self - _EPOCH) // timedelta(seconds=1)
            localtm: tuple[int, int, int, int, int, int, int, int, int] = _time.localtime(ts)
            local: 'datetime' = datetime(*localtm[:6])
            if len(localtm) > 9:
                gmtoff: int = localtm[10]
                zone: str = localtm[9]
                tz: tzinfo = timezone(timedelta(seconds=gmtoff), zone)
            else:
                delta: timedelta = local - datetime(*_time.gmtime(ts)[:6])
                dst: bool = _time.daylight and localtm[8] > 0
                gmtoff: int = -(_time.altzone if dst else _time.timezone)
                if delta == timedelta(seconds=gmtoff):
                    tz: tzinfo = timezone(delta, _time.tzname[dst])
                else:
                    tz: tzinfo = timezone(delta)
        elif not isinstance(tz, tzinfo):
            raise TypeError('tz argument must be an instance of tzinfo')
        mytz: tzinfo = self.tzinfo
        if mytz is None:
            raise ValueError('astimezone() requires an aware datetime')
        if tz is mytz:
            return self
        myoffset: timedelta = self.utcoffset()
        if myoffset is None:
            raise ValueError('astimezone() requires an aware datetime')
        utc: 'datetime' = (self - myoffset).replace(tzinfo=tz)
        return tz.fromutc(utc)

    def ctime(self) -> str:
        """Return ctime() style string."""
        weekday: int = self.toordinal() % 7 or 7
        return '{} {} {} {}:{}:{} {}'.format(_DAYNAMES[weekday], _MONTHNAMES[self._month], zfill(self._day, 2), zfill(self._hour, 2), zfill(self._minute, 2), zfill(self._second, 2), zfill(self._year, 4))

    def isoformat(self, sep: str = 'T') -> str:
        """Return the time formatted according to ISO.

        This is 'YYYY-MM-DD HH:MM:SS.mmmmmm', or 'YYYY-MM-DD HH:MM:SS' if
        self.microsecond == 0.

        If self.tzinfo is not None, the UTC offset is also attached, giving
        'YYYY-MM-DD HH:MM:SS.mmmmmm+HH:MM' or 'YYYY-MM-DD HH:MM:SS+HH:MM'.

        Optional argument sep specifies the separator between date and
        time, default 'T'.
        """
        s: str = '{}-{}-{}{}'.format(zfill(self._year, 4), zfill(self._month, 2), zfill(self._day, 2), sep) + _format_time(self._hour, self._minute, self._second, self._microsecond)
        off: str = self.utcoffset()
        if off is not None:
            if off.days < 0:
                sign: str = '-'
                off = -off
            else:
                sign: str = '+'
            hh, mm: tuple[int, int] = divmod(off, timedelta(hours=1))
            assert not mm % timedelta(minutes=1), 'whole minute'
            mm //= timedelta(minutes=1)
            s += '{}{}:{}'.format(sign, zfill(hh, 2), zfill(mm, 2))
        return s

    def __repr__(self) -> str:
        """Convert to formal string, for repr().

        >>> dt = datetime(2010, 1, 1)
        >>> repr(dt)
        'datetime.datetime(2010, 1, 1, 0, 0)'

        >>> dt = datetime(2010, 1, 1, tzinfo=timezone.utc)
        >>> repr(dt)
        'datetime.datetime(2010, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)'
        """
        L: list[str] = [str(self._year), str(self._month), str(self._day), str(self._hour), str(self._minute), str(self._second), str(self._microsecond)]
        if L[len(L) - 1] == '0':
            L.pop()
        if L[len(L) - 1] == '0':
            L.pop()
        s: str = 'datetime.datetime({})'.format(', '.join(L))
        if self._tzinfo is not None:
            assert s[-1:] == ')'
            s = s[:len(s) - 1] + ', tzinfo={}'.format(self._tzinfo.__repr__()) + ')'
        return s

    def __str__(self) -> str:
        """Convert to string, for str().

        >>> dt = datetime(2010, 1, 1)
        >>> str(dt)
        '2010-01-01'

        >>> dt = datetime(2010, 1, 1, tzinfo=timezone.utc)
        >>> str(dt)
        '2010-01-01 00:00:00+00:00'
        """
        return self.isoformat(sep=' ')

    @classmethod
    def strptime(cls, date_string: str, format: str) -> 'datetime':
        """string, format -> new datetime parsed from a string (like time.strptime())."""
        return cls(*_time.strptime(date_string, format)[:6])

    def utcoffset(self) -> timedelta:
        """Return the timezone offset in minutes east of UTC (negative west of
        UTC)."""
        if self._tzinfo is None:
            return None
        offset: timedelta = self._tzinfo.utcoffset(self)
        _check_utc_offset('utcoffset', offset)
        return offset

    def tzname(self) -> str:
        """Return the timezone name.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        if self._tzinfo is None:
            return None
        name: str = self._tzinfo.tzname(self)
        _check_tzname(name)
        return name

    def dst(self) -> timedelta:
        """Return 0 if DST is not in effect, or the DST offset (in minutes
        eastward) if DST is in effect.

        This is purely informational; the DST offset has already been added to
        the UTC offset returned by utcoffset() if applicable, so there's no
        need to consult dst() unless you're interested in displaying the DST
        info.
        """
        if self._tzinfo is None:
            return None
        offset: timedelta = self._tzinfo.dst(self)
        _check_utc_offset('dst', offset)
        return offset

    def __eq__(self, other: datetime) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=True) == 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            return False

    def __le__(self, other: datetime) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other) <= 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __lt__(self, other: datetime) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other) < 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __ge__(self, other: datetime) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other) >= 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __gt__(self, other: datetime) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other) > 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def _cmp(self, other: datetime, allow_mixed: bool = False) -> int:
        assert isinstance(other, datetime)
        mytz: tzinfo = self._tzinfo
        ottz: tzinfo = other._tzinfo
        myoff: timedelta = otoff: timedelta = None
        if mytz is ottz:
            base_compare: bool = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            base_compare: bool = myoff == otoff
        if base_compare:
            s1: str = '{}{}{}{}{}{}{}'.format(zfill(self._year, 4), zfill(self._month, 2), zfill(self._day, 2), zfill(self._hour, 2), zfill(self._minute, 2), zfill(self._second, 2), zfill(self._microsecond, 6))
            s2: str = '{}{}{}{}{}{}{}'.format(zfill(other._year, 4), zfill(other._month, 2), zfill(other._day, 2), zfill(other._hour, 2), zfill(other._minute, 2), zfill(other._second, 2), zfill(other._microsecond, 6))
            return _cmp(s1, s2)
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2
            else:
                raise TypeError('cannot compare naive and aware datetimes')
        diff: timedelta = self - other
        if diff.days < 0:
            return -1
        return diff and 1 or 0

    def __add__(self, other: timedelta) -> 'datetime':
        """Add a datetime and a timedelta."""
        if not isinstance(other, timedelta):
            return NotImplemented
        delta: timedelta = timedelta(self.toordinal(), hours=self._hour, minutes=self._minute, seconds=self._second, microseconds=self._microsecond)
        delta += other
        hour, rem: tuple[int, int] = divmod(delta.seconds, 3600)
        minute, second: tuple[int, int] = divmod(rem, 60)
        if 0 < delta.days <= _MAXORDINAL:
            return datetime.combine(date.fromordinal(delta.days), time(hour, minute, second, delta.microseconds, tzinfo=self._tzinfo))
        raise OverflowError('result out of range')

    def __radd__(self, other: timedelta) -> 'datetime':
        return self.__add__(other)

    def __sub__(self, other: datetime) -> timedelta:
        """Subtract two datetimes, or a datetime and a timedelta."""
        if not isinstance(other, datetime):
            if isinstance(other, timedelta):
                return self + -other
            return NotImplemented
        days1: int = self.toordinal()
        days2: int = other.toordinal()
        secs1: int = self._second + self._minute * 60 + self._hour * 3600
        secs2: int = other._second + other._minute * 60 + other._hour * 3600
        base: timedelta = timedelta(days1 - days2, secs1 - secs2, self._microsecond - other._microsecond)
        if self._tzinfo is other._tzinfo:
            return base
        myoff: timedelta = self.utcoffset()
        otoff: timedelta = other.utcoffset()
        if myoff == otoff:
            return base
        if myoff is None or otoff is None:
            raise TypeError('cannot mix naive and timezone-aware time')
        return base + otoff - myoff
    resolution: timedelta = timedelta(microseconds=1)
_dt_min: datetime = datetime(1, 1, 1)
_dt_max: datetime = datetime(9999, 12, 31, 23, 59, 59, 999999)

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
    """Abstract base class for time zone info classes.

    Subclasses must override the name(), utcoffset() and dst() methods.
    """

    def __init__(self, offset: timedelta, name: str = _Omitted) -> None:
        if not isinstance(offset, timedelta):
            raise TypeError('offset must be a timedelta')
        if name is _Omitted:
            if not offset:
                offset = self.utc
            name = None
        elif not isinstance(name, str):
            raise TypeError('name must be a string')
        if not self._minoffset <= offset <= self._maxoffset:
            raise ValueError('offset must be a timedelta strictly between -timedelta(hours=24) and timedelta(hours=24).')
        if offset.microseconds != 0 or offset.seconds % 60 != 0:
            raise ValueError('offset must be a timedelta representing a whole number of minutes')
        self._offset: timedelta = offset
        self._name: str = name

    @classmethod
    def _create(cls, offset: timedelta, name: str = _Omitted) -> 'timezone':
        return cls(offset, name)

    def __eq__(self, other: 'timezone') -> bool:
        if type(other) != timezone:
            return False
        return self._offset == other._offset

    def __repr__(self) -> str:
        """Convert to formal string, for repr().

        >>> tz = timezone.utc
        >>> repr(tz)
        'datetime.timezone.utc'
        >>> tz = timezone(timedelta(hours=-5), 'EST')
        >>> repr(tz)
        "datetime.timezone(datetime.timedelta(-1, 68400), 'EST')"
        """
        if self is self.utc:
            return 'datetime.timezone.utc'
        if self._name is None:
            return 'datetime.timezone({})'.format(self._offset.__repr__())
        return 'datetime.timezone({}, {})'.format(self._offset.__repr__(), self._name.__repr__())

    def __str__(self) -> str:
        return self.tzname(None)

    def utcoffset(self, dt: datetime) -> timedelta:
        if isinstance(dt, datetime) or dt is None:
            return self._offset
        raise TypeError('utcoffset() argument must be a datetime instance or None')

    def tzname(self, dt: datetime) -> str:
        if isinstance(dt, datetime) or dt is None:
            if self._name is None:
                return self._name_from_offset(self._offset)
            return self._name
        raise TypeError('tzname() argument must be a datetime instance or None')

    def dst(self, dt: datetime) -> timedelta:
        if isinstance(dt, datetime) or dt is None:
            return None
        raise TypeError('dst() argument must be a datetime instance or None')

    def fromutc(self, dt: datetime) -> datetime:
        if isinstance(dt, datetime):
            if dt.tzinfo is not self:
                raise ValueError('fromutc: dt.tzinfo is not self')
            return dt + self._offset
        raise TypeError('fromutc() argument must be a datetime instance or None')
    _maxoffset: timedelta = timedelta(hours=23, minutes=59)
    _minoffset: timedelta = -timedelta(hours=23, minutes=59)

    @staticmethod
    def _name_from_offset(delta: timedelta) -> str:
        if delta < timedelta(0):
            sign: str = '-'
            delta = -delta
        else:
            sign: str = '+'
        hours, rest: tuple[int, timedelta] = divmod(delta, timedelta(hours=1))
        minutes: int = rest // timedelta(minutes=1)
        return 'UTC{}{}:{}'.format(sign, zfill(hours, 2), zfill(minutes, 2))
_tz_utc: timezone = timezone._create(timedelta(0))
_tz_min: timezone = timezone._create(timezone._minoffset)
_tz_max: timezone = timezone._create(timezone._maxoffset)
_EPOCH: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
