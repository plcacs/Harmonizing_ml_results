"""Concrete date/time and related types.

See http://www.iana.org/time-zones/repository/tz-link.html for
time zone and DST data sources.
"""

import time as _time
import math as _math
from org.transcrypt.stubs.browser import __envir__
from typing import Optional, Union, Tuple, Any


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

# Utility functions, adapted from Python's Demo/classes/Dates.py, which
# also assumes the current Gregorian calendar indefinitely extended in
# both directions.  Difference:  Dates.py calls January 1 of year 0 day
# number 1.  The code here calls January 1 of year 1 day number 1.  This is
# to match the definition of the "proleptic Gregorian" calendar in Dershowitz
# and Reingold's "Calendrical Calculations", where it's the base calendar
# for all computations.  See the book for algorithms for converting between
# proleptic Gregorian ordinals and many other calendar systems.

# -1 is a placeholder for indexing purposes.
_DAYS_IN_MONTH: list[int] = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

_DAYS_BEFORE_MONTH: list[int] = [-1]  # -1 is a placeholder for indexing purposes.
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
    assert 1 <= day <= dim, ('day must be in 1..%d' % dim)
    return (_days_before_year(year) +
            _days_before_month(year, month) +
            day)


_DI400Y: int = _days_before_year(401)  # number of days in 400 years
_DI100Y: int = _days_before_year(101)  # "    "   "   " 100   "
_DI4Y: int = _days_before_year(5)  # "    "   "   "   4   "

# A 4-year cycle has an extra leap day over what we'd get from pasting
# together 4 single years.
assert _DI4Y == 4 * 365 + 1

# Similarly, a 400-year cycle has an extra leap day over what we'd get from
# pasting together 4 100-year cycles.
assert _DI400Y == 4 * _DI100Y + 1

# OTOH, a 100-year cycle has one fewer leap day than we'd get from
# pasting together 25 4-year cycles.
assert _DI100Y == 25 * _DI4Y - 1


def _ord2ymd(n: int) -> Tuple[int, int, int]:
    """ordinal -> (year, month, day), considering 01-Jan-0001 as day 1."""

    # n is a 1-based index, starting at 1-Jan-1.  The pattern of leap years
    # repeats exactly every 400 years.  The basic strategy is to find the
    # closest 400-year boundary at or before n, then work with the offset
    # from that boundary to n.  Life is much clearer if we subtract 1 from
    # n first -- then the values of n at 400-year boundaries are exactly
    # those divisible by _DI400Y:
    #
    #     D  M   Y            n              n-1
    #     -- --- ----        ----------     ----------------
    #     31 Dec -400        -_DI400Y       -_DI400Y -1
    #      1 Jan -399         -_DI400Y +1   -_DI400Y      400-year boundary
    #     ...
    #     30 Dec  000        -1             -2
    #     31 Dec  000         0             -1
    #      1 Jan  001         1              0            400-year boundary
    #      2 Jan  001         2              1
    #      3 Jan  001         3              2
    #     ...
    #     31 Dec  400         _DI400Y        _DI400Y -1
    #      1 Jan  401         _DI400Y +1     _DI400Y      400-year boundary
    n -= 1
    n400, n = divmod(n, _DI400Y)
    year: int = n400 * 400 + 1  # ..., -399, 1, 401, ...

    # Now n is the (non-negative) offset, in days, from January 1 of year, to
    # the desired date.  Now compute how many 100-year cycles precede n.
    # Note that it's possible for n100 to equal 4!  In that case 4 full
    # 100-year cycles precede the desired day, which implies the desired
    # day is December 31 at the end of a 400-year cycle.
    n100, n = divmod(n, _DI100Y)

    # Now compute how many 4-year cycles precede it.
    n4, n = divmod(n, _DI4Y)

    # And now how many single years.  Again n1 can be 4, and again meaning
    # that the desired day is December 31 at the end of the 4-year cycle.
    n1, n = divmod(n, 365)

    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        assert n == 0
        return year - 1, 12, 31

    # Now the year is correct, and n is the offset from January 1.  We find
    # the month via an estimate that's either exact or one too large.
    leapyear: bool = n1 == 3 and (n4 != 24 or n100 == 3)
    assert leapyear == _is_leap(year)
    month: int = (n + 50) >> 5
    preceding: int = _DAYS_BEFORE_MONTH[month] + (month > 2 and leapyear)
    if preceding > n:  # estimate is too large
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + (month == 2 and leapyear)
    n -= preceding
    assert 0 <= n < _days_in_month(year, month)

    # Now the year and month are correct, and n is the offset from the
    # start of that month:  we're done!
    return year, month, n + 1


# Month and day names.  For localized versions, see the calendar module.
_MONTHNAMES: list[Optional[str]] = [None, "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_DAYNAMES: list[Optional[str]] = [None, "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _build_struct_time(y: int, m: int, d: int, hh: int, mm: int, ss: int, dstflag: int) -> Tuple[int, int, int, int, int, int, int, int, int]:
    wday: int = (_ymd2ord(y, m, d) + 6) % 7
    dnum: int = _days_before_month(y, m) + d
    # return _time.struct_time((y, m, d, hh, mm, ss, wday, dnum, dstflag))
    return (y, m, d, hh, mm, ss, wday, dnum, dstflag)


def _format_time(hh: int, mm: int, ss: int, us: int) -> str:
    # Skip trailing microseconds when us==0.
    result: str = "{}:{}:{}".format(zfill(hh, 2), zfill(mm, 2), zfill(ss, 2))
    if us:
        result += ".{}".format(zfill(us, 6))
    return result


# Correctly substitute for %z and %Z escapes in strftime formats.
def _wrap_strftime(object: Any, format: str, timetuple: Tuple[int, int, int, int, int, int, int, int, int]) -> str:
    from datetime import timedelta  # Type hint requires timedelta
    # Don't call utcoffset() or tzname() unless actually needed.
    freplace: Optional[str] = None  # the string to use for %f
    zreplace: Optional[str] = None  # the string to use for %z
    Zreplace: Optional[str] = None  # the string to use for %Z

    # Scan format for %z and %Z escapes, replacing as needed.
    newformat: list[str] = []
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
                                m = m // timedelta(minutes=1)
                                zreplace = '{}{}{}'.format(sign, zfill(int(h), 2), zfill(int(m), 2))
                    assert '%' not in zreplace
                    newformat.append(zreplace)
                elif ch == 'Z':
                    if Zreplace is None:
                        Zreplace = ""
                        if hasattr(object, "tzname"):
                            s = object.tzname()
                            if s is not None:
                                # strftime is going to have at this: escape %
                                Zreplace = s.replace('%', '%%')
                    newformat.append(Zreplace)
                else:
                    newformat.append('%')
                    newformat.append(ch)
            else:
                newformat.append('%')
        else:
            newformat.append(ch)
    newformat_str: str = "".join(newformat)
    return _time.strftime(newformat_str, timetuple)


# Just raise TypeError if the arg isn't None or a string.
def _check_tzname(name: Any) -> None:
    if name is not None and not isinstance(name, str):
        raise TypeError("tzinfo.tzname() must return None or string, "
                        "not '{}'".format(type(name)))


# name is the offset-producing method, "utcoffset" or "dst".
# offset is what it returned.
# If offset isn't None or timedelta, raises TypeError.
# If offset is None, returns None.
# Else offset is checked for being in range, and a whole # of minutes.
# If it is, its integer value is returned.  Else ValueError is raised.
def _check_utc_offset(name: str, offset: Any) -> Optional[timedelta]:
    from datetime import timedelta
    assert name in ("utcoffset", "dst")
    if offset is None:
        return None
    if not isinstance(offset, timedelta):
        raise TypeError("tzinfo.{}() must return None "
                        "or timedelta, not '{}'".format(name, type(offset)))
    if offset.__mod__(timedelta(minutes=1)).microseconds or offset.microseconds:
        raise ValueError("tzinfo.{}() must return a whole number "
                         "of minutes, got {}".format(name, offset))
    # __pragma__('opov')
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError("{}()={}, must be must be strictly between "
                         "-timedelta(hours=24) and timedelta(hours=24)".format(name, offset))
    # __pragma__('noopov')


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


def _check_time_fields(hour: Any, minute: Any, second: Any, microsecond: Any) -> Tuple[int, int, int, int]:
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


def _check_tzinfo_arg(tz: Any) -> None:
    if tz is not None and not isinstance(tz, tzinfo):
        raise TypeError("tzinfo argument must be None or of a tzinfo subclass")


def _cmperror(x: Any, y: Any) -> None:
    raise TypeError("can't compare '{}' to '{}'".format(
        type(x).__name__, type(y).__name__))


def _divide_and_round(a: int, b: int) -> int:
    """divide a by b and round result to the nearest integer

    When the ratio is exactly half-way between two integers,
    the even integer is returned.
    """
    # Based on the reference implementation for divmod_near
    # in Objects/longobject.c.
    q, r = divmod(a, b)
    # round up if either r / b > 0.5, or r / b == 0.5 and q is odd.
    # The expression r / b > 0.5 is equivalent to 2 * r > b if b is
    # positive, 2 * r < b if b negative.
    r *= 2
    if b > 0:
        greater_than_half = r > b
    else:
        greater_than_half = r < b
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

    # __pragma__('kwargs')
    def __init__(self, days: Union[int, float] = 0, seconds: Union[int, float] = 0, microseconds: Union[int, float] = 0,
                milliseconds: Union[int, float] = 0, minutes: Union[int, float] = 0, hours: Union[int, float] = 0, weeks: Union[int, float] = 0) -> None:
        from datetime import timedelta as timedelta_cls
        # Doing this efficiently and accurately in C is going to be difficult
        # and error-prone, due to ubiquitous overflow possibilities, and that
        # C double doesn't have enough bits of precision to represent
        # microseconds over 10K years faithfully.  The code here tries to make
        # explicit where go-fast assumptions can be relied on, in order to
        # guide the C implementation; it's way more convoluted than speed-
        # ignoring auto-overflow-to-long idiomatic Python could be.

        # XXX Check that all inputs are ints or floats.

        # Final values, all integer.
        # s and us fit in 32-bit signed ints; d isn't bounded.
        d: int = 0
        s: int = 0
        us: int = 0

        # Normalize everything to days, seconds, microseconds.
        days += weeks * 7
        seconds += minutes * 60 + hours * 3600
        microseconds += milliseconds * 1000

        # Get rid of all fractions, and normalize s and us.
        # Take a deep breath <wink>.
        if isinstance(days, float):
            dayfrac, days = _math.modf(days)
            daysecondsfrac, daysecondswhole = _math.modf(dayfrac * (24. * 3600.))
            assert daysecondswhole == int(daysecondswhole)  # can't overflow
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
        # days isn't referenced again before redefinition

        if isinstance(seconds, float):
            secondsfrac, seconds = _math.modf(seconds)
            assert seconds == int(seconds)
            seconds = int(seconds)
            secondsfrac += daysecondsfrac
            assert abs(secondsfrac) <= 2.0
        else:
            secondsfrac = daysecondsfrac
        # daysecondsfrac isn't referenced again
        assert isinstance(secondsfrac, (float, int))
        assert abs(secondsfrac) <= 2.0

        assert isinstance(seconds, int)
        days, seconds = divmod(seconds, 24 * 3600)
        d += days
        s += int(seconds)  # can't overflow
        assert isinstance(s, int)
        assert abs(s) <= 2 * 24 * 3600
        # seconds isn't referenced again before redefinition

        usdouble: float = secondsfrac * 1e6
        assert abs(usdouble) < 2.1e6  # exact value not critical
        # secondsfrac isn't referenced again

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
        assert abs(microseconds) < 3_100_000

        # Just a little bit of carrying possible for microseconds and seconds.
        seconds, us = divmod(microseconds, 1000000)
        s += seconds
        days, s = divmod(s, 24 * 3600)
        d += days

        assert isinstance(d, int)
        assert isinstance(s, int) and 0 <= s < 24 * 3600
        assert isinstance(us, int) and 0 <= us < 1_000_000

        if abs(d) > 999_999_999:
            raise OverflowError("timedelta # of days is too large: %d" % d)

        self._days: int = d
        self._seconds: int = s
        self._microseconds: int = us
    # __pragma__('nokwargs')

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
        s = "{}:{}:{}".format(hh, zfill(mm, 2), zfill(ss, 2))
        if self._days:
            def plural(n: int) -> Tuple[int, str]:
                return n, "s" if abs(n) != 1 else ""

            s = ("{} day{}, ".format(plural(self._days))) + s
        if self._microseconds:
            s = s + ".{}".format(zfill(self._microseconds, 6))
        return s

    def total_seconds(self) -> float:
        """Total seconds in the duration."""
        return ((self.days * 86400 + self.seconds) * 10 ** 6 +
                self.microseconds) / 10 ** 6

    # Read-only field accessors
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

    def __add__(self, other: Any) -> Union['timedelta', Any]:
        if isinstance(other, timedelta):
            # for CPython compatibility, we cannot use
            # our __class__ here, but need a real timedelta
            return timedelta(self._days + other._days,
                             self._seconds + other._seconds,
                             self._microseconds + other._microseconds)
        return NotImplemented

    def __radd__(self, other: Any) -> Union['timedelta', Any]:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Union['timedelta', Any]:
        if isinstance(other, timedelta):
            # for CPython compatibility, we cannot use
            # our __class__ here, but need a real timedelta
            return timedelta(self._days - other._days,
                             self._seconds - other._seconds,
                             self._microseconds - other._microseconds)
        return NotImplemented

    def __rsub__(self, other: Any) -> Union['timedelta', Any]:
        if isinstance(other, timedelta):
            return -self + other
        return NotImplemented

    def __neg__(self) -> 'timedelta':
        # for CPython compatibility, we cannot use
        # our __class__ here, but need a real timedelta
        return timedelta(-self._days,
                         -self._seconds,
                         -self._microseconds)

    def __pos__(self) -> 'timedelta':
        return self

    def __abs__(self) -> 'timedelta':
        if self._days < 0:
            # __pragma__('opov')
            return -self
            # __pragma__('noopov')
        else:
            return self

    def __mul__(self, other: Any) -> Union['timedelta', Any]:
        from datetime import timedelta as timedelta_cls
        if isinstance(other, int):
            # for CPython compatibility, we cannot use
            # our __class__ here, but need a real timedelta
            return timedelta(self._days * other,
                             self._seconds * other,
                             self._microseconds * other)
        if isinstance(other, float):
            usec: int = self._to_microseconds()
            a, b = other.as_integer_ratio()
            return timedelta(0, 0, _divide_and_round(usec * a, b))
        return NotImplemented

    def __rmul__(self, other: Any) -> Union['timedelta', Any]:
        return self.__mul__(other)

    def _to_microseconds(self) -> int:
        return ((self._days * (24 * 3600) + self._seconds) * 1_000_000 +
                self._microseconds)

    def __floordiv__(self, other: Any) -> Union[int, 'timedelta', Any]:
        if not isinstance(other, (int, timedelta)):
            return NotImplemented
        usec: int = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec // other._to_microseconds()
        if isinstance(other, int):
            return timedelta(0, 0, usec // other)

    def __truediv__(self, other: Any) -> Union[float, 'timedelta', Any]:
        from datetime import timedelta as timedelta_cls
        if not isinstance(other, (int, float, timedelta)):
            return NotImplemented
        usec: int = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec / other._to_microseconds()
        if isinstance(other, int):
            return timedelta(0, 0, _divide_and_round(usec, other))
        if isinstance(other, float):
            a, b = other.as_integer_ratio()
            return timedelta(0, 0, _divide_and_round(b * usec, a))

    def __mod__(self, other: Any) -> Union['timedelta', Any]:
        if isinstance(other, timedelta):
            r: int = self._to_microseconds() % other._to_microseconds()
            return timedelta(0, 0, r)
        return NotImplemented

    def __divmod__(self, other: Any) -> Union[Tuple[int, 'timedelta'], Any]:
        if isinstance(other, timedelta):
            q, r = divmod(self._to_microseconds(),
                          other._to_microseconds())
            return q, timedelta(0, 0, r)
        return NotImplemented

    # Comparisons of timedelta objects with other.

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

    def _cmp(self, other: 'timedelta') -> int:
        assert isinstance(other, timedelta)
        return _cmp(self._to_microseconds(), other._to_microseconds())

    def __bool__(self) -> bool:
        return (self._days != 0 or
                self._seconds != 0 or
                self._microseconds != 0)


_td_min: timedelta = timedelta(-999_999_999)
_td_max: timedelta = timedelta(days=999_999_999, hours=23, minutes=59, seconds=59,
                          microseconds=999_999)
_td_resolution: timedelta = timedelta(microseconds=1)
# __pragma__ ('js', '{}', '''Object.defineProperty (timedelta, 'min', {get: function () {return _td_min;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (timedelta, 'max', {get: function () {return _td_max;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (timedelta, 'resolution', {get: function () {return _td_resolution;}})''')


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

    # __pragma__('kwargs')
    def __init__(self, year: Any, month: Optional[Any] = None, day: Optional[Any] = None) -> None:
        year, month, day = _check_date_fields(year, month, day)
        self._year: int = year
        self._month: int = month
        self._day: int = day
    # __pragma__('nokwargs')

    # Additional constructors

    @classmethod
    def fromtimestamp(cls, t: float) -> 'date':
        "Construct a date from a POSIX timestamp (like time.time())."
        y, m, d, hh, mm, ss, weekday, jday, dst = _time.localtime(t)
        return cls(y, m, d)

    @classmethod
    def today(cls) -> 'date':
        "Construct a date from time.time()."
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

    # Conversions to string

    def __repr__(self) -> str:
        """Convert to formal string, for repr().

        >>> dt = datetime(2010, 1, 1)
        >>> repr(dt)
        'datetime.datetime(2010, 1, 1, 0, 0)'

        >>> dt = datetime(2010, 1, 1, tzinfo=timezone.utc)
        >>> repr(dt)
        'datetime.datetime(2010, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)'
        """
        return "datetime.date({}, {}, {})".format(
                                      self._year,
                                      self._month,
                                      self._day)

    # XXX These shouldn't depend on time.localtime(), because that
    # clips the usable dates to [1970 .. 2038).  At least ctime() is
    # easily done without using strftime() -- that's better too because
    # strftime("%c", ...) is locale specific.

    def ctime(self) -> str:
        """Return ctime() style string."""
        weekday: int = self.toordinal() % 7 or 7
        return "{} {} {} 00:00:00 {}".format(
            _DAYNAMES[weekday],
            _MONTHNAMES[self._month],
            rjust(self._day, 2), zfill(self._year, 4))

    def strftime(self, fmt: str) -> str:
        """Format using strftime()."""
        return _wrap_strftime(self, fmt, self.timetuple())

    def __format__(self, fmt: str) -> str:
        if not isinstance(fmt, str):
            raise TypeError("must be str, not {}".format(type(fmt).__name__))
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
        return "{}-{}-{}".format(zfill(self._year, 4), zfill(self._month, 2), zfill(self._day, 2))

    def __str__(self) -> str:
        return self.isoformat()

    # Read-only field accessors
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

    # Standard conversions, __eq__, __le__, __lt__, __ge__, __gt__ (and helpers)

    def timetuple(self) -> Tuple[int, int, int, int, int, int, int, int, int]:
        """Return local time tuple compatible with time.localtime()."""
        return _build_struct_time(self._year, self._month, self._day,
                                  0, 0, 0, -1)

    def toordinal(self) -> int:
        """Return proleptic Gregorian ordinal for the year, month and day.

        January 1 of year 1 is day 1.  Only the year, month and day values
        contribute to the result.
        """
        return _ymd2ord(self._year, self._month, self._day)

    # __pragma__('kwargs')
    def replace(self, year: Optional[Any] = None, month: Optional[Any] = None, day: Optional[Any] = None) -> 'date':
        """Return a new date with new values for the specified fields."""
        if year is None:
            year = self._year
        if month is None:
            month = self._month
        if day is None:
            day = self._day
        return date(year, month, day)
    # __pragma__('nokwargs')

    # Comparisons of date objects with other.

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

    def _cmp(self, other: 'date') -> int:
        assert isinstance(other, date)
        y, m, d = self._year, self._month, self._day
        y2, m2, d2 = other._year, other._month, other._day
        return _cmp('{}{}{}'.format(zfill(y, 4), zfill(m, 2), zfill(d, 2)),
                    '{}{}{}'.format(zfill(y2, 4), zfill(m2, 2), zfill(d2, 2)))

    # Computations

    def __add__(self, other: Any) -> Union['date', Any]:
        """Add a date to a timedelta."""
        if isinstance(other, timedelta):
            o: int = self.toordinal() + other.days
            if 0 < o <= _MAXORDINAL:
                return date.fromordinal(o)
            raise OverflowError("result out of range")
        return NotImplemented

    def __radd__(self, other: Any) -> Union['date', Any]:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Union['timedelta', 'date', Any]:
        """Subtract two dates, or a date and a timedelta."""
        from datetime import timedelta as timedelta_cls
        if isinstance(other, timedelta):
            # __pragma__('opov')
            return self + timedelta(-other.days)
            # __pragma__('noopov')
        if isinstance(other, date):
            days1: int = self.toordinal()
            days2: int = other.toordinal()
            # __pragma__('opov')
            return timedelta(days1 - days2)
            # __pragma__('noopov')
        return NotImplemented

    def weekday(self) -> int:
        """Return day of the week, where Monday == 0 ... Sunday == 6."""
        return (self.toordinal() + 6) % 7

    # Day-of-the-week and week-of-the-year, according to ISO

    def isoweekday(self) -> int:
        """Return day of the week, where Monday == 1 ... Sunday == 7."""
        # 1-Jan-0001 is a Monday
        return self.toordinal() % 7 or 7

    def isocalendar(self) -> Tuple[int, int, int]:
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
        # Internally, week and day have origin 0
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


_date_class: Any = date  # so functions w/ args named "date" can get at the class

_d_min: date = date(1, 1, 1)
_d_max: date = date(9999, 12, 31)
# __pragma__ ('js', '{}', '''Object.defineProperty (date, 'min', {get: function () {return _d_min;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (date, 'max', {get: function () {return _d_max;}})''')


class tzinfo:
    """Abstract base class for time zone info classes.

    Subclasses must override the name(), utcoffset() and dst() methods.
    """

    def tzname(self, dt: Optional['datetime']) -> Optional[str]:
        """datetime -> string name of time zone."""
        raise NotImplementedError("tzinfo subclass must override tzname()")

    def utcoffset(self, dt: Optional['datetime']) -> Optional[timedelta]:
        """datetime -> minutes east of UTC (negative for west of UTC)"""
        raise NotImplementedError("tzinfo subclass must override utcoffset()")

    def dst(self, dt: Optional['datetime']) -> Optional[timedelta]:
        """datetime -> DST offset in minutes east of UTC.

        Return 0 if DST not in effect.  utcoffset() must include the DST
        offset.
        """
        raise NotImplementedError("tzinfo subclass must override dst()")

    def fromutc(self, dt: 'datetime') -> 'datetime':
        """datetime in UTC -> datetime in local time."""

        if not isinstance(dt, datetime):
            raise TypeError("fromutc() requires a datetime argument")
        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        dtoff: Optional[timedelta] = dt.utcoffset()
        if dtoff is None:
            raise ValueError("fromutc() requires a non-None utcoffset() "
                             "result")

        # See the long comment block at the end of this file for an
        # explanation of this algorithm.
        dtdst: Optional[timedelta] = dt.dst()
        if dtdst is None:
            raise ValueError("fromutc() requires a non-None dst() result")
        delta: Optional[timedelta] = dtoff - dtdst
        if delta:
            dt += delta
            dtdst = dt.dst()
            if dtdst is None:
                raise ValueError("fromutc(): dt.dst gave inconsistent "
                                 "results; cannot convert")
        return dt + dtdst


_tzinfo_class: Any = tzinfo


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

    # __pragma__('kwargs')
    def __init__(self, hour: Any = 0, minute: Any = 0, second: Any = 0, microsecond: Any = 0, tzinfo: Optional[tzinfo] = None) -> None:
        hour, minute, second, microsecond = _check_time_fields(
            hour, minute, second, microsecond)
        _check_tzinfo_arg(tzinfo)
        self._hour: int = hour
        self._minute: int = minute
        self._second: int = second
        self._microsecond: int = microsecond
        self._tzinfo: Optional[tzinfo] = tzinfo
    # __pragma__('nokwargs')

    # Read-only field accessors
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
    def tzinfo(self) -> Optional[tzinfo]:
        """timezone info object"""
        return self._tzinfo

    # Comparisons of time objects with other.

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=True) == 0
        else:
            return False

    def __le__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other) <= 0
        else:
            _cmperror(self, other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other) < 0
        else:
            _cmperror(self, other)

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other) >= 0
        else:
            _cmperror(self, other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, time):
            return self._cmp(other) > 0
        else:
            _cmperror(self, other)

    # __pragma__('kwargs')
    def _cmp(self, other: 'time', allow_mixed: bool = False) -> int:
        assert isinstance(other, time)
        mytz: Optional[tzinfo] = self._tzinfo
        ottz: Optional[tzinfo] = other._tzinfo
        myoff: Optional[timedelta] = None
        otoff: Optional[timedelta] = None

        if mytz is ottz:
            base_compare = True
        else:
            if mytz is not None:
                myoff = self.utcoffset()
            if ottz is not None:
                otoff = other.utcoffset()
            base_compare = myoff == otoff

        if base_compare:
            return _cmp((self._hour, self._minute, self._second,
                         self._microsecond),
                        (other._hour, other._minute, other._second,
                         other._microsecond))
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2  # arbitrary non-zero value
            else:
                raise TypeError("cannot compare naive and aware times")
        # __pragma__('opov')
        myhhmm: int = self._hour * 60 + self._minute - (myoff // timedelta(minutes=1)).__int__()
        othhmm: int = other._hour * 60 + other._minute - (otoff // timedelta(minutes=1)).__int__()
        # __pragma__('noopov')
        return _cmp((myhhmm, self._second, self._microsecond),
                    (othhmm, other._second, other._microsecond))
    # __pragma__('nokwargs')

    # Conversion to string

    def _tzstr(self, sep: str = ":") -> Optional[str]:
        """Return formatted timezone offset (+xx:xx) or None."""
        offset: Optional[timedelta] = self.utcoffset()
        if offset is not None:
            if offset.days < 0:
                sign: str = "-"
                offset = -offset
            else:
                sign = "+"
            h, m = divmod(offset, timedelta(hours=1))
            assert not m % timedelta(minutes=1), "whole minute"
            m = m // timedelta(minutes=1)
            assert 0 <= h < 24
            off: str = "{}{}{}{}".format(sign, zfill(int(h), 2), sep, zfill(int(m), 2))
            return off
        return None

    def __repr__(self) -> str:
        """Convert to formal string, for repr()."""
        components: list[str] = [str(self._hour), str(self._minute)]
        if self._microsecond != 0:
            components.extend([str(self._second), str(self._microsecond)])
        elif self._second != 0:
            components.append(str(self._second))
        s: str = "datetime.time({}, {}".format(self._hour, self._minute)
        if self._second != 0 or self._microsecond != 0:
            s += ", {}".format(self._second)
            if self._microsecond != 0:
                s += ", {}".format(self._microsecond)
        s += ")"
        if self._tzinfo is not None:
            assert s[-1:] == ")"
            s = s[:len(s)-1] + ", tzinfo={}".format(repr(self._tzinfo)) + ")"
        return s

    def isoformat(self, sep: str = 'T') -> str:
        """Return the time formatted according to ISO.

        This is 'HH:MM:SS.mmmmmm+zz:zz', or 'HH:MM:SS+zz:zz' if
        self.microsecond == 0.
        """
        s: str = _format_time(self._hour, self._minute, self._second,
                             self._microsecond)
        tz: Optional[str] = self._tzstr()
        if tz:
            s += tz
        return s

    def __str__(self) -> str:
        return self.isoformat()

    def strftime(self, fmt: str) -> str:
        """Format using strftime().  The date part of the timestamp passed
        to underlying strftime should not be used.
        """
        # The year must be >= 1000 else Python's strftime implementation
        # can raise a bogus exception.
        timetuple: Tuple[int, int, int, int, int, int, int, int, int] = (1900, 1, 1,
                                                                         self._hour, self._minute, self._second,
                                                                         0, 1, -1)
        return _wrap_strftime(self, fmt, timetuple)

    def __format__(self, fmt: str) -> str:
        if not isinstance(fmt, str):
            raise TypeError("must be str, not %s" % type(fmt).__name__)
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    # Timezone functions

    def utcoffset(self) -> Optional[timedelta]:
        """Return the timezone offset in minutes east of UTC (negative west of
        UTC)."""
        if self._tzinfo is None:
            return None
        offset: Optional[timedelta] = self._tzinfo.utcoffset(self)
        _check_utc_offset("utcoffset", offset)
        return offset

    def tzname(self) -> Optional[str]:
        """Return the timezone name.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        if self._tzinfo is None:
            return None
        name: Optional[str] = self._tzinfo.tzname(self)
        _check_tzname(name)
        return name

    def dst(self) -> Optional[timedelta]:
        """Return 0 if DST is not in effect, or the DST offset (in minutes eastward) if DST is in effect.

        This is purely informational; the DST offset has already been added to
        the UTC offset returned by utcoffset() if applicable, so there's no
        need to consult dst() unless you're interested in displaying the DST
        info.
        """
        if self._tzinfo is None:
            return None
        offset: Optional[timedelta] = self._tzinfo.dst(self)
        _check_utc_offset("dst", offset)
        return offset

    # __pragma__('kwargs')
    def replace(self, hour: Optional[Any] = None, minute: Optional[Any] = None, second: Optional[Any] = None, microsecond: Optional[Any] = None,
                tzinfo: Union[Optional[tzinfo], bool] = True) -> 'time':
        """Return a new time with new values for the specified fields."""
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
        return time(hour, minute, second, microsecond, tzinfo)
    # __pragma__('nokwargs')

    resolution: timedelta = timedelta(microseconds=1)


_time_class: Any = time  # so functions w/ args named "time" can get at the class

_tm_min: time = time(0, 0, 0)
_tm_max: time = time(23, 59, 59, 999_999)
# __pragma__ ('js', '{}', '''Object.defineProperty (time, 'min', {get: function () {return _tm_min;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (time, 'max', {get: function () {return _tm_max;}})''')


class datetime(date):
    """datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])

    The year, month and day arguments are required. tzinfo may be None, or an
    instance of a tzinfo subclass. The remaining arguments may be ints.
    """

    # __pragma__('kwargs')
    def __init__(self, year: Any, month: Optional[Any] = None, day: Optional[Any] = None, hour: Any = 0, minute: Any = 0, second: Any = 0,
                microsecond: Any = 0, tzinfo: Optional[tzinfo] = None) -> None:
        year, month, day = _check_date_fields(year, month, day)
        hour, minute, second, microsecond = _check_time_fields(
            hour, minute, second, microsecond)
        _check_tzinfo_arg(tzinfo)
        self._year: int = year
        self._month: int = month
        self._day: int = day
        self._hour: int = hour
        self._minute: int = minute
        self._second: int = second
        self._microsecond: int = microsecond
        self._tzinfo: Optional[tzinfo] = tzinfo
    # __pragma__('nokwargs')

    # Read-only field accessors
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
    def tzinfo(self) -> Optional[tzinfo]:
        """timezone info object"""
        return self._tzinfo

    @classmethod
    def _fromtimestamp(cls, t: float, utc: bool, tz: Optional[tzinfo]) -> 'datetime':
        """Construct a datetime from a POSIX timestamp (like time.time()).

        A timezone info object may be passed in as well.
        """
        frac, t_int = _math.modf(t)
        us: int = round(frac * 1_000_000)
        if us >= 1_000_000:
            t_int += 1
            us -= 1_000_000
        elif us < 0:
            t_int -= 1
            us += 1_000_000

        converter = _time.gmtime if utc else _time.localtime
        y, m, d, hh, mm, ss, weekday, jday, dst = converter(t_int)
        ss = min(ss, 59)  # clamp out leap seconds if the platform has them
        return cls(y, m, d, hh, mm, ss, us, tz)

    @classmethod
    def fromtimestamp(cls, t: float, tz: Optional[tzinfo] = None) -> 'datetime':
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
    def now(cls, tz: Optional[tzinfo] = None) -> 'datetime':
        """Construct a datetime from time.time() and optional time zone info."""
        t: float = _time.time()
        return cls.fromtimestamp(t, tz)

    @classmethod
    def utcnow(cls) -> 'datetime':
        """Construct a UTC datetime from time.time()."""
        t: float = _time.time()
        return cls.utcfromtimestamp(t)

    @classmethod
    def combine(cls, date_obj: 'date', time_obj: 'time') -> 'datetime':
        """Construct a datetime from a given date and a given time."""
        if not isinstance(date_obj, _date_class):
            raise TypeError("date argument must be a date instance")
        if not isinstance(time_obj, _time_class):
            raise TypeError("time argument must be a time instance")
        return cls(date_obj.year, date_obj.month, date_obj.day,
                   time_obj.hour, time_obj.minute, time_obj.second, time_obj.microsecond,
                   time_obj.tzinfo)

    def timetuple(self) -> Tuple[int, int, int, int, int, int, int, int, int]:
        """Return local time tuple compatible with time.localtime()."""
        dst_val: int
        dtdst: Optional[timedelta] = self.dst()
        if dtdst is None:
            dst_val = -1
        elif dtdst:
            dst_val = 1
        else:
            dst_val = 0
        return _build_struct_time(self.year, self.month, self.day,
                                  self.hour, self.minute, self.second,
                                  dst_val)

    def timestamp(self) -> float:
        """Return POSIX timestamp as float"""
        if self._tzinfo is None:
            return _time.mktime((self.year, self.month, self.day,
                                 self.hour, self.minute, self.second,
                                 -1, -1, -1)) + self.microsecond / 1e6
        else:
            # __pragma__('opov')
            return (self - _EPOCH).total_seconds()
            # __pragma__('noopov')

    def utctimetuple(self) -> Tuple[int, int, int, int, int, int, int, int, int]:
        """Return UTC time tuple compatible with time.gmtime()."""
        offset: Optional[timedelta] = self.utcoffset()
        if offset:
            # __pragma__('opov')
            self_minus_offset: 'datetime' = self - offset
            # __pragma__('noopov')
            y, m, d, hh, mm, ss, _, _, _ = self_minus_offset.year, self_minus_offset.month, self_minus_offset.day, self_minus_offset.hour, self_minus_offset.minute, self_minus_offset.second
            return _build_struct_time(y, m, d, hh, mm, ss, 0)
        y, m, d, hh, mm, ss = self.year, self.month, self.day, self.hour, self.minute, self.second
        return _build_struct_time(y, m, d, hh, mm, ss, 0)

    def date(self) -> 'date':
        """Return the date part."""
        return date(self._year, self._month, self._day)

    def time(self) -> 'time':
        """Return the time part, with tzinfo None."""
        return time(self.hour, self.minute, self.second, self.microsecond)

    def timetz(self) -> 'time':
        """Return the time part, with same tzinfo."""
        return time(self.hour, self.minute, self.second, self.microsecond,
                    self._tzinfo)

    # __pragma__('kwargs')
    def replace(self, year: Optional[Any] = None, month: Optional[Any] = None, day: Optional[Any] = None, hour: Optional[Any] = None,
                minute: Optional[Any] = None, second: Optional[Any] = None, microsecond: Optional[Any] = None, tzinfo: Union[Optional[tzinfo], bool] = True) -> 'datetime':
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
        return datetime(year, month, day, hour, minute, second, microsecond,
                        tzinfo)
    # __pragma__('nokwargs')

    # __pragma__('kwargs')
    def astimezone(self, tz: Optional[tzinfo] = None) -> 'datetime':
        if tz is None:
            if self.tzinfo is None:
                raise ValueError("astimezone() requires an aware datetime")

            # __pragma__('opov')
            ts: float = (self - _EPOCH) // timedelta(seconds=1)
            # __pragma__('noopov')

            localtm: Tuple[Any, ...] = _time.localtime(ts)
            local: 'datetime' = datetime(*(localtm[:6]))
            if len(localtm) > 9:
                # Extract TZ data if available
                gmtoff: Any = localtm[10]
                zone: Any = localtm[9]
                tz = timezone(timedelta(seconds=gmtoff), zone)
            else:
                # Compute UTC offset and compare with the value implied
                # by tm_isdst.  If the values match, use the zone name
                # implied by tm_isdst.
                # __pragma__('opov')
                delta: timedelta = local - datetime(*(_time.gmtime(ts)[:6]))
                dst_flag: bool = _time.daylight and localtm[8] > 0
                gmtoff: int = -(_time.altzone if dst_flag else _time.timezone)
                if delta == timedelta(seconds=gmtoff):
                    tz = timezone(timedelta(seconds=gmtoff), _time.tzname[dst_flag])
                else:
                    tz = timezone(delta)
                # __pragma__('noopov')

        elif not isinstance(tz, tzinfo):
            raise TypeError("tz argument must be an instance of tzinfo")

        mytz: Optional[tzinfo] = self.tzinfo
        if mytz is None:
            raise ValueError("astimezone() requires an aware datetime")

        if tz is mytz:
            return self

        # Convert self to UTC, and attach the new time zone object.
        myoff: Optional[timedelta] = self.utcoffset()
        if myoff is None:
            raise ValueError("astimezone() requires an aware datetime")
        # __pragma__('opov')
        utc: 'datetime' = (self - myoff).replace(tzinfo=tz)
        # __pragma__('noopov')

        # Convert from UTC to tz's local time.
        return tz.fromutc(utc)
    # __pragma__('nokwargs')

    # Ways to produce a string.

    def ctime(self) -> str:
        """Return ctime() style string."""
        weekday: int = self.toordinal() % 7 or 7
        return "{} {} {} {}:{}:{} {}".format(
            _DAYNAMES[weekday],
            _MONTHNAMES[self._month],
            zfill(self._day, 2),
            zfill(self._hour, 2), zfill(self._minute, 2), zfill(self._second, 2),
            zfill(self._year, 4))

    def isoformat(self, sep: str = 'T') -> str:
        """Return the time formatted according to ISO.

        This is 'YYYY-MM-DD HH:MM:SS.mmmmmm', or 'YYYY-MM-DD HH:MM:SS' if
        self.microsecond == 0.

        If self.tzinfo is not None, the UTC offset is also attached, giving
        'YYYY-MM-DD HH:MM:SS.mmmmmm+HH:MM' or 'YYYY-MM-DD HH:MM:SS+HH:MM'.

        Optional argument sep specifies the separator between date and
        time, default 'T'.
        """
        s: str = ("{}-{}-{}{}".format(zfill(self._year, 4), zfill(self._month, 2), zfill(self._day, 2), sep) +
             _format_time(self._hour, self._minute, self._second,
                          self._microsecond))
        offset: Optional[str] = self._tzstr(sep=':')
        if offset:
            s += offset
        return s

    def __repr__(self) -> str:
        """Convert to formal string, for repr()."""
        components: list[str] = [str(self._year), str(self._month), str(self._day),
                                 str(self._hour), str(self._minute), str(self._second), str(self._microsecond)]
        if components[-1] == '0':
            components.pop()
        if components[-1] == '0':
            components.pop()
        s: str = "datetime.datetime({})".format(", ".join(map(str, components)))
        if self._tzinfo is not None:
            assert s[-1:] == ")"
            s = s[:len(s)-1] + ", tzinfo={}".format(repr(self._tzinfo)) + ")"
        return s

    def __str__(self) -> str:
        """Convert to string, for str()."""
        return self.isoformat(sep=' ')

    @classmethod
    def strptime(cls, date_string: str, format: str) -> 'datetime':
        """string, format -> new datetime parsed from a string (like time.strptime())."""
        return cls(*(_time.strptime(date_string, format)[:6]))

    def utcoffset(self) -> Optional[timedelta]:
        """Return the timezone offset in minutes east of UTC (negative west of
        UTC)."""
        if self._tzinfo is None:
            return None
        offset: Optional[timedelta] = self._tzinfo.utcoffset(self)
        _check_utc_offset("utcoffset", offset)
        return offset

    def tzname(self) -> Optional[str]:
        """Return the timezone name.

        Note that the name is 100% informational -- there's no requirement that
        it mean anything in particular. For example, "GMT", "UTC", "-500",
        "-5:00", "EDT", "US/Eastern", "America/New York" are all valid replies.
        """
        if self._tzinfo is None:
            return None
        name: Optional[str] = self._tzinfo.tzname(self)
        _check_tzname(name)
        return name

    def dst(self) -> Optional[timedelta]:
        """Return 0 if DST is not in effect, or the DST offset (in minutes
        eastward) if DST is in effect.

        This is purely informational; the DST offset has already been added to
        the UTC offset returned by utcoffset() if applicable, so there's no
        need to consult dst() unless you're interested in displaying the DST
        info.
        """
        if self._tzinfo is None:
            return None
        offset: Optional[timedelta] = self._tzinfo.dst(self)
        _check_utc_offset("dst", offset)
        return offset

    # Comparisons of datetime objects with other.

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=True) == 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            return False

    def __le__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other) <= 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other) < 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other) >= 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, datetime):
            return self._cmp(other) > 0
        elif not isinstance(other, date):
            return NotImplemented
        else:
            _cmperror(self, other)

    def _cmp(self, other: 'datetime', allow_mixed: bool = False) -> int:
        assert isinstance(other, datetime)
        mytz: Optional[tzinfo] = self._tzinfo
        ottz: Optional[tzinfo] = other._tzinfo
        myoff: Optional[timedelta] = None
        otoff: Optional[timedelta] = None

        if mytz is ottz:
            base_compare = True
        else:
            if mytz is not None:
                myoff = self.utcoffset()
            if ottz is not None:
                otoff = other.utcoffset()
            base_compare = myoff == otoff

        if base_compare:
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
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2  # arbitrary non-zero value
            else:
                raise TypeError("cannot compare naive and timezone-aware datetimes")
        # XXX What follows could be done more efficiently...
        # __pragma__('opov')
        diff: timedelta = self - other  # this will take offsets into account
        # __pragma__('noopov')
        if diff.days < 0:
            return -1
        return 1 if diff.days > 0 or diff.seconds > 0 or diff.microseconds > 0 else 0

    def __add__(self, other: Any) -> Union['datetime', Any]:
        """Add a datetime and a timedelta."""
        if not isinstance(other, timedelta):
            return NotImplemented
        delta: timedelta = timedelta(self.toordinal(),
                                     seconds=self._hour * 3600 + self._minute * 60 + self._second,
                                     microseconds=self._microsecond)
        # __pragma__('opov')
        delta += other
        # __pragma__('noopov')
        hour, rem = divmod(delta.seconds, 3600)
        minute, second = divmod(rem, 60)
        if 0 < delta.days <= _MAXORDINAL:
            return datetime.combine(date.fromordinal(delta.days),
                                    time(hour, minute, second,
                                         delta.microseconds,
                                         tzinfo=self._tzinfo))
        raise OverflowError("result out of range")

    def __radd__(self, other: Any) -> Union['datetime', Any]:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Union['timedelta', 'datetime', Any]:
        """Subtract two datetimes, or a datetime and a timedelta."""
        from datetime import timedelta as timedelta_cls
        if not isinstance(other, datetime):
            if isinstance(other, timedelta):
                # __pragma__('opov')
                return self + -other
                # __pragma__('noopov')
            return NotImplemented

        days1: int = self.toordinal()
        days2: int = other.toordinal()
        secs1: int = self._second + self._minute * 60 + self._hour * 3600
        secs2: int = other._second + other._minute * 60 + other._hour * 3600
        base: timedelta = timedelta(days1 - days2,
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
        # __pragma__('opov')
        return base + otoff - myoff
        # __pragma__('noopov')

    resolution: timedelta = timedelta(microseconds=1)


_dt_min: datetime = datetime(1, 1, 1)
_dt_max: datetime = datetime(9999, 12, 31, 23, 59, 59, 999999)
# __pragma__ ('js', '{}', '''Object.defineProperty (datetime, 'min', {get: function () {return _dt_min;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (datetime, 'max', {get: function () {return _dt_max;}})''')


def _isoweek1monday(year: int) -> int:
    # Helper to calculate the day number of the Monday starting week 1
    # XXX This could be done more efficiently
    THURSDAY: int = 3
    firstday: int = _ymd2ord(year, 1, 1)
    firstweekday: int = (firstday + 6) % 7  # See weekday() above
    week1monday: int = firstday - firstweekday
    if firstweekday > THURSDAY:
        week1monday += 7
    return week1monday


# Sentinel value to disallow None
_Omitted: Any = '@#$^&$^'


class timezone(tzinfo):

    # __pragma__('kwargs')
    def __init__(self, offset: timedelta, name: Union[Optional[str], Any] = _Omitted) -> None:
        if not isinstance(offset, timedelta):
            raise TypeError("offset must be a timedelta")
        if name is _Omitted:
            if not offset:
                # offset = timedelta(0)  # utc
                offset = self.utc
            name = None
        elif not isinstance(name, str):
            raise TypeError("name must be a string")
        # __pragma__('opov')
        if not (self._minoffset <= offset <= self._maxoffset):
            raise ValueError("offset must be a timedelta "
                             "strictly between -timedelta(hours=24) and timedelta(hours=24).")
        # __pragma__('noopov')
        if (offset.microseconds != 0 or offset.seconds % 60 != 0):
            raise ValueError("offset must be a timedelta "
                             "representing a whole number of minutes")
        self._offset: timedelta = offset
        self._name: Optional[str] = name
    # __pragma__('nokwargs')

    # __pragma__('kwargs')
    @classmethod
    def _create(cls, offset: timedelta, name: Union[Optional[str], Any] = _Omitted) -> 'timezone':
        return cls(offset, name)
    # __pragma__('nokwargs')

    def __eq__(self, other: Any) -> bool:
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
            return "datetime.timezone({})".format(repr(self._offset))
        return "datetime.timezone({}, {})".format(repr(self._offset), repr(self._name))

    def __str__(self) -> str:
        return self.tzname(None)

    def utcoffset(self, dt: Optional['datetime']) -> Optional[timedelta]:
        if isinstance(dt, datetime) or dt is None:
            return self._offset
        raise TypeError("utcoffset() argument must be a datetime instance"
                        " or None")

    def tzname(self, dt: Optional['datetime']) -> Optional[str]:
        if isinstance(dt, datetime) or dt is None:
            if self._name is None:
                return self._name_from_offset(self._offset)
            return self._name
        raise TypeError("tzname() argument must be a datetime instance"
                        " or None")

    def dst(self, dt: Optional['datetime']) -> Optional[timedelta]:
        if isinstance(dt, datetime) or dt is None:
            return None
        raise TypeError("dst() argument must be a datetime instance"
                        " or None")

    _maxoffset: timedelta = timedelta(hours=23, minutes=59)
    # __pragma__('opov')
    _minoffset: timedelta = -timedelta(hours=23, minutes=59)
    # __pragma__('noopov')

    @staticmethod
    def _name_from_offset(delta: timedelta) -> str:
        # __pragma__('opov')
        if delta < timedelta(0):
            sign: str = '-'
            delta = -delta
        else:
            sign = '+'
        hours, remainder = divmod(delta, timedelta(hours=1))
        minutes = remainder // timedelta(minutes=1)
        # __pragma__('noopov')
        return 'UTC{}{}:{}'.format(sign, zfill(int(hours), 2), zfill(int(minutes), 2))


_tz_utc: timezone = timezone._create(timedelta(0))
_tz_min: timezone = timezone._create(timezone._minoffset)
_tz_max: timezone = timezone._create(timezone._maxoffset)
# __pragma__ ('js', '{}', '''Object.defineProperty (timezone, 'utc', {get: function () {return _tz_utc;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (timezone, 'min', {get: function () {return _tz_min;}})''')
# __pragma__ ('js', '{}', '''Object.defineProperty (timezone, 'max', {get: function () {return _tz_max;}})''')


_EPOCH: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
