from hypothesis import given, strategies as st
import numpy as np
import pytest
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas._libs.tslibs import NaT, OutOfBoundsDatetime, Timedelta, Timestamp, iNaT, to_offset
import pandas._testing as tm

class TestTimestampRound:

    def test_round_division_by_zero_raises(self) -> None:
        ts: Timestamp = Timestamp('2016-01-01')
        msg: str = 'Division by zero in rounding'
        with pytest.raises(ValueError, match=msg):
            ts.round('0ns')

    def test_round_frequencies(self, timestamp: str, freq: str, expected: str) -> None:
        dt: Timestamp = Timestamp(timestamp)
        result: Timestamp = dt.round(freq)
        expected: Timestamp = Timestamp(expected)
        assert result == expected

    def test_round_tzaware(self) -> None:
        dt: Timestamp = Timestamp('20130101 09:10:11', tz='US/Eastern')
        result: Timestamp = dt.round('D')
        expected: Timestamp = Timestamp('20130101', tz='US/Eastern')
        assert result == expected
        dt = Timestamp('20130101 09:10:11', tz='US/Eastern')
        result = dt.round('s')
        assert result == dt

    def test_round_30min(self) -> None:
        dt: Timestamp = Timestamp('20130104 12:32:00')
        result: Timestamp = dt.round('30Min')
        expected: Timestamp = Timestamp('20130104 12:30:00')
        assert result == expected

    def test_round_subsecond(self) -> None:
        result: Timestamp = Timestamp('2016-10-17 12:00:00.0015').round('ms')
        expected: Timestamp = Timestamp('2016-10-17 12:00:00.002000')
        assert result == expected
        result = Timestamp('2016-10-17 12:00:00.00149').round('ms')
        expected = Timestamp('2016-10-17 12:00:00.001000')
        assert result == expected
        ts: Timestamp = Timestamp('2016-10-17 12:00:00.0015')
        for freq in ['us', 'ns']:
            assert ts == ts.round(freq)
        result = Timestamp('2016-10-17 12:00:00.001501031').round('10ns')
        expected = Timestamp('2016-10-17 12:00:00.001501030')
        assert result == expected

    def test_round_nonstandard_freq(self) -> None:
        with tm.assert_produces_warning(False):
            Timestamp('2016-10-17 12:00:00.001501031').round('1010ns')

    def test_round_invalid_arg(self) -> None:
        stamp: Timestamp = Timestamp('2000-01-05 05:09:15.13')
        with pytest.raises(ValueError, match=INVALID_FREQ_ERR_MSG):
            stamp.round('foo')

    def test_ceil_floor_edge(self, test_input: str, rounder: str, freq: str, expected: str) -> None:
        dt: Timestamp = Timestamp(test_input)
        func = getattr(dt, rounder)
        result: Timestamp = func(freq)
        if dt is NaT:
            assert result is NaT
        else:
            expected: Timestamp = Timestamp(expected)
            assert result == expected

    def test_round_minute_freq(self, test_input: str, freq: str, expected: str, rounder: str) -> None:
        dt: Timestamp = Timestamp(test_input)
        expected: Timestamp = Timestamp(expected)
        func = getattr(dt, rounder)
        result: Timestamp = func(freq)
        assert result == expected

    def test_ceil(self, unit) -> None:
        dt: Timestamp = Timestamp('20130101 09:10:11').as_unit(unit)
        result: Timestamp = dt.ceil('D')
        expected: Timestamp = Timestamp('20130102')
        assert result == expected
        assert result._creso == dt._creso

    def test_floor(self, unit) -> None:
        dt: Timestamp = Timestamp('20130101 09:10:11').as_unit(unit)
        result: Timestamp = dt.floor('D')
        expected: Timestamp = Timestamp('20130101')
        assert result == expected
        assert result._creso == dt._creso

    def test_round_dst_border_ambiguous(self, method, unit) -> None:
        ts: Timestamp = Timestamp('2017-10-29 00:00:00', tz='UTC').tz_convert('Europe/Madrid')
        ts = ts.as_unit(unit)
        result = getattr(ts, method)('h', ambiguous=True)
        assert result == ts
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
        result = getattr(ts, method)('h', ambiguous=False)
        expected: Timestamp = Timestamp('2017-10-29 01:00:00', tz='UTC').tz_convert('Europe/Madrid')
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
        result = getattr(ts, method)('h', ambiguous='NaT')
        assert result is NaT
        msg: str = 'Cannot infer dst time'
        with pytest.raises(ValueError, match=msg):
            getattr(ts, method)('h', ambiguous='raise')

    def test_round_dst_border_nonexistent(self, method, ts_str, freq, unit) -> None:
        ts: Timestamp = Timestamp(ts_str, tz='America/Chicago').as_unit(unit)
        result = getattr(ts, method)(freq, nonexistent='shift_forward')
        expected: Timestamp = Timestamp('2018-03-11 03:00:00', tz='America/Chicago')
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
        result = getattr(ts, method)(freq, nonexistent='NaT')
        assert result is NaT
        msg: str = '2018-03-11 02:00:00'
        with pytest.raises(ValueError, match=msg):
            getattr(ts, method)(freq, nonexistent='raise')

    def test_round_int64(self, timestamp: str, freq: str) -> None:
        dt: Timestamp = Timestamp(timestamp).as_unit('ns')
        unit: int = to_offset(freq).nanos
        result: Timestamp = dt.floor(freq)
        assert result._value % unit == 0, f'floor not a {freq} multiple'
        assert 0 <= dt._value - result._value < unit, 'floor error'
        result = dt.ceil(freq)
        assert result._value % unit == 0, f'ceil not a {freq} multiple'
        assert 0 <= result._value - dt._value < unit, 'ceil error'
        result = dt.round(freq)
        assert result._value % unit == 0, f'round not a {freq} multiple'
        assert abs(result._value - dt._value) <= unit // 2, 'round error'
        if unit % 2 == 0 and abs(result._value - dt._value) == unit // 2:
            assert result._value // unit % 2 == 0, 'round half to even error'

    def test_round_implementation_bounds(self) -> None:
        result: Timestamp = Timestamp.min.ceil('s')
        expected: Timestamp = Timestamp(1677, 9, 21, 0, 12, 44)
        assert result == expected
        result = Timestamp.max.floor('s')
        expected = Timestamp.max - Timedelta(854775807)
        assert result == expected
        msg: str = 'Cannot round 1677-09-21 00:12:43.145224193 to freq=<Second>'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.floor('s')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.min.round('s')
        msg = 'Cannot round 2262-04-11 23:47:16.854775807 to freq=<Second>'
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.ceil('s')
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            Timestamp.max.round('s')

    @given(val=st.integers(iNaT + 1, lib.i8max))
    @pytest.mark.parametrize('method', [Timestamp.round, Timestamp.floor, Timestamp.ceil])
    def test_round_sanity(self, val: int, method) -> None:
        cls: type = Timestamp
        err_cls: type = OutOfBoundsDatetime
        val: np.int64 = np.int64(val)
        ts: Timestamp = cls(val)

        def checker(ts: Timestamp, nanos: int, unit: str) -> None:
            if nanos == 1:
                pass
            else:
                div, mod = divmod(ts._value, nanos)
                diff = int(nanos - mod)
                lb = ts._value - mod
                assert lb <= ts._value
                ub = ts._value + diff
                assert ub > ts._value
                msg: str = 'without overflow'
                if mod == 0:
                    pass
                elif method is cls.ceil:
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif method is cls.floor:
                    if lb < cls.min._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif mod >= diff:
                    if ub > cls.max._value:
                        with pytest.raises(err_cls, match=msg):
                            method(ts, unit)
                        return
                elif lb < cls.min._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return
            res: Timestamp = method(ts, unit)
            td: Timedelta = res - ts
            diff: int = abs(td._value)
            assert diff < nanos
            assert res._value % nanos == 0
            if method is cls.round:
                assert diff <= nanos / 2
            elif method is cls.floor:
                assert res <= ts
            elif method is cls.ceil:
                assert res >= ts
        nanos: int = 1
        checker(ts, nanos, 'ns')
        nanos = 1000
        checker(ts, nanos, 'us')
        nanos = 1000000
        checker(ts, nanos, 'ms')
        nanos = 1000000000
        checker(ts, nanos, 's')
        nanos = 60 * 1000000000
        checker(ts, nanos, 'min')
        nanos = 60 * 60 * 1000000000
        checker(ts, nanos, 'h')
        nanos = 24 * 60 * 60 * 1000000000
        checker(ts, nanos, 'D')
