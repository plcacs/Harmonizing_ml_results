from hypothesis import given, strategies as st
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import NaT, OutOfBoundsDatetime, Timedelta, Timestamp, iNaT, to_offset
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas._testing as tm
from typing import Any, Callable, List, Tuple, Union

class TestTimestampRound:

    def test_round_division_by_zero_raises(self) -> None:
        ts: Timestamp = Timestamp('2016-01-01')
        msg: str = 'Division by zero in rounding'
        with pytest.raises(ValueError, match=msg):
            ts.round('0ns')

    @pytest.mark.parametrize('timestamp, freq, expected', [('20130101 09:10:11', 'D', '20130101'), ('20130101 19:10:11', 'D', '20130102'), ('20130201 12:00:00', 'D', '20130202'), ('20130104 12:00:00', 'D', '20130105'), ('2000-01-05 05:09:15.13', 'D', '2000-01-05 00:00:00'), ('2000-01-05 05:09:15.13', 'h', '2000-01-05 05:00:00'), ('2000-01-05 05:09:15.13', 's', '2000-01-05 05:09:15')])
    def test_round_frequencies(self, timestamp: str, freq: str, expected: str) -> None:
        dt: Timestamp = Timestamp(timestamp)
        result: Timestamp = dt.round(freq)
        expected_ts: Timestamp = Timestamp(expected)
        assert result == expected_ts

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

    @pytest.mark.parametrize('test_input, rounder, freq, expected', [('2117-01-01 00:00:45', 'floor', '15s', '2117-01-01 00:00:45'), ('2117-01-01 00:00:45', 'ceil', '15s', '2117-01-01 00:00:45'), ('2117-01-01 00:00:45.000000012', 'floor', '10ns', '2117-01-01 00:00:45.000000010'), ('1823-01-01 00:00:01.000000012', 'ceil', '10ns', '1823-01-01 00:00:01.000000020'), ('1823-01-01 00:00:01', 'floor', '1s', '1823-01-01 00:00:01'), ('1823-01-01 00:00:01', 'ceil', '1s', '1823-01-01 00:00:01'), ('NaT', 'floor', '1s', 'NaT'), ('NaT', 'ceil', '1s', 'NaT')])
    def test_ceil_floor_edge(self, test_input: str, rounder: str, freq: str, expected: str) -> None:
        dt: Union[Timestamp, type(NaT)] = Timestamp(test_input)
        func: Callable[[str], Union[Timestamp, type(NaT)]] = getattr(dt, rounder)
        result: Union[Timestamp, type(NaT)] = func(freq)
        if dt is NaT:
            assert result is NaT
        else:
            expected_ts: Timestamp = Timestamp(expected)
            assert result == expected_ts

    @pytest.mark.parametrize('test_input, freq, expected', [('2018-01-01 00:02:06', '2s', '2018-01-01 00:02:06'), ('2018-01-01 00:02:00', '2min', '2018-01-01 00:02:00'), ('2018-01-01 00:04:00', '4min', '2018-01-01 00:04:00'), ('2018-01-01 00:15:00', '15min', '2018-01-01 00:15:00'), ('2018-01-01 00:20:00', '20min', '2018-01-01 00:20:00'), ('2018-01-01 03:00:00', '3h', '2018-01-01 03:00:00')])
    @pytest.mark.parametrize('rounder', ['ceil', 'floor', 'round'])
    def test_round_minute_freq(self, test_input: str, freq: str, expected: str, rounder: str) -> None:
        dt: Timestamp = Timestamp(test_input)
        expected_ts: Timestamp = Timestamp(expected)
        func: Callable[[str], Timestamp] = getattr(dt, rounder)
        result: Timestamp = func(freq)
        assert result == expected_ts

    def test_ceil(self, unit: str) -> None:
        dt: Timestamp = Timestamp('20130101 09:10:11').as_unit(unit)
        result: Timestamp = dt.ceil('D')
        expected: Timestamp = Timestamp('20130102')
        assert result == expected
        assert result._creso == dt._creso

    def test_floor(self, unit: str) -> None:
        dt: Timestamp = Timestamp('20130101 09:10:11').as_unit(unit)
        result: Timestamp = dt.floor('D')
        expected: Timestamp = Timestamp('20130101')
        assert result == expected
        assert result._creso == dt._creso

    @pytest.mark.parametrize('method', ['ceil', 'round', 'floor'])
    def test_round_dst_border_ambiguous(self, method: str, unit: str) -> None:
        ts: Timestamp = Timestamp('2017-10-29 00:00:00', tz='UTC').tz_convert('Europe/Madrid')
        ts = ts.as_unit(unit)
        result: Union[Timestamp, type(NaT)] = getattr(ts, method)('h', ambiguous=True)
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

    @pytest.mark.parametrize('method, ts_str, freq', [['ceil', '2018-03-11 01:59:00-0600', '5min'], ['round', '2018-03-11 01:59:00-0600', '5min'], ['floor', '2018-03-11 03:01:00-0500', '2h']])
    def test_round_dst_border_nonexistent(self, method: str, ts_str: str, freq: str, unit: str) -> None:
        ts: Timestamp = Timestamp(ts_str, tz='America/Chicago').as_unit(unit)
        result: Union[Timestamp, type(NaT)] = getattr(ts, method)(freq, nonexistent='shift_forward')
        expected: Timestamp = Timestamp('2018-03-11 03:00:00', tz='America/Chicago')
        assert result == expected
        assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
        result = getattr(ts, method)(freq, nonexistent='NaT')
        assert result is NaT
        msg: str = '2018-03-11 02:00:00'
        with pytest.raises(ValueError, match=msg):
            getattr(ts, method)(freq, nonexistent='raise')

    @pytest.mark.parametrize('timestamp', ['2018-01-01 0:0:0.124999360', '2018-01-01 0:0:0.125000367', '2018-01-01 0:0:0.125500', '2018-01-01 0:0:0.126500', '2018-01-01 12:00:00', '2019-01-01 12:00:00'])
    @pytest.mark.parametrize('freq', ['2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '250ns', '500ns', '750ns', '1us', '19us', '250us', '500us', '750us', '1s', '2s', '3s', '1D'])
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
    def test_round_sanity(self, val: int, method: Callable[[Timestamp, str], Timestamp]) -> None:
        cls: type = Timestamp
        err_cls: type = OutOfBoundsDatetime
        val_np: np.int64 = np.int64(val)
        ts: Timestamp = cls(val_np)

        def checker(ts: Timestamp, nanos: int, unit: str) -> None:
            if nanos == 1:
                pass
            else:
                div, mod = divmod(ts._value, nanos)
                diff: int = int(nanos - mod)
                lb: int = ts._value - mod
                assert lb <= ts._value
                ub: int = ts._value + diff
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
            diff = abs(td._value)
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
