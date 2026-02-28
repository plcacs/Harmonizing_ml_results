from datetime import datetime, timedelta
import numpy as np
import pytest
from pandas.compat import WASM
from pandas.errors import OutOfBoundsDatetime
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, NaT, Series, Timedelta, TimedeltaIndex, Timestamp, offsets, timedelta_range
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import assert_invalid_addsub_type, assert_invalid_comparison, get_upcast_box

def assert_dtype(obj: Series | Index | DataFrame, expected_dtype: str) -> None:
    """
    Helper to check the dtype for a Series, Index, or single-column DataFrame.
    """
    dtype = tm.get_dtype(obj)
    assert dtype == expected_dtype

def get_expected_name(box: type, names: list[str]) -> str:
    if box is DataFrame:
        exname = names[0]
    elif box in [tm.to_array, pd.array]:
        exname = names[1]
    else:
        exname = names[2]
    return exname

class TestTimedelta64ArrayLikeComparisons:

    def test_compare_timedelta64_zerodim(self, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = box_with_array if box_with_array not in [Index, pd.array] else np.ndarray
        tdi: TimedeltaIndex = timedelta_range('2h', periods=4)
        other: np.ndarray = np.array(tdi.to_numpy()[0])
        tdi: TimedeltaIndex = tm.box_expected(tdi, box)
        res: Series = tdi <= other
        expected: np.ndarray = np.array([True, False, False, False])
        expected: Series = tm.box_expected(expected, xbox)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize('td_scalar', [timedelta(days=1), Timedelta(days=1), Timedelta(days=1).to_timedelta64(), offsets.Hour(24)])
    def test_compare_timedeltalike_scalar(self, box_with_array: type, td_scalar: timedelta | Timedelta | np.timedelta64 | offsets.BaseOffset) -> None:
        box: type = box_with_array
        xbox: type = box if box not in [Index, pd.array] else np.ndarray
        ser: Series = Series([timedelta(days=1), timedelta(days=2)])
        ser: Series = tm.box_expected(ser, box)
        actual: Series = ser > td_scalar
        expected: Series = Series([False, True])
        expected: Series = tm.box_expected(expected, xbox)
        tm.assert_equal(actual, expected)

    @pytest.mark.parametrize('invalid', [345600000000000, 'a', Timestamp('2021-01-01'), Timestamp('2021-01-01').now('UTC'), Timestamp('2021-01-01').now().to_datetime64(), Timestamp('2021-01-01').now().to_pydatetime(), Timestamp('2021-01-01').date(), np.array(4)])
    def test_td64_comparisons_invalid(self, box_with_array: type, invalid: int | str | Timestamp | datetime | date | np.ndarray) -> None:
        box: type = box_with_array
        rng: TimedeltaIndex = timedelta_range('1 days', periods=10)
        obj: Series = tm.box_expected(rng, box)
        assert_invalid_comparison(obj, invalid, box)

    @pytest.mark.parametrize('other', [list(range(10)), np.arange(10), np.arange(10).astype(np.float32), np.arange(10).astype(object), pd.date_range('1970-01-01', periods=10, tz='UTC').array, np.array(pd.date_range('1970-01-01', periods=10)), list(pd.date_range('1970-01-01', periods=10)), pd.date_range('1970-01-01', periods=10).astype(object), pd.period_range('1971-01-01', freq='D', periods=10).array, pd.period_range('1971-01-01', freq='D', periods=10).astype(object)])
    def test_td64arr_cmp_arraylike_invalid(self, other: list[int] | np.ndarray | Index | Series | DataFrame | np.ndarray | list[datetime] | pd.Series | pd.DataFrame, box_with_array: type) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days', periods=10)._data
        rng: Series = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(rng, other, box_with_array)

    def test_td64arr_cmp_mixed_invalid(self) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days', periods=5)._data
        other: np.ndarray = np.array([0, 1, 2, rng[3], Timestamp('2021-01-01')])
        result: np.ndarray = rng == other
        expected: np.ndarray = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = rng != other
        tm.assert_numpy_array_equal(result, ~expected)
        msg: str = 'Invalid comparison between|Cannot compare type|not supported between'
        with pytest.raises(TypeError, match=msg):
            rng < other
        with pytest.raises(TypeError, match=msg):
            rng > other
        with pytest.raises(TypeError, match=msg):
            rng <= other
        with pytest.raises(TypeError, match=msg):
            rng >= other

class TestTimedelta64ArrayComparisons:

    @pytest.mark.parametrize('dtype', [None, object])
    def test_comp_nat(self, dtype: type[None] | type[object]) -> None:
        left: TimedeltaIndex = TimedeltaIndex([Timedelta('1 days'), NaT, Timedelta('3 days')])
        right: TimedeltaIndex = TimedeltaIndex([NaT, NaT, Timedelta('3 days')])
        lhs, rhs: TimedeltaIndex = (left, right)
        if dtype is object:
            lhs, rhs: TimedeltaIndex = (left.astype(object), right.astype(object))
        result: np.ndarray = rhs == lhs
        expected: np.ndarray = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = rhs != lhs
        expected: np.ndarray = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
        expected: np.ndarray = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs == NaT, expected)
        tm.assert_numpy_array_equal(NaT == rhs, expected)
        expected: np.ndarray = np.array([True, True, True])
        tm.assert_numpy_array_equal(lhs != NaT, expected)
        tm.assert_numpy_array_equal(NaT != lhs, expected)
        expected: np.ndarray = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs < NaT, expected)
        tm.assert_numpy_array_equal(NaT > lhs, expected)

    @pytest.mark.parametrize('idx2', [TimedeltaIndex(['2 day', '2 day', NaT, NaT, '1 day 00:00:02', '5 days 00:00:03']), np.array([np.timedelta64(2, 'D'), np.timedelta64(2, 'D'), np.timedelta64('nat'), np.timedelta64('nat'), np.timedelta64(1, 'D') + np.timedelta64(2, 's'), np.timedelta64(5, 'D') + np.timedelta64(3, 's')])])
    def test_comparisons_nat(self, idx2: TimedeltaIndex | np.ndarray) -> None:
        idx1: TimedeltaIndex = TimedeltaIndex(['1 day', NaT, '1 day 00:00:01', NaT, '1 day 00:00:01', '5 day 00:00:03'])
        result: np.ndarray = idx1 < idx2
        expected: np.ndarray = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = idx2 > idx1
        expected: np.ndarray = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = idx1 <= idx2
        expected: np.ndarray = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = idx2 >= idx1
        expected: np.ndarray = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = idx1 == idx2
        expected: np.ndarray = np.array([False, False, False, False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = idx1 != idx2
        expected: np.ndarray = np.array([True, True, True, True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_comparisons_coverage(self) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days', periods=10)
        result: np.ndarray = rng < rng[3]
        expected: np.ndarray = np.array([True, True, True] + [False] * 7)
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = rng == list(rng)
        exp: np.ndarray = rng == rng
        tm.assert_numpy_array_equal(result, exp)

class TestTimedelta64ArithmeticUnsorted:

    def test_ufunc_coercions(self) -> None:
        idx: TimedeltaIndex = TimedeltaIndex(['2h', '4h', '6h', '8h', '10h'], freq='2h', name='x')
        for result in [idx * 2, np.multiply(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp: TimedeltaIndex = TimedeltaIndex(['4h', '8h', '12h', '16h', '20h'], freq='4h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == '4h'
        for result in [idx / 2, np.divide(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp: TimedeltaIndex = TimedeltaIndex(['1h', '2h', '3h', '4h', '5h'], freq='h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == 'h'
        for result in [-idx, np.negative(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp: TimedeltaIndex = TimedeltaIndex(['-2h', '-4h', '-6h', '-8h', '-10h'], freq='-2h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == '-2h'
        idx: TimedeltaIndex = TimedeltaIndex(['-2h', '-1h', '0h', '1h', '2h'], freq='h', name='x')
        for result in [abs(idx), np.absolute(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp: TimedeltaIndex = TimedeltaIndex(['2h', '1h', '0h', '1h', '2h'], freq=None, name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq is None

    def test_subtraction_ops(self) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti: pd.DatetimeIndex = pd.date_range('20130101', periods=3, name='bar')
        td: Timedelta = Timedelta('1 days')
        dt: Timestamp = Timestamp('20130101')
        msg: str = 'cannot subtract a datelike from a TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi - dt
        with pytest.raises(TypeError, match=msg):
            tdi - dti
        msg: str = 'unsupported operand type\\(s\\) for -'
        with pytest.raises(TypeError, match=msg):
            td - dt
        msg: str = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            td - dti
        result: TimedeltaIndex = dt - dti
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', '-1 days', '-2 days'], name='bar')
        tm.assert_index_equal(result, expected)
        result: TimedeltaIndex = dti - dt
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', '1 days', '2 days'], name='bar')
        tm.assert_index_equal(result, expected)
        result: TimedeltaIndex = tdi - td
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', NaT, '1 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result: TimedeltaIndex = td - tdi
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', NaT, '-1 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = dti - td
        expected: pd.DatetimeIndex = DatetimeIndex(['20121231', '20130101', '20130102'], dtype='M8[ns]', freq='D', name='bar')
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = dt - tdi
        expected: pd.DatetimeIndex = DatetimeIndex(['20121231', NaT, '20121230'], dtype='M8[ns]', name='foo')
        tm.assert_index_equal(result, expected)

    def test_subtraction_ops_with_tz(self, box_with_array: type) -> None:
        dti: pd.DatetimeIndex = pd.date_range('20130101', periods=3)
        dti: Series = tm.box_expected(dti, box_with_array)
        ts: Timestamp = Timestamp('20130101')
        dt: datetime = ts.to_pydatetime()
        dti_tz: pd.DatetimeIndex = pd.date_range('20130101', periods=3).tz_localize('US/Eastern')
        dti_tz: Series = tm.box_expected(dti_tz, box_with_array)
        ts_tz: Timestamp = Timestamp('20130101').tz_localize('US/Eastern')
        ts_tz2: Timestamp = Timestamp('20130101').tz_localize('CET')
        dt_tz: datetime = ts_tz.to_pydatetime()
        td: Timedelta = Timedelta('1 days')

        def _check(result: Timedelta, expected: Timedelta) -> None:
            assert result == expected
            assert isinstance(result, Timedelta)

        result: Timedelta = ts - ts
        expected: Timedelta = Timedelta('0 days')
        _check(result, expected)
        result: Timedelta = dt_tz - ts_tz
        expected: Timedelta = Timedelta('0 days')
        _check(result, expected)
        result: Timedelta = ts_tz - dt_tz
        expected: Timedelta = Timedelta('0 days')
        _check(result, expected)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime-like objects.'
        with pytest.raises(TypeError, match=msg):
            dt_tz - ts
        msg: str = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt_tz - dt
        msg: str = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt - dt_tz
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime-like objects.'
        with pytest.raises(TypeError, match=msg):
            ts - dt_tz
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - ts
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - dt
        msg: str = 'Cannot subtract tz-naive and tz-aware'
        with pytest.raises(TypeError, match=msg):
            dti - ts_tz
        with pytest.raises(TypeError, match=msg):
            dti_tz - ts
        result: TimedeltaIndex = dti_tz - dt_tz
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', '1 days', '2 days'])
        expected: Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result: TimedeltaIndex = dt_tz - dti_tz
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', '-1 days', '-2 days'])
        expected: Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result: TimedeltaIndex = dti_tz - ts_tz
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', '1 days', '2 days'])
        expected: Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result: TimedeltaIndex = ts_tz - dti_tz
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', '-1 days', '-2 days'])
        expected: Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result: Timedelta = td - td
        expected: Timedelta = Timedelta('0 days')
        _check(result, expected)
        result: pd.DatetimeIndex = dti_tz - td
        expected: pd.DatetimeIndex = DatetimeIndex(['20121231', '20130101', '20130102'], tz='US/Eastern').as_unit('ns')
        expected: Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dti_tdi_numeric_ops(self) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti: pd.DatetimeIndex = pd.date_range('20130101', periods=3, name='bar')
        result: TimedeltaIndex = tdi - tdi
        expected: TimedeltaIndex = TimedeltaIndex(['0 days', NaT, '0 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result: TimedeltaIndex = tdi + tdi
        expected: TimedeltaIndex = TimedeltaIndex(['2 days', NaT, '4 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = dti - tdi
        expected: pd.DatetimeIndex = DatetimeIndex(['20121231', NaT, '20130101'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)

    def test_addition_ops(self) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti: pd.DatetimeIndex = pd.date_range('20130101', periods=3, name='bar')
        td: Timedelta = Timedelta('1 days')
        dt: Timestamp = Timestamp('20130101')
        result: pd.DatetimeIndex = tdi + dt
        expected: pd.DatetimeIndex = DatetimeIndex(['20130102', NaT, '20130103'], dtype='M8[ns]', name='foo')
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = dt + tdi
        expected: pd.DatetimeIndex = DatetimeIndex(['20130102', NaT, '20130103'], dtype='M8[ns]', name='foo')
        tm.assert_index_equal(result, expected)
        result: TimedeltaIndex = td + tdi
        expected: TimedeltaIndex = TimedeltaIndex(['2 days', NaT, '3 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result: TimedeltaIndex = tdi + td
        expected: TimedeltaIndex = TimedeltaIndex(['2 days', NaT, '3 days'], name='foo')
        tm.assert_index_equal(result, expected)
        msg: str = 'cannot add indices of unequal length'
        with pytest.raises(ValueError, match=msg):
            tdi + dti[0:1]
        with pytest.raises(ValueError, match=msg):
            tdi[0:1] + dti
        msg: str = 'Addition/subtraction of integers and integer-arrays'
        with pytest.raises(TypeError, match=msg):
            tdi + Index([1, 2, 3], dtype=np.int64)
        result: pd.DatetimeIndex = tdi + dti
        expected: pd.DatetimeIndex = DatetimeIndex(['20130102', NaT, '20130105'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = dti + tdi
        expected: pd.DatetimeIndex = DatetimeIndex(['20130102', NaT, '20130105'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)
        result: Timestamp = dt + td
        expected: Timestamp = Timestamp('20130102')
        assert result == expected
        result: Timestamp = td + dt
        expected: Timestamp = Timestamp('20130102')
        assert result == expected

    @pytest.mark.parametrize('freq', ['D', 'B'])
    def test_timedelta(self, freq: str) -> None:
        index: pd.DatetimeIndex = pd.date_range('1/1/2000', periods=50, freq=freq)
        shifted: pd.DatetimeIndex = index + timedelta(1)
        back: pd.DatetimeIndex = shifted + timedelta(-1)
        back: pd.DatetimeIndex = back._with_freq('infer')
        tm.assert_index_equal(index, back)
        if freq == 'D':
            expected: offsets.BaseOffset = pd.tseries.offsets.Day(1)
            assert index.freq == expected
            assert shifted.freq == expected
            assert back.freq == expected
        else:
            assert index.freq == pd.tseries.offsets.BusinessDay(1)
            assert shifted.freq is None
            assert back.freq == pd.tseries.offsets.BusinessDay(1)
        result: pd.DatetimeIndex = index - timedelta(1)
        expected: pd.DatetimeIndex = index + timedelta(-1)
        tm.assert_index_equal(result, expected)

    def test_timedelta_tick_arithmetic(self) -> None:
        rng: pd.DatetimeIndex = pd.date_range('2013', '2014')
        s: Series = Series(rng)
        result1: pd.DatetimeIndex = rng - offsets.Hour(1)
        result2: pd.DatetimeIndex = DatetimeIndex(s - np.timedelta64(100000000))
        result3: pd.DatetimeIndex = rng - np.timedelta64(100000000)
        result4: pd.DatetimeIndex = DatetimeIndex(s - offsets.Hour(1))
        assert result1.freq == rng.freq
        result1: pd.DatetimeIndex = result1._with_freq(None)
        tm.assert_index_equal(result1, result4)
        assert result3.freq == rng.freq
        result3: pd.DatetimeIndex = result3._with_freq(None)
        tm.assert_index_equal(result2, result3)

    def test_tda_add_sub_index(self) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex(['1 days', NaT, '2 days'])
        tda: np.ndarray = tdi.array
        dti: pd.DatetimeIndex = pd.date_range('1999-12-31', periods=3, freq='D')
        result: pd.DatetimeIndex = tda + dti
        expected: pd.DatetimeIndex = tdi + dti
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = tda + tdi
        expected: pd.DatetimeIndex = tdi + tdi
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = tda - tdi
        expected: pd.DatetimeIndex = tdi - tdi
        tm.assert_index_equal(result, expected)

    def test_tda_add_dt64_object_array(self, performance_warning: bool, box_with_array: type, tz_naive_fixture: str) -> None:
        box: type = box_with_array
        dti: pd.DatetimeIndex = pd.date_range('2016-01-01', periods=3, tz=tz_naive_fixture)
        dti: Series = dti._with_freq(None)
        tdi: pd.DatetimeIndex = dti - dti
        obj: Series = tm.box_expected(tdi, box)
        other: Series = tm.box_expected(dti, box)
        with tm.assert_produces_warning(performance_warning):
            result: Series = obj + other.astype(object)
        tm.assert_equal(result, other.astype(object))

    def test_tdi_iadd_timedeltalike(self, two_hours: Timedelta, box_with_array: type) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days')
        expected: TimedeltaIndex = timedelta_range('1 days 02:00:00', '10 days 02:00:00', freq='D')
        rng: Series = tm.box_expected(rng, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        orig_rng: TimedeltaIndex = rng
        rng += two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            tm.assert_equal(orig_rng, expected)

    def test_tdi_isub_timedeltalike(self, two_hours: Timedelta, box_with_array: type) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days')
        expected: TimedeltaIndex = timedelta_range('0 days 22:00:00', '9 days 22:00:00')
        rng: Series = tm.box_expected(rng, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        orig_rng: TimedeltaIndex = rng
        rng -= two_hours
        tm.assert_equal(rng, expected)
        if box_with_array is not Index:
            tm.assert_equal(orig_rng, expected)

    def test_tdi_ops_attributes(self) -> None:
        rng: TimedeltaIndex = timedelta_range('2 days', periods=5, freq='2D', name='x')
        result: TimedeltaIndex = rng + 1 * rng.freq
        exp: TimedeltaIndex = timedelta_range('4 days', periods=5, freq='2D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == '2D'
        result: TimedeltaIndex = rng - 2 * rng.freq
        exp: TimedeltaIndex = timedelta_range('-2 days', periods=5, freq='2D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == '2D'
        result: TimedeltaIndex = rng * 2
        exp: TimedeltaIndex = timedelta_range('4 days', periods=5, freq='4D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == '4D'
        result: TimedeltaIndex = rng / 2
        exp: TimedeltaIndex = timedelta_range('1 days', periods=5, freq='D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == 'D'
        result: TimedeltaIndex = -rng
        exp: TimedeltaIndex = timedelta_range('-2 days', periods=5, freq='-2D', name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq == '-2D'
        rng: TimedeltaIndex = timedelta_range('-2 days', periods=5, freq='D', name='x')
        result: TimedeltaIndex = abs(rng)
        exp: TimedeltaIndex = TimedeltaIndex(['2 days', '1 days', '0 days', '1 days', '2 days'], name='x')
        tm.assert_index_equal(result, exp)
        assert result.freq is None

class TestAddSubNaTMasking:

    @pytest.mark.parametrize('str_ts', ['1950-01-01', '1980-01-01'])
    def test_tdarr_add_timestamp_nat_masking(self, box_with_array: type, str_ts: str) -> None:
        tdinat: pd.Series = pd.to_timedelta(['24658 days 11:15:00', 'NaT'])
        tdobj: Series = tm.box_expected(tdinat, box_with_array)
        ts: Timestamp = Timestamp(str_ts)
        ts_variants: list[Timestamp] = [ts, ts.to_pydatetime(), ts.to_datetime64().astype('datetime64[ns]'), ts.to_datetime64().astype('datetime64[D]')]
        for variant in ts_variants:
            res: Series = tdobj + variant
            if box_with_array is DataFrame:
                assert res.iloc[1, 1] is NaT
            else:
                assert res[1] is NaT

    def test_tdi_add_overflow(self) -> None:
        with pytest.raises(OutOfBoundsDatetime, match='10155196800000000000'):
            pd.to_timedelta(106580, 'D') + Timestamp('2000')
        with pytest.raises(OutOfBoundsDatetime, match='10155196800000000000'):
            Timestamp('2000') + pd.to_timedelta(106580, 'D')
        _NaT: np.timedelta64 = NaT._value + 1
        msg: str = 'Overflow in int64 addition'
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([106580], 'D') + Timestamp('2000')
        with pytest.raises(OverflowError, match=msg):
            Timestamp('2000') + pd.to_timedelta([106580], 'D')
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([_NaT]) - Timedelta('1 days')
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta(['5 days', _NaT]) - Timedelta('1 days')
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([_NaT, '5 days', '1 hours']) - pd.to_timedelta(['7 seconds', _NaT, '4 hours'])
        exp: TimedeltaIndex = TimedeltaIndex([NaT])
        result: TimedeltaIndex = pd.to_timedelta([NaT]) - Timedelta('1 days')
        tm.assert_index_equal(result, exp)
        exp: TimedeltaIndex = TimedeltaIndex(['4 days', NaT])
        result: TimedeltaIndex = pd.to_timedelta(['5 days', NaT]) - Timedelta('1 days')
        tm.assert_index_equal(result, exp)
        exp: TimedeltaIndex = TimedeltaIndex([NaT, NaT, '5 hours'])
        result: TimedeltaIndex = pd.to_timedelta([NaT, '5 days', '1 hours']) + pd.to_timedelta(['7 seconds', NaT, '4 hours'])
        tm.assert_index_equal(result, exp)

class TestTimedeltaArraylikeAddSubOps:

    def test_sub_nat_retain_unit(self) -> None:
        ser: Series = pd.to_timedelta(Series(['00:00:01'])).astype('m8[s]')
        result: Series = ser - NaT
        expected: Series = Series([NaT], dtype='m8[s]')
        tm.assert_series_equal(result, expected)

    def test_timedelta_ops_with_missing_values(self) -> None:
        s1: Series = pd.to_timedelta(Series(['00:00:01']))
        s2: Series = pd.to_timedelta(Series(['00:00:02']))
        sn: Series = pd.to_timedelta(Series([NaT], dtype='m8[ns]'))
        df1: DataFrame = DataFrame(['00:00:01']).apply(pd.to_timedelta)
        df2: DataFrame = DataFrame(['00:00:02']).apply(pd.to_timedelta)
        dfn: DataFrame = DataFrame([NaT._value]).apply(pd.to_timedelta)
        scalar1: Timedelta = pd.to_timedelta('00:00:01')
        scalar2: Timedelta = pd.to_timedelta('00:00:02')
        timedelta_NaT: np.timedelta64 = pd.to_timedelta('NaT')
        actual: Series = scalar1 + scalar1
        assert actual == scalar2
        actual: Series = scalar2 - scalar1
        assert actual == scalar1
        actual: Series = s1 + s1
        tm.assert_series_equal(actual, s2)
        actual: Series = s2 - s1
        tm.assert_series_equal(actual, s1)
        actual: Series = s1 + scalar1
        tm.assert_series_equal(actual, s2)
        actual: Series = scalar1 + s1
        tm.assert_series_equal(actual, s2)
        actual: Series = s2 - scalar1
        tm.assert_series_equal(actual, s1)
        actual: Series = -scalar1 + s2
        tm.assert_series_equal(actual, s1)
        actual: Series = s1 + timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual: Series = timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        actual: Series = s1 - timedelta_NaT
        tm.assert_series_equal(actual, sn)
        actual: Series = -timedelta_NaT + s1
        tm.assert_series_equal(actual, sn)
        msg: str = 'unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            s1 + np.nan
        with pytest.raises(TypeError, match=msg):
            np.nan + s1
        with pytest.raises(TypeError, match=msg):
            s1 - np.nan
        with pytest.raises(TypeError, match=msg):
            -np.nan + s1
        actual: Series = s1 + NaT
        tm.assert_series_equal(actual, sn)
        actual: Series = s2 - NaT
        tm.assert_series_equal(actual, sn)
        actual: DataFrame = s1 + df1
        tm.assert_frame_equal(actual, df2)
        actual: DataFrame = s2 - df1
        tm.assert_frame_equal(actual, df1)
        actual: DataFrame = df1 + s1
        tm.assert_frame_equal(actual, df2)
        actual: DataFrame = df2 - s1
        tm.assert_frame_equal(actual, df1)
        actual: DataFrame = df1 + df1
        tm.assert_frame_equal(actual, df2)
        actual: DataFrame = df2 - df1
        tm.assert_frame_equal(actual, df1)
        actual: DataFrame = df1 + scalar1
        tm.assert_frame_equal(actual, df2)
        actual: DataFrame = df2 - scalar1
        tm.assert_frame_equal(actual, df1)
        actual: DataFrame = df1 + timedelta_NaT
        tm.assert_frame_equal(actual, dfn)
        actual: DataFrame = df1 - timedelta_NaT
        tm.assert_frame_equal(actual, dfn)

    def test_operators_timedelta64(self) -> None:
        v1: pd.Series = pd.date_range('2012-1-1', periods=3, freq='D')
        v2: pd.Series = pd.date_range('2012-1-2', periods=3, freq='D')
        rs: Series = Series(v2) - Series(v1)
        xp: Series = Series(1000000000.0 * 3600 * 24, rs.index).astype('int64').astype('timedelta64[ns]')
        tm.assert_series_equal(rs, xp)
        assert rs.dtype == 'timedelta64[ns]'
        df: DataFrame = DataFrame({'A': v1})
        td: Series = Series([timedelta(days=i) for i in range(3)])
        assert td.dtype == 'timedelta64[ns]'
        result: Series = df['A'] - df['A'].shift()
        assert result.dtype == 'timedelta64[ns]'
        result: Series = df['A'] + td
        assert result.dtype == 'M8[ns]'
        maxa: Timestamp = df['A'].max()
        assert isinstance(maxa, Timestamp)
        resultb: Series = df['A'] - df['A'].max()
        assert resultb.dtype == 'timedelta64[ns]'
        result: Series = resultb + df['A']
        values: list[datetime] = [Timestamp('20111230'), Timestamp('20120101'), Timestamp('20120103')]
        expected: Series = Series(values, dtype='M8[ns]', name='A')
        tm.assert_series_equal(result, expected)
        result: Series = df['A'] - datetime(2001, 1, 1)
        expected: Series = Series([timedelta(days=4017 + i) for i in range(3)], name='A')
        tm.assert_series_equal(result, expected)
        assert result.dtype == 'm8[ns]'
        d: datetime = datetime(2001, 1, 1, 3, 4)
        resulta: Series = df['A'] - d
        assert resulta.dtype == 'm8[ns]'
        resultb: Series = resulta + d
        tm.assert_series_equal(df['A'], resultb)
        td: timedelta = timedelta(days=1)
        resulta: Series = df['A'] + td
        resultb: Series = resulta - td
        tm.assert_series_equal(resultb, df['A'])
        assert resultb.dtype == 'M8[ns]'
        td: timedelta = timedelta(minutes=5, seconds=3)
        resulta: Series = df['A'] + td
        resultb: Series = resulta - td
        tm.assert_series_equal(df['A'], resultb)
        assert resultb.dtype == 'M8[ns]'
        value: np.timedelta64 = rs[2] + np.timedelta64(timedelta(minutes=5, seconds=1))
        rs[2] += np.timedelta64(timedelta(minutes=5, seconds=1))
        assert rs[2] == value

    def test_timedelta64_ops_nat(self) -> None:
        timedelta_series: Series = Series([NaT, Timedelta('1s')])
        nat_series_dtype_timedelta: Series = Series([NaT, NaT], dtype='timedelta64[ns]')
        single_nat_dtype_timedelta: Series = Series([NaT], dtype='timedelta64[ns]')
        tm.assert_series_equal(timedelta_series - NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(-NaT + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series - single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(-single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + NaT, nat_series_dtype_timedelta)
        tm.assert_series_equal(NaT + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timedelta * 1.0, nat_series_dtype_timedelta)
        tm.assert_series_equal(1.0 * nat_series_dtype_timedelta, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series * 1, timedelta_series)
        tm.assert_series_equal(1 * timedelta_series, timedelta_series)
        tm.assert_series_equal(timedelta_series * 1.5, Series([NaT, Timedelta('1.5s')]))
        tm.assert_series_equal(1.5 * timedelta_series, Series([NaT, Timedelta('1.5s')]))
        tm.assert_series_equal(timedelta_series * np.nan, nat_series_dtype_timedelta)
        tm.assert_series_equal(np.nan * timedelta_series, nat_series_dtype_timedelta)
        tm.assert_series_equal(timedelta_series / 2, Series([NaT, Timedelta('0.5s')]))
        tm.assert_series_equal(timedelta_series / 2.0, Series([NaT, Timedelta('0.5s')]))
        tm.assert_series_equal(timedelta_series / np.nan, nat_series_dtype_timedelta)

    @pytest.mark.parametrize('cls', [Timestamp, datetime, np.datetime64])
    def test_td64arr_add_sub_datetimelike_scalar(self, cls: type[datetime | Timestamp | np.datetime64], box_with_array: type, tz_naive_fixture: str) -> None:
        tz: str = tz_naive_fixture
        dt_scalar: Timestamp = Timestamp('2012-01-01', tz=tz)
        if cls is datetime:
            ts: datetime = dt_scalar.to_pydatetime()
        elif cls is np.datetime64:
            if tz_naive_fixture is not None:
                pytest.skip(f'{cls} doesn support {tz_naive_fixture}')
            ts: np.datetime64 = dt_scalar.to_datetime64()
        else:
            ts: Timestamp = dt_scalar
        tdi: TimedeltaIndex = timedelta_range('1 day', periods=3)
        expected: pd.Series = pd.date_range('2012-01-02', periods=3, tz=tz)
        tdarr: Series = tm.box_expected(tdi, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(ts + tdarr, expected)
        tm.assert_equal(tdarr + ts, expected)
        expected2: pd.Series = pd.date_range('2011-12-31', periods=3, freq='-1D', tz=tz)
        expected2: Series = tm.box_expected(expected2, box_with_array)
        tm.assert_equal(ts - tdarr, expected2)
        tm.assert_equal(ts + -tdarr, expected2)
        msg: str = 'cannot subtract a datelike'
        with pytest.raises(TypeError, match=msg):
            tdarr - ts

    def test_td64arr_add_datetime64_nat(self, box_with_array: type) -> None:
        other: np.datetime64 = np.datetime64('NaT')
        tdi: TimedeltaIndex = timedelta_range('1 day', periods=3)
        expected: pd.Series = DatetimeIndex(['NaT', 'NaT', 'NaT'], dtype='M8[ns]')
        tdser: Series = tm.box_expected(tdi, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(tdser + other, expected)
        tm.assert_equal(other + tdser, expected)

    def test_td64arr_sub_dt64_array(self, box_with_array: type) -> None:
        dti: pd.Series = pd.date_range('2016-01-01', periods=3)
        tdi: TimedeltaIndex = TimedeltaIndex(['-1 Day'] * 3)
        dtarr: np.ndarray = dti.values
        expected: pd.Series = DatetimeIndex(dtarr) - tdi
        tdi: Series = tm.box_expected(tdi, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        msg: str = 'cannot subtract a datelike from'
        with pytest.raises(TypeError, match=msg):
            tdi - dtarr
        result: np.ndarray = dtarr - tdi
        tm.assert_equal(result, expected)

    def test_td64arr_add_dt64_array(self, box_with_array: type) -> None:
        dti: pd.Series = pd.date_range('2016-01-01', periods=3)
        tdi: TimedeltaIndex = TimedeltaIndex(['-1 Day'] * 3)
        dtarr: np.ndarray = dti.values
        expected: pd.Series = DatetimeIndex(dtarr) + tdi
        tdi: Series = tm.box_expected(tdi, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = tdi + dtarr
        tm.assert_equal(result, expected)
        result: Series = dtarr + tdi
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('tdi_freq', [None, 'h'])
    def test_td64arr_sub_periodlike(self, box_with_array: type, box_with_array2: type, tdi_freq: str | None, pi_freq: str) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex(['1 hours', '2 hours'], freq=tdi_freq)
        dti: pd.Series = Timestamp('2018-03-07 17:16:40') + tdi
        pi: pd.Series = dti.to_period(pi_freq)
        per: pd.Period = pi[0]
        tdi: Series = tm.box_expected(tdi, box_with_array)
        pi: Series = tm.box_expected(pi, box_with_array2)
        msg: str = 'cannot subtract|unsupported operand type'
        with pytest.raises(TypeError, match=msg):
            tdi - pi
        with pytest.raises(TypeError, match=msg):
            tdi - per

    @pytest.mark.parametrize('other', ['a', 1, 1.5, np.array(2)])
    def test_td64arr_addsub_numeric_scalar_invalid(self, box_with_array: type, other: int | float | np.ndarray | str) -> None:
        tdser: Series = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        tdarr: Series = tm.box_expected(tdser, box_with_array)
        assert_invalid_addsub_type(tdarr, other)

    @pytest.mark.parametrize('vec', [np.array([1, 2, 3]), Index([1, 2, 3]), Series([1, 2, 3]), DataFrame([[1, 2, 3]])], ids=lambda x: type(x).__name__)
    def test_td64arr_addsub_numeric_arr_invalid(self, box_with_array: type, vec: np.ndarray | Index | Series | DataFrame, any_real_numpy_dtype: type[np.number]) -> None:
        tdser: Series = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        tdarr: Series = tm.box_expected(tdser, box_with_array)
        vector: np.ndarray | Index | Series | DataFrame = vec.astype(any_real_numpy_dtype)
        assert_invalid_addsub_type(tdarr, vector)

    def test_td64arr_add_sub_int(self, box_with_array: type, one: int) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days 09:00:00', freq='h', periods=10)
        tdarr: Series = tm.box_expected(rng, box_with_array)
        msg: str = 'Addition/subtraction of integers'
        assert_invalid_addsub_type(tdarr, one, msg)
        with pytest.raises(TypeError, match=msg):
            tdarr += one
        with pytest.raises(TypeError, match=msg):
            tdarr -= one

    def test_td64arr_add_sub_integer_array(self, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        rng: TimedeltaIndex = timedelta_range('1 days 09:00:00', freq='h', periods=3)
        tdarr: Series = tm.box_expected(rng, box)
        other: Series = tm.box_expected([4, 3, 2], xbox)
        msg: str = 'Addition/subtraction of integers and integer-arrays'
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_addsub_integer_array_no_freq(self, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        tdi: TimedeltaIndex = TimedeltaIndex(['1 Day', 'NaT', '3 Hours'])
        tdarr: Series = tm.box_expected(tdi, box)
        other: Series = tm.box_expected([14, -1, 16], xbox)
        msg: str = 'Addition/subtraction of integers'
        assert_invalid_addsub_type(tdarr, other, msg)

    def test_td64arr_add_sub_td64_array(self, box_with_array: type) -> None:
        box: type = box_with_array
        dti: pd.Series = pd.date_range('2016-01-01', periods=3)
        tdi: TimedeltaIndex = dti - dti.shift(1)
        tdarr: np.ndarray = tdi.values
        expected: TimedeltaIndex = 2 * tdi
        tdi: Series = tm.box_expected(tdi, box)
        expected: Series = tm.box_expected(expected, box)
        result: Series = tdi + tdarr
        tm.assert_equal(result, expected)
        result: Series = tdarr + tdi
        tm.assert_equal(result, expected)
        expected_sub: TimedeltaIndex = 0 * tdi
        result: Series = tdi - tdarr
        tm.assert_equal(result, expected_sub)
        result: Series = tdarr - tdi
        tm.assert_equal(result, expected_sub)

    def test_td64arr_add_sub_tdi(self, box_with_array: type, names: list[str]) -> None:
        box: type = box_with_array
        exname: str = get_expected_name(box, names)
        tdi: TimedeltaIndex = TimedeltaIndex(['0 days', '1 day'], name=names[1])
        tdi: np.ndarray = tdi if box in [tm.to_array, pd.array] else tdi
        ser: Series = Series([Timedelta(hours=3), Timedelta(hours=4)], name=names[0])
        expected: Series = Series([Timedelta(hours=3), Timedelta(days=1, hours=4)], name=exname)
        ser: Series = tm.box_expected(ser, box)
        expected: Series = tm.box_expected(expected, box)
        result: Series = tdi + ser
        tm.assert_equal(result, expected)
        assert result.dtype == 'timedelta64[ns]'
        result: Series = ser + tdi
        tm.assert_equal(result, expected)
        assert result.dtype == 'timedelta64[ns]'
        expected: Series = Series([Timedelta(hours=-3), Timedelta(days=1, hours=-4)], name=exname)
        expected: Series = tm.box_expected(expected, box)
        result: Series = tdi - ser
        tm.assert_equal(result, expected)
        assert result.dtype == 'timedelta64[ns]'
        result: Series = ser - tdi
        tm.assert_equal(result, -expected)
        assert result.dtype == 'timedelta64[ns]'

    @pytest.mark.parametrize('tdnat', [np.timedelta64('NaT'), NaT])
    def test_td64arr_add_sub_td64_nat(self, box_with_array: type, tdnat: np.timedelta64 | NaT) -> None:
        box: type = box_with_array
        tdi: TimedeltaIndex = TimedeltaIndex([NaT, Timedelta('1s')])
        expected: TimedeltaIndex = TimedeltaIndex(['NaT'] * 2)
        obj: Series = tm.box_expected(tdi, box)
        expected: Series = tm.box_expected(expected, box)
        result: Series = obj + tdnat
        tm.assert_equal(result, expected)
        result: Series = tdnat + obj
        tm.assert_equal(result, expected)
        result: Series = obj - tdnat
        tm.assert_equal(result, expected)
        result: Series = tdnat - obj
        tm.assert_equal(result, expected)

    def test_td64arr_add_timedeltalike(self, two_hours: Timedelta, box_with_array: type) -> None:
        box: type = box_with_array
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days')
        expected: TimedeltaIndex = timedelta_range('1 days 02:00:00', '10 days 02:00:00', freq='D')
        rng: Series = tm.box_expected(rng, box)
        expected: Series = tm.box_expected(expected, box)
        result: Series = rng + two_hours
        tm.assert_equal(result, expected)
        result: Series = two_hours + rng
        tm.assert_equal(result, expected)

    def test_td64arr_sub_timedeltalike(self, two_hours: Timedelta, box_with_array: type) -> None:
        box: type = box_with_array
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days')
        expected: TimedeltaIndex = timedelta_range('0 days 22:00:00', '9 days 22:00:00')
        rng: Series = tm.box_expected(rng, box)
        expected: Series = tm.box_expected(expected, box)
        result: Series = rng - two_hours
        tm.assert_equal(result, expected)
        result: Series = two_hours - rng
        tm.assert_equal(result, -expected)

    def test_td64arr_add_sub_offset_index(self, performance_warning: bool, names: list[str], box_with_array: type) -> None:
        box: type = box_with_array
        exname: str = get_expected_name(box, names)
        tdi: TimedeltaIndex = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'], name=names[0])
        other: Index = Index([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        other: np.ndarray = other if box in [tm.to_array, pd.array] else other
        expected: TimedeltaIndex = TimedeltaIndex([tdi[n] + other[n] for n in range(len(tdi))], freq='infer', name=exname)
        expected_sub: TimedeltaIndex = TimedeltaIndex([tdi[n] - other[n] for n in range(len(tdi))], freq='infer', name=exname)
        tdi: Series = tm.box_expected(tdi, box)
        expected: Series = tm.box_expected(expected, box).astype(object)
        expected_sub: Series = tm.box_expected(expected_sub, box).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res: Series = tdi + other
        tm.assert_equal(res, expected)
        with tm.assert_produces_warning(performance_warning):
            res2: Series = other + tdi
        tm.assert_equal(res2, expected)
        with tm.assert_produces_warning(performance_warning):
            res_sub: Series = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_add_sub_offset_array(self, performance_warning: bool, box_with_array: type) -> None:
        box: type = box_with_array
        tdi: TimedeltaIndex = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'])
        other: np.ndarray = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])
        expected: TimedeltaIndex = TimedeltaIndex([tdi[n] + other[n] for n in range(len(tdi))], freq='infer')
        expected_sub: TimedeltaIndex = TimedeltaIndex([tdi[n] - other[n] for n in range(len(tdi))], freq='infer')
        tdi: Series = tm.box_expected(tdi, box)
        expected: Series = tm.box_expected(expected, box).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res: Series = tdi + other
        tm.assert_equal(res, expected)
        with tm.assert_produces_warning(performance_warning):
            res2: Series = other + tdi
        tm.assert_equal(res2, expected)
        expected_sub: Series = tm.box_expected(expected_sub, box_with_array).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res_sub: Series = tdi - other
        tm.assert_equal(res_sub, expected_sub)

    def test_td64arr_with_offset_series(self, performance_warning: bool, names: list[str], box_with_array: type) -> None:
        box: type = box_with_array
        box2: type = Series if box in [Index, tm.to_array, pd.array] else box
        exname: str = get_expected_name(box, names)
        tdi: TimedeltaIndex = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'], name=names[0])
        other: Series = Series([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        expected_add: Series = Series([tdi[n] + other[n] for n in range(len(tdi))], name=exname, dtype=object)
        obj: Series = tm.box_expected(tdi, box)
        expected_add: Series = tm.box_expected(expected_add, box2).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res: Series = obj + other
        tm.assert_equal(res, expected_add)
        with tm.assert_produces_warning(performance_warning):
            res2: Series = other + obj
        tm.assert_equal(res2, expected_add)
        expected_sub: Series = Series([tdi[n] - other[n] for n in range(len(tdi))], name=exname, dtype=object)
        expected_sub: Series = tm.box_expected(expected_sub, box2).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res3: Series = obj - other
        tm.assert_equal(res3, expected_sub)

    @pytest.mark.parametrize('obox', [np.array, Index, Series])
    def test_td64arr_addsub_anchored_offset_arraylike(self, performance_warning: bool, obox: type[np.ndarray | Index | Series], box_with_array: type) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex(['1 days 00:00:00', '3 days 04:00:00'])
        tdi: Series = tm.box_expected(tdi, box_with_array)
        anchored: np.ndarray | Index | Series = obox([offsets.MonthEnd(), offsets.Day(n=2)])
        msg: str = 'has incorrect type|cannot add the type MonthEnd'
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdi + anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                anchored + tdi
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdi - anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                anchored - tdi

    def test_td64arr_add_sub_object_array(self, performance_warning: bool, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        tdi: TimedeltaIndex = timedelta_range('1 day', periods=3, freq='D')
        tdarr: Series = tm.box_expected(tdi, box)
        other: np.ndarray = np.array([Timedelta(days=1), offsets.Day(2), Timestamp('2000-01-04')])
        with tm.assert_produces_warning(performance_warning):
            result: Series = tdarr + other
        expected: Index = Index([Timedelta(days=2), Timedelta(days=4), Timestamp('2000-01-07')])
        expected: Series = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)
        msg: str = 'unsupported operand type|cannot subtract a datelike'
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                tdarr - other
        with tm.assert_produces_warning(performance_warning):
            result: Series = other - tdarr
        expected: Index = Index([Timedelta(0), Timedelta(0), Timestamp('2000-01-01')])
        expected: Series = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(result, expected)

class TestTimedeltaArraylikeMulDivOps:

    def test_td64arr_mul_int(self, box_with_array: type) -> None:
        idx: TimedeltaIndex = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx: Series = tm.box_expected(idx, box_with_array)
        result: Series = idx * 1
        tm.assert_equal(result, idx)
        result: Series = 1 * idx
        tm.assert_equal(result, idx)

    def test_td64arr_mul_tdlike_scalar_raises(self, two_hours: Timedelta, box_with_array: type) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days', name='foo')
        rng: Series = tm.box_expected(rng, box_with_array)
        msg: str = '|'.join(['argument must be an integer', 'cannot use operands with types dtype', 'Cannot multiply with'])
        with pytest.raises(TypeError, match=msg):
            rng * two_hours

    def test_tdi_mul_int_array_zerodim(self, box_with_array: type) -> None:
        rng5: np.ndarray = np.arange(5, dtype='int64')
        idx: TimedeltaIndex = TimedeltaIndex(rng5)
        expected: TimedeltaIndex = TimedeltaIndex(rng5 * 5)
        idx: Series = tm.box_expected(idx, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = idx * np.array(5, dtype='int64')
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_array(self, box_with_array: type) -> None:
        rng5: np.ndarray = np.arange(5, dtype='int64')
        idx: TimedeltaIndex = TimedeltaIndex(rng5)
        expected: TimedeltaIndex = TimedeltaIndex(rng5 ** 2)
        idx: Series = tm.box_expected(idx, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = idx * rng5
        tm.assert_equal(result, expected)

    def test_tdi_mul_int_series(self, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = Series if box in [Index, tm.to_array, pd.array] else box
        idx: TimedeltaIndex = TimedeltaIndex(np.arange(5, dtype='int64'))
        expected: TimedeltaIndex = TimedeltaIndex(np.arange(5, dtype='int64') ** 2)
        idx: Series = tm.box_expected(idx, box)
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = idx * Series(np.arange(5, dtype='int64'))
        tm.assert_equal(result, expected)

    def test_tdi_mul_float_series(self, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = Series if box in [Index, tm.to_array, pd.array] else box
        idx: TimedeltaIndex = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx: Series = tm.box_expected(idx, box)
        rng5f: np.ndarray = np.arange(5, dtype='float64')
        expected: TimedeltaIndex = TimedeltaIndex(rng5f * (rng5f + 1.0))
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = idx * Series(rng5f + 1.0)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('other', [np.arange(1, 11), Index(np.arange(1, 11), np.int64), Index(range(1, 11), np.uint64), Index(range(1, 11), np.float64), pd.RangeIndex(1, 11)])
    def test_tdi_rmul_arraylike(self, other: np.ndarray | Index | Series | pd.RangeIndex, box_with_array: type) -> None:
        box: type = box_with_array
        tdi: TimedeltaIndex = TimedeltaIndex(['1 Day'] * 10)
        expected: TimedeltaIndex = timedelta_range('1 days', '10 days')._with_freq(None)
        tdi: Series = tm.box_expected(tdi, box)
        xbox: type = get_upcast_box(tdi, other)
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = other * tdi
        tm.assert_equal(result, expected)
        commute: Series = tdi * other
        tm.assert_equal(commute, expected)

    def test_td64arr_div_nat_invalid(self, box_with_array: type) -> None:
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days', name='foo')
        rng: Series = tm.box_expected(rng, box_with_array)
        with pytest.raises(TypeError, match='unsupported operand type'):
            rng / NaT
        with pytest.raises(TypeError, match='Cannot divide NaTType by'):
            NaT / rng
        dt64nat: np.datetime64 = np.datetime64('NaT', 'ns')
        msg: str = '|'.join(["ufunc '(true_divide|divide)' cannot use operands", 'cannot perform __r?truediv__', 'Cannot divide datetime64 by TimedeltaArray'])
        with pytest.raises(TypeError, match=msg):
            rng / dt64nat
        with pytest.raises(TypeError, match=msg):
            dt64nat / rng

    def test_td64arr_div_td64nat(self, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days')
        rng: Series = tm.box_expected(rng, box)
        other: np.timedelta64 = np.timedelta64('NaT')
        expected: np.ndarray = np.array([np.nan] * 10)
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = rng / other
        tm.assert_equal(result, expected)
        result: Series = other / rng
        tm.assert_equal(result, expected)

    def test_td64arr_div_int(self, box_with_array: type) -> None:
        idx: TimedeltaIndex = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx: Series = tm.box_expected(idx, box_with_array)
        result: Series = idx / 1
        tm.assert_equal(result, idx)
        with pytest.raises(TypeError, match='Cannot divide'):
            1 / idx

    def test_td64arr_div_tdlike_scalar(self, two_hours: Timedelta, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        rng: TimedeltaIndex = timedelta_range('1 days', '10 days', name='foo')
        expected: Index = (np.arange(10) + 1) * 12
        expected: Series = tm.box_expected(expected, xbox)
        rng: Series = tm.box_expected(rng, box)
        result: Series = rng / two_hours
        tm.assert_equal(result, expected)
        result: Series = two_hours / rng
        expected: Series = 1 / expected
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('m', [1, 3, 10])
    @pytest.mark.parametrize('unit', ['D', 'h', 'm', 's', 'ms', 'us', 'ns'])
    def test_td64arr_div_td64_scalar(self, m: int, unit: str, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        ser: Series = Series([Timedelta(days=59)] * 3)
        ser[2] = np.nan
        flat: Series = ser
        ser: Series = tm.box_expected(ser, box)
        expected: Series = Series([x / np.timedelta64(m, unit) for x in flat])
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = ser / np.timedelta64(m, unit)
        tm.assert_equal(result, expected)
        expected: Series = Series([Timedelta(np.timedelta64(m, unit)) / x for x in flat])
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = np.timedelta64(m, unit) / ser
        tm.assert_equal(result, expected)

    def test_td64arr_div_tdlike_scalar_with_nat(self, two_hours: Timedelta, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        rng: TimedeltaIndex = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        expected: Index = [12, np.nan, 24]
        expected: Series = tm.box_expected(expected, xbox)
        rng: Series = tm.box_expected(rng, box)
        result: Series = rng / two_hours
        tm.assert_equal(result, expected)
        result: Series = two_hours / rng
        expected: Series = 1 / expected
        tm.assert_equal(result, expected)

    def test_td64arr_div_td64_ndarray(self, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        rng: TimedeltaIndex = TimedeltaIndex(['1 days', NaT, '2 days'])
        rng: Series = tm.box_expected(rng, box)
        other: np.ndarray = np.array([2, 4, 2], dtype='m8[h]')
        expected: Index = [12, np.nan, 24]
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = rng / other
        tm.assert_equal(result, expected)
        result: Series = rng / tm.box_expected(other, box)
        tm.assert_equal(result, expected)
        result: Series = rng / other.astype(object)
        tm.assert_equal(result, expected.astype(object))
        result: Series = rng / list(other)
        tm.assert_equal(result, expected)
        expected: Series = 1 / expected
        result: Series = other / rng
        tm.assert_equal(result, expected)
        result: Series = tm.box_expected(other, box) / rng
        tm.assert_equal(result, expected)
        result: Series = other.astype(object) / rng
        tm.assert_equal(result, expected)
        result: Series = list(other) / rng
        tm.assert_equal(result, expected)

    def test_tdarr_div_length_mismatch(self, box_with_array: type) -> None:
        rng: TimedeltaIndex = TimedeltaIndex(['1 days', NaT, '2 days'])
        mismatched: list[int] = [1, 2, 3, 4]
        rng: Series = tm.box_expected(rng, box_with_array)
        msg: str = 'Cannot divide vectors|Unable to coerce to Series'
        for obj in [mismatched, mismatched[:2]]:
            for other in [obj, np.array(obj), Index(obj)]:
                with pytest.raises(ValueError, match=msg):
                    rng / other
                with pytest.raises(ValueError, match=msg):
                    other / rng

    def test_td64_div_object_mixed_result(self, box_with_array: type) -> None:
        orig: TimedeltaIndex = timedelta_range('1 Day', periods=3).insert(1, NaT)
        tdi: Series = tm.box_expected(orig, box_with_array, transpose=False)
        other: np.ndarray = np.array([orig[0], 1.5, 2.0, orig[2]], dtype=object)
        other: Series = tm.box_expected(other, box_with_array, transpose=False)
        res: Series = tdi / other
        expected: Index = [1.0, np.timedelta64('NaT', 'ns'), orig[0], 1.5]
        expected: Series = tm.box_expected(expected, box_with_array, transpose=False)
        if isinstance(expected, NumpyExtensionArray):
            expected: np.ndarray = expected.to_numpy()
        tm.assert_equal(res, expected)
        if box_with_array is DataFrame:
            assert isinstance(res.iloc[1, 0], np.timedelta64)
        res: Series = tdi // other
        expected: Index = [1, np.timedelta64('NaT', 'ns'), orig[0], 1]
        expected: Series = tm.box_expected(expected, box_with_array, transpose=False)
        if isinstance(expected, NumpyExtensionArray):
            expected: np.ndarray = expected.to_numpy()
        tm.assert_equal(res, expected)
        if box_with_array is DataFrame:
            assert isinstance(res.iloc[1, 0], np.timedelta64)

    @pytest.mark.skipif(WASM, reason='no fp exception support in wasm')
    def test_td64arr_floordiv_td64arr_with_nat(self, box_with_array: type) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        left: Series = Series([1000, 222330, 30], dtype='timedelta64[ns]')
        right: Series = Series([1000, 222330, None], dtype='timedelta64[ns]')
        left: Series = tm.box_expected(left, box)
        right: Series = tm.box_expected(right, box)
        expected: np.ndarray = np.array([1.0, 1.0, np.nan], dtype=np.float64)
        expected: Series = tm.box_expected(expected, xbox)
        with tm.maybe_produces_warning(RuntimeWarning, box is pd.array, check_stacklevel=False):
            result: Series = left // right
        tm.assert_equal(result, expected)
        with tm.maybe_produces_warning(RuntimeWarning, box is pd.array, check_stacklevel=False):
            result: np.ndarray = np.asarray(left) // right
        tm.assert_equal(result, expected)

    @pytest.mark.filterwarnings('ignore:invalid value encountered:RuntimeWarning')
    def test_td64arr_floordiv_tdscalar(self, box_with_array: type, scalar_td: Timedelta) -> None:
        box: type = box_with_array
        xbox: type = np.ndarray if box is pd.array else box
        td: Timedelta = Timedelta('5m3s')
        td1: Series = Series([td, td, NaT], dtype='m8[ns]')
        td1: Series = tm.box_expected(td1, box, transpose=False)
        expected: Series = Series([0, 0, np.nan])
        expected: Series = tm.box_expected(expected, xbox, transpose=False)
        result: Series = td1 // scalar_td
        tm.assert_equal(result, expected)
        expected: Series = Series([2, 2, np.nan])
        expected: Series = tm.box_expected(expected, xbox, transpose=False)
        result: Series = scalar_td // td1
        tm.assert_equal(result, expected)
        result: Series = td1.__rfloordiv__(scalar_td)
        tm.assert_equal(result, expected)

    def test_td64arr_floordiv_int(self, box_with_array: type) -> None:
        idx: TimedeltaIndex = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx: Series = tm.box_expected(idx, box_with_array)
        result: Series = idx // 1
        tm.assert_equal(result, idx)
        pattern: str = 'floor_divide cannot use operands|Cannot divide int by Timedelta*'
        with pytest.raises(TypeError, match=pattern):
            1 // idx

    def test_td64arr_mod_tdscalar(self, performance_warning: bool, box_with_array: type, three_days: Timedelta) -> None:
        tdi: TimedeltaIndex = timedelta_range('1 Day', '9 days')
        tdarr: Series = tm.box_expected(tdi, box_with_array)
        expected: TimedeltaIndex = TimedeltaIndex(['1 Day', '2 Days', '0 Days'] * 3)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = tdarr % three_days
        tm.assert_equal(result, expected)
        if box_with_array is DataFrame and isinstance(three_days, pd.DateOffset):
            expected: Series = expected.astype(object)
        else:
            performance_warning: bool = False
        with tm.assert_produces_warning(performance_warning):
            result: tuple[Series, Series] = divmod(tdarr, three_days)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // three_days)

    def test_td64arr_mod_int(self, box_with_array: type) -> None:
        tdi: TimedeltaIndex = timedelta_range('1 ns', '10 ns', periods=10)
        tdarr: Series = tm.box_expected(tdi, box_with_array)
        expected: TimedeltaIndex = TimedeltaIndex(['1 ns', '0 ns'] * 5)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = tdarr % 2
        tm.assert_equal(result, expected)
        msg: str = 'Cannot divide int by'
        with pytest.raises(TypeError, match=msg):
            2 % tdarr
        result: tuple[Series, Series] = divmod(tdarr, 2)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], tdarr // 2)

    def test_td64arr_rmod_tdscalar(self, box_with_array: type, three_days: Timedelta) -> None:
        tdi: TimedeltaIndex = timedelta_range('1 Day', '9 days')
        tdarr: Series = tm.box_expected(tdi, box_with_array)
        expected: TimedeltaIndex = ['0 Days', '1 Day', '0 Days'] + ['3 Days'] * 6
        expected: TimedeltaIndex = TimedeltaIndex(expected)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = three_days % tdarr
        tm.assert_equal(result, expected)
        result: tuple[Series, Series] = divmod(three_days, tdarr)
        tm.assert_equal(result[1], expected)
        tm.assert_equal(result[0], three_days // tdarr)

    def test_td64arr_mul_tdscalar_invalid(self, box_with_array: type, scalar_td: Timedelta) -> None:
        td1: Series = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan
        td1: Series = tm.box_expected(td1, box_with_array)
        pattern: str = 'operate|unsupported|cannot|not supported'
        with pytest.raises(TypeError, match=pattern):
            td1 * scalar_td
        with pytest.raises(TypeError, match=pattern):
            scalar_td * td1

    def test_td64arr_mul_too_short_raises(self, box_with_array: type) -> None:
        idx: TimedeltaIndex = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx: Series = tm.box_expected(idx, box_with_array)
        msg: str = '|'.join(['cannot use operands with types dtype', 'Cannot multiply with unequal lengths', 'Unable to coerce to Series'])
        with pytest.raises(TypeError, match=msg):
            idx * idx[:3]
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    def test_td64arr_mul_td64arr_raises(self, box_with_array: type) -> None:
        idx: TimedeltaIndex = TimedeltaIndex(np.arange(5, dtype='int64'))
        idx: Series = tm.box_expected(idx, box_with_array)
        msg: str = 'cannot use operands with types dtype'
        with pytest.raises(TypeError, match=msg):
            idx * idx

    def test_td64arr_mul_numeric_scalar(self, box_with_array: type, one: int) -> None:
        tdser: Series = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        expected: Series = Series(['-59 Days', '-59 Days', 'NaT'], dtype='timedelta64[ns]')
        tdser: Series = tm.box_expected(tdser, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = tdser * -one
        tm.assert_equal(result, expected)
        result: Series = -one * tdser
        tm.assert_equal(result, expected)
        expected: Series = Series(['118 Days', '118 Days', 'NaT'], dtype='timedelta64[ns]')
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = tdser * (2 * one)
        tm.assert_equal(result, expected)
        result: Series = 2 * one * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('two', [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_div_numeric_scalar(self, box_with_array: type, two: int | float | np.ndarray | np.ndarray) -> None:
        tdser: Series = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        expected: Series = Series(['29.5D', '29.5D', 'NaT'], dtype='timedelta64[ns]')
        tdser: Series = tm.box_expected(tdser, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = tdser / two
        tm.assert_equal(result, expected)
        with pytest.raises(TypeError, match='Cannot divide'):
            two / tdser

    @pytest.mark.parametrize('two', [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_floordiv_numeric_scalar(self, box_with_array: type, two: int | float | np.ndarray | np.ndarray) -> None:
        tdser: Series = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        expected: Series = Series(['29.5D', '29.5D', 'NaT'], dtype='timedelta64[ns]')
        tdser: Series = tm.box_expected(tdser, box_with_array)
        expected: Series = tm.box_expected(expected, box_with_array)
        result: Series = tdser // two
        tm.assert_equal(result, expected)
        with pytest.raises(TypeError, match='Cannot divide'):
            two // tdser

    @pytest.mark.parametrize('klass', [np.array, Index, Series])
    def test_td64arr_rmul_numeric_array(self, box_with_array: type, klass: type[np.ndarray | Index | Series], any_real_numpy_dtype: type[np.number]) -> None:
        vector: np.ndarray | Index | Series = klass([20, 30, 40])
        vector: np.ndarray | Index | Series = vector.astype(any_real_numpy_dtype)
        tdser: Series = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        expected: Series = Series(['1180 Days', '1770 Days', 'NaT'], dtype='timedelta64[ns]')
        tdser: Series = tm.box_expected(tdser, box_with_array)
        xbox: type = get_upcast_box(tdser, vector)
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = tdser * vector
        tm.assert_equal(result, expected)
        result: Series = vector * tdser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('klass', [np.array, Index, Series])
    def test_td64arr_div_numeric_array(self, box_with_array: type, klass: type[np.ndarray | Index | Series], any_real_numpy_dtype: type[np.number]) -> None:
        vector: np.ndarray | Index | Series = klass([20, 30, 40])
        vector: np.ndarray | Index | Series = vector.astype(any_real_numpy_dtype)
        tdser: Series = Series(['59 Days', '59 Days', 'NaT'], dtype='m8[ns]')
        expected: Series = Series(['2.95D', '1D 23h 12m', 'NaT'], dtype='timedelta64[ns]')
        tdser: Series = tm.box_expected(tdser, box_with_array)
        xbox: type = get_upcast_box(tdser, vector)
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = tdser / vector
        tm.assert_equal(result, expected)
        pattern: str = '|'.join(["true_divide'? cannot use operands", 'cannot perform __div__', 'cannot perform __truediv__', 'unsupported operand', 'Cannot divide', "ufunc 'divide' cannot use operands with types"])
        with pytest.raises(TypeError, match=pattern):
            vector / tdser
        result: Series = tdser / vector.astype(object)
        if box_with_array is DataFrame:
            expected: list[np.timedelta64] = [tdser.iloc[0, n] / vector[n] for n in range(len(vector))]
            expected: Series = tm.box_expected(expected, xbox).astype(object)
            expected[2] = expected[2].fillna(np.timedelta64('NaT', 'ns'))
        else:
            expected: list[np.timedelta64] = [tdser[n] / vector[n] for n in range(len(tdser))]
            expected: list[np.timedelta64] = [x if x is not NaT else np.timedelta64('NaT', 'ns') for x in expected]
            if xbox is tm.to_array:
                expected: Series = tm.to_array(expected).astype(object)
            else:
                expected: Series = xbox(expected, dtype=object)
        tm.assert_equal(result, expected)
        with pytest.raises(TypeError, match=pattern):
            vector.astype(object) / tdser

    def test_td64arr_mul_int_series(self, box_with_array: type, names: list[str]) -> None:
        box: type = box_with_array
        exname: str = get_expected_name(box, names)
        tdi: TimedeltaIndex = TimedeltaIndex(['0days', '1day', '2days', '3days', '4days'], name=names[0])
        ser: Series = Series([0, 1, 2, 3, 4], dtype=np.int64, name=names[1])
        expected: Series = Series(['0days', '1day', '4days', '9days', '16days'], dtype='timedelta64[ns]', name=exname)
        tdi: Series = tm.box_expected(tdi, box)
        xbox: type = get_upcast_box(tdi, ser)
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = ser * tdi
        tm.assert_equal(result, expected)
        result: Series = tdi * ser
        tm.assert_equal(result, expected)

    def test_float_series_rdiv_td64arr(self, box_with_array: type, names: list[str]) -> None:
        box: type = box_with_array
        tdi: TimedeltaIndex = TimedeltaIndex(['0days', '1day', '2days', '3days', '4days'], name=names[0])
        ser: Series = Series([1.5, 3, 4.5, 6, 7.5], dtype=np.float64, name=names[1])
        xname: str = names[2] if box not in [tm.to_array, pd.array] else names[1]
        expected: Series = Series([tdi[n] / ser[n] for n in range(len(ser))], dtype='timedelta64[ns]', name=xname)
        tdi: Series = tm.box_expected(tdi, box)
        xbox: type = get_upcast_box(tdi, ser)
        expected: Series = tm.box_expected(expected, xbox)
        result: Series = ser.__rtruediv__(tdi)
        if box is DataFrame:
            assert result is NotImplemented
        else:
            tm.assert_equal(result, expected)

    def test_td64arr_all_nat_div_object_dtype_numeric(self, box_with_array: type) -> None:
        tdi: TimedeltaIndex = TimedeltaIndex([NaT, NaT])
        left: Series = tm.box_expected(tdi, box_with_array)
        right: np.ndarray = np.array([2, 2.0], dtype=object)
        tdnat: np.timedelta64 = np.timedelta64('NaT', 'ns')
        expected: Index = [tdnat] * 2
        if box_with_array is not Index:
            expected: Series = tm.box_expected(expected, box_with_array).astype(object)
            if box_with_array in [Series, DataFrame]:
                expected: Series = expected.fillna(tdnat)
        result: Series = left / right
        tm.assert_equal(result, expected)
        result: Series = left // right
        tm.assert_equal(result, expected)

class TestTimedelta64ArrayLikeArithmetic:

    def test_td64arr_pow_invalid(self, scalar_td: Timedelta, box_with_array: type) -> None:
        td1: Series = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan
        td1: Series = tm.box_expected(td1, box_with_array)
        pattern: str = 'operate|unsupported|cannot|not supported'
        with pytest.raises(TypeError, match=pattern):
            scalar_td ** td1
        with pytest.raises(TypeError, match=pattern):
            td1 ** scalar_td

def test_add_timestamp_to_timedelta() -> None:
    timestamp: Timestamp = Timestamp('2021-01-01')
    result: pd.Series = timestamp + timedelta_range('0s', '1s', periods=31)
    expected: pd.Series = DatetimeIndex([timestamp + (pd.to_timedelta('0.033333333s') * i + pd.to_timedelta('0.000000001s') * divmod(i, 3)[0]) for i in range(31)])
    tm.assert_index_equal(result, expected)
