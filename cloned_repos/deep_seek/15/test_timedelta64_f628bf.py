from datetime import datetime, timedelta
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pytest
from pandas.compat import WASM
from pandas.errors import OutOfBoundsDatetime
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, NaT, Series, Timedelta, TimedeltaIndex, Timestamp, offsets, timedelta_range
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import assert_invalid_addsub_type, assert_invalid_comparison, get_upcast_box

def assert_dtype(obj: Any, expected_dtype: str) -> None:
    """
    Helper to check the dtype for a Series, Index, or single-column DataFrame.
    """
    dtype = tm.get_dtype(obj)
    assert dtype == expected_dtype

def get_expected_name(box: Any, names: Sequence[str]) -> str:
    if box is DataFrame:
        exname = names[0]
    elif box in [tm.to_array, pd.array]:
        exname = names[1]
    else:
        exname = names[2]
    return exname

class TestTimedelta64ArrayLikeComparisons:

    def test_compare_timedelta64_zerodim(self, box_with_array: Any) -> None:
        box = box_with_array
        xbox = box_with_array if box_with_array not in [Index, pd.array] else np.ndarray
        tdi = timedelta_range('2h', periods=4)
        other = np.array(tdi.to_numpy()[0])
        tdi = tm.box_expected(tdi, box)
        res = tdi <= other
        expected = np.array([True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize('td_scalar', [timedelta(days=1), Timedelta(days=1), Timedelta(days=1).to_timedelta64(), offsets.Hour(24)])
    def test_compare_timedeltalike_scalar(self, box_with_array: Any, td_scalar: Any) -> None:
        box = box_with_array
        xbox = box if box not in [Index, pd.array] else np.ndarray
        ser = Series([timedelta(days=1), timedelta(days=2)])
        ser = tm.box_expected(ser, box)
        actual = ser > td_scalar
        expected = Series([False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(actual, expected)

    @pytest.mark.parametrize('invalid', [345600000000000, 'a', Timestamp('2021-01-01'), Timestamp('2021-01-01').now('UTC'), Timestamp('2021-01-01').now().to_datetime64(), Timestamp('2021-01-01').now().to_pydatetime(), Timestamp('2021-01-01').date(), np.array(4)])
    def test_td64_comparisons_invalid(self, box_with_array: Any, invalid: Any) -> None:
        box = box_with_array
        rng = timedelta_range('1 days', periods=10)
        obj = tm.box_expected(rng, box)
        assert_invalid_comparison(obj, invalid, box)

    @pytest.mark.parametrize('other', [list(range(10)), np.arange(10), np.arange(10).astype(np.float32), np.arange(10).astype(object), pd.date_range('1970-01-01', periods=10, tz='UTC').array, np.array(pd.date_range('1970-01-01', periods=10)), list(pd.date_range('1970-01-01', periods=10)), pd.date_range('1970-01-01', periods=10).astype(object), pd.period_range('1971-01-01', freq='D', periods=10).array, pd.period_range('1971-01-01', freq='D', periods=10).astype(object)])
    def test_td64arr_cmp_arraylike_invalid(self, other: Any, box_with_array: Any) -> None:
        rng = timedelta_range('1 days', periods=10)._data
        rng = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(rng, other, box_with_array)

    def test_td64arr_cmp_mixed_invalid(self) -> None:
        rng = timedelta_range('1 days', periods=5)._data
        other = np.array([0, 1, 2, rng[3], Timestamp('2021-01-01')])
        result = rng == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = rng != other
        tm.assert_numpy_array_equal(result, ~expected)
        msg = 'Invalid comparison between|Cannot compare type|not supported between'
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
    def test_comp_nat(self, dtype: Optional[str]) -> None:
        left = TimedeltaIndex([Timedelta('1 days'), NaT, Timedelta('3 days')])
        right = TimedeltaIndex([NaT, NaT, Timedelta('3 days')])
        lhs, rhs = (left, right)
        if dtype is object:
            lhs, rhs = (left.astype(object), right.astype(object))
        result = rhs == lhs
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = rhs != lhs
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs == NaT, expected)
        tm.assert_numpy_array_equal(NaT == rhs, expected)
        expected = np.array([True, True, True])
        tm.assert_numpy_array_equal(lhs != NaT, expected)
        tm.assert_numpy_array_equal(NaT != lhs, expected)
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs < NaT, expected)
        tm.assert_numpy_array_equal(NaT > lhs, expected)

    @pytest.mark.parametrize('idx2', [TimedeltaIndex(['2 day', '2 day', NaT, NaT, '1 day 00:00:02', '5 days 00:00:03']), np.array([np.timedelta64(2, 'D'), np.timedelta64(2, 'D'), np.timedelta64('nat'), np.timedelta64('nat'), np.timedelta64(1, 'D') + np.timedelta64(2, 's'), np.timedelta64(5, 'D') + np.timedelta64(3, 's')])])
    def test_comparisons_nat(self, idx2: Any) -> None:
        idx1 = TimedeltaIndex(['1 day', NaT, '1 day 00:00:01', NaT, '1 day 00:00:01', '5 day 00:00:03'])
        result = idx1 < idx2
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = idx2 > idx1
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 <= idx2
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx2 >= idx1
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 == idx2
        expected = np.array([False, False, False, False, False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = idx1 != idx2
        expected = np.array([True, True, True, True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    def test_comparisons_coverage(self) -> None:
        rng = timedelta_range('1 days', periods=10)
        result = rng < rng[3]
        expected = np.array([True, True, True] + [False] * 7)
        tm.assert_numpy_array_equal(result, expected)
        result = rng == list(rng)
        exp = rng == rng
        tm.assert_numpy_array_equal(result, exp)

class TestTimedelta64ArithmeticUnsorted:

    def test_ufunc_coercions(self) -> None:
        idx = TimedeltaIndex(['2h', '4h', '6h', '8h', '10h'], freq='2h', name='x')
        for result in [idx * 2, np.multiply(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(['4h', '8h', '12h', '16h', '20h'], freq='4h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == '4h'
        for result in [idx / 2, np.divide(idx, 2)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(['1h', '2h', '3h', '4h', '5h'], freq='h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == 'h'
        for result in [-idx, np.negative(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(['-2h', '-4h', '-6h', '-8h', '-10h'], freq='-2h', name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq == '-2h'
        idx = TimedeltaIndex(['-2h', '-1h', '0h', '1h', '2h'], freq='h', name='x')
        for result in [abs(idx), np.absolute(idx)]:
            assert isinstance(result, TimedeltaIndex)
            exp = TimedeltaIndex(['2h', '1h', '0h', '1h', '2h'], freq=None, name='x')
            tm.assert_index_equal(result, exp)
            assert result.freq is None

    def test_subtraction_ops(self) -> None:
        tdi = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti = pd.date_range('20130101', periods=3, name='bar')
        td = Timedelta('1 days')
        dt = Timestamp('20130101')
        msg = 'cannot subtract a datelike from a TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi - dt
        with pytest.raises(TypeError, match=msg):
            tdi - dti
        msg = 'unsupported operand type\\(s\\) for -'
        with pytest.raises(TypeError, match=msg):
            td - dt
        msg = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            td - dti
        result = dt - dti
        expected = TimedeltaIndex(['0 days', '-1 days', '-2 days'], name='bar')
        tm.assert_index_equal(result, expected)
        result = dti - dt
        expected = TimedeltaIndex(['0 days', '1 days', '2 days'], name='bar')
        tm.assert_index_equal(result, expected)
        result = tdi - td
        expected = TimedeltaIndex(['0 days', NaT, '1 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = td - tdi
        expected = TimedeltaIndex(['0 days', NaT, '-1 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = dti - td
        expected = DatetimeIndex(['20121231', '20130101', '20130102'], dtype='M8[ns]', freq='D', name='bar')
        tm.assert_index_equal(result, expected)
        result = dt - tdi
        expected = DatetimeIndex(['20121231', NaT, '20121230'], dtype='M8[ns]', name='foo')
        tm.assert_index_equal(result, expected)

    def test_subtraction_ops_with_tz(self, box_with_array: Any) -> None:
        dti = pd.date_range('20130101', periods=3)
        dti = tm.box_expected(dti, box_with_array)
        ts = Timestamp('20130101')
        dt = ts.to_pydatetime()
        dti_tz = pd.date_range('20130101', periods=3).tz_localize('US/Eastern')
        dti_tz = tm.box_expected(dti_tz, box_with_array)
        ts_tz = Timestamp('20130101').tz_localize('US/Eastern')
        ts_tz2 = Timestamp('20130101').tz_localize('CET')
        dt_tz = ts_tz.to_pydatetime()
        td = Timedelta('1 days')

        def _check(result: Any, expected: Any) -> None:
            assert result == expected
            assert isinstance(result, Timedelta)
        result = ts - ts
        expected = Timedelta('0 days')
        _check(result, expected)
        result = dt_tz - ts_tz
        expected = Timedelta('0 days')
        _check(result, expected)
        result = ts_tz - dt_tz
        expected = Timedelta('0 days')
        _check(result, expected)
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects.'
        with pytest.raises(TypeError, match=msg):
            dt_tz - ts
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt_tz - dt
        msg = "can't subtract offset-naive and offset-aware datetimes"
        with pytest.raises(TypeError, match=msg):
            dt - dt_tz
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects.'
        with pytest.raises(TypeError, match=msg):
            ts - dt_tz
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - ts
        with pytest.raises(TypeError, match=msg):
            ts_tz2 - dt
        msg = 'Cannot subtract tz-naive and tz-aware'
        with pytest.raises(TypeError, match=msg):
            dti - ts_tz
        with pytest.raises(TypeError, match=msg):
            dti_tz - ts
        result = dti_tz - dt_tz
        expected = TimedeltaIndex(['0 days', '1 days', '2 days'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = dt_tz - dti_tz
        expected = TimedeltaIndex(['0 days', '-1 days', '-2 days'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = dti_tz - ts_tz
        expected = TimedeltaIndex(['0 days', '1 days', '2 days'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = ts_tz - dti_tz
        expected = TimedeltaIndex(['0 days', '-1 days', '-2 days'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = td - td
        expected = Timedelta('0 days')
        _check(result, expected)
        result = dti_tz - td
        expected = DatetimeIndex(['20121231', '20130101', '20130102'], tz='US/Eastern').as_unit('ns')
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dti_tdi_numeric_ops(self) -> None:
        tdi = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti = pd.date_range('20130101', periods=3, name='bar')
        result = tdi - tdi
        expected = TimedeltaIndex(['0 days', NaT, '0 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = tdi + tdi
        expected = TimedeltaIndex(['2 days', NaT, '4 days'], name='foo')
        tm.assert_index_equal(result, expected)
        result = dti - tdi
        expected = DatetimeIndex(['20121231', NaT, '20130101'], dtype='M8[ns]')
        tm.assert_index_equal(result, expected)

    def test_addition_ops(self) -> None:
        tdi = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
        dti = pd.date_range('20130101', periods=3, name='bar')
        td = Timedelta('1 days')
        dt = Timestamp('20130101')
        result = tdi + dt
        expected = DatetimeIndex(['20130102', NaT, '20130103'], dtype='M8[ns]', name='foo')
        tm.assert_index_equal(result, expected)
        result = dt + tdi
        expected = DatetimeIndex(['