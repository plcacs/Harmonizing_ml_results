#!/usr/bin/env python
from datetime import datetime, time, timedelta, timezone
from itertools import product, starmap
import operator
from typing import Any, Union, List
import numpy as np
import pytest
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
import pandas as pd
from pandas import DateOffset, DatetimeIndex, NaT, Period, Series, Timedelta, TimedeltaIndex, Timestamp, date_range
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import assert_cannot_add, assert_invalid_addsub_type, assert_invalid_comparison, get_upcast_box

class TestDatetime64ArrayLikeComparisons:
    def test_compare_zerodim(self, tz_naive_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_naive_fixture
        box: Any = box_with_array
        dti: DatetimeIndex = date_range('20130101', periods=3, tz=tz)
        other: np.ndarray = np.array(dti.to_numpy()[0])
        dtarr: Any = tm.box_expected(dti, box)
        xbox: Any = get_upcast_box(dtarr, other, True)
        result: Any = dtarr <= other
        expected: np.ndarray = np.array([True, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('other', ['foo', -1, 99, 4.0, object(), timedelta(days=2), datetime(2001, 1, 1).date(), None, np.nan])
    def test_dt64arr_cmp_scalar_invalid(self, other: Any, tz_naive_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_naive_fixture
        rng: DatetimeIndex = date_range('1/1/2000', periods=10, tz=tz)
        dtarr: Any = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(dtarr, other, box_with_array)

    @pytest.mark.parametrize('other', [
        list(range(10)), 
        np.arange(10), 
        np.arange(10).astype(np.float32), 
        np.arange(10).astype(object), 
        pd.timedelta_range('1ns', periods=10).array, 
        np.array(pd.timedelta_range('1ns', periods=10)), 
        list(pd.timedelta_range('1ns', periods=10)), 
        pd.timedelta_range('1 Day', periods=10).astype(object), 
        pd.period_range('1971-01-01', freq='D', periods=10).array, 
        pd.period_range('1971-01-01', freq='D', periods=10).astype(object)
    ])
    def test_dt64arr_cmp_arraylike_invalid(self, other: Any, tz_naive_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_naive_fixture
        dta: Any = date_range('1970-01-01', freq='ns', periods=10, tz=tz)._data
        obj: Any = tm.box_expected(dta, box_with_array)
        assert_invalid_comparison(obj, other, box_with_array)

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        dta: Any = date_range('1970-01-01', freq='h', periods=5, tz=tz)._data
        other: np.ndarray = np.array([0, 1, 2, dta[3], Timedelta(days=1)])
        result: np.ndarray = dta == other
        expected: np.ndarray = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = dta != other
        tm.assert_numpy_array_equal(result, ~expected)
        msg: str = 'Invalid comparison between|Cannot compare type|not supported between'
        with pytest.raises(TypeError, match=msg):
            dta < other
        with pytest.raises(TypeError, match=msg):
            dta > other
        with pytest.raises(TypeError, match=msg):
            dta <= other
        with pytest.raises(TypeError, match=msg):
            dta >= other

    def test_dt64arr_nat_comparison(self, tz_naive_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_naive_fixture
        box: Any = box_with_array
        ts: Timestamp = Timestamp('2021-01-01', tz=tz)
        ser: Series = Series([ts, NaT])
        obj: Any = tm.box_expected(ser, box)
        xbox: Any = get_upcast_box(obj, ts, True)
        expected: Series = Series([True, False], dtype=np.bool_)
        expected = tm.box_expected(expected, xbox)
        result: Any = obj == ts
        tm.assert_equal(result, expected)

class TestDatetime64SeriesComparison:
    @pytest.mark.parametrize('pair', [
        ([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [NaT, NaT, Timestamp('2011-01-03')]),
        ([Timedelta('1 days'), NaT, Timedelta('3 days')], [NaT, NaT, Timedelta('3 days')]),
        ([Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')], [NaT, NaT, Period('2011-03', freq='M')])
    ])
    @pytest.mark.parametrize('reverse', [True, False])
    @pytest.mark.parametrize('dtype', [None, object])
    @pytest.mark.parametrize('op, expected', [
        (operator.eq, [False, False, True]), 
        (operator.ne, [True, True, False]), 
        (operator.lt, [False, False, False]), 
        (operator.gt, [False, False, False]), 
        (operator.ge, [False, False, True]), 
        (operator.le, [False, False, True])
    ])
    def test_nat_comparisons(self, dtype: Any, index_or_series: Any, reverse: bool, pair: List[Any], op: Any, expected: List[bool]) -> None:
        box: Any = index_or_series
        lhs, rhs = pair
        if reverse:
            lhs, rhs = (rhs, lhs)
        left: Series = Series(lhs, dtype=dtype)
        right: Any = box(rhs, dtype=dtype)
        result: Any = op(left, right)
        tm.assert_series_equal(result, Series(expected))

    @pytest.mark.parametrize('data', [
        [Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], 
        [Timedelta('1 days'), NaT, Timedelta('3 days')], 
        [Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')]
    ])
    @pytest.mark.parametrize('dtype', [None, object])
    def test_nat_comparisons_scalar(self, dtype: Any, data: List[Any], box_with_array: Any) -> None:
        box: Any = box_with_array
        left: Series = Series(data, dtype=dtype)
        left = tm.box_expected(left, box)
        xbox: Any = get_upcast_box(left, NaT, True)
        expected: List[bool] = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype='bool')
        tm.assert_equal(left == NaT, expected)
        tm.assert_equal(NaT == left, expected)
        expected = [True, True, True]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype='bool')
        tm.assert_equal(left != NaT, expected)
        tm.assert_equal(NaT != left, expected)
        expected = [False, False, False]
        expected = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype='bool')
        tm.assert_equal(left < NaT, expected)
        tm.assert_equal(NaT > left, expected)
        tm.assert_equal(left <= NaT, expected)
        tm.assert_equal(NaT >= left, expected)
        tm.assert_equal(left > NaT, expected)
        tm.assert_equal(NaT < left, expected)
        tm.assert_equal(left >= NaT, expected)
        tm.assert_equal(NaT <= left, expected)

    @pytest.mark.parametrize('val', [datetime(2000, 1, 4), datetime(2000, 1, 5)])
    def test_series_comparison_scalars(self, val: Union[datetime, Any]) -> None:
        series: Series = Series(date_range('1/1/2000', periods=10))
        result: Any = series > val
        expected: Series = Series([x > val for x in series])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('left,right', [('lt', 'gt'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
    def test_timestamp_compare_series(self, left: str, right: str) -> None:
        ser: Series = Series(date_range('20010101', periods=10), name='dates')
        s_nat: Series = ser.copy(deep=True)
        ser[0] = Timestamp('nat')
        ser[3] = Timestamp('nat')
        left_f: Any = getattr(operator, left)
        right_f: Any = getattr(operator, right)
        expected: Any = left_f(ser, Timestamp('20010109'))
        result: Any = right_f(Timestamp('20010109'), ser)
        tm.assert_series_equal(result, expected)
        expected = left_f(ser, Timestamp('nat'))
        result = right_f(Timestamp('nat'), ser)
        tm.assert_series_equal(result, expected)
        expected = left_f(s_nat, Timestamp('20010109'))
        result = right_f(Timestamp('20010109'), s_nat)
        tm.assert_series_equal(result, expected)
        expected = left_f(s_nat, NaT)
        result = right_f(NaT, s_nat)
        tm.assert_series_equal(result, expected)

    def test_dt64arr_timestamp_equality(self, box_with_array: Any) -> None:
        box: Any = box_with_array
        ser: Series = Series([Timestamp('2000-01-29 01:59:00'), Timestamp('2000-01-30'), NaT])
        ser = tm.box_expected(ser, box)
        xbox: Any = get_upcast_box(ser, ser, True)
        result: Any = ser != ser
        expected: Any = tm.box_expected([False, False, True], xbox)
        tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser != ser[0]
        else:
            result = ser != ser[0]
            expected = tm.box_expected([False, True, True], xbox)
            tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser != ser[2]
        else:
            result = ser != ser[2]
            expected = tm.box_expected([True, True, True], xbox)
            tm.assert_equal(result, expected)
        result = ser == ser
        expected = tm.box_expected([True, True, False], xbox)
        tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser == ser[0]
        else:
            result = ser == ser[0]
            expected = tm.box_expected([True, False, False], xbox)
            tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser == ser[2]
        else:
            result = ser == ser[2]
            expected = tm.box_expected([False, False, False], xbox)
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize('datetimelike', [
        Timestamp('20130101'), 
        datetime(2013, 1, 1), 
        np.datetime64('2013-01-01T00:00', 'ns')
    ])
    @pytest.mark.parametrize('op,expected', [
        (operator.lt, [True, False, False, False]), 
        (operator.le, [True, True, False, False]), 
        (operator.eq, [False, True, False, False]), 
        (operator.gt, [False, False, False, True])
    ])
    def test_dt64_compare_datetime_scalar(self, datetimelike: Any, op: Any, expected: List[bool]) -> None:
        ser: Series = Series([Timestamp('20120101'), Timestamp('20130101'), np.nan, Timestamp('20130103')], name='A')
        result: Any = op(ser, datetimelike)
        expected_ser: Series = Series(expected, name='A')
        tm.assert_series_equal(result, expected_ser)

    def test_ts_series_numpy_maximum(self) -> None:
        ts: Timestamp = Timestamp('2024-07-01')
        ts_series: Series = Series(['2024-06-01', '2024-07-01', '2024-08-01'], dtype='datetime64[us]')
        expected: Series = Series(['2024-07-01', '2024-07-01', '2024-08-01'], dtype='datetime64[us]')
        tm.assert_series_equal(expected, np.maximum(ts, ts_series))

class TestDatetimeIndexComparisons:
    def test_comparators(self, comparison_op: Any) -> None:
        index: DatetimeIndex = date_range('2020-01-01', periods=10)
        element: Any = index[len(index) // 2]
        element = Timestamp(element).to_datetime64()
        arr: np.ndarray = np.array(index)
        arr_result: Any = comparison_op(arr, element)
        index_result: Any = comparison_op(index, element)
        assert isinstance(index_result, np.ndarray)
        tm.assert_numpy_array_equal(arr_result, index_result)

    @pytest.mark.parametrize('other', [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
    def test_dti_cmp_datetimelike(self, other: Any, tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = date_range('2016-01-01', periods=2, tz=tz)
        if tz is not None:
            if isinstance(other, np.datetime64):
                pytest.skip(f'{type(other).__name__} is not tz aware')
            other = localize_pydatetime(other, dti.tzinfo)
        result: np.ndarray = dti == other
        expected: np.ndarray = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = dti > other
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(result, expected)
        result = dti >= other
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
        result = dti < other
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)
        result = dti <= other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('dtype', [None, object])
    def test_dti_cmp_nat(self, dtype: Any, box_with_array: Any) -> None:
        left: DatetimeIndex = DatetimeIndex([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')])
        right: DatetimeIndex = DatetimeIndex([NaT, NaT, Timestamp('2011-01-03')])
        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        xbox: Any = get_upcast_box(left, right, True)
        lhs: Any = left
        rhs: Any = right
        if dtype is object:
            lhs, rhs = (left.astype(object), right.astype(object))
        result: Any = rhs == lhs
        expected: np.ndarray = np.array([False, False, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)
        result = lhs != rhs
        expected = np.array([True, True, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)
        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs == NaT, expected)
        tm.assert_equal(NaT == rhs, expected)
        expected = np.array([True, True, True])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs != NaT, expected)
        tm.assert_equal(NaT != lhs, expected)
        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs < NaT, expected)
        tm.assert_equal(NaT > lhs, expected)

    def test_dti_cmp_nat_behaves_like_float_cmp_nan(self) -> None:
        fidx1: pd.Index = pd.Index([1.0, np.nan, 3.0, np.nan, 5.0, 7.0])
        fidx2: pd.Index = pd.Index([2.0, 3.0, np.nan, np.nan, 6.0, 7.0])
        didx1: DatetimeIndex = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
        didx2: DatetimeIndex = DatetimeIndex(['2014-02-01', '2014-03-01', NaT, NaT, '2014-06-01', '2014-07-01'])
        darr: np.ndarray = np.array([np.datetime64('2014-02-01 00:00'), np.datetime64('2014-03-01 00:00'), np.datetime64('nat'), np.datetime64('nat'), np.datetime64('2014-06-01 00:00'), np.datetime64('2014-07-01 00:00')])
        cases: List[Any] = [(fidx1, fidx2), (didx1, didx2), (didx1, darr)]
        with tm.assert_produces_warning(None):
            for idx1, idx2 in cases:
                result: Any = idx1 < idx2
                expected: np.ndarray = np.array([True, False, False, False, True, False])
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
        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, np.nan), (didx1, NaT)]:
                result = idx1 < val
                expected = np.array([False, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 > val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 <= val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 >= val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 == val
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 != val
                expected = np.array([True, True, True, True, True, True])
                tm.assert_numpy_array_equal(result, expected)
        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, 3), (didx1, datetime(2014, 3, 1))]:
                result = idx1 < val
                expected = np.array([True, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 > val
                expected = np.array([False, False, False, False, True, True])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 <= val
                expected = np.array([True, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 >= val
                expected = np.array([False, False, True, False, True, True])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 == val
                expected = np.array([False, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result = idx1 != val
                expected = np.array([True, True, False, True, True, True])
                tm.assert_numpy_array_equal(result, expected)

    def test_comparison_tzawareness_compat(self, comparison_op: Any, box_with_array: Any) -> None:
        op: Any = comparison_op
        box: Any = box_with_array
        dr: DatetimeIndex = date_range('2016-01-01', periods=6)
        dz: DatetimeIndex = dr.tz_localize('US/Pacific')
        dr = tm.box_expected(dr, box)
        dz = tm.box_expected(dz, box)
        if box is pd.DataFrame:
            tolist = lambda x: x.astype(object).values.tolist()[0]
        else:
            tolist = list
        if op not in [operator.eq, operator.ne]:
            msg: str = 'Invalid comparison between dtype=datetime64\\[ns.*\\] and (Timestamp|DatetimeArray|list|ndarray)'
            with pytest.raises(TypeError, match=msg):
                op(dr, dz)
            with pytest.raises(TypeError, match=msg):
                op(dr, tolist(dz))
            with pytest.raises(TypeError, match=msg):
                op(dr, np.array(tolist(dz), dtype=object))
            with pytest.raises(TypeError, match=msg):
                op(dz, dr)
            with pytest.raises(TypeError, match=msg):
                op(dz, tolist(dr))
            with pytest.raises(TypeError, match=msg):
                op(dz, np.array(tolist(dr), dtype=object))
        assert np.all(dr == dr)
        assert np.all(dr == tolist(dr))
        assert np.all(tolist(dr) == dr)
        assert np.all(np.array(tolist(dr), dtype=object) == dr)
        assert np.all(dr == np.array(tolist(dr), dtype=object))
        assert np.all(dz == dz)
        assert np.all(dz == tolist(dz))
        assert np.all(tolist(dz) == dz)
        assert np.all(np.array(tolist(dz), dtype=object) == dz)
        assert np.all(dz == np.array(tolist(dz), dtype=object))

    def test_comparison_tzawareness_compat_scalars(self, comparison_op: Any, box_with_array: Any) -> None:
        op: Any = comparison_op
        dr: DatetimeIndex = date_range('2016-01-01', periods=6)
        dz: DatetimeIndex = dr.tz_localize('US/Pacific')
        dr = tm.box_expected(dr, box_with_array)
        dz = tm.box_expected(dz, box_with_array)
        ts: Timestamp = Timestamp('2000-03-14 01:59')
        ts_tz: Timestamp = Timestamp('2000-03-14 01:59', tz='Europe/Amsterdam')
        assert np.all(dr > ts)
        msg: str = 'Invalid comparison between dtype=datetime64\\[ns, .*\\] and Timestamp'
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dr, ts_tz)
        assert np.all(dz > ts_tz)
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dz, ts)
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(ts, dz)

    @pytest.mark.parametrize('other', [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
    @pytest.mark.filterwarnings('ignore:elementwise comp:DeprecationWarning')
    def test_scalar_comparison_tzawareness(self, comparison_op: Any, other: Any, tz_aware_fixture: Any, box_with_array: Any) -> None:
        op: Any = comparison_op
        tz: Any = tz_aware_fixture
        dti: DatetimeIndex = date_range('2016-01-01', periods=2, tz=tz)
        dtarr: Any = tm.box_expected(dti, box_with_array)
        xbox: Any = get_upcast_box(dtarr, other, True)
        if op in [operator.eq, operator.ne]:
            exbool: bool = (op is operator.ne)
            expected: np.ndarray = np.array([exbool, exbool], dtype=bool)
            expected = tm.box_expected(expected, xbox)
            result: Any = op(dtarr, other)
            tm.assert_equal(result, expected)
            result = op(other, dtarr)
            tm.assert_equal(result, expected)
        else:
            msg: str = f'Invalid comparison between dtype=datetime64\\[ns, .*\\] and {type(other).__name__}'
            with pytest.raises(TypeError, match=msg):
                op(dtarr, other)
            with pytest.raises(TypeError, match=msg):
                op(other, dtarr)

    def test_nat_comparison_tzawareness(self, comparison_op: Any) -> None:
        op: Any = comparison_op
        dti: DatetimeIndex = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
        expected: np.ndarray = np.array([op == operator.ne] * len(dti))
        result: Any = op(dti, NaT)
        tm.assert_numpy_array_equal(result, expected)
        result = op(dti.tz_localize('US/Pacific'), NaT)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_str(self, tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        rng: DatetimeIndex = date_range('1/1/2000', periods=10, tz=tz)
        other: str = '1/1/2000'
        result: Any = rng == other
        expected: np.ndarray = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result = rng != other
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result = rng < other
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)
        result = rng <= other
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result = rng > other
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result = rng >= other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_list(self) -> None:
        rng: DatetimeIndex = date_range('1/1/2000', periods=10)
        result: Any = rng == list(rng)
        expected: Any = rng == rng
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other', [
        pd.timedelta_range('1D', periods=10), 
        pd.timedelta_range('1D', periods=10).to_series(), 
        pd.timedelta_range('1D', periods=10).asi8.view('m8[ns]')
    ], ids=lambda x: type(x).__name__)
    def test_dti_cmp_tdi_tzawareness(self, other: Any) -> None:
        dti: DatetimeIndex = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
        result: Any = dti == other
        expected: np.ndarray = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)
        result = dti != other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)
        msg: str = 'Invalid comparison between'
        with pytest.raises(TypeError, match=msg):
            dti < other
        with pytest.raises(TypeError, match=msg):
            dti <= other
        with pytest.raises(TypeError, match=msg):
            dti > other
        with pytest.raises(TypeError, match=msg):
            dti >= other

    def test_dti_cmp_object_dtype(self) -> None:
        dti: DatetimeIndex = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
        other: Any = dti.astype('O')
        result: Any = dti == other
        expected: np.ndarray = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)
        other = dti.tz_localize(None)
        result = dti != other
        tm.assert_numpy_array_equal(result, expected)
        other = np.array(list(dti[:5]) + [Timedelta(days=1)] * 5)
        result = dti == other
        expected = np.array([True] * 5 + [False] * 5)
        tm.assert_numpy_array_equal(result, expected)
        msg: str = ">=' not supported between instances of 'Timestamp' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            dti >= other

class TestDatetime64Arithmetic:
    @pytest.mark.arm_slow
    def test_dt64arr_add_timedeltalike_scalar(self, tz_naive_fixture: Any, two_hours: Any, box_with_array: Any) -> None:
        tz: Any = tz_naive_fixture
        rng: DatetimeIndex = date_range('2000-01-01', '2000-02-01', tz=tz)
        expected: DatetimeIndex = date_range('2000-01-01 02:00', '2000-02-01 02:00', tz=tz)
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result: Any = rng + two_hours
        tm.assert_equal(result, expected)
        result = two_hours + rng
        tm.assert_equal(result, expected)
        rng += two_hours
        tm.assert_equal(rng, expected)

    def test_dt64arr_sub_timedeltalike_scalar(self, tz_naive_fixture: Any, two_hours: Any, box_with_array: Any) -> None:
        tz: Any = tz_naive_fixture
        rng: DatetimeIndex = date_range('2000-01-01', '2000-02-01', tz=tz)
        expected: DatetimeIndex = date_range('1999-12-31 22:00', '2000-01-31 22:00', tz=tz)
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result: Any = rng - two_hours
        tm.assert_equal(result, expected)
        rng -= two_hours
        tm.assert_equal(rng, expected)

    def test_dt64_array_sub_dt_with_different_timezone(self, box_with_array: Any) -> None:
        t1: DatetimeIndex = date_range('20130101', periods=3).tz_localize('US/Eastern')
        t1 = tm.box_expected(t1, box_with_array)
        t2: Timestamp = Timestamp('20130101').tz_localize('CET')
        tnaive: Timestamp = Timestamp(20130101)
        result: Any = t1 - t2
        expected: TimedeltaIndex = TimedeltaIndex(['0 days 06:00:00', '1 days 06:00:00', '2 days 06:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = t2 - t1
        expected = TimedeltaIndex(['-1 days +18:00:00', '-2 days +18:00:00', '-3 days +18:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64_array_sub_dt64_array_with_different_timezone(self, box_with_array: Any) -> None:
        t1: DatetimeIndex = date_range('20130101', periods=3).tz_localize('US/Eastern')
        t1 = tm.box_expected(t1, box_with_array)
        t2: DatetimeIndex = date_range('20130101', periods=3).tz_localize('CET')
        t2 = tm.box_expected(t2, box_with_array)
        tnaive: DatetimeIndex = date_range('20130101', periods=3)
        result: Any = t1 - t2
        expected: TimedeltaIndex = TimedeltaIndex(['0 days 06:00:00', '0 days 06:00:00', '0 days 06:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = t2 - t1
        expected = TimedeltaIndex(['-1 days +18:00:00', '-1 days +18:00:00', '-1 days +18:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64arr_add_sub_td64_nat(self, box_with_array: Any, tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = date_range('1994-04-01', periods=9, tz=tz, freq='QS')
        other: np.timedelta64 = np.timedelta64('NaT')
        expected: DatetimeIndex = DatetimeIndex(['NaT'] * 9, tz=tz).as_unit('ns')
        obj: Any = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result: Any = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        msg: str = 'cannot subtract'
        with pytest.raises(TypeError, match=msg):
            other - obj

    def test_dt64arr_add_sub_td64ndarray(self, tz_naive_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz=tz)
        tdi: TimedeltaIndex = TimedeltaIndex(['-1 Day', '-1 Day', '-1 Day'])
        tdarr: np.ndarray = tdi.values
        expected: DatetimeIndex = date_range('2015-12-31', '2016-01-02', periods=3, tz=tz)
        dtarr: Any = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result: Any = dtarr + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + dtarr
        tm.assert_equal(result, expected)
        expected = date_range('2016-01-02', '2016-01-04', periods=3, tz=tz)
        expected = tm.box_expected(expected, box_with_array)
        result = dtarr - tdarr
        tm.assert_equal(result, expected)
        msg: str = 'cannot subtract|(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            tdarr - dtarr

    @pytest.mark.parametrize('ts', [
        Timestamp('2013-01-01'), 
        Timestamp('2013-01-01').to_pydatetime(), 
        Timestamp('2013-01-01').to_datetime64(), 
        np.datetime64('2013-01-01', 'D')
    ])
    def test_dt64arr_sub_dtscalar(self, box_with_array: Any, ts: Any) -> None:
        idx: DatetimeIndex = date_range('2013-01-01', periods=3)._with_freq(None)
        idx = tm.box_expected(idx, box_with_array)
        expected: TimedeltaIndex = TimedeltaIndex(['0 Days', '1 Day', '2 Days'])
        expected = tm.box_expected(expected, box_with_array)
        result: Any = idx - ts
        tm.assert_equal(result, expected)
        result = ts - idx
        tm.assert_equal(result, -expected)
        tm.assert_equal(result, -expected)

    def test_dt64arr_sub_timestamp_tzaware(self, box_with_array: Any) -> None:
        ser: DatetimeIndex = date_range('2014-03-17', periods=2, freq='D', tz='US/Eastern')
        ser = ser._with_freq(None)
        ts: Timestamp = ser[0]
        ser = tm.box_expected(ser, box_with_array)
        delta_series: Series = Series([np.timedelta64(0, 'D'), np.timedelta64(1, 'D')])
        expected: Any = tm.box_expected(delta_series, box_with_array)
        tm.assert_equal(ser - ts, expected)
        tm.assert_equal(ts - ser, -expected)

    def test_dt64arr_sub_NaT(self, box_with_array: Any, unit: str) -> None:
        dti: DatetimeIndex = DatetimeIndex([NaT, Timestamp('19900315')]).as_unit(unit)
        ser: Any = tm.box_expected(dti, box_with_array)
        result: Any = ser - NaT
        expected: Series = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        dti_tz: DatetimeIndex = dti.tz_localize('Asia/Tokyo')
        ser_tz: Any = tm.box_expected(dti_tz, box_with_array)
        result = ser_tz - NaT
        expected = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dt64arr_sub_dt64object_array(self, performance_warning: Any, box_with_array: Any, tz_naive_fixture: Any) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz=tz_naive_fixture)
        expected: DatetimeIndex = dti - dti
        obj: Any = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array).astype(object)
        with tm.assert_produces_warning(performance_warning):
            result: Any = obj - obj.astype(object)
        tm.assert_equal(result, expected)

    def test_dt64arr_naive_sub_dt64ndarray(self, box_with_array: Any) -> None:
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz=None)
        dt64vals: np.ndarray = dti.values
        dtarr: Any = tm.box_expected(dti, box_with_array)
        expected: Any = dtarr - dtarr
        result: Any = dtarr - dt64vals
        tm.assert_equal(result, expected)
        result = dt64vals - dtarr
        tm.assert_equal(result, expected)

    def test_dt64arr_aware_sub_dt64ndarray_raises(self, tz_aware_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_aware_fixture
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz=tz)
        dt64vals: np.ndarray = dti.values
        dtarr: Any = tm.box_expected(dti, box_with_array)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime'
        with pytest.raises(TypeError, match=msg):
            dtarr - dt64vals
        with pytest.raises(TypeError, match=msg):
            dt64vals - dtarr

    def test_dt64arr_add_dtlike_raises(self, tz_naive_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = date_range('2016-01-01', periods=3, tz=tz)
        if tz is None:
            dti2: DatetimeIndex = dti.tz_localize('US/Eastern')
        else:
            dti2 = dti.tz_localize(None)
        dtarr: Any = tm.box_expected(dti, box_with_array)
        assert_cannot_add(dtarr, dti.values)
        assert_cannot_add(dtarr, dti)
        assert_cannot_add(dtarr, dtarr)
        assert_cannot_add(dtarr, dti[0])
        assert_cannot_add(dtarr, dti[0].to_pydatetime())
        assert_cannot_add(dtarr, dti[0].to_datetime64())
        assert_cannot_add(dtarr, dti2[0])
        assert_cannot_add(dtarr, dti2[0].to_pydatetime())
        assert_cannot_add(dtarr, np.datetime64('2011-01-01', 'D'))

    @pytest.mark.parametrize('freq', ['h', 'D', 'W', '2ME', 'MS', 'QE', 'B', None])
    @pytest.mark.parametrize('dtype', [None, 'uint8'])
    def test_dt64arr_addsub_intlike(self, dtype: Any, index_or_series_or_array: Any, freq: Union[str, None], tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        if freq is None:
            dti: DatetimeIndex = DatetimeIndex(['NaT', '2017-04-05 06:07:08'], tz=tz)
        else:
            dti = date_range('2016-01-01', periods=2, freq=freq, tz=tz)
        obj: Any = index_or_series_or_array(dti)
        other: np.ndarray = np.array([4, -1])
        if dtype is not None:
            other = other.astype(dtype)
        msg: str = '|'.join([
            'Addition/subtraction of integers', 
            'cannot subtract DatetimeArray from', 
            'can only perform ops with numeric values', 
            'unsupported operand type.*Categorical', 
            "unsupported operand type\\(s\\) for -: 'int' and 'Timestamp'"
        ])
        assert_invalid_addsub_type(obj, 1, msg)
        assert_invalid_addsub_type(obj, np.int64(2), msg)
        assert_invalid_addsub_type(obj, np.array(3, dtype=np.int64), msg)
        assert_invalid_addsub_type(obj, other, msg)
        assert_invalid_addsub_type(obj, np.array(other), msg)
        assert_invalid_addsub_type(obj, pd.array(other), msg)
        assert_invalid_addsub_type(obj, pd.Categorical(other), msg)
        assert_invalid_addsub_type(obj, pd.Index(other), msg)
        assert_invalid_addsub_type(obj, Series(other), msg)

    @pytest.mark.parametrize('other', [3.14, np.array([2.0, 3.0]), Period('2011-01-01', freq='D'), time(1, 2, 3)])
    @pytest.mark.parametrize('dti_freq', [None, 'D'])
    def test_dt64arr_add_sub_invalid(self, dti_freq: Any, other: Any, box_with_array: Any) -> None:
        dti: DatetimeIndex = date_range('2011-01-01', periods=2, freq=dti_freq)
        dtarr: Any = tm.box_expected(dti, box_with_array)
        msg: str = '|'.join([
            'unsupported operand type', 
            'cannot (add|subtract)', 
            'cannot use operands with types', 
            "ufunc '?(add|subtract)'? cannot use operands with types", 
            'Concatenation operation is not implemented for NumPy arrays'
        ])
        assert_invalid_addsub_type(dtarr, other, msg)

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('dti_freq', [None, 'D'])
    def test_dt64arr_add_sub_parr(self, dti_freq: Any, pi_freq: str, box_with_array: Any, box_with_array2: Any) -> None:
        dti: DatetimeIndex = date_range('2011-01-01', periods=2, freq=dti_freq)
        pi: Any = dti.to_period(pi_freq)
        dtarr: Any = tm.box_expected(dti, box_with_array)
        parr: Any = tm.box_expected(pi, box_with_array2)
        msg: str = '|'.join([
            'cannot (add|subtract)', 
            'unsupported operand', 
            'descriptor.*requires', 
            'ufunc.*cannot use operands'
        ])
        assert_invalid_addsub_type(dtarr, parr, msg)

    @pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
    def test_dt64arr_addsub_time_objects_raises(self, box_with_array: Any, tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        obj1: DatetimeIndex = date_range('2012-01-01', periods=3, tz=tz)
        obj2: List[time] = [time(i, i, i) for i in range(3)]
        obj1 = tm.box_expected(obj1, box_with_array)
        obj2 = tm.box_expected(obj2, box_with_array)
        msg: str = '|'.join(['unsupported operand', 'cannot subtract DatetimeArray from ndarray'])
        assert_invalid_addsub_type(obj1, obj2, msg=msg)

    @pytest.mark.parametrize('dt64_series', [
        Series([Timestamp('19900315'), Timestamp('19900315')]),
        Series([NaT, Timestamp('19900315')]),
        Series([NaT, NaT], dtype='datetime64[ns]')
    ])
    @pytest.mark.parametrize('one', [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(self, one: Any, dt64_series: Series, box_with_array: Any) -> None:
        obj: Any = tm.box_expected(dt64_series, box_with_array)
        msg: str = 'cannot perform .* with this index type'
        with pytest.raises(TypeError, match=msg):
            obj * one
        with pytest.raises(TypeError, match=msg):
            one * obj
        with pytest.raises(TypeError, match=msg):
            obj / one
        with pytest.raises(TypeError, match=msg):
            one / obj

class TestDatetime64DateOffsetArithmetic:
    def test_dt64arr_series_add_tick_DateOffset(self, box_with_array: Any, unit: str) -> None:
        ser: Series = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')]).dt.as_unit(unit)
        expected: Series = Series([Timestamp('20130101 9:01:05'), Timestamp('20130101 9:02:05')]).dt.as_unit(unit)
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result: Any = ser + pd.offsets.Second(5)
        tm.assert_equal(result, expected)
        result2: Any = pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)

    def test_dt64arr_series_sub_tick_DateOffset(self, box_with_array: Any) -> None:
        ser: Series = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        expected: Series = Series([Timestamp('20130101 9:00:55'), Timestamp('20130101 9:01:55')])
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result: Any = ser - pd.offsets.Second(5)
        tm.assert_equal(result, expected)
        result2: Any = -pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)
        msg: str = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            pd.offsets.Second(5) - ser

    @pytest.mark.parametrize('cls_name', ['Day', 'Hour', 'Minute', 'Second', 'Milli', 'Micro', 'Nano'])
    def test_dt64arr_add_sub_tick_DateOffset_smoke(self, cls_name: Any, box_with_array: Any) -> None:
        ser: Series = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        ser = tm.box_expected(ser, box_with_array)
        offset_cls: Any = getattr(pd.offsets, cls_name)
        ser + offset_cls(5)
        offset_cls(5) + ser
        ser - offset_cls(5)

    def test_dti_add_tick_tzaware(self, tz_aware_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_aware_fixture
        if tz == 'US/Pacific':
            dates: DatetimeIndex = date_range('2012-11-01', periods=3, tz=tz)
            offset: DatetimeIndex = dates + pd.offsets.Hour(5)
            assert dates[0] + pd.offsets.Hour(5) == offset[0]
        dates = date_range('2010-11-01 00:00', periods=3, tz=tz, freq='h')
        expected: DatetimeIndex = DatetimeIndex(['2010-11-01 05:00', '2010-11-01 06:00', '2010-11-01 07:00'], freq='h', tz=tz).as_unit('ns')
        dates = tm.box_expected(dates, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        for scalar in [pd.offsets.Hour(5), np.timedelta64(5, 'h'), timedelta(hours=5)]:
            offset = dates + scalar
            tm.assert_equal(offset, expected)
            offset = scalar + dates
            tm.assert_equal(offset, expected)
            roundtrip = offset - scalar
            tm.assert_equal(roundtrip, dates)
            msg: str = '|'.join(['bad operand type for unary -', 'cannot subtract DatetimeArray'])
            with pytest.raises(TypeError, match=msg):
                scalar - dates

    def test_dt64arr_add_sub_relativedelta_offsets(self, box_with_array: Any, unit: str) -> None:
        vec: DatetimeIndex = DatetimeIndex([
            Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'),
            Timestamp('2000-01-01'), Timestamp('2000-03-31'), Timestamp('2000-02-29'),
            Timestamp('2000-12-31'), Timestamp('2000-05-15'), Timestamp('2001-06-15')
        ]).as_unit(unit)
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec
        relative_kwargs = [('years', 2), ('months', 5), ('days', 3), ('hours', 5), ('minutes', 10), ('seconds', 2), ('microseconds', 5)]
        for i, (offset_unit, value) in enumerate(relative_kwargs):
            off: DateOffset = DateOffset(**{offset_unit: value})
            exp_unit: str = unit
            if offset_unit == 'microseconds' and unit != 'ns':
                exp_unit = 'us'
            expected: DatetimeIndex = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)
            expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)
            off = DateOffset(**dict(relative_kwargs[:i + 1]))
            expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)
            expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)
            msg: str = '(bad|unsupported) operand type for unary'
            with pytest.raises(TypeError, match=msg):
                off - vec

    @pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
    @pytest.mark.parametrize('cls_and_kwargs', [
        'YearBegin', ('YearBegin', {'month': 5}), 'YearEnd', ('YearEnd', {'month': 5}),
        'MonthBegin', 'MonthEnd', 'SemiMonthEnd', 'SemiMonthBegin', 'Week',
        ('Week', {'weekday': 3}), ('Week', {'weekday': 6}), 'BusinessDay', 'BDay',
        'QuarterEnd', 'QuarterBegin', 'CustomBusinessDay', 'CDay', 'CBMonthEnd',
        'CBMonthBegin', 'BMonthBegin', 'BMonthEnd', 'BusinessHour', 'BYearBegin',
        'BYearEnd', 'BQuarterBegin', ('LastWeekOfMonth', {'weekday': 2}),
        ('FY5253Quarter', {'qtr_with_extra_week': 1, 'startingMonth': 1, 'weekday': 2, 'variation': 'nearest'}),
        ('FY5253', {'weekday': 0, 'startingMonth': 2, 'variation': 'nearest'}),
        ('WeekOfMonth', {'weekday': 2, 'week': 2}), 'Easter', ('DateOffset', {'day': 4}),
        ('DateOffset', {'month': 5})
    ])
    @pytest.mark.parametrize('normalize', [True, False])
    @pytest.mark.parametrize('n', [0, 5])
    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_dt64arr_add_sub_DateOffsets(self, box_with_array: Any, n: int, normalize: bool, cls_and_kwargs: Any, unit: str, tz: Any) -> None:
        if isinstance(cls_and_kwargs, tuple):
            cls_name, kwargs = cls_and_kwargs
        else:
            cls_name = cls_and_kwargs
            kwargs = {}
        if n == 0 and cls_name in ['WeekOfMonth', 'LastWeekOfMonth', 'FY5253Quarter', 'FY5253']:
            return
        vec: DatetimeIndex = DatetimeIndex([
            Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'),
            Timestamp('2000-01-01'), Timestamp('2000-03-31'), Timestamp('2000-02-29'),
            Timestamp('2000-12-31'), Timestamp('2000-05-15'), Timestamp('2001-06-15')
        ]).as_unit(unit).tz_localize(tz)
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec
        offset_cls: Any = getattr(pd.offsets, cls_name)
        offset: Any = offset_cls(n, normalize=normalize, **kwargs)
        expected: DatetimeIndex = DatetimeIndex([x + offset for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec + offset)
        tm.assert_equal(expected, offset + vec)
        expected = DatetimeIndex([x - offset for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec - offset)
        expected = DatetimeIndex([offset + x for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, offset + vec)
        msg: str = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            offset - vec

    @pytest.mark.parametrize('other', [
        [pd.offsets.MonthEnd(), pd.offsets.Day(n=2)],
        [pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()],
        [pd.offsets.DateOffset(years=1), pd.offsets.DateOffset(years=1)]
    ])
    @pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
    def test_dt64arr_add_sub_offset_array(self, performance_warning: Any, tz_naive_fixture: Any, box_with_array: Any, op: Any, other: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = date_range('2017-01-01', periods=2, tz=tz)
        dtarr: Any = tm.box_expected(dti, box_with_array)
        other = np.array(other)
        expected: DatetimeIndex = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))])
        expected = tm.box_expected(expected, box_with_array).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res: Any = op(dtarr, other)
        tm.assert_equal(res, expected)
        other = tm.box_expected(other, box_with_array)
        if box_with_array is pd.array and op is roperator.radd:
            expected = pd.array(expected, dtype=object)
        with tm.assert_produces_warning(performance_warning):
            res = op(dtarr, other)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize('op, offset, exp, exp_freq', [
        ('__add__', DateOffset(months=3, days=10), [Timestamp('2014-04-11'), Timestamp('2015-04-11'), Timestamp('2016-04-11'), Timestamp('2017-04-11')], None),
        ('__add__', DateOffset(months=3), [Timestamp('2014-04-01'), Timestamp('2015-04-01'), Timestamp('2016-04-01'), Timestamp('2017-04-01')], 'YS-APR'),
        ('__sub__', DateOffset(months=3, days=10), [Timestamp('2013-09-21'), Timestamp('2014-09-21'), Timestamp('2015-09-21'), Timestamp('2016-09-21')], None),
        ('__sub__', DateOffset(months=3), [Timestamp('2013-10-01'), Timestamp('2014-10-01'), Timestamp('2015-10-01'), Timestamp('2016-10-01')], 'YS-OCT')
    ])
    def test_dti_add_sub_nonzero_mth_offset(self, op: str, offset: DateOffset, exp: List[Timestamp], exp_freq: Any, tz_aware_fixture: Any, box_with_array: Any) -> None:
        tz: Any = tz_aware_fixture
        date: DatetimeIndex = date_range(start='01 Jan 2014', end='01 Jan 2017', freq='YS', tz=tz)
        date = tm.box_expected(date, box_with_array, False)
        mth: Any = getattr(date, op)
        result: Any = mth(offset)
        expected: DatetimeIndex = DatetimeIndex(exp, tz=tz).as_unit('ns')
        expected = tm.box_expected(expected, box_with_array, False)
        tm.assert_equal(result, expected)

    def test_dt64arr_series_add_DateOffset_with_milli(self) -> None:
        dti: DatetimeIndex = DatetimeIndex([
            '2000-01-01 00:00:00.012345678', 
            '2000-01-31 00:00:00.012345678', 
            '2000-02-29 00:00:00.012345678'
        ], dtype='datetime64[ns]')
        result: DatetimeIndex = dti + DateOffset(milliseconds=4)
        expected: DatetimeIndex = DatetimeIndex([
            '2000-01-01 00:00:00.016345678', 
            '2000-01-31 00:00:00.016345678', 
            '2000-02-29 00:00:00.016345678'
        ], dtype='datetime64[ns]')
        tm.assert_index_equal(result, expected)
        result = dti + DateOffset(days=1, milliseconds=4)
        expected = DatetimeIndex([
            '2000-01-02 00:00:00.016345678', 
            '2000-02-01 00:00:00.016345678', 
            '2000-03-01 00:00:00.016345678'
        ], dtype='datetime64[ns]')
        tm.assert_index_equal(result, expected)

class TestDatetime64OverflowHandling:
    def test_dt64_overflow_masking(self, box_with_array: Any) -> None:
        left: Series = Series([Timestamp('1969-12-31')], dtype='M8[ns]')
        right: Series = Series([NaT])
        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        expected: TimedeltaIndex = TimedeltaIndex([NaT], dtype='m8[ns]')
        expected = tm.box_expected(expected, box_with_array)
        result: Any = left - right
        tm.assert_equal(result, expected)

    def test_dt64_series_arith_overflow(self) -> None:
        dt: Timestamp = Timestamp('1700-01-31')
        td: Timedelta = Timedelta('20000 Days')
        dti: DatetimeIndex = date_range('1949-09-30', freq='100YE', periods=4)
        ser: Series = Series(dti)
        msg: str = 'Overflow in int64 addition'
        with pytest.raises(OverflowError, match=msg):
            ser - dt
        with pytest.raises(OverflowError, match=msg):
            dt - ser
        with pytest.raises(OverflowError, match=msg):
            ser + td
        with pytest.raises(OverflowError, match=msg):
            td + ser
        ser.iloc[-1] = NaT
        expected: Series = Series(['2004-10-03', '2104-10-04', '2204-10-04', 'NaT'], dtype='datetime64[ns]')
        res: Series = ser + td
        tm.assert_series_equal(res, expected)
        res = td + ser
        tm.assert_series_equal(res, expected)
        ser.iloc[1:] = NaT
        expected = Series(['91279 Days', 'NaT', 'NaT', 'NaT'], dtype='timedelta64[ns]')
        res = ser - dt
        tm.assert_series_equal(res, expected)
        res = dt - ser
        tm.assert_series_equal(res, -expected)

    def test_datetimeindex_sub_timestamp_overflow(self) -> None:
        dtimax: DatetimeIndex = pd.to_datetime(['2021-12-28 17:19', Timestamp.max]).as_unit('ns')
        dtimin: DatetimeIndex = pd.to_datetime(['2021-12-28 17:19', Timestamp.min]).as_unit('ns')
        tsneg: Timestamp = Timestamp('1950-01-01').as_unit('ns')
        ts_neg_variants: List[Any] = [
            tsneg, tsneg.to_pydatetime(), tsneg.to_datetime64().astype('datetime64[ns]'), tsneg.to_datetime64().astype('datetime64[D]')
        ]
        tspos: Timestamp = Timestamp('1980-01-01').as_unit('ns')
        ts_pos_variants: List[Any] = [
            tspos, tspos.to_pydatetime(), tspos.to_datetime64().astype('datetime64[ns]'), tspos.to_datetime64().astype('datetime64[D]')
        ]
        msg: str = 'Overflow in int64 addition'
        for variant in ts_neg_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimax - variant
        expected_value: int = Timestamp.max._value - tspos._value
        for variant in ts_pos_variants:
            res = dtimax - variant
            assert res[1]._value == expected_value
        expected_value = Timestamp.min._value - tsneg._value
        for variant in ts_neg_variants:
            res = dtimin - variant
            assert res[1]._value == expected_value
        for variant in ts_pos_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimin - variant
        tmin: DatetimeIndex = pd.to_datetime([Timestamp.min])
        t1: Any = tmin + Timedelta.max + Timedelta('1us')
        with pytest.raises(OverflowError, match=msg):
            t1 - tmin
        tmax: DatetimeIndex = pd.to_datetime([Timestamp.max])
        t2: Any = tmax + Timedelta.min - Timedelta('1us')
        with pytest.raises(OverflowError, match=msg):
            tmax - t2

    def test_datetimeindex_sub_datetimeindex_overflow(self) -> None:
        dtimax: DatetimeIndex = pd.to_datetime(['2021-12-28 17:19', Timestamp.max]).as_unit('ns')
        dtimin: DatetimeIndex = pd.to_datetime(['2021-12-28 17:19', Timestamp.min]).as_unit('ns')
        ts_neg: DatetimeIndex = pd.to_datetime(['1950-01-01', '1950-01-01']).as_unit('ns')
        ts_pos: DatetimeIndex = pd.to_datetime(['1980-01-01', '1980-01-01']).as_unit('ns')
        expected_value: int = Timestamp.max._value - ts_pos[1]._value
        result: Any = dtimax - ts_pos
        assert result[1]._value == expected_value
        expected_value = Timestamp.min._value - ts_neg[1]._value
        result = dtimin - ts_neg
        assert result[1]._value == expected_value
        msg: str = 'Overflow in int64 addition'
        with pytest.raises(OverflowError, match=msg):
            dtimax - ts_neg
        with pytest.raises(OverflowError, match=msg):
            dtimin - ts_pos
        tmin: DatetimeIndex = pd.to_datetime([Timestamp.min])
        t1: Any = tmin + Timedelta.max + Timedelta('1us')
        with pytest.raises(OverflowError, match=msg):
            t1 - tmin
        tmax: DatetimeIndex = pd.to_datetime([Timestamp.max])
        t2: Any = tmax + Timedelta.min - Timedelta('1us')
        with pytest.raises(OverflowError, match=msg):
            tmax - t2

class TestTimestampSeriesArithmetic:
    def test_empty_series_add_sub(self, box_with_array: Any) -> None:
        a: Series = Series(dtype='M8[ns]')
        b: Series = Series(dtype='m8[ns]')
        a = box_with_array(a)
        b = box_with_array(b)
        tm.assert_equal(a, a + b)
        tm.assert_equal(a, a - b)
        tm.assert_equal(a, b + a)
        msg: str = 'cannot subtract'
        with pytest.raises(TypeError, match=msg):
            b - a

    def test_operators_datetimelike(self) -> None:
        td1: Series = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan
        dt1: Series = Series([Timestamp('20111230'), Timestamp('20120101'), Timestamp('20120103')])
        dt1.iloc[2] = np.nan
        dt2: Series = Series([Timestamp('20111231'), Timestamp('20120102'), Timestamp('20120104')])
        dt1 - dt2
        dt2 - dt1
        dt1 + td1
        td1 + dt1
        dt1 - td1
        td1 + dt1
        dt1 + td1

    def test_dt64ser_sub_datetime_dtype(self, unit: str) -> None:
        ts: Timestamp = Timestamp(datetime(1993, 1, 7, 13, 30, 0))
        dt: datetime = datetime(1993, 6, 22, 13, 30)
        ser: Series = Series([ts], dtype=f'M8[{unit}]')
        result: Any = ser - dt
        exp_unit: str = tm.get_finest_unit(unit, 'us')
        assert result.dtype == f'timedelta64[{exp_unit}]'

    @pytest.mark.parametrize('left, right, op_fail', [
        [
            [Timestamp('20111230'), Timestamp('20120101'), NaT],
            [Timestamp('20111231'), Timestamp('20120102'), Timestamp('20120104')],
            ['__sub__', '__rsub__']
        ],
        [
            [Timestamp('20111230'), Timestamp('20120101'), NaT],
            [timedelta(minutes=5, seconds=3), timedelta(minutes=5, seconds=3), NaT],
            ['__add__', '__radd__', '__sub__']
        ],
        [
            [Timestamp('20111230', tz='US/Eastern'), Timestamp('20111230', tz='US/Eastern'), NaT],
            [timedelta(minutes=5, seconds=3), NaT, timedelta(minutes=5, seconds=3)],
            ['__add__', '__radd__', '__sub__']
        ]
    ])
    def test_operators_datetimelike_invalid(self, left: List[Any], right: List[Any], op_fail: List[str], all_arithmetic_operators: str) -> None:
        op_str: str = all_arithmetic_operators
        arg1: Series = Series(left)
        arg2: Series = Series(right)
        op: Any = getattr(arg1, op_str, None)
        if op_str not in op_fail:
            with pytest.raises(TypeError, match='operate|[cC]annot|unsupported operand'):
                op(arg2)
        else:
            op(arg2)

    def test_sub_single_tz(self, unit: str) -> None:
        s1: Series = Series([Timestamp('2016-02-10', tz='America/Sao_Paulo')]).dt.as_unit(unit)
        s2: Series = Series([Timestamp('2016-02-08', tz='America/Sao_Paulo')]).dt.as_unit(unit)
        result: Any = s1 - s2
        expected: Series = Series([Timedelta('2days')]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)
        result = s2 - s1
        expected = Series([Timedelta('-2days')]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)

    def test_dt64tz_series_sub_dtitz(self) -> None:
        dti: DatetimeIndex = date_range('1999-09-30', periods=10, tz='US/Eastern')
        ser: Series = Series(dti)
        expected: Series = Series(TimedeltaIndex(['0days'] * 10))
        res: Any = dti - ser
        tm.assert_series_equal(res, expected)
        res = ser - dti
        tm.assert_series_equal(res, expected)

    def test_sub_datetime_compat(self, unit: str) -> None:
        ser: Series = Series([datetime(2016, 8, 23, 12, tzinfo=timezone.utc), NaT]).dt.as_unit(unit)
        dt: datetime = datetime(2016, 8, 22, 12, tzinfo=timezone.utc)
        exp_unit: str = tm.get_finest_unit(unit, 'us')
        exp: Series = Series([Timedelta('1 days'), NaT]).dt.as_unit(exp_unit)
        result: Any = ser - dt
        tm.assert_series_equal(result, exp)
        result2: Any = ser - Timestamp(dt)
        tm.assert_series_equal(result2, exp)

    def test_dt64_series_add_mixed_tick_DateOffset(self) -> None:
        s: Series = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        result: Series = s + pd.offsets.Milli(5)
        result2: Series = pd.offsets.Milli(5) + s
        expected: Series = Series([Timestamp('20130101 9:01:00.005'), Timestamp('20130101 9:02:00.005')])
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)
        result = s + pd.offsets.Minute(5) + pd.offsets.Milli(5)
        expected = Series([Timestamp('20130101 9:06:00.005'), Timestamp('20130101 9:07:00.005')])
        tm.assert_series_equal(result, expected)

    def test_datetime64_ops_nat(self, unit: str) -> None:
        datetime_series: Series = Series([NaT, Timestamp('19900315')]).dt.as_unit(unit)
        nat_series_dtype_timestamp: Series = Series([NaT, NaT], dtype=f'datetime64[{unit}]')
        single_nat_dtype_datetime: Series = Series([NaT], dtype=f'datetime64[{unit}]')
        tm.assert_series_equal(-NaT + datetime_series, nat_series_dtype_timestamp)
        msg: str = "bad operand type for unary -: 'DatetimeArray'"
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + datetime_series
        tm.assert_series_equal(-NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + nat_series_dtype_timestamp
        tm.assert_series_equal(nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp)
        tm.assert_series_equal(NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
        tm.assert_series_equal(nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp)
        tm.assert_series_equal(NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp)

    def test_operators_datetimelike_with_timezones(self) -> None:
        tz: str = 'US/Eastern'
        dt1: Series = Series(date_range('2000-01-01 09:00:00', periods=5, tz=tz), name='foo')
        dt2: Series = dt1.copy()
        dt2.iloc[2] = np.nan
        td1: Series = Series(pd.timedelta_range('1 days 1 min', periods=5, freq='h'))
        td2: Series = td1.copy()
        td2.iloc[1] = np.nan
        assert td2._values.freq is None
        result: Any = dt1 + td1[0]
        exp: Any = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt2 + td2[0]
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = td1[0] + dt1
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = td2[0] + dt2
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt1 - td1[0]
        exp = (dt1.dt.tz_localize(None) - td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg: str = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            td1[0] - dt1
        result = dt2 - td2[0]
        exp = (dt2.dt.tz_localize(None) - td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        with pytest.raises(TypeError, match=msg):
            td2[0] - dt2
        result = dt1 + td1
        exp = (dt1.dt.tz_localize(None) + td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt2 + td2
        exp = (dt2.dt.tz_localize(None) + td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt1 - td1
        exp = (dt1.dt.tz_localize(None) - td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result = dt2 - td2
        exp = (dt2.dt.tz_localize(None) - td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg = 'cannot (add|subtract)'
        with pytest.raises(TypeError, match=msg):
            td1 - dt1
        with pytest.raises(TypeError, match=msg):
            td2 - dt2

class TestDatetimeIndexArithmetic:
    def test_dti_add_tdi(self, tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        tdi: TimedeltaIndex = pd.timedelta_range('0 days', periods=10)
        expected: DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz)
        expected = expected._with_freq(None)
        result: Any = dti + tdi
        tm.assert_index_equal(result, expected)
        result = tdi + dti
        tm.assert_index_equal(result, expected)
        result = dti + tdi.values
        tm.assert_index_equal(result, expected)
        result = tdi.values + dti
        tm.assert_index_equal(result, expected)

    def test_dti_iadd_tdi(self, tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        tdi: TimedeltaIndex = pd.timedelta_range('0 days', periods=10)
        expected: DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz)
        expected = expected._with_freq(None)
        result: DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        result += tdi
        tm.assert_index_equal(result, expected)
        result = pd.timedelta_range('0 days', periods=10)
        result += dti
        tm.assert_index_equal(result, expected)
        result = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        result += tdi.values
        tm.assert_index_equal(result, expected)
        result = pd.timedelta_range('0 days', periods=10)
        result += dti
        tm.assert_index_equal(result, expected)

    def test_dti_sub_tdi(self, tz_naive_fixture: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        tdi: TimedeltaIndex = pd.timedelta_range('0 days', periods=10)
        expected: DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz, freq='-1D')
        expected = expected._with_freq(None)
        result: Any = dti - tdi
        tm.assert_index_equal(result, expected)
        msg: str = 'cannot subtract .*TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi - dti
        result = dti - tdi.values
        tm.assert_index_equal(result, expected)
        msg = 'cannot subtract a datelike from a TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi.values - dti

    def test_dti_isub_tdi(self, tz_naive_fixture: Any, unit: str) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz).as_unit(unit)
        tdi: TimedeltaIndex = pd.timedelta_range('0 days', periods=10, unit=unit)
        expected: DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz, freq='-1D', unit=unit)
        expected = expected._with_freq(None)
        result: DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10).as_unit(unit)
        result -= tdi
        tm.assert_index_equal(result, expected)
        dta: Any = dti._data.copy()
        dta -= tdi
        tm.assert_datetime_array_equal(dta, expected._data)
        out: Any = dti._data.copy()
        np.subtract(out, tdi, out=out)
        tm.assert_datetime_array_equal(out, expected._data)
        msg: str = 'cannot subtract a datelike from a TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi -= dti
        result = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10).as_unit(unit)
        result -= tdi.values
        tm.assert_index_equal(result, expected)
        with pytest.raises(TypeError, match=msg):
            tdi.values -= dti
        with pytest.raises(TypeError, match=msg):
            tdi._values -= dti

    def test_dta_add_sub_index(self, tz_naive_fixture: Any) -> None:
        dti: DatetimeIndex = date_range('20130101', periods=3, tz=tz_naive_fixture)
        dta: Any = dti.array
        result: Any = dta - dti
        expected: DatetimeIndex = dti - dti
        tm.assert_index_equal(result, expected)
        tdi: Any = result
        result = dta + tdi
        expected = dti + tdi
        tm.assert_index_equal(result, expected)
        result = dta - tdi
        expected = dti - tdi
        tm.assert_index_equal(result, expected)

    def test_sub_dti_dti(self, unit: str) -> None:
        dti: DatetimeIndex = date_range('20130101', periods=3, unit=unit)
        dti_tz: DatetimeIndex = date_range('20130101', periods=3, unit=unit).tz_localize('US/Eastern')
        expected: TimedeltaIndex = TimedeltaIndex([0, 0, 0]).as_unit(unit)
        result: Any = dti - dti
        tm.assert_index_equal(result, expected)
        result = dti_tz - dti_tz
        tm.assert_index_equal(result, expected)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            dti_tz - dti
        with pytest.raises(TypeError, match=msg):
            dti - dti_tz
        dti -= dti
        tm.assert_index_equal(dti, expected)
        dti1: DatetimeIndex = date_range('20130101', periods=3, unit=unit)
        dti2: DatetimeIndex = date_range('20130101', periods=4, unit=unit)
        msg = 'cannot add indices of unequal length'
        with pytest.raises(ValueError, match=msg):
            dti1 - dti2
        dti1 = DatetimeIndex(['2012-01-01', np.nan, '2012-01-03']).as_unit(unit)
        dti2 = DatetimeIndex(['2012-01-02', '2012-01-03', np.nan]).as_unit(unit)
        expected = TimedeltaIndex(['1 days', np.nan, np.nan]).as_unit(unit)
        result = dti2 - dti1
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('op', [operator.add, operator.sub])
    def test_timedelta64_equal_timedelta_supported_ops(self, op: Any, box_with_array: Any) -> None:
        ser: Series = Series([Timestamp('20130301'), Timestamp('20130228 23:00:00'), Timestamp('20130228 22:00:00'), Timestamp('20130228 21:00:00')])
        obj: Any = box_with_array(ser)
        intervals: List[str] = ['D', 'h', 'm', 's', 'us']
        def timedelta64(*args: Any) -> np.timedelta64:
            return np.sum(list(starmap(np.timedelta64, zip(args, intervals))))
        for d, h, m, s, us in product(*[range(2)] * 5):
            nptd: np.timedelta64 = timedelta64(d, h, m, s, us)
            pytd: timedelta = timedelta(days=d, hours=h, minutes=m, seconds=s, microseconds=us)
            lhs: Any = op(obj, nptd)
            rhs: Any = op(obj, pytd)
            tm.assert_equal(lhs, rhs)

    def test_ops_nat_mixed_datetime64_timedelta64(self) -> None:
        timedelta_series: Series = Series([NaT, Timedelta('1s')])
        datetime_series: Series = Series([NaT, Timestamp('19900315')])
        nat_series_dtype_timedelta: Series = Series([NaT, NaT], dtype='timedelta64[ns]')
        nat_series_dtype_timestamp: Series = Series([NaT, NaT], dtype='datetime64[ns]')
        single_nat_dtype_datetime: Series = Series([NaT], dtype='datetime64[ns]')
        single_nat_dtype_timedelta: Series = Series([NaT], dtype='timedelta64[ns]')
        tm.assert_series_equal(datetime_series - single_nat_dtype_datetime, nat_series_dtype_timedelta)
        tm.assert_series_equal(datetime_series - single_nat_dtype_timedelta, nat_series_dtype_timestamp)
        tm.assert_series_equal(-single_nat_dtype_timedelta + datetime_series, nat_series_dtype_timestamp)
        tm.assert_series_equal(nat_series_dtype_timestamp - single_nat_dtype_datetime, nat_series_dtype_timedelta)
        tm.assert_series_equal(nat_series_dtype_timestamp - single_nat_dtype_timedelta, nat_series_dtype_timestamp)
        tm.assert_series_equal(-single_nat_dtype_timedelta + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
        msg: str = 'cannot subtract a datelike'
        with pytest.raises(TypeError, match=msg):
            timedelta_series - single_nat_dtype_datetime
        tm.assert_series_equal(nat_series_dtype_timestamp + single_nat_dtype_timedelta, nat_series_dtype_timestamp)
        tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
        tm.assert_series_equal(nat_series_dtype_timestamp + single_nat_dtype_timedelta, nat_series_dtype_timestamp)
        tm.assert_series_equal(single_nat_dtype_timedelta + nat_series_dtype_timestamp, nat_series_dtype_timestamp)
        tm.assert_series_equal(nat_series_dtype_timedelta + single_nat_dtype_datetime, nat_series_dtype_timestamp)
        tm.assert_series_equal(single_nat_dtype_datetime + nat_series_dtype_timedelta, nat_series_dtype_timestamp)

    def test_ufunc_coercions(self, unit: str) -> None:
        idx: DatetimeIndex = date_range('2011-01-01', periods=3, freq='2D', name='x', unit=unit)
        delta: np.timedelta64 = np.timedelta64(1, 'D')
        exp: DatetimeIndex = date_range('2011-01-02', periods=3, freq='2D', name='x', unit=unit)
        for result in [idx + delta, np.add(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == '2D'
        exp = date_range('2010-12-31', periods=3, freq='2D', name='x', unit=unit)
        for result in [idx - delta, np.subtract(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == '2D'
        idx = idx._with_freq(None)
        delta = np.array([np.timedelta64(1, 'D'), np.timedelta64(2, 'D'), np.timedelta64(3, 'D')])
        exp = DatetimeIndex(['2011-01-02', '2011-01-05', '2011-01-08'], name='x').as_unit(unit)
        for result in [idx + delta, np.add(idx, delta)]:
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq
        exp = DatetimeIndex(['2010-12-31', '2011-01-01', '2011-01-02'], name='x').as_unit(unit)
        for result in [idx - delta, np.subtract(idx, delta)]:
            assert isinstance(result, DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

    def test_dti_add_series(self, tz_naive_fixture: Any, names: List[str]) -> None:
        tz: Any = tz_naive_fixture
        index: DatetimeIndex = DatetimeIndex(['2016-06-28 05:30', '2016-06-28 05:31'], tz=tz, name=names[0]).as_unit('ns')
        ser: Series = Series([Timedelta(seconds=5)] * 2, index=index, name=names[1])
        expected: Series = Series(index + Timedelta(seconds=5), index=index, name=names[2])
        expected.name = names[2]
        assert expected.dtype == index.dtype
        result: Series = ser + index
        tm.assert_series_equal(result, expected)
        result = index + ser
        tm.assert_series_equal(result, expected)
        expected = index + Timedelta(seconds=5)
        result = ser.values + index
        tm.assert_index_equal(result, expected)
        result = index + ser.values
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
    def test_dti_addsub_offset_arraylike(self, performance_warning: Any, tz_naive_fixture: Any, names: List[str], op: Any, index_or_series: Any) -> None:
        other_box: Any = index_or_series
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = date_range('2017-01-01', periods=2, tz=tz, name=names[0])
        other: Any = other_box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)], name=names[1])
        xbox: Any = get_upcast_box(dti, other)
        with tm.assert_produces_warning(performance_warning):
            res: Any = op(dti, other)
        expected: DatetimeIndex = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))], name=names[2], freq='infer')
        expected = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize('other_box', [pd.Index, np.array])
    def test_dti_addsub_object_arraylike(self, performance_warning: Any, tz_naive_fixture: Any, box_with_array: Any, other_box: Any) -> None:
        tz: Any = tz_naive_fixture
        dti: DatetimeIndex = date_range('2017-01-01', periods=2, tz=tz)
        dtarr: Any = tm.box_expected(dti, box_with_array)
        other: Any = other_box([pd.offsets.MonthEnd(), Timedelta(days=4)])
        xbox: Any = get_upcast_box(dtarr, other)
        expected: DatetimeIndex = DatetimeIndex(['2017-01-31', '2017-01-06'], tz=tz_naive_fixture)
        expected = tm.box_expected(expected, xbox).astype(object)
        with tm.assert_produces_warning(performance_warning):
            result: Any = dtarr + other
        tm.assert_equal(result, expected)
        expected = DatetimeIndex(['2016-12-31', '2016-12-29'], tz=tz_naive_fixture)
        expected = tm.box_expected(expected, xbox).astype(object)
        with tm.assert_produces_warning(performance_warning):
            result = dtarr - other
        tm.assert_equal(result, expected)

@pytest.mark.parametrize('years', [-1, 0, 1])
@pytest.mark.parametrize('months', [-2, 0, 2])
def test_shift_months(years: int, months: int, unit: str) -> None:
    dti: DatetimeIndex = DatetimeIndex([
        Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'),
        Timestamp('2000-01-01'), Timestamp('2000-02-29'), Timestamp('2000-12-31')
    ]).as_unit(unit)
    shifted: np.ndarray = shift_months(dti.asi8, years * 12 + months, reso=dti._data._creso)
    shifted_dt64: np.ndarray = shifted.view(f'M8[{dti.unit}]')
    actual: DatetimeIndex = DatetimeIndex(shifted_dt64)
    raw: List[Timestamp] = [x + pd.offsets.DateOffset(years=years, months=months) for x in dti]
    expected: DatetimeIndex = DatetimeIndex(raw).as_unit(dti.unit)
    tm.assert_index_equal(actual, expected)

def test_dt64arr_addsub_object_dtype_2d(performance_warning: Any) -> None:
    dti: DatetimeIndex = date_range('1994-02-13', freq='2W', periods=4)
    dta: Any = dti._data.reshape((4, 1))
    other: np.ndarray = np.array([[pd.offsets.Day(n)] for n in range(4)])
    assert other.shape == dta.shape
    with tm.assert_produces_warning(performance_warning):
        result: Any = dta + other
    with tm.assert_produces_warning(performance_warning):
        expected: Any = (dta[:, 0] + other[:, 0]).reshape(-1, 1)
    tm.assert_numpy_array_equal(result, expected)
    with tm.assert_produces_warning(performance_warning):
        result2: Any = dta - dta.astype(object)
    assert result2.shape == (4, 1)
    assert all((td._value == 0 for td in result2.ravel()))

def test_non_nano_dt64_addsub_np_nat_scalars() -> None:
    ser: Series = Series([1233242342344, 232432434324, 332434242344], dtype='datetime64[ms]')
    result: Series = ser - np.datetime64('nat', 'ms')
    expected: Series = Series([NaT] * 3, dtype='timedelta64[ms]')
    tm.assert_series_equal(result, expected)
    result = ser + np.timedelta64('nat', 'ms')
    expected = Series([NaT] * 3, dtype='datetime64[ms]')
    tm.assert_series_equal(result, expected)

def test_non_nano_dt64_addsub_np_nat_scalars_unitless() -> None:
    ser: Series = Series([1233242342344, 232432434324, 332434242344], dtype='datetime64[ms]')
    result: Series = ser - np.datetime64('nat')
    expected: Series = Series([NaT] * 3, dtype='timedelta64[ns]')
    tm.assert_series_equal(result, expected)
    result = ser + np.timedelta64('nat')
    expected = Series([NaT] * 3, dtype='datetime64[ns]')
    tm.assert_series_equal(result, expected)

def test_non_nano_dt64_addsub_np_nat_scalars_unsupported_unit() -> None:
    ser: Series = Series([12332, 23243, 33243], dtype='datetime64[s]')
    result: Series = ser - np.datetime64('nat', 'D')
    expected: Series = Series([NaT] * 3, dtype='timedelta64[s]')
    tm.assert_series_equal(result, expected)
    result = ser + np.timedelta64('nat', 'D')
    expected = Series([NaT] * 3, dtype='datetime64[s]')
    tm.assert_series_equal(result, expected)