from datetime import datetime, time, timedelta, timezone
from itertools import product, starmap
import operator
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

    def test_compare_zerodim(self, tz_naive_fixture: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_naive_fixture
        box = box_with_array
        dti = date_range('20130101', periods=3, tz=tz)
        other = np.array(dti.to_numpy()[0])
        dtarr = tm.box_expected(dti, box)
        xbox = get_upcast_box(dtarr, other, True)
        result = dtarr <= other
        expected = np.array([True, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('other', ['foo', -1, 99, 4.0, object(), timedelta(days=2), datetime(2001, 1, 1).date(), None, np.nan])
    def test_dt64arr_cmp_scalar_invalid(self, other: Any, tz_naive_fixture: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_naive_fixture
        rng = date_range('1/1/2000', periods=10, tz=tz)
        dtarr = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(dtarr, other, box_with_array)

    @pytest.mark.parametrize('other', [list(range(10)), np.arange(10), np.arange(10).astype(np.float32), np.arange(10).astype(object), pd.timedelta_range('1ns', periods=10).array, np.array(pd.timedelta_range('1ns', periods=10)), list(pd.timedelta_range('1ns', periods=10)), pd.timedelta_range('1 Day', periods=10).astype(object), pd.period_range('1971-01-01', freq='D', periods=10).array, pd.period_range('1971-01-01', freq='D', periods=10).astype(object)])
    def test_dt64arr_cmp_arraylike_invalid(self, other: Any, tz_naive_fixture: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_naive_fixture
        dta = date_range('1970-01-01', freq='ns', periods=10, tz=tz)._data
        obj = tm.box_expected(dta, box_with_array)
        assert_invalid_comparison(obj, other, box_with_array)

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture: Optional[str]) -> None:
        tz = tz_naive_fixture
        dta = date_range('1970-01-01', freq='h', periods=5, tz=tz)._data
        other = np.array([0, 1, 2, dta[3], Timedelta(days=1)])
        result = dta == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result = dta != other
        tm.assert_numpy_array_equal(result, ~expected)
        msg = 'Invalid comparison between|Cannot compare type|not supported between'
        with pytest.raises(TypeError, match=msg):
            dta < other
        with pytest.raises(TypeError, match=msg):
            dta > other
        with pytest.raises(TypeError, match=msg):
            dta <= other
        with pytest.raises(TypeError, match=msg):
            dta >= other

    def test_dt64arr_nat_comparison(self, tz_naive_fixture: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_naive_fixture
        box = box_with_array
        ts = Timestamp('2021-01-01', tz=tz)
        ser = Series([ts, NaT])
        obj = tm.box_expected(ser, box)
        xbox = get_upcast_box(obj, ts, True)
        expected = Series([True, False], dtype=np.bool_)
        expected = tm.box_expected(expected, xbox)
        result = obj == ts
        tm.assert_equal(result, expected)

class TestDatetime64SeriesComparison:

    @pytest.mark.parametrize('pair', [([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [NaT, NaT, Timestamp('2011-01-03')]), ([Timedelta('1 days'), NaT, Timedelta('3 days')], [NaT, NaT, Timedelta('3 days')]), ([Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')], [NaT, NaT, Period('2011-03', freq='M')])])
    @pytest.mark.parametrize('reverse', [True, False])
    @pytest.mark.parametrize('dtype', [None, object])
    @pytest.mark.parametrize('op, expected', [(operator.eq, [False, False, True]), (operator.ne, [True, True, False]), (operator.lt, [False, False, False]), (operator.gt, [False, False, False]), (operator.ge, [False, False, True]), (operator.le, [False, False, True])])
    def test_nat_comparisons(self, dtype: Optional[str], index_or_series: Callable[[Index], Box], reverse: bool, pair: Tuple[List[Any], List[Any]], op: Callable, expected: List[bool]) -> None:
        box = index_or_series
        lhs, rhs = pair
        if reverse:
            lhs, rhs = (rhs, lhs)
        left = Series(lhs, dtype=dtype)
        right = box(rhs, dtype=dtype)
        result = op(left, right)
        tm.assert_series_equal(result, Series(expected))

    @pytest.mark.parametrize('data', [[Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [Timedelta('1 days'), NaT, Timedelta('3 days')], [Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')]])
    @pytest.mark.parametrize('dtype', [None, object])
    def test_nat_comparisons_scalar(self, dtype: Optional[str], data: List[Any], box_with_array: Callable[[Index], Box]) -> None:
        box = box_with_array
        left = Series(data, dtype=dtype)
        left = tm.box_expected(left, box)
        xbox = get_upcast_box(left, NaT, True)
        expected = [False, False, False]
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
    def test_series_comparison_scalars(self, val: datetime) -> None:
        series = Series(date_range('1/1/2000', periods=10))
        result = series > val
        expected = Series([x > val for x in series])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('left,right', [('lt', 'gt'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
    def test_timestamp_compare_series(self, left: str, right: str) -> None:
        ser = Series(date_range('20010101', periods=10), name='dates')
        s_nat = ser.copy(deep=True)
        ser[0] = Timestamp('nat')
        ser[3] = Timestamp('nat')
        left_f = getattr(operator, left)
        right_f = getattr(operator, right)
        expected = left_f(ser, Timestamp('20010109'))
        result = right_f(Timestamp('20010109'), ser)
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

    def test_dt64arr_timestamp_equality(self, box_with_array: Callable[[Index], Box]) -> None:
        box = box_with_array
        ser = Series([Timestamp('2000-01-29 01:59:00'), Timestamp('2000-01-30'), NaT])
        ser = tm.box_expected(ser, box)
        xbox = get_upcast_box(ser, ser, True)
        result = ser != ser
        expected = tm.box_expected([False, False, True], xbox)
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

    @pytest.mark.parametrize('datetimelike', [Timestamp('20130101'), datetime(2013, 1, 1), np.datetime64('2013-01-01T00:00', 'ns')])
    @pytest.mark.parametrize('op,expected', [(operator.lt, [True, False, False, False]), (operator.le, [True, True, False, False]), (operator.eq, [False, True, False, False]), (operator.gt, [False, False, False, True])])
    def test_dt64_compare_datetime_scalar(self, datetimelike: Any, op: Callable, expected: List[bool]) -> None:
        ser = Series([Timestamp('20120101'), Timestamp('20130101'), np.nan, Timestamp('20130103')], name='A')
        result = op(ser, datetimelike)
        expected = Series(expected, name='A')
        tm.assert_series_equal(result, expected)

    def test_ts_series_numpy_maximum(self) -> None:
        ts = Timestamp('2024-07-01')
        ts_series = Series(['2024-06-01', '2024-07-01', '2024-08-01'], dtype='datetime64[us]')
        expected = Series(['2024-07-01', '2024-07-01', '2024-08-01'], dtype='datetime64[us]')
        tm.assert_series_equal(expected, np.maximum(ts, ts_series))

class TestDatetimeIndexComparisons:

    def test_comparators(self, comparison_op: Callable) -> None:
        index = date_range('2020-01-01', periods=10)
        element = index[len(index) // 2]
        element = Timestamp(element).to_datetime64()
        arr = np.array(index)
        arr_result = comparison_op(arr, element)
        index_result = comparison_op(index, element)
        assert isinstance(index_result, np.ndarray)
        tm.assert_numpy_array_equal(arr_result, index_result)

    @pytest.mark.parametrize('other', [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
    def test_dti_cmp_datetimelike(self, other: Any, tz_naive_fixture: Optional[str]) -> None:
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=2, tz=tz)
        if tz is not None:
            if isinstance(other, np.datetime64):
                pytest.skip(f'{type(other).__name__} is not tz aware')
            other = localize_pydatetime(other, dti.tzinfo)
        result = dti == other
        expected = np.array([True, False])
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
    def test_dti_cmp_nat(self, dtype: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        left = DatetimeIndex([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')])
        right = DatetimeIndex([NaT, NaT, Timestamp('2011-01-03')])
        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        xbox = get_upcast_box(left, right, True)
        lhs, rhs = (left, right)
        if dtype is object:
            lhs, rhs = (left.astype(object), right.astype(object))
        result = rhs == lhs
        expected = np.array([False, False, True])
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
        fidx1 = pd.Index([1.0, np.nan, 3.0, np.nan, 5.0, 7.0])
        fidx2 = pd.Index([2.0, 3.0, np.nan, np.nan, 6.0, 7.0])
        didx1 = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
        didx2 = DatetimeIndex(['2014-02-01', '2014-03-01', NaT, NaT, '2014-06-01', '2014-07-01'])
        darr = np.array([np.datetime64('2014-02-01 00:00'), np.datetime64('2014-03-01 00:00'), np.datetime64('nat'), np.datetime64('nat'), np.datetime64('2014-06-01 00:00'), np.datetime64('2014-07-01 00:00')])
        cases = [(fidx1, fidx2), (didx1, didx2), (didx1, darr)]
        with tm.assert_produces_warning(None):
            for idx1, idx2 in cases:
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

    def test_comparison_tzawareness_compat(self, comparison_op: Callable, box_with_array: Callable[[Index], Box]) -> None:
        op = comparison_op
        box = box_with_array
        dr = date_range('2016-01-01', periods=6)
        dz = dr.tz_localize('US/Pacific')
        dr = tm.box_expected(dr, box)
        dz = tm.box_expected(dz, box)
        if box is pd.DataFrame:
            tolist = lambda x: x.astype(object).values.tolist()[0]
        else:
            tolist = list
        if op not in [operator.eq, operator.ne]:
            msg = 'Invalid comparison between dtype=datetime64\\[ns.*\\] and (Timestamp|DatetimeArray|list|ndarray)'
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

    def test_comparison_tzawareness_compat_scalars(self, comparison_op: Callable, box_with_array: Callable[[Index], Box]) -> None:
        op = comparison_op
        dr = date_range('2016-01-01', periods=6)
        dz = dr.tz_localize('US/Pacific')
        dr = tm.box_expected(dr, box_with_array)
        dz = tm.box_expected(dz, box_with_array)
        ts = Timestamp('2000-03-14 01:59')
        ts_tz = Timestamp('2000-03-14 01:59', tz='Europe/Amsterdam')
        assert np.all(dr > ts)
        msg = 'Invalid comparison between dtype=datetime64\\[ns.*\\] and Timestamp'
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
    def test_scalar_comparison_tzawareness(self, comparison_op: Callable, other: Any, tz_aware_fixture: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        op = comparison_op
        tz = tz_aware_fixture
        dti = date_range('2016-01-01', periods=2, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)
        xbox = get_upcast_box(dtarr, other, True)
        if op in [operator.eq, operator.ne]:
            exbool = op is operator.ne
            expected = np.array([exbool, exbool], dtype=bool)
            expected = tm.box_expected(expected, xbox)
            result = op(dtarr, other)
            tm.assert_equal(result, expected)
            result = op(other, dtarr)
            tm.assert_equal(result, expected)
        else:
            msg = f'Invalid comparison between dtype=datetime64\\[ns, .*\\] and {type(other).__name__}'
            with pytest.raises(TypeError, match=msg):
                op(dtarr, other)
            with pytest.raises(TypeError, match=msg):
                op(other, dtarr)

    def test_nat_comparison_tzawareness(self, comparison_op: Callable) -> None:
        op = comparison_op
        dti = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
        expected = np.array([op == operator.ne] * len(dti))
        result = op(dti, NaT)
        tm.assert_numpy_array_equal(result, expected)
        result = op(dti.tz_localize('US/Pacific'), NaT)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_str(self, tz_naive_fixture: Optional[str]) -> None:
        tz = tz_naive_fixture
        rng = date_range('1/1/2000', periods=10, tz=tz)
        other = '1/1/2000'
        result = rng == other
        expected = np.array([True] + [False] * 9)
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
        rng = date_range('1/1/2000', periods=10)
        result = rng == list(rng)
        expected = rng == rng
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other', [pd.timedelta_range('1D', periods=10), pd.timedelta_range('1D', periods=10).to_series(), pd.timedelta_range('1D', periods=10).asi8.view('m8[ns]')], ids=lambda x: type(x).__name__)
    def test_dti_cmp_tdi_tzawareness(self, other: Any) -> None:
        dti = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
        result = dti == other
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)
        result = dti != other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)
        msg = 'Invalid comparison between'
        with pytest.raises(TypeError, match=msg):
            dti < other
        with pytest.raises(TypeError, match=msg):
            dti <= other
        with pytest.raises(TypeError, match=msg):
            dti > other
        with pytest.raises(TypeError, match=msg):
            dti >= other

    def test_dti_cmp_object_dtype(self) -> None:
        dti = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
        other = dti.astype('O')
        result = dti == other
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)
        other = dti.tz_localize(None)
        result = dti != other
        tm.assert_numpy_array_equal(result, expected)
        other = np.array(list(dti[:5]) + [Timedelta(days=1)] * 5)
        result = dti == other
        expected = np.array([True] * 5 + [False] * 5)
        tm.assert_numpy_array_equal(result, expected)
        msg = ">=' not supported between instances of 'Timestamp' and 'Timedelta'"
        with pytest.raises(TypeError, match=msg):
            dti >= other

class TestDatetime64Arithmetic:

    @pytest.mark.arm_slow
    def test_dt64arr_add_timedeltalike_scalar(self, tz_naive_fixture: Optional[str], two_hours: Timedelta, box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_naive_fixture
        rng = date_range('2000-01-01', '2000-02-01', tz=tz)
        expected = date_range('2000-01-01 02:00', '2000-02-01 02:00', tz=tz)
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = rng + two_hours
        tm.assert_equal(result, expected)
        result = two_hours + rng
        tm.assert_equal(result, expected)
        rng += two_hours
        tm.assert_equal(rng, expected)

    def test_dt64arr_sub_timedeltalike_scalar(self, tz_naive_fixture: Optional[str], two_hours: Timedelta, box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_naive_fixture
        rng = date_range('2000-01-01', '2000-02-01', tz=tz)
        expected = date_range('1999-12-31 22:00', '2000-01-31 22:00', tz=tz)
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = rng - two_hours
        tm.assert_equal(result, expected)
        rng -= two_hours
        tm.assert_equal(rng, expected)

    def test_dt64_array_sub_dt_with_different_timezone(self, box_with_array: Callable[[Index], Box]) -> None:
        t1 = date_range('20130101', periods=3).tz_localize('US/Eastern')
        t1 = tm.box_expected(t1, box_with_array)
        t2 = Timestamp('20130101').tz_localize('CET')
        tnaive = Timestamp(20130101)
        result = t1 - t2
        expected = TimedeltaIndex(['0 days 06:00:00', '1 days 06:00:00', '2 days 06:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = t2 - t1
        expected = TimedeltaIndex(['-1 days +18:00:00', '-2 days +18:00:00', '-3 days +18:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64_array_sub_dt64_array_with_different_timezone(self, box_with_array: Callable[[Index], Box]) -> None:
        t1 = date_range('20130101', periods=3).tz_localize('US/Eastern')
        t1 = tm.box_expected(t1, box_with_array)
        t2 = date_range('20130101', periods=3).tz_localize('CET')
        t2 = tm.box_expected(t2, box_with_array)
        tnaive = date_range('20130101', periods=3)
        result = t1 - t2
        expected = TimedeltaIndex(['0 days 06:00:00', '0 days 06:00:00', '0 days 06:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result = t2 - t1
        expected = TimedeltaIndex(['-1 days +18:00:00', '-1 days +18:00:00', '-1 days +18:00:00'])
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64arr_add_sub_td64_nat(self, box_with_array: Callable[[Index], Box], tz_naive_fixture: Optional[str]) -> None:
        tz = tz_naive_fixture
        dti = date_range('1994-04-01', periods=9, tz=tz, freq='QS')
        other = np.timedelta64('NaT')
        expected = DatetimeIndex(['NaT'] * 9, tz=tz).as_unit('ns')
        obj = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = obj + other
        tm.assert_equal(result, expected)
        result = other + obj
        tm.assert_equal(result, expected)
        result = obj - other
        tm.assert_equal(result, expected)
        msg = 'cannot subtract'
        with pytest.raises(TypeError, match=msg):
            other - obj

    def test_dt64arr_add_sub_td64ndarray(self, tz_naive_fixture: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        tdi = TimedeltaIndex(['-1 Day', '-1 Day', '-1 Day'])
        tdarr = tdi.values
        expected = date_range('2015-12-31', '2016-01-02', periods=3, tz=tz)
        dtarr = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = dtarr + tdarr
        tm.assert_equal(result, expected)
        result = tdarr + dtarr
        tm.assert_equal(result, expected)
        expected = date_range('2016-01-02', '2016-01-04', periods=3, tz=tz)
        expected = tm.box_expected(expected, box_with_array)
        result = dtarr - tdarr
        tm.assert_equal(result, expected)
        msg = 'cannot subtract|(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            tdarr - dtarr

    @pytest.mark.parametrize('ts', [Timestamp('2013-01-01'), Timestamp('2013-01-01').to_pydatetime(), Timestamp('2013-01-01').to_datetime64(), np.datetime64('2013-01-01', 'D')])
    def test_dt64arr_sub_dtscalar(self, box_with_array: Callable[[Index], Box], ts: Any) -> None:
        idx = date_range('2013-01-01', periods=3)._with_freq(None)
        idx = tm.box_expected(idx, box_with_array)
        expected = TimedeltaIndex(['0 Days', '1 Day', '2 Days'])
        expected = tm.box_expected(expected, box_with_array)
        result = idx - ts
        tm.assert_equal(result, expected)
        result = ts - idx
        tm.assert_equal(result, -expected)
        tm.assert_equal(result, -expected)

    def test_dt64arr_sub_timestamp_tzaware(self, box_with_array: Callable[[Index], Box]) -> None:
        ser = date_range('2014-03-17', periods=2, freq='D', tz='US/Eastern')
        ser = ser._with_freq(None)
        ts = ser[0]
        ser = tm.box_expected(ser, box_with_array)
        delta_series = Series([np.timedelta64(0, 'D'), np.timedelta64(1, 'D')])
        expected = tm.box_expected(delta_series, box_with_array)
        tm.assert_equal(ser - ts, expected)
        tm.assert_equal(ts - ser, -expected)

    def test_dt64arr_sub_NaT(self, box_with_array: Callable[[Index], Box], unit: str) -> None:
        dti = DatetimeIndex([NaT, Timestamp('19900315')]).as_unit(unit)
        ser = tm.box_expected(dti, box_with_array)
        result = ser - NaT
        expected = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        dti_tz = dti.tz_localize('Asia/Tokyo')
        ser_tz = tm.box_expected(dti_tz, box_with_array)
        result = ser_tz - NaT
        expected = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dt64arr_sub_dt64object_array(self, performance_warning: Any, box_with_array: Callable[[Index], Box], tz_naive_fixture: Optional[str]) -> None:
        dti = date_range('2016-01-01', periods=3, tz=tz_naive_fixture)
        expected = dti - dti
        obj = tm.box_expected(dti, box_with_array)
        expected = tm.box_expected(expected, box_with_array).astype(object)
        with tm.assert_produces_warning(performance_warning):
            result = obj - obj.astype(object)
        tm.assert_equal(result, expected)

    def test_dt64arr_naive_sub_dt64ndarray(self, box_with_array: Callable[[Index], Box]) -> None:
        dti = date_range('2016-01-01', periods=3, tz=None)
        dt64vals = dti.values
        dtarr = tm.box_expected(dti, box_with_array)
        expected = dtarr - dtarr
        result = dtarr - dt64vals
        tm.assert_equal(result, expected)
        result = dt64vals - dtarr
        tm.assert_equal(result, expected)

    def test_dt64arr_aware_sub_dt64ndarray_raises(self, tz_aware_fixture: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_aware_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        dt64vals = dti.values
        dtarr = tm.box_expected(dti, box_with_array)
        msg = 'Cannot subtract tz-naive and tz-aware datetime'
        with pytest.raises(TypeError, match=msg):
            dtarr - dt64vals
        with pytest.raises(TypeError, match=msg):
            dt64vals - dtarr

    def test_dt64arr_add_dtlike_raises(self, tz_naive_fixture: Optional[str], box_with_array: Callable[[Index], Box]) -> None:
        tz = tz_naive_fixture
        dti = date_range('2016-01-01', periods=3, tz=tz)
        if tz is None:
            dti2 = dti.tz_localize('US/Eastern')
        else:
            dti2 = dti.tz_localize(None)
        dtarr = tm.box_expected(dti, box_with_array)
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
    def test_dt64arr_addsub_intlike(self, dtype: Optional[str], index_or_series_or_array: Callable[[Index], Box], freq: Optional[str], tz_naive_fixture: Optional[str]) -> None:
        tz = tz_naive_fixture
        if freq is None:
            dti = DatetimeIndex(['NaT', '2017-04-05 06:07:08'], tz=tz)
        else:
            dti = date_range('2016-01-01', periods=2, freq=freq, tz=tz)
        obj = index_or_series_or_array(dti)
        other = np.array([4, -1])
        if dtype is not None:
            other = other.astype(dtype)
        msg = '|'.join(['Addition/subtraction of integers', 'cannot subtract DatetimeArray from', 'can only perform ops with numeric values', 'unsupported operand type.*Categorical', "unsupported operand type\\(s\\) for -: 'int' and 'Timestamp'"])
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
    def test_dt64arr_add_sub_invalid(self, dti_freq: Optional[str], other: Any, box_with_array: Callable[[Index], Box]) -> None:
        dti = DatetimeIndex(['2011-01-01', '2011-01-02'], freq=dti_freq)
        dtarr = tm.box_expected(dti, box_with_array)
        msg = '|'.join(['unsupported operand type', 'cannot (add|subtract)', 'cannot use operands with types', "ufunc '?(add|subtract)'? cannot use operands with types", 'Concatenation operation is not implemented for NumPy arrays'])
        assert_invalid_addsub_type(dtarr, other, msg=msg)

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('dti_freq', [None, 'D'])
    def test_dt64arr_add_sub_parr(self, dti_freq: Optional[str], pi_freq: str, box_with_array: Callable[[Index], Box], box_with_array2: Callable[[Index], Box]) -> None:
        dti = DatetimeIndex(['2011-01-01', '2011-01-02'], freq=dti_freq)
        pi = dti.to_period(pi_freq)
        dtarr = tm.box_expected(dti, box_with_array)
        parr = tm.box_expected(pi, box_with_array2)
        msg = '|'.join(['cannot (add|subtract)', 'unsupported operand', 'descriptor.*requires', 'ufunc.*cannot use operands'])
        assert_invalid_addsub_type(dtarr, parr, msg=msg)

    @pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
    def test_dt64arr_addsub_time_objects_raises(self, box_with_array: Callable[[Index], Box], tz_naive_fixture: Optional[str]) -> None:
        tz = tz_naive_fixture
        obj1 = date_range('2012-01-01', periods=3, tz=tz)
        obj2 = [time(i, i, i) for i in range(3)]
        obj1 = tm.box_expected(obj1, box_with_array)
        obj2 = tm.box_expected(obj2, box_with_array)
        msg = '|'.join(['unsupported operand', 'cannot subtract DatetimeArray from ndarray'])
        assert_invalid_addsub_type(obj1, obj2, msg=msg)

    @pytest.mark.parametrize('dt64_series', [Series([Timestamp('19900315'), Timestamp('19900315')]), Series([NaT, Timestamp('19900315')]), Series([NaT, NaT], dtype='datetime64[ns]')])
    @pytest.mark.parametrize('one', [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(self, one: Any, dt64_series: Series, box_with_array: Callable[[Index], Box]) -> None:
        obj = tm.box_expected(dt64_series, box_with_array)
        msg = 'cannot perform .* with this index type'
        with pytest.raises(TypeError, match=msg):
            obj * one
        with pytest.raises(TypeError, match=msg):
            one * obj
        with pytest.raises(TypeError, match=msg):
            obj / one
        with pytest.raises(TypeError, match=msg):
            one / obj

class TestDatetime64DateOffsetArithmetic:

    def test_dt64arr_series_add_tick_DateOffset(self, box_with_array: Callable[[Index], Box], unit: str) -> None:
        ser = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')]).dt.as_unit(unit)
        expected = Series([Timestamp('20130101 9:01:05'), Timestamp('20130101 9:02:05')]).dt.as_unit(unit)
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)
        result = ser + pd.offsets.Second(5)
        tm.assert_equal(result, expected