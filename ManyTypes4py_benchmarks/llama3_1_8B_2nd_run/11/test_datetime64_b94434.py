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
    def test_compare_zerodim(self, tz_naive_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Compare zero-dimensional array"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        box: pd.Series = box_with_array
        dti: pd.DatetimeIndex = date_range('20130101', periods=3, tz=tz)
        other: np.ndarray = np.array(dti.to_numpy()[0])
        dtarr: pd.Series = tm.box_expected(dti, box)
        xbox: pd.Series = get_upcast_box(dtarr, other, True)
        result: pd.Series = dtarr <= other
        expected: np.ndarray = np.array([True, False, False])
        expected: pd.Series = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('other', ['foo', -1, 99, 4.0, object(), timedelta(days=2), datetime(2001, 1, 1).date(), None, np.nan])
    def test_dt64arr_cmp_scalar_invalid(self, other: object, tz_naive_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Invalid scalar comparisons"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        rng: pd.DatetimeIndex = date_range('1/1/2000', periods=10, tz=tz)
        dtarr: pd.Series = tm.box_expected(rng, box_with_array)
        assert_invalid_comparison(dtarr, other, box_with_array)

    @pytest.mark.parametrize('other', [list(range(10)), np.arange(10), np.arange(10).astype(np.float32), np.arange(10).astype(object), pd.timedelta_range('1ns', periods=10).array, np.array(pd.timedelta_range('1ns', periods=10)), list(pd.timedelta_range('1ns', periods=10)), pd.timedelta_range('1 Day', periods=10).astype(object), pd.period_range('1971-01-01', freq='D', periods=10).array, pd.period_range('1971-01-01', freq='D', periods=10).astype(object)])
    def test_dt64arr_cmp_arraylike_invalid(self, other: object, tz_naive_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Invalid array-like comparisons"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dta: pd.DatetimeIndex = date_range('1970-01-01', freq='ns', periods=10, tz=tz)._data
        obj: pd.Series = tm.box_expected(dta, box_with_array)
        assert_invalid_comparison(obj, other, box_with_array)

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Mixed type comparisons"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dta: pd.DatetimeIndex = date_range('1970-01-01', freq='h', periods=5, tz=tz)._data
        other: np.ndarray = np.array([0, 1, 2, dta[3], Timedelta(days=1)])
        result: pd.Series = dta == other
        expected: np.ndarray = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)
        result: pd.Series = dta != other
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

    def test_dt64arr_nat_comparison(self, tz_naive_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """NaT comparisons"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        box: pd.Series = box_with_array
        ts: pd.Timestamp = Timestamp('2021-01-01', tz=tz)
        ser: pd.Series = Series([ts, NaT])
        obj: pd.Series = tm.box_expected(ser, box)
        xbox: pd.Series = get_upcast_box(obj, ts, True)
        expected: pd.Series = Series([True, False], dtype=np.bool_)
        expected: pd.Series = tm.box_expected(expected, xbox)
        result: pd.Series = obj == ts
        tm.assert_equal(result, expected)

class TestDatetime64SeriesComparison:
    @pytest.mark.parametrize('pair', [([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [NaT, NaT, Timestamp('2011-01-03')]), ([Timedelta('1 days'), NaT, Timedelta('3 days')], [NaT, NaT, Timedelta('3 days')]), ([Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')], [NaT, NaT, Period('2011-03', freq='M')])])
    @pytest.mark.parametrize('reverse', [True, False])
    @pytest.mark.parametrize('dtype', [None, object])
    @pytest.mark.parametrize('op, expected', [(operator.eq, [False, False, True]), (operator.ne, [True, True, False]), (operator.lt, [False, False, False]), (operator.gt, [False, False, False]), (operator.ge, [False, False, True]), (operator.le, [False, False, True])])
    def test_nat_comparisons(self, dtype: object, index_or_series: pd.Series, reverse: bool, pair: tuple, op: operator, expected: list) -> None:
        """NaT comparisons"""
        box: pd.Series = index_or_series
        lhs, rhs: list = pair
        if reverse:
            lhs, rhs = (rhs, lhs)
        left: pd.Series = Series(lhs, dtype=dtype)
        right: pd.Series = box(rhs, dtype=dtype)
        result: pd.Series = op(left, right)
        tm.assert_series_equal(result, Series(expected))

    @pytest.mark.parametrize('data', [[Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')], [Timedelta('1 days'), NaT, Timedelta('3 days')], [Period('2011-01', freq='M'), NaT, Period('2011-03', freq='M')]])
    @pytest.mark.parametrize('dtype', [None, object])
    def test_nat_comparisons_scalar(self, dtype: object, data: list, box_with_array) -> None:
        """NaT comparisons with scalar"""
        box: pd.Series = box_with_array
        left: pd.Series = Series(data, dtype=dtype)
        left: pd.Series = tm.box_expected(left, box)
        xbox: pd.Series = get_upcast_box(left, NaT, True)
        expected: list = [False, False, False]
        expected: pd.Series = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected: pd.Series = pd.array(expected, dtype='bool')
        tm.assert_equal(left == NaT, expected)
        tm.assert_equal(NaT == left, expected)
        expected: list = [True, True, True]
        expected: pd.Series = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected: pd.Series = pd.array(expected, dtype='bool')
        tm.assert_equal(left != NaT, expected)
        tm.assert_equal(NaT != left, expected)
        expected: list = [False, False, False]
        expected: pd.Series = tm.box_expected(expected, xbox)
        if box is pd.array and dtype is object:
            expected: pd.Series = pd.array(expected, dtype='bool')
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
        """Series comparisons with scalar"""
        series: pd.Series = Series(date_range('1/1/2000', periods=10))
        result: pd.Series = series > val
        expected: pd.Series = Series([x > val for x in series])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('left,right', [('lt', 'gt'), ('le', 'ge'), ('eq', 'eq'), ('ne', 'ne')])
    def test_timestamp_compare_series(self, left: str, right: str) -> None:
        """Timestamp comparisons with series"""
        ser: pd.Series = Series(date_range('20010101', periods=10), name='dates')
        s_nat: pd.Series = ser.copy(deep=True)
        ser[0] = Timestamp('nat')
        ser[3] = Timestamp('nat')
        left_f: operator = getattr(operator, left)
        right_f: operator = getattr(operator, right)
        expected: pd.Series = left_f(ser, Timestamp('20010109'))
        result: pd.Series = right_f(Timestamp('20010109'), ser)
        tm.assert_series_equal(result, expected)
        expected: pd.Series = left_f(ser, Timestamp('nat'))
        result: pd.Series = right_f(Timestamp('nat'), ser)
        tm.assert_series_equal(result, expected)
        expected: pd.Series = left_f(s_nat, Timestamp('20010109'))
        result: pd.Series = right_f(Timestamp('20010109'), s_nat)
        tm.assert_series_equal(result, expected)
        expected: pd.Series = left_f(s_nat, NaT)
        result: pd.Series = right_f(NaT, s_nat)
        tm.assert_series_equal(result, expected)

    def test_dt64arr_timestamp_equality(self, box_with_array) -> None:
        """Timestamp equality"""
        box: pd.Series = box_with_array
        ser: pd.Series = Series([Timestamp('2000-01-29 01:59:00'), Timestamp('2000-01-30'), NaT])
        ser: pd.Series = tm.box_expected(ser, box)
        xbox: pd.Series = get_upcast_box(ser, ser, True)
        result: pd.Series = ser != ser
        expected: pd.Series = tm.box_expected([False, False, True], xbox)
        tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser != ser[0]
        else:
            result: pd.Series = ser != ser[0]
            expected: pd.Series = tm.box_expected([False, True, True], xbox)
            tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser != ser[2]
        else:
            result: pd.Series = ser != ser[2]
            expected: pd.Series = tm.box_expected([True, True, True], xbox)
            tm.assert_equal(result, expected)
        result: pd.Series = ser == ser
        expected: pd.Series = tm.box_expected([True, True, False], xbox)
        tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser == ser[0]
        else:
            result: pd.Series = ser == ser[0]
            expected: pd.Series = tm.box_expected([True, False, False], xbox)
            tm.assert_equal(result, expected)
        if box is pd.DataFrame:
            with pytest.raises(ValueError, match='not aligned'):
                ser == ser[2]
        else:
            result: pd.Series = ser == ser[2]
            expected: pd.Series = tm.box_expected([False, False, False], xbox)
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize('datetimelike', [Timestamp('20130101'), datetime(2013, 1, 1), np.datetime64('2013-01-01T00:00', 'ns')])
    @pytest.mark.parametrize('op,expected', [(operator.lt, [True, False, False, False]), (operator.le, [True, True, False, False]), (operator.eq, [False, True, False, False]), (operator.gt, [False, False, False, True])])
    def test_dt64_compare_datetime_scalar(self, datetimelike: object, op: operator, expected: list) -> None:
        """Compare datetime64 scalar with series"""
        ser: pd.Series = Series([Timestamp('20120101'), Timestamp('20130101'), np.nan, Timestamp('20130103')], name='A')
        result: pd.Series = op(ser, datetimelike)
        expected: pd.Series = Series(expected, name='A')
        tm.assert_series_equal(result, expected)

    def test_ts_series_numpy_maximum(self) -> None:
        """Timestamp series maximum"""
        ts: pd.Timestamp = Timestamp('2024-07-01')
        ts_series: pd.Series = Series(['2024-06-01', '2024-07-01', '2024-08-01'], dtype='datetime64[us]')
        expected: pd.Series = Series(['2024-07-01', '2024-07-01', '2024-08-01'], dtype='datetime64[us]')
        tm.assert_series_equal(expected, np.maximum(ts, ts_series))

class TestDatetimeIndexComparisons:
    def test_comparators(self, comparison_op: operator) -> None:
        """Compare datetimeindex with scalar"""
        index: pd.DatetimeIndex = date_range('2020-01-01', periods=10)
        element: pd.Timestamp = Timestamp(element).to_datetime64()
        arr: np.ndarray = np.array(index)
        arr_result: np.ndarray = comparison_op(arr, element)
        index_result: np.ndarray = comparison_op(index, element)
        assert isinstance(index_result, np.ndarray)
        tm.assert_numpy_array_equal(arr_result, index_result)

    @pytest.mark.parametrize('other', [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
    def test_dti_cmp_datetimelike(self, other: object, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Compare datetimeindex with datetime-like"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = date_range('2016-01-01', periods=2, tz=tz)
        if tz is not None:
            if isinstance(other, np.datetime64):
                pytest.skip(f'{type(other).__name__} is not tz aware')
            other: pd.Timestamp = localize_pydatetime(other, dti.tzinfo)
        result: np.ndarray = dti == other
        expected: np.ndarray = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = dti > other
        expected: np.ndarray = np.array([False, True])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = dti >= other
        expected: np.ndarray = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = dti < other
        expected: np.ndarray = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = dti <= other
        expected: np.ndarray = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('dtype', [None, object])
    def test_dti_cmp_nat(self, dtype: object, box_with_array) -> None:
        """Compare datetimeindex with NaT"""
        left: pd.DatetimeIndex = DatetimeIndex([Timestamp('2011-01-01'), NaT, Timestamp('2011-01-03')])
        right: pd.DatetimeIndex = DatetimeIndex([NaT, NaT, Timestamp('2011-01-03')])
        left: pd.Series = tm.box_expected(left, box_with_array)
        right: pd.Series = tm.box_expected(right, box_with_array)
        xbox: pd.Series = get_upcast_box(left, right, True)
        lhs, rhs: pd.Series = (left, right)
        if dtype is object:
            lhs, rhs = (left.astype(object), right.astype(object))
        result: pd.Series = rhs == lhs
        expected: np.ndarray = np.array([False, False, True])
        expected: pd.Series = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)
        result: pd.Series = lhs != rhs
        expected: np.ndarray = np.array([True, True, False])
        expected: pd.Series = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)
        expected: np.ndarray = np.array([False, False, False])
        expected: pd.Series = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs == NaT, expected)
        tm.assert_equal(NaT == rhs, expected)
        expected: np.ndarray = np.array([True, True, True])
        expected: pd.Series = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs != NaT, expected)
        tm.assert_equal(NaT != lhs, expected)
        expected: np.ndarray = np.array([False, False, False])
        expected: pd.Series = tm.box_expected(expected, xbox)
        tm.assert_equal(lhs < NaT, expected)
        tm.assert_equal(NaT > lhs, expected)

    def test_dti_cmp_nat_behaves_like_float_cmp_nan(self) -> None:
        """Compare datetimeindex with NaT, behaves like float comparison with NaN"""
        fidx1: pd.Index = pd.Index([1.0, np.nan, 3.0, np.nan, 5.0, 7.0])
        fidx2: pd.Index = pd.Index([2.0, 3.0, np.nan, np.nan, 6.0, 7.0])
        didx1: pd.DatetimeIndex = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
        didx2: pd.DatetimeIndex = DatetimeIndex(['2014-02-01', '2014-03-01', NaT, NaT, '2014-06-01', '2014-07-01'])
        darr: np.ndarray = np.array([np.datetime64('2014-02-01 00:00'), np.datetime64('2014-03-01 00:00'), np.datetime64('nat'), np.datetime64('nat'), np.datetime64('2014-06-01 00:00'), np.datetime64('2014-07-01 00:00')])
        cases: list = [(fidx1, fidx2), (didx1, didx2), (didx1, darr)]
        with tm.assert_produces_warning(None):
            for idx1, idx2 in cases:
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
        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, np.nan), (didx1, NaT)]:
                result: np.ndarray = idx1 < val
                expected: np.ndarray = np.array([False, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 > val
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 <= val
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 >= val
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 == val
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 != val
                expected: np.ndarray = np.array([True, True, True, True, True, True])
                tm.assert_numpy_array_equal(result, expected)
        with tm.assert_produces_warning(None):
            for idx1, val in [(fidx1, 3), (didx1, datetime(2014, 3, 1))]:
                result: np.ndarray = idx1 < val
                expected: np.ndarray = np.array([True, False, False, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 > val
                expected: np.ndarray = np.array([False, False, False, False, True, True])
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 <= val
                expected: np.ndarray = np.array([True, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 >= val
                expected: np.ndarray = np.array([False, False, True, False, True, True])
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 == val
                expected: np.ndarray = np.array([False, False, True, False, False, False])
                tm.assert_numpy_array_equal(result, expected)
                result: np.ndarray = idx1 != val
                expected: np.ndarray = np.array([True, True, False, True, True, True])
                tm.assert_numpy_array_equal(result, expected)

    def test_comparison_tzawareness_compat(self, comparison_op: operator, box_with_array) -> None:
        """Compare datetimeindex with datetimeindex, tz-awareness compatibility"""
        op: operator = comparison_op
        box: pd.Series = box_with_array
        dr: pd.DatetimeIndex = date_range('2016-01-01', periods=6)
        dz: pd.DatetimeIndex = dr.tz_localize('US/Pacific')
        dr: pd.Series = tm.box_expected(dr, box)
        dz: pd.Series = tm.box_expected(dz, box)
        if box is pd.DataFrame:
            tolist: callable = lambda x: x.astype(object).values.tolist()[0]
        else:
            tolist: callable = list
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

    def test_comparison_tzawareness_compat_scalars(self, comparison_op: operator, box_with_array) -> None:
        """Compare datetimeindex with scalar, tz-awareness compatibility"""
        op: operator = comparison_op
        dr: pd.DatetimeIndex = date_range('2016-01-01', periods=6)
        dz: pd.DatetimeIndex = dr.tz_localize('US/Pacific')
        dr: pd.Series = tm.box_expected(dr, box_with_array)
        dz: pd.Series = tm.box_expected(dz, box_with_array)
        ts: pd.Timestamp = Timestamp('2000-03-14 01:59')
        ts_tz: pd.Timestamp = Timestamp('2000-03-14 01:59', tz='Europe/Amsterdam')
        assert np.all(dr > ts)
        msg: str = 'Invalid comparison between dtype=datetime64\\[ns.*\\] and Timestamp'
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
    def test_scalar_comparison_tzawareness(self, comparison_op: operator, other: object, tz_aware_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Compare scalar with datetimeindex, tz-awareness"""
        op: operator = comparison_op
        tz: pd.Timestamp.tz_localize = tz_aware_fixture
        dti: pd.DatetimeIndex = date_range('2016-01-01', periods=2, tz=tz)
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
        xbox: pd.Series = get_upcast_box(dtarr, other, True)
        if op in [operator.eq, operator.ne]:
            exbool: bool = op is operator.ne
            expected: np.ndarray = np.array([exbool, exbool], dtype=bool)
            expected: pd.Series = tm.box_expected(expected, xbox)
            result: pd.Series = op(dtarr, other)
            tm.assert_equal(result, expected)
            result: pd.Series = op(other, dtarr)
            tm.assert_equal(result, expected)
        else:
            msg: str = f'Invalid comparison between dtype=datetime64\\[ns, .*\\] and {type(other).__name__}'
            with pytest.raises(TypeError, match=msg):
                op(dtarr, other)
            with pytest.raises(TypeError, match=msg):
                op(other, dtarr)

    def test_nat_comparison_tzawareness(self, comparison_op: operator) -> None:
        """Compare datetimeindex with NaT, tz-awareness"""
        op: operator = comparison_op
        dti: pd.DatetimeIndex = DatetimeIndex(['2014-01-01', NaT, '2014-03-01', NaT, '2014-05-01', '2014-07-01'])
        expected: np.ndarray = np.array([op == operator.ne] * len(dti))
        result: np.ndarray = op(dti, NaT)
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = op(dti.tz_localize('US/Pacific'), NaT)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_str(self, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Compare datetimeindex with str"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        rng: pd.DatetimeIndex = date_range('1/1/2000', periods=10, tz=tz)
        other: str = '1/1/2000'
        result: np.ndarray = rng == other
        expected: np.ndarray = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = rng != other
        expected: np.ndarray = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = rng < other
        expected: np.ndarray = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = rng <= other
        expected: np.ndarray = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = rng > other
        expected: np.ndarray = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = rng >= other
        expected: np.ndarray = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

    def test_dti_cmp_list(self) -> None:
        """Compare datetimeindex with list"""
        rng: pd.DatetimeIndex = date_range('1/1/2000', periods=10)
        result: pd.Series = rng == list(rng)
        expected: pd.Series = rng == rng
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('other', [pd.timedelta_range('1D', periods=10), pd.timedelta_range('1D', periods=10).to_series(), pd.timedelta_range('1D', periods=10).asi8.view('m8[ns]')], ids=lambda x: type(x).__name__)
    def test_dti_cmp_tdi_tzawareness(self, other: object) -> None:
        """Compare datetimeindex with timedelta, tz-awareness"""
        dti: pd.DatetimeIndex = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
        result: np.ndarray = dti == other
        expected: np.ndarray = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)
        result: np.ndarray = dti != other
        expected: np.ndarray = np.array([True] * 10)
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
        """Compare datetimeindex with object-dtype"""
        dti: pd.DatetimeIndex = date_range('2000-01-01', periods=10, tz='Asia/Tokyo')
        other: pd.DatetimeIndex = dti.astype('O')
        result: np.ndarray = dti == other
        expected: np.ndarray = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)
        other: pd.DatetimeIndex = dti.tz_localize(None)
        result: np.ndarray = dti != other
        tm.assert_numpy_array_equal(result, expected)
        other: np.ndarray = np.array(list(dti[:5]) + [Timedelta(days=1)] * 5)
        result: np.ndarray = dti == other
        expected: np.ndarray = np.array([True] * 5 + [False] * 5)
        tm.assert_numpy_array_equal(result, expected)
        msg: str = '>=' 'not supported between instances of ' 'Timestamp' 'and' 'Timedelta"'
        with pytest.raises(TypeError, match=msg):
            dti >= other

class TestDatetime64Arithmetic:
    @pytest.mark.arm_slow
    def test_dt64arr_add_timedeltalike_scalar(self, tz_naive_fixture: pd.Timestamp.tz_localize, two_hours: pd.Timedelta, box_with_array) -> None:
        """Add timedelta-like scalar to datetime64 array"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        rng: pd.DatetimeIndex = date_range('2000-01-01', '2000-02-01', tz=tz)
        expected: pd.DatetimeIndex = date_range('2000-01-01 02:00', '2000-02-01 02:00', tz=tz)
        rng: pd.Series = tm.box_expected(rng, box_with_array)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = rng + two_hours
        tm.assert_equal(result, expected)
        result: pd.Series = two_hours + rng
        tm.assert_equal(result, expected)
        rng += two_hours
        tm.assert_equal(rng, expected)

    def test_dt64arr_sub_timedeltalike_scalar(self, tz_naive_fixture: pd.Timestamp.tz_localize, two_hours: pd.Timedelta, box_with_array) -> None:
        """Subtract timedelta-like scalar from datetime64 array"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        rng: pd.DatetimeIndex = date_range('2000-01-01', '2000-02-01', tz=tz)
        expected: pd.DatetimeIndex = date_range('1999-12-31 22:00', '2000-01-31 22:00', tz=tz)
        rng: pd.Series = tm.box_expected(rng, box_with_array)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = rng - two_hours
        tm.assert_equal(result, expected)
        rng -= two_hours
        tm.assert_equal(rng, expected)

    def test_dt64_array_sub_dt_with_different_timezone(self, box_with_array) -> None:
        """Subtract datetime with different timezone from datetime64 array"""
        t1: pd.DatetimeIndex = date_range('20130101', periods=3).tz_localize('US/Eastern')
        t1: pd.Series = tm.box_expected(t1, box_with_array)
        t2: pd.Timestamp = Timestamp('20130101').tz_localize('CET')
        tnaive: pd.Timestamp = Timestamp(20130101)
        result: pd.TimedeltaIndex = t1 - t2
        expected: pd.TimedeltaIndex = pd.timedelta_range('0 days 06:00:00', periods=3)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result: pd.TimedeltaIndex = t2 - t1
        expected: pd.TimedeltaIndex = pd.timedelta_range('-1 days +18:00:00', periods=3)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64_array_sub_dt64_array_with_different_timezone(self, box_with_array) -> None:
        """Subtract datetime64 array with different timezone from datetime64 array"""
        t1: pd.DatetimeIndex = date_range('20130101', periods=3).tz_localize('US/Eastern')
        t1: pd.Series = tm.box_expected(t1, box_with_array)
        t2: pd.DatetimeIndex = date_range('20130101', periods=3).tz_localize('CET')
        t2: pd.Series = tm.box_expected(t2, box_with_array)
        tnaive: pd.DatetimeIndex = date_range('20130101', periods=3)
        result: pd.TimedeltaIndex = t1 - t2
        expected: pd.TimedeltaIndex = pd.timedelta_range('0 days 06:00:00', periods=3)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        result: pd.TimedeltaIndex = t2 - t1
        expected: pd.TimedeltaIndex = pd.timedelta_range('-1 days +18:00:00', periods=3)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    def test_dt64arr_add_sub_td64_nat(self, box_with_array, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Add/subtract timedelta64 NaT from datetime64 array"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = date_range('1994-04-01', periods=9, tz=tz, freq='QS')
        other: np.timedelta64 = np.timedelta64('NaT')
        expected: pd.DatetimeIndex = DatetimeIndex(['NaT'] * 9, tz=tz).as_unit('ns')
        obj: pd.Series = tm.box_expected(dti, box_with_array)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = obj + other
        tm.assert_equal(result, expected)
        result: pd.Series = other + obj
        tm.assert_equal(result, expected)
        result: pd.Series = obj - other
        tm.assert_equal(result, expected)
        msg: str = 'cannot subtract'
        with pytest.raises(TypeError, match=msg):
            other - obj

    def test_dt64arr_add_sub_td64ndarray(self, tz_naive_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Add/subtract timedelta64 ndarray from datetime64 array"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = date_range('2016-01-01', periods=3, tz=tz)
        tdi: pd.TimedeltaIndex = pd.timedelta_range('0 days', periods=3)
        tdarr: np.ndarray = tdi.values
        expected: pd.DatetimeIndex = date_range('2015-12-31', '2016-01-02', periods=3, tz=tz)
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = dtarr + tdarr
        tm.assert_equal(result, expected)
        result: pd.Series = tdarr + dtarr
        tm.assert_equal(result, expected)
        expected: pd.DatetimeIndex = date_range('2016-01-02', '2016-01-04', periods=3, tz=tz)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = dtarr - tdarr
        tm.assert_equal(result, expected)
        msg: str = 'cannot subtract|(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            tdarr - dtarr

    @pytest.mark.parametrize('ts', [Timestamp('2013-01-01'), Timestamp('2013-01-01').to_pydatetime(), Timestamp('2013-01-01').to_datetime64(), np.datetime64('2013-01-01', 'D')])
    def test_dt64arr_sub_dtscalar(self, box_with_array, ts: object) -> None:
        """Subtract datetime scalar from datetime64 array"""
        idx: pd.DatetimeIndex = date_range('2013-01-01', periods=3)._with_freq(None)
        idx: pd.Series = tm.box_expected(idx, box_with_array)
        expected: pd.TimedeltaIndex = pd.timedelta_range('0 Days', periods=3)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = idx - ts
        tm.assert_equal(result, expected)
        result: pd.Series = ts - idx
        tm.assert_equal(result, -expected)
        tm.assert_equal(result, -expected)

    def test_dt64arr_sub_timestamp_tzaware(self, box_with_array) -> None:
        """Subtract tz-aware timestamp from datetime64 array"""
        ser: pd.Series = date_range('2014-03-17', periods=2, freq='D', tz='US/Eastern')
        ser: pd.Series = ser._with_freq(None)
        ts: pd.Timestamp = ser[0]
        ser: pd.Series = tm.box_expected(ser, box_with_array)
        delta_series: pd.Series = Series([np.timedelta64(0, 'D'), np.timedelta64(1, 'D')])
        expected: pd.Series = tm.box_expected(delta_series, box_with_array)
        tm.assert_equal(ser - ts, expected)
        tm.assert_equal(ts - ser, -expected)

    def test_dt64arr_sub_NaT(self, box_with_array, unit: str) -> None:
        """Subtract NaT from datetime64 array"""
        dti: pd.DatetimeIndex = DatetimeIndex([NaT, Timestamp('19900315')]).as_unit(unit)
        ser: pd.Series = tm.box_expected(dti, box_with_array)
        result: pd.Series = ser - NaT
        expected: pd.Series = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)
        dti_tz: pd.DatetimeIndex = dti.tz_localize('Asia/Tokyo')
        ser_tz: pd.Series = tm.box_expected(dti_tz, box_with_array)
        result: pd.Series = ser_tz - NaT
        expected: pd.Series = Series([NaT, NaT], dtype=f'timedelta64[{unit}]')
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

    def test_dt64arr_sub_dt64object_array(self, performance_warning: str, box_with_array, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Subtract datetime64 object array from datetime64 array"""
        dti: pd.DatetimeIndex = date_range('2016-01-01', periods=3, tz=tz_naive_fixture)
        expected: pd.DatetimeIndex = dti - dti
        obj: pd.Series = tm.box_expected(dti, box_with_array)
        expected: pd.Series = tm.box_expected(expected, box_with_array).astype(object)
        with tm.assert_produces_warning(performance_warning):
            result: pd.Series = obj - obj.astype(object)
        tm.assert_equal(result, expected)

    def test_dt64arr_naive_sub_dt64ndarray(self, box_with_array) -> None:
        """Subtract datetime64 naive ndarray from datetime64 array"""
        dti: pd.DatetimeIndex = date_range('2016-01-01', periods=3, tz=None)
        dt64vals: np.ndarray = dti.values
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
        expected: pd.DatetimeIndex = dtarr - dtarr
        result: pd.Series = dtarr - dt64vals
        tm.assert_equal(result, expected)
        result: pd.Series = dt64vals - dtarr
        tm.assert_equal(result, expected)

    def test_dt64arr_aware_sub_dt64ndarray_raises(self, tz_aware_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Subtract datetime64 aware ndarray from datetime64 array raises"""
        tz: pd.Timestamp.tz_localize = tz_aware_fixture
        dti: pd.DatetimeIndex = date_range('2016-01-01', periods=3, tz=tz)
        dt64vals: np.ndarray = dti.values
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime'
        with pytest.raises(TypeError, match=msg):
            dtarr - dt64vals
        with pytest.raises(TypeError, match=msg):
            dt64vals - dtarr

    def test_dt64arr_add_dtlike_raises(self, tz_naive_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Add datetime-like raises"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = date_range('2016-01-01', periods=3, tz=tz)
        if tz is None:
            dti2: pd.DatetimeIndex = dti.tz_localize('US/Eastern')
        else:
            dti2: pd.DatetimeIndex = dti.tz_localize(None)
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
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
    def test_dt64arr_addsub_intlike(self, dtype: object, index_or_series_or_array: pd.Series, freq: str, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Add/subtract int-like to datetime64 array"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        if freq is None:
            dti: pd.DatetimeIndex = DatetimeIndex(['NaT', '2017-04-05 06:07:08'], tz=tz)
        else:
            dti: pd.DatetimeIndex = date_range('2016-01-01', periods=2, freq=freq, tz=tz)
        obj: pd.Series = index_or_series_or_array(dti)
        other: np.ndarray = np.array([4, -1])
        if dtype is not None:
            other: np.ndarray = other.astype(dtype)
        msg: str = '|'.join(['Addition/subtraction of integers', 'cannot subtract DatetimeArray from', 'can only perform ops with numeric values', 'unsupported operand type.*Categorical', "unsupported operand type\\(s\\) for -: 'int' and 'Timestamp'"])
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
    def test_dt64arr_add_sub_invalid(self, dti_freq: str, other: object, box_with_array) -> None:
        """Add/subtract invalid types to datetime64 array"""
        dti: pd.DatetimeIndex = DatetimeIndex(['2011-01-01', '2011-01-02'], freq=dti_freq)
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
        msg: str = '|'.join(['unsupported operand type', 'cannot (add|subtract)', 'cannot use operands with types', "ufunc '?(add|subtract)'? cannot use operands with types", 'Concatenation operation is not implemented for NumPy arrays'])
        assert_invalid_addsub_type(dtarr, other, msg)

    @pytest.mark.parametrize('pi_freq', ['D', 'W', 'Q', 'h'])
    @pytest.mark.parametrize('dti_freq', [None, 'D'])
    def test_dt64arr_add_sub_parr(self, dti_freq: str, pi_freq: str, box_with_array, box_with_array2) -> None:
        """Add/subtract PeriodArray to datetime64 array"""
        dti: pd.DatetimeIndex = DatetimeIndex(['2011-01-01', '2011-01-02'], freq=dti_freq)
        pi: pd.PeriodIndex = dti.to_period(pi_freq)
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
        parr: pd.Series = tm.box_expected(pi, box_with_array2)
        msg: str = '|'.join(['cannot (add|subtract)', 'unsupported operand', 'descriptor.*requires', 'ufunc.*cannot use operands'])
        assert_invalid_addsub_type(dtarr, parr, msg)

    @pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
    def test_dt64arr_addsub_time_objects_raises(self, box_with_array, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Add/subtract time objects raises"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        obj1: pd.Series = date_range('2012-01-01', periods=3, tz=tz)
        obj2: pd.Series = [time(i, i, i) for i in range(3)]
        obj1: pd.Series = tm.box_expected(obj1, box_with_array)
        obj2: pd.Series = tm.box_expected(obj2, box_with_array)
        msg: str = '|'.join(['unsupported operand', 'cannot subtract DatetimeArray from ndarray'])
        assert_invalid_addsub_type(obj1, obj2, msg=msg)

    @pytest.mark.parametrize('dt64_series', [Series([Timestamp('19900315'), Timestamp('19900315')]), Series([NaT, Timestamp('19900315')]), Series([NaT, NaT], dtype='datetime64[ns]')])
    @pytest.mark.parametrize('one', [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(self, one: object, dt64_series: pd.Series, box_with_array) -> None:
        """Multiply/divide datetime64 series with numeric invalid"""
        obj: pd.Series = tm.box_expected(dt64_series, box_with_array)
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
    def test_dt64arr_series_add_tick_DateOffset(self, box_with_array, unit: str) -> None:
        """Add tick DateOffset to datetime64 series"""
        ser: pd.Series = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')]).dt.as_unit(unit)
        expected: pd.Series = Series([Timestamp('20130101 9:01:05'), Timestamp('20130101 9:02:05')]).dt.as_unit(unit)
        ser: pd.Series = tm.box_expected(ser, box_with_array)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = ser + pd.offsets.Second(5)
        tm.assert_equal(result, expected)
        result2: pd.Series = pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)

    def test_dt64arr_series_sub_tick_DateOffset(self, box_with_array) -> None:
        """Subtract tick DateOffset from datetime64 series"""
        ser: pd.Series = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        expected: pd.Series = Series([Timestamp('20130101 9:00:55'), Timestamp('20130101 9:01:55')])
        ser: pd.Series = tm.box_expected(ser, box_with_array)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = ser - pd.offsets.Second(5)
        tm.assert_equal(result, expected)
        result2: pd.Series = -pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)
        msg: str = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            pd.offsets.Second(5) - ser

    @pytest.mark.parametrize('cls_name', ['Day', 'Hour', 'Minute', 'Second', 'Milli', 'Micro', 'Nano'])
    def test_dt64arr_add_sub_tick_DateOffset_smoke(self, cls_name: str, box_with_array) -> None:
        """Add/subtract tick DateOffset to datetime64 array"""
        ser: pd.Series = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        ser: pd.Series = tm.box_expected(ser, box_with_array)
        offset_cls: pd.offsets.BaseOffset = getattr(pd.offsets, cls_name)
        ser + offset_cls(5)
        offset_cls(5) + ser
        ser - offset_cls(5)

    def test_dti_add_tick_tzaware(self, tz_aware_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Add tick DateOffset to datetimeindex, tz-aware"""
        tz: pd.Timestamp.tz_localize = tz_aware_fixture
        if tz == 'US/Pacific':
            dates: pd.DatetimeIndex = date_range('2012-11-01', periods=3, tz=tz)
            offset: pd.DatetimeIndex = dates + pd.offsets.Hour(5)
            assert dates[0] + pd.offsets.Hour(5) == offset[0]
        dates: pd.DatetimeIndex = date_range('2010-11-01 00:00', periods=3, tz=tz, freq='h')
        expected: pd.DatetimeIndex = DatetimeIndex(['2010-11-01 05:00', '2010-11-01 06:00', '2010-11-01 07:00'], freq='h', tz=tz).as_unit('ns')
        dates: pd.Series = tm.box_expected(dates, box_with_array)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        for scalar in [pd.offsets.Hour(5), np.timedelta64(5, 'h'), timedelta(hours=5)]:
            offset: pd.DatetimeIndex = dates + scalar
            tm.assert_equal(offset, expected)
            offset: pd.DatetimeIndex = scalar + dates
            tm.assert_equal(offset, expected)
            roundtrip: pd.DatetimeIndex = offset - scalar
            tm.assert_equal(roundtrip, dates)
            msg: str = '|'.join(['bad operand type for unary -', 'cannot subtract DatetimeArray'])
            with pytest.raises(TypeError, match=msg):
                scalar - dates

    def test_dt64arr_add_sub_relativedelta_offsets(self, box_with_array, unit: str) -> None:
        """Add/subtract relativedelta DateOffset to datetime64 array"""
        vec: pd.Series = DatetimeIndex([Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'), Timestamp('2000-01-01'), Timestamp('2000-03-31'), Timestamp('2000-02-29'), Timestamp('2000-12-31'), Timestamp('2000-05-15'), Timestamp('2001-06-15')]).as_unit(unit)
        vec: pd.Series = tm.box_expected(vec, box_with_array)
        vec_items: pd.Timestamp = vec.iloc[0] if box_with_array is pd.DataFrame else vec
        relative_kwargs: list = [('years', 2), ('months', 5), ('days', 3), ('hours', 5), ('minutes', 10), ('seconds', 2), ('microseconds', 5)]
        for i, (offset_unit: str, value: int) in enumerate(relative_kwargs):
            off: pd.offsets.DateOffset = DateOffset(**{offset_unit: value})
            exp_unit: str = unit
            if offset_unit == 'microseconds' and unit != 'ns':
                exp_unit: str = 'us'
            expected: pd.Series = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected: pd.Series = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)
            expected: pd.Series = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected: pd.Series = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)
            off: pd.offsets.DateOffset = DateOffset(**dict(relative_kwargs[:i + 1]))
            expected: pd.Series = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected: pd.Series = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)
            expected: pd.Series = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected: pd.Series = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)
            msg: str = '(bad|unsupported) operand type for unary'
            with pytest.raises(TypeError, match=msg):
                off - vec

    @pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
    @pytest.mark.parametrize('cls_and_kwargs', ['YearBegin', ('YearBegin', {'month': 5}), 'YearEnd', ('YearEnd', {'month': 5}), 'MonthBegin', 'MonthEnd', 'SemiMonthEnd', 'SemiMonthBegin', 'Week', ('Week', {'weekday': 3}), ('Week', {'weekday': 6}), 'BusinessDay', 'BDay', 'QuarterEnd', 'QuarterBegin', 'CustomBusinessDay', 'CDay', 'CBMonthEnd', 'CBMonthBegin', 'BMonthBegin', 'BMonthEnd', 'BusinessHour', 'BYearBegin', 'BYearEnd', 'BQuarterBegin', ('LastWeekOfMonth', {'weekday': 2}), ('FY5253Quarter', {'qtr_with_extra_week': 1, 'startingMonth': 1, 'weekday': 2, 'variation': 'nearest'}), ('FY5253', {'weekday': 0, 'startingMonth': 2, 'variation': 'nearest'}), ('WeekOfMonth', {'weekday': 2, 'week': 2}), 'Easter', ('DateOffset', {'day': 4}), ('DateOffset', {'month': 5})])
    @pytest.mark.parametrize('normalize', [True, False])
    @pytest.mark.parametrize('n', [0, 5])
    @pytest.mark.parametrize('tz', [None, 'US/Central'])
    def test_dt64arr_add_sub_DateOffsets(self, box_with_array, n: int, normalize: bool, cls_and_kwargs: object, unit: str, tz: pd.Timestamp.tz_localize) -> None:
        """Add/subtract DateOffset to datetime64 array"""
        if isinstance(cls_and_kwargs, tuple):
            cls_name: str = cls_and_kwargs[0]
            kwargs: dict = cls_and_kwargs[1]
        else:
            cls_name: str = cls_and_kwargs
            kwargs: dict = {}
        if n == 0 and cls_name in ['WeekOfMonth', 'LastWeekOfMonth', 'FY5253Quarter', 'FY5253']:
            return
        vec: pd.Series = DatetimeIndex([Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'), Timestamp('2000-01-01'), Timestamp('2000-03-31'), Timestamp('2000-02-29'), Timestamp('2000-12-31'), Timestamp('2000-05-15'), Timestamp('2001-06-15')]).as_unit(unit).tz_localize(tz)
        vec: pd.Series = tm.box_expected(vec, box_with_array)
        vec_items: pd.Timestamp = vec.iloc[0] if box_with_array is pd.DataFrame else vec
        offset_cls: pd.offsets.BaseOffset = getattr(pd.offsets, cls_name)
        offset: pd.offsets.BaseOffset = offset_cls(n, normalize=normalize, **kwargs)
        expected: pd.Series = DatetimeIndex([x + offset for x in vec_items]).as_unit(unit)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec + offset)
        tm.assert_equal(expected, offset + vec)
        expected: pd.Series = DatetimeIndex([x - offset for x in vec_items]).as_unit(unit)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, vec - offset)
        expected: pd.Series = DatetimeIndex([offset + x for x in vec_items]).as_unit(unit)
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        tm.assert_equal(expected, offset + vec)
        msg: str = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            offset - vec

    @pytest.mark.parametrize('other', [[pd.offsets.MonthEnd(), pd.offsets.Day(n=2)], [pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()], [pd.offsets.DateOffset(years=1), pd.offsets.DateOffset(years=1)]])
    @pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
    def test_dt64arr_add_sub_offset_array(self, performance_warning: str, tz_naive_fixture: pd.Timestamp.tz_localize, box_with_array, op: operator, other: list) -> None:
        """Add/subtract DateOffset array to datetime64 array"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = date_range('2017-01-01', periods=2, tz=tz)
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
        other: np.ndarray = np.array(other)
        expected: pd.Series = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))])
        expected: pd.Series = tm.box_expected(expected, box_with_array).astype(object)
        with tm.assert_produces_warning(performance_warning):
            res: pd.Series = op(dtarr, other)
        tm.assert_equal(res, expected)
        other: pd.Series = tm.box_expected(other, box_with_array)
        if box_with_array is pd.array and op is roperator.radd:
            expected: pd.Series = pd.array(expected, dtype=object)
        with tm.assert_produces_warning(performance_warning):
            res: pd.Series = op(dtarr, other)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize('op, offset, exp, exp_freq', [('__add__', DateOffset(months=3, days=10), [Timestamp('2014-04-11'), Timestamp('2015-04-11'), Timestamp('2016-04-11'), Timestamp('2017-04-11')], None), ('__add__', DateOffset(months=3), [Timestamp('2014-04-01'), Timestamp('2015-04-01'), Timestamp('2016-04-01'), Timestamp('2017-04-01')], 'YS-APR'), ('__sub__', DateOffset(months=3, days=10), [Timestamp('2013-09-21'), Timestamp('2014-09-21'), Timestamp('2015-09-21'), Timestamp('2016-09-21')], None), ('__sub__', DateOffset(months=3), [Timestamp('2013-10-01'), Timestamp('2014-10-01'), Timestamp('2015-10-01'), Timestamp('2016-10-01')], 'YS-OCT')])
    def test_dti_add_sub_nonzero_mth_offset(self, op: str, offset: pd.offsets.BaseOffset, exp: list, exp_freq: str, tz_aware_fixture: pd.Timestamp.tz_localize, box_with_array) -> None:
        """Add/subtract nonzero month DateOffset to datetimeindex"""
        tz: pd.Timestamp.tz_localize = tz_aware_fixture
        date: pd.DatetimeIndex = date_range(start='01 Jan 2014', end='01 Jan 2017', freq='YS', tz=tz)
        date: pd.Series = tm.box_expected(date, box_with_array, False)
        mth: pd.Series = getattr(date, op)
        result: pd.Series = mth(offset)
        expected: pd.Series = DatetimeIndex(exp, tz=tz).as_unit('ns')
        expected: pd.Series = tm.box_expected(expected, box_with_array, False)
        tm.assert_equal(result, expected)

    def test_dt64arr_series_add_DateOffset_with_milli(self) -> None:
        """Add DateOffset with milli to datetime64 series"""
        dti: pd.DatetimeIndex = DatetimeIndex(['2000-01-01 00:00:00.012345678', '2000-01-31 00:00:00.012345678', '2000-02-29 00:00:00.012345678'], dtype='datetime64[ns]')
        result: pd.DatetimeIndex = dti + DateOffset(milliseconds=4)
        expected: pd.DatetimeIndex = DatetimeIndex(['2000-01-01 00:00:00.016345678', '2000-01-31 00:00:00.016345678', '2000-02-29 00:00:00.016345678'], dtype='datetime64[ns]')
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = dti + DateOffset(days=1, milliseconds=4)
        expected: pd.DatetimeIndex = DatetimeIndex(['2000-01-02 00:00:00.016345678', '2000-02-01 00:00:00.016345678', '2000-03-01 00:00:00.016345678'], dtype='datetime64[ns]')
        tm.assert_index_equal(result, expected)

class TestDatetime64OverflowHandling:
    def test_dt64_overflow_masking(self, box_with_array) -> None:
        """Datetime64 overflow masking"""
        left: pd.Series = Series([Timestamp('1969-12-31')], dtype='M8[ns]')
        right: pd.Series = Series([NaT])
        left: pd.Series = tm.box_expected(left, box_with_array)
        right: pd.Series = tm.box_expected(right, box_with_array)
        expected: pd.Series = TimedeltaIndex([NaT], dtype='m8[ns]')
        expected: pd.Series = tm.box_expected(expected, box_with_array)
        result: pd.Series = left - right
        tm.assert_equal(result, expected)

    def test_dt64_series_arith_overflow(self) -> None:
        """Datetime64 series arithmetic overflow"""
        dt: pd.Timestamp = Timestamp('1700-01-31')
        td: pd.Timedelta = Timedelta('20000 Days')
        dti: pd.DatetimeIndex = date_range('1949-09-30', freq='100YE', periods=4)
        ser: pd.Series = Series(dti)
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
        expected: pd.Series = Series(['2004-10-03', '2104-10-04', '2204-10-04', 'NaT'], dtype='datetime64[ns]')
        res: pd.Series = ser + td
        tm.assert_series_equal(res, expected)
        res: pd.Series = td + ser
        tm.assert_series_equal(res, expected)
        ser.iloc[1:] = NaT
        expected: pd.Series = Series(['91279 Days', 'NaT', 'NaT', 'NaT'], dtype='timedelta64[ns]')
        res: pd.Series = ser - dt
        tm.assert_series_equal(res, expected)
        res: pd.Series = dt - ser
        tm.assert_series_equal(res, -expected)

    def test_datetimeindex_sub_timestamp_overflow(self) -> None:
        """Datetimeindex subtract timestamp overflow"""
        dtimax: pd.DatetimeIndex = pd.to_datetime(['2021-12-28 17:19', Timestamp.max]).as_unit('ns')
        dtimin: pd.DatetimeIndex = pd.to_datetime(['2021-12-28 17:19', Timestamp.min]).as_unit('ns')
        tsneg: pd.Timestamp = Timestamp('1950-01-01').as_unit('ns')
        ts_neg_variants: list = [tsneg, tsneg.to_pydatetime(), tsneg.to_datetime64().astype('datetime64[ns]'), tsneg.to_datetime64().astype('datetime64[D]')]
        tspos: pd.Timestamp = Timestamp('1980-01-01').as_unit('ns')
        ts_pos_variants: list = [tspos, tspos.to_pydatetime(), tspos.to_datetime64().astype('datetime64[ns]'), tspos.to_datetime64().astype('datetime64[D]')]
        msg: str = 'Overflow in int64 addition'
        for variant in ts_neg_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimax - variant
        expected: int = Timestamp.max._value - tspos._value
        for variant in ts_pos_variants:
            res: pd.DatetimeIndex = dtimax - variant
            assert res[1]._value == expected
        expected: int = Timestamp.min._value - tsneg._value
        for variant in ts_neg_variants:
            res: pd.DatetimeIndex = dtimin - variant
            assert res[1]._value == expected
        for variant in ts_pos_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimin - variant

    def test_datetimeindex_sub_datetimeindex_overflow(self) -> None:
        """Datetimeindex subtract datetimeindex overflow"""
        dtimax: pd.DatetimeIndex = pd.to_datetime(['2021-12-28 17:19', Timestamp.max]).as_unit('ns')
        dtimin: pd.DatetimeIndex = pd.to_datetime(['2021-12-28 17:19', Timestamp.min]).as_unit('ns')
        ts_neg: pd.DatetimeIndex = pd.to_datetime(['1950-01-01', '1950-01-01']).as_unit('ns')
        ts_pos: pd.DatetimeIndex = pd.to_datetime(['1980-01-01', '1980-01-01']).as_unit('ns')
        expected: int = Timestamp.max._value - ts_pos[1]._value
        result: pd.DatetimeIndex = dtimax - ts_pos
        assert result[1]._value == expected
        expected: int = Timestamp.min._value - ts_neg[1]._value
        result: pd.DatetimeIndex = dtimin - ts_neg
        assert result[1]._value == expected
        msg: str = 'Overflow in int64 addition'
        with pytest.raises(OverflowError, match=msg):
            dtimax - ts_neg
        with pytest.raises(OverflowError, match=msg):
            dtimin - ts_pos
        tmin: pd.DatetimeIndex = pd.to_datetime([Timestamp.min])
        t1: pd.DatetimeIndex = tmin + Timedelta.max + Timedelta('1us')
        with pytest.raises(OverflowError, match=msg):
            t1 - tmin
        tmax: pd.DatetimeIndex = pd.to_datetime([Timestamp.max])
        t2: pd.DatetimeIndex = tmax + Timedelta.min - Timedelta('1us')
        with pytest.raises(OverflowError, match=msg):
            tmax - t2

class TestTimestampSeriesArithmetic:
    def test_empty_series_add_sub(self, box_with_array) -> None:
        """Empty series add/sub"""
        a: pd.Series = Series(dtype='M8[ns]')
        b: pd.Series = Series(dtype='m8[ns]')
        a: pd.Series = box_with_array(a)
        b: pd.Series = box_with_array(b)
        tm.assert_equal(a, a + b)
        tm.assert_equal(a, a - b)
        tm.assert_equal(a, b + a)
        msg: str = 'cannot subtract'
        with pytest.raises(TypeError, match=msg):
            b - a

    def test_operators_datetimelike(self) -> None:
        """Operators with datetime-like"""
        td1: pd.Series = Series([timedelta(minutes=5, seconds=3)] * 3)
        td1.iloc[2] = np.nan
        dt1: pd.Series = Series([Timestamp('20111230'), Timestamp('20120101'), Timestamp('20120103')])
        dt1.iloc[2] = np.nan
        dt2: pd.Series = Series([Timestamp('20111231'), Timestamp('20120102'), Timestamp('20120104')])
        dt1 - dt2
        dt2 - dt1
        dt1 + td1
        td1 + dt1
        dt1 - td1
        td1 + dt1
        dt1 + td1

    def test_dt64ser_sub_datetime_dtype(self, unit: str) -> None:
        """Subtract datetime scalar from datetime64 series"""
        ts: pd.Timestamp = Timestamp(datetime(1993, 1, 7, 13, 30, 0))
        dt: datetime = datetime(1993, 6, 22, 13, 30)
        ser: pd.Series = Series([ts], dtype=f'M8[{unit}]')
        result: pd.Series = ser - dt
        exp_unit: str = tm.get_finest_unit(unit, 'us')
        assert result.dtype == f'timedelta64[{exp_unit}]'

    @pytest.mark.parametrize('left, right, op_fail', [[[Timestamp('20111230'), NaT, Timestamp('20120101')], [Timestamp('20111231'), Timestamp('20120102'), Timestamp('20120104')], ['__sub__', '__rsub__']], [[Timestamp('20111230'), Timestamp('20120101'), NaT], [timedelta(minutes=5, seconds=3), timedelta(minutes=5, seconds=3), NaT], ['__add__', '__radd__', '__sub__']], [[Timestamp('20111230', tz='US/Eastern'), Timestamp('20111230', tz='US/Eastern'), NaT], [timedelta(minutes=5, seconds=3), NaT, timedelta(minutes=5, seconds=3)], ['__add__', '__radd__', '__sub__']]])
    def test_operators_datetimelike_invalid(self, left: list, right: list, op_fail: list, all_arithmetic_operators: str) -> None:
        """Operators with datetime-like invalid"""
        op_str: str = all_arithmetic_operators
        arg1: pd.Series = Series(left)
        arg2: pd.Series = Series(right)
        op: operator = getattr(arg1, op_str, None)
        if op_str not in op_fail:
            with pytest.raises(TypeError, match='operate|[cC]annot|unsupported operand'):
                op(arg2)
        else:
            op(arg2)

    def test_sub_single_tz(self, unit: str) -> None:
        """Subtract single tz-aware timestamp"""
        s1: pd.Series = Series([Timestamp('2016-02-10', tz='America/Sao_Paulo')]).dt.as_unit(unit)
        s2: pd.Series = Series([Timestamp('2016-02-08', tz='America/Sao_Paulo')]).dt.as_unit(unit)
        result: pd.Series = s1 - s2
        expected: pd.Series = Series([Timedelta('2days')]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)
        result: pd.Series = s2 - s1
        expected: pd.Series = Series([Timedelta('-2days')]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)

    def test_dt64tz_series_sub_dtitz(self) -> None:
        """Subtract tz-aware datetimeindex from tz-aware datetime series"""
        dti: pd.DatetimeIndex = date_range('1999-09-30', periods=10, tz='US/Pacific')
        ser: pd.Series = Series(dti)
        expected: pd.Series = Series(TimedeltaIndex(['0days'] * 10))
        res: pd.Series = dti - ser
        tm.assert_series_equal(res, expected)
        res: pd.Series = ser - dti
        tm.assert_series_equal(res, expected)

    def test_sub_datetime_compat(self, unit: str) -> None:
        """Subtract datetime scalar from datetime series"""
        ser: pd.Series = Series([datetime(2016, 8, 23, 12, tzinfo=timezone.utc), NaT]).dt.as_unit(unit)
        dt: datetime = datetime(2016, 8, 22, 12, tzinfo=timezone.utc)
        exp_unit: str = tm.get_finest_unit(unit, 'us')
        exp: pd.Series = Series([Timedelta('1 days'), NaT]).dt.as_unit(exp_unit)
        result: pd.Series = ser - dt
        tm.assert_series_equal(result, exp)
        result2: pd.Series = ser - Timestamp(dt)
        tm.assert_series_equal(result2, exp)

    def test_dt64_series_add_mixed_tick_DateOffset(self) -> None:
        """Add mixed tick DateOffset to datetime64 series"""
        s: pd.Series = Series([Timestamp('20130101 9:01'), Timestamp('20130101 9:02')])
        result: pd.Series = s + pd.offsets.Milli(5)
        result2: pd.Series = pd.offsets.Milli(5) + s
        expected: pd.Series = Series([Timestamp('20130101 9:01:00.005'), Timestamp('20130101 9:02:00.005')])
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)
        result: pd.Series = s + pd.offsets.Minute(5) + pd.offsets.Milli(5)
        expected: pd.Series = Series([Timestamp('20130101 9:06:00.005'), Timestamp('20130101 9:07:00.005')])
        tm.assert_series_equal(result, expected)

    def test_datetime64_ops_nat(self, unit: str) -> None:
        """Datetime64 operations with NaT"""
        datetime_series: pd.Series = Series([NaT, Timestamp('19900315')]).dt.as_unit(unit)
        nat_series_dtype_timestamp: pd.Series = Series([NaT, NaT], dtype=f'datetime64[{unit}]')
        single_nat_dtype_datetime: pd.Series = Series([NaT], dtype=f'datetime64[{unit}]')
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
        """Operators with datetime-like and timezones"""
        tz: str = 'US/Eastern'
        dt1: pd.Series = Series(date_range('2000-01-01 09:00:00', periods=5, tz=tz), name='foo')
        dt2: pd.Series = dt1.copy()
        dt2.iloc[2] = np.nan
        td1: pd.Series = Series(pd.timedelta_range('1 days 1 min', periods=5, freq='h'))
        td2: pd.Series = td1.copy()
        td2.iloc[1] = np.nan
        assert td2._values.freq is None
        result: pd.Series = dt1 + td1[0]
        exp: pd.Series = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result: pd.Series = dt2 + td2[0]
        exp: pd.Series = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result: pd.Series = td1[0] + dt1
        exp: pd.Series = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result: pd.Series = td2[0] + dt2
        exp: pd.Series = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result: pd.Series = dt1 - td1[0]
        exp: pd.Series = (dt1.dt.tz_localize(None) - td1[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg: str = '(bad|unsupported) operand type for unary'
        with pytest.raises(TypeError, match=msg):
            td1[0] - dt1
        result: pd.Series = dt2 - td2[0]
        exp: pd.Series = (dt2.dt.tz_localize(None) - td2[0]).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        with pytest.raises(TypeError, match=msg):
            td2[0] - dt2
        result: pd.Series = dt1 + td1
        exp: pd.Series = (dt1.dt.tz_localize(None) + td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result: pd.Series = dt2 + td2
        exp: pd.Series = (dt2.dt.tz_localize(None) + td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result: pd.Series = dt1 - td1
        exp: pd.Series = (dt1.dt.tz_localize(None) - td1).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        result: pd.Series = dt2 - td2
        exp: pd.Series = (dt2.dt.tz_localize(None) - td2).dt.tz_localize(tz)
        tm.assert_series_equal(result, exp)
        msg: str = 'cannot (add|subtract)'
        with pytest.raises(TypeError, match=msg):
            td1 - dt1
        with pytest.raises(TypeError, match=msg):
            td2 - dt2

class TestDatetimeIndexArithmetic:
    def test_dti_add_tdi(self, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Add timedelta to datetimeindex"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        tdi: pd.TimedeltaIndex = pd.timedelta_range('0 days', periods=10)
        expected: pd.DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz)
        expected: pd.DatetimeIndex = expected._with_freq(None)
        result: pd.DatetimeIndex = dti + tdi
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = tdi + dti
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = dti + tdi.values
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = tdi.values + dti
        tm.assert_index_equal(result, expected)

    def test_dti_iadd_tdi(self, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Inplace add timedelta to datetimeindex"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        tdi: pd.TimedeltaIndex = pd.timedelta_range('0 days', periods=10)
        expected: pd.DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz)
        expected: pd.DatetimeIndex = expected._with_freq(None)
        result: pd.DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        result += tdi
        tm.assert_index_equal(result, expected)
        result: pd.TimedeltaIndex = pd.timedelta_range('0 days', periods=10)
        result += dti
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        result += tdi.values
        tm.assert_index_equal(result, expected)
        result: pd.TimedeltaIndex = pd.timedelta_range('0 days', periods=10)
        result += dti
        tm.assert_index_equal(result, expected)

    def test_dti_sub_tdi(self, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Subtract timedelta from datetimeindex"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10)
        tdi: pd.TimedeltaIndex = pd.timedelta_range('0 days', periods=10)
        expected: pd.DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz, freq='-1D')
        expected: pd.DatetimeIndex = expected._with_freq(None)
        result: pd.DatetimeIndex = dti - tdi
        tm.assert_index_equal(result, expected)
        msg: str = 'cannot subtract .*TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi - dti
        result: pd.DatetimeIndex = dti - tdi.values
        tm.assert_index_equal(result, expected)
        msg: str = 'cannot subtract a datelike from a TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi.values - dti

    def test_dti_isub_tdi(self, tz_naive_fixture: pd.Timestamp.tz_localize, unit: str) -> None:
        """Inplace subtract timedelta from datetimeindex"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10).as_unit(unit)
        tdi: pd.TimedeltaIndex = pd.timedelta_range('0 days', periods=10, unit=unit)
        expected: pd.DatetimeIndex = date_range('2017-01-01', periods=10, tz=tz, freq='-1D', unit=unit)
        expected: pd.DatetimeIndex = expected._with_freq(None)
        result: pd.DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10).as_unit(unit)
        result -= tdi
        tm.assert_index_equal(result, expected)
        dta: np.ndarray = dti._data.copy()
        dta -= tdi
        tm.assert_datetime_array_equal(dta, expected._data)
        out: np.ndarray = dti._data.copy()
        np.subtract(out, tdi, out=out)
        tm.assert_datetime_array_equal(out, expected._data)
        msg: str = 'cannot subtract a datelike from a TimedeltaArray'
        with pytest.raises(TypeError, match=msg):
            tdi -= dti
        result: pd.DatetimeIndex = DatetimeIndex([Timestamp('2017-01-01', tz=tz)] * 10).as_unit(unit)
        result -= tdi.values
        tm.assert_index_equal(result, expected)
        with pytest.raises(TypeError, match=msg):
            tdi.values -= dti
        with pytest.raises(TypeError, match=msg):
            tdi._values -= dti

    def test_dta_add_sub_index(self, tz_naive_fixture: pd.Timestamp.tz_localize) -> None:
        """Add/subtract index from datetimearray"""
        dti: pd.DatetimeIndex = date_range('20130101', periods=3, tz=tz_naive_fixture)
        dta: np.ndarray = dti.array
        result: pd.DatetimeIndex = dta - dti
        expected: pd.DatetimeIndex = dti - dti
        tm.assert_index_equal(result, expected)
        tdi: pd.TimedeltaIndex = result
        result: pd.DatetimeIndex = dta + tdi
        expected: pd.DatetimeIndex = dti + tdi
        tm.assert_index_equal(result, expected)

    def test_sub_dti_dti(self, unit: str) -> None:
        """Subtract datetimeindex from datetimeindex"""
        dti: pd.DatetimeIndex = date_range('20130101', periods=3, unit=unit)
        dti_tz: pd.DatetimeIndex = date_range('20130101', periods=3, unit=unit).tz_localize('US/Eastern')
        expected: pd.TimedeltaIndex = TimedeltaIndex([0, 0, 0]).as_unit(unit)
        result: pd.DatetimeIndex = dti - dti
        tm.assert_index_equal(result, expected)
        result: pd.DatetimeIndex = dti_tz - dti_tz
        tm.assert_index_equal(result, expected)
        msg: str = 'Cannot subtract tz-naive and tz-aware datetime-like objects'
        with pytest.raises(TypeError, match=msg):
            dti_tz - dti
        with pytest.raises(TypeError, match=msg):
            dti - dti_tz
        dti -= dti
        tm.assert_index_equal(dti, expected)
        dti1: pd.DatetimeIndex = date_range('20130101', periods=3, unit=unit)
        dti2: pd.DatetimeIndex = date_range('20130101', periods=4, unit=unit)
        msg: str = 'cannot add indices of unequal length'
        with pytest.raises(ValueError, match=msg):
            dti1 - dti2
        dti1: pd.DatetimeIndex = DatetimeIndex(['2012-01-01', np.nan, '2012-01-03']).as_unit(unit)
        dti2: pd.DatetimeIndex = DatetimeIndex(['2012-01-02', '2012-01-03', np.nan]).as_unit(unit)
        expected: pd.TimedeltaIndex = TimedeltaIndex(['1 days', np.nan, np.nan]).as_unit(unit)
        result: pd.DatetimeIndex = dti2 - dti1
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize('op', [operator.add, operator.sub])
    def test_timedelta64_equal_timedelta_supported_ops(self, op: operator, box_with_array) -> None:
        """Timedelta64 equal timedelta supported ops"""
        ser: pd.Series = Series([Timestamp('20130301'), Timestamp('20130228 23:00:00'), Timestamp('20130228 22:00:00'), Timestamp('20130228 21:00:00')])
        obj: pd.Series = box_with_array(ser)
        intervals: list = ['D', 'h', 'm', 's', 'us']

        def timedelta64(*args: object) -> np.timedelta64:
            return np.sum(list(starmap(np.timedelta64, zip(args, intervals))))

        for d, h, m, s, us in product(*[range(2)] * 5):
            nptd: np.timedelta64 = timedelta64(d, h, m, s, us)
            pytd: timedelta = timedelta(days=d, hours=h, minutes=m, seconds=s, microseconds=us)
            lhs: pd.Series = op(obj, nptd)
            rhs: pd.Series = op(obj, pytd)
            tm.assert_equal(lhs, rhs)

    def test_ops_nat_mixed_datetime64_timedelta64(self) -> None:
        """Operations with NaT and mixed datetime64/timedelta64"""
        timedelta_series: pd.Series = Series([NaT, Timedelta('1s')])
        datetime_series: pd.Series = Series([NaT, Timestamp('19900315')])
        nat_series_dtype_timedelta: pd.Series = Series([NaT, NaT], dtype='timedelta64[ns]')
        nat_series_dtype_timestamp: pd.Series = Series([NaT, NaT], dtype='datetime64[ns]')
        single_nat_dtype_datetime: pd.Series = Series([NaT], dtype='datetime64[ns]')
        single_nat_dtype_timedelta: pd.Series = Series([NaT], dtype='timedelta64[ns]')
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
        """Ufunc coercions"""
        idx: pd.DatetimeIndex = date_range('2011-01-01', periods=3, freq='2D', name='x', unit=unit)
        delta: np.timedelta64 = np.timedelta64(1, 'D')
        exp: pd.DatetimeIndex = date_range('2011-01-02', periods=3, freq='2D', name='x', unit=unit)
        for result in [idx + delta, np.add(idx, delta)]:
            assert isinstance(result, pd.DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == '2D'
        exp: pd.DatetimeIndex = date_range('2010-12-31', periods=3, freq='2D', name='x', unit=unit)
        for result in [idx - delta, np.subtract(idx, delta)]:
            assert isinstance(result, pd.DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == '2D'
        idx: pd.DatetimeIndex = idx._with_freq(None)
        delta: np.ndarray = np.array([np.timedelta64(1, 'D'), np.timedelta64(2, 'D'), np.timedelta64(3, 'D')])
        exp: pd.DatetimeIndex = DatetimeIndex(['2011-01-02', '2011-01-05', '2011-01-08'], name='x').as_unit(unit)
        for result in [idx + delta, np.add(idx, delta)]:
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq
        exp: pd.DatetimeIndex = DatetimeIndex(['2010-12-31', '2011-01-01', '2011-01-02'], name='x').as_unit(unit)
        for result in [idx - delta, np.subtract(idx, delta)]:
            assert isinstance(result, pd.DatetimeIndex)
            tm.assert_index_equal(result, exp)
            assert result.freq == exp.freq

    def test_dti_add_series(self, tz_naive_fixture: pd.Timestamp.tz_localize, names: list, box_with_array) -> None:
        """Add series to datetimeindex"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        index: pd.DatetimeIndex = DatetimeIndex(['2016-06-28 05:30', '2016-06-28 05:31'], tz=tz, name=names[0]).as_unit('ns')
        ser: pd.Series = Series([Timedelta(seconds=5)] * 2, index=index, name=names[1])
        expected: pd.Series = Series(index + Timedelta(seconds=5), index=index, name=names[2])
        expected.name = names[2]
        assert expected.dtype == index.dtype
        result: pd.Series = ser + index
        tm.assert_series_equal(result, expected)
        result2: pd.Series = index + ser
        tm.assert_series_equal(result2, expected)
        expected: pd.DatetimeIndex = index + Timedelta(seconds=5)
        result3: pd.Index = ser.values + index
        tm.assert_index_equal(result3, expected)
        result4: pd.Index = index + ser.values
        tm.assert_index_equal(result4, expected)

    @pytest.mark.parametrize('op', [operator.add, roperator.radd, operator.sub])
    def test_dti_addsub_offset_arraylike(self, performance_warning: str, tz_naive_fixture: pd.Timestamp.tz_localize, names: list, op: operator, index_or_series: pd.Series) -> None:
        """Add/subtract offset array-like to datetimeindex"""
        other_box: pd.Series = index_or_series
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = date_range('2017-01-01', periods=2, tz=tz, name=names[0])
        other: pd.Series = other_box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)], name=names[1])
        xbox: pd.Series = get_upcast_box(dti, other)
        with tm.assert_produces_warning(performance_warning):
            res: pd.Series = op(dti, other)
        expected: pd.Series = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))], name=names[2], freq='infer')
        expected: pd.Series = tm.box_expected(expected, xbox).astype(object)
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize('other_box', [pd.Index, np.array])
    def test_dti_addsub_object_arraylike(self, performance_warning: str, tz_naive_fixture: pd.Timestamp.tz_localize, box_with_array, other_box: pd.Series) -> None:
        """Add/subtract object array-like to datetimeindex"""
        tz: pd.Timestamp.tz_localize = tz_naive_fixture
        dti: pd.DatetimeIndex = date_range('2017-01-01', periods=2, tz=tz)
        dtarr: pd.Series = tm.box_expected(dti, box_with_array)
        other: pd.Series = other_box([pd.offsets.MonthEnd(), Timedelta(days=4)])
        xbox: pd.Series = get_upcast_box(dtarr, other)
        expected: pd.DatetimeIndex = DatetimeIndex(['2017-01-31', '2017-01-06'], tz=tz_naive_fixture)
        expected: pd.Series = tm.box_expected(expected, xbox).astype(object)
        with tm.assert_produces_warning(performance_warning):
            result: pd.Series = dtarr + other
        tm.assert_equal(result, expected)
        expected: pd.DatetimeIndex = DatetimeIndex(['2016-12-31', '2016-12-29'], tz=tz_naive_fixture)
        expected: pd.Series = tm.box_expected(expected, xbox).astype(object)
        with tm.assert_produces_warning(performance_warning):
            result: pd.Series = dtarr - other
        tm.assert_equal(result, expected)

@pytest.mark.parametrize('years', [-1, 0, 1])
@pytest.mark.parametrize('months', [-2, 0, 2])
def test_shift_months(years: int, months: int, unit: str) -> None:
    """Shift months"""
    dti: pd.DatetimeIndex = DatetimeIndex([Timestamp('2000-01-05 00:15:00'), Timestamp('2000-01-31 00:23:00'), Timestamp('2000-01-01'), Timestamp('2000-02-29'), Timestamp('2000-12-31')]).as_unit(unit)
    shifted: np.ndarray = shift_months(dti.asi8, years * 12 + months, reso=dti._data._creso)
    shifted_dt64: np.ndarray = shifted.view(f'M8[{dti.unit}]')
    actual: pd.DatetimeIndex = DatetimeIndex(shifted_dt64)
    raw: list = [x + pd.offsets.DateOffset(years=years, months=months) for x in dti]
    expected: pd.DatetimeIndex = DatetimeIndex(raw).as_unit(dti.unit)
    tm.assert_index_equal(actual, expected)

def test_dt64arr_addsub_object_dtype_2d(performance_warning: str) -> None:
    """Add/subtract object-dtype 2D array"""
    dti: pd.DatetimeIndex = date_range('1994-02-13', freq='2W', periods=4)
    dta: np.ndarray = dti._data.reshape((4, 1))
    other: np.ndarray = np.array([[pd.offsets.Day(n)] for n in range(4)])
    assert other.shape == dta.shape
    with tm.assert_produces_warning(performance_warning):
        result: pd.Series = dta + other
    with tm.assert_produces_warning(performance_warning):
        expected: pd.Series = (dta[:, 0] + other[:, 0]).reshape(-1, 1)
    tm.assert_numpy_array_equal(result, expected)
    with tm.assert_produces_warning(performance_warning):
        result2: pd.Series = dta - dta.astype(object)
    assert result2.shape == (4, 1)
    assert all((td._value == 0 for td in result2.ravel()))

def test_non_nano_dt64_addsub_np_nat_scalars() -> None:
    """Non-nano datetime64 add/sub np.nan scalar"""
    ser: pd.Series = Series([1233242342344, 232432434324, 332434242344], dtype='datetime64[ms]')
    result: pd.Series = ser - np.datetime64('nat', 'ms')
    expected: pd.Series = Series([NaT] * 3, dtype='timedelta64[ms]')
    tm.assert_series_equal(result, expected)
    result: pd.Series = ser + np.timedelta64('nat', 'ms')
    expected: pd.Series = Series([NaT] * 3, dtype='datetime64[ms]')
    tm.assert_series_equal(result, expected)

def test_non_nano_dt64_addsub_np_nat_scalars_unitless() -> None:
    """Non-nano datetime64 add/sub np.nan scalar unitless"""
    ser: pd.Series = Series([1233242342344, 232432434324, 332434242344], dtype='datetime64[ms]')
    result: pd.Series = ser - np.datetime64('nat')
    expected: pd.Series = Series([NaT] * 3, dtype='timedelta64[ns]')
    tm.assert_series_equal(result, expected)
    result: pd.Series = ser + np.timedelta64('nat')
    expected: pd.Series = Series([NaT] * 3, dtype='datetime64[ns]')
    tm.assert_series_equal(result, expected)

def test_non_nano_dt64_addsub_np_nat_scalars_unsupported_unit() -> None:
    """Non-nano datetime64 add/sub np.nan scalar unsupported unit"""
    ser: pd.Series = Series([12332, 23243, 33243], dtype='datetime64[s]')
    result: pd.Series = ser - np.datetime64('nat', 'D')
    expected: pd.Series = Series([NaT] * 3, dtype='timedelta64[s]')
    tm.assert_series_equal(result, expected)
    result: pd.Series = ser + np.timedelta64('nat', 'D')
    expected: pd.Series = Series([NaT] * 3, dtype='datetime64[s]')
    tm.assert_series_equal(result, expected)
