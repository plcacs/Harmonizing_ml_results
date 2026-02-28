from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import Categorical, DataFrame, DatetimeIndex, Index, NaT, Period, PeriodIndex, RangeIndex, Series, Timedelta, TimedeltaIndex, Timestamp, date_range, isna, period_range, timedelta_range, to_timedelta
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics

def get_objs() -> list:
    indexes = [Index([True, False] * 5, name='a'), Index(np.arange(10), dtype=np.int64, name='a'), Index(np.arange(10), dtype=np.float64, name='a'), DatetimeIndex(date_range('2020-01-01', periods=10), name='a'), DatetimeIndex(date_range('2020-01-01', periods=10), name='a').tz_localize(tz='US/Eastern'), PeriodIndex(period_range('2020-01-01', periods=10, freq='D'), name='a'), Index([str(i) for i in range(10)], name='a')]
    arr = np.random.default_rng(2).standard_normal(10)
    series = [Series(arr, index=idx, name='a') for idx in indexes]
    objs = indexes + series
    return objs

class TestReductions:
    @pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
    @pytest.mark.parametrize('opname: str', ['max', 'min'])
    @pytest.mark.parametrize('obj: object', get_objs())
    def test_ops(self, opname: str, obj: object) -> None:
        result = getattr(obj, opname)()
        if not isinstance(obj, PeriodIndex):
            if isinstance(obj.values, ArrowStringArrayNumpySemantics):
                expected = getattr(np.array(obj.values), opname)()
            else:
                expected = getattr(obj.values, opname)()
        else:
            expected = Period(ordinal=getattr(obj.asi8, opname)(), freq=obj.freq)
        if getattr(obj, 'tz', None) is not None:
            expected = expected.astype('M8[ns]').astype('int64')
            assert result._value == expected
        else:
            assert result == expected

    @pytest.mark.parametrize('opname: str', ['max', 'min'])
    def test_nanminmax(self, opname: str, dtype: str, val: object, index_or_series: type) -> None:
        klass = index_or_series
        def check_missing(res: object) -> bool:
            if dtype == 'datetime64[ns]':
                return res is NaT
            elif dtype in ['Int64', 'boolean']:
                return res is pd.NA
            else:
                return isna(res)
        obj = klass([None], dtype=dtype)
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))
        obj = klass([], dtype=dtype)
        assert check_missing(getattr(obj, opname)())
        assert check_missing(getattr(obj, opname)(skipna=False))
        if dtype == 'object':
            return
        obj = klass([None, val], dtype=dtype)
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))
        obj = klass([None, val, None], dtype=dtype)
        assert getattr(obj, opname)() == val
        assert check_missing(getattr(obj, opname)(skipna=False))

    @pytest.mark.parametrize('opname: str', ['max', 'min'])
    def test_nanargminmax(self, opname: str, index_or_series: type) -> None:
        klass = index_or_series
        arg_op = 'arg' + opname if klass is Index else 'idx' + opname
        obj = klass([NaT, datetime(2011, 11, 1)])
        assert getattr(obj, arg_op)() == 1
        with pytest.raises(ValueError, match='Encountered an NA value'):
            getattr(obj, arg_op)(skipna=False)
        obj = klass([NaT, datetime(2011, 11, 1), NaT])
        assert getattr(obj, arg_op)() == 1
        with pytest.raises(ValueError, match='Encountered an NA value'):
            getattr(obj, arg_op)(skipna=False)

    @pytest.mark.parametrize('opname: str', ['max', 'min'])
    @pytest.mark.parametrize('dtype: str', ['M8[ns]', 'datetime64[ns, UTC]'])
    def test_nanops_empty_object(self, opname: str, index_or_series: type, dtype: str) -> None:
        klass = index_or_series
        arg_op = 'arg' + opname if klass is Index else 'idx' + opname
        obj = klass([], dtype=dtype)
        assert getattr(obj, opname)() is NaT
        assert getattr(obj, opname)(skipna=False) is NaT
        with pytest.raises(ValueError, match='empty sequence'):
            getattr(obj, arg_op)()
        with pytest.raises(ValueError, match='empty sequence'):
            getattr(obj, arg_op)(skipna=False)

    def test_argminmax(self) -> None:
        obj = Index(np.arange(5, dtype='int64'))
        assert obj.argmin() == 0
        assert obj.argmax() == 4
        obj = Index([np.nan, 1, np.nan, 2])
        assert obj.argmin() == 1
        assert obj.argmax() == 3
        with pytest.raises(ValueError, match='Encountered an NA value'):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match='Encountered an NA value'):
            obj.argmax(skipna=False)
        obj = Index([np.nan])
        with pytest.raises(ValueError, match='Encountered all NA values'):
            obj.argmin()
        with pytest.raises(ValueError, match='Encountered all NA values'):
            obj.argmax()
        with pytest.raises(ValueError, match='Encountered an NA value'):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match='Encountered an NA value'):
            obj.argmax(skipna=False)
        obj = Index([NaT, datetime(2011, 11, 1), datetime(2011, 11, 2), NaT])
        assert obj.argmin() == 1
        assert obj.argmax() == 2
        with pytest.raises(ValueError, match='Encountered an NA value'):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match='Encountered an NA value'):
            obj.argmax(skipna=False)
        obj = Index([NaT])
        with pytest.raises(ValueError, match='Encountered all NA values'):
            obj.argmin()
        with pytest.raises(ValueError, match='Encountered all NA values'):
            obj.argmax()
        with pytest.raises(ValueError, match='Encountered an NA value'):
            obj.argmin(skipna=False)
        with pytest.raises(ValueError, match='Encountered an NA value'):
            obj.argmax(skipna=False)

    @pytest.mark.parametrize('op: str', ['max', 'min'])
    def test_same_tz_min_max_axis_1(self, op: str, expected_col: str) -> None:
        df = DataFrame(date_range('2016-01-01 00:00:00', periods=3, tz='UTC'), columns=['a'])
        df['b'] = df.a.subtract(Timedelta(seconds=3600))
        result = getattr(df, op)(axis=1)
        expected = df[expected_col].rename(None)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('func: str', ['maximum', 'minimum'])
    def test_numpy_reduction_with_tz_aware_dtype(self, tz_aware_fixture: object, func: str) -> None:
        tz = tz_aware_fixture
        arg = pd.to_datetime(['2019']).tz_localize(tz)
        expected = Series(arg)
        result = getattr(np, func)(expected, expected)
        tm.assert_series_equal(result, expected)

    def test_nan_int_timedelta_sum(self) -> None:
        df = DataFrame({'A': Series([1, 2, NaT], dtype='timedelta64[ns]'), 'B': Series([1, 2, np.nan], dtype='Int64')})
        expected = Series({'A': Timedelta(3), 'B': 3})
        result = df.sum()
        tm.assert_series_equal(result, expected)

class TestIndexReductions:
    @pytest.mark.parametrize('start: int', [0, 400, 500])
    @pytest.mark.parametrize('stop: int', [400, 0, 10 ** 6])
    @pytest.mark.parametrize('step: int', [3, -6, 4])
    def test_max_min_range(self, start: int, stop: int, step: int) -> None:
        idx = RangeIndex(start, stop, step)
        expected = idx._values.max()
        result = idx.max()
        assert result == expected
        result2 = idx.max(skipna=False)
        assert result2 == expected
        expected = idx._values.min()
        result = idx.min()
        assert result == expected
        result2 = idx.min(skipna=False)
        assert result2 == expected
        idx = RangeIndex(start, stop, -step)
        assert isna(idx.max())
        assert isna(idx.min())

    def test_minmax_timedelta64(self) -> None:
        idx1 = TimedeltaIndex(['1 days', '2 days', '3 days'])
        assert idx1.is_monotonic_increasing
        idx2 = TimedeltaIndex(['1 days', np.nan, '3 days', 'NaT'])
        assert not idx2.is_monotonic_increasing
        for idx in [idx1, idx2]:
            assert idx.min() == Timedelta('1 days')
            assert idx.max() == Timedelta('3 days')
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize('op: str', ['min', 'max'])
    def test_minmax_timedelta_empty_or_na(self, op: str) -> None:
        obj = TimedeltaIndex([])
        assert getattr(obj, op)() is NaT
        obj = TimedeltaIndex([NaT])
        assert getattr(obj, op)() is NaT
        obj = TimedeltaIndex([NaT, NaT, NaT])
        assert getattr(obj, op)() is NaT

    def test_numpy_minmax_timedelta64(self) -> None:
        td = timedelta_range('16815 days', '16820 days', freq='D')
        assert np.min(td) == Timedelta('16815 days')
        assert np.max(td) == Timedelta('16820 days')
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(td, out=0)
        assert np.argmin(td) == 0
        assert np.argmax(td) == 5
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(td, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(td, out=0)

    def test_timedelta_ops(self) -> None:
        s = Series([Timestamp('20130101') + timedelta(seconds=i * i) for i in range(10)])
        td = s.diff()
        result = td.mean()
        expected = to_timedelta(timedelta(seconds=9))
        assert result == expected
        result = td.to_frame().mean()
        assert result[0] == expected
        result = td.quantile(0.1)
        expected = Timedelta(np.timedelta64(2600, 'ms'))
        assert result == expected
        result = td.median()
        expected = to_timedelta('00:00:09')
        assert result == expected
        result = td.to_frame().median()
        assert result[0] == expected
        result = td.sum()
        expected = to_timedelta('00:01:21')
        assert result == expected
        result = td.to_frame().sum()
        assert result[0] == expected
        result = td.std()
        expected = to_timedelta(Series(td.dropna().values).std())
        assert result == expected
        result = td.to_frame().std()
        assert result[0] == expected
        s = Series([Timestamp('2015-02-03'), Timestamp('2015-02-07')])
        assert s.diff().median() == timedelta(days=4)
        s = Series([Timestamp('2015-02-03'), Timestamp('2015-02-07'), Timestamp('2015-02-15')])
        assert s.diff().median() == timedelta(days=6)

    @pytest.mark.parametrize('opname: str', ['skew', 'kurt', 'sem', 'prod', 'var'])
    def test_invalid_td64_reductions(self, opname: str, s: Series) -> None:
        td = s.diff()
        msg = '|'.join([f"reduction operation '{opname}' not allowed for this dtype", f'cannot perform {opname} with type timedelta64\\[ns\\]', f"does not support operation '{opname}'"])
        with pytest.raises(TypeError, match=msg):
            getattr(td, opname)()
        with pytest.raises(TypeError, match=msg):
            getattr(td.to_frame(), opname)(numeric_only=False)

    def test_minmax_tz(self, tz_naive_fixture: object) -> None:
        tz = tz_naive_fixture
        idx1 = DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], tz=tz)
        assert idx1.is_monotonic_increasing
        idx2 = DatetimeIndex(['2011-01-01', NaT, '2011-01-03', '2011-01-02', NaT], tz=tz)
        assert not idx2.is_monotonic_increasing
        for idx in [idx1, idx2]:
            assert idx.min() == Timestamp('2011-01-01', tz=tz)
            assert idx.max() == Timestamp('2011-01-03', tz=tz)
            assert idx.argmin() == 0
            assert idx.argmax() == 2

    @pytest.mark.parametrize('op: str', ['min', 'max'])
    def test_minmax_nat_datetime64(self, op: str) -> None:
        obj = DatetimeIndex([])
        assert isna(getattr(obj, op)())
        obj = DatetimeIndex([NaT])
        assert isna(getattr(obj, op)())
        obj = DatetimeIndex([NaT, NaT, NaT])
        assert isna(getattr(obj, op)())

    def test_numpy_minmax_integer(self) -> None:
        idx = Index([1, 2, 3])
        expected = idx.values.max()
        result = np.max(idx)
        assert result == expected
        expected = idx.values.min()
        result = np.min(idx)
        assert result == expected
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)
        expected = idx.values.argmax()
        result = np.argmax(idx)
        assert result == expected
        expected = idx.values.argmin()
        result = np.argmin(idx)
        assert result == expected
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(idx, out=0)

    def test_numpy_minmax_range(self) -> None:
        idx = RangeIndex(0, 10, 3)
        result = np.max(idx)
        assert result == 9
        result = np.min(idx)
        assert result == 0
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(idx, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(idx, out=0)

    def test_numpy_minmax_datetime64(self) -> None:
        dr = date_range(start='2016-01-15', end='2016-01-20')
        assert np.min(dr) == Timestamp('2016-01-15 00:00:00')
        assert np.max(dr) == Timestamp('2016-01-20 00:00:00')
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(dr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(dr, out=0)
        assert np.argmin(dr) == 0
        assert np.argmax(dr) == 5
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(dr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(dr, out=0)

    def test_minmax_period(self) -> None:
        idx1 = PeriodIndex([NaT, '2011-01-01', '2011-01-02', '2011-01-03'], freq='D')
        assert not idx1.is_monotonic_increasing
        assert idx1[1:].is_monotonic_increasing
        idx2 = PeriodIndex(['2011-01-01', NaT, '2011-01-03', '2011-01-02', NaT], freq='D')
        assert not idx2.is_monotonic_increasing
        for idx in [idx1, idx2]:
            assert idx.min() == Period('2011-01-01', freq='D')
            assert idx.max() == Period('2011-01-03', freq='D')
        assert idx1.argmin() == 1
        assert idx2.argmin() == 0
        assert idx1.argmax() == 3
        assert idx2.argmax() == 2

    @pytest.mark.parametrize('op: str', ['min', 'max'])
    @pytest.mark.parametrize('data: list', [[], [NaT], [NaT, NaT, NaT]])
    def test_minmax_period_empty_nat(self, op: str, data: list) -> None:
        obj = PeriodIndex(data, freq='M')
        result = getattr(obj, op)()
        assert result is NaT

    def test_numpy_minmax_period(self) -> None:
        pr = period_range(start='2016-01-15', end='2016-01-20')
        assert np.min(pr) == Period('2016-01-15', freq='D')
        assert np.max(pr) == Period('2016-01-20', freq='D')
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.min(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.max(pr, out=0)
        assert np.argmin(pr) == 0
        assert np.argmax(pr) == 5
        errmsg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=errmsg):
            np.argmin(pr, out=0)
        with pytest.raises(ValueError, match=errmsg):
            np.argmax(pr, out=0)

    def test_min_max_categorical(self) -> None:
        ci = pd.CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=False)
        msg = 'Categorical is not ordered for operation min\\nyou can use .as_ordered\\(\\) to change the Categorical to an ordered one\\n'
        with pytest.raises(TypeError, match=msg):
            ci.min()
        msg = 'Categorical is not ordered for operation max\\nyou can use .as_ordered\\(\\) to change the Categorical to an ordered one\\n'
        with pytest.raises(TypeError, match=msg):
            ci.max()
        ci = pd.CategoricalIndex(list('aabbca'), categories=list('cab'), ordered=True)
        assert ci.min() == 'c'
        assert ci.max() == 'b'

class TestSeriesReductions:
    def test_sum_inf(self) -> None:
        s = Series(np.random.default_rng(2).standard_normal(10))
        s2 = s.copy()
        s[5:8] = np.inf
        s2[5:8] = np.nan
        assert np.isinf(s.sum())
        arr = np.random.default_rng(2).standard_normal((100, 100)).astype('f4')
        arr[:, 2] = np.inf
        res = nanops.nansum(arr, axis=1)
        assert np.isinf(res).all()

    @pytest.mark.parametrize('dtype: str', ['float64', 'Float32', 'Int64', 'boolean', 'object'])
    @pytest.mark.parametrize('use_bottleneck: bool', [True, False])
    @pytest.mark.parametrize('method: str', ['sum', 'prod'])
    def test_empty(self, method: str, unit: str, use_bottleneck: bool, dtype: str) -> None:
        with pd.option_context('use_bottleneck', use_bottleneck):
            s = Series([], dtype=dtype)
            result = getattr(s, method)()
            assert result == 0.0
            result = getattr(s, method)(min_count=0)
            assert result == 0.0
            result = getattr(s, method)(min_count=1)
            assert isna(result)
            result = getattr(s, method)(skipna=True)
            result == 0.0
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == 0.0
            result = getattr(s, method)(skipna=True, min_count=1)
            assert isna(result)
            result = getattr(s, method)(skipna=False, min_count=0)
            assert result == 0.0
            result = getattr(s, method)(skipna=False, min_count=1)
            assert isna(result)
            s = Series([np.nan], dtype=dtype)
            result = getattr(s, method)()
            assert result == 0.0
            result = getattr(s, method)(min_count=0)
            assert result == 0.0
            result = getattr(s, method)(min_count=1)
            assert isna(result)
            result = getattr(s, method)(skipna=True)
            result == 0.0
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == 0.0
            result = getattr(s, method)(skipna=True, min_count=1)
            assert isna(result)
            s = Series([np.nan, 1], dtype=dtype)
            result = getattr(s, method)()
            assert result == 1.0
            result = getattr(s, method)(min_count=0)
            assert result == 1.0
            result = getattr(s, method)(min_count=1)
            assert result == 1.0
            result = getattr(s, method)(skipna=True)
            assert result == 1.0
            result = getattr(s, method)(skipna=True, min_count=0)
            assert result == 1.0
            df = DataFrame(np.empty((10, 0)), dtype=dtype)
            assert (getattr(df, method)(axis=1) == 0.0).all()
            s = Series([1], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)
            result = getattr(s, method)(skipna=False, min_count=2)
            assert isna(result)
            s = Series([np.nan], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)
            s = Series([np.nan, 1], dtype=dtype)
            result = getattr(s, method)(min_count=2)
            assert isna(result)

    @pytest.mark.parametrize('method: str', ['mean', 'var'])
    @pytest.mark.parametrize('dtype: str', ['Float64', 'Int64', 'boolean'])
    def test_ops_consistency_on_empty_nullable(self, method: str, dtype: str) -> None:
        eser = Series([], dtype=dtype)
        result = getattr(eser, method)()
        assert result is pd.NA
        nser = Series([np.nan], dtype=dtype)
        result = getattr(nser, method)()
        assert result is pd.NA

    @pytest.mark.parametrize('method: str', ['mean', 'median', 'std', 'var'])
    def test_ops_consistency_on_empty(self, method: str) -> None:
        result = getattr(Series(dtype=float), method)()
        assert isna(result)
        tdser = Series([], dtype='m8[ns]')
        if method == 'var':
            msg = '|'.join(["operation 'var' not allowed", 'cannot perform var with type timedelta64\\[ns\\]', "does not support operation 'var'"])
            with pytest.raises(TypeError, match=msg):
                getattr(tdser, method)()
        else:
            result = getattr(tdser, method)()
            assert result is NaT

    def test_nansum_buglet(self) -> None:
        ser = Series([1.0, np.nan], index=[0, 1])
        result = np.nansum(ser)
        tm.assert_almost_equal(result, 1)

    @pytest.mark.parametrize('use_bottleneck: bool', [True, False])
    @pytest.mark.parametrize('dtype: str', ['int32', 'int64'])
    def test_sum_overflow_int(self, use_bottleneck: bool, dtype: str) -> None:
        with pd.option_context('use_bottleneck', use_bottleneck):
            v = np.arange(5000000, dtype=dtype)
            s = Series(v)
            result = s.sum(skipna=False)
            assert int(result) == v.sum(dtype='int64')
            result = s.min(skipna=False)
            assert int(result) == 0
            result = s.max(skipna=False)
            assert int(result) == v[-1]

    @pytest.mark.parametrize('use_bottleneck: bool', [True, False])
    @pytest.mark.parametrize('dtype: str', ['float32', 'float64'])
    def test_sum_overflow_float(self, use_bottleneck: bool, dtype: str) -> None:
        with pd.option_context('use_bottleneck', use_bottleneck):
            v = np.arange(5000000, dtype=dtype)
            s = Series(v)
            result = s.sum(skipna=False)
            assert result == v.sum(dtype=dtype)
            result = s.min(skipna=False)
            assert np.allclose(float(result), 0.0)
            result = s.max(skipna=False)
            assert np.allclose(float(result), v[-1])

    def test_mean_masked_overflow(self) -> None:
        val = 100000000000000000
        n_elements = 100
        na = np.array([val] * n_elements)
        ser = Series([val] * n_elements, dtype='Int64')
        result_numpy = np.mean(na)
        result_masked = ser.mean()
        assert result_masked - result_numpy == 0
        assert result_masked == 1e+17

    @pytest.mark.parametrize('ddof: int', [1, 0])
    def test_var_masked_array(self, ddof: int) -> None:
        ser = Series([1, 2, 3, 4, 5], dtype='Int64')
        ser_numpy_dtype = Series([1, 2, 3, 4, 5], dtype='int64')
        result = ser.var(ddof=ddof)
        result_numpy_dtype = ser_numpy_dtype.var(ddof=ddof)
        assert result == result_numpy_dtype
        assert result == 2.5

    @pytest.mark.parametrize('dtype: str', ('m8[ns]', 'M8[ns]', 'M8[ns, UTC]'))
    def test_empty_timeseries_reductions_return_nat(self, dtype: str, skipna: bool) -> None:
        assert Series([], dtype=dtype).min(skipna=skipna) is NaT
        assert Series([], dtype=dtype).max(skipna=skipna) is NaT

    def test_numpy_argmin(self) -> None:
        data = np.arange(1, 11)
        s = Series(data, index=data)
        result = np.argmin(s)
        expected = np.argmin(data)
        assert result == expected
        result = s.argmin()
        assert result == expected
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmin(s, out=data)

    def test_numpy_argmax(self) -> None:
        data = np.arange(1, 11)
        ser = Series(data, index=data)
        result = np.argmax(ser)
        expected = np.argmax(data)
        assert result == expected
        result = ser.argmax()
        assert result == expected
        msg = "the 'out' parameter is not supported"
        with pytest.raises(ValueError, match=msg):
            np.argmax(ser, out=data)

    def test_idxmin_dt64index(self, unit: str) -> None:
        dti = DatetimeIndex(['NaT', '2015-02-08', 'NaT']).as_unit(unit)
        ser = Series([1.0, 2.0, np.nan], index=dti)
        with pytest.raises(ValueError, match='Encountered an NA value'):
            ser.idxmin(skipna=False)
        with pytest.raises(ValueError, match='Encountered an NA value'):
            ser.idxmax(skipna=False)
        df = ser.to_frame()
        with pytest.raises(ValueError, match='Encountered an NA value'):
            df.idxmin(skipna=False)
        with pytest.raises(ValueError, match='Encountered an NA value'):
            df.idxmax(skipna=False)

    def test_idxmin(self) -> None:
        string_series = Series(range(20), dtype=np.float64, name='series')
        string_series[5:15] = np.nan
        assert string_series[string_series.idxmin()] == string_series.min()
        with pytest.raises(ValueError, match='Encountered an NA value'):
            string_series.idxmin(skipna=False)
        nona = string_series.dropna()
        assert nona[nona.idxmin()] == nona.min()
        assert nona.index.values.tolist().index(nona.idxmin()) == nona.values.argmin()
        allna = string_series * np.nan
        msg = 'Encountered all NA values'
        with pytest.raises(ValueError, match=msg):
            allna.idxmin()
        s = Series(date_range('20130102', periods=6))
        result = s.idxmin()
        assert result == 0
        s[0] = np.nan
        result = s.idxmin()
        assert result == 1

    def test_idxmax(self) -> None:
        string_series = Series(range(20), dtype=np.float64, name='series')
        string_series[5:15] = np.nan
        assert string_series[string_series.idxmax()] == string_series.max()
        with pytest.raises(ValueError, match='Encountered an NA value'):
            assert isna(string_series.idxmax(skipna=False))
        nona = string_series.dropna()
        assert nona[nona.idxmax()] == nona.max()
        assert nona.index.values.tolist().index(nona.idxmax()) == nona.values.argmax()
        allna = string_series * np.nan
        msg = 'Encountered all NA values'
        with pytest.raises(ValueError, match=msg):
            allna.idxmax()
        s = Series(date_range('20130102', periods=6))
        result = s.idxmax()
        assert result == 5
        s[5] = np.nan
        result = s.idxmax()
        assert result == 4
        s = Series([1, 2, 3], [1.1, 2.1, 3.1])
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1
        s = Series(s.index, s.index)
        result = s.idxmax()
        assert result == 3.1
        result = s.idxmin()
        assert result == 1.1

    def test_all_any(self) -> None:
        ts = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        bool_series = ts > 0
        assert not bool_series.all()
        assert bool_series.any()
        s = Series(['abc', True])
        assert s.any()

    def test_numpy_all_any(self, index_or_series: type) -> None:
        idx = index_or_series([0, 1, 2])
        assert not np.all(idx)
        assert np.any(idx)
        idx = Index([1, 2, 3])
        assert np.all(idx)

    def test_all_any_skipna(self) -> None:
        s1 = Series([np.nan, True])
        s2 = Series([np.nan, False])
        assert s1.all(skipna=False)
        assert s1.all(skipna=True)
        assert s2.any(skipna=False)
        assert not s2.any(skipna=True)

    def test_all_any_bool_only(self) -> None:
        df = DataFrame({'A': [True, False], 'B': [1, 2]})
        result = df.any(axis=1, bool_only=True)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_any_axis1_bool_only(self) -> None:
        df = DataFrame({'A': [True, False], 'B': [1, 2]})
        result = df.any(axis=1, bool_only=True)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_any_all_datetimelike(self) -> None:
        dta = date_range('1995-01-02', periods=3)._data
        ser = Series(dta)
        df = DataFrame(ser)
        msg = "datetime64 type does not support operation '(any|all)'"
        with pytest.raises(TypeError, match=msg):
            dta.all()
        with pytest.raises(TypeError, match=msg):
            dta.any()
        with pytest.raises(TypeError, match=msg):
            ser.all()
        with pytest.raises(TypeError, match=msg):
            ser.any()
        with pytest.raises(TypeError, match=msg):
            df.any().all()
        with pytest.raises(TypeError, match=msg):
            df.all().all()
        dta = dta.tz_localize('UTC')
        ser = Series(dta)
        df = DataFrame(ser)
        with pytest.raises(TypeError, match=msg):
            dta.all()
        with pytest.raises(TypeError, match=msg):
            dta.any()
        with pytest.raises(TypeError, match=msg):
            ser.all()
        with pytest.raises(TypeError, match=msg):
            ser.any()
        with pytest.raises(TypeError, match=msg):
            df.any().all()
        with pytest.raises(TypeError, match=msg):
            df.all().all()
        tda = dta - dta[0]
        ser = Series(tda)
        df = DataFrame(ser)
        assert tda.any()
        assert not tda.all()
        assert ser.any()
        assert not ser.all()
        assert df.any().all()
        assert not df.all().any()

    def test_any_all_string_dtype(self, any_string_dtype: pd.StringDtype) -> None:
        if isinstance(any_string_dtype, pd.StringDtype) and any_string_dtype.na_value is pd.NA:
            ser = Series(['a', 'b'], dtype=any_string_dtype)
            with pytest.raises(TypeError):
                ser.any()
            with pytest.raises(TypeError):
                ser.all()
            return
        ser = Series(['', 'a'], dtype=any_string_dtype)
        assert ser.any()
        assert not ser.all()
        assert ser.any(skipna=False)
        assert not ser.all(skipna=False)
        ser = Series([np.nan, 'a'], dtype=any_string_dtype)
        assert ser.any()
        assert ser.all()
        assert ser.any(skipna=False)
        assert ser.all(skipna=False)
        ser = Series([np.nan, ''], dtype=any_string_dtype)
        assert not ser.any()
        assert not ser.all()
        assert ser.any(skipna=False)
        assert not ser.all(skipna=False)
        ser = Series(['a', 'b'], dtype=any_string_dtype)
        assert ser.any()
        assert ser.all()
        assert ser.any(skipna=False)
        assert ser.all(skipna=False)
        ser = Series([], dtype=any_string_dtype)
        assert not ser.any()
        assert ser.all()
        assert not ser.any(skipna=False)
        assert ser.all(skipna=False)
        ser = Series([''], dtype=any_string_dtype)
        assert not ser.any()
        assert not ser.all()
        assert not ser.any(skipna=False)
        assert not ser.all(skipna=False)
        ser = Series([np.nan], dtype=any_string_dtype)
        assert not ser.any()
        assert ser.all()
        assert ser.any(skipna=False)
        assert ser.all(skipna=False)

    def test_timedelta64_analytics(self) -> None:
        dti = date_range('2012-1-1', periods=3, freq='D')
        td = Series(dti) - Timestamp('20120101')
        result = td.idxmin()
        assert result == 0
        result = td.idxmax()
        assert result == 2
        td[0] = np.nan
        result = td.idxmin()
        assert result == 1
        result = td.idxmax()
        assert result == 2
        s1 = Series(date_range('20120101', periods=3))
        s2 = Series(date_range('20120102', periods=3))
        expected = Series(s2 - s1)
        result = np.abs(s1 - s2)
        tm.assert_series_equal(result, expected)
        result = (s1 - s2).abs()
        tm.assert_series_equal(result, expected)
        result = td.max()
        expected = Timedelta('2 days')
        assert result == expected
        result = td.min()
        expected = Timedelta('1 days')
        assert result == expected

    def test_assert_idxminmax_empty_raises(self) -> None:
        """
        Cases where ``Series.argmax`` and related should raise an exception
        """
        test_input = Series([], dtype='float64')
        msg = 'attempt to get argmin of an empty sequence'
        with pytest.raises(ValueError, match=msg):
            test_input.idxmin()
        with pytest.raises(ValueError, match=msg):
            test_input.idxmin(skipna=False)
        msg = 'attempt to get argmax of an empty sequence'
        with pytest.raises(ValueError, match=msg):
            test_input.idxmax()
        with pytest.raises(ValueError, match=msg):
            test_input.idxmax(skipna=False)

    def test_idxminmax_object_dtype(self, using_infer_string: bool) -> None:
        ser = Series(['foo', 'bar', 'baz'])
        assert ser.idxmax() == 0
        assert ser.idxmax(skipna=False) == 0
        assert ser.idxmin() == 1
        assert ser.idxmin(skipna=False) == 1
        ser2 = Series([(1,), (2,)])
        assert ser2.idxmax() == 1
        assert ser2.idxmax(skipna=False) == 1
        assert ser2.idxmin() == 0
        assert ser2.idxmin(skipna=False) == 0
        if not using_infer_string:
            ser3 = Series(['foo', 'foo', 'bar', 'bar', None, np.nan, 'baz'])
            msg = "'>' not supported between instances of 'float' and 'str'"
            with pytest.raises(TypeError, match=msg):
                ser3.idxmax()
            with pytest.raises(TypeError, match=msg):
                ser3.idxmax(skipna=False)
            msg = "'<' not supported between instances of 'float' and 'str'"
            with pytest.raises(TypeError, match=msg):
                ser3.idxmin()
            with pytest.raises(TypeError, match=msg):
                ser3.idxmin(skipna=False)

    def test_idxminmax_object_tuples(self) -> None:
        ser = Series([(1, 3), (2, 2), (3, 1)])
        assert ser.idxmax() == 2
        assert ser.idxmin() == 0
        assert ser.idxmax(skipna=False) == 2
        assert ser.idxmin(skipna=False) == 0

    def test_idxminmax_object_decimals(self) -> None:
        df = DataFrame({'idx': [0, 1], 'x': [Decimal('8.68'), Decimal('42.23')], 'y': [Decimal('7.11'), Decimal('79.61')]})
        res = df.idxmax()
        exp = Series({'idx': 1, 'x': 1, 'y': 1})
        tm.assert_series_equal(res, exp)
        res2 = df.idxmin()
        exp2 = exp - 1
        tm.assert_series_equal(res2, exp2)

    def test_argminmax_object_ints(self) -> None:
        ser = Series([0, 1], dtype='object')
        assert ser.argmax() == 1
        assert ser.argmin() == 0
        assert ser.argmax(skipna=False) == 1
        assert ser.argmin(skipna=False) == 0

    def test_idxminmax_with_inf(self) -> None:
        s = Series([0, -np.inf, np.inf, np.nan])
        assert s.idxmin() == 1
        with pytest.raises(ValueError, match='Encountered an NA value'):
            s.idxmin(skipna=False)
        assert s.idxmax() == 2
        with pytest.raises(ValueError, match='Encountered an NA value'):
            s.idxmax(skipna=False)

    def test_sum_uint64(self) -> None:
        s = Series([10000000000000000000], dtype='uint64')
        result = s.sum()
        expected = np.uint64(10000000000000000000)
        tm.assert_almost_equal(result, expected)

    def test_signedness_preserved_after_sum(self) -> None:
        ser = Series([1, 2, 3, 4])
        assert ser.astype('uint8').sum().dtype == 'uint64'

class TestDatetime64SeriesReductions:
    @pytest.mark.parametrize('nat_ser: Series', [Series([NaT, NaT]), Series([NaT, Timedelta('nat')]), Series([Timedelta('nat'), Timedelta('nat')])])
    def test_minmax_nat_series(self, nat_ser: Series) -> None:
        assert nat_ser.min() is NaT
        assert nat_ser.max() is NaT
        assert nat_ser.min(skipna=False) is NaT
        assert nat_ser.max(skipna=False) is NaT

    @pytest.mark.parametrize('nat_df: list', [[NaT, NaT], [NaT, Timedelta('nat')], [Timedelta('nat'), Timedelta('nat')]])
    def test_minmax_nat_dataframe(self, nat_df: list) -> None:
        nat_df = DataFrame(nat_df)
        assert nat_df.min()[0] is NaT
        assert nat_df.max()[0] is NaT
        assert nat_df.min(skipna=False)[0] is NaT
        assert nat_df.max(skipna=False)[0] is NaT

    def test_min_max(self) -> None:
        rng = date_range('1/1/2000', '12/31/2000')
        rng2 = rng.take(np.random.default_rng(2).permutation(len(rng)))
        the_min = rng2.min()
        the_max = rng2.max()
        assert isinstance(the_min, Timestamp)
        assert isinstance(the_max, Timestamp)
        assert the_min == rng[0]
        assert the_max == rng[-1]
        assert rng.min() == rng[0]
        assert rng.max() == rng[-1]

    def test_min_max_series(self) -> None:
        rng = date_range('1/1/2000', periods=10, freq='4h')
        lvls = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
        df = DataFrame({'TS': rng, 'V': np.random.default_rng(2).standard_normal(len(rng)), 'L': lvls})
        result = df.TS.max()
        exp = Timestamp(df.TS.iat[-1])
        assert isinstance(result, Timestamp)
        assert result == exp
        result = df.TS.min()
        exp = Timestamp(df.TS.iat[0])
        assert isinstance(result, Timestamp)
        assert result == exp

class TestCategoricalSeriesReductions:
    @pytest.mark.parametrize('function: str', ['min', 'max'])
    def test_min_max_unordered_raises(self, function: str) -> None:
        cat = Series(Categorical(['a', 'b', 'c', 'd'], ordered=False))
        msg = f'Categorical is not ordered for operation {function}'
        with pytest.raises(TypeError, match=msg):
            getattr(cat, function)()

    @pytest.mark.parametrize('values: list', [list('abc'), list('abc') + [np.nan]])
    @pytest.mark.parametrize('categories: list', [list('abc'), list('cba')])
    @pytest.mark.parametrize('function: str', ['min', 'max'])
    def test_min_max_ordered(self, values: list, categories: list, function: str) -> None:
        cat = Series(Categorical(values, categories=categories, ordered=True))
        result = getattr(cat, function)(skipna=True)
        expected = categories[0] if function == 'min' else categories[2]
        assert result == expected

    @pytest.mark.parametrize('function: str', ['min', 'max'])
    def test_min_max_ordered_with_nan_only(self, function: str, skipna: bool) -> None:
        cat = Series(Categorical([np.nan], categories=[1, 2], ordered=True))
        result = getattr(cat, function)(skipna=skipna)
        assert result is np.nan

    @pytest.mark.parametrize('function: str', ['min', 'max'])
    def test_min_max_skipna(self, function: str, skipna: bool) -> None:
        cat = Series(Categorical(['a', 'b', np.nan, 'a'], categories=['b', 'a'], ordered=True))
        result = getattr(cat, function)(skipna=skipna)
        if skipna is True:
            expected = 'b' if function == 'min' else 'a'
            assert result == expected
        else:
            assert result is np.nan

class TestSeriesMode:
    def test_mode_empty(self, dropna: bool) -> None:
        s = Series([], dtype=np.float64)
        result = s.mode(dropna)
        tm.assert_series_equal(result, s)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    @pytest.mark.parametrize('data: list', [[1, 1, 1, 2], [1, 1, 1, 2, 3, 3, 3]])
    def test_mode_numerical(self, dropna: bool, data: list, any_real_numpy_dtype: np.dtype) -> None:
        s = Series(data, dtype=any_real_numpy_dtype)
        result = s.mode(dropna)
        expected = Series(data, dtype=any_real_numpy_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    def test_mode_numerical_nan(self, dropna: bool) -> None:
        s = Series([1, 1, 2, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series([1.0])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    def test_mode_object(self, dropna: bool) -> None:
        data = ['a'] * 2 + ['b'] * 3
        s = Series(data, dtype='c')
        result = s.mode(dropna)
        expected = Series(['b'], dtype='c')
        tm.assert_series_equal(result, expected)
        data = ['foo', 'bar', 'bar', np.nan, np.nan, np.nan]
        s = Series(data, dtype=object)
        result = s.mode(dropna)
        expected = Series(['bar'], dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    def test_mode_string(self, dropna: bool, any_string_dtype: pd.StringDtype) -> None:
        data = ['a'] * 2 + ['b'] * 3
        s = Series(data, dtype=any_string_dtype)
        result = s.mode(dropna)
        expected = Series(['b'], dtype=any_string_dtype)
        tm.assert_series_equal(result, expected)
        data = ['foo', 'bar', 'bar', np.nan, np.nan, np.nan]
        s = Series(data, dtype=any_string_dtype)
        result = s.mode(dropna)
        expected = Series(['bar'], dtype=any_string_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    def test_mode_mixeddtype(self, dropna: bool) -> None:
        s = Series([1, 'foo', 'foo'])
        result = s.mode(dropna)
        expected = Series(['foo'], dtype=object)
        tm.assert_series_equal(result, expected)
        s = Series([1, 'foo', 'foo', np.nan, np.nan, np.nan])
        result = s.mode(dropna)
        expected = Series(['foo'], dtype=object)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    def test_mode_datetime(self, dropna: bool) -> None:
        s = Series(['2011-01-03', '2013-01-02', '1900-05-03', 'nan', 'nan'], dtype='M8[ns]')
        result = s.mode(dropna)
        expected = Series(['2011-01-03'], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)
        s = Series(['2011-01-03', '2013-01-02', '1900-05-03', '2011-01-03', '2013-01-02', 'nan', 'nan'], dtype='M8[ns]')
        result = s.mode(dropna)
        expected = Series(['2011-01-03'], dtype='M8[ns]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    def test_mode_timedelta(self, dropna: bool) -> None:
        s = Series(['1 days', '-1 days', '0 days', 'nan', 'nan'], dtype='timedelta64[ns]')
        result = s.mode(dropna)
        expected = Series(['0 days'], dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected)
        s = Series(['1 day', '1 day', '-1 day', '-1 day 2 min', '2 min', '2 min', 'nan', 'nan'], dtype='timedelta64[ns]')
        result = s.mode(dropna)
        expected = Series(['2 min'], dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    def test_mode_category(self, dropna: bool) -> None:
        s = Series(Categorical([1, 2, np.nan, np.nan]))
        result = s.mode(dropna)
        expected = Series(Categorical([1], categories=[1, 2], ordered=True), dtype='category')
        tm.assert_series_equal(result, expected)
        s = Series(Categorical([1, 'a', 'a', np.nan, np.nan]))
        result = s.mode(dropna)
        expected = Series(Categorical(['a'], categories=['a'], ordered=True), dtype='category')
        tm.assert_series_equal(result, expected)
        s = Series(Categorical([1, 1, 2, 3, 3, np.nan, np.nan], categories=[3, 2, 1], ordered=True))
        result = s.mode(dropna)
        expected = Series(Categorical([3, 1], categories=[3, 2, 1], ordered=True), dtype='category')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('dropna: bool', [True, False])
    def test_mode_intoverflow(self, dropna: bool) -> None:
        s = Series([1, 2 ** 63, 2 ** 63], dtype=np.uint64)
        result = s.mode(dropna)
        expected = Series([2 ** 63], dtype=np.uint64)
        tm.assert_series_equal(result, expected)
        s = Series([1, 2 ** 63], dtype=np.uint64)
        result = s.mode(dropna)
        expected = Series([2 ** 63], dtype=np.uint64)
        tm.assert_series_equal(result, expected)

    def test_mode_sort_with_na(self) -> None:
        s = Series([1, 'foo', 'foo', np.nan, np.nan])
        expected = Series(['foo', np.nan], dtype=object)
        result = s.mode(dropna=False)
        tm.assert_series_equal(result, expected)

    def test_mode_boolean_with_na(self) -> None:
        ser = Series([True, False, True, pd.NA], dtype='boolean')
        result = ser.mode()
        expected = Series([True], dtype='boolean')
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('array: list', [[0, 1j, 1, 1, 1 + 1j, 1 + 2j], [0, 1j, 1, 1, 1 + 1j, 1 + 2j]])
    @pytest.mark.parametrize('dtype: str', ['complex128', 'complex64'])
    def test_single_mode_value_complex(self, array: list, dtype: str) -> None:
        result = Series(array, dtype=dtype).mode()
        expected = Series([1], dtype=dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('array: list', [[0, 1j, 1, 1, 1 + 1j, 1 + 2j], [1 + 1j, 2j, 1 + 1j]])
    @pytest.mark.parametrize('dtype: str', ['complex128', 'complex64'])
    def test_multimode_complex(self, array: list, dtype: str) -> None:
        result = Series(array, dtype=dtype).mode()
        expected = Series([1 + 1j], dtype=dtype)
        tm.assert_series_equal(result, expected)
