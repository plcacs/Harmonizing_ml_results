from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas.compat import IS64, is_platform_windows
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Categorical, CategoricalDtype, DataFrame, DatetimeIndex, Index, PeriodIndex, RangeIndex, Series, Timestamp, date_range, isna, notna, to_datetime, to_timedelta
import pandas._testing as tm
from pandas.core import algorithms, nanops
from typing import Any, Callable, Optional, Dict, List, Sequence

is_windows_np2_or_is32: bool = is_platform_windows() and (not np_version_gt2) or not IS64
is_windows_or_is32: bool = is_platform_windows() or not IS64


def make_skipna_wrapper(
    alternative: Callable[[Any], Any],
    skipna_alternative: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Series], Any]:
    """
    Create a function for calling on an array.

    Parameters
    ----------
    alternative : function
        The function to be called on the array with no NaNs.
        Only used when 'skipna_alternative' is None.
    skipna_alternative : function
        The function to be called on the original array

    Returns
    -------
    function
    """
    if skipna_alternative:

        def skipna_wrapper(x: Series) -> Any:
            return skipna_alternative(x.values)

    else:

        def skipna_wrapper(x: Series) -> Any:
            nona = x.dropna()
            if len(nona) == 0:
                return np.nan
            return alternative(nona)

    return skipna_wrapper


def assert_stat_op_calc(
    opname: str,
    alternative: Callable[[Any], Any],
    frame: DataFrame,
    has_skipna: bool = True,
    check_dtype: bool = True,
    check_dates: bool = False,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    skipna_alternative: Optional[Callable[[Any], Any]] = None,
) -> None:
    """
    Check that operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    check_dtype : bool, default True
        Whether the dtypes of the result of "frame.opname()" and
        "alternative(frame)" should be checked.
    check_dates : bool, default false
        Whether opname should be tested on a Datetime Series
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    skipna_alternative : function, default None
        NaN-safe version of alternative
    """
    f = getattr(frame, opname)
    if check_dates:
        df = DataFrame({'b': date_range('1/1/2001', periods=2)})
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)
        df['a'] = range(len(df))
        with tm.assert_produces_warning(None):
            result = getattr(df, opname)()
        assert isinstance(result, Series)
        assert len(result)
    if has_skipna:

        def wrapper(x: Series) -> Any:
            return alternative(x.values)

        skipna_wrapper = make_skipna_wrapper(alternative, skipna_alternative)
        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)
        tm.assert_series_equal(result0, frame.apply(wrapper), check_dtype=check_dtype, rtol=rtol, atol=atol)
        tm.assert_series_equal(result1, frame.apply(wrapper, axis=1), rtol=rtol, atol=atol)
    else:
        skipna_wrapper = alternative
    result0 = f(axis=0)
    result1 = f(axis=1)
    tm.assert_series_equal(result0, frame.apply(skipna_wrapper), check_dtype=check_dtype, rtol=rtol, atol=atol)
    if opname in ['sum', 'prod']:
        expected = frame.apply(skipna_wrapper, axis=1)
        tm.assert_series_equal(result1, expected, check_dtype=False, rtol=rtol, atol=atol)
    if check_dtype:
        lcd_dtype = frame.values.dtype
        assert lcd_dtype == result0.dtype
        assert lcd_dtype == result1.dtype
    with pytest.raises(ValueError, match='No axis named 2'):
        f(axis=2)
    if has_skipna:
        all_na = frame * np.nan
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname in ['sum', 'prod']:
            unit = 1 if opname == 'prod' else 0
            expected = Series(unit, index=r0.index, dtype=r0.dtype)
            tm.assert_series_equal(r0, expected)
            expected = Series(unit, index=r1.index, dtype=r1.dtype)
            tm.assert_series_equal(r1, expected)


@pytest.fixture
def bool_frame_with_na() -> DataFrame:
    """
    Fixture for DataFrame of booleans with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(
        np.concatenate([np.ones((15, 4), dtype=bool), np.zeros((15, 4), dtype=bool)], axis=0),
        index=Index([f'foo_{i}' for i in range(30)], dtype=object),
        columns=Index(list('ABCD'), dtype=object),
        dtype=object,
    )
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


@pytest.fixture
def float_frame_with_na() -> DataFrame:
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f'foo_{i}' for i in range(30)], dtype=object),
        columns=Index(list('ABCD'), dtype=object),
    )
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df


class TestDataFrameAnalytics:
    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize(
        'opname',
        [
            'count',
            'sum',
            'mean',
            'product',
            'median',
            'min',
            'max',
            'nunique',
            'var',
            'std',
            'sem',
            pytest.param('skew', marks=td.skip_if_no('scipy')),
            pytest.param('kurt', marks=td.skip_if_no('scipy')),
        ],
    )
    def test_stat_op_api_float_string_frame(self, float_string_frame: DataFrame, axis: int, opname: str) -> None:
        if opname in ('sum', 'min', 'max') and axis == 0 or opname in ('count', 'nunique'):
            getattr(float_string_frame, opname)(axis=axis)
        else:
            if opname in ['var', 'std', 'sem', 'skew', 'kurt']:
                msg: Any = "could not convert string to float: 'bar'"
            elif opname == 'product':
                if axis == 1:
                    msg = "can't multiply sequence by non-int of type 'float'"
                else:
                    msg = "can't multiply sequence by non-int of type 'str'"
            elif opname == 'sum':
                msg = "unsupported operand type\\(s\\) for \\+: 'float' and 'str'"
            elif opname == 'mean':
                if axis == 0:
                    msg = '|'.join(["Could not convert \\['.*'\\] to numeric", "Could not convert string '(bar){30}' to numeric"])
                else:
                    msg = "unsupported operand type\\(s\\) for \\+: 'float' and 'str'"
            elif opname in ['min', 'max']:
                msg = "'[><]=' not supported between instances of 'float' and 'str'"
            elif opname == 'median':
                msg = re.compile('Cannot convert \\[.*\\] to numeric|does not support|Cannot perform', flags=re.S)
            if not isinstance(msg, re.Pattern):
                msg = msg + '|does not support|Cannot perform reduction'
            with pytest.raises(TypeError, match=msg):
                getattr(float_string_frame, opname)(axis=axis)
        if opname != 'nunique':
            getattr(float_string_frame, opname)(axis=axis, numeric_only=True)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize(
        'opname',
        [
            'count',
            'sum',
            'mean',
            'product',
            'median',
            'min',
            'max',
            'var',
            'std',
            'sem',
            pytest.param('skew', marks=td.skip_if_no('scipy')),
            pytest.param('kurt', marks=td.skip_if_no('scipy')),
        ],
    )
    def test_stat_op_api_float_frame(self, float_frame: DataFrame, axis: int, opname: str) -> None:
        getattr(float_frame, opname)(axis=axis, numeric_only=False)

    def test_stat_op_calc(self, float_frame_with_na: DataFrame, mixed_float_frame: DataFrame) -> None:
        def count(s: Series) -> int:
            return int(notna(s).sum())

        def nunique(s: Series) -> int:
            return len(algorithms.unique1d(s.dropna()))

        def var(x: Any) -> float:
            return float(np.var(x, ddof=1))

        def std(x: Any) -> float:
            return float(np.std(x, ddof=1))

        def sem(x: Any) -> float:
            return float(np.std(x, ddof=1) / np.sqrt(len(x)))

        assert_stat_op_calc('nunique', nunique, float_frame_with_na, has_skipna=False, check_dtype=False, check_dates=True)
        assert_stat_op_calc('sum', np.sum, mixed_float_frame.astype('float32'), check_dtype=False, rtol=0.001)
        assert_stat_op_calc('sum', np.sum, float_frame_with_na, skipna_alternative=np.nansum)
        assert_stat_op_calc('mean', np.mean, float_frame_with_na, check_dates=True)
        assert_stat_op_calc('product', np.prod, float_frame_with_na, skipna_alternative=np.nanprod)
        assert_stat_op_calc('var', var, float_frame_with_na)
        assert_stat_op_calc('std', std, float_frame_with_na)
        assert_stat_op_calc('sem', sem, float_frame_with_na)
        assert_stat_op_calc('count', count, float_frame_with_na, has_skipna=False, check_dtype=False, check_dates=True)

    def test_stat_op_calc_skew_kurtosis(self, float_frame_with_na: DataFrame) -> None:
        sp_stats = pytest.importorskip('scipy.stats')

        def skewness(x: Any) -> float:
            if len(x) < 3:
                return np.nan
            return float(sp_stats.skew(x, bias=False))

        def kurt(x: Any) -> float:
            if len(x) < 4:
                return np.nan
            return float(sp_stats.kurtosis(x, bias=False))

        assert_stat_op_calc('skew', skewness, float_frame_with_na)
        assert_stat_op_calc('kurt', kurt, float_frame_with_na)

    def test_median(self, float_frame_with_na: DataFrame, int_frame: DataFrame) -> None:
        def wrapper(x: Any) -> float:
            if isna(x).any():
                return np.nan
            return float(np.median(x))

        assert_stat_op_calc('median', wrapper, float_frame_with_na, check_dates=True)
        assert_stat_op_calc('median', wrapper, int_frame, check_dtype=False, check_dates=True)

    @pytest.mark.parametrize('method', ['sum', 'mean', 'prod', 'var', 'std', 'skew', 'min', 'max'])
    @pytest.mark.parametrize(
        'df',
        [
            DataFrame(
                {
                    'a': [-0.0004998754019959134, -0.001646725777291983, 0.0006769587077588301],
                    'b': [-0, -0, 0.0],
                    'c': [0.00031111847529610595, 0.0014902627951905339, -0.0009409920003597969],
                },
                index=['foo', 'bar', 'baz'],
                dtype='O',
            ),
            DataFrame({0: [np.nan, 2], 1: [np.nan, 3], 2: [np.nan, 4]}, dtype=object),
        ],
    )
    @pytest.mark.filterwarnings('ignore:Mismatched null-like values:FutureWarning')
    def test_stat_operators_attempt_obj_array(self, method: str, df: DataFrame, axis: int | str) -> None:
        assert df.values.dtype == np.object_
        result = getattr(df, method)(axis=axis)
        expected = getattr(df.astype('f8'), method)(axis=axis).astype(object)
        if axis in [1, 'columns'] and method in ['min', 'max']:
            expected[expected.isna()] = None
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('op', ['mean', 'std', 'var', 'skew', 'kurt', 'sem'])
    def test_mixed_ops(self, op: str) -> None:
        df = DataFrame({'int': [1, 2, 3, 4], 'float': [1.0, 2.0, 3.0, 4.0], 'str': ['a', 'b', 'c', 'd']})
        msg = '|'.join(['Could not convert', 'could not convert', "can't multiply sequence by non-int", 'does not support', 'Cannot perform'])
        with pytest.raises(TypeError, match=msg):
            getattr(df, op)()
        with pd.option_context('use_bottleneck', False):
            with pytest.raises(TypeError, match=msg):
                getattr(df, op)()

    def test_reduce_mixed_frame(self) -> None:
        df = DataFrame({'bool_data': [True, True, False, False, False], 'int_data': [10, 20, 30, 40, 50], 'string_data': ['a', 'b', 'c', 'd', 'e']})
        df.reindex(columns=['bool_data', 'int_data', 'string_data'])
        test = df.sum(axis=0)
        tm.assert_numpy_array_equal(test.values, np.array([2, 150, 'abcde'], dtype=object))
        alt = df.T.sum(axis=1)
        tm.assert_series_equal(test, alt)

    def test_nunique(self) -> None:
        df = DataFrame({'A': [1, 1, 1], 'B': [1, 2, 3], 'C': [1, np.nan, 3]})
        tm.assert_series_equal(df.nunique(), Series({'A': 1, 'B': 3, 'C': 2}))
        tm.assert_series_equal(df.nunique(dropna=False), Series({'A': 1, 'B': 3, 'C': 3}))
        tm.assert_series_equal(df.nunique(axis=1), Series([1, 2, 2]))
        tm.assert_series_equal(df.nunique(axis=1, dropna=False), Series([1, 3, 2]))

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_mean_mixed_datetime_numeric(self, tz: Optional[str]) -> None:
        df = DataFrame({'A': [1, 1], 'B': [Timestamp('2000', tz=tz)] * 2})
        result = df.mean()
        expected = Series([1.0, Timestamp('2000', tz=tz)], index=['A', 'B'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('tz', [None, 'UTC'])
    def test_mean_includes_datetimes(self, tz: Optional[str]) -> None:
        df = DataFrame({'A': [Timestamp('2000', tz=tz)] * 2})
        result = df.mean()
        expected = Series([Timestamp('2000', tz=tz)], index=['A'])
        tm.assert_series_equal(result, expected)

    def test_mean_mixed_string_decimal(self) -> None:
        d = [
            {'A': 2, 'B': None, 'C': Decimal('628.00')},
            {'A': 1, 'B': None, 'C': Decimal('383.00')},
            {'A': 3, 'B': None, 'C': Decimal('651.00')},
            {'A': 2, 'B': None, 'C': Decimal('575.00')},
            {'A': 4, 'B': None, 'C': Decimal('1114.00')},
            {'A': 1, 'B': 'TEST', 'C': Decimal('241.00')},
            {'A': 2, 'B': None, 'C': Decimal('572.00')},
            {'A': 4, 'B': None, 'C': Decimal('609.00')},
            {'A': 3, 'B': None, 'C': Decimal('820.00')},
            {'A': 5, 'B': None, 'C': Decimal('1223.00')},
        ]
        df = DataFrame(d)
        with pytest.raises(TypeError, match='unsupported operand type|does not support|Cannot perform'):
            df.mean()
        result = df[['A', 'C']].mean()
        expected = Series([2.7, 681.6], index=['A', 'C'], dtype=object)
        tm.assert_series_equal(result, expected)

    def test_var_std(self, datetime_frame: DataFrame) -> None:
        result = datetime_frame.std(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4))
        tm.assert_almost_equal(result, expected)
        result = datetime_frame.var(ddof=4)
        expected = datetime_frame.apply(lambda x: x.var(ddof=4))
        tm.assert_almost_equal(result, expected)
        arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
        result = nanops.nanvar(arr, axis=0)
        assert not (result < 0).any()
        with pd.option_context('use_bottleneck', False):
            result = nanops.nanvar(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize('meth', ['sem', 'var', 'std'])
    def test_numeric_only_flag(self, meth: str) -> None:
        df1 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['foo', 'bar', 'baz'])
        df1 = df1.astype({'foo': object})
        df1.loc[0, 'foo'] = '100'
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['foo', 'bar', 'baz'])
        df2 = df2.astype({'foo': object})
        df2.loc[0, 'foo'] = 'a'
        result = getattr(df1, meth)(axis=1, numeric_only=True)
        expected = getattr(df1[['bar', 'baz']], meth)(axis=1)
        tm.assert_series_equal(expected, result)
        result = getattr(df2, meth)(axis=1, numeric_only=True)
        expected = getattr(df2[['bar', 'baz']], meth)(axis=1)
        tm.assert_series_equal(expected, result)
        msg = "unsupported operand type\\(s\\) for -: 'float' and 'str'"
        with pytest.raises(TypeError, match=msg):
            getattr(df1, meth)(axis=1, numeric_only=False)
        msg = "could not convert string to float: 'a'"
        with pytest.raises(TypeError, match=msg):
            getattr(df2, meth)(axis=1, numeric_only=False)

    def test_sem(self, datetime_frame: DataFrame) -> None:
        result = datetime_frame.sem(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4) / np.sqrt(len(x)))
        tm.assert_almost_equal(result, expected)
        arr = np.repeat(np.random.default_rng(2).random((1, 1000)), 1000, 0)
        result = nanops.nansem(arr, axis=0)
        assert not (result < 0).any()
        with pd.option_context('use_bottleneck', False):
            result = nanops.nansem(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize(
        'dropna, expected',
        [
            (
                True,
                {
                    'A': [12],
                    'B': [10.0],
                    'C': [1.0],
                    'D': ['a'],
                    'E': Categorical(['a'], categories=['a']),
                    'F': DatetimeIndex(['2000-01-02'], dtype='M8[ns]'),
                    'G': to_timedelta(['1 days']),
                },
            ),
            (
                False,
                {
                    'A': [12],
                    'B': [10.0],
                    'C': [np.nan],
                    'D': Series([np.nan], dtype='str'),
                    'E': Categorical([np.nan], categories=['a']),
                    'F': DatetimeIndex([pd.NaT], dtype='M8[ns]'),
                    'G': to_timedelta([pd.NaT]),
                },
            ),
            (
                True,
                {
                    'H': [8, 9, np.nan, np.nan],
                    'I': [8, 9, np.nan, np.nan],
                    'J': [1, np.nan, np.nan, np.nan],
                    'K': Categorical(['a', np.nan, np.nan, np.nan], categories=['a']),
                    'L': DatetimeIndex(['2000-01-02', 'NaT', 'NaT', 'NaT'], dtype='M8[ns]'),
                    'M': to_timedelta(['1 days', 'nan', 'nan', 'nan']),
                    'N': [0, 1, 2, 3],
                },
            ),
            (
                False,
                {
                    'H': [8, 9, np.nan, np.nan],
                    'I': [8, 9, np.nan, np.nan],
                    'J': [1, np.nan, np.nan, np.nan],
                    'K': Categorical([np.nan, 'a', np.nan, np.nan], categories=['a']),
                    'L': DatetimeIndex(['NaT', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'),
                    'M': to_timedelta(['nan', '1 days', 'nan', 'nan']),
                    'N': [0, 1, 2, 3],
                },
            ),
        ],
    )
    def test_mode_dropna(self, dropna: bool, expected: Dict[str, Any]) -> None:
        df = DataFrame(
            {
                'A': [12, 12, 19, 11],
                'B': [10, 10, np.nan, 3],
                'C': [1, np.nan, np.nan, np.nan],
                'D': Series([np.nan, np.nan, 'a', np.nan], dtype='str'),
                'E': Categorical([np.nan, np.nan, 'a', np.nan]),
                'F': DatetimeIndex(['NaT', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'),
                'G': to_timedelta(['1 days', 'nan', 'nan', 'nan']),
                'H': [8, 8, 9, 9],
                'I': [9, 9, 8, 8],
                'J': [1, 1, np.nan, np.nan],
                'K': Categorical(['a', np.nan, 'a', np.nan]),
                'L': DatetimeIndex(['2000-01-02', '2000-01-02', 'NaT', 'NaT'], dtype='M8[ns]'),
                'M': to_timedelta(['1 days', 'nan', '1 days', 'nan']),
                'N': np.arange(4, dtype='int64'),
            }
        )
        result = df[sorted(expected.keys())].mode(dropna=dropna)
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    def test_mode_sort_with_na(self, using_infer_string: bool) -> None:
        df = DataFrame({'A': [np.nan, np.nan, 'a', 'a']})
        expected = DataFrame({'A': ['a', np.nan]})
        result = df.mode(dropna=False)
        tm.assert_frame_equal(result, expected)

    def test_mode_empty_df(self) -> None:
        df = DataFrame([], columns=['a', 'b'])
        expected = df.copy()
        result = df.mode()
        tm.assert_frame_equal(result, expected)

    def test_operators_timedelta64(self) -> None:
        df = DataFrame(
            {
                'A': date_range('2012-1-1', periods=3, freq='D'),
                'B': date_range('2012-1-2', periods=3, freq='D'),
                'C': Timestamp('20120101') - timedelta(minutes=5, seconds=5),
            }
        )
        diffs = DataFrame({'A': df['A'] - df['C'], 'B': df['A'] - df['B']})
        result = diffs.min()
        assert result.iloc[0] == diffs.loc[0, 'A']
        assert result.iloc[1] == diffs.loc[0, 'B']
        result = diffs.min(axis=1)
        assert (result == diffs.loc[0, 'B']).all()
        result = diffs.max()
        assert result.iloc[0] == diffs.loc[2, 'A']
        assert result.iloc[1] == diffs.loc[2, 'B']
        result = diffs.max(axis=1)
        assert (result == diffs['A']).all()
        result = diffs.abs()
        result2 = abs(diffs)
        expected = DataFrame({'A': df['A'] - df['C'], 'B': df['B'] - df['A']})
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)
        mixed = diffs.copy()
        mixed['C'] = 'foo'
        mixed['D'] = 1
        mixed['E'] = 1.0
        mixed['F'] = Timestamp('20130101')
        result = mixed.min()
        expected = Series(
            [pd.Timedelta(timedelta(seconds=5 * 60 + 5)), pd.Timedelta(timedelta(days=-1)), 'foo', 1, 1.0, Timestamp('20130101')],
            index=mixed.columns,
        )
        tm.assert_series_equal(result, expected)
        result = mixed.min(axis=1, numeric_only=True)
        expected = Series([1, 1, 1.0])
        tm.assert_series_equal(result, expected)
        result = mixed[['A', 'B']].min(axis=1)
        expected = Series([timedelta(days=-1)] * 3)
        tm.assert_series_equal(result, expected)
        result = mixed[['A', 'B']].min()
        expected = Series([timedelta(seconds=5 * 60 + 5), timedelta(days=-1)], index=['A', 'B'])
        tm.assert_series_equal(result, expected)
        df = DataFrame({'time': date_range('20130102', periods=5), 'time2': date_range('20130105', periods=5)})
        df['off1'] = df['time2'] - df['time']
        assert df['off1'].dtype == 'timedelta64[ns]'
        df['off2'] = df['time'] - df['time2']
        df._consolidate_inplace()
        assert df['off1'].dtype == 'timedelta64[ns]'
        assert df['off2'].dtype == 'timedelta64[ns]'

    def test_std_timedelta64_skipna_false(self) -> None:
        tdi = pd.timedelta_range('1 Day', periods=10)
        df = DataFrame({'A': tdi, 'B': tdi}, copy=True)
        df.iloc[-2, -1] = pd.NaT
        result = df.std(skipna=False)
        expected = Series([df['A'].std(), pd.NaT], index=['A', 'B'], dtype='timedelta64[ns]')
        tm.assert_series_equal(result, expected)
        result = df.std(axis=1, skipna=False)
        expected = Series([pd.Timedelta(0)] * 8 + [pd.NaT, pd.Timedelta(0)])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('values', [['2022-01-01', '2022-01-02', pd.NaT, '2022-01-03'], 4 * [pd.NaT]])
    def test_std_datetime64_with_nat(self, values: Sequence[object], skipna: bool, request: pytest.FixtureRequest, unit: str) -> None:
        dti = to_datetime(values).as_unit(unit)
        df = DataFrame({'a': dti})
        result = df.std(skipna=skipna)
        if not skipna or all((value is pd.NaT for value in values)):
            expected = Series({'a': pd.NaT}, dtype=f'timedelta64[{unit}]')
        else:
            expected = Series({'a': 86400000000000}, dtype=f'timedelta64[{unit}]')
        tm.assert_series_equal(result, expected)

    def test_sum_corner(self) -> None:
        empty_frame = DataFrame()
        axis0 = empty_frame.sum(axis=0)
        axis1 = empty_frame.sum(axis=1)
        assert isinstance(axis0, Series)
        assert isinstance(axis1, Series)
        assert len(axis0) == 0
        assert len(axis1) == 0

    @pytest.mark.parametrize('index', [RangeIndex(0), DatetimeIndex([]), Index([], dtype=np.int64), Index([], dtype=np.float64), DatetimeIndex([], freq='ME'), PeriodIndex([], freq='D')])
    def test_axis_1_empty(self, all_reductions: str, index: Index) -> None:
        df = DataFrame(columns=['a'], index=index)
        result = getattr(df, all_reductions)(axis=1)
        if all_reductions in ('any', 'all'):
            expected_dtype = 'bool'
        elif all_reductions == 'count':
            expected_dtype = 'int64'
        else:
            expected_dtype = 'object'
        expected = Series([], index=index, dtype=expected_dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('method, unit', [('sum', 0), ('prod', 1)])
    @pytest.mark.parametrize('numeric_only', [None, True, False])
    def test_sum_prod_nanops(self, method: str, unit: int, numeric_only: Optional[bool]) -> None:
        idx = ['a', 'b', 'c']
        df = DataFrame({'a': [unit, unit], 'b': [unit, np.nan], 'c': [np.nan, np.nan]})
        result = getattr(df, method)(numeric_only=numeric_only)
        expected = Series([unit, unit, unit], index=idx, dtype='float64')
        tm.assert_series_equal(result, expected)
        result = getattr(df, method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, unit, np.nan], index=idx)
        tm.assert_series_equal(result, expected)
        result = getattr(df, method)(numeric_only=numeric_only, min_count=0)
        expected = Series([unit, unit, unit], index=idx, dtype='float64')
        tm.assert_series_equal(result, expected)
        result = getattr(df.iloc[1:], method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, np.nan, np.nan], index=idx)
        tm.assert_series_equal(result, expected)
        df = DataFrame({'A': [unit] * 10, 'B': [unit] * 5 + [np.nan] * 5})
        result = getattr(df, method)(numeric_only=numeric_only, min_count=5)
        expected = Series(result, index=['A', 'B'])
        tm.assert_series_equal(result, expected)
        result = getattr(df, method)(numeric_only=numeric_only, min_count=6)
        expected = Series(result, index=['A', 'B'])
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_timedelta(self) -> None:
        idx = ['a', 'b', 'c']
        df = DataFrame({'a': [0, 0], 'b': [0, np.nan], 'c': [np.nan, np.nan]})
        df2 = df.apply(to_timedelta)
        result = df2.sum()
        expected = Series([0, 0, 0], dtype='m8[ns]', index=idx)
        tm.assert_series_equal(result, expected)
        result = df2.sum(min_count=0)
        tm.assert_series_equal(result, expected)
        result = df2.sum(min_count=1)
        expected = Series([0, 0, np.nan], dtype='m8[ns]', index=idx)
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_min_count(self) -> None:
        df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        result = df.sum(min_count=10)
        expected = Series([np.nan, np.nan], index=['x', 'y'])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('float_type', ['float16', 'float32', 'float64'])
    @pytest.mark.parametrize('kwargs, expected_result', [({'axis': 1, 'min_count': 2}, [3.2, 5.3, np.nan]), ({'axis': 1, 'min_count': 3}, [np.nan, np.nan, np.nan]), ({'axis': 1, 'skipna': False}, [3.2, 5.3, np.nan])])
    def test_sum_nanops_dtype_min_count(self, float_type: str, kwargs: Dict[str, Any], expected_result: List[float]) -> None:
        df = DataFrame({'a': [1.0, 2.3, 4.4], 'b': [2.2, 3, np.nan]}, dtype=float_type)
        result = df.sum(**kwargs)
        expected = Series(expected_result).astype(float_type)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('float_type', ['float16', 'float32', 'float64'])
    @pytest.mark.parametrize('kwargs, expected_result', [({'axis': 1, 'min_count': 2}, [2.0, 4.0, np.nan]), ({'axis': 1, 'min_count': 3}, [np.nan, np.nan, np.nan]), ({'axis': 1, 'skipna': False}, [2.0, 4.0, np.nan])])
    def test_prod_nanops_dtype_min_count(self, float_type: str, kwargs: Dict[str, Any], expected_result: List[float]) -> None:
        df = DataFrame({'a': [1.0, 2.0, 4.4], 'b': [2.0, 2.0, np.nan]}, dtype=float_type)
        result = df.prod(**kwargs)
        expected = Series(expected_result).astype(float_type)
        tm.assert_series_equal(result, expected)

    def test_sum_object(self, float_frame: DataFrame) -> None:
        values = float_frame.values.astype(int)
        frame = DataFrame(values, index=float_frame.index, columns=float_frame.columns)
        deltas = frame * timedelta(1)
        deltas.sum()

    def test_sum_bool(self, float_frame: DataFrame) -> None:
        bools = np.isnan(float_frame)
        bools.sum(axis=1)
        bools.sum(axis=0)

    def test_sum_mixed_datetime(self) -> None:
        df = DataFrame({'A': date_range('2000', periods=4), 'B': [1, 2, 3, 4]}).reindex([2, 3, 4])
        with pytest.raises(TypeError, match="does not support operation 'sum'"):
            df.sum()

    def test_mean_corner(self, float_frame: DataFrame, float_string_frame: DataFrame) -> None:
        msg = 'Could not convert|does not support|Cannot perform'
        with pytest.raises(TypeError, match=msg):
            float_string_frame.mean(axis=0)
        with pytest.raises(TypeError, match='unsupported operand type'):
            float_string_frame.mean(axis=1)
        float_frame['bool'] = float_frame['A'] > 0
        means = float_frame.mean(axis=0)
        assert means['bool'] == float_frame['bool'].values.mean()

    def test_mean_datetimelike(self) -> None:
        df = DataFrame({'A': np.arange(3), 'B': date_range('2016-01-01', periods=3), 'C': pd.timedelta_range('1D', periods=3), 'D': pd.period_range('2016', periods=3, freq='Y')})
        result = df.mean(numeric_only=True)
        expected = Series({'A': 1.0})
        tm.assert_series_equal(result, expected)
        with pytest.raises(TypeError, match='mean is not implemented for PeriodArray'):
            df.mean()

    def test_mean_datetimelike_numeric_only_false(self) -> None:
        df = DataFrame({'A': np.arange(3), 'B': date_range('2016-01-01', periods=3), 'C': pd.timedelta_range('1D', periods=3)})
        result = df.mean(numeric_only=False)
        expected = Series({'A': 1, 'B': df.loc[1, 'B'], 'C': df.loc[1, 'C']})
        tm.assert_series_equal(result, expected)
        df['D'] = pd.period_range('2016', periods=3, freq='Y')
        with pytest.raises(TypeError, match='mean is not implemented for Period'):
            df.mean(numeric_only=False)

    def test_mean_extensionarray_numeric_only_true(self) -> None:
        arr = np.random.default_rng(2).integers(1000, size=(10, 5))
        df = DataFrame(arr, dtype='Int64')
        result = df.mean(numeric_only=True)
        expected = DataFrame(arr).mean().astype('Float64')
        tm.assert_series_equal(result, expected)

    def test_stats_mixed_type(self, float_string_frame: DataFrame) -> None:
        with pytest.raises(TypeError, match='could not convert'):
            float_string_frame.std(axis=1)
        with pytest.raises(TypeError, match='could not convert'):
            float_string_frame.var(axis=1)
        with pytest.raises(TypeError, match='unsupported operand type'):
            float_string_frame.mean(axis=1)
        with pytest.raises(TypeError, match='could not convert'):
            float_string_frame.skew(axis=1)

    def test_sum_bools(self) -> None:
        df = DataFrame(index=range(1), columns=range(10))
        bools = isna(df)
        assert bools.sum(axis=1)[0] == 10

    @pytest.mark.parametrize('axis', [0, 1])
    def test_idxmin(self, float_frame: DataFrame, int_frame: DataFrame, skipna: bool, axis: int) -> None:
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for df in [frame, int_frame]:
            if (not skipna or axis == 1) and df is not int_frame:
                if skipna:
                    msg = 'Encountered all NA values'
                else:
                    msg = 'Encountered an NA value'
                with pytest.raises(ValueError, match=msg):
                    df.idxmin(axis=axis, skipna=skipna)
                with pytest.raises(ValueError, match=msg):
                    df.idxmin(axis=axis, skipna=skipna)
            else:
                result = df.idxmin(axis=axis, skipna=skipna)
                expected = df.apply(Series.idxmin, axis=axis, skipna=skipna)
                expected = expected.astype(df.index.dtype)
                tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_idxmin_empty(self, index: Index, skipna: bool, axis: int) -> None:
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)
        result = frame.idxmin(axis=axis, skipna=skipna)
        expected = Series(dtype=index.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('numeric_only', [True, False])
    def test_idxmin_numeric_only(self, numeric_only: bool) -> None:
        df = DataFrame({'a': [2, 3, 1], 'b': [2, 1, 1], 'c': list('xyx')})
        result = df.idxmin(numeric_only=numeric_only)
        if numeric_only:
            expected = Series([2, 1], index=['a', 'b'])
        else:
            expected = Series([2, 1, 0], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)

    def test_idxmin_axis_2(self, float_frame: DataFrame) -> None:
        frame = float_frame
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            frame.idxmin(axis=2)

    @pytest.mark.parametrize('axis', [0, 1])
    def test_idxmax(self, float_frame: DataFrame, int_frame: DataFrame, skipna: bool, axis: int) -> None:
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for df in [frame, int_frame]:
            if (skipna is False or axis == 1) and df is frame:
                if skipna:
                    msg = 'Encountered all NA values'
                else:
                    msg = 'Encountered an NA value'
                with pytest.raises(ValueError, match=msg):
                    df.idxmax(axis=axis, skipna=skipna)
                return
            result = df.idxmax(axis=axis, skipna=skipna)
            expected = df.apply(Series.idxmax, axis=axis, skipna=skipna)
            expected = expected.astype(df.index.dtype)
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_idxmax_empty(self, index: Index, skipna: bool, axis: int) -> None:
        if axis == 0:
            frame = DataFrame(index=index)
        else:
            frame = DataFrame(columns=index)
        result = frame.idxmax(axis=axis, skipna=skipna)
        expected = Series(dtype=index.dtype)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('numeric_only', [True, False])
    def test_idxmax_numeric_only(self, numeric_only: bool) -> None:
        df = DataFrame({'a': [2, 3, 1], 'b': [2, 1, 1], 'c': list('xyx')})
        result = df.idxmax(numeric_only=numeric_only)
        if numeric_only:
            expected = Series([1, 0], index=['a', 'b'])
        else:
            expected = Series([1, 0, 1], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)

    def test_idxmax_arrow_types(self) -> None:
        pytest.importorskip('pyarrow')
        df = DataFrame({'a': [2, 3, 1], 'b': [2, 1, 1]}, dtype='int64[pyarrow]')
        result = df.idxmax()
        expected = Series([1, 0], index=['a', 'b'])
        tm.assert_series_equal(result, expected)
        result = df.idxmin()
        expected = Series([2, 1], index=['a', 'b'])
        tm.assert_series_equal(result, expected)
        df = DataFrame({'a': ['b', 'c', 'a']}, dtype='string[pyarrow]')
        result = df.idxmax(numeric_only=False)
        expected = Series([1], index=['a'])
        tm.assert_series_equal(result, expected)
        result = df.idxmin(numeric_only=False)
        expected = Series([2], index=['a'])
        tm.assert_series_equal(result, expected)

    def test_idxmax_axis_2(self, float_frame: DataFrame) -> None:
        frame = float_frame
        msg = 'No axis named 2 for object type DataFrame'
        with pytest.raises(ValueError, match=msg):
            frame.idxmax(axis=2)

    def test_idxmax_mixed_dtype(self) -> None:
        dti = date_range('2016-01-01', periods=3)
        df = DataFrame({1: [0, 2, 1], 2: range(3)[::-1], 3: dti})
        result = df.idxmax()
        expected = Series([1, 0, 2], index=range(1, 4))
        tm.assert_series_equal(result, expected)
        result = df.idxmin()
        expected = Series([0, 2, 0], index=range(1, 4))
        tm.assert_series_equal(result, expected)
        df.loc[0, 3] = pd.NaT
        result = df.idxmax()
        expected = Series([1, 0, 2], index=range(1, 4))
        tm.assert_series_equal(result, expected)
        result = df.idxmin()
        expected = Series([0, 2, 1], index=range(1, 4))
        tm.assert_series_equal(result, expected)
        df[4] = dti[::-1]
        df._consolidate_inplace()
        result = df.idxmax()
        expected = Series([1, 0, 2, 0], index=range(1, 5))
        tm.assert_series_equal(result, expected)
        result = df.idxmin()
        expected = Series([0, 2, 1, 2], index=range(1, 5))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('op, expected_value', [('idxmax', [0, 4]), ('idxmin', [0, 5])])
    def test_idxmax_idxmin_convert_dtypes(self, op: str, expected_value: List[int]) -> None:
        df = DataFrame({'ID': [100, 100, 100, 200, 200, 200], 'value': [0, 0, 0, 1, 2, 0]}, dtype='Int64')
        df = df.groupby('ID')
        result = getattr(df, op)()
        expected = DataFrame({'value': expected_value}, index=Index([100, 200], name='ID', dtype='Int64'))
        tm.assert_frame_equal(result, expected)

    def test_idxmax_dt64_multicolumn_axis1(self) -> None:
        dti = date_range('2016-01-01', periods=3)
        df = DataFrame({3: dti, 4: dti[::-1]}, copy=True)
        df.iloc[0, 0] = pd.NaT
        df._consolidate_inplace()
        result = df.idxmax(axis=1)
        expected = Series([4, 3, 3])
        tm.assert_series_equal(result, expected)
        result = df.idxmin(axis=1)
        expected = Series([4, 3, 4])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('axis', [0, 1])
    @pytest.mark.parametrize('bool_only', [False, True])
    def test_any_all_mixed_float(self, all_boolean_reductions: str, axis: int, bool_only: bool, float_string_frame: DataFrame) -> None:
        mixed = float_string_frame
        mixed['_bool_'] = np.random.default_rng(2).standard_normal(len(mixed)) > 0.5
        getattr(mixed, all_boolean_reductions)(axis=axis, bool_only=bool_only)

    @pytest.mark.parametrize('axis', [0, 1])
    def test_any_all_bool_with_na(self, all_boolean_reductions: str, axis: int, bool_frame_with_na: DataFrame) -> None:
        getattr(bool_frame_with_na, all_boolean_reductions)(axis=axis, bool_only=False)

    def test_any_all_bool_frame(self, all_boolean_reductions: str, bool_frame_with_na: DataFrame) -> None:
        frame = bool_frame_with_na.fillna(True)
        alternative = getattr(np, all_boolean_reductions)
        f = getattr(frame, all_boolean_reductions)

        def skipna_wrapper(x: Series) -> Any:
            nona = x.dropna().values
            return alternative(nona)

        def wrapper(x: Series) -> Any:
            return alternative(x.values)

        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)
        tm.assert_series_equal(result0, frame.apply(wrapper))
        tm.assert_series_equal(result1, frame.apply(wrapper, axis=1))
        result0 = f(axis=0)
        result1 = f(axis=1)
        tm.assert_series_equal(result0, frame.apply(skipna_wrapper))
        tm.assert_series_equal(result1, frame.apply(skipna_wrapper, axis=1), check_dtype=False)
        with pytest.raises(ValueError, match='No axis named 2'):
            f(axis=2)
        all_na = frame * np.nan
        r0 = getattr(all_na, all_boolean_reductions)(axis=0)
        r1 = getattr(all_na, all_boolean_reductions)(axis=1)
        if all_boolean_reductions == 'any':
            assert not r0.any()
            assert not r1.any()
        else:
            assert r0.all()
            assert r1.all()

    def test_any_all_extra(self) -> None:
        df = DataFrame({'A': [True, False, False], 'B': [True, True, False], 'C': [True, True, True]}, index=['a', 'b', 'c'])
        result = df[['A', 'B']].any(axis=1)
        expected = Series([True, True, False], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)
        result = df[['A', 'B']].any(axis=1, bool_only=True)
        tm.assert_series_equal(result, expected)
        result = df.all(axis=1)
        expected = Series([True, False, False], index=['a', 'b', 'c'])
        tm.assert_series_equal(result, expected)
        result = df.all(axis=1, bool_only=True)
        tm.assert_series_equal(result, expected)
        result = df.all(axis=None).item()
        assert result is False
        result = df.any(axis=None).item()
        assert result is True
        result = df[['C']].all(axis=None).item()
        assert result is True

    @pytest.mark.parametrize('axis', [0, 1])
    def test_any_all_object_dtype(self, axis: int, all_boolean_reductions: str, skipna: bool) -> None:
        df = DataFrame(data=[[1, np.nan, np.nan, True], [np.nan, 2, np.nan, True], [np.nan, np.nan, np.nan, True], [np.nan, np.nan, '5', np.nan]])
        result = getattr(df, all_boolean_reductions)(axis=axis, skipna=skipna)
        expected = Series([True, True, True, True])
        tm.assert_series_equal(result, expected)

    def test_any_datetime(self) -> None:
        float_data = [1, np.nan, 3, np.nan]
        datetime_data = [Timestamp('1960-02-15'), Timestamp('1960-02-16'), pd.NaT, pd.NaT]
        df = DataFrame({'A': float_data, 'B': datetime_data})
        msg = "datetime64 type does not support operation 'any'"
        with pytest.raises(TypeError, match=msg):
            df.any(axis=1)

    def test_any_all_bool_only(self) -> None:
        df = DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [None, None, None]}, columns=Index(['col1', 'col2', 'col3'], dtype=object))
        result = df.all(bool_only=True)
        expected = Series(dtype=np.bool_, index=[])
        tm.assert_series_equal(result, expected)
        df = DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [None, None, None], 'col4': [False, False, True]})
        result = df.all(bool_only=True)
        expected = Series({'col4': False})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'func, data, expected',
        [
            (np.any, {}, False),
            (np.all, {}, True),
            (np.any, {'A': []}, False),
            (np.all, {'A': []}, True),
            (np.any, {'A': [False, False]}, False),
            (np.all, {'A': [False, False]}, False),
            (np.any, {'A': [True, False]}, True),
            (np.all, {'A': [True, False]}, False),
            (np.any, {'A': [True, True]}, True),
            (np.all, {'A': [True, True]}, True),
            (np.any, {'A': [False], 'B': [False]}, False),
            (np.all, {'A': [False], 'B': [False]}, False),
            (np.any, {'A': [False, False], 'B': [False, True]}, True),
            (np.all, {'A': [False, False], 'B': [False, True]}, False),
            (np.all, {'A': Series([0.0, 1.0], dtype='float')}, False),
            (np.any, {'A': Series([0.0, 1.0], dtype='float')}, True),
            (np.all, {'A': Series([0, 1], dtype=int)}, False),
            (np.any, {'A': Series([0, 1], dtype=int)}, True),
            pytest.param(np.all, {'A': Series([0, 1], dtype='M8[ns]')}, False),
            pytest.param(np.all, {'A': Series([0, 1], dtype='M8[ns, UTC]')}, False),
            pytest.param(np.any, {'A': Series([0, 1], dtype='M8[ns]')}, True),
            pytest.param(np.any, {'A': Series([0, 1], dtype='M8[ns, UTC]')}, True),
            pytest.param(np.all, {'A': Series([1, 2], dtype='M8[ns]')}, True),
            pytest.param(np.all, {'A': Series([1, 2], dtype='M8[ns, UTC]')}, True),
            pytest.param(np.any, {'A': Series([1, 2], dtype='M8[ns]')}, True),
            pytest.param(np.any, {'A': Series([1, 2], dtype='M8[ns, UTC]')}, True),
            pytest.param(np.all, {'A': Series([0, 1], dtype='m8[ns]')}, False),
            pytest.param(np.any, {'A': Series([0, 1], dtype='m8[ns]')}, True),
            pytest.param(np.all, {'A': Series([1, 2], dtype='m8[ns]')}, True),
            pytest.param(np.any, {'A': Series([1, 2], dtype='m8[ns]')}, True),
            (np.all, {'A': Series([0, 1], dtype='category')}, True),
            (np.any, {'A': Series([0, 1], dtype='category')}, False),
            (np.all, {'A': Series([1, 2], dtype='category')}, True),
            (np.any, {'A': Series([1, 2], dtype='category')}, False),
            pytest.param(np.all, {'A': Series([10, 20], dtype='M8[ns]'), 'B': Series([10, 20], dtype='m8[ns]')}, True),
        ],
    )
    def test_any_all_np_func(self, func: Callable[..., Any], data: Dict[str, Any], expected: bool) -> None:
        data = DataFrame(data)
        if any((isinstance(x, CategoricalDtype) for x in data.dtypes)):
            with pytest.raises(TypeError, match='.* dtype category does not support operation'):
                func(data)
            with pytest.raises(TypeError, match='.* dtype category does not support operation'):
                getattr(DataFrame(data), func.__name__)(axis=None)
        if data.dtypes.apply(lambda x: x.kind == 'M').any():
            msg = "datetime64 type does not support operation '(any|all)'"
            with pytest.raises(TypeError, match=msg):
                func(data)
            with pytest.raises(TypeError, match=msg):
                getattr(DataFrame(data), func.__name__)(axis=None)
        elif data.dtypes.apply(lambda x: x != 'category').any():
            result = func(data)
            assert isinstance(result, np.bool_)
            assert result.item() is expected
            result = getattr(DataFrame(data), func.__name__)(axis=None)
            assert isinstance(result, np.bool_)
            assert result.item() is expected

    def test_any_all_object(self) -> None:
        result = np.all(DataFrame(columns=['a', 'b'])).item()
        assert result is True
        result = np.any(DataFrame(columns=['a', 'b'])).item()
        assert result is False

    def test_any_all_object_bool_only(self) -> None:
        df = DataFrame({'A': ['foo', 2], 'B': [True, False]}).astype(object)
        df._consolidate_inplace()
        df['C'] = Series([True, True])
        df['D'] = df['C'].astype('category')
        res = df._get_bool_data()
        expected = df[['C']]
        tm.assert_frame_equal(res, expected)
        res = df.all(bool_only=True, axis=0)
        expected = Series([True], index=['C'])
        tm.assert_series_equal(res, expected)
        res = df[['B', 'C']].all(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)
        assert df.all(bool_only=True, axis=None)
        res = df.any(bool_only=True, axis=0)
        expected = Series([True], index=['C'])
        tm.assert_series_equal(res, expected)
        res = df[['C']].any(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)
        assert df.any(bool_only=True, axis=None)

    def test_series_broadcasting(self) -> None:
        df = DataFrame([1.0, 1.0, 1.0])
        df_nan = DataFrame({'A': [np.nan, 2.0, np.nan]})
        s = Series([1, 1, 1])
        s_nan = Series([np.nan, np.nan, 1])
        with tm.assert_produces_warning(None):
            df_nan.clip(lower=s, axis=0)
            for op in ['lt', 'le', 'gt', 'ge', 'eq', 'ne']:
                getattr(df, op)(s_nan, axis=0)


class TestDataFrameReductions:
    def test_min_max_dt64_with_NaT(self) -> None:
        df = DataFrame({'foo': [pd.NaT, pd.NaT, Timestamp('2012-05-01')]})
        res = df.min()
        exp = Series([Timestamp('2012-05-01')], index=['foo'])
        tm.assert_series_equal(res, exp)
        res = df.max()
        exp = Series([Timestamp('2012-05-01')], index=['foo'])
        tm.assert_series_equal(res, exp)
        df = DataFrame({'foo': [pd.NaT, pd.NaT]})
        res = df.min()
        exp = Series([pd.NaT], index=['foo'])
        tm.assert_series_equal(res, exp)
        res = df.max()
        exp = Series([pd.NaT], index=['foo'])
        tm.assert_series_equal(res, exp)

    def test_min_max_dt64_with_NaT_precision(self) -> None:
        df = DataFrame({'foo': [pd.NaT, pd.NaT, Timestamp('2012-05-01 09:20:00.123456789')]}, dtype='datetime64[ns]')
        res = df.min(axis=1)
        exp = df.foo.rename(None)
        tm.assert_series_equal(res, exp)
        res = df.max(axis=1)
        exp = df.foo.rename(None)
        tm.assert_series_equal(res, exp)

    def test_min_max_td64_with_NaT_precision(self) -> None:
        df = DataFrame({'foo': [pd.NaT, pd.NaT, to_timedelta('10000 days 06:05:01.123456789')]}, dtype='timedelta64[ns]')
        res = df.min(axis=1)
        exp = df.foo.rename(None)
        tm.assert_series_equal(res, exp)
        res = df.max(axis=1)
        exp = df.foo.rename(None)
        tm.assert_series_equal(res, exp)

    def test_min_max_dt64_with_NaT_skipna_false(self, request: pytest.FixtureRequest, tz_naive_fixture: Any) -> None:
        tz = tz_naive_fixture
        if isinstance(tz, tzlocal) and is_platform_windows():
            pytest.skip('GH#37659 OSError raised within tzlocal bc Windows chokes in times before 1970-01-01')
        df = DataFrame({'a': [Timestamp('2020-01-01 08:00:00', tz=tz), Timestamp('1920-02-01 09:00:00', tz=tz)], 'b': [Timestamp('2020-02-01 08:00:00', tz=tz), pd.NaT]})
        res = df.min(axis=1, skipna=False)
        expected = Series([df.loc[0, 'a'], pd.NaT])
        assert expected.dtype == df['a'].dtype
        tm.assert_series_equal(res, expected)
        res = df.max(axis=1, skipna=False)
        expected = Series([df.loc[0, 'b'], pd.NaT])
        assert expected.dtype == df['a'].dtype
        tm.assert_series_equal(res, expected)

    def test_min_max_dt64_api_consistency_with_NaT(self) -> None:
        df = DataFrame({'x': to_datetime([])})
        expected_dt_series = Series(to_datetime([]))
        assert (df.min(axis=0).x is pd.NaT) == (expected_dt_series.min() is pd.NaT)
        assert (df.max(axis=0).x is pd.NaT) == (expected_dt_series.max() is pd.NaT)
        tm.assert_series_equal(df.min(axis=1), expected_dt_series)
        tm.assert_series_equal(df.max(axis=1), expected_dt_series)

    def test_min_max_dt64_api_consistency_empty_df(self) -> None:
        df = DataFrame({'x': []})
        expected_float_series = Series([], dtype=float)
        assert np.isnan(df.min(axis=0).x) == np.isnan(expected_float_series.min())
        assert np.isnan(df.max(axis=0).x) == np.isnan(expected_float_series.max())
        tm.assert_series_equal(df.min(axis=1), expected_float_series)
        tm.assert_series_equal(df.min(axis=1), expected_float_series)

    @pytest.mark.parametrize('initial', ['2018-10-08 13:36:45+00:00', '2018-10-08 13:36:45+03:00'])
    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_preserve_timezone(self, initial: str, method: str) -> None:
        initial_dt = to_datetime(initial)
        expected = Series([initial_dt])
        df = DataFrame([expected])
        result = getattr(df, method)(axis=1)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_minmax_tzaware_skipna_axis_1(self, method: str, skipna: bool) -> None:
        val = to_datetime('1900-01-01', utc=True)
        df = DataFrame({'a': Series([pd.NaT, pd.NaT, val]), 'b': Series([pd.NaT, val, val])})
        op = getattr(df, method)
        result = op(axis=1, skipna=skipna)
        if skipna:
            expected = Series([pd.NaT, val, val])
        else:
            expected = Series([pd.NaT, pd.NaT, val])
        tm.assert_series_equal(result, expected)

    def test_frame_any_with_timedelta(self) -> None:
        df = DataFrame({'a': Series([0, 0]), 't': Series([to_timedelta(0, 's'), to_timedelta(1, 'ms')])})
        result = df.any(axis=0)
        expected = Series(data=[False, True], index=['a', 't'])
        tm.assert_series_equal(result, expected)
        result = df.any(axis=1)
        expected = Series(data=[False, True])
        tm.assert_series_equal(result, expected)

    def test_reductions_skipna_none_raises(self, request: pytest.FixtureRequest, frame_or_series: Callable[[List[int]], Any], all_reductions: str) -> None:
        if all_reductions == 'count':
            request.applymarker(pytest.mark.xfail(reason='Count does not accept skipna'))
        obj = frame_or_series([1, 2, 3])
        msg = 'For argument "skipna" expected type bool, received type NoneType.'
        with pytest.raises(ValueError, match=msg):
            getattr(obj, all_reductions)(skipna=None)

    def test_reduction_timestamp_smallest_unit(self) -> None:
        df = DataFrame({'a': Series([Timestamp('2019-12-31')], dtype='datetime64[s]'), 'b': Series([Timestamp('2019-12-31 00:00:00.123')], dtype='datetime64[ms]')})
        result = df.max()
        expected = Series([Timestamp('2019-12-31'), Timestamp('2019-12-31 00:00:00.123')], dtype='datetime64[ms]', index=['a', 'b'])
        tm.assert_series_equal(result, expected)

    def test_reduction_timedelta_smallest_unit(self) -> None:
        df = DataFrame({'a': Series([pd.Timedelta('1 days')], dtype='timedelta64[s]'), 'b': Series([pd.Timedelta('1 days')], dtype='timedelta64[ms]')})
        result = df.max()
        expected = Series([pd.Timedelta('1 days'), pd.Timedelta('1 days')], dtype='timedelta64[ms]', index=['a', 'b'])
        tm.assert_series_equal(result, expected)


class TestNuisanceColumns:
    def test_any_all_categorical_dtype_nuisance_column(self, all_boolean_reductions: str) -> None:
        ser = Series([0, 1], dtype='category', name='A')
        df = ser.to_frame()
        with pytest.raises(TypeError, match='does not support operation'):
            getattr(ser, all_boolean_reductions)()
        with pytest.raises(TypeError, match='does not support operation'):
            getattr(np, all_boolean_reductions)(ser)
        with pytest.raises(TypeError, match='does not support operation'):
            getattr(df, all_boolean_reductions)(bool_only=False)
        with pytest.raises(TypeError, match='does not support operation'):
            getattr(df, all_boolean_reductions)(bool_only=None)
        with pytest.raises(TypeError, match='does not support operation'):
            getattr(np, all_boolean_reductions)(df, axis=0)

    def test_median_categorical_dtype_nuisance_column(self) -> None:
        df = DataFrame({'A': Categorical([1, 2, 2, 2, 3])})
        ser = df['A']
        with pytest.raises(TypeError, match='does not support operation'):
            ser.median()
        with pytest.raises(TypeError, match='does not support operation'):
            df.median(numeric_only=False)
        with pytest.raises(TypeError, match='does not support operation'):
            df.median()
        df['B'] = df['A'].astype(int)
        with pytest.raises(TypeError, match='does not support operation'):
            df.median(numeric_only=False)
        with pytest.raises(TypeError, match='does not support operation'):
            df.median()

    @pytest.mark.parametrize('method', ['min', 'max'])
    def test_min_max_categorical_dtype_non_ordered_nuisance_column(self, method: str) -> None:
        cat = Categorical(['a', 'b', 'c', 'b'], ordered=False)
        ser = Series(cat)
        df = ser.to_frame('A')
        with pytest.raises(TypeError, match='is not ordered for operation'):
            getattr(ser, method)()
        with pytest.raises(TypeError, match='is not ordered for operation'):
            getattr(np, method)(ser)
        with pytest.raises(TypeError, match='is not ordered for operation'):
            getattr(df, method)(numeric_only=False)
        with pytest.raises(TypeError, match='is not ordered for operation'):
            getattr(df, method)()
        with pytest.raises(TypeError, match='is not ordered for operation'):
            getattr(np, method)(df, axis=0)
        df['B'] = df['A'].astype(object)
        with pytest.raises(TypeError, match='is not ordered for operation'):
            getattr(df, method)()
        with pytest.raises(TypeError, match='is not ordered for operation'):
            getattr(np, method)(df, axis=0)


class TestEmptyDataFrameReductions:
    @pytest.mark.parametrize(
        'opname, dtype, exp_value, exp_dtype',
        [
            ('sum', np.int8, 0, np.int64),
            ('prod', np.int8, 1, np.int_),
            ('sum', np.int64, 0, np.int64),
            ('prod', np.int64, 1, np.int64),
            ('sum', np.uint8, 0, np.uint64),
            ('prod', np.uint8, 1, np.uint),
            ('sum', np.uint64, 0, np.uint64),
            ('prod', np.uint64, 1, np.uint64),
            ('sum', np.float32, 0, np.float32),
            ('prod', np.float32, 1, np.float32),
            ('sum', np.float64, 0, np.float64),
        ],
    )
    def test_df_empty_min_count_0(self, opname: str, dtype: Any, exp_value: Any, exp_dtype: Any) -> None:
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=0)
        expected = Series([exp_value, exp_value], dtype=exp_dtype, index=range(2))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'opname, dtype, exp_dtype',
        [
            ('sum', np.int8, np.float64),
            ('prod', np.int8, np.float64),
            ('sum', np.int64, np.float64),
            ('prod', np.int64, np.float64),
            ('sum', np.uint8, np.float64),
            ('prod', np.uint8, np.float64),
            ('sum', np.uint64, np.float64),
            ('prod', np.uint64, np.float64),
            ('sum', np.float32, np.float32),
            ('prod', np.float32, np.float32),
            ('sum', np.float64, np.float64),
        ],
    )
    def test_df_empty_min_count_1(self, opname: str, dtype: Any, exp_dtype: Any) -> None:
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=1)
        expected = Series([np.nan, np.nan], dtype=exp_dtype, index=Index([0, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'opname, dtype, exp_value, exp_dtype',
        [
            ('sum', 'Int8', 0, 'Int32' if is_windows_np2_or_is32 else 'Int64'),
            ('prod', 'Int8', 1, 'Int32' if is_windows_np2_or_is32 else 'Int64'),
            ('sum', 'Int64', 0, 'Int64'),
            ('prod', 'Int64', 1, 'Int64'),
            ('sum', 'UInt8', 0, 'UInt32' if is_windows_np2_or_is32 else 'UInt64'),
            ('prod', 'UInt8', 1, 'UInt32' if is_windows_np2_or_is32 else 'UInt64'),
            ('sum', 'UInt64', 0, 'UInt64'),
            ('prod', 'UInt64', 1, 'UInt64'),
            ('sum', 'Float32', 0, 'Float32'),
            ('prod', 'Float32', 1, 'Float32'),
            ('sum', 'Float64', 0, 'Float64'),
        ],
    )
    def test_df_empty_nullable_min_count_0(self, opname: str, dtype: str, exp_value: Any, exp_dtype: str) -> None:
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=0)
        expected = Series([exp_value, exp_value], dtype=exp_dtype, index=Index([0, 1]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        'opname, dtype, exp_dtype',
        [
            ('sum', 'Int8', 'Int32' if is_windows_or_is32 else 'Int64'),
            ('prod', 'Int8', 'Int32' if is_windows_or_is32 else 'Int64'),
            ('sum', 'Int64', 'Int64'),
            ('prod', 'Int64', 'Int64'),
            ('sum', 'UInt8', 'UInt32' if is_windows_or_is32 else 'UInt64'),
            ('prod', 'UInt8', 'UInt32' if is_windows_or_is32 else 'UInt64'),
            ('sum', 'UInt64', 'UInt64'),
            ('prod', 'UInt64', 'UInt64'),
            ('sum', 'Float32', 'Float32'),
            ('prod', 'Float32', 'Float32'),
            ('sum', 'Float64', 'Float64'),
        ],
    )
    def test_df_empty_nullable_min_count_1(self, opname: str, dtype: str, exp_dtype: str) -> None:
        df = DataFrame({0: [], 1: []}, dtype=dtype)
        result = getattr(df, opname)(min_count=1)
        expected = Series([pd.NA, pd.NA], dtype=exp_dtype, index=Index([0, 1]))
        tm.assert_series_equal(result, expected)


def test_sum_timedelta64_skipna_false() -> None:
    arr = np.arange(8).astype(np.int64).view('m8[s]').reshape(4, 2)
    arr[-1, -1] = 'Nat'
    df = DataFrame(arr)
    assert (df.dtypes == arr.dtype).all()
    result = df.sum(skipna=False)
    expected = Series([pd.Timedelta(seconds=12), pd.NaT], dtype='m8[s]')
    tm.assert_series_equal(result, expected)
    result = df.sum(axis=0, skipna=False)
    tm.assert_series_equal(result, expected)
    result = df.sum(axis=1, skipna=False)
    expected = Series([pd.Timedelta(seconds=1), pd.Timedelta(seconds=5), pd.Timedelta(seconds=9), pd.NaT], dtype='m8[s]')
    tm.assert_series_equal(result, expected)


def test_mixed_frame_with_integer_sum() -> None:
    df = DataFrame([['a', 1]], columns=list('ab'))
    df = df.astype({'b': 'Int64'})
    result = df.sum()
    expected = Series(['a', 1], index=['a', 'b'])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('numeric_only', [True, False, None])
@pytest.mark.parametrize('method', ['min', 'max'])
def test_minmax_extensionarray(method: str, numeric_only: Optional[bool]) -> None:
    int64_info = np.iinfo('int64')
    ser = Series([int64_info.max, None, int64_info.min], dtype=pd.Int64Dtype())
    df = DataFrame({'Int64': ser})
    result = getattr(df, method)(numeric_only=numeric_only)
    expected = Series([getattr(int64_info, method)], dtype='Int64', index=Index(['Int64']))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('ts_value', [Timestamp('2000-01-01'), pd.NaT])
def test_frame_mixed_numeric_object_with_timestamp(ts_value: object) -> None:
    df = DataFrame({'a': [1], 'b': [1.1], 'c': ['foo'], 'd': [ts_value]})
    with pytest.raises(TypeError, match='does not support operation|Cannot perform'):
        df.sum()


def test_prod_sum_min_count_mixed_object() -> None:
    df = DataFrame([1, 'a', True])
    result = df.prod(axis=0, min_count=1, numeric_only=False)
    expected = Series(['a'], dtype=object)
    tm.assert_series_equal(result, expected)
    msg = re.escape("unsupported operand type(s) for +: 'int' and 'str'")
    with pytest.raises(TypeError, match=msg):
        df.sum(axis=0, min_count=1, numeric_only=False)


@pytest.mark.parametrize('method', ['min', 'max', 'mean', 'median', 'skew', 'kurt'])
@pytest.mark.parametrize('numeric_only', [True, False])
@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
def test_reduction_axis_none_returns_scalar(method: str, numeric_only: bool, dtype: str) -> None:
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), dtype=dtype)
    result = getattr(df, method)(axis=None, numeric_only=numeric_only)
    np_arr = df.to_numpy(dtype=np.float64)
    if method in {'skew', 'kurt'}:
        comp_mod = pytest.importorskip('scipy.stats')
        if method == 'kurt':
            method = 'kurtosis'
        expected = getattr(comp_mod, method)(np_arr, bias=False, axis=None)
        tm.assert_almost_equal(result, expected)
    else:
        expected = getattr(np, method)(np_arr, axis=None)
        assert result == expected


@pytest.mark.parametrize('kernel', ['corr', 'corrwith', 'cov', 'idxmax', 'idxmin', 'kurt', 'max', 'mean', 'median', 'min', 'prod', 'quantile', 'sem', 'skew', 'std', 'sum', 'var'])
def test_fails_on_non_numeric(kernel: str) -> None:
    df = DataFrame({'a': [1, 2, 3], 'b': object})
    args = (df,) if kernel == 'corrwith' else ()
    msg = '|'.join(['not allowed for this dtype', 'argument must be a string or a number', 'not supported between instances of', 'unsupported operand type', 'argument must be a string or a real number'])
    if kernel == 'median':
        msg1 = "Cannot convert \\[\\[<class 'object'> <class 'object'> <class 'object'>\\]\\] to numeric"
        msg2 = "Cannot convert \\[<class 'object'> <class 'object'> <class 'object'>\\] to numeric"
        msg = '|'.join([msg1, msg2])
    with pytest.raises(TypeError, match=msg):
        getattr(df, kernel)(*args)


@pytest.mark.parametrize('method', ['all', 'any', 'count', 'idxmax', 'idxmin', 'kurt', 'kurtosis', 'max', 'mean', 'median', 'min', 'nunique', 'prod', 'product', 'sem', 'skew', 'std', 'sum', 'var'])
@pytest.mark.parametrize('min_count', [0, 2])
def test_numeric_ea_axis_1(method: str, skipna: bool, min_count: int, any_numeric_ea_dtype: str) -> None:
    df = DataFrame({'a': Series([0, 1, 2, 3], dtype=any_numeric_ea_dtype), 'b': Series([0, 1, pd.NA, 3], dtype=any_numeric_ea_dtype)})
    expected_df = DataFrame({'a': [0.0, 1.0, 2.0, 3.0], 'b': [0.0, 1.0, np.nan, 3.0]})
    if method in ('count', 'nunique'):
        expected_dtype = 'int64'
    elif method in ('all', 'any'):
        expected_dtype = 'boolean'
    elif method in ('kurt', 'kurtosis', 'mean', 'median', 'sem', 'skew', 'std', 'var') and (not any_numeric_ea_dtype.startswith('Float')):
        expected_dtype = 'Float64'
    else:
        expected_dtype = any_numeric_ea_dtype
    kwargs: Dict[str, Any] = {}
    if method not in ('count', 'nunique', 'quantile'):
        kwargs['skipna'] = skipna
    if method in ('prod', 'product', 'sum'):
        kwargs['min_count'] = min_count
    if not skipna and method in ('idxmax', 'idxmin'):
        msg = f'The behavior of DataFrame.{method} with all-NA values'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            getattr(df, method)(axis=1, **kwargs)
        with pytest.raises(ValueError, match='Encountered an NA value'):
            getattr(expected_df, method)(axis=1, **kwargs)
        return
    result = getattr(df, method)(axis=1, **kwargs)
    expected = getattr(expected_df, method)(axis=1, **kwargs)
    if method not in ('idxmax', 'idxmin'):
        expected = expected.astype(expected_dtype)
    tm.assert_series_equal(result, expected)