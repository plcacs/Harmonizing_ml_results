from datetime import datetime, timedelta
import inspect
from typing import Any, List, Dict, Union

import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import IS64, is_platform_windows
from pandas.compat.numpy import np_version_gt2
import pandas as pd
from pandas import Categorical, CategoricalIndex, DataFrame, Index, MultiIndex, Series, date_range, isna
import pandas._testing as tm
from pandas.api.types import CategoricalDtype


class TestReindexSetIndex:

    def test_dti_set_index_reindex_datetimeindex(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).random(6))
        idx1: pd.DatetimeIndex = date_range('2011/01/01', periods=6, freq='ME', tz='US/Eastern')
        idx2: pd.DatetimeIndex = date_range('2013', periods=6, freq='YE', tz='Asia/Tokyo')
        df = df.set_index(idx1)
        tm.assert_index_equal(df.index, idx1)
        df = df.reindex(idx2)
        tm.assert_index_equal(df.index, idx2)

    def test_dti_set_index_reindex_freq_with_tz(self) -> None:
        index: pd.DatetimeIndex = date_range(datetime(2015, 10, 1), datetime(2015, 10, 1, 23), freq='h', tz='US/Eastern')
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((24, 1)), columns=['a'], index=index)
        new_index: pd.DatetimeIndex = date_range(datetime(2015, 10, 2), datetime(2015, 10, 2, 23), freq='h', tz='US/Eastern')
        result: DataFrame = df.set_index(new_index)
        assert result.index.freq == index.freq

    def test_set_reset_index_intervalindex(self) -> None:
        df: DataFrame = DataFrame({'A': range(10)})
        ser: Series = pd.cut(df.A, 5)
        df['B'] = ser
        df = df.set_index('B')
        df = df.reset_index()

    def test_setitem_reset_index_dtypes(self) -> None:
        df: DataFrame = DataFrame(columns=['a', 'b', 'c']).astype({'a': 'datetime64[ns]', 'b': np.int64, 'c': np.float64})
        df1: DataFrame = df.set_index(['a'])
        df1['d'] = []
        result: DataFrame = df1.reset_index()
        expected: DataFrame = DataFrame(columns=['a', 'b', 'c', 'd'], index=range(0)).astype({'a': 'datetime64[ns]', 'b': np.int64, 'c': np.float64, 'd': np.float64})
        tm.assert_frame_equal(result, expected)
        df2: DataFrame = df.set_index(['a', 'b'])
        df2['d'] = []
        result = df2.reset_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('timezone, year, month, day, hour', [
        ['America/Chicago', 2013, 11, 3, 1],
        ['America/Santiago', 2021, 4, 3, 23]
    ])
    def test_reindex_timestamp_with_fold(self, timezone: str, year: int, month: int, day: int, hour: int) -> None:
        test_timezone = gettz(timezone)
        transition_1: pd.Timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=0, tzinfo=test_timezone)
        transition_2: pd.Timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=1, tzinfo=test_timezone)
        df: DataFrame = DataFrame({'index': [transition_1, transition_2], 'vals': ['a', 'b']}).set_index('index').reindex(['1', '2'])
        exp: DataFrame = DataFrame({'index': ['1', '2'], 'vals': [np.nan, np.nan]}).set_index('index')
        exp = exp.astype(df.vals.dtype)
        tm.assert_frame_equal(df, exp)


class TestDataFrameSelectReindex:

    @pytest.mark.xfail(not IS64 or (is_platform_windows() and (not np_version_gt2)), reason='Passes int32 values to DatetimeArray in make_na_array on windows, 32bit linux builds')
    def test_reindex_tzaware_fill_value(self) -> None:
        df: DataFrame = DataFrame([[1]])
        ts: pd.Timestamp = pd.Timestamp('2023-04-10 17:32', tz='US/Pacific')
        res: DataFrame = df.reindex([0, 1], axis=1, fill_value=ts)
        assert res.dtypes[1] == pd.DatetimeTZDtype(unit='s', tz='US/Pacific')
        expected: DataFrame = DataFrame({0: [1], 1: [ts]})
        expected[1] = expected[1].astype(res.dtypes[1])
        tm.assert_frame_equal(res, expected)
        per: pd.Period = ts.tz_localize(None).to_period('s')
        res = df.reindex([0, 1], axis=1, fill_value=per)
        assert res.dtypes[1] == pd.PeriodDtype('s')
        expected = DataFrame({0: [1], 1: [per]})
        tm.assert_frame_equal(res, expected)
        interval = pd.Interval(ts, ts + pd.Timedelta(seconds=1))
        res = df.reindex([0, 1], axis=1, fill_value=interval)
        assert res.dtypes[1] == pd.IntervalDtype('datetime64[s, US/Pacific]', 'right')
        expected = DataFrame({0: [1], 1: [interval]})
        expected[1] = expected[1].astype(res.dtypes[1])
        tm.assert_frame_equal(res, expected)

    def test_reindex_date_fill_value(self) -> None:
        arr: np.ndarray = date_range('2016-01-01', periods=6).values.reshape(3, 2)
        df: DataFrame = DataFrame(arr, columns=['A', 'B'], index=range(3))
        ts: Any = df.iloc[0, 0]
        fv: Any = ts.date()
        res: DataFrame = df.reindex(index=range(4), columns=['A', 'B', 'C'], fill_value=fv)
        expected: DataFrame = DataFrame({'A': df['A'].tolist() + [fv], 'B': df['B'].tolist() + [fv], 'C': [fv] * 4}, dtype=object)
        tm.assert_frame_equal(res, expected)
        res = df.reindex(index=range(4), fill_value=fv)
        tm.assert_frame_equal(res, expected[['A', 'B']])
        res = df.reindex(index=range(4), columns=['A', 'B', 'C'], fill_value='2016-01-01')
        expected = DataFrame({'A': df['A'].tolist() + [ts], 'B': df['B'].tolist() + [ts], 'C': [ts] * 4})
        tm.assert_frame_equal(res, expected)

    def test_reindex_with_multi_index(self) -> None:
        df: DataFrame = DataFrame({
            'a': [-1] * 7 + [0] * 7 + [1] * 7,
            'b': list(range(7)) * 3,
            'c': ['A', 'B', 'C', 'D', 'E', 'F', 'G'] * 3
        }).set_index(['a', 'b'])
        new_index: List[float] = [0.5, 2.0, 5.0, 5.8]
        new_multi_index: MultiIndex = MultiIndex.from_product([[0], new_index], names=['a', 'b'])
        reindexed: DataFrame = df.reindex(new_multi_index)
        expected: DataFrame = DataFrame({'a': [0] * 4, 'b': new_index, 'c': [np.nan, 'C', 'F', np.nan]}).set_index(['a', 'b'])
        tm.assert_frame_equal(expected, reindexed)
        expected = DataFrame({'a': [0] * 4, 'b': new_index, 'c': ['B', 'C', 'F', 'G']}).set_index(['a', 'b'])
        reindexed_with_backfilling: DataFrame = df.reindex(new_multi_index, method='bfill')
        tm.assert_frame_equal(expected, reindexed_with_backfilling)
        reindexed_with_backfilling = df.reindex(new_multi_index, method='backfill')
        tm.assert_frame_equal(expected, reindexed_with_backfilling)
        expected = DataFrame({'a': [0] * 4, 'b': new_index, 'c': ['A', 'C', 'F', 'F']}).set_index(['a', 'b'])
        reindexed_with_padding: DataFrame = df.reindex(new_multi_index, method='pad')
        tm.assert_frame_equal(expected, reindexed_with_padding)
        reindexed_with_padding = df.reindex(new_multi_index, method='ffill')
        tm.assert_frame_equal(expected, reindexed_with_padding)

    @pytest.mark.parametrize('method,expected_values', [
        ('nearest', [0, 1, 1, 2]),
        ('pad', [np.nan, 0, 1, 1]),
        ('backfill', [0, 1, 2, 2])
    ])
    def test_reindex_methods(self, method: Union[str, None], expected_values: List[Any]) -> None:
        df: DataFrame = DataFrame({'x': list(range(5))})
        target: np.ndarray = np.array([-0.1, 0.9, 1.1, 1.5])
        expected: DataFrame = DataFrame({'x': expected_values}, index=target)
        actual: DataFrame = df.reindex(target, method=method)
        tm.assert_frame_equal(expected, actual)
        actual = df.reindex(target, method=method, tolerance=1)
        tm.assert_frame_equal(expected, actual)
        actual = df.reindex(target, method=method, tolerance=[1, 1, 1, 1])
        tm.assert_frame_equal(expected, actual)
        e2: DataFrame = expected[::-1]
        actual = df.reindex(target[::-1], method=method)
        tm.assert_frame_equal(e2, actual)
        new_order: List[int] = [3, 0, 2, 1]
        e2 = expected.iloc[new_order]
        actual = df.reindex(target[new_order], method=method)
        tm.assert_frame_equal(e2, actual)
        switched_method: Union[str, None] = 'pad' if method == 'backfill' else 'backfill' if method == 'pad' else method
        actual = df[::-1].reindex(target, method=switched_method)
        tm.assert_frame_equal(expected, actual)

    def test_reindex_methods_nearest_special(self) -> None:
        df: DataFrame = DataFrame({'x': list(range(5))})
        target: np.ndarray = np.array([-0.1, 0.9, 1.1, 1.5])
        expected: DataFrame = DataFrame({'x': [0, 1, 1, np.nan]}, index=target)
        actual: DataFrame = df.reindex(target, method='nearest', tolerance=0.2)
        tm.assert_frame_equal(expected, actual)
        expected = DataFrame({'x': [0, np.nan, 1, np.nan]}, index=target)
        actual = df.reindex(target, method='nearest', tolerance=[0.5, 0.01, 0.4, 0.1])
        tm.assert_frame_equal(expected, actual)

    def test_reindex_nearest_tz(self, tz_aware_fixture: Any) -> None:
        tz = tz_aware_fixture
        idx: pd.DatetimeIndex = date_range('2019-01-01', periods=5, tz=tz)
        df: DataFrame = DataFrame({'x': list(range(5))}, index=idx)
        expected: DataFrame = df.head(3)
        actual: DataFrame = df.reindex(idx[:3], method='nearest')
        tm.assert_frame_equal(expected, actual)

    def test_reindex_nearest_tz_empty_frame(self) -> None:
        dti: pd.DatetimeIndex = pd.DatetimeIndex(['2016-06-26 14:27:26+00:00'])
        df: DataFrame = DataFrame(index=pd.DatetimeIndex(['2016-07-04 14:00:59+00:00']))
        expected: DataFrame = DataFrame(index=dti)
        result: DataFrame = df.reindex(dti, method='nearest')
        tm.assert_frame_equal(result, expected)

    def test_reindex_frame_add_nat(self) -> None:
        rng: pd.DatetimeIndex = date_range('1/1/2000 00:00:00', periods=10, freq='10s')
        df: DataFrame = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': rng})
        result: DataFrame = df.reindex(range(15))
        assert np.issubdtype(result['B'].dtype, np.dtype('M8[ns]'))
        mask: Series = isna(result)['B']
        assert mask[-5:].all()
        assert not mask[:-5].any()

    @pytest.mark.parametrize('method, exp_values', [
        ('ffill', [0, 1, 2, 3]),
        ('bfill', [1.0, 2.0, 3.0, np.nan])
    ])
    def test_reindex_frame_tz_ffill_bfill(self, frame_or_series: Any, method: str, exp_values: List[Any]) -> None:
        obj: Any = frame_or_series([0, 1, 2, 3],
                                   index=date_range('2020-01-01 00:00:00', periods=4, freq='h', tz='UTC'))
        new_index: pd.DatetimeIndex = date_range('2020-01-01 00:01:00', periods=4, freq='h', tz='UTC')
        result: Any = obj.reindex(new_index, method=method, tolerance=pd.Timedelta('1 hour'))
        expected: Any = frame_or_series(exp_values, index=new_index)
        tm.assert_equal(result, expected)

    def test_reindex_limit(self) -> None:
        data: List[List[Any]] = [['A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C', 'C'], ['D', 'D', 'D']]
        exp_data: List[List[Any]] = [['A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C', 'C'], ['D', 'D', 'D'], ['D', 'D', 'D'], [np.nan, np.nan, np.nan]]
        df: DataFrame = DataFrame(data)
        result: DataFrame = df.reindex([0, 1, 2, 3, 4, 5], method='ffill', limit=1)
        expected: DataFrame = DataFrame(exp_data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('idx, check_index_type', [
        [['C', 'B', 'A'], True],
        [['F', 'C', 'A', 'D'], True],
        [['A'], True],
        [['A', 'B', 'C'], True],
        [['C', 'A', 'B'], True],
        [['C', 'B'], True],
        [['C', 'A'], True],
        [['A', 'B'], True],
        [['B', 'A', 'C'], True],
        [['D', 'F'], False],
        [['A', 'C', 'B'], False]
    ])
    def test_reindex_level_verify_first_level(self, idx: List[str], check_index_type: bool) -> None:
        df: DataFrame = DataFrame({
            'jim': list('B' * 4 + 'A' * 2 + 'C' * 3),
            'joe': list('abcdeabcd')[::-1],
            'jolie': [10, 20, 30] * 3,
            'joline': np.random.default_rng(2).integers(0, 1000, 9)
        })
        icol: List[str] = ['jim', 'joe', 'jolie']

        def f(val: str) -> np.ndarray:
            return np.nonzero((df['jim'] == val).to_numpy())[0]
        i: np.ndarray = np.concatenate(list(map(f, idx)))
        left: DataFrame = df.set_index(icol).reindex(idx, level='jim')
        right: DataFrame = df.iloc[i].set_index(icol)
        tm.assert_frame_equal(left, right, check_index_type=check_index_type)

    @pytest.mark.parametrize('idx', [
        ('mid',), ('mid', 'btm'), ('mid', 'btm', 'top'),
        ('mid', 'top'), ('mid', 'top', 'btm'),
        ('btm',), ('btm', 'mid'), ('btm', 'mid', 'top'),
        ('btm', 'top'), ('btm', 'top', 'mid'),
        ('top',), ('top', 'mid'), ('top', 'mid', 'btm'),
        ('top', 'btm'), ('top', 'btm', 'mid')
    ])
    def test_reindex_level_verify_first_level_repeats(self, idx: Any) -> None:
        df: DataFrame = DataFrame({
            'jim': ['mid'] * 5 + ['btm'] * 8 + ['top'] * 7,
            'joe': ['3rd'] * 2 + ['1st'] * 3 + ['2nd'] * 3 + ['1st'] * 2 + ['3rd'] * 3 + ['1st'] * 2 + ['3rd'] * 3 + ['2nd'] * 2,
            'jolie': np.concatenate([np.random.default_rng(2).choice(1000, x, replace=False) for x in [2, 3, 3, 2, 3, 2, 3, 2]]),
            'joline': np.random.default_rng(2).standard_normal(20).round(3) * 10
        })
        icol: List[str] = ['jim', 'joe', 'jolie']

        def f(val: str) -> np.ndarray:
            return np.nonzero((df['jim'] == val).to_numpy())[0]
        i: np.ndarray = np.concatenate(list(map(f, idx)))
        left: DataFrame = df.set_index(icol).reindex(idx, level='jim')
        right: DataFrame = df.iloc[i].set_index(icol)
        tm.assert_frame_equal(left, right)

    @pytest.mark.parametrize('idx, indexer', [
        [['1st', '2nd', '3rd'], [2, 3, 4, 0, 1, 8, 9, 5, 6, 7, 10, 11, 12, 13, 14, 18, 19, 15, 16, 17]],
        [['3rd', '2nd', '1st'], [0, 1, 2, 3, 4, 10, 11, 12, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 13, 14]],
        [['2nd', '3rd'], [0, 1, 5, 6, 7, 10, 11, 12, 18, 19, 15, 16, 17]],
        [['3rd', '1st'], [0, 1, 2, 3, 4, 10, 11, 12, 8, 9, 15, 16, 17, 13, 14]]
    ])
    def test_reindex_level_verify_repeats(self, idx: List[str], indexer: List[int]) -> None:
        df: DataFrame = DataFrame({
            'jim': ['mid'] * 5 + ['btm'] * 8 + ['top'] * 7,
            'joe': ['3rd'] * 2 + ['1st'] * 3 + ['2nd'] * 3 + ['1st'] * 2 + ['3rd'] * 3 + ['1st'] * 2 + ['3rd'] * 3 + ['2nd'] * 2,
            'jolie': np.concatenate([np.random.default_rng(2).choice(1000, x, replace=False) for x in [2, 3, 3, 2, 3, 2, 3, 2]]),
            'joline': np.random.default_rng(2).standard_normal(20).round(3) * 10
        })
        icol: List[str] = ['jim', 'joe', 'jolie']
        left: DataFrame = df.set_index(icol).reindex(idx, level='joe')
        right: DataFrame = df.iloc[indexer].set_index(icol)
        tm.assert_frame_equal(left, right)

    @pytest.mark.parametrize('idx, indexer, check_index_type', [
        [list('abcde'), [3, 2, 1, 0, 5, 4, 8, 7, 6], True],
        [list('abcd'), [3, 2, 1, 0, 5, 8, 7, 6], True],
        [list('abc'), [3, 2, 1, 8, 7, 6], True],
        [list('eca'), [1, 3, 4, 6, 8], True],
        [list('edc'), [0, 1, 4, 5, 6], True],
        [list('eadbc'), [3, 0, 2, 1, 4, 5, 8, 7, 6], True],
        [list('edwq'), [0, 4, 5], True],
        [list('wq'), [], False]
    ])
    def test_reindex_level_verify(self, idx: List[str], indexer: List[int], check_index_type: bool) -> None:
        df: DataFrame = DataFrame({
            'jim': list('B' * 4 + 'A' * 2 + 'C' * 3),
            'joe': list('abcdeabcd')[::-1],
            'jolie': [10, 20, 30] * 3,
            'joline': np.random.default_rng(2).integers(0, 1000, 9)
        })
        icol: List[str] = ['jim', 'joe', 'jolie']
        left: DataFrame = df.set_index(icol).reindex(idx, level='joe')
        right: DataFrame = df.iloc[indexer].set_index(icol)
        tm.assert_frame_equal(left, right, check_index_type=check_index_type)

    def test_non_monotonic_reindex_methods(self) -> None:
        dr: pd.DatetimeIndex = date_range('2013-08-01', periods=6, freq='B')
        data: np.ndarray = np.random.default_rng(2).standard_normal((6, 1))
        df: DataFrame = DataFrame(data, index=dr, columns=list('A'))
        df_rev: DataFrame = DataFrame(data, index=dr[[3, 4, 5] + [0, 1, 2]], columns=list('A'))
        msg: str = 'index must be monotonic increasing or decreasing'
        with pytest.raises(ValueError, match=msg):
            df_rev.reindex(df.index, method='pad')
        with pytest.raises(ValueError, match=msg):
            df_rev.reindex(df.index, method='ffill')
        with pytest.raises(ValueError, match=msg):
            df_rev.reindex(df.index, method='bfill')
        with pytest.raises(ValueError, match=msg):
            df_rev.reindex(df.index, method='nearest')

    def test_reindex_sparse(self) -> None:
        df: DataFrame = DataFrame({'A': [0, 1], 'B': pd.array([0, 1], dtype=pd.SparseDtype('int64', 0))})
        result: DataFrame = df.reindex([0, 2])
        expected: DataFrame = DataFrame({'A': [0.0, np.nan], 'B': pd.array([0.0, np.nan], dtype=pd.SparseDtype('float64', 0.0))}, index=[0, 2])
        tm.assert_frame_equal(result, expected)

    def test_reindex(self, float_frame: DataFrame) -> None:
        datetime_series: Series = Series(np.arange(30, dtype=np.float64), index=date_range('2020-01-01', periods=30))
        newFrame: DataFrame = float_frame.reindex(datetime_series.index)
        for col in newFrame.columns:
            for idx, val in newFrame[col].items():
                if idx in float_frame.index:
                    if np.isnan(val):
                        assert np.isnan(float_frame[col][idx])
                    else:
                        assert val == float_frame[col][idx]
                else:
                    assert np.isnan(val)
        for col, series in newFrame.items():
            tm.assert_index_equal(series.index, newFrame.index)
        emptyFrame: DataFrame = float_frame.reindex(Index([]))
        assert len(emptyFrame.index) == 0
        nonContigFrame: DataFrame = float_frame.reindex(datetime_series.index[::2])
        for col in nonContigFrame.columns:
            for idx, val in nonContigFrame[col].items():
                if idx in float_frame.index:
                    if np.isnan(val):
                        assert np.isnan(float_frame[col][idx])
                    else:
                        assert val == float_frame[col][idx]
                else:
                    assert np.isnan(val)
        for col, series in nonContigFrame.items():
            tm.assert_index_equal(series.index, nonContigFrame.index)
        newFrame = float_frame.reindex(float_frame.index)
        assert newFrame.index.is_(float_frame.index)
        newFrame = float_frame.reindex([])
        assert newFrame.empty
        assert len(newFrame.columns) == len(float_frame.columns)
        newFrame = float_frame.reindex([])
        newFrame = newFrame.reindex(float_frame.index)
        assert len(newFrame.index) == len(float_frame.index)
        assert len(newFrame.columns) == len(float_frame.columns)
        newFrame = float_frame.reindex(list(datetime_series.index))
        expected_index: Index = datetime_series.index._with_freq(None)
        tm.assert_index_equal(newFrame.index, expected_index)
        result: DataFrame = float_frame.reindex()
        tm.assert_frame_equal(result, float_frame)
        assert result is not float_frame

    def test_reindex_nan(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 5], [7, 11], [9, 23]], index=[2, np.nan, 1, 5], columns=['joe', 'jim'])
        i: List[Any] = [np.nan, 5, 5, np.nan, 1, 2, np.nan]
        j: List[int] = [1, 3, 3, 1, 2, 0, 1]
        tm.assert_frame_equal(df.reindex(i), df.iloc[j])
        df.index = df.index.astype('object')
        tm.assert_frame_equal(df.reindex(i), df.iloc[j], check_index_type=False)
        df = DataFrame({'other': ['a', 'b', np.nan, 'c'], 'date': ['2015-03-22', np.nan, '2012-01-08', np.nan], 'amount': [2, 3, 4, 5]})
        df['date'] = pd.to_datetime(df.date)
        df['delta'] = (pd.to_datetime('2015-06-18') - df['date']).shift(1)
        left: DataFrame = df.set_index(['delta', 'other', 'date']).reset_index()
        right: DataFrame = df.reindex(columns=['delta', 'other', 'date', 'amount'])
        tm.assert_frame_equal(left, right)

    def test_reindex_name_remains(self) -> None:
        s: Series = Series(np.random.default_rng(2).random(10))
        df: DataFrame = DataFrame(s, index=np.arange(len(s)))
        i: Series = Series(np.arange(10), name='iname')
        df = df.reindex(i)
        assert df.index.name == 'iname'
        df = df.reindex(Index(np.arange(10), name='tmpname'))
        assert df.index.name == 'tmpname'
        s = Series(np.random.default_rng(2).random(10))
        df = DataFrame(s.T, index=np.arange(len(s)))
        i = Series(np.arange(10), name='iname')
        df = df.reindex(columns=i)
        assert df.columns.name == 'iname'

    def test_reindex_int(self, int_frame: DataFrame) -> None:
        smaller: DataFrame = int_frame.reindex(int_frame.index[::2])
        assert smaller['A'].dtype == np.int64
        bigger: DataFrame = smaller.reindex(int_frame.index)
        assert bigger['A'].dtype == np.float64
        smaller = int_frame.reindex(columns=['A', 'B'])
        assert smaller['A'].dtype == np.int64

    def test_reindex_columns(self, float_frame: DataFrame) -> None:
        new_frame: DataFrame = float_frame.reindex(columns=['A', 'B', 'E'])
        tm.assert_series_equal(new_frame['B'], float_frame['B'])
        assert np.isnan(new_frame['E']).all()
        assert 'C' not in new_frame
        new_frame = float_frame.reindex(columns=[])
        assert new_frame.empty

    def test_reindex_columns_method(self) -> None:
        df: DataFrame = DataFrame(data=[[11, 12, 13], [21, 22, 23], [31, 32, 33]], index=[1, 2, 4], columns=[1, 2, 4], dtype=float)
        result: DataFrame = df.reindex(columns=range(6))
        expected: DataFrame = DataFrame(data=[[np.nan, 11, 12, np.nan, 13, np.nan],
                                               [np.nan, 21, 22, np.nan, 23, np.nan],
                                               [np.nan, 31, 32, np.nan, 33, np.nan]],
                                         index=[1, 2, 4], columns=range(6), dtype=float)
        tm.assert_frame_equal(result, expected)
        result = df.reindex(columns=range(6), method='ffill')
        expected = DataFrame(data=[[np.nan, 11, 12, 12, 13, 13],
                                   [np.nan, 21, 22, 22, 23, 23],
                                   [np.nan, 31, 32, 32, 33, 33]],
                             index=[1, 2, 4], columns=range(6), dtype=float)
        tm.assert_frame_equal(result, expected)
        result = df.reindex(columns=range(6), method='bfill')
        expected = DataFrame(data=[[11, 11, 12, 13, 13, np.nan],
                                   [21, 21, 22, 23, 23, np.nan],
                                   [31, 31, 32, 33, 33, np.nan]],
                             index=[1, 2, 4], columns=range(6), dtype=float)
        tm.assert_frame_equal(result, expected)

    def test_reindex_axes(self) -> None:
        df: DataFrame = DataFrame(np.ones((3, 3)), index=[datetime(2012, 1, 1), datetime(2012, 1, 2), datetime(2012, 1, 3)], columns=['a', 'b', 'c'])
        msg: str = "'d' is deprecated and will be removed in a future version."
        with tm.assert_produces_warning(FutureWarning, match=msg):
            time_freq = date_range('2012-01-01', '2012-01-03', freq='d')
        some_cols: List[str] = ['a', 'b']
        index_freq = df.reindex(index=time_freq).index.freq
        both_freq = df.reindex(index=time_freq, columns=some_cols).index.freq
        seq_freq = df.reindex(index=time_freq).reindex(columns=some_cols).index.freq
        assert index_freq == both_freq
        assert index_freq == seq_freq

    def test_reindex_fill_value(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        result: DataFrame = df.reindex(list(range(15)))
        assert np.isnan(result.values[-5:]).all()
        result = df.reindex(range(15), fill_value=0)
        expected: DataFrame = df.reindex(range(15)).fillna(0)
        tm.assert_frame_equal(result, expected)
        result = df.reindex(columns=range(5), fill_value=0.0)
        expected = df.copy()
        expected[4] = 0.0
        tm.assert_frame_equal(result, expected)
        result = df.reindex(columns=range(5), fill_value=0)
        expected = df.copy()
        expected[4] = 0
        tm.assert_frame_equal(result, expected)
        result = df.reindex(columns=range(5), fill_value='foo')
        expected = df.copy()
        expected[4] = 'foo'
        tm.assert_frame_equal(result, expected)
        df['foo'] = 'foo'
        result = df.reindex(range(15), fill_value='0')
        expected = df.reindex(range(15)).fillna('0')
        tm.assert_frame_equal(result, expected)

    def test_reindex_uint_dtypes_fill_value(self, any_unsigned_int_numpy_dtype: Any) -> None:
        df: DataFrame = DataFrame({'a': [1, 2], 'b': [1, 2]}, dtype=any_unsigned_int_numpy_dtype)
        result: DataFrame = df.reindex(columns=list('abcd'), index=[0, 1, 2, 3], fill_value=10)
        expected: DataFrame = DataFrame({'a': [1, 2, 10, 10], 'b': [1, 2, 10, 10], 'c': 10, 'd': 10}, dtype=any_unsigned_int_numpy_dtype)
        tm.assert_frame_equal(result, expected)

    def test_reindex_single_column_ea_index_and_columns(self, any_numeric_ea_dtype: Any) -> None:
        df: DataFrame = DataFrame({'a': [1, 2]}, dtype=any_numeric_ea_dtype)
        result: DataFrame = df.reindex(columns=list('ab'), index=[0, 1, 2], fill_value=10)
        expected: DataFrame = DataFrame({'a': Series([1, 2, 10], dtype=any_numeric_ea_dtype), 'b': 10})
        tm.assert_frame_equal(result, expected)

    def test_reindex_dups(self) -> None:
        arr: np.ndarray = np.random.default_rng(2).standard_normal(10)
        df: DataFrame = DataFrame(arr, index=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        result: DataFrame = df.copy()
        result.index = list(range(len(df)))
        expected: DataFrame = DataFrame(arr, index=list(range(len(df))))
        tm.assert_frame_equal(result, expected)
        msg: str = 'cannot reindex on an axis with duplicate labels'
        with pytest.raises(ValueError, match=msg):
            df.reindex(index=list(range(len(df))))

    def test_reindex_with_duplicate_columns(self) -> None:
        df: DataFrame = DataFrame([[1, 5, 7.0], [1, 5, 7.0], [1, 5, 7.0]], columns=['bar', 'a', 'a'])
        msg: str = 'cannot reindex on an axis with duplicate labels'
        with pytest.raises(ValueError, match=msg):
            df.reindex(columns=['bar'])
        with pytest.raises(ValueError, match=msg):
            df.reindex(columns=['bar', 'foo'])

    def test_reindex_axis_style(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        expected: DataFrame = DataFrame({'A': [1, 2, np.nan], 'B': [4, 5, np.nan]}, index=[0, 1, 3])
        result: DataFrame = df.reindex([0, 1, 3])
        tm.assert_frame_equal(result, expected)
        result = df.reindex([0, 1, 3], axis=0)
        tm.assert_frame_equal(result, expected)
        result = df.reindex([0, 1, 3], axis='index')
        tm.assert_frame_equal(result, expected)

    def test_reindex_positional_raises(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        msg: str = 'reindex\\(\\) takes from 1 to 2 positional arguments but 3 were given'
        with pytest.raises(TypeError, match=msg):
            df.reindex([0, 1], ['A', 'B', 'C'])

    def test_reindex_axis_style_raises(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex([0, 1], columns=['A'], axis=1)
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex([0, 1], columns=['A'], axis='index')
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis='index')
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis='columns')
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(columns=[0, 1], axis='columns')
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], columns=[0, 1], axis='columns')
        with pytest.raises(TypeError, match='Cannot specify all'):
            df.reindex(labels=[0, 1], index=[0], columns=['A'])
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis='index')
        with pytest.raises(TypeError, match="Cannot specify both 'axis'"):
            df.reindex(index=[0, 1], axis='columns')
        with pytest.raises(TypeError, match='multiple values'):
            df.reindex([0, 1], labels=[0, 1])

    def test_reindex_single_named_indexer(self) -> None:
        df: DataFrame = DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3]})
        result: DataFrame = df.reindex([0, 1], columns=['A'])
        expected: DataFrame = DataFrame({'A': [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_reindex_api_equivalence(self) -> None:
        df: DataFrame = DataFrame([[1, 2, 3], [3, 4, 5], [5, 6, 7]], index=['a', 'b', 'c'], columns=['d', 'e', 'f'])
        res1: DataFrame = df.reindex(['b', 'a'])
        res2: DataFrame = df.reindex(index=['b', 'a'])
        res3: DataFrame = df.reindex(labels=['b', 'a'])
        res4: DataFrame = df.reindex(labels=['b', 'a'], axis=0)
        res5: DataFrame = df.reindex(['b', 'a'], axis=0)
        for res in [res2, res3, res4, res5]:
            tm.assert_frame_equal(res1, res)
        res1 = df.reindex(columns=['e', 'd'])
        res2 = df.reindex(['e', 'd'], axis=1)
        res3 = df.reindex(labels=['e', 'd'], axis=1)
        for res in [res2, res3]:
            tm.assert_frame_equal(res1, res)
        res1 = df.reindex(index=['b', 'a'], columns=['e', 'd'])
        res2 = df.reindex(columns=['e', 'd'], index=['b', 'a'])
        res3 = df.reindex(labels=['b', 'a'], axis=0).reindex(labels=['e', 'd'], axis=1)
        for res in [res2, res3]:
            tm.assert_frame_equal(res1, res)

    def test_reindex_boolean(self) -> None:
        frame: DataFrame = DataFrame(np.ones((10, 2), dtype=bool), index=np.arange(0, 20, 2), columns=[0, 2])
        reindexed: DataFrame = frame.reindex(np.arange(10))
        assert reindexed.values.dtype == np.object_
        assert isna(reindexed[0][1])
        reindexed = frame.reindex(columns=range(3))
        assert reindexed.values.dtype == np.object_
        assert isna(reindexed[1]).all()

    def test_reindex_objects(self, float_string_frame: DataFrame) -> None:
        reindexed: DataFrame = float_string_frame.reindex(columns=['foo', 'A', 'B'])
        assert 'foo' in reindexed
        reindexed = float_string_frame.reindex(columns=['A', 'B'])
        assert 'foo' not in reindexed

    def test_reindex_corner(self, int_frame: DataFrame) -> None:
        index_: Index = Index(['a', 'b', 'c'])
        dm: DataFrame = DataFrame({}).reindex(index=[1, 2, 3])
        reindexed: DataFrame = dm.reindex(columns=index_)
        tm.assert_index_equal(reindexed.columns, index_)
        smaller: DataFrame = int_frame.reindex(columns=['A', 'B', 'E'])
        assert smaller['E'].dtype == np.float64

    def test_reindex_with_nans(self) -> None:
        df: DataFrame = DataFrame([[1, 2], [3, 4], [np.nan, np.nan], [7, 8], [9, 10]], columns=['a', 'b'], index=[100.0, 101.0, np.nan, 102.0, 103.0])
        result: DataFrame = df.reindex(index=[101.0, 102.0, 103.0])
        expected: DataFrame = df.iloc[[1, 3, 4]]
        tm.assert_frame_equal(result, expected)
        result = df.reindex(index=[103.0])
        expected = df.iloc[[4]]
        tm.assert_frame_equal(result, expected)
        result = df.reindex(index=[101.0])
        expected = df.iloc[[1]]
        tm.assert_frame_equal(result, expected)

    def test_reindex_without_upcasting(self) -> None:
        df: DataFrame = DataFrame(np.zeros((10, 10), dtype=np.float32))
        result: DataFrame = df.reindex(columns=np.arange(5, 15))
        assert result.dtypes.eq(np.float32).all()

    def test_reindex_multi(self) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
        result: DataFrame = df.reindex(index=range(4), columns=range(4))
        expected: DataFrame = df.reindex(list(range(4))).reindex(columns=range(4))
        tm.assert_frame_equal(result, expected)
        df = DataFrame(np.random.default_rng(2).integers(0, 10, (3, 3)))
        result = df.reindex(index=range(4), columns=range(4))
        expected = df.reindex(list(range(4))).reindex(columns=range(4))
        tm.assert_frame_equal(result, expected)
        df = DataFrame(np.random.default_rng(2).integers(0, 10, (3, 3)))
        result = df.reindex(index=range(2), columns=range(2))
        expected = df.reindex(range(2)).reindex(columns=range(2))
        tm.assert_frame_equal(result, expected)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)) + 1j, columns=['a', 'b', 'c'])
        result = df.reindex(index=[0, 1], columns=['a', 'b'])
        expected = df.reindex([0, 1]).reindex(columns=['a', 'b'])
        tm.assert_frame_equal(result, expected)

    def test_reindex_multi_categorical_time(self) -> None:
        midx: MultiIndex = MultiIndex.from_product([Categorical(['a', 'b', 'c']), Categorical(date_range('2012-01-01', periods=3, freq='h'))])
        df: DataFrame = DataFrame({'a': range(len(midx))}, index=midx)
        df2: DataFrame = df.iloc[[0, 1, 2, 3, 4, 5, 6, 8]]
        result: DataFrame = df2.reindex(midx)
        expected: DataFrame = DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, np.nan, 8]}, index=midx)
        tm.assert_frame_equal(result, expected)

    def test_reindex_with_categoricalindex(self) -> None:
        df: DataFrame = DataFrame({'A': np.arange(3, dtype='int64')}, index=CategoricalIndex(list('abc'), dtype=CategoricalDtype(list('cabe')), name='B'))
        result: DataFrame = df.reindex(['a', 'b', 'e'])
        expected: DataFrame = DataFrame({'A': [0, 1, np.nan], 'B': Series(list('abe'))}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(['a', 'b'])
        expected = DataFrame({'A': [0, 1], 'B': Series(list('ab'))}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(['e'])
        expected = DataFrame({'A': [np.nan], 'B': Series(['e'])}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(['d'])
        expected = DataFrame({'A': [np.nan], 'B': Series(['d'])}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        cats: List[str] = list('cabe')
        result = df.reindex(Categorical(['a', 'e'], categories=cats))
        expected = DataFrame({'A': [0, np.nan], 'B': Series(list('ae')).astype(CategoricalDtype(cats))}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(Categorical(['a'], categories=cats))
        expected = DataFrame({'A': [0], 'B': Series(list('a')).astype(CategoricalDtype(cats))}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(['a', 'b', 'e'])
        expected = DataFrame({'A': [0, 1, np.nan], 'B': Series(list('abe'))}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(['a', 'b'])
        expected = DataFrame({'A': [0, 1], 'B': Series(list('ab'))}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(['e'])
        expected = DataFrame({'A': [np.nan], 'B': Series(['e'])}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(Categorical(['a', 'e'], categories=cats, ordered=True))
        expected = DataFrame({'A': [0, np.nan], 'B': Series(list('ae')).astype(CategoricalDtype(cats, ordered=True))}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        result = df.reindex(Categorical(['a', 'd'], categories=['a', 'd']))
        expected = DataFrame({'A': [0, np.nan], 'B': Series(list('ad')).astype(CategoricalDtype(['a', 'd']))}).set_index('B')
        tm.assert_frame_equal(result, expected, check_index_type=True)
        df2 = DataFrame({'A': np.arange(6, dtype='int64')}, index=CategoricalIndex(list('aabbca'), dtype=CategoricalDtype(list('cabe')), name='B'))
        msg: str = 'cannot reindex on an axis with duplicate labels'
        with pytest.raises(ValueError, match=msg):
            df2.reindex(['a', 'b'])
        msg = 'argument {} is not implemented for CategoricalIndex\\.reindex'
        with pytest.raises(NotImplementedError, match=msg.format('method')):
            df.reindex(['a'], method='ffill')
        with pytest.raises(NotImplementedError, match=msg.format('level')):
            df.reindex(['a'], level=1)
        with pytest.raises(NotImplementedError, match=msg.format('limit')):
            df.reindex(['a'], limit=2)

    def test_reindex_signature(self) -> None:
        sig: inspect.Signature = inspect.signature(DataFrame.reindex)
        parameters: set = set(sig.parameters)
        assert parameters == {'self', 'labels', 'index', 'columns', 'axis', 'limit', 'copy', 'level', 'method', 'fill_value', 'tolerance'}

    def test_reindex_multiindex_ffill_added_rows(self) -> None:
        mi: MultiIndex = MultiIndex.from_tuples([('a', 'b'), ('d', 'e')])
        df: DataFrame = DataFrame([[0, 7], [3, 4]], index=mi, columns=['x', 'y'])
        mi2: MultiIndex = MultiIndex.from_tuples([('a', 'b'), ('d', 'e'), ('h', 'i')])
        result: DataFrame = df.reindex(mi2, axis=0, method='ffill')
        expected: DataFrame = DataFrame([[0, 7], [3, 4], [3, 4]], index=mi2, columns=['x', 'y'])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('kwargs', [
        {'method': 'pad', 'tolerance': timedelta(seconds=9)},
        {'method': 'backfill', 'tolerance': timedelta(seconds=9)},
        {'method': 'nearest'},
        {'method': None}
    ])
    def test_reindex_empty_frame(self, kwargs: Dict[str, Any]) -> None:
        idx: pd.DatetimeIndex = date_range(start='2020', freq='30s', periods=3)
        df: DataFrame = DataFrame([], index=Index([], name='time'), columns=['a'])
        result: DataFrame = df.reindex(idx, **kwargs)
        expected: DataFrame = DataFrame({'a': [np.nan] * 3}, index=idx, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('src_idx', [Index, CategoricalIndex])
    @pytest.mark.parametrize('cat_idx', [Index([]), CategoricalIndex([]), Index(['A', 'B']), CategoricalIndex(['A', 'B']), Index(['A', 'A']), CategoricalIndex(['A', 'A'])])
    def test_reindex_empty(self, src_idx: Any, cat_idx: Any) -> None:
        df: DataFrame = DataFrame(columns=src_idx([]), index=['K'], dtype='f8')
        result: DataFrame = df.reindex(columns=cat_idx)
        expected: DataFrame = DataFrame(index=['K'], columns=cat_idx, dtype='f8')
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('dtype', ['m8[ns]', 'M8[ns]'])
    def test_reindex_datetimelike_to_object(self, dtype: str) -> None:
        mi: MultiIndex = MultiIndex.from_product([list('ABCDE'), range(2)])
        dti: pd.DatetimeIndex = date_range('2016-01-01', periods=10)
        fv: Any = np.timedelta64('NaT', 'ns')
        if dtype == 'm8[ns]':
            dti = dti - dti[0]
            fv = np.datetime64('NaT', 'ns')
        ser: Series = Series(dti, index=mi)
        ser[::3] = pd.NaT
        df: DataFrame = ser.unstack()
        index: Index = df.index.append(Index([1]))
        columns: Index = df.columns.append(Index(['foo']))
        res: DataFrame = df.reindex(index=index, columns=columns, fill_value=fv)
        expected: DataFrame = DataFrame({0: df[0].tolist() + [fv], 1: df[1].tolist() + [fv], 'foo': np.array(['NaT'] * 6, dtype=fv.dtype)}, index=index)
        assert (res.dtypes[[0, 1]] == object).all()
        assert res.iloc[0, 0] is pd.NaT
        assert res.iloc[-1, 0] is fv
        assert res.iloc[-1, 1] is fv
        tm.assert_frame_equal(res, expected)

    @pytest.mark.parametrize('klass', [Index, CategoricalIndex])
    @pytest.mark.parametrize('data', ['A', 'B'])
    def test_reindex_not_category(self, klass: Any, data: str) -> None:
        df: DataFrame = DataFrame(index=CategoricalIndex([], categories=['A']))
        idx: Any = klass([data])
        result: DataFrame = df.reindex(index=idx)
        expected: DataFrame = DataFrame(index=idx)
        tm.assert_frame_equal(result, expected)

    def test_invalid_method(self) -> None:
        df: DataFrame = DataFrame({'A': [1, np.nan, 2]})
        msg: str = 'Invalid fill method'
        with pytest.raises(ValueError, match=msg):
            df.reindex([1, 0, 2], method='asfreq')