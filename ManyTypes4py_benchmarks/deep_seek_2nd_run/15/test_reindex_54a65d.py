from datetime import datetime, timedelta
import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import IS64, is_platform_windows
from pandas.compat.numpy import np_version_gt2
import pandas as pd
from pandas import Categorical, CategoricalIndex, DataFrame, Index, MultiIndex, Series, date_range, isna
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.common import is_datetime64_dtype, is_timedelta64_dtype
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.series import Series
from pandas.core.frame import DataFrame

class TestReindexSetIndex:

    def test_dti_set_index_reindex_datetimeindex(self) -> None:
        df = DataFrame(np.random.default_rng(2).random(6))
        idx1 = date_range('2011/01/01', periods=6, freq='ME', tz='US/Eastern')
        idx2 = date_range('2013', periods=6, freq='YE', tz='Asia/Tokyo')
        df = df.set_index(idx1)
        tm.assert_index_equal(df.index, idx1)
        df = df.reindex(idx2)
        tm.assert_index_equal(df.index, idx2)

    def test_dti_set_index_reindex_freq_with_tz(self) -> None:
        index = date_range(datetime(2015, 10, 1), datetime(2015, 10, 1, 23), freq='h', tz='US/Eastern')
        df = DataFrame(np.random.default_rng(2).standard_normal((24, 1)), columns=['a'], index=index)
        new_index = date_range(datetime(2015, 10, 2), datetime(2015, 10, 2, 23), freq='h', tz='US/Eastern')
        result = df.set_index(new_index)
        assert result.index.freq == index.freq

    def test_set_reset_index_intervalindex(self) -> None:
        df = DataFrame({'A': range(10)})
        ser = pd.cut(df.A, 5)
        df['B'] = ser
        df = df.set_index('B')
        df = df.reset_index()

    def test_setitem_reset_index_dtypes(self) -> None:
        df = DataFrame(columns=['a', 'b', 'c']).astype({'a': 'datetime64[ns]', 'b': np.int64, 'c': np.float64})
        df1 = df.set_index(['a'])
        df1['d'] = []
        result = df1.reset_index()
        expected = DataFrame(columns=['a', 'b', 'c', 'd'], index=range(0)).astype({'a': 'datetime64[ns]', 'b': np.int64, 'c': np.float64, 'd': np.float64})
        tm.assert_frame_equal(result, expected)
        df2 = df.set_index(['a', 'b'])
        df2['d'] = []
        result = df2.reset_index()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('timezone, year, month, day, hour', [['America/Chicago', 2013, 11, 3, 1], ['America/Santiago', 2021, 4, 3, 23]])
    def test_reindex_timestamp_with_fold(self, timezone: str, year: int, month: int, day: int, hour: int) -> None:
        test_timezone = gettz(timezone)
        transition_1 = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=0, tzinfo=test_timezone)
        transition_2 = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=1, tzinfo=test_timezone)
        df = DataFrame({'index': [transition_1, transition_2], 'vals': ['a', 'b']}).set_index('index').reindex(['1', '2'])
        exp = DataFrame({'index': ['1', '2'], 'vals': [np.nan, np.nan]}).set_index('index')
        exp = exp.astype(df.vals.dtype)
        tm.assert_frame_equal(df, exp)

class TestDataFrameSelectReindex:

    @pytest.mark.xfail(not IS64 or (is_platform_windows() and (not np_version_gt2)), reason='Passes int32 values to DatetimeArray in make_na_array on windows, 32bit linux builds')
    def test_reindex_tzaware_fill_value(self) -> None:
        df = DataFrame([[1]])
        ts = pd.Timestamp('2023-04-10 17:32', tz='US/Pacific')
        res = df.reindex([0, 1], axis=1, fill_value=ts)
        assert res.dtypes[1] == pd.DatetimeTZDtype(unit='s', tz='US/Pacific')
        expected = DataFrame({0: [1], 1: [ts]})
        expected[1] = expected[1].astype(res.dtypes[1])
        tm.assert_frame_equal(res, expected)
        per = ts.tz_localize(None).to_period('s')
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
        arr = date_range('2016-01-01', periods=6).values.reshape(3, 2)
        df = DataFrame(arr, columns=['A', 'B'], index=range(3))
        ts = df.iloc[0, 0]
        fv = ts.date()
        res = df.reindex(index=range(4), columns=['A', 'B', 'C'], fill_value=fv)
        expected = DataFrame({'A': df['A'].tolist() + [fv], 'B': df['B'].tolist() + [fv], 'C': [fv] * 4}, dtype=object)
        tm.assert_frame_equal(res, expected)
        res = df.reindex(index=range(4), fill_value=fv)
        tm.assert_frame_equal(res, expected[['A', 'B']])
        res = df.reindex(index=range(4), columns=['A', 'B', 'C'], fill_value='2016-01-01')
        expected = DataFrame({'A': df['A'].tolist() + [ts], 'B': df['B'].tolist() + [ts], 'C': [ts] * 4})
        tm.assert_frame_equal(res, expected)

    def test_reindex_with_multi_index(self) -> None:
        df = DataFrame({'a': [-1] * 7 + [0] * 7 + [1] * 7, 'b': list(range(7)) * 3, 'c': ['A', 'B', 'C', 'D', 'E', 'F', 'G'] * 3}).set_index(['a', 'b'])
        new_index = [0.5, 2.0, 5.0, 5.8]
        new_multi_index = MultiIndex.from_product([[0], new_index], names=['a', 'b'])
        reindexed = df.reindex(new_multi_index)
        expected = DataFrame({'a': [0] * 4, 'b': new_index, 'c': [np.nan, 'C', 'F', np.nan]}).set_index(['a', 'b'])
        tm.assert_frame_equal(expected, reindexed)
        expected = DataFrame({'a': [0] * 4, 'b': new_index, 'c': ['B', 'C', 'F', 'G']}).set_index(['a', 'b'])
        reindexed_with_backfilling = df.reindex(new_multi_index, method='bfill')
        tm.assert_frame_equal(expected, reindexed_with_backfilling)
        reindexed_with_backfilling = df.reindex(new_multi_index, method='backfill')
        tm.assert_frame_equal(expected, reindexed_with_backfilling)
        expected = DataFrame({'a': [0] * 4, 'b': new_index, 'c': ['A', 'C', 'F', 'F']}).set_index(['a', 'b'])
        reindexed_with_padding = df.reindex(new_multi_index, method='pad')
        tm.assert_frame_equal(expected, reindexed_with_padding)
        reindexed_with_padding = df.reindex(new_multi_index, method='ffill')
        tm.assert_frame_equal(expected, reindexed_with_padding)

    @pytest.mark.parametrize('method,expected_values', [('nearest', [0, 1, 1, 2]), ('pad', [np.nan, 0, 1, 1]), ('backfill', [0, 1, 2, 2])])
    def test_reindex_methods(self, method: str, expected_values: List[Union[int, float]]) -> None:
        df = DataFrame({'x': list(range(5))})
        target = np.array([-0.1, 0.9, 1.1, 1.5])
        expected = DataFrame({'x': expected_values}, index=target)
        actual = df.reindex(target, method=method)
        tm.assert_frame_equal(expected, actual)
        actual = df.reindex(target, method=method, tolerance=1)
        tm.assert_frame_equal(expected, actual)
        actual = df.reindex(target, method=method, tolerance=[1, 1, 1, 1])
        tm.assert_frame_equal(expected, actual)
        e2 = expected[::-1]
        actual = df.reindex(target[::-1], method=method)
        tm.assert_frame_equal(e2, actual)
        new_order = [3, 0, 2, 1]
        e2 = expected.iloc[new_order]
        actual = df.reindex(target[new_order], method=method)
        tm.assert_frame_equal(e2, actual)
        switched_method = 'pad' if method == 'backfill' else 'backfill' if method == 'pad' else method
        actual = df[::-1].reindex(target, method=switched_method)
        tm.assert_frame_equal(expected, actual)

    def test_reindex_methods_nearest_special(self) -> None:
        df = DataFrame({'x': list(range(5))})
        target = np.array([-0.1, 0.9, 1.1, 1.5])
        expected = DataFrame({'x': [0, 1, 1, np.nan]}, index=target)
        actual = df.reindex(target, method='nearest', tolerance=0.2)
        tm.assert_frame_equal(expected, actual)
        expected = DataFrame({'x': [0, np.nan, 1, np.nan]}, index=target)
        actual = df.reindex(target, method='nearest', tolerance=[0.5, 0.01, 0.4, 0.1])
        tm.assert_frame_equal(expected, actual)

    def test_reindex_nearest_tz(self, tz_aware_fixture: Any) -> None:
        tz = tz_aware_fixture
        idx = date_range('2019-01-01', periods=5, tz=tz)
        df = DataFrame({'x': list(range(5))}, index=idx)
        expected = df.head(3)
        actual = df.reindex(idx[:3], method='nearest')
        tm.assert_frame_equal(expected, actual)

    def test_reindex_nearest_tz_empty_frame(self) -> None:
        dti = pd.DatetimeIndex(['2016-06-26 14:27:26+00:00'])
        df = DataFrame(index=pd.DatetimeIndex(['2016-07-04 14:00:59+00:00']))
        expected = DataFrame(index=dti)
        result = df.reindex(dti, method='nearest')
        tm.assert_frame_equal(result, expected)

    def test_reindex_frame_add_nat(self) -> None:
        rng = date_range('1/1/2000 00:00:00', periods=10, freq='10s')
        df = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': rng})
        result = df.reindex(range(15))
        assert np.issubdtype(result['B'].dtype, np.dtype('M8[ns]'))
        mask = isna(result)['B']
        assert mask[-5:].all()
        assert not mask[:-5].any()

    @pytest.mark.parametrize('method, exp_values', [('ffill', [0, 1, 2, 3]), ('bfill', [1.0, 2.0, 3.0, np.nan])])
    def test_reindex_frame_tz_ffill_bfill(self, frame_or_series: Any, method: str, exp_values: List[Union[int, float]]) -> None:
        obj = frame_or_series([0, 1, 2, 3], index=date_range('2020-01-01 00:00:00', periods=4, freq='h', tz='UTC'))
        new_index = date_range('2020-01-01 00:01:00', periods=4, freq='h', tz='UTC')
        result = obj.reindex(new_index, method=method, tolerance=pd.Timedelta('1 hour'))
        expected = frame_or_series(exp_values, index=new_index)
        tm.assert_equal(result, expected)

    def test_reindex_limit(self) -> None:
        data = [['A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C', 'C'], ['D', 'D', 'D']]
        exp_data = [['A', 'A', 'A'], ['B', 'B', 'B'], ['C', 'C', 'C'], ['D', 'D', 'D'], ['D', 'D', 'D'], [np.nan, np.nan, np.nan]]
        df = DataFrame(data)
        result = df.reindex([0, 1, 2, 3, 4, 5], method='ffill', limit=1)
        expected = DataFrame(exp_data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('idx, check_index_type', [[['C', 'B', 'A'], True], [['F', 'C', 'A', 'D'], True], [['A'], True], [['A', 'B', 'C'], True], [['C', 'A', 'B'], True], [['C', 'B'], True], [['C', 'A'], True], [['A', 'B'], True], [['B', 'A', 'C'], True], [['D', 'F'], False], [['A', 'C', 'B'], False]])
    def test_reindex_level_verify_first_level(self, idx: List[str], check_index_type: bool) -> None:
        df = DataFrame({'jim': list('B' * 4 + 'A' * 2 + 'C' * 3), 'joe': list('abcdeabcd')[::-1], 'jolie': [10, 20, 30] * 3, 'joline': np.random.default_rng(2).integers(0, 1000, 9)})
        icol = ['jim', 'joe', 'jolie']

        def f(val: str) -> np.ndarray:
            return np.nonzero((df['jim'] == val).to_numpy())[0]
        i = np.concatenate(list(map(f, idx)))
        left = df.set_index(icol).reindex(idx, level='jim')
        right = df.iloc[i].set_index(icol)
        tm.assert_frame_equal(left, right, check_index_type=check_index_type)

    @pytest.mark.parametrize('idx', [('mid',), ('mid', 'btm'), ('mid', 'btm', 'top'), ('mid', 'top'), ('mid', 'top', 'btm'), ('btm',), ('btm', 'mid'), ('btm', 'mid', 'top'), ('btm', 'top'), ('btm', 'top', 'mid'), ('top',), ('top', 'mid'), ('top', 'mid', 'btm'), ('top', 'btm'), ('top', 'btm', 'mid')])
    def test_reindex_level_verify_first_level_repeats(self, idx: Tuple[str, ...]) -> None:
        df = DataFrame({'jim': ['mid'] * 5 + ['btm'] * 8 + ['top'] * 7, 'joe': ['3rd'] * 2 + ['1st'] * 3 + ['2nd'] * 3 + ['1st'] * 2 + ['3rd'] * 3 + ['1st'] * 2 + ['3rd'] * 3 + ['2nd'] * 2, 'jolie': np.concatenate([np.random.default_rng(2).choice(1000, x, replace=False) for x in [2, 3, 3, 2, 3, 2, 3, 2]]), 'joline': np.random.default_rng(2).standard_normal(20).round(3)