#!/usr/bin/env python3
"""
Tests for statistical reductions of 2nd moment or higher: var, skew, kurt, ...
"""

import inspect
from typing import Any, Callable
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Series, date_range
import pandas._testing as tm


class TestDatetimeLikeStatReductions:
    def test_dt64_mean(self, tz_naive_fixture: Any, index_or_series_or_array: Callable[[Any, Any], Any]) -> None:
        tz = tz_naive_fixture
        dti = date_range('2001-01-01', periods=11, tz=tz)
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
        dtarr = dti._data
        obj = index_or_series_or_array(dtarr)
        assert obj.mean() == pd.Timestamp('2001-01-06', tz=tz)
        assert obj.mean(skipna=False) == pd.Timestamp('2001-01-06', tz=tz)
        dtarr[-2] = pd.NaT
        obj = index_or_series_or_array(dtarr)
        assert obj.mean() == pd.Timestamp('2001-01-06 07:12:00', tz=tz)
        assert obj.mean(skipna=False) is pd.NaT

    @pytest.mark.parametrize('freq', ['s', 'h', 'D', 'W', 'B'])
    def test_period_mean(self, index_or_series_or_array: Callable[[Any, Any], Any], freq: str) -> None:
        dti = date_range('2001-01-01', periods=11)
        dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
        warn = FutureWarning if freq == 'B' else None
        msg = 'PeriodDtype\\[B\\] is deprecated'
        with tm.assert_produces_warning(warn, match=msg):
            parr = dti._data.to_period(freq)
        obj = index_or_series_or_array(parr)
        with pytest.raises(TypeError, match='ambiguous'):
            obj.mean()
        with pytest.raises(TypeError, match='ambiguous'):
            obj.mean(skipna=True)
        parr[-2] = pd.NaT
        with pytest.raises(TypeError, match='ambiguous'):
            obj.mean()
        with pytest.raises(TypeError, match='ambiguous'):
            obj.mean(skipna=True)

    def test_td64_mean(self, index_or_series_or_array: Callable[[Any, bool], Any]) -> None:
        m8values = np.array([0, 3, -2, -7, 1, 2, -1, 3, 5, -2, 4], 'm8[D]')
        tdi = pd.TimedeltaIndex(m8values).as_unit('ns')
        tdarr = tdi._data
        obj = index_or_series_or_array(tdarr, copy=False)
        result = obj.mean()
        expected = np.array(tdarr).mean()
        assert result == expected
        tdarr[0] = pd.NaT
        assert obj.mean(skipna=False) is pd.NaT
        result2 = obj.mean(skipna=True)
        assert result2 == tdi[1:].mean()
        assert result2.round('us') == (result * 11.0 / 10).round('us')


class TestSeriesStatReductions:
    def _check_stat_op(
        self,
        name: str,
        alternate: Callable[[Any], Any],
        string_series_: Series,
        check_objects: bool = False,
        check_allna: bool = False
    ) -> None:
        with pd.option_context('use_bottleneck', False):
            f = getattr(Series, name)
            string_series_[5:15] = np.nan
            if name not in ['max', 'min', 'mean', 'median', 'std']:
                ds = Series(date_range('1/1/2001', periods=10))
                msg = f"does not support operation '{name}'"
                with pytest.raises(TypeError, match=msg):
                    f(ds)
            assert pd.notna(f(string_series_))
            assert pd.isna(f(string_series_, skipna=False))
            nona = string_series_.dropna()
            tm.assert_almost_equal(f(nona), alternate(nona.values))
            tm.assert_almost_equal(f(string_series_), alternate(nona.values))
            allna = string_series_ * np.nan
            if check_allna:
                assert np.isnan(f(allna))
            s = Series([1, 2, 3, None, 5])
            f(s)
            items = [0]
            items.extend(range(2 ** 40, 2 ** 40 + 1000))
            s = Series(items, dtype='int64')
            tm.assert_almost_equal(float(f(s)), float(alternate(s.values)))
            if check_objects:
                s = Series(pd.bdate_range('1/1/2000', periods=10))
                res = f(s)
                exp = alternate(s)
                assert res == exp
            if name not in ['sum', 'min', 'max']:
                with pytest.raises(TypeError, match=None):
                    f(Series(list('abc')))
            msg = 'No axis named 1 for object type Series'
            with pytest.raises(ValueError, match=msg):
                f(string_series_, axis=1)
            if 'numeric_only' in inspect.getfullargspec(f).args:
                f(string_series_, numeric_only=True)

    def test_sum(self) -> None:
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('sum', np.sum, string_series, check_allna=False)

    def test_mean(self) -> None:
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('mean', np.mean, string_series)

    def test_median(self) -> None:
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('median', np.median, string_series)
        int_ts: Series = Series(np.ones(10, dtype=int), index=range(10))
        tm.assert_almost_equal(np.median(int_ts), int_ts.median())

    def test_prod(self) -> None:
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('prod', np.prod, string_series)

    def test_min(self) -> None:
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('min', np.min, string_series, check_objects=True)

    def test_max(self) -> None:
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        self._check_stat_op('max', np.max, string_series, check_objects=True)

    def test_var_std(self) -> None:
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        datetime_series: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        alt_std = lambda x: np.std(x, ddof=1)
        self._check_stat_op('std', alt_std, string_series)
        alt_var = lambda x: np.var(x, ddof=1)
        self._check_stat_op('var', alt_var, string_series)
        result = datetime_series.std(ddof=4)
        expected = np.std(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)
        result = datetime_series.var(ddof=4)
        expected = np.var(datetime_series.values, ddof=4)
        tm.assert_almost_equal(result, expected)
        s: Series = datetime_series.iloc[[0]]
        result = s.var(ddof=1)
        assert pd.isna(result)
        result = s.std(ddof=1)
        assert pd.isna(result)

    def test_sem(self) -> None:
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        datetime_series: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
        alt_sem = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
        self._check_stat_op('sem', alt_sem, string_series)
        result = datetime_series.sem(ddof=4)
        expected = np.std(datetime_series.values, ddof=4) / np.sqrt(len(datetime_series.values))
        tm.assert_almost_equal(result, expected)
        s: Series = datetime_series.iloc[[0]]
        result = s.sem(ddof=1)
        assert pd.isna(result)

    def test_skew(self) -> None:
        sp_stats = pytest.importorskip('scipy.stats')
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        alt_skew = lambda x: sp_stats.skew(x, bias=False)
        self._check_stat_op('skew', alt_skew, string_series)
        min_N: int = 3
        for i in range(1, min_N + 1):
            s: Series = Series(np.ones(i))
            df: DataFrame = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.skew())
                assert np.isnan(df.skew()).all()
            else:
                assert 0 == s.skew()
                assert isinstance(s.skew(), np.float64)
                assert (df.skew() == 0).all()

    def test_kurt(self) -> None:
        sp_stats = pytest.importorskip('scipy.stats')
        string_series: Series = Series(range(20), dtype=np.float64, name='series')
        alt_kurt = lambda x: sp_stats.kurtosis(x, bias=False)
        self._check_stat_op('kurt', alt_kurt, string_series)

    def test_kurt_corner(self) -> None:
        min_N: int = 4
        for i in range(1, min_N + 1):
            s: Series = Series(np.ones(i))
            df: DataFrame = DataFrame(np.ones((i, i)))
            if i < min_N:
                assert np.isnan(s.kurt())
                assert np.isnan(df.kurt()).all()
            else:
                assert 0 == s.kurt()
                assert isinstance(s.kurt(), np.float64)
                assert (df.kurt() == 0).all()