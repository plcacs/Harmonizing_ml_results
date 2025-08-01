from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import Series, date_range, period_range
import pandas._testing as tm
from typing import Optional, List

@pytest.mark.parametrize('first_slice,second_slice', [[[2, None], [None, -5]], [[None, 0], [None, -5]], [[None, -5], [None, 0]], [[None, 0], [None, 0]]])
@pytest.mark.parametrize('fill', [None, -1])
def test_align(datetime_series: Series, first_slice: List[Optional[int]], second_slice: List[Optional[int]], join_type: str, fill: Optional[int]) -> None:
    a: Series = datetime_series[slice(*first_slice)]
    b: Series = datetime_series[slice(*second_slice)]
    aa, ab = a.align(b, join=join_type, fill_value=fill)
    join_index = a.index.join(b.index, how=join_type)
    if fill is not None:
        diff_a = aa.index.difference(join_index)
        diff_b = ab.index.difference(join_index)
        if len(diff_a) > 0:
            assert (aa.reindex(diff_a) == fill).all()
        if len(diff_b) > 0:
            assert (ab.reindex(diff_b) == fill).all()
    ea = a.reindex(join_index)
    eb = b.reindex(join_index)
    if fill is not None:
        ea = ea.fillna(fill)
        eb = eb.fillna(fill)
    tm.assert_series_equal(aa, ea)
    tm.assert_series_equal(ab, eb)
    assert aa.name == 'ts'
    assert ea.name == 'ts'
    assert ab.name == 'ts'
    assert eb.name == 'ts'

def test_align_nocopy(datetime_series: Series) -> None:
    b: Series = datetime_series[:5].copy()
    a: Series = datetime_series.copy()
    ra, _ = a.align(b, join='left')
    ra[:5] = 5
    assert not (a[:5] == 5).any()
    a = datetime_series.copy()
    ra, _ = a.align(b, join='left')
    ra[:5] = 5
    assert not (a[:5] == 5).any()
    a = datetime_series.copy()
    b = datetime_series[:5].copy()
    _, rb = a.align(b, join='right')
    rb[:3] = 5
    assert not (b[:3] == 5).any()
    a = datetime_series.copy()
    b = datetime_series[:5].copy()
    _, rb = a.align(b, join='right')
    rb[:2] = 5
    assert not (b[:2] == 5).any()

def test_align_same_index(datetime_series: Series) -> None:
    a, b = datetime_series.align(datetime_series)
    assert a.index.is_(datetime_series.index)
    assert b.index.is_(datetime_series.index)
    a, b = datetime_series.align(datetime_series)
    assert a.index is not datetime_series.index
    assert b.index is not datetime_series.index
    assert a.index.is_(datetime_series.index)
    assert b.index.is_(datetime_series.index)

def test_align_multiindex() -> None:
    midx = pd.MultiIndex.from_product([range(2), range(3), range(2)], names=('a', 'b', 'c'))
    idx = pd.Index(range(2), name='b')
    s1: Series = Series(np.arange(12, dtype='int64'), index=midx)
    s2: Series = Series(np.arange(2, dtype='int64'), index=idx)
    res1l, res1r = s1.align(s2, join='left')
    res2l, res2r = s2.align(s1, join='right')
    expl: Series = s1
    tm.assert_series_equal(expl, res1l)
    tm.assert_series_equal(expl, res2r)
    expr: Series = Series([0, 0, 1, 1, np.nan, np.nan] * 2, index=midx)
    tm.assert_series_equal(expr, res1r)
    tm.assert_series_equal(expr, res2l)
    res1l, res1r = s1.align(s2, join='right')
    res2l, res2r = s2.align(s1, join='left')
    exp_idx = pd.MultiIndex.from_product([range(2), range(2), range(2)], names=('a', 'b', 'c'))
    expl = Series([0, 1, 2, 3, 6, 7, 8, 9], index=exp_idx)
    tm.assert_series_equal(expl, res1l)
    tm.assert_series_equal(expl, res2r)
    expr = Series([0, 0, 1, 1] * 2, index=exp_idx)
    tm.assert_series_equal(expr, res1r)
    tm.assert_series_equal(expr, res2l)

def test_align_dt64tzindex_mismatched_tzs() -> None:
    idx1 = date_range('2001', periods=5, freq='h', tz='US/Eastern')
    ser: Series = Series(np.random.default_rng(2).standard_normal(len(idx1)), index=idx1)
    ser_central: Series = ser.tz_convert('US/Central')
    new1, new2 = ser.align(ser_central)
    assert new1.index.tz is timezone.utc
    assert new2.index.tz is timezone.utc

def test_align_periodindex(join_type: str) -> None:
    rng = period_range('1/1/2000', '1/1/2010', freq='Y')
    ts: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    ts.align(ts[::2], join=join_type)

def test_align_stringindex(any_string_dtype: str) -> None:
    left: Series = Series(range(3), index=pd.Index(['a', 'b', 'd'], dtype=any_string_dtype))
    right: Series = Series(range(3), index=pd.Index(['a', 'b', 'c'], dtype=any_string_dtype))
    result_left, result_right = left.align(right)
    expected_idx = pd.Index(['a', 'b', 'c', 'd'], dtype=any_string_dtype)
    expected_left: Series = Series([0, 1, np.nan, 2], index=expected_idx)
    expected_right: Series = Series([0, 1, 2, np.nan], index=expected_idx)
    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)

def test_align_left_fewer_levels() -> None:
    left: Series = Series([2], index=pd.MultiIndex.from_tuples([(1, 3)], names=['a', 'c']))
    right: Series = Series([1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=['a', 'b', 'c']))
    result_left, result_right = left.align(right)
    expected_right: Series = Series([1], index=pd.MultiIndex.from_tuples([(1, 3, 2)], names=['a', 'c', 'b']))
    expected_left: Series = Series([2], index=pd.MultiIndex.from_tuples([(1, 3, 2)], names=['a', 'c', 'b']))
    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)

def test_align_left_different_named_levels() -> None:
    left: Series = Series([2], index=pd.MultiIndex.from_tuples([(1, 4, 3)], names=['a', 'd', 'c']))
    right: Series = Series([1], index=pd.MultiIndex.from_tuples([(1, 2, 3)], names=['a', 'b', 'c']))
    result_left, result_right = left.align(right)
    expected_left: Series = Series([2], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=['a', 'd', 'c', 'b']))
    expected_right: Series = Series([1], index=pd.MultiIndex.from_tuples([(1, 4, 3, 2)], names=['a', 'd', 'c', 'b']))
    tm.assert_series_equal(result_left, expected_left)
    tm.assert_series_equal(result_right, expected_right)