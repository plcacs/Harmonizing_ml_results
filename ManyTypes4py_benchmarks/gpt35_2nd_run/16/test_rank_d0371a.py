from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, NaT, Series, concat
import pandas._testing as tm

def test_rank_unordered_categorical_typeerror() -> None:
    cat: pd.Categorical = pd.Categorical([], ordered=False)
    ser: Series = Series(cat)
    df: DataFrame = ser.to_frame()
    msg: str = 'Cannot perform rank with non-ordered Categorical'
    gb: SeriesGroupBy = ser.groupby(cat, observed=False)
    with pytest.raises(TypeError, match=msg):
        gb.rank()
    gb2: DataFrameGroupBy = df.groupby(cat, observed=False)
    with pytest.raises(TypeError, match=msg):
        gb2.rank()

def test_rank_apply() -> None:
    lev1: np.ndarray = np.array(['a' * 10] * 100, dtype=object)
    lev2: np.ndarray = np.array(['b' * 10] * 130, dtype=object)
    lab1: np.ndarray = np.random.default_rng(2).integers(0, 100, size=500, dtype=int)
    lab2: np.ndarray = np.random.default_rng(2).integers(0, 130, size=500, dtype=int)
    df: DataFrame = DataFrame({'value': np.random.default_rng(2).standard_normal(500), 'key1': lev1.take(lab1), 'key2': lev2.take(lab2)})
    result: Series = df.groupby(['key1', 'key2']).value.rank()
    expected: DataFrame = [piece.value.rank() for key, piece in df.groupby(['key1', 'key2'])]
    expected = concat(expected, axis=0)
    expected = expected.reindex(result.index)
    tm.assert_series_equal(result, expected)
    result = df.groupby(['key1', 'key2']).value.rank(pct=True)
    expected = [piece.value.rank(pct=True) for key, piece in df.groupby(['key1', 'key2'])]
    expected = concat(expected, axis=0)
    expected = expected.reindex(result.index)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('grps', [['qux'], ['qux', 'quux']])
@pytest.mark.parametrize('vals', [np.array([2, 2, 8, 2, 6], dtype=dtype) for dtype in ['i8', 'i4', 'i2', 'i1', 'u8', 'u4', 'u2', 'u1', 'f8', 'f4', 'f2']] + [[pd.Timestamp('2018-01-02'), pd.Timestamp('2018-01-02'), pd.Timestamp('2018-01-08'), pd.Timestamp('2018-01-02'), pd.Timestamp('2018-01-06')], [pd.Timestamp('2018-01-02', tz='US/Pacific'), pd.Timestamp('2018-01-02', tz='US/Pacific'), pd.Timestamp('2018-01-08', tz='US/Pacific'), pd.Timestamp('2018-01-02', tz='US/Pacific'), pd.Timestamp('2018-01-06', tz='US/Pacific')], [pd.Timestamp('2018-01-02') - pd.Timestamp(0), pd.Timestamp('2018-01-02') - pd.Timestamp(0), pd.Timestamp('2018-01-08') - pd.Timestamp(0), pd.Timestamp('2018-01-02') - pd.Timestamp(0), pd.Timestamp('2018-01-06') - pd.Timestamp(0)], [pd.Timestamp('2018-01-02').to_period('D'), pd.Timestamp('2018-01-02').to_period('D'), pd.Timestamp('2018-01-08').to_period('D'), pd.Timestamp('2018-01-02').to_period('D'), pd.Timestamp('2018-01-06').to_period('D')]], ids=lambda x: type(x[0]))
@pytest.mark.parametrize('ties_method,ascending,pct,exp', [('average', True, False, [2.0, 2.0, 5.0, 2.0, 4.0]), ('average', True, True, [0.4, 0.4, 1.0, 0.4, 0.8]), ('average', False, False, [4.0, 4.0, 1.0, 4.0, 2.0]), ('average', False, True, [0.8, 0.8, 0.2, 0.8, 0.4]), ('min', True, False, [1.0, 1.0, 5.0, 1.0, 4.0]), ('min', True, True, [0.2, 0.2, 1.0, 0.2, 0.8]), ('min', False, False, [3.0, 3.0, 1.0, 3.0, 2.0]), ('min', False, True, [0.6, 0.6, 0.2, 0.6, 0.4]), ('max', True, False, [3.0, 3.0, 5.0, 3.0, 4.0]), ('max', True, True, [0.6, 0.6, 1.0, 0.6, 0.8]), ('max', False, False, [5.0, 5.0, 1.0, 5.0, 2.0]), ('max', False, True, [1.0, 1.0, 0.2, 1.0, 0.4]), ('first', True, False, [1.0, 2.0, 5.0, 3.0, 4.0]), ('first', True, True, [0.2, 0.4, 1.0, 0.6, 0.8]), ('first', False, False, [3.0, 4.0, 1.0, 5.0, 2.0]), ('first', False, True, [0.6, 0.8, 0.2, 1.0, 0.4]), ('dense', True, False, [1.0, 1.0, 3.0, 1.0, 2.0]), ('dense', True, True, [1.0 / 3.0, 1.0 / 3.0, 3.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]), ('dense', False, False, [3.0, 3.0, 1.0, 3.0, 2.0]), ('dense', False, True, [3.0 / 3.0, 3.0 / 3.0, 1.0 / 3.0, 3.0 / 3.0, 2.0 / 3.0])])
def test_rank_args(grps: List[str], vals: np.ndarray, ties_method: str, ascending: bool, pct: bool, exp: List[float]) -> None:
    key: np.ndarray = np.repeat(grps, len(vals))
    orig_vals: np.ndarray = vals
    vals = list(vals) * len(grps)
    if isinstance(orig_vals, np.ndarray):
        vals = np.array(vals, dtype=orig_vals.dtype)
    df: DataFrame = DataFrame({'key': key, 'val': vals})
    result: DataFrame = df.groupby('key').rank(method=ties_method, ascending=ascending, pct=pct)
    exp_df: DataFrame = DataFrame(exp * len(grps), columns=['val'])
    tm.assert_frame_equal(result, exp_df)

@pytest.mark.parametrize('grps', [['qux'], ['qux', 'quux']])
@pytest.mark.parametrize('vals', [[-np.inf, -np.inf, np.nan, 1.0, np.nan, np.inf, np.inf]])
@pytest.mark.parametrize('ties_method,ascending,na_option,exp', [('average', True, 'keep', [1.5, 1.5, np.nan, 3, np.nan, 4.5, 4.5]), ('average', True, 'top', [3.5, 3.5, 1.5, 5.0, 1.5, 6.5, 6.5]), ('average', True, 'bottom', [1.5, 1.5, 6.5, 3.0, 6.5, 4.5, 4.5]), ('average', False, 'keep', [4.5, 4.5, np.nan, 3, np.nan, 1.5, 1.5]), ('average', False, 'top', [6.5, 6.5, 1.5, 5.0, 1.5, 3.5, 3.5]), ('average', False, 'bottom', [4.5, 4.5, 6.5, 3.0, 6.5, 1.5, 1.5]), ('min', True, 'keep', [1.0, 1.0, np.nan, 3.0, np.nan, 4.0, 4.0]), ('min', True, 'top', [3.0, 3.0, 1.0, 5.0, 1.0, 6.0, 6.0]), ('min', True, 'bottom', [1.0, 1.0, 6.0, 3.0, 6.0, 4.0, 4.0]), ('min', False, 'keep', [4.0, 4.0, np.nan, 3.0, np.nan, 1.0, 1.0]), ('min', False, 'top', [6.0, 6.0, 1.0, 5.0, 1.0, 3.0, 3.0]), ('min', False, 'bottom', [4.0, 4.0, 6.0, 3.0, 6.0, 1.0, 1.0]), ('max', True, 'keep', [2.0, 2.0, np.nan, 3.0, np.nan, 5.0, 5.0]), ('max', True, 'top', [4.0, 4.0, 2.0, 5.0, 2.0, 7.0, 7.0]), ('max', True, 'bottom', [2.0, 2.0, 7.0, 3.0, 7.0, 5.0, 5.0]), ('max', False, 'keep', [5.0, 5.0, np.nan, 3.0, np.nan, 2.0, 2.0]), ('max', False, 'top', [7.0, 7.0, 2.0, 5.0, 2.0, 4.0, 4.0]), ('max', False, 'bottom', [5.0, 5.0, 7.0, 3.0, 7.0, 2.0, 2.0]), ('first', True, 'keep', [1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0]), ('first', True, 'top', [3.0, 4.0, 1.0, 5.0, 2.0, 6.0, 7.0]), ('first', True, 'bottom', [1.0, 2.0, 6.0, 3.0, 7.0, 4.0, 5.0]), ('first', False, 'keep', [4.0, 5.0, np.nan, 3.0, np.nan, 1.0, 2.0]), ('first', False, 'top', [6.0, 7.0, 1.0, 5.0, 2.0, 3.0, 4.0]), ('first', False, 'bottom', [4.0, 5.0, 6.0, 3.0, 7.0, 1.0, 2.0]), ('dense', True, 'keep', [1.0, 1.0, np.nan, 2.0, np.nan, 3.0, 3.0]), ('dense', True, 'top', [2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 4.0]), ('dense', True, 'bottom', [1.0, 1.0, 4.0, 2.0, 4.0, 3.0, 3.0]), ('dense', False, 'keep', [3.0, 3.0, np.nan, 2.0, np.nan, 1.0, 1.0]), ('dense', False, 'top', [4.0, 4.0, 1.0, 3.0, 1.0, 2.0, 2.0]), ('dense', False, 'bottom', [3.0, 3.0, 4.0, 2.0, 4.0, 1.0, 1.0])])
def test_infs_n_nans(grps: List[str], vals: List[float], ties_method: str, ascending: bool, na_option: str, exp: List[float]) -> None:
    key: np.ndarray = np.repeat(grps, len(vals))
    vals: List[float] = vals * len(grps)
    df: DataFrame = DataFrame({'key': key, 'val': vals})
    result: DataFrame = df.groupby('key').rank(method=ties_method, ascending=ascending, na_option=na_option)
    exp_df: DataFrame = DataFrame(exp * len(grps), columns=['val'])
    tm.assert_frame_equal(result, exp_df)

@pytest.mark.parametrize('grps', [['qux'], ['qux', 'quux']])
@pytest.mark.parametrize('vals', [np.array([2, 2, np.nan, 8, 2, 6, np.nan, np.nan], dtype=dtype) for dtype in ['f8', 'f4', 'f2']] + [[pd.Timestamp('2018-01-02'), pd.Timestamp('2018-01-02'), np.nan, pd.Timestamp('2018-01-08'), pd.Timestamp('2018-01-02'), pd.Timestamp('2018-01-06'), np.nan, np.nan], [pd.Timestamp('2018-01-02', tz='US/Pacific'), pd.Timestamp('2018-01-02', tz='US/Pacific'), np.nan, pd.Timestamp('2018-01-08', tz='US/Pacific'), pd.Timestamp('2018-01-02', tz='US/Pacific'), pd.Timestamp('2018-01-06', tz='US/Pacific'), np.nan, np.nan], [pd.Timestamp('2018-01-02') - pd.Timestamp(0), pd.Timestamp('2018-01-02') - pd.Timestamp(0), np.nan, pd.Timestamp('2018-01-08') - pd.Timestamp(0), pd.Timestamp('2018-01-02') - pd.Timestamp(0), pd.Timestamp('2018-01-06') - pd.Timestamp(0), np.nan, np.nan], [pd.Timestamp('2018-01-02').to_period('D'), pd.Timestamp('2018-01-02').to_period('D'), np.nan, pd.Timestamp('2018-01-08').to_period('D'), pd.Timestamp('2018-01-02').to_period('D'), pd.Timestamp('2018-01-06').to_period('D'), np.nan, np.nan]], ids=lambda x: type(x[0]))
@pytest.mark.parametrize('ties_method,ascending,na_option,pct,exp', [('average', True, 'keep', False, [2.0, 2.0, np.nan, 5.0, 2.0, 4.0, np.nan, np.nan]), ('average', True, 'keep', True, [0.4, 0.4, np.nan, 1.0, 0.4, 0.8, np.nan, np.nan]), ('average', False, 'keep', False, [4.0, 4.0, np.nan, 1.0, 4.0, 2.0, np.nan, np.nan]), ('average', False, 'keep', True, [0.8, 0.8, np.nan, 0.2, 0.8, 0.4, np.nan, np.nan]), ('min', True, 'keep', False, [1.0, 1.0, np.nan, 5.0, 1.0, 4.0, np.nan, np.nan]), ('min', True, 'keep', True, [0.2, 0.2, np.nan, 1.0, 0.2, 0.8, np.nan, np.nan]), ('min', False, 'keep', False, [3.0, 3.0, np.nan, 1.0, 3.0, 2.0, np.nan, np.nan]), ('min', False, 'keep', True, [0.6, 0.6, np.nan, 0.2, 0.6, 0.4, np.nan, np.nan]), ('max', True, 'keep', False, [3.0, 3.0, np.nan, 5.0, 3.0, 4.0, np.nan, np.nan]), ('max', True, 'keep', True, [0.6, 0.6, np.nan, 1.0, 0.6, 0.8, np.nan, np.nan]), ('max', False, 'keep', False, [5.0, 5.0, np.nan, 1.0, 5.0, 2.0, np.nan, np.nan]), ('max', False, 'keep', True, [1.0, 1.0, np.nan, 0