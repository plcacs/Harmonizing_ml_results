import builtins
import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, Timestamp, date_range, isna
import pandas._testing as tm
from pandas.util import _test_decorators as td
from typing import Any, Callable, Dict, List, Tuple, Union

@pytest.mark.parametrize('dtype', ['int64', 'int32', 'float64', 'float32'])
def func_6yjt6r1u(dtype: str) -> None:
    data: Series = Series(np.arange(9) // 3, index=np.arange(9), dtype=dtype)
    index: np.ndarray = np.arange(9)
    np.random.default_rng(2).shuffle(index)
    data = data.reindex(index)
    grouped = data.groupby(lambda x: x // 3, group_keys=False)
    for k, v in grouped:
        assert len(v) == 3
    agged: Series = grouped.aggregate(np.mean)
    assert agged[1] == 1
    expected: Series = grouped.agg(np.mean)
    tm.assert_series_equal(agged, expected)
    tm.assert_series_equal(agged, grouped.mean())
    result: Series = grouped.sum()
    expected = grouped.agg(np.sum)
    if dtype == 'int32':
        expected = expected.astype('int32')
    tm.assert_series_equal(result, expected)
    expected = grouped.apply(lambda x: x * x.sum())
    transformed: Series = grouped.transform(lambda x: x * x.sum())
    assert transformed[7] == 12
    tm.assert_series_equal(transformed, expected)
    value_grouped = data.groupby(data)
    result = value_grouped.aggregate(np.mean)
    tm.assert_series_equal(result, agged, check_index_type=False)
    agged = grouped.aggregate([np.mean, np.std])
    msg = 'nested renamer is not supported'
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        grouped.aggregate({'one': np.mean, 'two': np.std})
    msg = 'Must produce aggregated value'
    with pytest.raises(Exception, match=msg):
        grouped.aggregate(lambda x: x * 2)

@pytest.mark.parametrize('vals', [['foo', 'bar', 'baz'], ['foo', '', ''], [
    '', '', ''], [1, 2, 3], [1, 0, 0], [0, 0, 0], [1.0, 2.0, 3.0], [1.0, 
    0.0, 0.0], [0.0, 0.0, 0.0], [True, True, True], [True, False, False], [
    False, False, False], [np.nan, np.nan, np.nan]])
def func_oxh3uhuw(skipna: bool, all_boolean_reductions: str, vals: List[Any]) -> None:
    df: DataFrame = DataFrame({'key': ['a'] * 3 + ['b'] * 3, 'val': vals * 2})
    exp: Any = getattr(builtins, all_boolean_reductions)(vals)
    if skipna and all(isna(v) for v in vals) and all_boolean_reductions == 'any':
        exp = False
    expected: DataFrame = DataFrame([exp] * 2, columns=['val'], index=pd.Index(['a', 'b'], name='key'))
    result = getattr(df.groupby('key'), all_boolean_reductions)(skipna=skipna)
    tm.assert_frame_equal(result, expected)

def func_3tcu3nlg() -> None:
    df: DataFrame = DataFrame([[1, 2, 'foo'], [1, np.nan, 'bar'], [3, np.nan, 'baz']],
        columns=['A', 'B', 'C'])
    expected: DataFrame = DataFrame([[True, True], [False, True]], columns=['B', 'C'],
        index=[1, 3])
    expected.index.name = 'A'
    result: DataFrame = df.groupby('A').any()
    tm.assert_frame_equal(result, expected)

def func_wk8vudxn(all_boolean_reductions: str) -> None:
    df: DataFrame = DataFrame([[True, True]], columns=['a', 'a'])
    grp_by = df.groupby([0])
    result = getattr(grp_by, all_boolean_reductions)()
    expected: DataFrame = df.set_axis(np.array([0]))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data', [[False, False, False], [True, True, True],
    [pd.NA, pd.NA, pd.NA], [False, pd.NA, False], [True, pd.NA, True], [
    True, pd.NA, False]])
def func_euzechig(all_boolean_reductions: str, skipna: bool, data: List[Any]) -> None:
    ser: Series = Series(data, dtype='boolean')
    expected_data: Any = getattr(ser, all_boolean_reductions)(skipna=skipna)
    expected: Series = Series(expected_data, index=np.array([0]), dtype='boolean')
    result: Series = ser.groupby([0, 0, 0]).agg(all_boolean_reductions, skipna=skipna)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('dtype1,dtype2,exp_col1,exp_col2', [('float',
    'Float64', np.array([True], dtype=bool), pd.array([pd.NA], dtype=
    'boolean')), ('Int64', 'float', pd.array([pd.NA], dtype='boolean'), np.
    array([True], dtype=bool)), ('Int64', 'Int64', pd.array([pd.NA], dtype=
    'boolean'), pd.array([pd.NA], dtype='boolean')), ('Float64', 'boolean',
    pd.array([pd.NA], dtype='boolean'), pd.array([pd.NA], dtype='boolean'))])
def func_ke9rcru2(dtype1: str, dtype2: str, exp_col1: np.ndarray, exp_col2: pd.ArrayLike) -> None:
    data: List[Union[float, int]] = [1.0, np.nan]
    df: DataFrame = DataFrame({'col1': pd.array(data, dtype=dtype1), 'col2': pd.array(
        data, dtype=dtype2)})
    result = df.groupby([1, 1]).agg('all', skipna=False)
    expected: DataFrame = DataFrame({'col1': exp_col1, 'col2': exp_col2}, index=pd.
        array([1]))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'boolean'])
def func_vgt5qevt(all_boolean_reductions: str, dtype: str, skipna: bool, frame_or_series: Callable[[List[Any]], Union[DataFrame, Series]]) -> None:
    obj: Union[DataFrame, Series] = frame_or_series([pd.NA, 1], dtype=dtype)
    expected_res: Union[bool, Any] = True
    if not skipna and all_boolean_reductions == 'all':
        expected_res = pd.NA
    expected: Union[DataFrame, Series] = frame_or_series([expected_res], index=np.array([1]), dtype=
        'boolean')
    result = obj.groupby([1, 1]).agg(all_boolean_reductions, skipna=skipna)
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('bool_agg_func,data,expected_res', [('any', [pd.NA,
    np.nan], False), ('any', [pd.NA, 1, np.nan], True), ('all', [pd.NA, pd.
    NaT], True), ('all', [pd.NA, False, pd.NaT], False)])
def func_dyo3dw8q(bool_agg_func: str, data: List[Any], expected_res: bool, frame_or_series: Callable[[List[Any]], Union[DataFrame, Series]]) -> None:
    obj: Union[DataFrame, Series] = frame_or_series(data, dtype=object)
    result = obj.groupby([1] * len(data)).agg(bool_agg_func)
    expected: Union[DataFrame, Series] = frame_or_series([expected_res], index=np.array([1]), dtype=
        'bool')
    tm.assert_equal(result, expected)

def func_75n1glmh(all_boolean_reductions: str) -> None:
    ser: Series = Series([pd.NA], dtype=object)
    with pytest.raises(TypeError, match='boolean value of NA is ambiguous'):
        ser.groupby([1]).agg(all_boolean_reductions, skipna=False)

def func_8yzjqni6(frame_or_series: Callable[..., Union[DataFrame, Series]], all_boolean_reductions: str) -> None:
    kwargs: Dict[str, List[str]] = {'columns': ['a']} if frame_or_series is DataFrame else {'name': 'a'}
    obj: Union[DataFrame, Series] = frame_or_series(**kwargs, dtype=object)
    result: Union[DataFrame, Series] = getattr(obj.groupby(obj.index), all_boolean_reductions)()
    expected: Union[DataFrame, Series] = frame_or_series(**kwargs, dtype=bool)
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('how', ['idxmin', 'idxmax'])
def func_2h7geiyv(how: str, any_real_numpy_dtype: type) -> None:
    if any_real_numpy_dtype is int or any_real_numpy_dtype is float:
        return
    info = np.iinfo if 'int' in any_real_numpy_dtype.__name__ else np.finfo
    min_value: Union[int, float] = info(any_real_numpy_dtype).min
    max_value: Union[int, float] = info(any_real_numpy_dtype).max
    df: DataFrame = DataFrame({'a': [2, 1, 1, 2], 'b': [min_value, max_value,
        max_value, min_value]}, dtype=any_real_numpy_dtype)
    gb = df.groupby('a')
    result: DataFrame = getattr(gb, how)()
    expected: DataFrame = DataFrame({'b': [1, 0]}, index=pd.Index([1, 2], name='a',
        dtype=any_real_numpy_dtype))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('how', ['idxmin', 'idxmax'])
def func_foywifcb(skipna: bool, how: str, float_numpy_dtype: type) -> None:
    min_value: float = np.finfo(float_numpy_dtype).min
    max_value: float = np.finfo(float_numpy_dtype).max
    df: DataFrame = DataFrame({'a': Series(np.repeat(range(1, 6), repeats=2), dtype='intp'), 'b': Series([np.nan, min_value, np.nan, max_value,
        min_value, np.nan, max_value, np.nan, np.nan, np.nan], dtype=
        float_numpy_dtype)})
    gb = df.groupby('a')
    if not skipna:
        msg = f'DataFrameGroupBy.{how} with skipna=False'
        with pytest.raises(ValueError, match=msg):
            getattr(gb, how)(skipna=skipna)
        return
    result: DataFrame = getattr(gb, how)(skipna=skipna)
    expected: DataFrame = DataFrame({'b': [1, 3, 4, 6, np.nan]}, index=pd.Index(range(
        1, 6), name='a', dtype='intp'))
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func, values', [('idxmin', {'c_int': [0, 2],
    'c_float': [1, 3], 'c_date': [1, 2]}), ('idxmax', {'c_int': [1, 3],
    'c_float': [0, 2], 'c_date': [0, 3]})])
@pytest.mark.parametrize('numeric_only', [True, False])
def func_n7ql243w(func: str, values: Dict[str, List[Any]], numeric_only: bool) -> None:
    df: DataFrame = DataFrame({'name': ['A', 'A', 'B', 'B'], 'c_int': [1, 2, 3, 4],
        'c_float': [4.02, 3.03, 2.04, 1.05], 'c_date': ['2019', '2018',
        '2016', '2017']})
    df['c_date'] = pd.to_datetime(df['c_date'])
    df['c_date_tz'] = df['c_date'].dt.tz_localize('US/Pacific')
    df['c_timedelta'] = df['c_date'] - df['c_date'].iloc[0]
    df['c_period'] = df['c_date'].dt.to_period('W')
    df['c_Integer'] = df['c_int'].astype('Int64')
    df['c_Floating'] = df['c_float'].astype('Float64')
    result: DataFrame = getattr(df.groupby('name'), func)(numeric_only=numeric_only)
    expected: DataFrame = DataFrame(values, index=pd.Index(['A', 'B'], name='name'))
    if numeric_only:
        expected = expected.drop(columns=['c_date'])
    else:
        expected['c_date_tz'] = expected['c_date']
        expected['c_timedelta'] = expected['c_date']
        expected['c_period'] = expected['c_date']
    expected['c_Integer'] = expected['c_int']
    expected['c_Floating'] = expected['c_float']
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('data', [[0, 1, np.nan, 3,
    4, 5, 6, 7, 8, 9], 'float64', 'float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7,
    8, 9], 'Float64', 'Float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9],
    'Int64', 'Float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9],
    'timedelta64[ns]', 'timedelta64[ns]'), (pd.to_datetime(['2019-05-09',
    pd.NaT, '2019-05-11', '2019-05-12', '2019-05-13', '2019-05-14',
    '2019-05-15', '2019-05-16', '2019-05-17', '2019-05-18']),
    'datetime64[ns]', 'datetime64[ns]')])
def func_g9yxd604(values: List[Any], dtype: str, result_dtype: str, skipna: bool) -> None:
    df: DataFrame = DataFrame({'val': values, 'cat': ['A', 'B'] * 5}).astype({'val': dtype})
    expected: Series = df.groupby('cat')['val'].apply(lambda x: x.mean(skipna=skipna)).astype(result_dtype)
    result: Series = df.groupby('cat')['val'].mean(skipna=skipna)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('values, dtype', [([0, 1, np.nan, 3, 4, 5, 6, 7, 8,
    9], 'float64'), ([0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'Float64'), ([0, 
    1, np.nan, 3, 4, 5, 6, 7, 8, 9], 'Int64'), ([0, 1, np.nan, 3, 4, 5, 6, 
    7, 8, 9], 'timedelta64[ns]')])
def func_n5ja9x46(values: List[Any], dtype: str, skipna: bool) -> None:
    df: DataFrame = DataFrame({'val': values, 'cat': ['A', 'B'] * 5}).astype({'val': dtype})
    expected: Series = df.groupby('cat')['val'].apply(lambda x: x.sum(skipna=skipna)).astype(dtype)
    result: Series = df.groupby('cat')['val'].sum(skipna=skipna)
    tm.assert_series_equal(result, expected)

def func_ym15q8rh(skipna: bool) -> None:
    df: DataFrame = DataFrame({'val': ['a', 'b', np.nan, 'd', 'e', 'f', 'g', 'h', 'i',
        'j'], 'cat': ['A', 'B'] * 5}).astype({'val': object})
    if skipna:
        expected: Series = Series(['aegi', 'bdfhj'], index=pd.Index(['A', 'B'],
            name='cat'), name='val').astype(object)
    else:
        expected = Series([np.nan, 'bdfhj'], index=pd.Index(['A', 'B'],
            name='cat'), name='val').astype(object)
    result: Series = df.groupby('cat')['val'].sum(skipna=skipna)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func, values, dtype, result_dtype', [('prod', [0,
    1, 3, np.nan, 4, 5, 6, 7, -8, 9], 'float64', 'float64'), ('prod', [0, -
    1, 3, 4, 5, np.nan, 6, 7, 8, 9], 'Float64', 'Float64'), ('prod', [0, 1,
    3, -4, 5, 6, 7, -8, np.nan, 9], 'Int64', 'Int64'), ('prod', [np.nan] * 
    10, 'float64', 'float64'), ('prod', [np.nan] * 10, 'Float64', 'Float64'
    ), ('prod', [np.nan] * 10, 'Int64', 'Int64'), ('var', [0, -1, 3, 4, np.
    nan, 5, 6, 7, 8, 9], 'float64', 'float64'), ('var', [0, 1, 3, -4, 5, 6,
    7, -8, 9, np.nan], 'Float64', 'Float64'), ('var', [0, -1, 3, 4, 5, -6, 
    7, np.nan, 8, 9], 'Int64', 'Float64'), ('var', [np.nan] * 10, 'float64',
    'float64'), ('var', [np.nan] * 10, 'Float64', 'Float64'), ('var', [np.
    nan] * 10, 'Int64', 'Float64'), ('std', [0, 1, 3, -4, 5, 6, 7, -8, np.
    nan, 9], 'float64', 'float64'), ('std', [0, -1, 3, 4, 5, -6, 7, np.nan,
    8, 9], 'Float64', 'Float64'), ('std', [0, 1, 3, -4, 5, 6, 7, -8, 9, np.
    nan], 'Int64', 'Float64'), ('std', [np.nan] * 10, 'float64', 'float64'),
    ('std', [np.nan] * 10, 'Float64', 'Float64'), ('std', [np.nan] * 10,
    'Int64', 'Float64'), ('sem', [0, -1, 3, 4, 5, -6, 7, np.nan, 8, 9],
    'float64', 'float64'), ('sem', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9],
    'Float64', 'Float64'), ('sem', [0, -1, 3, 4, 5, -6, 7, 8, 9, np.nan],
    'Int64', 'Float64'), ('sem', [np.nan] * 10, 'float64', 'float64'), (
    'sem', [np.nan] * 10, 'Float64', 'Float64'), ('sem', [np.nan] * 10,
    'Int64', 'Float64'), ('min', [0, -1, 3, 4, 5, -6, 7, np.nan, 8, 9],
    'float64', 'float64'), ('min', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9],
    'Float64', 'Float64'), ('min', [0, -1, 3, 4, 5, -6, 7, 8, 9, np.nan],
    'Int64', 'Int64'), ('min', [0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9],
    'timedelta64[ns]', 'timedelta64[ns]'), ('min', pd.to_datetime([
    '2019-05-09', pd.NaT, '2019-05-11', '2019-05-12', '2019-05-13',
    '2019-05-14', '2019-05-15', '2019-05-16', '2019-05-17', '2019-05-18']),
    'datetime64[ns]', 'datetime64[ns]'), ('min', [np.nan] * 10, 'float64',
    'float64'), ('min', [np.nan] * 10, 'Float64', 'Float64'), ('min', [np.
    nan] * 10, 'Int64', 'Int64'), ('max', [0, -1, 3, 4, 5, -6, 7, np.nan, 8,
    9], 'float64', 'float64'), ('max', [0, 1, 3, -4, 5, 6, 7, -8, np.nan, 9
    ], 'Float64', 'Float64'), ('max', [0, -1, 3, 4, 5, -6, 7, 8, 9, np.nan],
    'Int64', 'Int64'), ('max', [0, 1, np.nan, 3, 4, 5, 6, 7, 8, 9],
    'timedelta64[ns]', 'timedelta64[ns]'), ('max', pd.to_datetime([
    '2019-05-09', pd.NaT, '2019-05-11', '2019-05-12', '2019-05-13',
    '2019-05-14', '2019-05-15', '2019-05-16', '2019-05-17', '2019-05-18']),
    'datetime64[ns]', 'datetime64[ns]'), ('max', [np.nan] * 10, 'float64',
    'float64'), ('max', [np.nan] * 10, 'Float64', 'Float64'), ('max', [np.
    nan] * 10, 'Int64', 'Int64'), ('median', [0, -1, 3, 4, 5, -6, 7, np.nan,
    8, 9], 'float64', 'float64'), ('median', [0, 1, 3, -4, 5, 6, 7, -8, np.
    nan, 9], 'Float64', 'Float64'), ('median', [0, -1, 3, 4, 5, -6, 7, 8, 9,
    np.nan], 'Int64', 'Float64'), ('median', [0, 1, np.nan, 3, 4, 5, 6, 7, 
    8, 9], 'timedelta64[ns]', 'timedelta64[ns]'), ('median', pd.to_datetime
    (['2019-05-09', pd.NaT, '2019-05-11', '2019-05-12', '2019-05-13',
    '2019-05-14', '2019-05-15', '2019-05-16', '2019-05-17', '2019-05-18']),
    'datetime64[ns]', 'datetime64[ns]'), ('median', [np.nan] * 10,
    'float64', 'float64'), ('median', [np.nan] * 10, 'Float64', 'Float64'),
    ('median', [np.nan] * 10, 'Int64', 'Float64')])
def func_dlicr02t(func: str, values: List[Any], dtype: str, result_dtype: str, skipna: bool) -> None:
    df: DataFrame = DataFrame({'val': values, 'cat': ['A', 'B'] * 5}).astype({'val': dtype})
    expected: Series = df.groupby('cat')['val'].apply(lambda x: getattr(x, func)(
        skipna=skipna)).astype(result_dtype)
    result: Series = getattr(df.groupby('cat')['val'], func)(skipna=skipna)
    tm.assert_series_equal(result, expected)

def func_761pc611() -> None:
    arr: np.ndarray = np.random.default_rng(2).standard_normal(1000)
    arr[::2] = np.nan
    df: DataFrame = DataFrame(arr)
    labels: np.ndarray = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    labels[::17] = np.nan
    result: DataFrame = df.groupby(labels).median()
    exp: DataFrame = df.groupby(labels).agg(np.nanmedian)
    tm.assert_frame_equal(result, exp)
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 5)))
    rs: DataFrame = df.groupby(labels).agg(np.median)
    xp: DataFrame = df.groupby(labels).median()
    tm.assert_frame_equal(rs, xp)

def func_4dgx9iqk(observed: bool) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).integers(0, 44, 500))
    grps: List[int] = list(range(0, 55, 5))
    bins: pd.IntervalIndex = pd.cut(df[0], grps)
    result: DataFrame = df.groupby(bins, observed=observed).median()
    expected: DataFrame = df.groupby(bins, observed=observed).agg(lambda x: x.median())
    tm.assert_frame_equal(result, expected)

def func_ahx4o83z() -> None:
    aa: DataFrame = DataFrame({'nn': [11, 11, 22, 22], 'ii': [1, 2, 3, 4], 'ss': 4 * [
        'mama']})
    result: DataFrame = aa.groupby('nn').max()
    assert 'ss' in result
    result = aa.groupby('nn').max(numeric_only=False)
    assert 'ss' in result
    result = aa.groupby('nn').min()
    assert 'ss' in result
    result = aa.groupby('nn').min(numeric_only=False)
    assert 'ss' in result

def func_gzrf0mzv(using_infer_string: bool) -> None:
    df: DataFrame = DataFrame({'A': [1, 1, 2, 2, 3], 'B': [1, 'foo', 2, 'bar', False],
        'C': ['a', 'b', 'c', 'd', 'e']})
    df._consolidate_inplace()
    assert len(df._mgr.blocks) == 3 if using_infer_string else 2
    gb = df.groupby('A')
    result: DataFrame = gb[['C']].max()
    ei: pd.Index = pd.Index([1, 2, 3], name='A')
    expected: DataFrame = DataFrame({'C': ['b', 'd', 'e']}, index=ei)
    tm.assert_frame_equal(result, expected)
    result = gb[['C']].min()
    ei = pd.Index([1, 2, 3], name='A')
    expected = DataFrame({'C': ['a', 'c', 'e']}, index=ei)
    tm.assert_frame_equal(result, expected)

def func_3gim2mvq() -> None:
    dates: Series = pd.to_datetime(Series(['2019-05-09', '2019-05-09', '2019-05-09'
        ]), format='%Y-%m-%d').dt.date
    df: DataFrame = DataFrame({'a': [np.nan, '1', np.nan], 'b': [0, 1, 1], 'c': dates})
    result: Series = df.groupby('b', as_index=False)['c'].min()['c']
    expected: Series = pd.to_datetime(Series(['2019-05-09', '2019-05-09'], name='c'
        ), format='%Y-%m-%d').dt.date
    tm.assert_series_equal(result, expected)
    result = df.groupby('b')['c'].min()
    expected.index.name = 'b'
    tm.assert_series_equal(result, expected)

def func_zwn0o0cg() -> None:
    ser: Series = Series([1, iNaT])
    key: np.ndarray = np.array([1, 1], dtype=np.int64)
    gb = ser.groupby(key)
    result: Series = gb.max(min_count=2)
    expected: Series = Series({(1): 1}, dtype=np.int64)
    tm.assert_series_equal(result, expected, check_exact=True)
    result = gb.min(min_count=2)
    expected = Series({(1): iNaT}, dtype=np.int64)
    tm.assert_series_equal(result, expected, check_exact=True)
    result = gb.min(min_count=3)
    expected = Series({(1): np.nan})
    tm.assert_series_equal(result, expected, check_exact=True)

def func_2t8qgv55() -> None:
    ser: Series = Series([1, iNaT, 2, iNaT + 1])
    gb = ser.groupby([1, 2, 3, 3])
    result: Series = gb.min(min_count=2)
    expected: Series = Series({(1): np.nan, (2): np.nan, (3): iNaT + 1})
    expected.index = expected.index.astype(int)
    tm.assert_series_equal(result, expected, check_exact=True)

@pytest.mark.parametrize('func', ['min', 'max'])
def func_6uigkjrk(func: str) -> None:
    groups: List[int] = [1, 2]
    periods: pd.PeriodIndex = pd.period_range('2020', periods=2, freq='Y')
    df: DataFrame = DataFrame({'a': groups, 'b': periods})
    result = getattr(df.groupby('a')['b'], func)()
    idx: pd.Index = pd.Index([1, 2], name='a')
    expected: Series = Series(periods, index=idx, name='b')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func', ['min', 'max'])
def func_f7cey4du(func: str) -> None:
    groups: List[int] = [1, 2]
    periods: pd.PeriodIndex = pd.period_range('2020', periods=2, freq='Y')
    df: DataFrame = DataFrame({'a': groups, 'b': periods})
    result: DataFrame = getattr(df.groupby('a'), func)()
    idx: pd.Index = pd.Index([1, 2], name='a')
    expected: DataFrame = DataFrame({'b': periods}, index=idx)
    tm.assert_frame_equal(result, expected)

def func_g7tfwyjo() -> None:
    df: DataFrame = DataFrame({'key': ['A', 'A', 'B', 'B'], 'col1': list('abcd'),
        'col2': [np.nan] * 4}).astype(object)
    result: DataFrame = df.groupby('key').min()
    expected: DataFrame = DataFrame({'key': ['A', 'B'], 'col1': ['a', 'c'], 'col2': [
        np.nan, np.nan]}).set_index('key').astype(object)
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'key': ['A', 'A', 'B', 'B'], 'col1': list('abcd'),
        'col2': range(4)}).astype(object)
    result = df.groupby('key').min()
    expected = DataFrame({'key': ['A', 'B'], 'col1': ['a', 'c'], 'col2': [0,
        2]}).set_index('key').astype(object)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func', ['min', 'max'])
def func_tt49c7w4(func: str) -> None:
    ds: pd.Categorical = Series(['b'], dtype='category').cat.as_ordered()
    df: DataFrame = DataFrame({'A': [1997], 'B': ds})
    result: DataFrame = df.groupby('A').agg({'B': func})
    expected: DataFrame = DataFrame({'B': ['b']}, index=pd.Index([1997], name='A'))
    expected['B'] = expected['B'].astype(ds.dtype)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('dtype', ['Int64', 'Int32', 'Float64', 'Float32',
    'boolean'])
def func_5nvvodzx(dtype: str) -> None:
    if dtype == 'Int64':
        ts: Union[int, Any] = 1618556707013635762
    elif dtype == 'boolean':
        ts = 0
    else:
        ts = 4.0
    df: DataFrame = DataFrame({'id': [2, 2], 'ts': [ts, ts + 1]})
    df['ts'] = df['ts'].astype(dtype)
    gb = df.groupby('id')
    result: DataFrame = gb.min()
    expected: DataFrame = df.iloc[:1].set_index('id')
    tm.assert_frame_equal(result, expected)
    res_max: DataFrame = gb.max()
    expected_max: DataFrame = df.iloc[1:].set_index('id')
    tm.assert_frame_equal(res_max, expected_max)
    result2: DataFrame = gb.min(min_count=3)
    expected2: DataFrame = DataFrame({'ts': [pd.NA]}, index=expected.index, dtype=dtype)
    tm.assert_frame_equal(result2, expected2)
    res_max2: DataFrame = gb.max(min_count=3)
    tm.assert_frame_equal(res_max2, expected2)
    df2: DataFrame = DataFrame({'id': [2, 2, 2], 'ts': [ts, pd.NA, ts + 1]})
    df2['ts'] = df2['ts'].astype(dtype)
    gb2 = df2.groupby('id')
    result3: DataFrame = gb2.min()
    tm.assert_frame_equal(result3, expected)
    res_max3: DataFrame = gb2.max()
    tm.assert_frame_equal(res_max3, expected_max)
    result4: DataFrame = gb2.min(min_count=100)
    tm.assert_frame_equal(result4, expected2)
    res_max4: DataFrame = gb2.max(min_count=100)
    tm.assert_frame_equal(res_max4, expected2)

def func_2bz9t391() -> None:
    cat: pd.Categorical = pd.Categorical([0] * 10, categories=[0, 1])
    df: DataFrame = DataFrame({'A': cat, 'B': pd.array(np.arange(10, dtype=np.uint64))})
    gb = df.groupby('A', observed=False)
    res: DataFrame = gb.min()
    idx: pd.CategoricalIndex = pd.CategoricalIndex([0, 1], dtype=cat.dtype, name='A')
    expected: DataFrame = DataFrame({'B': pd.array([0, pd.NA], dtype='UInt64')}, index=idx)
    tm.assert_frame_equal(res, expected)
    res = gb.max()
    expected.iloc[0, 0] = 9
    tm.assert_frame_equal(res, expected)

@pytest.mark.parametrize('func', ['first', 'last', 'min', 'max'])
def func_egoo5h4q(func: str) -> None:
    df: DataFrame = DataFrame({'col1': pd.Categorical(['A'], categories=list('AB'),
        ordered=True), 'col2': pd.Categorical([1], categories=[1, 2],
        ordered=True), 'value': 0.1})
    result: DataFrame = getattr(df.groupby('col1', observed=False), func)()
    idx: pd.CategoricalIndex = pd.CategoricalIndex(data=['A', 'B'], name='col1', ordered=True)
    expected: DataFrame = DataFrame({'col2': pd.Categorical([1, None], categories=[1, 
        2], ordered=True), 'value': [0.1, None]}, index=idx)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('func', ['min', 'max'])
def func_ehyan4xp(func: str, string_dtype_no_object: str) -> None:
    dtype: str = string_dtype_no_object
    df: DataFrame = DataFrame({'a': ['a'], 'b': 'a', 'c': 'a'}, dtype=dtype).iloc[:0]
    result: DataFrame = getattr(df.groupby('a'), func)()
    expected: DataFrame = DataFrame(columns=['b', 'c'], dtype=dtype, index=pd.Index([],
        dtype=dtype, name='a'))
    tm.assert_frame_equal(result, expected)

def func_qe8k5xn2() -> None:
    df: DataFrame = DataFrame({'Unnamed: 0': ['-04-23', '-05-06', '-05-07'], 'Date': [
        '2013-04-23 00:00:00', '2013-05-06 00:00:00', '2013-05-07 00:00:00'
        ], 'app': Series([np.nan, np.nan, 'OE']), 'File': ['log080001.log',
        'log.log', 'xlsx']})
    gb = df.groupby('Date')
    r: DataFrame = gb[['File']].max()
    e: DataFrame = gb['File'].max().to_frame()
    tm.assert_frame_equal(r, e)
    assert not r['File'].isna().any()

@pytest.mark.slow
@pytest.mark.parametrize('with_nan', [True, False])
@pytest.mark.parametrize('keys', [['joe'], ['joe', 'jim']])
def func_wqlnfn4b(sort: bool, dropna: bool, as_index: bool, with_nan: bool, keys: List[str]) -> None:
    n: int = 100
    m: int = 10
    days: pd.DatetimeIndex = date_range('2015-08-23', periods=10)
    df: DataFrame = DataFrame({'jim': np.random.default_rng(2).choice(list(
        ascii_lowercase), n), 'joe': np.random.default_rng(2).choice(days,
        n), 'julie': np.random.default_rng(2).integers(0, m, n)})
    if with_nan:
        df = df.astype({'julie': float})
        df.loc[1::17, 'jim'] = None
        df.loc[3::37, 'joe'] = None
        df.loc[7::19, 'julie'] = None
        df.loc[8::19, 'julie'] = None
        df.loc[9::19, 'julie'] = None
    original_df: DataFrame = df.copy()
    gr: pd.core.groupby.generic.DataFrameGroupBy = df.groupby(keys, as_index=as_index, sort=sort)
    left: Union[Series, DataFrame] = gr['julie'].nunique(dropna=dropna)
    gr = df.groupby(keys, as_index=as_index, sort=sort)
    right: Union[Series, DataFrame] = gr['julie'].apply(Series.nunique, dropna=dropna)
    if not as_index:
        right = right.reset_index(drop=True)
    if as_index:
        tm.assert_series_equal(left, right, check_names=False)
    else:
        tm.assert_frame_equal(left, right, check_names=False)
    tm.assert_frame_equal(df, original_df)

def func_dazzwshu() -> None:
    df: DataFrame = DataFrame({'A': list('abbacc'), 'B': list('abxacc'), 'C': list(
        'abbacx')})
    expected: DataFrame = DataFrame({'A': list('abc'), 'B': [1, 2, 1], 'C': [1, 1, 2]})
    result: DataFrame = df.groupby('A', as_index=False).nunique()
    tm.assert_frame_equal(result, expected)
    expected.index = list('abc')
    expected.index.name = 'A'
    expected = expected.drop(columns='A')
    result = df.groupby('A').nunique()
    tm.assert_frame_equal(result, expected)
    result = df.replace({'x': None}).groupby('A').nunique(dropna=False)
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({'B': [1] * 3, 'C': [1] * 3}, index=list('abc'))
    expected.index.name = 'A'
    result = df.replace({'x': None}).groupby('A').nunique()
    tm.assert_frame_equal(result, expected)

def func_75ga3eb8() -> None:
    data: List[List[int]] = [[1, 11], [1, 41], [1, 17], [1, 37], [1, 7], [1, 29], [1, 31], [
        1, 2], [1, 3], [1, 43], [1, 5], [1, 47], [1, 19], [1, 88]]
    df: DataFrame = DataFrame(data, columns=['amount', 'id', 'name'])
    result: Series = df.groupby(['id', 'amount'])['name'].nunique()
    index = MultiIndex.from_arrays([df.id, df.amount])
    expected: Series = Series([1] * 5, name='name', index=index)
    tm.assert_series_equal(result, expected)

def func_91sp1gvj() -> None:
    data: Series = Series(name='name', dtype=object)
    result: Series = data.groupby(level=0).nunique()
    expected: Series = Series(name='name', dtype='int64')
    tm.assert_series_equal(result, expected)

def func_xlw1k1x8() -> None:
    test: DataFrame = DataFrame({'time': [Timestamp('2016-06-28 09:35:35'), Timestamp(
        '2016-06-28 16:09:30'), Timestamp('2016-06-28 16:46:28')], 'data':
        ['1', '2', '3']}).set_index('time')
    result: Series = test.groupby(pd.Grouper(freq='h'))['data'].nunique()
    expected: Series = test.groupby(pd.Grouper(freq='h'))['data'].apply(Series.nunique)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('key, data, dropna, expected', [(['x', 'x', 'x'],
    [Timestamp('2019-01-01'), pd.NaT, Timestamp('2019-01-01')], True,
    Series([1], index=pd.Index(['x'], name='key'), name='data')), (['x',
    'x', 'x'], [dt.date(2019, 1, 1), pd.NaT, dt.date(2019, 1, 1)], True,
    Series([1], index=pd.Index(['x'], name='key'), name='data')), (['x',
    'x', 'x', 'y', 'y'], [dt.date(2019, 1, 1), pd.NaT, dt.date(2019, 1, 1),
    pd.NaT, dt.date(2019, 1, 1)], False, Series([2, 2], index=pd.Index(['x',
    'y'], name='key'), name='data')), (['x', 'x', 'x', 'x', 'y'], [dt.date(
    2019, 1, 1), pd.NaT, dt.date(2019, 1, 1), pd.NaT, dt.date(2019, 1, 1)],
    False, Series([2, 1], index=pd.Index(['x', 'y'], name='key'), name=
    'data'))])
def func_gnaqi79c(key: List[str], data: List[Any], dropna: bool, expected: Series) -> None:
    df: DataFrame = DataFrame({'key': key, 'data': data})
    result: Series = df.groupby(['key'])['data'].nunique(dropna=dropna)
    tm.assert_series_equal(result, expected)

def func_kpylbji2() -> None:
    test: DataFrame = DataFrame([1, 2, 2], columns=pd.Index(['A'], name='level_0'))
    result: DataFrame = test.groupby([0, 0, 0]).nunique()
    expected: DataFrame = DataFrame([2], index=np.array([0]), columns=test.columns)
    tm.assert_frame_equal(result, expected)

def func_70qwafu1() -> None:
    df: DataFrame = DataFrame(date_range('2008-12-31', '2009-01-02'), columns=['date'])
    result: Series = df.groupby([0, 0, 1])['date'].transform('nunique')
    expected: Series = Series([2, 2, 1], name='date')
    tm.assert_series_equal(result, expected)

def func_q06c8j3u(observed: bool) -> None:
    cat: pd.Categorical = Series([1]).astype('category')
    ser: Series = cat[:0]
    gb = ser.groupby(ser, observed=observed)
    result: Series = gb.nunique()
    if observed:
        expected: Series = Series([], index=cat[:0], dtype='int64')
    else:
        expected = Series([0], index=cat, dtype='int64')
    tm.assert_series_equal(result, expected)

def func_jkj74clh() -> None:
    s: Series = Series([1.0, 2.0, np.nan, 3.0])
    grouped = s.groupby([0, 1, 2, 2])
    result: Series = grouped.agg(builtins.sum)
    result2: Series = grouped.apply(builtins.sum)
    expected: Series = Series([1.0, 2.0, np.nan], index=np.array([0, 1, 2]))
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)

@pytest.mark.parametrize('min_count', [0, 10])
def func_y1ddgd0r(min_count: int) -> None:
    b: bool = True
    a: bool = False
    na: float = np.nan
    dfg: pd.array = pd.array([b, b, na, na, a, a, b], dtype='boolean')
    df: DataFrame = DataFrame({'A': [1, 1, 2, 2, 3, 3, 1], 'B': dfg})
    result: DataFrame = df.groupby('A').sum(min_count=min_count)
    if min_count == 0:
        expected: DataFrame = DataFrame({'B': pd.array([3, 0, 0], dtype='Int64')},
            index=pd.Index([1, 2, 3], name='A'))
        tm.assert_frame_equal(result, expected)
    else:
        expected = DataFrame({'B': pd.array([pd.NA] * 3, dtype='Int64')},
            index=pd.Index([1, 2, 3], name='A'))
        tm.assert_frame_equal(result, expected)

def func_7rfr77ev() -> None:
    df: DataFrame = DataFrame({'a': [0, 1, 2], 'b': [0, 1, 2], 'c': [0, 1, 2]}, dtype=
        'Int64')
    grouped = df.groupby('a')
    idx: pd.Index = pd.Index([0, 1, 2], name='a', dtype='Int64')
    result: Series = grouped['b'].sum(min_count=2)
    expected: Series = Series([pd.NA] * 3, dtype='Int64', index=idx, name='b')
    tm.assert_series_equal(result, expected)
    result = grouped.sum(min_count=2)
    expected = DataFrame({'b': [pd.NA] * 3, 'c': [pd.NA] * 3}, dtype=
        'Int64', index=idx)
    tm.assert_frame_equal(result, expected)

def func_lseioxkx() -> None:
    df: DataFrame = DataFrame({'a': [1, 1, 2, 2], 'b': [pd.Timedelta('1D'), pd.
        Timedelta('2D'), pd.Timedelta('3D'), pd.NaT]})
    td3: pd.Timedelta = pd.Timedelta(days=3)
    gb = df.groupby('a')
    res: DataFrame = gb.sum()
    expected: DataFrame = DataFrame({'b': [td3, td3]}, index=pd.Index([1, 2], name='a'))
    tm.assert_frame_equal(res, expected)
    res = gb['b'].sum()
    tm.assert_series_equal(res, expected['b'])

@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64',
    'float32', 'float64', 'uint64'])
@pytest.mark.parametrize('method,data', [('first', {'df': [{'a': 1, 'b': 1},
    {'a': 2, 'b': 3}]}), ('last', {'df': [{'a': 1, 'b': 2}, {'a': 2, 'b': 4
    }]}), ('min', {'df': [{'a': 1, 'b': 1}, {'a': 2, 'b': 3}]}), ('max', {
    'df': [{'a': 1, 'b': 2}, {'a': 2, 'b': 4}]}), ('count', {'df': [{'a': 1,
    'b': 2}, {'a': 2, 'b': 2}], 'out_type': 'int64'})])
def func_gbc9u4o8(dtype: str, method: str, data: Dict[str, Any]) -> None:
    df: DataFrame = DataFrame([{'a': 1, 'b': 1}, {'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {
        'a': 2, 'b': 4}])
    df['b'] = df.b.astype(dtype)
    if 'args' not in data:
        data['args'] = []
    if 'out_type' in data:
        out_type: str = data['out_type']
    else:
        out_type = dtype
    exp: Dict[str, List[Any]] = data['df']
    df_out: DataFrame = DataFrame(exp)
    df_out['b'] = df_out.b.astype(out_type)
    df_out.set_index('a', inplace=True)
    grpd = df.groupby('a')
    t: DataFrame = getattr(grpd, method)(*data['args'])
    tm.assert_frame_equal(t, df_out)

def func_wfiiscik(*args: Any, **kwargs: Any) -> Any:
    from scipy.stats import sem
    return sem(*args, ddof=1, **kwargs)

@pytest.mark.parametrize('op,targop', [('mean', np.mean), ('median', np.
    median), ('std', np.std), ('var', np.var), ('sum', np.sum), ('prod', np
    .prod), ('min', np.min), ('max', np.max), ('first', lambda x: x.iloc[0]
    ), ('last', lambda x: x.iloc[-1]), ('count', np.size), pytest.param(
    'sem', func_wfiiscik, marks=td.skip_if_no('scipy'))])
def func_8hrpqnrq(op: str, targop: Callable, skipna: bool, sort: bool) -> None:
    df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal(1000))
    labels: np.ndarray = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    result: DataFrame = getattr(df.groupby(labels), op)()
    kwargs: Dict[str, Any] = {'ddof': 1, 'axis': 0} if op in ['std', 'var'] else {}
    expected: DataFrame = df.groupby(labels).agg(targop, **kwargs)
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('values', [{'a': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'b':
    [1, pd.NA, 2, 1, pd.NA, 2, 1, pd.NA, 2]}, {'a': [1, 1, 2, 2, 3, 3], 'b':
    [1, 2, 1, 2, 1, 2]}])
@pytest.mark.parametrize('function', ['mean', 'median', 'var'])
def func_t1j3fs79(values: Dict[str, List[Any]], function: str) -> None:
    output: float = 0.5 if function == 'var' else 1.5
    arr: np.ndarray = np.array([output] * 3, dtype=float)
    idx: pd.Index = pd.Index([1, 2, 3], name='a', dtype='Int64')
    expected: DataFrame = DataFrame({'b': arr}, index=idx).astype('Float64')
    groups: DataFrameGroupBy = DataFrame(values, dtype='Int64').groupby('a')
    result: DataFrame = getattr(groups, function)()
    tm.assert_frame_equal(result, expected)
    result = groups.agg(function)
    tm.assert_frame_equal(result, expected)
    result = groups.agg([function])
    expected.columns = MultiIndex.from_tuples([('b', function)])
    tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('op', ['sum', 'prod', 'min', 'max', 'median',
    'mean', 'skew', 'kurt', 'std', 'var', 'sem'])
def func_odg39x3v(op: str, skipna: bool, sort: bool) -> None:
    frame: DataFrame = DataFrame([0])
    grouped = frame.groupby(level=0, sort=sort)
    if op in ['skew', 'kurt', 'sum', 'mean']:
        result = getattr(grouped, op)(skipna=skipna)
        expected = frame.groupby(level=0).apply(lambda h: getattr(h, op)(
            skipna=skipna))
        if sort:
            expected = expected.sort_index()
        tm.assert_frame_equal(result, expected)
    else:
        result = getattr(grouped, op)()
        expected = frame.groupby(level=0).apply(lambda h: getattr(h, op)())
        if sort:
            expected = expected.sort_index()
        tm.assert_frame_equal(result, expected)

def func_qg6bbrfj() -> None:
    data: List[List[int]] = [[1, 11], [1, 41], [1, 17], [1, 37], [1, 7], [1, 29], [1, 31], [
        1, 2], [1, 3], [1, 43], [1, 5], [1, 47], [1, 19], [1, 88]]
    df: DataFrame = DataFrame(data, columns=['A', 'B'], dtype='int64')
    result: DataFrame = df.groupby(['A']).prod().reset_index()
    expected: DataFrame = DataFrame({'A': [1], 'B': [180970905912331920]}, dtype='int64')
    tm.assert_frame_equal(result, expected)

def func_awjkmz0g() -> None:
    tdi: pd.TimedeltaIndex = pd.timedelta_range('1 Day', periods=10000)
    ser: Series = Series(tdi)
    ser[::5] *= 2
    df: DataFrame = ser.to_frame('A').copy()
    df['B'] = ser + Timestamp(0)
    df['C'] = ser + Timestamp(0, tz='UTC')
    df.iloc[-1] = pd.NaT
    gb = df.groupby(list(range(5)) * 2000)
    result: DataFrame = gb.std()
    td1: pd.Timedelta = pd.Timedelta('2887 days 11:21:02.326710176')
    td4: pd.Timedelta = pd.Timedelta('2886 days 00:42:34.664668096')
    exp_ser: Series = Series([td1 * 2, td1, td1, td1, td4], index=np.arange(5))
    expected: DataFrame = DataFrame({'A': exp_ser, 'B': exp_ser, 'C': exp_ser})
    tm.assert_frame_equal(result, expected)
