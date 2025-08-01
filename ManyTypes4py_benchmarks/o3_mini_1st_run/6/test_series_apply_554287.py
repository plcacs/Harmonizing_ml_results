#!/usr/bin/env python3
from typing import Any, Callable, Dict, List, Union, Literal
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, concat, date_range, timedelta_range
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels

@pytest.fixture(params=[False, 'compat'])
def by_row(request: pytest.FixtureRequest) -> Union[bool, Literal['compat']]:
    return request.param

def test_series_map_box_timedelta(by_row: Union[bool, Literal['compat']]) -> None:
    ser: Series = Series(timedelta_range('1 day 1 s', periods=3, freq='h'))

    def f(x: Any) -> Any:
        return x.total_seconds() if by_row else x.dt.total_seconds()
    result: Series = ser.apply(f, by_row=by_row)
    expected: Series = ser.map(lambda x: x.total_seconds())
    tm.assert_series_equal(result, expected)
    expected = Series([86401.0, 90001.0, 93601.0])
    tm.assert_series_equal(result, expected)

def test_apply(datetime_series: Series, by_row: Union[bool, Literal['compat']]) -> None:
    result: Series = datetime_series.apply(np.sqrt, by_row=by_row)
    with np.errstate(all='ignore'):
        expected: Series = np.sqrt(datetime_series)
    tm.assert_series_equal(result, expected)
    result = datetime_series.apply(np.exp, by_row=by_row)
    expected = np.exp(datetime_series)
    tm.assert_series_equal(result, expected)
    s: Series = Series(dtype=object, name='foo', index=Index([], name='bar'))
    rs: Series = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(s, rs)
    assert s is not rs
    assert s.index is rs.index
    assert s.dtype == rs.dtype
    assert s.name == rs.name
    s = Series(index=[1, 2, 3], dtype=np.float64)
    rs = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(s, rs)

def test_apply_map_same_length_inference_bug() -> None:
    s: Series = Series([1, 2])

    def f(x: Any) -> Any:
        return (x, x + 1)
    result: Series = s.apply(f, by_row='compat')
    expected: Series = s.map(f)
    tm.assert_series_equal(result, expected)

def test_apply_args() -> None:
    s: Series = Series(['foo,bar'])
    result: Series = s.apply(str.split, args=(',',))
    assert result[0] == ['foo', 'bar']
    assert isinstance(result[0], list)

@pytest.mark.parametrize('args, kwargs, increment', [
    ((), {}, 0),
    ((), {'a': 1}, 1),
    ((2, 3), {}, 32),
    ((1,), {'c': 2}, 201)
])
def test_agg_args(args: tuple, kwargs: Dict[str, Any], increment: int) -> None:

    def f(x: Any, a: int = 0, b: int = 0, c: int = 0) -> Any:
        return x + a + 10 * b + 100 * c
    s: Series = Series([1, 2])
    result: Series = s.agg(f, 0, *args, **kwargs)
    expected: Series = s + increment
    tm.assert_series_equal(result, expected)

def test_agg_mapping_func_deprecated() -> None:
    s: Series = Series([1, 2, 3])

    def foo1(x: Any, a: int = 1, c: int = 0) -> Any:
        return x + a + c

    def foo2(x: Any, b: int = 2, c: int = 0) -> Any:
        return x + b + c
    s.agg(foo1, 0, 3, c=4)
    s.agg([foo1, foo2], 0, 3, c=4)
    s.agg({'a': foo1, 'b': foo2}, 0, 3, c=4)

def test_series_apply_map_box_timestamps(by_row: Union[bool, Literal['compat']]) -> None:
    ser: Series = Series(date_range('1/1/2000', periods=10))

    def func(x: Any) -> Any:
        return (x.hour, x.day, x.month)
    if not by_row:
        msg = "Series' object has no attribute 'hour'"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(func, by_row=by_row)
        return
    result: Series = ser.apply(func, by_row=by_row)
    expected: Series = ser.map(func)
    tm.assert_series_equal(result, expected)

def test_apply_box_dt64() -> None:
    vals: List[pd.Timestamp] = [pd.Timestamp('2011-01-01'), pd.Timestamp('2011-01-02')]
    ser: Series = Series(vals, dtype='M8[ns]')
    assert ser.dtype == 'datetime64[ns]'
    res: Series = ser.apply(lambda x: f'{type(x).__name__}_{x.day}_{x.tz}', by_row='compat')
    exp: Series = Series(['Timestamp_1_None', 'Timestamp_2_None'])
    tm.assert_series_equal(res, exp)

def test_apply_box_dt64tz() -> None:
    vals: List[pd.Timestamp] = [pd.Timestamp('2011-01-01', tz='US/Eastern'),
                                  pd.Timestamp('2011-01-02', tz='US/Eastern')]
    ser: Series = Series(vals, dtype='M8[ns, US/Eastern]')
    assert ser.dtype == 'datetime64[ns, US/Eastern]'
    res: Series = ser.apply(lambda x: f'{type(x).__name__}_{x.day}_{x.tz}', by_row='compat')
    exp: Series = Series(['Timestamp_1_US/Eastern', 'Timestamp_2_US/Eastern'])
    tm.assert_series_equal(res, exp)

def test_apply_box_td64() -> None:
    vals: List[pd.Timedelta] = [pd.Timedelta('1 days'), pd.Timedelta('2 days')]
    ser: Series = Series(vals)
    assert ser.dtype == 'timedelta64[ns]'
    res: Series = ser.apply(lambda x: f'{type(x).__name__}_{x.days}', by_row='compat')
    exp: Series = Series(['Timedelta_1', 'Timedelta_2'])
    tm.assert_series_equal(res, exp)

def test_apply_box_period() -> None:
    vals: List[pd.Period] = [pd.Period('2011-01-01', freq='M'), pd.Period('2011-01-02', freq='M')]
    ser: Series = Series(vals)
    assert ser.dtype == 'Period[M]'
    res: Series = ser.apply(lambda x: f'{type(x).__name__}_{x.freqstr}', by_row='compat')
    exp: Series = Series(['Period_M', 'Period_M'])
    tm.assert_series_equal(res, exp)

def test_apply_datetimetz(by_row: Union[bool, Literal['compat']]) -> None:
    values: pd.DatetimeIndex = date_range('2011-01-01', '2011-01-02', freq='h').tz_localize('Asia/Tokyo')
    s: Series = Series(values, name='XX')
    result: Series = s.apply(lambda x: x + pd.offsets.Day(), by_row=by_row)
    exp_values: pd.DatetimeIndex = date_range('2011-01-02', '2011-01-03', freq='h').tz_localize('Asia/Tokyo')
    exp: Series = Series(exp_values, name='XX')
    tm.assert_series_equal(result, exp)
    result = s.apply(lambda x: x.hour if by_row else x.dt.hour, by_row=by_row)
    exp = Series(list(range(24)) + [0], name='XX', dtype='int64' if by_row else 'int32')
    tm.assert_series_equal(result, exp)

    def f(x: Any) -> str:
        return str(x.tz) if by_row else str(x.dt.tz)
    result = s.apply(f, by_row=by_row)
    if by_row:
        exp = Series(['Asia/Tokyo'] * 25, name='XX')
        tm.assert_series_equal(result, exp)
    else:
        assert result == 'Asia/Tokyo'

def test_apply_categorical(by_row: Union[bool, Literal['compat']],
                           using_infer_string: Any) -> None:
    values: pd.Categorical = pd.Categorical(list('ABBABCD'), categories=list('DCBA'), ordered=True)
    ser: Series = Series(values, name='XX', index=list('abcdefg'))
    if not by_row:
        msg = "Series' object has no attribute 'lower"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(lambda x: x.lower(), by_row=by_row)
        assert ser.apply(lambda x: 'A', by_row=by_row) == 'A'
        return
    result: Series = ser.apply(lambda x: x.lower(), by_row=by_row)
    values = pd.Categorical(list('abbabcd'), categories=list('dcba'), ordered=True)
    exp: Series = Series(values, name='XX', index=list('abcdefg'))
    tm.assert_series_equal(result, exp)
    tm.assert_categorical_equal(result.values, exp.values)
    result = ser.apply(lambda x: 'A')
    exp = Series(['A'] * 7, name='XX', index=list('abcdefg'))
    tm.assert_series_equal(result, exp)
    if using_infer_string:
        assert result.dtype == 'str'
    else:
        assert result.dtype == object

@pytest.mark.parametrize('series', [['1-1', '1-1', np.nan], ['1-1', '1-2', np.nan]])
def test_apply_categorical_with_nan_values(series: List[Union[str, float]], by_row: Union[bool, Literal['compat']]) -> None:
    s: Series = Series(series, dtype='category')
    if not by_row:
        msg = "'Series' object has no attribute 'split'"
        with pytest.raises(AttributeError, match=msg):
            s.apply(lambda x: x.split('-')[0], by_row=by_row)
        return
    result: Series = s.apply(lambda x: x.split('-')[0] if pd.notna(x) else False, by_row=by_row)
    result = result.astype(object)
    expected: Series = Series(['1', '1', False], dtype='category')
    expected = expected.astype(object)
    tm.assert_series_equal(result, expected)

def test_apply_empty_integer_series_with_datetime_index(by_row: Union[bool, Literal['compat']]) -> None:
    s: Series = Series([], index=date_range(start='2018-01-01', periods=0), dtype=int)
    result: Series = s.apply(lambda x: x, by_row=by_row)
    tm.assert_series_equal(result, s)

def test_apply_dataframe_iloc() -> None:
    uintDF: DataFrame = DataFrame(np.uint64([1, 2, 3, 4, 5]), columns=['Numbers'])
    indexDF: DataFrame = DataFrame([2, 3, 2, 1, 2], columns=['Indices'])

    def retrieve(targetRow: int, targetDF: DataFrame) -> Any:
        val = targetDF['Numbers'].iloc[targetRow]
        return val
    result: Series = indexDF['Indices'].apply(retrieve, args=(uintDF,))
    expected: Series = Series([3, 4, 3, 2, 3], name='Indices', dtype='uint64')
    tm.assert_series_equal(result, expected)

def test_transform(string_series: Series, by_row: Union[bool, Literal['compat']]) -> None:
    with np.errstate(all='ignore'):
        f_sqrt: Series = np.sqrt(string_series)
        f_abs: Series = np.abs(string_series)
        result: Series = string_series.apply(np.sqrt, by_row=by_row)
        expected: Series = f_sqrt.copy()
        tm.assert_series_equal(result, expected)
        result = string_series.apply([np.sqrt], by_row=by_row)
        expected_df: DataFrame = f_sqrt.to_frame().copy()
        expected_df.columns = ['sqrt']
        tm.assert_frame_equal(result, expected_df)
        result = string_series.apply(['sqrt'], by_row=by_row)
        tm.assert_frame_equal(result, expected_df)
        expected_multi: DataFrame = concat([f_sqrt, f_abs], axis=1)
        expected_multi.columns = ['sqrt', 'absolute']
        result = string_series.apply([np.sqrt, np.abs], by_row=by_row)
        tm.assert_frame_equal(result, expected_multi)
        expected_multi2: DataFrame = concat([f_sqrt, f_abs], axis=1)
        expected_multi2.columns = ['foo', 'bar']
        expected_multi2 = expected_multi2.unstack().rename('series')
        result = string_series.apply({'foo': np.sqrt, 'bar': np.abs}, by_row=by_row)
        tm.assert_series_equal(result.reindex_like(expected_multi2), expected_multi2)

@pytest.mark.parametrize('op', series_transform_kernels)
def test_transform_partial_failure(op: str, request: pytest.FixtureRequest) -> None:
    if op in ('ffill', 'bfill', 'shift'):
        request.applymarker(pytest.mark.xfail(reason=f'{op} is successful on any dtype'))
    ser: Series = Series(3 * [object])
    if op in ('fillna', 'ngroup'):
        error: type = ValueError
        msg: str = 'Transform function failed'
    else:
        error = TypeError
        msg = '|'.join(["not supported between instances of 'type' and 'type'", 'unsupported operand type'])
    with pytest.raises(error, match=msg):
        ser.transform([op, 'shift'])
    with pytest.raises(error, match=msg):
        ser.transform({'A': op, 'B': 'shift'})
    with pytest.raises(error, match=msg):
        ser.transform({'A': [op], 'B': ['shift']})
    with pytest.raises(error, match=msg):
        ser.transform({'A': [op, 'shift'], 'B': [op]})

def test_transform_partial_failure_valueerror() -> None:

    def noop(x: Any) -> Any:
        return x

    def raising_op(_: Any) -> Any:
        raise ValueError
    ser: Series = Series(3 * [object])
    msg: str = 'Transform function failed'
    with pytest.raises(ValueError, match=msg):
        ser.transform([noop, raising_op])
    with pytest.raises(ValueError, match=msg):
        ser.transform({'A': raising_op, 'B': noop})
    with pytest.raises(ValueError, match=msg):
        ser.transform({'A': [raising_op], 'B': [noop]})
    with pytest.raises(ValueError, match=msg):
        ser.transform({'A': [noop, raising_op], 'B': [noop]})

def test_demo() -> None:
    s: Series = Series(range(6), dtype='int64', name='series')
    result: Series = s.agg(['min', 'max'])
    expected: Series = Series([0, 5], index=['min', 'max'], name='series')
    tm.assert_series_equal(result, expected)
    result = s.agg({'foo': 'min'})
    expected = Series([0], index=['foo'], name='series')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('func', [str, lambda x: str(x)])
def test_apply_map_evaluate_lambdas_the_same(string_series: Series, func: Callable[[Any], str],
                                               by_row: Union[bool, Literal['compat']]) -> None:
    result: Any = string_series.apply(func, by_row=by_row)
    if by_row:
        expected: Series = string_series.map(func)
        tm.assert_series_equal(result, expected)
    else:
        assert result == str(string_series)

def test_agg_evaluate_lambdas(string_series: Series) -> None:
    result: Any = string_series.agg(lambda x: type(x))
    assert result is Series
    result = string_series.agg(type)
    assert result is Series

@pytest.mark.parametrize('op_name', ['agg', 'apply'])
def test_with_nested_series(datetime_series: Series, op_name: str) -> None:
    result: Any = getattr(datetime_series, op_name)(lambda x: Series([x, x ** 2], index=['x', 'x^2']))
    if op_name == 'apply':
        expected: DataFrame = DataFrame({'x': datetime_series, 'x^2': datetime_series ** 2})
        tm.assert_frame_equal(result, expected)
    else:
        expected: Series = Series([datetime_series, datetime_series ** 2], index=['x', 'x^2'])
        tm.assert_series_equal(result, expected)

def test_replicate_describe(string_series: Series) -> None:
    expected: Series = string_series.describe()
    result: Series = string_series.apply({
        'count': 'count',
        'mean': 'mean',
        'std': 'std',
        'min': 'min',
        '25%': lambda x: x.quantile(0.25),
        '50%': 'median',
        '75%': lambda x: x.quantile(0.75),
        'max': 'max'
    })
    tm.assert_series_equal(result, expected)

def test_reduce(string_series: Series) -> None:
    result: Series = string_series.agg(['sum', 'mean'])
    expected: Series = Series([string_series.sum(), string_series.mean()], ['sum', 'mean'], name=string_series.name)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('how, kwds', [
    ('agg', {}),
    ('apply', {'by_row': 'compat'}),
    ('apply', {'by_row': False})
])
def test_non_callable_aggregates(how: str, kwds: Dict[str, Any]) -> None:
    s: Series = Series([1, 2, None])
    result: Any = getattr(s, how)('size', **kwds)
    expected: int = s.size
    assert result == expected
    result = getattr(s, how)(['size', 'count', 'mean'], **kwds)
    expected = Series({'size': 3.0, 'count': 2.0, 'mean': 1.5})
    tm.assert_series_equal(result, expected)
    result = getattr(s, how)({'size': 'size', 'count': 'count', 'mean': 'mean'}, **kwds)
    tm.assert_series_equal(result, expected)

def test_series_apply_no_suffix_index(by_row: Union[bool, Literal['compat']]) -> None:
    s: Series = Series([4] * 3)
    result: Series = s.apply(['sum', lambda x: x.sum(), lambda x: x.sum()], by_row=by_row)
    expected: Series = Series([12, 12, 12], index=['sum', '<lambda>', '<lambda>'])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('dti,exp', [
    (Series([1, 2], index=pd.DatetimeIndex([0, 31536000000])),
     DataFrame(np.repeat([[1, 2]], 2, axis=0), dtype='int64')),
    (Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts'),
     DataFrame(np.repeat([[1, 2]], 10, axis=0), dtype='int64'))
])
@pytest.mark.parametrize('aware', [True, False])
def test_apply_series_on_date_time_index_aware_series(dti: Series, exp: DataFrame, aware: bool) -> None:
    if aware:
        index = dti.tz_localize('UTC').index
    else:
        index = dti.index
    result: DataFrame = Series(index).apply(lambda x: Series([1, 2]))
    tm.assert_frame_equal(result, exp)

@pytest.mark.parametrize('by_row, expected', [
    ('compat', Series(np.ones(10), dtype='int64')),
    (False, 1)
])
def test_apply_scalar_on_date_time_index_aware_series(by_row: Union[bool, Literal['compat']], expected: Any) -> None:
    series: Series = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10, tz='UTC'))
    result: Any = Series(series.index).apply(lambda x: 1, by_row=by_row)
    tm.assert_equal(result, expected)

def test_apply_to_timedelta(by_row: Union[bool, Literal['compat']]) -> None:
    list_of_valid_strings: List[str] = ['00:00:01', '00:00:02']
    a: pd.TimedeltaIndex = pd.to_timedelta(list_of_valid_strings)
    b: Series = Series(list_of_valid_strings).apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)
    list_of_strings: List[Any] = ['00:00:01', np.nan, pd.NaT, pd.NaT]
    a = pd.to_timedelta(list_of_strings)
    ser: Series = Series(list_of_strings)
    b = ser.apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)

@pytest.mark.parametrize('ops, names', [
    ([np.sum], ['sum']),
    ([np.sum, np.mean], ['sum', 'mean']),
    (np.array([np.sum]), ['sum']),
    (np.array([np.sum, np.mean]), ['sum', 'mean'])
])
@pytest.mark.parametrize('how, kwargs', [
    ('agg', {}),
    ('apply', {'by_row': 'compat'}),
    ('apply', {'by_row': False})
])
def test_apply_listlike_reducer(string_series: Series, ops: Union[List[Callable[[Series], Any]], np.ndarray],
                                names: List[str], how: str, kwargs: Dict[str, Any]) -> None:
    expected: Series = Series({name: op(string_series) for name, op in zip(names, ops)})
    expected.name = 'series'
    result: Series = getattr(string_series, how)(ops, **kwargs)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ops', [
    {'A': np.sum},
    {'A': np.sum, 'B': np.mean},
    Series({'A': np.sum}),
    Series({'A': np.sum, 'B': np.mean})
])
@pytest.mark.parametrize('how, kwargs', [
    ('agg', {}),
    ('apply', {'by_row': 'compat'}),
    ('apply', {'by_row': False})
])
def test_apply_dictlike_reducer(string_series: Series, ops: Union[Dict[str, Callable[[Series], Any]], Series],
                                how: str, kwargs: Dict[str, Any], by_row: Union[bool, Literal['compat']]) -> None:
    expected: Series = Series({name: op(string_series) for name, op in ops.items()})
    expected.name = string_series.name
    result: Series = getattr(string_series, how)(ops, **kwargs)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ops, names', [
    ([np.sqrt], ['sqrt']),
    ([np.abs, np.sqrt], ['absolute', 'sqrt']),
    (np.array([np.sqrt]), ['sqrt']),
    (np.array([np.abs, np.sqrt]), ['absolute', 'sqrt'])
])
def test_apply_listlike_transformer(string_series: Series, ops: Union[List[Callable[[Series], Any]], np.ndarray],
                                    names: List[str], by_row: Union[bool, Literal['compat']]) -> None:
    with np.errstate(all='ignore'):
        expected: DataFrame = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        result: DataFrame = string_series.apply(ops, by_row=by_row)
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize('ops, expected', [
    ([lambda x: x], DataFrame({'<lambda>': [1, 2, 3]})),
    ([lambda x: x.sum()], Series([6], index=['<lambda>']))
])
def test_apply_listlike_lambda(ops: List[Callable[[Any], Any]], expected: Union[DataFrame, Series],
                               by_row: Union[bool, Literal['compat']]) -> None:
    ser: Series = Series([1, 2, 3])
    result: Any = ser.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)

@pytest.mark.parametrize('ops', [
    {'A': np.sqrt},
    {'A': np.sqrt, 'B': np.exp},
    Series({'A': np.sqrt}),
    Series({'A': np.sqrt, 'B': np.exp})
])
def test_apply_dictlike_transformer(string_series: Series,
                                    ops: Union[Dict[str, Callable[[Series], Any]], Series],
                                    by_row: Union[bool, Literal['compat']]) -> None:
    with np.errstate(all='ignore'):
        expected: Union[Series, DataFrame] = concat({name: op(string_series) for name, op in ops.items()})
        expected.name = string_series.name
        result: Series = string_series.apply(ops, by_row=by_row)
        tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('ops, expected', [
    ({'a': lambda x: x}, Series([1, 2, 3], index=MultiIndex.from_arrays([['a'] * 3, list(range(3))]))),
    ({'a': lambda x: x.sum()}, Series([6], index=['a']))
])
def test_apply_dictlike_lambda(ops: Dict[str, Callable[[Any], Any]], by_row: Union[bool, Literal['compat']],
                               expected: Series) -> None:
    ser: Series = Series([1, 2, 3])
    result: Any = ser.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)

def test_apply_retains_column_name(by_row: Union[bool, Literal['compat']]) -> None:
    df: DataFrame = DataFrame({'x': range(3)}, Index(range(3), name='x'))
    result: DataFrame = df.x.apply(lambda x: Series(range(x + 1), Index(range(x + 1), name='y')))
    expected: DataFrame = DataFrame([[0.0, np.nan, np.nan],
                                     [0.0, 1.0, np.nan],
                                     [0.0, 1.0, 2.0]],
                                    columns=Index(range(3), name='y'),
                                    index=Index(range(3), name='x'))
    tm.assert_frame_equal(result, expected)

def test_apply_type() -> None:
    s: Series = Series([3, 'string', float], index=['a', 'b', 'c'])
    result: Series = s.apply(type)
    expected: Series = Series([int, str, type], index=['a', 'b', 'c'])
    tm.assert_series_equal(result, expected)

def test_series_apply_unpack_nested_data() -> None:
    ser: Series = Series([[1, 2, 3], [4, 5, 6, 7]])
    result: DataFrame = ser.apply(lambda x: Series(x))
    expected: DataFrame = DataFrame({0: [1.0, 4.0],
                                     1: [2.0, 5.0],
                                     2: [3.0, 6.0],
                                     3: [np.nan, 7]})
    tm.assert_frame_equal(result, expected)