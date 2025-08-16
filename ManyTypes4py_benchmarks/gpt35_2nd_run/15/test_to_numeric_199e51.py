import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import ArrowDtype, DataFrame, Index, Series, option_context, to_numeric
import pandas._testing as tm

@pytest.fixture(params=[None, 'raise', 'coerce'])
def errors(request: pytest.FixtureRequest) -> str:
    return request.param

@pytest.fixture(params=[True, False])
def signed(request: pytest.FixtureRequest) -> bool:
    return request.param

@pytest.fixture(params=[lambda x: x, str], ids=['identity', 'str'])
def transform(request: pytest.FixtureRequest) -> callable:
    return request.param

@pytest.fixture(params=[47393996303418497800, 100000000000000000000])
def large_val(request: pytest.FixtureRequest) -> int:
    return request.param

@pytest.fixture(params=[True, False])
def multiple_elts(request: pytest.FixtureRequest) -> bool:
    return request.param

@pytest.fixture(params=[(lambda x: Index(x, name='idx'), tm.assert_index_equal), (lambda x: Series(x, name='ser'), tm.assert_series_equal), (lambda x: np.array(Index(x).values), tm.assert_numpy_array_equal)])
def transform_assert_equal(request: pytest.FixtureRequest) -> tuple:
    return request.param

@pytest.mark.parametrize('input_kwargs,result_kwargs', [({}, {'dtype': np.int64}), ({'errors': 'coerce', 'downcast': 'integer'}, {'dtype': np.int8})])
def test_empty(input_kwargs: dict, result_kwargs: dict) -> None:
    ser = Series([], dtype=object)
    result = to_numeric(ser, **input_kwargs)
    expected = Series([], **result_kwargs)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('last_val', ['7', 7])
def test_series(last_val: str, infer_string: bool) -> None:
    with option_context('future.infer_string', infer_string):
        ser = Series(['1', '-3.14', last_val])
        result = to_numeric(ser)
    expected = Series([1, -3.14, 7])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('data', [[1, 3, 4, 5], [1.0, 3.0, 4.0, 5.0], [True, False, True, True]])
def test_series_numeric(data: list) -> None:
    ser = Series(data, index=list('ABCD'), name='EFG')
    result = to_numeric(ser)
    tm.assert_series_equal(result, ser)

@pytest.mark.parametrize('data,msg', [([1, -3.14, 'apple'], 'Unable to parse string "apple" at position 2'), (['orange', 1, -3.14, 'apple'], 'Unable to parse string "orange" at position 0')])
def test_error(data: list, msg: str) -> None:
    ser = Series(data)
    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, errors='raise')

def test_ignore_error() -> None:
    ser = Series([1, -3.14, 'apple'])
    result = to_numeric(ser, errors='coerce')
    expected = Series([1, -3.14, np.nan])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('errors,exp', [('raise', 'Unable to parse string "apple" at position 2'), ('coerce', [1.0, 0.0, np.nan])])
def test_bool_handling(errors: str, exp: str) -> None:
    ser = Series([True, False, 'apple'])
    if isinstance(exp, str):
        with pytest.raises(ValueError, match=exp):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        expected = Series(exp)
        tm.assert_series_equal(result, expected)

def test_list() -> None:
    ser = ['1', '-3.14', '7']
    res = to_numeric(ser)
    expected = np.array([1, -3.14, 7])
    tm.assert_numpy_array_equal(res, expected)

@pytest.mark.parametrize('data,arr_kwargs', [([1, 3, 4, 5], {'dtype': np.int64}), ([1.0, 3.0, 4.0, 5.0], {}), ([True, False, True, True], {})])
def test_list_numeric(data: list, arr_kwargs: dict) -> None:
    result = to_numeric(data)
    expected = np.array(data, **arr_kwargs)
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize('kwargs', [{'dtype': 'O'}, {}])
def test_numeric(kwargs: dict) -> None:
    data = [1, -3.14, 7]
    ser = Series(data, **kwargs)
    result = to_numeric(ser)
    expected = Series(data)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('columns', ['a', ['a', 'b']])
def test_numeric_df_columns(columns: str) -> None:
    df = DataFrame({'a': [1.2, decimal.Decimal('3.14'), decimal.Decimal('infinity'), '0.1'], 'b': [1.0, 2.0, 3.0, 4.0]})
    expected = DataFrame({'a': [1.2, 3.14, np.inf, 0.1], 'b': [1.0, 2.0, 3.0, 4.0]})
    df[columns] = df[columns].apply(to_numeric)
    tm.assert_frame_equal(df, expected)

@pytest.mark.parametrize('data,exp_data', [([[decimal.Decimal('3.14'), 1.0], decimal.Decimal('1.6'), 0.1], [[3.14, 1.0], 1.6, 0.1]), ([np.array([decimal.Decimal('3.14'), 1.0]), 0.1], [[3.14, 1.0], 0.1])])
def test_numeric_embedded_arr_likes(data: list, exp_data: list) -> None:
    df = DataFrame({'a': data})
    df['a'] = df['a'].apply(to_numeric)
    expected = DataFrame({'a': exp_data})
    tm.assert_frame_equal(df, expected)

def test_all_nan() -> None:
    ser = Series(['a', 'b', 'c'])
    result = to_numeric(ser, errors='coerce')
    expected = Series([np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

def test_type_check(errors: str) -> None:
    df = DataFrame({'a': [1, -3.14, 7], 'b': ['4', '5', '6']})
    kwargs = {'errors': errors} if errors is not None else {}
    with pytest.raises(TypeError, match='1-d array'):
        to_numeric(df, **kwargs)

@pytest.mark.parametrize('val', [1, 1.1, 20001])
def test_scalar(val: int, signed: bool, transform: callable) -> None:
    val = -val if signed else val
    assert to_numeric(transform(val)) == float(val)

def test_really_large_scalar(large_val: int, signed: bool, transform: callable, errors: str) -> None:
    kwargs = {'errors': errors} if errors is not None else {}
    val = -large_val if signed else large_val
    val = transform(val)
    val_is_string = isinstance(val, str)
    if val_is_string and errors in (None, 'raise'):
        msg = 'Integer out of range. at position 0'
        with pytest.raises(ValueError, match=msg):
            to_numeric(val, **kwargs)
    else:
        expected = float(val) if errors == 'coerce' and val_is_string else val
        tm.assert_almost_equal(to_numeric(val, **kwargs), expected)

def test_really_large_in_arr(large_val: int, signed: bool, transform: callable, multiple_elts: bool, errors: str) -> None:
    kwargs = {'errors': errors} if errors is not None else {}
    val = -large_val if signed else large_val
    val = transform(val)
    extra_elt = 'string'
    arr = [val] + multiple_elts * [extra_elt]
    val_is_string = isinstance(val, str)
    coercing = errors == 'coerce'
    if errors in (None, 'raise') and (val_is_string or multiple_elts):
        if val_is_string:
            msg = 'Integer out of range. at position 0'
        else:
            msg = 'Unable to parse string "string" at position 1'
        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        result = to_numeric(arr, **kwargs)
        exp_val = float(val) if coercing and val_is_string else val
        expected = [exp_val]
        if multiple_elts:
            if coercing:
                expected.append(np.nan)
                exp_dtype = float
            else:
                expected.append(extra_elt)
                exp_dtype = object
        else:
            exp_dtype = float if isinstance(exp_val, (int, float)) else object
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype)

def test_really_large_in_arr_consistent(large_val: int, signed: bool, multiple_elts: bool, errors: str) -> None:
    kwargs = {'errors': errors} if errors is not None else {}
    arr = [str(-large_val if signed else large_val)]
    if multiple_elts:
        arr.insert(0, large_val)
    if errors in (None, 'raise'):
        index = int(multiple_elts)
        msg = f'Integer out of range. at position {index}'
        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        result = to_numeric(arr, **kwargs)
        expected = [float(i) for i in arr]
        exp_dtype = float
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype)

@pytest.mark.parametrize('errors,checker', [('raise', 'Unable to parse string "fail" at position 0'), ('coerce', lambda x: np.isnan(x))])
def test_scalar_fail(errors: str, checker: str) -> None:
    scalar = 'fail'
    if isinstance(checker, str):
        with pytest.raises(ValueError, match=checker):
            to_numeric(scalar, errors=errors)
    else:
        assert checker(to_numeric(scalar, errors=errors)

@pytest.mark.parametrize('data', [[1, 2, 3], [1.0, np.nan, 3, np.nan]])
def test_numeric_dtypes(data: list, transform_assert_equal: tuple) -> None:
    transform, assert_equal = transform_assert_equal
    data = transform(data)
    result = to_numeric(data)
    assert_equal(result, data)

@pytest.mark.parametrize('data,exp', [(['1', '2', '3'], np.array([1, 2, 3], dtype='int64')), (['1.5', '2.7', '3.4'], np.array([1.5, 2.7, 3.4]))])
def test_str(data: list, exp: np.array, transform_assert_equal: tuple) -> None:
    transform, assert_equal = transform_assert_equal
    result = to_numeric(transform(data))
    expected = transform(exp)
    assert_equal(result, expected)

def test_datetime_like(tz_naive_fixture, transform_assert_equal: tuple) -> None:
    transform, assert_equal = transform_assert_equal
    idx = pd.date_range('20130101', periods=3, tz=tz_naive_fixture)
    result = to_numeric(transform(idx))
    expected = transform(idx.asi8)
    assert_equal(result, expected)

def test_timedelta(transform_assert_equal: tuple) -> None:
    transform, assert_equal = transform_assert_equal
    idx = pd.timedelta_range('1 days', periods=3, freq='D')
    result = to_numeric(transform(idx))
    expected = transform(idx.asi8)
    assert_equal(result, expected)

@pytest.mark.parametrize('scalar', [pd.Timedelta(1, 'D'), pd.Timestamp('2017-01-01T12'), pd.Timestamp('2017-01-01T12', tz='US/Pacific')])
def test_timedelta_timestamp_scalar(scalar: pd.Timestamp) -> None:
    result = to_numeric(scalar)
    expected = to_numeric(Series(scalar))[0]
    assert result == expected

def test_period(request: pytest.FixtureRequest, transform_assert_equal: tuple) -> None:
    transform, assert_equal = transform_assert_equal
    idx = pd.period_range('2011-01', periods=3, freq='M', name='')
    inp = transform(idx)
    if not isinstance(inp, Index):
        request.applymarker(pytest.mark.xfail(reason='Missing PeriodDtype support in to_numeric'))
    result = to_numeric(inp)
    expected = transform(idx.asi8)
    assert_equal(result, expected)

@pytest.mark.parametrize('errors,expected', [('raise', 'Invalid object type at position 0'), ('coerce', Series([np.nan, 1.0, np.nan]))])
def test_non_hashable(errors: str, expected: str) -> None:
    ser = Series([[10.0, 2], 1.0, 'apple'])
    if isinstance(expected, str):
        with pytest.raises(TypeError, match=expected):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        tm.assert_series_equal(result, expected)

def test_downcast_invalid_cast() -> None:
    data = ['1', 2, 3]
    invalid_downcast = 'unsigned-integer'
    msg = 'invalid downcasting method provided'
    with pytest.raises(ValueError, match=msg):
        to_numeric(data, downcast=invalid_downcast)

def test_errors_invalid_value() -> None:
    data = ['1', 2, 3]
    invalid_error_value = 'invalid'
    msg = 'invalid error value specified'
    with pytest.raises(ValueError, match=msg):
        to_numeric(data, errors=invalid_error_value)

@pytest.mark.parametrize('data', [['1', '2', None], ['1', '2', '3'], ['1', '2', 3], ['1', '2', 3.5], ['1', None, 3.5], ['1', '2', '3.5']])
def test_to_numeric_from_nullable_string(data: list, nullable_string_dtype: ArrowDtype, expected: Series) -> None:
    s = Series(data, dtype=nullable_string_dtype)
    result = to_numeric(s)
    tm.assert_series_equal(result, expected)

def test_to_numeric_from_nullable_string_coerce(nullable_string_dtype: ArrowDtype) -> None:
    values = ['a', '1']
    ser = Series(values, dtype=nullable_string_dtype)
    result = to_numeric(ser, errors='coerce')
    expected = Series([pd.NA, 1], dtype='Int64')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('data, input_dtype, downcast, expected_dtype', (([1, 1], 'Int64', 'integer', 'Int8'), ([1.0, pd.NA], 'Float64', 'integer', 'Int8'), ([1.0, 1.1], 'Float64', 'integer', 'Float64'), ([1, pd.NA], 'Int64', 'integer', 'Int8'), ([450, 300], 'Int64', 'integer', 'Int16'), ([1, 1], 'Float64', 'integer', 'Int8'), ([np.iinfo(np.int64).max - 1, 1], 'Int64', 'integer', 'Int64'), ([1, pd.NA], 'Int64', 'integer', 'Int8'), ([450, -300], 'Int64', 'integer', 'Int16'), ([np.iinfo(np.uint64).max - 1, 1], 'UInt64', 'signed', 'UInt64'), ([1, 1], 'Int64', 'unsigned', 'UInt8'), ([1.0, pd.NA], 'Float64', 'unsigned', 'UInt8'), ([1.0, 1.1], 'Float64', 'unsigned', 'Float64'), ([1, pd.NA], 'Int64', 'unsigned', 'UInt8'), ([450, -300], 'Int64', 'unsigned', 'Int64'), ([-1, -1], 'Int32', 'unsigned', 'Int32'), ([1, 1], 'Float64', 'float', 'Float32'), ([1, 1.1], 'Float64', 'float', 'Float32'), ([1, 1], 'Float32', 'float', 'Float32'), ([1, 1.1], 'Float32', 'float', 'Float32')))
def test_downcast_nullable_numeric(data: list, input_dtype: str, downcast: str, expected_dtype: str) -> None:
    arr = pd.array(data, dtype=input_dtype)
    result = to_numeric(arr, downcast=downcast)
    expected = pd.array(data, dtype=expected_dtype)
    tm.assert_extension_array_equal(result, expected)

def test_downcast_nullable_mask_is_copied() -> None:
    arr = pd.array([1, 2, pd.NA], dtype='Int64')
    result = to_numeric(arr, downcast='integer')
    expected = pd.array([1, 2, pd.NA], dtype='Int8')
    tm.assert_extension_array_equal(result, expected)

def test_to_numeric_scientific_notation() -> None:
    result = to_numeric('1.7e+308')
    expected = np.float64(1.7e+308)
    assert result == expected

@pytest.mark.parametrize('val', [9876543210.0, 2.0 ** 128])
def test_to_numeric_large_float_not_downcast_to_float_32(val: float) -> None:
    expected = Series([val])
    result = to_numeric(expected, downcast='float')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('val, dtype', [(1, 'Int64'), (1.5, 'Float64'), (True, 'boolean')])
def test_to_numeric_dtype_backend(val: int, dtype: str) -> None:
    ser = Series([val], dtype=object)
    result = to_numeric(ser, dtype_backend='numpy_nullable')
    expected = Series([val], dtype=dtype)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('val, dtype', [(1, 'Int64'), (1.5, 'Float64'), (True, 'boolean'), (1, 'int64[pyarrow]