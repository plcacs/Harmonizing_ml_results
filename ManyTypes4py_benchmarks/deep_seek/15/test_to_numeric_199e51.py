import decimal
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import ArrowDtype, DataFrame, Index, Series, option_context, to_numeric
import pandas._testing as tm

@pytest.fixture(params=[None, 'raise', 'coerce'])
def errors(request: pytest.FixtureRequest) -> Optional[str]:
    return cast(Optional[str], request.param)

@pytest.fixture(params=[True, False])
def signed(request: pytest.FixtureRequest) -> bool:
    return cast(bool, request.param)

@pytest.fixture(params=[lambda x: x, str], ids=['identity', 'str'])
def transform(request: pytest.FixtureRequest) -> Callable[[Any], Any]:
    return cast(Callable[[Any], Any], request.param)

@pytest.fixture(params=[47393996303418497800, 100000000000000000000])
def large_val(request: pytest.FixtureRequest) -> int:
    return cast(int, request.param)

@pytest.fixture(params=[True, False])
def multiple_elts(request: pytest.FixtureRequest) -> bool:
    return cast(bool, request.param)

@pytest.fixture(params=[(lambda x: Index(x, name='idx'), tm.assert_index_equal), (lambda x: Series(x, name='ser'), tm.assert_series_equal), (lambda x: np.array(Index(x).values), tm.assert_numpy_array_equal)])
def transform_assert_equal(request: pytest.FixtureRequest) -> Tuple[Callable[[Any], Any], Callable[[Any, Any], None]:
    return cast(Tuple[Callable[[Any], Any], Callable[[Any, Any], None], request.param)

@pytest.mark.parametrize('input_kwargs,result_kwargs', [({}, {'dtype': np.int64}), ({'errors': 'coerce', 'downcast': 'integer'}, {'dtype': np.int8})])
def test_empty(input_kwargs: Dict[str, Any], result_kwargs: Dict[str, Any]) -> None:
    ser = Series([], dtype=object)
    result = to_numeric(ser, **input_kwargs)
    expected = Series([], **result_kwargs)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('last_val', ['7', 7])
def test_series(last_val: Union[str, int], infer_string: bool) -> None:
    with option_context('future.infer_string', infer_string):
        ser = Series(['1', '-3.14', last_val])
        result = to_numeric(ser)
    expected = Series([1, -3.14, 7])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('data', [[1, 3, 4, 5], [1.0, 3.0, 4.0, 5.0], [True, False, True, True]])
def test_series_numeric(data: List[Union[int, float, bool]]) -> None:
    ser = Series(data, index=list('ABCD'), name='EFG')
    result = to_numeric(ser)
    tm.assert_series_equal(result, ser)

@pytest.mark.parametrize('data,msg', [([1, -3.14, 'apple'], 'Unable to parse string "apple" at position 2'), (['orange', 1, -3.14, 'apple'], 'Unable to parse string "orange" at position 0')])
def test_error(data: List[Union[int, float, str]], msg: str) -> None:
    ser = Series(data)
    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, errors='raise')

def test_ignore_error() -> None:
    ser = Series([1, -3.14, 'apple'])
    result = to_numeric(ser, errors='coerce')
    expected = Series([1, -3.14, np.nan])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('errors,exp', [('raise', 'Unable to parse string "apple" at position 2'), ('coerce', [1.0, 0.0, np.nan])])
def test_bool_handling(errors: str, exp: Union[str, List[Union[float, type(np.nan)]]]) -> None:
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
def test_list_numeric(data: List[Union[int, float, bool]], arr_kwargs: Dict[str, Any]) -> None:
    result = to_numeric(data)
    expected = np.array(data, **arr_kwargs)
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize('kwargs', [{'dtype': 'O'}, {}])
def test_numeric(kwargs: Dict[str, Any]) -> None:
    data = [1, -3.14, 7]
    ser = Series(data, **kwargs)
    result = to_numeric(ser)
    expected = Series(data)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('columns', ['a', ['a', 'b']])
def test_numeric_df_columns(columns: Union[str, List[str]]) -> None:
    df = DataFrame({'a': [1.2, decimal.Decimal('3.14'), decimal.Decimal('infinity'), '0.1'], 'b': [1.0, 2.0, 3.0, 4.0]})
    expected = DataFrame({'a': [1.2, 3.14, np.inf, 0.1], 'b': [1.0, 2.0, 3.0, 4.0]})
    df[columns] = df[columns].apply(to_numeric)
    tm.assert_frame_equal(df, expected)

@pytest.mark.parametrize('data,exp_data', [([[decimal.Decimal('3.14'), 1.0], decimal.Decimal('1.6'), 0.1], [[3.14, 1.0], 1.6, 0.1]), ([np.array([decimal.Decimal('3.14'), 1.0]), 0.1], [[3.14, 1.0], 0.1])])
def test_numeric_embedded_arr_likes(data: List[Union[List[Union[decimal.Decimal, float]], decimal.Decimal, float]], exp_data: List[Union[List[float], float]]) -> None:
    df = DataFrame({'a': data})
    df['a'] = df['a'].apply(to_numeric)
    expected = DataFrame({'a': exp_data})
    tm.assert_frame_equal(df, expected)

def test_all_nan() -> None:
    ser = Series(['a', 'b', 'c'])
    result = to_numeric(ser, errors='coerce')
    expected = Series([np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

def test_type_check(errors: Optional[str]) -> None:
    df = DataFrame({'a': [1, -3.14, 7], 'b': ['4', '5', '6']})
    kwargs = {'errors': errors} if errors is not None else {}
    with pytest.raises(TypeError, match='1-d array'):
        to_numeric(df, **kwargs)

@pytest.mark.parametrize('val', [1, 1.1, 20001])
def test_scalar(val: Union[int, float], signed: bool, transform: Callable[[Any], Any]) -> None:
    val = -val if signed else val
    assert to_numeric(transform(val)) == float(val)

def test_really_large_scalar(large_val: int, signed: bool, transform: Callable[[Any], Any], errors: Optional[str]) -> None:
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

def test_really_large_in_arr(large_val: int, signed: bool, transform: Callable[[Any], Any], multiple_elts: bool, errors: Optional[str]) -> None:
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
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))

def test_really_large_in_arr_consistent(large_val: int, signed: bool, multiple_elts: bool, errors: Optional[str]) -> None:
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
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))

@pytest.mark.parametrize('errors,checker', [('raise', 'Unable to parse string "fail" at position 0'), ('coerce', lambda x: np.isnan(x))])
def test_scalar_fail(errors: str, checker: Union[str, Callable[[Any], bool]]) -> None:
    scalar = 'fail'
    if isinstance(checker, str):
        with pytest.raises(ValueError, match=checker):
            to_numeric(scalar, errors=errors)
    else:
        assert checker(to_numeric(scalar, errors=errors))

@pytest.mark.parametrize('data', [[1, 2, 3], [1.0, np.nan, 3, np.nan]])
def test_numeric_dtypes(data: List[Union[int, float, type(np.nan)]], transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    data = transform(data)
    result = to_numeric(data)
    assert_equal(result, data)

@pytest.mark.parametrize('data,exp', [(['1', '2', '3'], np.array([1, 2, 3], dtype='int64')), (['1.5', '2.7', '3.4'], np.array([1.5, 2.7, 3.4]))])
def test_str(data: List[str], exp: np.ndarray, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    result = to_numeric(transform(data))
    expected = transform(exp)
    assert_equal(result, expected)

def test_datetime_like(tz_naive_fixture: Any, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    idx = pd.date_range('20130101', periods=3, tz=tz_naive_fixture)
    result = to_numeric(transform(idx))
    expected = transform(idx.asi8)
    assert_equal(result, expected)

def test_timedelta(transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    idx = pd.timedelta_range('1 days', periods=3, freq='D')
    result = to_numeric(transform(idx))
    expected = transform(idx.asi8)
    assert_equal(result, expected)

@pytest.mark.parametrize('scalar', [pd.Timedelta(1, 'D'), pd.Timestamp('2017-01-01T12'), pd.Timestamp('2017-01-01T12', tz='US/Pacific')])
def test_timedelta_timestamp_scalar(scalar: Union[pd.Timedelta, pd.Timestamp]) -> None:
    result = to_numeric(scalar)
    expected = to_numeric(Series(scalar))[0]
    assert result == expected

def test_period(request: pytest.FixtureRequest, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    idx = pd.period_range('2011-01', periods=3, freq='M', name='')
    inp = transform(idx)
    if not isinstance(inp, Index):
        request.applymarker(pytest.mark.xfail(reason='Missing PeriodDtype support in to_numeric'))
    result = to_numeric(inp)
    expected = transform(idx.asi8)
    assert_equal(result, expected)

@pytest.mark.parametrize('errors,expected', [('raise', 'Invalid object type at position 0'), ('coerce', Series([np.nan, 1.0, np.nan]))])
def test_non_hashable(errors: str, expected: Union[str, Series]) -> None:
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

@pytest.mark.parametrize('data', [['1', 2, 3], [1, 2, 3], np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')])
@pytest.mark.parametrize('kwargs,exp_dtype', [({}, np.int64), ({'downcast': None}, np.int64), ({'downcast': 'float'}, np.dtype(np.float32).char), ({'downcast': 'unsigned'}, np.dtype(np.typecodes['UnsignedInteger'][0]))])
def test_downcast_basic(data: Union[List[Union[str, int]], np.ndarray], kwargs: Dict[str, Any], exp_dtype: np.dtype) -> None:
    result = to_numeric(data, **kwargs)
    expected = np.array([1, 2, 3], dtype=exp_dtype)
    tm.assert_numpy_array_equal(result, expected)

@pytest.mark.parametrize('signed_downcast', ['integer', 'signed'])
@pytest.mark.parametrize('data', [['1', 2, 3], [1, 2, 3], np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')])
def test_signed_downcast(data: Union[List[Union[str, int]], np.ndarray], signed_downcast: str) -> None:
    smallest_int_dtype = np.dtype(np.typecodes['Integer'][0])
