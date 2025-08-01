#!/usr/bin/env python3
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import decimal
import numpy as np
from numpy import iinfo
import pandas as pd
import pytest
from pandas import ArrowDtype, DataFrame, Index, Series, option_context, to_numeric
import pandas._testing as tm
import pandas.util._test_decorators as td


@pytest.fixture(params=[None, 'raise', 'coerce'])
def errors(request: pytest.FixtureRequest) -> Optional[str]:
    return request.param


@pytest.fixture(params=[True, False])
def signed(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[lambda x: x, str], ids=['identity', 'str'])
def transform(request: pytest.FixtureRequest) -> Callable[[Any], Any]:
    return request.param


@pytest.fixture(params=[47393996303418497800, 100000000000000000000])
def large_val(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[True, False])
def multiple_elts(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(
    params=[
        (lambda x: Index(x, name='idx'), tm.assert_index_equal),
        (lambda x: Series(x, name='ser'), tm.assert_series_equal),
        (lambda x: np.array(Index(x).values), tm.assert_numpy_array_equal),
    ]
)
def transform_assert_equal(request: pytest.FixtureRequest) -> Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]:
    return request.param


@pytest.mark.parametrize('input_kwargs,result_kwargs', [({}, {'dtype': np.int64}), ({'errors': 'coerce', 'downcast': 'integer'}, {'dtype': np.int8})])
def test_empty(input_kwargs: Dict[str, Any], result_kwargs: Dict[str, Any]) -> None:
    ser: Series = Series([], dtype=object)
    result: Series = to_numeric(ser, **input_kwargs)
    expected: Series = Series([], **result_kwargs)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('last_val', ['7', 7])
def test_series(last_val: Union[str, int], infer_string: bool) -> None:
    with option_context('future.infer_string', infer_string):
        ser: Series = Series(['1', '-3.14', last_val])
        result: Series = to_numeric(ser)
    expected: Series = Series([1, -3.14, 7])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('data', [[1, 3, 4, 5], [1.0, 3.0, 4.0, 5.0], [True, False, True, True]])
def test_series_numeric(data: List[Any]) -> None:
    ser: Series = Series(data, index=list('ABCD'), name='EFG')
    result: Series = to_numeric(ser)
    tm.assert_series_equal(result, ser)


@pytest.mark.parametrize('data,msg', [([1, -3.14, 'apple'], 'Unable to parse string "apple" at position 2'), (['orange', 1, -3.14, 'apple'], 'Unable to parse string "orange" at position 0')])
def test_error(data: List[Any], msg: str) -> None:
    ser: Series = Series(data)
    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, errors='raise')


def test_ignore_error() -> None:
    ser: Series = Series([1, -3.14, 'apple'])
    result: Series = to_numeric(ser, errors='coerce')
    expected: Series = Series([1, -3.14, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('errors,exp', [('raise', 'Unable to parse string "apple" at position 2'), ('coerce', [1.0, 0.0, np.nan])])
def test_bool_handling(errors: str, exp: Union[str, List[float]]) -> None:
    ser: Series = Series([True, False, 'apple'])
    if isinstance(exp, str):
        with pytest.raises(ValueError, match=exp):
            to_numeric(ser, errors=errors)
    else:
        result: Series = to_numeric(ser, errors=errors)
        expected: Series = Series(exp)
        tm.assert_series_equal(result, expected)


def test_list() -> None:
    ser: List[str] = ['1', '-3.14', '7']
    res: Any = to_numeric(ser)
    expected: np.ndarray = np.array([1, -3.14, 7])
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize('data,arr_kwargs', [([1, 3, 4, 5], {'dtype': np.int64}), ([1.0, 3.0, 4.0, 5.0], {}), ([True, False, True, True], {})])
def test_list_numeric(data: List[Any], arr_kwargs: Dict[str, Any]) -> None:
    result: Any = to_numeric(data)
    expected: np.ndarray = np.array(data, **arr_kwargs)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize('kwargs', [{'dtype': 'O'}, {}])
def test_numeric(kwargs: Dict[str, Any]) -> None:
    data: List[Any] = [1, -3.14, 7]
    ser: Series = Series(data, **kwargs)
    result: Series = to_numeric(ser)
    expected: Series = Series(data)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('columns', ['a', ['a', 'b']])
def test_numeric_df_columns(columns: Union[str, List[str]]) -> None:
    df: DataFrame = DataFrame({
        'a': [1.2, decimal.Decimal('3.14'), decimal.Decimal('infinity'), '0.1'],
        'b': [1.0, 2.0, 3.0, 4.0]
    })
    expected: DataFrame = DataFrame({
        'a': [1.2, 3.14, np.inf, 0.1],
        'b': [1.0, 2.0, 3.0, 4.0]
    })
    df[columns] = df[columns].apply(to_numeric)
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize('data,exp_data', [
    ([[decimal.Decimal('3.14'), 1.0], decimal.Decimal('1.6'), 0.1], [[3.14, 1.0], 1.6, 0.1]),
    ([np.array([decimal.Decimal('3.14'), 1.0]), 0.1], [[3.14, 1.0], 0.1])
])
def test_numeric_embedded_arr_likes(data: List[Any], exp_data: Any) -> None:
    df: DataFrame = DataFrame({'a': data})
    df['a'] = df['a'].apply(to_numeric)
    expected: DataFrame = DataFrame({'a': exp_data})
    tm.assert_frame_equal(df, expected)


def test_all_nan() -> None:
    ser: Series = Series(['a', 'b', 'c'])
    result: Series = to_numeric(ser, errors='coerce')
    expected: Series = Series([np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_type_check(errors: Optional[str]) -> None:
    df: DataFrame = DataFrame({'a': [1, -3.14, 7], 'b': ['4', '5', '6']})
    kwargs: Dict[str, Any] = {'errors': errors} if errors is not None else {}
    with pytest.raises(TypeError, match='1-d array'):
        to_numeric(df, **kwargs)


@pytest.mark.parametrize('val', [1, 1.1, 20001])
def test_scalar(val: Union[int, float], signed: bool, transform: Callable[[Any], Any]) -> None:
    val = -val if signed else val
    result: Any = to_numeric(transform(val))
    assert result == float(val)


def test_really_large_scalar(large_val: int, signed: bool, transform: Callable[[Any], Any], errors: Optional[str]) -> None:
    kwargs: Dict[str, Any] = {'errors': errors} if errors is not None else {}
    val: Union[int, float] = -large_val if signed else large_val
    val = transform(val)
    val_is_string: bool = isinstance(val, str)
    if val_is_string and errors in (None, 'raise'):
        msg: str = 'Integer out of range. at position 0'
        with pytest.raises(ValueError, match=msg):
            to_numeric(val, **kwargs)
    else:
        expected: Any = float(val) if errors == 'coerce' and val_is_string else val
        tm.assert_almost_equal(to_numeric(val, **kwargs), expected)


def test_really_large_in_arr(large_val: int, signed: bool, transform: Callable[[Any], Any], multiple_elts: bool, errors: Optional[str]) -> None:
    kwargs: Dict[str, Any] = {'errors': errors} if errors is not None else {}
    val: Union[int, float] = -large_val if signed else large_val
    val = transform(val)
    extra_elt: str = 'string'
    arr: List[Any] = [val] + multiple_elts * [extra_elt]
    val_is_string: bool = isinstance(val, str)
    coercing: bool = errors == 'coerce'
    if errors in (None, 'raise') and (val_is_string or multiple_elts):
        if val_is_string:
            msg: str = 'Integer out of range. at position 0'
        else:
            msg: str = 'Unable to parse string "string" at position 1'
        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        result: Any = to_numeric(arr, **kwargs)
        exp_val: Any = float(val) if coercing and val_is_string else val
        expected: List[Any] = [exp_val]
        if multiple_elts:
            if coercing:
                expected.append(np.nan)
                exp_dtype: Any = float
            else:
                expected.append(extra_elt)
                exp_dtype = object
        else:
            exp_dtype = float if isinstance(exp_val, (int, float)) else object
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))


def test_really_large_in_arr_consistent(large_val: int, signed: bool, multiple_elts: bool, errors: Optional[str]) -> None:
    kwargs: Dict[str, Any] = {'errors': errors} if errors is not None else {}
    arr: List[Any] = [str(-large_val if signed else large_val)]
    if multiple_elts:
        arr.insert(0, large_val)
    if errors in (None, 'raise'):
        index: int = int(multiple_elts)
        msg: str = f'Integer out of range. at position {index}'
        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        result: Any = to_numeric(arr, **kwargs)
        expected: List[float] = [float(i) for i in arr]
        exp_dtype: Any = float
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))


@pytest.mark.parametrize('errors,checker', [
    ('raise', 'Unable to parse string "fail" at position 0'),
    ('coerce', lambda x: np.isnan(x))
])
def test_scalar_fail(errors: str, checker: Union[str, Callable[[Any], bool]]) -> None:
    scalar: str = 'fail'
    if isinstance(checker, str):
        with pytest.raises(ValueError, match=checker):
            to_numeric(scalar, errors=errors)
    else:
        result: Any = to_numeric(scalar, errors=errors)
        assert checker(result)


@pytest.mark.parametrize('data', [[1, 2, 3], [1.0, np.nan, 3, np.nan]])
def test_numeric_dtypes(data: List[Any], transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    data_transformed: Any = transform(data)
    result: Any = to_numeric(data_transformed)
    assert_equal(result, data_transformed)


@pytest.mark.parametrize('data,exp', [
    (['1', '2', '3'], np.array([1, 2, 3], dtype='int64')),
    (['1.5', '2.7', '3.4'], np.array([1.5, 2.7, 3.4]))
])
def test_str(data: List[str], exp: np.ndarray, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    result: Any = to_numeric(transform(data))
    expected: Any = transform(exp)
    assert_equal(result, expected)


def test_datetime_like(tz_naive_fixture: Any, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    idx: Index = pd.date_range('20130101', periods=3, tz=tz_naive_fixture)
    result: Any = to_numeric(transform(idx))
    expected: Any = transform(idx.asi8)
    assert_equal(result, expected)


def test_timedelta(transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    idx: Index = pd.timedelta_range('1 days', periods=3, freq='D')
    result: Any = to_numeric(transform(idx))
    expected: Any = transform(idx.asi8)
    assert_equal(result, expected)


@pytest.mark.parametrize('scalar', [pd.Timedelta(1, 'D'), pd.Timestamp('2017-01-01T12'), pd.Timestamp('2017-01-01T12', tz='US/Pacific')])
def test_timedelta_timestamp_scalar(scalar: Union[pd.Timedelta, pd.Timestamp]) -> None:
    result: Any = to_numeric(scalar)
    expected: Any = to_numeric(Series(scalar))[0]
    assert result == expected


def test_period(request: pytest.FixtureRequest, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    transform, assert_equal = transform_assert_equal
    idx: Index = pd.period_range('2011-01', periods=3, freq='M', name='')
    inp: Any = transform(idx)
    if not isinstance(inp, Index):
        request.applymarker(pytest.mark.xfail(reason='Missing PeriodDtype support in to_numeric'))
    result: Any = to_numeric(inp)
    expected: Any = transform(idx.asi8)
    assert_equal(result, expected)


@pytest.mark.parametrize('errors,expected', [
    ('raise', 'Invalid object type at position 0'),
    ('coerce', Series([np.nan, 1.0, np.nan]))
])
def test_non_hashable(errors: str, expected: Union[str, Series]) -> None:
    ser: Series = Series([[10.0, 2], 1.0, 'apple'])
    if isinstance(expected, str):
        with pytest.raises(TypeError, match=expected):
            to_numeric(ser, errors=errors)
    else:
        result: Series = to_numeric(ser, errors=errors)
        tm.assert_series_equal(result, expected)


def test_downcast_invalid_cast() -> None:
    data: List[Any] = ['1', 2, 3]
    invalid_downcast: str = 'unsigned-integer'
    msg: str = 'invalid downcasting method provided'
    with pytest.raises(ValueError, match=msg):
        to_numeric(data, downcast=invalid_downcast)


def test_errors_invalid_value() -> None:
    data: List[Any] = ['1', 2, 3]
    invalid_error_value: str = 'invalid'
    msg: str = 'invalid error value specified'
    with pytest.raises(ValueError, match=msg):
        to_numeric(data, errors=invalid_error_value)


@pytest.mark.parametrize('data', [
    ['1', 2, 3],
    [1, 2, 3],
    np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')
])
@pytest.mark.parametrize('kwargs,exp_dtype', [
    ({}, np.int64),
    ({'downcast': None}, np.int64),
    ({'downcast': 'float'}, np.dtype(np.float32).char),
    ({'downcast': 'unsigned'}, np.dtype(np.typecodes['UnsignedInteger'][0]))
])
def test_downcast_basic(data: Any, kwargs: Dict[str, Any], exp_dtype: Any) -> None:
    result: np.ndarray = to_numeric(data, **kwargs)
    expected: np.ndarray = np.array([1, 2, 3], dtype=exp_dtype)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize('signed_downcast', ['integer', 'signed'])
@pytest.mark.parametrize('data', [
    ['1', 2, 3],
    [1, 2, 3],
    np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')
])
def test_signed_downcast(data: Any, signed_downcast: str) -> None:
    smallest_int_dtype: Any = np.dtype(np.typecodes['Integer'][0])
    expected: np.ndarray = np.array([1, 2, 3], dtype=smallest_int_dtype)
    res: np.ndarray = to_numeric(data, downcast=signed_downcast)
    tm.assert_numpy_array_equal(res, expected)


def test_ignore_downcast_neg_to_unsigned() -> None:
    data: List[Any] = ['-1', 2, 3]
    expected: np.ndarray = np.array([-1, 2, 3], dtype=np.int64)
    res: np.ndarray = to_numeric(data, downcast='unsigned')
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize('downcast', ['integer', 'signed', 'unsigned'])
@pytest.mark.parametrize('data,expected', [
    (['1.1', 2, 3], np.array([1.1, 2, 3], dtype=np.float64)),
    ([10000.0, 20000, 3000, 40000.36, 50000, 50000.0], np.array([10000.0, 20000, 3000, 40000.36, 50000, 50000.0], dtype=np.float64))
])
def test_ignore_downcast_cannot_convert_float(data: List[Any], expected: np.ndarray, downcast: str) -> None:
    res: Any = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize('downcast,expected_dtype', [
    ('integer', np.int16),
    ('signed', np.int16),
    ('unsigned', np.uint16)
])
def test_downcast_not8bit(downcast: str, expected_dtype: Any) -> None:
    data: List[Any] = ['256', 257, 258]
    expected: np.ndarray = np.array([256, 257, 258], dtype=expected_dtype)
    res: np.ndarray = to_numeric(data, downcast=downcast)
    tm.assert_numpy_array_equal(res, expected)


@pytest.mark.parametrize('dtype,downcast,min_max', [
    ('int8', 'integer', [iinfo(np.int8).min, iinfo(np.int8).max]),
    ('int16', 'integer', [iinfo(np.int16).min, iinfo(np.int16).max]),
    ('int32', 'integer', [iinfo(np.int32).min, iinfo(np.int32).max]),
    ('int64', 'integer', [iinfo(np.int64).min, iinfo(np.int64).max]),
    ('uint8', 'unsigned', [iinfo(np.uint8).min, iinfo(np.uint8).max]),
    ('uint16', 'unsigned', [iinfo(np.uint16).min, iinfo(np.uint16).max]),
    ('uint32', 'unsigned', [iinfo(np.uint32).min, iinfo(np.uint32).max]),
    ('uint64', 'unsigned', [iinfo(np.uint64).min, iinfo(np.uint64).max]),
    ('int16', 'integer', [iinfo(np.int8).min, iinfo(np.int8).max + 1]),
    ('int32', 'integer', [iinfo(np.int16).min, iinfo(np.int16).max + 1]),
    ('int64', 'integer', [iinfo(np.int32).min, iinfo(np.int32).max + 1]),
    ('int16', 'integer', [iinfo(np.int8).min - 1, iinfo(np.int16).max]),
    ('int32', 'integer', [iinfo(np.int16).min - 1, iinfo(np.int32).max]),
    ('int64', 'integer', [iinfo(np.int32).min - 1, iinfo(np.int64).max]),
    ('uint16', 'unsigned', [iinfo(np.uint8).min, iinfo(np.uint8).max + 1]),
    ('uint32', 'unsigned', [iinfo(np.uint16).min, iinfo(np.uint16).max + 1]),
    ('uint64', 'unsigned', [iinfo(np.uint32).min, iinfo(np.uint32).max + 1])
])
def test_downcast_limits(dtype: str, downcast: str, min_max: List[Any]) -> None:
    series: Series = to_numeric(Series(min_max), downcast=downcast)
    assert series.dtype == dtype


def test_downcast_float64_to_float32() -> None:
    series: Series = Series([16777217.0, np.finfo(np.float64).max, np.nan], dtype=np.float64)
    result: Series = to_numeric(series, downcast='float')
    assert series.dtype == result.dtype


def test_downcast_uint64() -> None:
    ser: Series = Series([0, 9223372036854775808])
    result: Series = to_numeric(ser, downcast='unsigned')
    expected: Series = Series([0, 9223372036854775808], dtype=np.uint64)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('data,exp_data', [
    ([200, 300, '', 'NaN', 30000000000000000000], [200, 300, np.nan, np.nan, 30000000000000000000]),
    (['12345678901234567890', '1234567890', 'ITEM'], [12345678901234567890, 1234567890, np.nan])
])
def test_coerce_uint64_conflict(data: List[Any], exp_data: List[Any]) -> None:
    result: Series = to_numeric(Series(data), errors='coerce')
    expected: Series = Series(exp_data, dtype=float)
    tm.assert_series_equal(result, expected)


def test_non_coerce_uint64_conflict() -> None:
    ser: Series = Series(['12345678901234567890', '1234567890', 'ITEM'])
    with pytest.raises(ValueError, match='Unable to parse string'):
        to_numeric(ser, errors='raise')


@pytest.mark.parametrize('dc1', ['integer', 'float', 'unsigned'])
@pytest.mark.parametrize('dc2', ['integer', 'float', 'unsigned'])
def test_downcast_empty(dc1: str, dc2: str) -> None:
    tm.assert_numpy_array_equal(to_numeric([], downcast=dc1), to_numeric([], downcast=dc2), check_dtype=False)


def test_failure_to_convert_uint64_string_to_NaN() -> None:
    result: Any = to_numeric('uint64', errors='coerce')
    assert np.isnan(result)
    ser: Series = Series([32, 64, np.nan])
    result = to_numeric(Series(['32', '64', 'uint64']), errors='coerce')
    tm.assert_series_equal(result, ser)


@pytest.mark.parametrize('strrep', [
    '243.164', '245.968', '249.585', '259.745', '265.742', '272.567',
    '279.196', '280.366', '275.034', '271.351', '272.889', '270.627',
    '280.828', '290.383', '308.153', '319.945', '336.0', '344.09',
    '351.385', '356.178', '359.82', '361.03', '367.701', '380.812',
    '387.98', '391.749', '391.171', '385.97', '385.345', '386.121',
    '390.996', '399.734', '413.073', '421.532', '430.221', '437.092',
    '439.746', '446.01', '451.191', '460.463', '469.779', '472.025',
    '479.49', '474.864', '467.54', '471.978'
])
def test_precision_float_conversion(strrep: str) -> None:
    result: Any = to_numeric(strrep)
    assert result == float(strrep)


@pytest.mark.parametrize('values, expected', [
    (['1', '2', None], Series([1, 2, np.nan], dtype='Int64')),
    (['1', '2', '3'], Series([1, 2, 3], dtype='Int64')),
    (['1', '2', 3], Series([1, 2, 3], dtype='Int64')),
    (['1', '2', 3.5], Series([1, 2, 3.5], dtype='Float64')),
    (['1', None, 3.5], Series([1, np.nan, 3.5], dtype='Float64')),
    (['1', '2', '3.5'], Series([1, 2, 3.5], dtype='Float64'))
])
def test_to_numeric_from_nullable_string(values: List[Any], nullable_string_dtype: str, expected: Series) -> None:
    s: Series = Series(values, dtype=nullable_string_dtype)
    result: Series = to_numeric(s)
    tm.assert_series_equal(result, expected)


def test_to_numeric_from_nullable_string_coerce(nullable_string_dtype: str) -> None:
    values: List[str] = ['a', '1']
    ser: Series = Series(values, dtype=nullable_string_dtype)
    result: Series = to_numeric(ser, errors='coerce')
    expected: Series = Series([pd.NA, 1], dtype='Int64')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('data, input_dtype, downcast, expected_dtype', (
    ([1, 1], 'Int64', 'integer', 'Int8'),
    ([1.0, pd.NA], 'Float64', 'integer', 'Int8'),
    ([1.0, 1.1], 'Float64', 'integer', 'Float64'),
    ([1, pd.NA], 'Int64', 'integer', 'Int8'),
    ([450, 300], 'Int64', 'integer', 'Int16'),
    ([1, 1], 'Float64', 'integer', 'Int8'),
    ([np.iinfo(np.int64).max - 1, 1], 'Int64', 'integer', 'Int64'),
    ([1, 1], 'Int64', 'signed', 'Int8'),
    ([1.0, 1.0], 'Float32', 'signed', 'Int8'),
    ([1.0, 1.1], 'Float64', 'signed', 'Float64'),
    ([1, pd.NA], 'Int64', 'signed', 'Int8'),
    ([450, -300], 'Int64', 'signed', 'Int16'),
    ([np.iinfo(np.uint64).max - 1, 1], 'UInt64', 'signed', 'UInt64'),
    ([1, 1], 'Int64', 'unsigned', 'UInt8'),
    ([1.0, 1.0], 'Float32', 'unsigned', 'UInt8'),
    ([1.0, 1.1], 'Float64', 'unsigned', 'Float64'),
    ([1, pd.NA], 'Int64', 'unsigned', 'UInt8'),
    ([450, -300], 'Int64', 'unsigned', 'Int64'),
    ([-1, -1], 'Int32', 'unsigned', 'Int32'),
    ([1, 1], 'Float64', 'float', 'Float32'),
    ([1, 1.1], 'Float64', 'float', 'Float32'),
    ([1, 1], 'Float32', 'float', 'Float32'),
    ([1, 1.1], 'Float32', 'float', 'Float32')
))
def test_downcast_nullable_numeric(data: List[Any], input_dtype: str, downcast: str, expected_dtype: str) -> None:
    arr = pd.array(data, dtype=input_dtype)
    result = to_numeric(arr, downcast=downcast)
    expected = pd.array(data, dtype=expected_dtype)
    tm.assert_extension_array_equal(result, expected)


def test_downcast_nullable_mask_is_copied() -> None:
    arr = pd.array([1, 2, pd.NA], dtype='Int64')
    result = to_numeric(arr, downcast='integer')
    expected = pd.array([1, 2, pd.NA], dtype='Int8')
    tm.assert_extension_array_equal(result, expected)
    arr[1] = pd.NA
    tm.assert_extension_array_equal(result, expected)


def test_to_numeric_scientific_notation() -> None:
    result: Any = to_numeric('1.7e+308')
    expected: np.float64 = np.float64(1.7e+308)
    assert result == expected


@pytest.mark.parametrize('val', [9876543210.0, 2.0 ** 128])
def test_to_numeric_large_float_not_downcast_to_float_32(val: float) -> None:
    expected: Series = Series([val])
    result: Series = to_numeric(expected, downcast='float')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('val, dtype', [(1, 'Int64'), (1.5, 'Float64'), (True, 'boolean')])
def test_to_numeric_dtype_backend(val: Union[int, float, bool], dtype: str) -> None:
    ser: Series = Series([val], dtype=object)
    result: Series = to_numeric(ser, dtype_backend='numpy_nullable')
    expected: Series = Series([val], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('val, dtype', [
    (1, 'Int64'), (1.5, 'Float64'), (True, 'boolean'),
    (1, 'int64[pyarrow]'), (1.5, 'float64[pyarrow]'), (True, 'bool[pyarrow]')
])
def test_to_numeric_dtype_backend_na(val: Union[int, float, bool], dtype: str) -> None:
    if 'pyarrow' in dtype:
        pytest.importorskip('pyarrow')
        dtype_backend: str = 'pyarrow'
    else:
        dtype_backend = 'numpy_nullable'
    ser: Series = Series([val, None], dtype=object)
    result: Series = to_numeric(ser, dtype_backend=dtype_backend)
    expected: Series = Series([val, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('val, dtype, downcast', [
    (1, 'Int8', 'integer'),
    (1.5, 'Float32', 'float'),
    (1, 'Int8', 'signed'),
    (1, 'int8[pyarrow]', 'integer'),
    (1.5, 'float[pyarrow]', 'float'),
    (1, 'int8[pyarrow]', 'signed')
])
def test_to_numeric_dtype_backend_downcasting(val: Union[int, float], dtype: str, downcast: str) -> None:
    if 'pyarrow' in dtype:
        pytest.importorskip('pyarrow')
        dtype_backend: str = 'pyarrow'
    else:
        dtype_backend = 'numpy_nullable'
    ser: Series = Series([val, None], dtype=object)
    result: Series = to_numeric(ser, dtype_backend=dtype_backend, downcast=downcast)
    expected: Series = Series([val, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('smaller, dtype_backend', [
    ['UInt8', 'numpy_nullable'],
    ['uint8[pyarrow]', 'pyarrow']
])
def test_to_numeric_dtype_backend_downcasting_uint(smaller: str, dtype_backend: str) -> None:
    if dtype_backend == 'pyarrow':
        pytest.importorskip('pyarrow')
    ser: Series = Series([1, pd.NA], dtype='UInt64')
    result: Series = to_numeric(ser, dtype_backend=dtype_backend, downcast='unsigned')
    expected: Series = Series([1, pd.NA], dtype=smaller)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('dtype', [
    'Int64', 'UInt64', 'Float64', 'boolean',
    'int64[pyarrow]', 'uint64[pyarrow]', 'float64[pyarrow]', 'bool[pyarrow]'
])
def test_to_numeric_dtype_backend_already_nullable(dtype: str) -> None:
    if 'pyarrow' in dtype:
        pytest.importorskip('pyarrow')
    ser: Series = Series([1, pd.NA], dtype=dtype)
    result: Series = to_numeric(ser, dtype_backend='numpy_nullable')
    expected: Series = Series([1, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_to_numeric_dtype_backend_error(dtype_backend: str = 'numpy_nullable') -> None:
    ser: Series = Series(['a', 'b', ''])
    expected: Series = ser.copy()
    with pytest.raises(ValueError, match='Unable to parse string'):
        to_numeric(ser, dtype_backend=dtype_backend)
    result: Series = to_numeric(ser, dtype_backend=dtype_backend, errors='coerce')
    if dtype_backend == 'pyarrow':
        dtype: str = 'double[pyarrow]'
    else:
        dtype = 'Float64'
    expected = Series([np.nan, np.nan, np.nan], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_invalid_dtype_backend() -> None:
    ser: Series = Series([1, 2, 3])
    msg: str = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        to_numeric(ser, dtype_backend='numpy')


def test_coerce_pyarrow_backend() -> None:
    pa = pytest.importorskip('pyarrow')
    ser: Series = Series(list('12x'), dtype=ArrowDtype(pa.string()))
    result: Series = to_numeric(ser, errors='coerce', dtype_backend='pyarrow')
    expected: Series = Series([1, 2, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)