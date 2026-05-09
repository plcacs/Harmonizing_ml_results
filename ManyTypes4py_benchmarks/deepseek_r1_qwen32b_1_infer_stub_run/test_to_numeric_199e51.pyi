from __future__ import annotations
from decimal import Decimal
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)
import numpy as np
import pytest
from pandas import (
    ArrowDtype,
    DataFrame,
    Index,
    Series,
    to_numeric,
)
from pandas._testing import (
    tm,
)

@pytest.fixture
def errors(request: Any) -> Optional[str]:
    ...

@pytest.fixture
def signed(request: Any) -> bool:
    ...

@pytest.fixture
def transform(request: Any) -> Callable[[Any], Union[str, Any]]:
    ...

@pytest.fixture
def large_val(request: Any) -> int:
    ...

@pytest.fixture
def multiple_elts(request: Any) -> bool:
    ...

@pytest.fixture
def transform_assert_equal(request: Any) -> Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]:
    ...

@pytest.mark.parametrize('input_kwargs,result_kwargs', [({}, {'dtype': np.int64}), ({'errors': 'coerce', 'downcast': 'integer'}, {'dtype': np.int8})])
def test_empty(input_kwargs: Dict[str, Any], result_kwargs: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
@pytest.mark.parametrize('last_val', ['7', 7])
def test_series(last_val: Union[str, int], infer_string: bool) -> None:
    ...

@pytest.mark.parametrize('data', [[1, 3, 4, 5], [1.0, 3.0, 4.0, 5.0], [True, False, True, True]])
def test_series_numeric(data: List[Union[int, float, bool]]) -> None:
    ...

@pytest.mark.parametrize('data,msg', [([1, -3.14, 'apple'], 'Unable to parse string "apple" at position 2'), (['orange', 1, -3.14, 'apple'], 'Unable to parse string "orange" at position 0')])
def test_error(data: List[Union[int, float, str]], msg: str) -> None:
    ...

def test_ignore_error() -> None:
    ...

@pytest.mark.parametrize('errors,exp', [('raise', 'Unable to parse string "apple" at position 2'), ('coerce', [1.0, 0.0, np.nan])])
def test_bool_handling(errors: str, exp: Union[str, List[float]]) -> None:
    ...

def test_list() -> None:
    ...

@pytest.mark.parametrize('data,arr_kwargs', [([1, 3, 4, 5], {'dtype': np.int64}), ([1.0, 3.0, 4.0, 5.0], {}), ([True, False, True, True], {})])
def test_list_numeric(data: List[Union[int, float, bool]], arr_kwargs: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('kwargs', [{'dtype': 'O'}, {}])
def test_numeric(kwargs: Dict[str, str]) -> None:
    ...

@pytest.mark.parametrize('columns', ['a', ['a', 'b']])
def test_numeric_df_columns(columns: Union[str, List[str]]) -> None:
    ...

@pytest.mark.parametrize('data,exp_data', [([[decimal.Decimal('3.14'), 1.0], decimal.Decimal('1.6'), 0.1], [[3.14, 1.0], 1.6, 0.1]), ([np.array([decimal.Decimal('3.14'), 1.0]), 0.1], [[3.14, 1.0], 0.1])])
def test_numeric_embedded_arr_likes(data: List[Union[List[decimal.Decimal], decimal.Decimal, float]], exp_data: List[Union[List[float], float]]) -> None:
    ...

def test_all_nan() -> None:
    ...

def test_type_check(errors: Optional[str]) -> None:
    ...

@pytest.mark.parametrize('val', [1, 1.1, 20001])
def test_scalar(val: Union[int, float], signed: bool, transform: Callable[[Any], Union[str, Any]]) -> None:
    ...

def test_really_large_scalar(large_val: int, signed: bool, transform: Callable[[Any], Union[str, Any]], errors: Optional[str]) -> None:
    ...

def test_really_large_in_arr(large_val: int, signed: bool, transform: Callable[[Any], Union[str, Any]], multiple_elts: bool, errors: Optional[str]) -> None:
    ...

def test_really_large_in_arr_consistent(large_val: int, signed: bool, multiple_elts: bool, errors: Optional[str]) -> None:
    ...

@pytest.mark.parametrize('errors,checker', [('raise', 'Unable to parse string "fail" at position 0'), ('coerce', lambda x: np.isnan(x))])
def test_scalar_fail(errors: str, checker: Union[str, Callable[[Any], bool]]) -> None:
    ...

@pytest.mark.parametrize('data', [[1, 2, 3], [1.0, np.nan, 3, np.nan]])
def test_numeric_dtypes(data: List[Union[int, float, np.nan]], transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...

@pytest.mark.parametrize('data,exp', [(['1', '2', '3'], np.array([1, 2, 3], dtype='int64')), (['1.5', '2.7', '3.4'], np.array([1.5, 2.7, 3.4]))])
def test_str(data: List[str], exp: np.ndarray, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...

def test_datetime_like(tz_naive_fixture: Any, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...

def test_timedelta(transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...

@pytest.mark.parametrize('scalar', [pd.Timedelta(1, 'D'), pd.Timestamp('2017-01-01T12'), pd.Timestamp('2017-01-01T12', tz='US/Pacific')])
def test_timedelta_timestamp_scalar(scalar: Union[pd.Timedelta, pd.Timestamp]) -> None:
    ...

def test_period(request: pytest.FixtureRequest, transform_assert_equal: Tuple[Callable[[Any], Any], Callable[[Any, Any], None]]) -> None:
    ...

@pytest.mark.parametrize('errors,expected', [('raise', 'Invalid object type at position 0'), ('coerce', Series([np.nan, 1.0, np.nan]))])
def test_non_hashable(errors: str, expected: Union[str, Series]) -> None:
    ...

def test_downcast_invalid_cast() -> None:
    ...

def test_errors_invalid_value() -> None:
    ...

@pytest.mark.parametrize('data', [['1', 2, 3], [1, 2, 3], np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')])
@pytest.mark.parametrize('kwargs,exp_dtype', [({}, np.int64), ({'downcast': None}, np.int64), ({'downcast': 'float'}, np.dtype(np.float32).char), ({'downcast': 'unsigned'}, np.dtype(np.typecodes['UnsignedInteger'][0]))])
def test_downcast_basic(data: Union[List[Union[str, int]], np.ndarray], kwargs: Dict[str, Any], exp_dtype: np.dtype) -> None:
    ...

@pytest.mark.parametrize('signed_downcast', ['integer', 'signed'])
@pytest.mark.parametrize('data', [['1', 2, 3], [1, 2, 3], np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')])
def test_signed_downcast(data: Union[List[Union[str, int]], np.ndarray], signed_downcast: str) -> None:
    ...

def test_ignore_downcast_neg_to_unsigned() -> None:
    ...

@pytest.mark.parametrize('downcast', ['integer', 'signed', 'unsigned'])
@pytest.mark.parametrize('data,expected', [(['1.1', 2, 3], np.array([1.1, 2, 3], dtype=np.float64)), ([10000.0, 20000, 3000, 40000.36, 50000, 50000.0], np.array([10000.0, 20000, 3000, 40000.36, 50000, 50000.0], dtype=np.float64))])
def test_ignore_downcast_cannot_convert_float(data: List[Union[str, int, float]], expected: np.ndarray, downcast: str) -> None:
    ...

@pytest.mark.parametrize('downcast,expected_dtype', [('integer', np.int16), ('signed', np.int16), ('unsigned', np.uint16)])
def test_downcast_not8bit(downcast: str, expected_dtype: np.dtype) -> None:
    ...

@pytest.mark.parametrize('dtype,downcast,min_max', [('int8', 'integer', [iinfo(np.int8).min, iinfo(np.int8).max]), ('int16', 'integer', [iinfo(np.int16).min, iinfo(np.int16).max]), ('int32', 'integer', [iinfo(np.int32).min, iinfo(np.int32).max]), ('int64', 'integer', [iinfo(np.int64).min, iinfo(np.int64).max]), ('uint8', 'unsigned', [iinfo(np.uint8).min, iinfo(np.uint8).max]), ('uint16', 'unsigned', [iinfo(np.uint16).min, iinfo(np.uint16).max]), ('uint32', 'unsigned', [iinfo(np.uint32).min, iinfo(np.uint32).max]), ('uint64', 'unsigned', [iinfo(np.uint64).min, iinfo(np.uint64).max]), ('int16', 'integer', [iinfo(np.int8).min, iinfo(np.int8).max + 1]), ('int32', 'integer', [iinfo(np.int16).min, iinfo(np.int16).max + 1]), ('int64', 'integer', [iinfo(np.int32).min, iinfo(np.int32).max + 1]), ('int16', 'integer', [iinfo(np.int8).min - 1, iinfo(np.int16).max]), ('int32', 'integer', [iinfo(np.int16).min - 1, iinfo(np.int32).max]), ('int64', 'integer', [iinfo(np.int32).min - 1, iinfo(np.int64).max]), ('uint16', 'unsigned', [iinfo(np.uint8).min, iinfo(np.uint8).max + 1]), ('uint32', 'unsigned', [iinfo(np.uint16).min, iinfo(np.uint16).max + 1]), ('uint64', 'unsigned', [iinfo(np.uint32).min, iinfo(np.uint32).max + 1])])
def test_downcast_limits(dtype: str, downcast: str, min_max: List[int]) -> None:
    ...

def test_downcast_float64_to_float32() -> None:
    ...

def test_downcast_uint64() -> None:
    ...

@pytest.mark.parametrize('data,exp_data', [([200, 300, '', 'NaN', 30000000000000000000], [200, 300, np.nan, np.nan, 30000000000000000000]), (['12345678901234567890', '1234567890', 'ITEM'], [12345678901234567890, 1234567890, np.nan])])
def test_coerce_uint64_conflict(data: List[Union[int, str]], exp_data: List[Union[int, float]]) -> None:
    ...

def test_non_coerce_uint64_conflict() -> None:
    ...

@pytest.mark.parametrize('dc1', ['integer', 'float', 'unsigned'])
@pytest.mark.parametrize('dc2', ['integer', 'float', 'unsigned'])
def test_downcast_empty(dc1: str, dc2: str) -> None:
    ...

def test_failure_to_convert_uint64_string_to_NaN() -> None:
    ...

@pytest.mark.parametrize('strrep', ['243.164', '245.968', '249.585', '259.745', '265.742', '272.567', '279.196', '280.366', '275.034', '271.351', '272.889', '270.627', '280.828', '290.383', '308.153', '319.945', '336.0', '344.09', '351.385', '356.178', '359.82', '361.03', '367.701', '380.812', '387.98', '391.749', '391.171', '385.97', '385.345', '386.121', '390.996', '399.734', '413.073', '421.532', '430.221', '437.092', '439.746', '446.01', '451.191', '460.463', '469.779', '472.025', '479.49', '474.864', '467.54', '471.978'])
def test_precision_float_conversion(strrep: str) -> None:
    ...

@pytest.mark.parametrize('values, expected', [(['1', '2', None], Series([1, 2, np.nan], dtype='Int64')), (['1', '2', '3'], Series([1, 2, 3], dtype='Int64')), (['1', '2', 3], Series([1, 2, 3], dtype='Int64')), (['1', '2', 3.5], Series([1, 2, 3.5], dtype='Float64')), (['1', None, 3.5], Series([1, np.nan, 3.5], dtype='Float64')), (['1', '2', '3.5'], Series([1, 2, 3.5], dtype='Float64'))])
def test_to_numeric_from_nullable_string(values: List[Union[str, int, float, None]], nullable_string_dtype: Any, expected: Series) -> None:
    ...

def test_to_numeric_from_nullable_string_coerce(nullable_string_dtype: Any) -> None:
    ...

@pytest.mark.parametrize('data, input_dtype, downcast, expected_dtype', (([1, 1], 'Int64', 'integer', 'Int8'), ([1.0, pd.NA], 'Float64', 'integer', 'Int8'), ([1.0, 1.1], 'Float64', 'integer', 'Float64'), ([1, pd.NA], 'Int64', 'integer', 'Int8'), ([450, 300], 'Int64', 'integer', 'Int16'), ([1, 1], 'Float64', 'integer', 'Int8'), ([np.iinfo(np.int64).max - 1, 1], 'Int64', 'integer', 'Int64'), ([1, 1], 'Int64', 'signed', 'Int8'), ([1.0, 1.0], 'Float32', 'signed', 'Int8'), ([1.0, 1.1], 'Float64', 'signed', 'Float64'), ([1, pd.NA], 'Int64', 'signed', 'Int8'), ([450, -300], 'Int64', 'signed', 'Int16'), ([np.iinfo(np.uint64).max - 1, 1], 'UInt64', 'signed', 'UInt64'), ([1, 1], 'Int64', 'unsigned', 'UInt8'), ([1.0, 1.0], 'Float32', 'unsigned', 'UInt8'), ([1.0, 1.1], 'Float64', 'unsigned', 'Float64'), ([1, pd.NA], 'Int64', 'unsigned', 'UInt8'), ([450, -300], 'Int64', 'unsigned', 'Int64'), ([-1, -1], 'Int32', 'unsigned', 'Int32'), ([1, 1], 'Float64', 'float', 'Float32'), ([1, 1.1], 'Float64', 'float', 'Float32'), ([1, 1], 'Float32', 'float', 'Float32'), ([1, 1.1], 'Float32', 'float', 'Float32')))
def test_downcast_nullable_numeric(data: List[Union[int, float, pd.NA]], input_dtype: str, downcast: str, expected_dtype: str) -> None:
    ...

def test_downcast_nullable_mask_is_copied() -> None:
    ...

def test_to_numeric_scientific_notation() -> None:
    ...

@pytest.mark.parametrize('val', [9876543210.0, 2.0 ** 128])
def test_to_numeric_large_float_not_downcast_to_float_32(val: float) -> None:
    ...

@pytest.mark.parametrize('val, dtype', [(1, 'Int64'), (1.5, 'Float64'), (True, 'boolean')])
def test_to_numeric_dtype_backend(val: Union[int, float, bool], dtype: str) -> None:
    ...

@pytest.mark.parametrize('val, dtype', [(1, 'Int64'), (1.5, 'Float64'), (True, 'boolean'), (1, 'int64[pyarrow]'), (1.5, 'float64[pyarrow]'), (True, 'bool[pyarrow]')])
def test_to_numeric_dtype_backend_na(val: Union[int, float, bool], dtype: str) -> None:
    ...

@pytest.mark.parametrize('val, dtype, downcast', [(1, 'Int8', 'integer'), (1.5, 'Float32', 'float'), (1, 'Int8', 'signed'), (1, 'int8[pyarrow]', 'integer'), (1.5, 'float[pyarrow]', 'float'), (1, 'int8[pyarrow]', 'signed')])
def test_to_numeric_dtype_backend_downcasting(val: Union[int, float], dtype: str, downcast: str) -> None:
    ...

@pytest.mark.parametrize('smaller, dtype_backend', [['UInt8', 'numpy_nullable'], ['uint8[pyarrow]', 'pyarrow']])
def test_to_numeric_dtype_backend_downcasting_uint(smaller: str, dtype_backend: str) -> None:
    ...

@pytest.mark.parametrize('dtype', ['Int64', 'UInt64', 'Float64', 'boolean', 'int64[pyarrow]', 'uint64[pyarrow]', 'float64[pyarrow]', 'bool[pyarrow]'])
def test_to_numeric_dtype_backend_already_nullable(dtype: str) -> None:
    ...

def test_to_numeric_dtype_backend_error(dtype_backend: str) -> None:
    ...

def test_invalid_dtype_backend() -> None:
    ...

def test_coerce_pyarrow_backend() -> None:
    ...