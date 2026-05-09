"""
Stub file for test_dtypes_basic_4a9ebb.py
"""

from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas import DataFrame, Timestamp
import pandas._testing as tm
from pandas.core.arrays import IntegerArray
from pandas.core.dtypes.dtypes import (
    AnyDtype,
    ArrowDtype,
    BooleanDtype,
    Float64Dtype,
    Int64Dtype,
    IntervalDtype,
    NumpyExtensionDtype,
    PandasDtype,
    PeriodDtype,
    StringDtype,
    TimedeltaDtype,
)

@pytest.mark.parametrize('dtype', [str, object])
@pytest.mark.parametrize('check_orig', [True, False])
def test_dtype_all_columns(all_parsers: pytest.fixture, dtype: type, check_orig: bool, using_infer_string: bool) -> None:
    ...

def test_dtype_per_column(all_parsers: pytest.fixture) -> None:
    ...

def test_invalid_dtype_per_column(all_parsers: pytest.fixture) -> None:
    ...

def test_raise_on_passed_int_dtype_with_nas(all_parsers: pytest.fixture) -> None:
    ...

def test_dtype_with_converters(all_parsers: pytest.fixture) -> None:
    ...

def test_numeric_dtype(all_parsers: pytest.fixture, any_real_numpy_dtype: np.dtype) -> None:
    ...

def test_boolean_dtype(all_parsers: pytest.fixture) -> None:
    ...

def test_delimiter_with_usecols_and_parse_dates(all_parsers: pytest.fixture) -> None:
    ...

@pytest.mark.parametrize('thousands', ['_', None])
def test_decimal_and_exponential(request: pytest.fixture, python_parser_only: pytest.fixture, numeric_decimal: tuple[str, str | float]) -> None:
    ...

@pytest.mark.parametrize('thousands', ['_', None])
@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
def test_1000_sep_decimal_float_precision(request: pytest.fixture, c_parser_only: pytest.fixture, numeric_decimal: tuple[str, str | float], float_precision: str | None, thousands: str | None) -> None:
    ...

def decimal_number_check(request: pytest.fixture, parser: object, numeric_decimal: tuple[str, str | float], thousands: str | None, float_precision: str | None) -> None:
    ...

@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
def test_skip_whitespace(c_parser_only: pytest.fixture, float_precision: str | None) -> None:
    ...

def test_true_values_cast_to_bool(all_parsers: pytest.fixture) -> None:
    ...

@pytest.mark.parametrize('dtypes, exp_value', [({}, '1'), ({'a.1': 'int64'}, 1)])
def test_dtype_mangle_dup_cols(all_parsers: pytest.fixture, dtypes: dict[str, str], exp_value: str | int) -> None:
    ...

def test_dtype_mangle_dup_cols_single_dtype(all_parsers: pytest.fixture) -> None:
    ...

def test_dtype_multi_index(all_parsers: pytest.fixture) -> None:
    ...

def test_nullable_int_dtype(all_parsers: pytest.fixture, any_int_ea_dtype: AnyDtype) -> None:
    ...

@pytest.mark.parametrize('default', ['float', 'float64'])
def test_dtypes_defaultdict(all_parsers: pytest.fixture, default: str) -> None:
    ...

def test_dtypes_defaultdict_mangle_dup_cols(all_parsers: pytest.fixture) -> None:
    ...

def test_dtypes_defaultdict_invalid(all_parsers: pytest.fixture) -> None:
    ...

def test_dtype_backend(all_parsers: pytest.fixture) -> None:
    ...

def test_dtype_backend_and_dtype(all_parsers: pytest.fixture) -> None:
    ...

def test_dtype_backend_string(all_parsers: pytest.fixture, string_storage: str) -> None:
    ...

def test_dtype_backend_ea_dtype_specified(all_parsers: pytest.fixture) -> None:
    ...

def test_dtype_backend_pyarrow(all_parsers: pytest.fixture, request: pytest.fixture) -> None:
    ...

def test_ea_int_avoid_overflow(all_parsers: pytest.fixture) -> None:
    ...

def test_string_inference(all_parsers: pytest.fixture) -> None:
    ...

@pytest.mark.parametrize('dtype', ['O', object, 'object', np.object_, str, np.str_])
def test_string_inference_object_dtype(all_parsers: pytest.fixture, dtype: type, using_infer_string: bool) -> None:
    ...

def test_accurate_parsing_of_large_integers(all_parsers: pytest.fixture) -> None:
    ...

def test_dtypes_with_usecols(all_parsers: pytest.fixture) -> None:
    ...

def test_index_col_with_dtype_no_rangeindex(all_parsers: pytest.fixture) -> None:
    ...