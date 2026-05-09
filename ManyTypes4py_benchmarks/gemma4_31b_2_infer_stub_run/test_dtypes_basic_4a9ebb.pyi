import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp
from typing import Any, Union, Optional, Callable, Dict, List, Tuple, Type, Sequence
from collections import defaultdict
import pytest

pytestmark: Any = ...
xfail_pyarrow: Any = ...

def test_dtype_all_columns(
    all_parsers: Any, 
    dtype: Union[Type, str], 
    check_orig: bool, 
    using_infer_string: bool
) -> None: ...

def test_dtype_per_column(all_parsers: Any) -> None: ...

def test_invalid_dtype_per_column(all_parsers: Any) -> None: ...

def test_raise_on_passed_int_dtype_with_nas(all_parsers: Any) -> None: ...

def test_dtype_with_converters(all_parsers: Any) -> None: ...

def test_numeric_dtype(all_parsers: Any, any_real_numpy_dtype: Any) -> None: ...

def test_boolean_dtype(all_parsers: Any) -> None: ...

def test_delimiter_with_usecols_and_parse_dates(all_parsers: Any) -> None: ...

@pytest.mark.parametrize('thousands', ['_', None])
def test_decimal_and_exponential(
    request: Any, 
    python_parser_only: Any, 
    numeric_decimal: Tuple[str, Any], 
    thousands: Optional[str]
) -> None: ...

@pytest.mark.parametrize('thousands', ['_', None])
@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
def test_1000_sep_decimal_float_precision(
    request: Any, 
    c_parser_only: Any, 
    numeric_decimal: Tuple[str, Any], 
    float_precision: Optional[str], 
    thousands: Optional[str]
) -> None: ...

def decimal_number_check(
    request: Any, 
    parser: Any, 
    numeric_decimal: Tuple[str, Any], 
    thousands: Optional[str], 
    float_precision: Optional[str]
) -> None: ...

@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
def test_skip_whitespace(c_parser_only: Any, float_precision: Optional[str]) -> None: ...

@pytest.mark.usefixtures('pyarrow_xfail')
def test_true_values_cast_to_bool(all_parsers: Any) -> None: ...

@pytest.mark.usefixtures('pyarrow_xfail')
@pytest.mark.parametrize('dtypes, exp_value', [({}, '1'), ({'a.1': 'int64'}, 1)])
def test_dtype_mangle_dup_cols(
    all_parsers: Any, 
    dtypes: Dict[str, Any], 
    exp_value: Union[str, int]
) -> None: ...

@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtype_mangle_dup_cols_single_dtype(all_parsers: Any) -> None: ...

@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtype_multi_index(all_parsers: Any) -> None: ...

def test_nullable_int_dtype(all_parsers: Any, any_int_ea_dtype: Any) -> None: ...

@pytest.mark.usefixtures('pyarrow_xfail')
@pytest.mark.parametrize('default', ['float', 'float64'])
def test_dtypes_defaultdict(all_parsers: Any, default: str) -> None: ...

@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtypes_defaultdict_mangle_dup_cols(all_parsers: Any) -> None: ...

@pytest.mark.usefixtures('pyarrow_xfail')
def test_dtypes_defaultdict_invalid(all_parsers: Any) -> None: ...

def test_dtype_backend(all_parsers: Any) -> None: ...

def test_dtype_backend_and_dtype(all_parsers: Any) -> None: ...

def test_dtype_backend_string(all_parsers: Any, string_storage: str) -> None: ...

def test_dtype_backend_ea_dtype_specified(all_parsers: Any) -> None: ...

def test_dtype_backend_pyarrow(all_parsers: Any, request: Any) -> None: ...

@pytest.mark.usefixtures('pyarrow_xfail')
def test_ea_int_avoid_overflow(all_parsers: Any) -> None: ...

def test_string_inference(all_parsers: Any) -> None: ...

@pytest.mark.parametrize('dtype', ['O', object, 'object', np.object_, str, np.str_])
def test_string_inference_object_dtype(all_parsers: Any, dtype: Any, using_infer_string: bool) -> None: ...

@xfail_pyarrow
def test_accurate_parsing_of_large_integers(all_parsers: Any) -> None: ...

def test_dtypes_with_usecols(all_parsers: Any) -> None: ...

def test_index_col_with_dtype_no_rangeindex(all_parsers: Any) -> None: ...