"""
Stub file for 'test_na_values_e137af' module
"""

from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import DataFrame, Index, MultiIndex
import pandas._testing as tm

pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
xfail_pyarrow = pytest.mark.usefixtures('pyarrow_xfail')
skip_pyarrow = pytest.mark.usefixtures('pyarrow_skip')

def test_string_nas(all_parsers: Any) -> None:
    ...

def test_detect_string_na(all_parsers: Any) -> None:
    ...

def test_non_string_na_values(all_parsers: Any, data: str, na_values: list[Union[int, float, str]], request: Any) -> None:
    ...

def test_default_na_values(all_parsers: Any) -> None:
    ...

def test_custom_na_values(all_parsers: Any, na_values: Union[str, list[str]]) -> None:
    ...

def test_bool_na_values(all_parsers: Any) -> None:
    ...

def test_na_value_dict(all_parsers: Any) -> None:
    ...

def test_na_value_dict_multi_index(all_parsers: Any, index_col: Union[list[int], list[str]], expected: DataFrame) -> None:
    ...

def test_na_values_keep_default(all_parsers: Any, kwargs: dict, expected: DataFrame, request: Any, using_infer_string: bool) -> None:
    ...

def test_no_na_values_no_keep_default(all_parsers: Any) -> None:
    ...

def test_no_keep_default_na_dict_na_values(all_parsers: Any, data: str, na_values: dict) -> None:
    ...

def test_no_keep_default_na_dict_na_scalar_values(all_parsers: Any, data: str, na_values: dict) -> None:
    ...

def test_no_keep_default_na_dict_na_values_diff_reprs(all_parsers: Any, col_zero_na_values: Union[int, str], data: str, na_values: dict) -> None:
    ...

def test_na_values_na_filter_override(request: Any, all_parsers: Any, na_filter: bool, row_data: list[list[Union[str, float, np.nan]]], using_infer_string: bool) -> None:
    ...

def test_na_trailing_columns(all_parsers: Any) -> None:
    ...

def test_na_values_scalar(all_parsers: Any, na_values: Union[int, dict], row_data: list[list[Union[float, np.nan]]]) -> None:
    ...

def test_na_values_dict_aliasing(all_parsers: Any, na_values: dict, na_values_copy: dict, names: list[str], data: str, expected: DataFrame) -> None:
    ...

def test_na_values_dict_null_column_name(all_parsers: Any, data: str, names: list[Union[None, str]], na_values: dict, dtype: dict, keep_default_na: bool, expected: DataFrame) -> None:
    ...

def test_na_values_dict_col_index(all_parsers: Any, data: str, na_values: dict, expected: DataFrame) -> None:
    ...

def test_na_values_uint64(all_parsers: Any, data: str, kwargs: dict, expected: DataFrame, request: Any) -> None:
    ...

def test_empty_na_values_no_default_with_index(all_parsers: Any) -> None:
    ...

def test_no_na_filter_on_index(all_parsers: Any, na_filter: bool, index_data: list[Union[str, float]], request: Any, expected: DataFrame) -> None:
    ...

def test_inf_na_values_with_int_index(all_parsers: Any) -> None:
    ...

def test_na_values_with_dtype_str_and_na_filter(all_parsers: Any, na_filter: bool) -> None:
    ...

def test_cast_NA_to_bool_raises_error(all_parsers: Any, data: str, na_values: Union[None, str, list[str], dict]) -> None:
    ...

def test_str_nan_dropped(all_parsers: Any) -> None:
    ...

def test_nan_multi_index(all_parsers: Any, na_values: dict) -> None:
    ...

def test_bool_and_nan_to_bool(all_parsers: Any) -> None:
    ...

def test_bool_and_nan_to_int(all_parsers: Any) -> None:
    ...

def test_bool_and_nan_to_float(all_parsers: Any) -> None:
    ...

def test_na_values_dict_without_dtype(all_parsers: Any, na_values: list[Union[int, float]]) -> None:
    ...