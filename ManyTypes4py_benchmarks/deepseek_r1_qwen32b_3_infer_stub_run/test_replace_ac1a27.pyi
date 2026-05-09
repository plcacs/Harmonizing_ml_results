from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp, date_range
from pandas._testing import tm

@pytest.fixture
def mix_ab() -> dict[str, list[int | str]]:
    ...

@pytest.fixture
def mix_abc() -> dict[str, list[int | str | float]]:
    ...

class TestDataFrameReplace:
    def test_replace_inplace(self, datetime_frame: DataFrame, float_string_frame: DataFrame) -> None:
        ...

    @pytest.mark.parametrize('to_replace,values,expected', [
        (list[str], list[Any], dict[str, list[Any]]),
        (list[str], list[Any], dict[str, list[Any]]),
        (list[str], list[Any], dict[str, list[Any]])
    ])
    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('use_value_regex_args', [True, False])
    def test_regex_replace_list_obj(self, to_replace: list[str], values: list[Any], expected: dict[str, list[Any]], inplace: bool, use_value_regex_args: bool) -> None:
        ...

    def test_regex_replace_list_mixed(self, mix_ab: dict[str, list[int | str]]) -> None:
        ...

    def test_regex_replace_list_mixed_inplace(self, mix_ab: dict[str, list[int | str]]) -> None:
        ...

    def test_regex_replace_dict_mixed(self, mix_abc: dict[str, list[int | str | float]]) -> None:
        ...

    def test_regex_replace_dict_nested(self, mix_abc: dict[str, list[int | str | float]]) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['any_string_dtype'])
    def test_regex_replace_dict_nested_non_first_character(self, any_string_dtype: str, using_infer_string: bool) -> None:
        ...

    def test_regex_replace_dict_nested_gh4115(self) -> None:
        ...

    def test_regex_replace_list_to_scalar(self, mix_abc: dict[str, list[int | str | float]]) -> None:
        ...

    def test_regex_replace_str_to_numeric(self, mix_abc: dict[str, list[int | str | float]]) -> None:
        ...

    def test_regex_replace_regex_list_to_numeric(self, mix_abc: dict[str, list[int | str | float]]) -> None:
        ...

    def test_regex_replace_series_of_regexes(self, mix_abc: dict[str, list[int | str | float]]) -> None:
        ...

    def test_regex_replace_numeric_to_object_conversion(self, mix_abc: dict[str, list[int | str | float]]) -> None:
        ...

    @pytest.mark.parametrize('to_replace', [dict[str, Any], dict[str, Any]])
    def test_joint_simple_replace_and_regex_replace(self, to_replace: dict[str, Any]) -> None:
        ...

    @pytest.mark.parametrize('metachar', ['[]', '()', '\\d', '\\w', '\\s'])
    def test_replace_regex_metachar(self, metachar: str) -> None:
        ...

    @pytest.mark.parametrize('data,to_replace,expected', [
        (list[str], dict[str, Any], list[str]),
        (list[str], dict[str, Any], list[str])
    ])
    def test_regex_replace_string_types(self, data: list[str], to_replace: dict[str, Any], expected: list[str], frame_or_series: Any, any_string_dtype: str, using_infer_string: bool, request: Any) -> None:
        ...

    def test_replace(self, datetime_frame: DataFrame) -> None:
        ...

    def test_replace_list(self) -> None:
        ...

    def test_replace_with_empty_list(self, frame_or_series: Any) -> None:
        ...

    def test_replace_series_dict(self) -> None:
        ...

    def test_replace_convert(self, any_string_dtype: str) -> None:
        ...

    def test_replace_mixed(self, float_string_frame: DataFrame) -> None:
        ...

    def test_replace_mixed_int_block_upcasting(self) -> None:
        ...

    def test_replace_mixed_int_block_splitting(self) -> None:
        ...

    def test_replace_mixed2(self) -> None:
        ...

    def test_replace_mixed3(self) -> None:
        ...

    def test_replace_nullable_int_with_string_doesnt_cast(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['boolean', 'Int64', 'Float64'])
    def test_replace_with_nullable_column(self, dtype: str) -> None:
        ...

    def test_replace_simple_nested_dict(self) -> None:
        ...

    def test_replace_simple_nested_dict_with_nonexistent_value(self) -> None:
        ...

    def test_replace_NA_with_None(self) -> None:
        ...

    def test_replace_NAT_with_None(self) -> None:
        ...

    def test_replace_with_None_keeps_categorical(self) -> None:
        ...

    def test_replace_all_NA(self) -> None:
        ...

    def test_replace_value_is_none(self, datetime_frame: DataFrame) -> None:
        ...

    def test_replace_for_new_dtypes(self, datetime_frame: DataFrame) -> None:
        ...

    @pytest.mark.parametrize('frame, to_replace, value, expected', [
        (DataFrame, int, int, DataFrame),
        (DataFrame, int, int, DataFrame),
        (DataFrame, int, int, DataFrame),
        (DataFrame, bool, bool, DataFrame),
        (DataFrame, complex, complex, DataFrame),
        (DataFrame, datetime, datetime, DataFrame),
        (DataFrame, str, str, DataFrame),
        (DataFrame, Timestamp, Timestamp, DataFrame),
        (DataFrame, float, float, DataFrame),
        (DataFrame, float, float, DataFrame),
        (DataFrame, float, float, DataFrame),
        (DataFrame, float, float, DataFrame)
    ])
    def test_replace_dtypes(self, frame: DataFrame, to_replace: Any, value: Any, expected: DataFrame) -> None:
        ...

    def test_replace_input_formats_listlike(self) -> None:
        ...

    def test_replace_input_formats_scalar(self) -> None:
        ...

    def test_replace_limit(self) -> None:
        ...

    def test_replace_dict_no_regex(self, any_string_dtype: str) -> None:
        ...

    def test_replace_series_no_regex(self, any_string_dtype: str) -> None:
        ...

    def test_replace_dict_tuple_list_ordering_remains_the_same(self) -> None:
        ...

    def test_replace_doesnt_replace_without_regex(self) -> None:
        ...

    def test_replace_bool_with_string(self) -> None:
        ...

    def test_replace_pure_bool_with_string_no_op(self) -> None:
        ...

    def test_replace_bool_with_bool(self) -> None:
        ...

    def test_replace_with_dict_with_bool_keys(self) -> None:
        ...

    def test_replace_dict_strings_vs_ints(self) -> None:
        ...

    def test_replace_truthy(self) -> None:
        ...

    def test_nested_dict_overlapping_keys_replace_int(self) -> None:
        ...

    def test_nested_dict_overlapping_keys_replace_str(self) -> None:
        ...

    def test_replace_swapping_bug(self) -> None:
        ...

    def test_replace_datetimetz(self) -> None:
        ...

    def test_replace_with_empty_dictlike(self, mix_abc: dict[str, list[int | str | float]]) -> None:
        ...

    @pytest.mark.parametrize('df, to_replace, exp', [
        (dict[str, list[int | float]], dict[int, int], dict[str, list[int | float]]),
        (dict[str, list[str]], dict[str, str], dict[str, list[str]])
    ])
    def test_replace_commutative(self, df: dict[str, list[Any]], to_replace: dict[Any, Any], exp: dict[str, list[Any]]) -> None:
        ...

    @pytest.mark.parametrize('replacer', [Timestamp, np.int8, np.int16, np.float32, np.float64])
    def test_replace_replacer_dtype(self, replacer: Any) -> None:
        ...

    def test_replace_after_convert_dtypes(self) -> None:
        ...

    def test_replace_invalid_to_replace(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', ['float', 'float64', 'int64', 'Int64', 'boolean'])
    @pytest.mark.parametrize('value', [np.nan, pd.NA])
    def test_replace_no_replacement_dtypes(self, dtype: str, value: Any) -> None:
        ...

    @pytest.mark.parametrize('replacement', [np.nan, int])
    def test_replace_with_duplicate_columns(self, replacement: Any) -> None:
        ...

    @pytest.mark.parametrize('value', [pd.Period, pd.Interval])
    def test_replace_ea_ignore_float(self, frame_or_series: Any, value: Any) -> None:
        ...

    @pytest.mark.parametrize('replace_dict, final_data', [
        (dict[str, int], list[list[int]]),
        (dict[str, int], list[list[int]])
    ])
    def test_categorical_replace_with_dict(self, replace_dict: dict[str, int], final_data: list[list[int]]) -> None:
        ...

    def test_replace_value_category_type(self) -> None:
        ...

    def test_replace_dict_category_type(self) -> None:
        ...

    def test_replace_with_compiled_regex(self) -> None:
        ...

    def test_replace_intervals(self) -> None:
        ...

    def test_replace_unicode(self) -> None:
        ...

    def test_replace_bytes(self, frame_or_series: Any) -> None:
        ...

    @pytest.mark.parametrize('data, to_replace, value, expected', [
        (list[int], list[float], list[int], list[int]),
        (list[int], list[int], list[int], list[int]),
        (list[float], list[float], list[int], list[int]),
        (list[float], list[int], list[int], list[int])
    ])
    @pytest.mark.parametrize('box', [list, tuple, np.ndarray])
    def test_replace_list_with_mixed_type(self, data: list[Any], to_replace: list[Any], value: list[Any], expected: list[Any], box: type, frame_or_series: Any) -> None:
        ...

    @pytest.mark.parametrize('val', [int, np.nan, float])
    def test_replace_value_none_dtype_numeric(self, val: Any) -> None:
        ...

    def test_replace_with_nil_na(self) -> None:
        ...

class TestDataFrameReplaceRegex:
    @pytest.mark.parametrize('data', [
        dict[str, list[str]],
        dict[str, list[int]]
    ])
    @pytest.mark.parametrize('to_replace,value', [
        (str, np.nan),
        (str, str)
    ])
    @pytest.mark.parametrize('compile_regex', [True, False])
    @pytest.mark.parametrize('regex_kwarg', [True, False])
    @pytest.mark.parametrize('inplace', [True, False])
    def test_regex_replace_scalar(self, data: dict[str, list[Any]], to_replace: str, value: Any, compile_regex: bool, regex_kwarg: bool, inplace: bool) -> None:
        ...

    @pytest.mark.parametrize('regex', [False, True])
    @pytest.mark.parametrize('value', [int, str])
    def test_replace_regex_dtype_frame(self, regex: bool, value: Any) -> None:
        ...

    def test_replace_with_value_also_being_replaced(self) -> None:
        ...

    def test_replace_categorical_no_replacement(self) -> None:
        ...

    def test_replace_object_splitting(self, using_infer_string: bool) -> None:
        ...