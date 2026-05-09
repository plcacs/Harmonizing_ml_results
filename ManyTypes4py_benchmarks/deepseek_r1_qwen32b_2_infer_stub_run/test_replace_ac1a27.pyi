from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import pytest
import re

@pytest.fixture
def mix_ab() -> Dict[str, Union[List[int], List[str]]]:
    ...

@pytest.fixture
def mix_abc() -> Dict[str, Union[List[int], List[str], List[Union[str, float]]]]:
    ...

class TestDataFrameReplace:
    def test_replace_inplace(self, datetime_frame: pd.DataFrame, float_string_frame: pd.DataFrame) -> None:
        ...

    @pytest.mark.parametrize('to_replace,values,expected', [
        (List[Union[str, re.Pattern]], List[Union[np.nan, str]], Dict[str, List[Union[str, float]]]),
        (List[Union[str, re.Pattern]], List[Union[str, re.Pattern]], Dict[str, List[str]]),
        (List[Union[str, re.Pattern]], List[Union[str, re.Pattern]], Dict[str, List[str]])
    ])
    @pytest.mark.parametrize('inplace', [True, False])
    @pytest.mark.parametrize('use_value_regex_args', [True, False])
    def test_regex_replace_list_obj(
        self,
        to_replace: List[Union[str, re.Pattern]],
        values: List[Any],
        expected: Dict[str, List[Any]],
        inplace: bool,
        use_value_regex_args: bool
    ) -> None:
        ...

    def test_regex_replace_list_mixed(self, mix_ab: Dict[str, Union[List[int], List[str]]]) -> None:
        ...

    def test_regex_replace_list_mixed_inplace(self, mix_ab: Dict[str, Union[List[int], List[str]]]) -> None:
        ...

    def test_regex_replace_dict_mixed(self, mix_abc: Dict[str, Union[List[int], List[str], List[Union[str, float]]]]) -> None:
        ...

    def test_regex_replace_dict_nested(self, mix_abc: Dict[str, Union[List[int], List[str], List[Union[str, float]]]]) -> None:
        ...

    @pytest.mark.parametrize('dtype', [Any])
    def test_regex_replace_dict_nested_non_first_character(self, any_string_dtype: Any, using_infer_string: bool) -> None:
        ...

    def test_regex_replace_dict_nested_gh4115(self) -> None:
        ...

    def test_regex_replace_list_to_scalar(self, mix_abc: Dict[str, Union[List[int], List[str], List[Union[str, float]]]]) -> None:
        ...

    def test_regex_replace_str_to_numeric(self, mix_abc: Dict[str, Union[List[int], List[str], List[Union[str, float]]]]) -> None:
        ...

    def test_regex_replace_regex_list_to_numeric(self, mix_abc: Dict[str, Union[List[int], List[str], List[Union[str, float]]]]) -> None:
        ...

    def test_regex_replace_series_of_regexes(self, mix_abc: Dict[str, Union[List[int], List[str], List[Union[str, float]]]]) -> None:
        ...

    def test_regex_replace_numeric_to_object_conversion(self, mix_abc: Dict[str, Union[List[int], List[str], List[Union[str, float]]]]) -> None:
        ...

    @pytest.mark.parametrize('to_replace', [Dict[str, Union[str, float]]])
    def test_joint_simple_replace_and_regex_replace(self, to_replace: Dict[str, Union[str, float]]) -> None:
        ...

    @pytest.mark.parametrize('metachar', ['[]', '()', '\\d', '\\w', '\\s'])
    def test_replace_regex_metachar(self, metachar: str) -> None:
        ...

    @pytest.mark.parametrize('data,to_replace,expected', [
        (List[str], Dict[str, str], List[str]),
        (List[str], Dict[str, pd.NA], List[Union[str, pd.NA]])
    ])
    def test_regex_replace_string_types(
        self,
        data: List[str],
        to_replace: Dict[str, Union[str, pd.NA]],
        expected: List[Union[str, pd.NA]],
        frame_or_series: Any,
        any_string_dtype: Any,
        using_infer_string: bool,
        request: Any
    ) -> None:
        ...

    def test_replace(self, datetime_frame: pd.DataFrame) -> None:
        ...

    def test_replace_list(self) -> None:
        ...

    def test_replace_with_empty_list(self, frame_or_series: Any) -> None:
        ...

    def test_replace_series_dict(self) -> None:
        ...

    @pytest.mark.parametrize('dtype', [Any])
    def test_replace_convert(self, any_string_dtype: Any) -> None:
        ...

    def test_replace_mixed(self, float_string_frame: pd.DataFrame) -> None:
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

    def test_replace_value_is_none(self, datetime_frame: pd.DataFrame) -> None:
        ...

    def test_replace_for_new_dtypes(self, datetime_frame: pd.DataFrame) -> None:
        ...

    @pytest.mark.parametrize('frame, to_replace, value, expected', [
        (pd.DataFrame, Any, Any, pd.DataFrame),
        (pd.DataFrame, Any, Any, pd.DataFrame)
    ])
    def test_replace_dtypes(
        self,
        frame: pd.DataFrame,
        to_replace: Any,
        value: Any,
        expected: pd.DataFrame
    ) -> None:
        ...

    def test_replace_input_formats_listlike(self) -> None:
        ...

    def test_replace_input_formats_scalar(self) -> None:
        ...

    def test_replace_limit(self) -> None:
        ...

    def test_replace_dict_no_regex(self, any_string_dtype: Any) -> None:
        ...

    def test_replace_series_no_regex(self, any_string_dtype: Any) -> None:
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

    def test_replace_with_empty_dictlike(self, mix_abc: Dict[str, Union[List[int], List[str], List[Union[str, float]]]]) -> None:
        ...

    @pytest.mark.parametrize('df, to_replace, exp', [
        (Dict[str, List[int]], Dict[int, int], Dict[str, List[int]]),
        (Dict[str, List[str]], Dict[str, str], Dict[str, List[str]])
    ])
    def test_replace_commutative(
        self,
        df: Dict[str, List[Any]],
        to_replace: Dict[Any, Any],
        exp: Dict[str, List[Any]]
    ) -> None:
        ...

    @pytest.mark.parametrize('replacer', [pd.Timestamp, np.int8, np.int16, np.float32, np.float64])
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

    @pytest.mark.parametrize('value', [pd.Period, pd.Interval])
    def test_replace_ea_ignore_float(self, frame_or_series: Any, value: Any) -> None:
        ...

    @pytest.mark.parametrize('replace_dict, final_data', [
        (Dict[str, int], List[List[int]]),
        (Dict[str, int], List[List[int]])
    ])
    def test_categorical_replace_with_dict(
        self,
        replace_dict: Dict[str, int],
        final_data: List[List[int]]
    ) -> None:
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
        (List[int], List[float], List[int], List[int]),
        (List[int], List[int], List[int], List[int]),
        (List[float], List[float], List[float], List[float]),
        (List[float], List[int], List[float], List[float])
    ])
    @pytest.mark.parametrize('box', [list, tuple, np.ndarray])
    def test_replace_list_with_mixed_type(
        self,
        data: List[Any],
        to_replace: List[Any],
        value: List[Any],
        expected: List[Any],
        box: Any,
        frame_or_series: Any
    ) -> None:
        ...

    @pytest.mark.parametrize('val', [int, np.nan, float])
    def test_replace_value_none_dtype_numeric(self, val: Any) -> None:
        ...

    def test_replace_with_nil_na(self) -> None:
        ...

class TestDataFrameReplaceRegex:
    @pytest.mark.parametrize('data', [
        Dict[str, List[str]],
        Dict[str, Union[List[str], List[int]]]
    ])
    @pytest.mark.parametrize('to_replace,value', [
        (str, np.nan),
        (str, str)
    ])
    @pytest.mark.parametrize('compile_regex', [True, False])
    @pytest.mark.parametrize('regex_kwarg', [True, False])
    @pytest.mark.parametrize('inplace', [True, False])
    def test_regex_replace_scalar(
        self,
        data: Dict[str, List[Any]],
        to_replace: Union[str, re.Pattern],
        value: Any,
        compile_regex: bool,
        regex_kwarg: bool,
        inplace: bool
    ) -> None:
        ...

    @pytest.mark.parametrize('regex', [True, False])
    @pytest.mark.parametrize('value', [int, str])
    def test_replace_regex_dtype_frame(self, regex: bool, value: Any) -> None:
        ...

    def test_replace_with_value_also_being_replaced(self) -> None:
        ...

    def test_replace_categorical_no_replacement(self) -> None:
        ...

    def test_replace_object_splitting(self, using_infer_string: bool) -> None:
        ...