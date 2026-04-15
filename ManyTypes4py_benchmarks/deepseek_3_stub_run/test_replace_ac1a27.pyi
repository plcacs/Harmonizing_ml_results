from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Pattern,
    Sequence,
    Mapping,
    TypeVar,
)
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index, Timestamp
from pandas._testing import assert_frame_equal, assert_series_equal, assert_equal
import pytest
import re

T = TypeVar("T")

@pytest.fixture
def mix_ab() -> Dict[str, List[Union[int, str]]]: ...

@pytest.fixture
def mix_abc() -> Dict[str, List[Union[int, str, float]]]: ...

class TestDataFrameReplace:
    def test_replace_inplace(
        self,
        datetime_frame: DataFrame,
        float_string_frame: DataFrame,
    ) -> None: ...

    @pytest.mark.parametrize(
        "to_replace,values,expected",
        [
            (
                List[str],
                List[Union[str, float]],
                Dict[str, List[Union[str, float]]],
            ),
            (
                List[str],
                List[str],
                Dict[str, List[str]],
            ),
            (
                List[str],
                List[str],
                Dict[str, List[str]],
            ),
        ],
    )
    @pytest.mark.parametrize("inplace", [bool])
    @pytest.mark.parametrize("use_value_regex_args", [bool])
    def test_regex_replace_list_obj(
        self,
        to_replace: List[str],
        values: List[Union[str, float]],
        expected: Dict[str, List[Union[str, float]]],
        inplace: bool,
        use_value_regex_args: bool,
    ) -> None: ...

    def test_regex_replace_list_mixed(
        self,
        mix_ab: Dict[str, List[Union[int, str]]],
    ) -> None: ...

    def test_regex_replace_list_mixed_inplace(
        self,
        mix_ab: Dict[str, List[Union[int, str]]],
    ) -> None: ...

    def test_regex_replace_dict_mixed(
        self,
        mix_abc: Dict[str, List[Union[int, str, float]]],
    ) -> None: ...

    def test_regex_replace_dict_nested(
        self,
        mix_abc: Dict[str, List[Union[int, str, float]]],
    ) -> None: ...

    def test_regex_replace_dict_nested_non_first_character(
        self,
        any_string_dtype: str,
        using_infer_string: bool,
    ) -> None: ...

    def test_regex_replace_dict_nested_gh4115(self) -> None: ...

    def test_regex_replace_list_to_scalar(
        self,
        mix_abc: Dict[str, List[Union[int, str, float]]],
    ) -> None: ...

    def test_regex_replace_str_to_numeric(
        self,
        mix_abc: Dict[str, List[Union[int, str, float]]],
    ) -> None: ...

    def test_regex_replace_regex_list_to_numeric(
        self,
        mix_abc: Dict[str, List[Union[int, str, float]]],
    ) -> None: ...

    def test_regex_replace_series_of_regexes(
        self,
        mix_abc: Dict[str, List[Union[int, str, float]]],
    ) -> None: ...

    def test_regex_replace_numeric_to_object_conversion(
        self,
        mix_abc: Dict[str, List[Union[int, str, float]]],
    ) -> None: ...

    @pytest.mark.parametrize(
        "to_replace",
        [
            Dict[str, Union[str, float]],
            Dict[str, Union[str, float]],
        ],
    )
    def test_joint_simple_replace_and_regex_replace(
        self,
        to_replace: Dict[str, Union[str, float]],
    ) -> None: ...

    @pytest.mark.parametrize("metachar", [str])
    def test_replace_regex_metachar(self, metachar: str) -> None: ...

    @pytest.mark.parametrize(
        "data,to_replace,expected",
        [
            (
                List[str],
                Dict[str, str],
                List[str],
            ),
            (
                List[str],
                Dict[str, Union[str, pd._libs.missing.NAType]],
                List[Union[str, pd._libs.missing.NAType]],
            ),
        ],
    )
    def test_regex_replace_string_types(
        self,
        data: List[str],
        to_replace: Dict[str, str],
        expected: List[Union[str, pd._libs.missing.NAType]],
        frame_or_series: Any,
        any_string_dtype: str,
        using_infer_string: bool,
        request: Any,
    ) -> None: ...

    def test_replace(self, datetime_frame: DataFrame) -> None: ...

    def test_replace_list(self) -> None: ...

    def test_replace_with_empty_list(self, frame_or_series: Any) -> None: ...

    def test_replace_series_dict(self) -> None: ...

    def test_replace_convert(self, any_string_dtype: str) -> None: ...

    def test_replace_mixed(self, float_string_frame: DataFrame) -> None: ...

    def test_replace_mixed_int_block_upcasting(self) -> None: ...

    def test_replace_mixed_int_block_splitting(self) -> None: ...

    def test_replace_mixed2(self) -> None: ...

    def test_replace_mixed3(self) -> None: ...

    def test_replace_nullable_int_with_string_doesnt_cast(self) -> None: ...

    @pytest.mark.parametrize("dtype", [str])
    def test_replace_with_nullable_column(self, dtype: str) -> None: ...

    def test_replace_simple_nested_dict(self) -> None: ...

    def test_replace_simple_nested_dict_with_nonexistent_value(self) -> None: ...

    def test_replace_NA_with_None(self) -> None: ...

    def test_replace_NAT_with_None(self) -> None: ...

    def test_replace_with_None_keeps_categorical(self) -> None: ...

    def test_replace_all_NA(self) -> None: ...

    def test_replace_value_is_none(self, datetime_frame: DataFrame) -> None: ...

    def test_replace_for_new_dtypes(self, datetime_frame: DataFrame) -> None: ...

    @pytest.mark.parametrize(
        "frame, to_replace, value, expected",
        [
            (
                DataFrame,
                Union[int, float, bool, complex, Timestamp, str],
                Union[int, float, bool, complex, Timestamp, str],
                DataFrame,
            ),
        ],
    )
    def test_replace_dtypes(
        self,
        frame: DataFrame,
        to_replace: Union[int, float, bool, complex, Timestamp, str],
        value: Union[int, float, bool, complex, Timestamp, str],
        expected: DataFrame,
    ) -> None: ...

    def test_replace_input_formats_listlike(self) -> None: ...

    def test_replace_input_formats_scalar(self) -> None: ...

    def test_replace_limit(self) -> None: ...

    def test_replace_dict_no_regex(self, any_string_dtype: str) -> None: ...

    def test_replace_series_no_regex(self, any_string_dtype: str) -> None: ...

    def test_replace_dict_tuple_list_ordering_remains_the_same(self) -> None: ...

    def test_replace_doesnt_replace_without_regex(self) -> None: ...

    def test_replace_bool_with_string(self) -> None: ...

    def test_replace_pure_bool_with_string_no_op(self) -> None: ...

    def test_replace_bool_with_bool(self) -> None: ...

    def test_replace_with_dict_with_bool_keys(self) -> None: ...

    def test_replace_dict_strings_vs_ints(self) -> None: ...

    def test_replace_truthy(self) -> None: ...

    def test_nested_dict_overlapping_keys_replace_int(self) -> None: ...

    def test_nested_dict_overlapping_keys_replace_str(self) -> None: ...

    def test_replace_swapping_bug(self) -> None: ...

    def test_replace_datetimetz(self) -> None: ...

    def test_replace_with_empty_dictlike(
        self,
        mix_abc: Dict[str, List[Union[int, str, float]]],
    ) -> None: ...

    @pytest.mark.parametrize(
        "df, to_replace, exp",
        [
            (
                Dict[str, List[Union[int, str]]],
                Dict[Union[int, str], Union[int, str]],
                Dict[str, List[Union[int, str]]],
            ),
        ],
    )
    def test_replace_commutative(
        self,
        df: Dict[str, List[Union[int, str]]],
        to_replace: Dict[Union[int, str], Union[int, str]],
        exp: Dict[str, List[Union[int, str]]],
    ) -> None: ...

    @pytest.mark.parametrize(
        "replacer",
        [
            Timestamp,
            np.int8,
            np.int16,
            np.float32,
            np.float64,
        ],
    )
    def test_replace_replacer_dtype(self, replacer: Any) -> None: ...

    def test_replace_after_convert_dtypes(self) -> None: ...

    def test_replace_invalid_to_replace(self) -> None: ...

    @pytest.mark.parametrize("dtype", [str])
    @pytest.mark.parametrize("value", [Union[float, pd._libs.missing.NAType]])
    def test_replace_no_replacement_dtypes(
        self,
        dtype: str,
        value: Union[float, pd._libs.missing.NAType],
    ) -> None: ...

    @pytest.mark.parametrize("replacement", [Union[float, int]])
    def test_replace_with_duplicate_columns(self, replacement: Union[float, int]) -> None: ...

    @pytest.mark.parametrize(
        "value",
        [
            pd.Period,
            pd.Interval,
        ],
    )
    def test_replace_ea_ignore_float(
        self,
        frame_or_series: Any,
        value: Union[pd.Period, pd.Interval],
    ) -> None: ...

    @pytest.mark.parametrize(
        "replace_dict, final_data",
        [
            (Dict[str, int], List[List[int]]),
            (Dict[str, int], List[List[int]]),
        ],
    )
    def test_categorical_replace_with_dict(
        self,
        replace_dict: Dict[str, int],
        final_data: List[List[int]],
    ) -> None: ...

    def test_replace_value_category_type(self) -> None: ...

    def test_replace_dict_category_type(self) -> None: ...

    def test_replace_with_compiled_regex(self) -> None: ...

    def test_replace_intervals(self) -> None: ...

    def test_replace_unicode(self) -> None: ...

    def test_replace_bytes(self, frame_or_series: Any) -> None: ...

    @pytest.mark.parametrize(
        "data, to_replace, value, expected",
        [
            (List[int], List[float], List[int], List[int]),
            (List[int], List[int], List[int], List[int]),
            (List[float], List[float], List[int], List[float]),
            (List[float], List[int], List[int], List[float]),
        ],
    )
    @pytest.mark.parametrize("box", [List, Tuple, np.ndarray])
    def test_replace_list_with_mixed_type(
        self,
        data: List[Union[int, float]],
        to_replace: List[Union[int, float]],
        value: List[int],
        expected: List[Union[int, float]],
        box: Any,
        frame_or_series: Any,
    ) -> None: ...

    @pytest.mark.parametrize("val", [int, float])
    def test_replace_value_none_dtype_numeric(self, val: Union[int, float]) -> None: ...

    def test_replace_with_nil_na(self) -> None: ...

class TestDataFrameReplaceRegex:
    @pytest.mark.parametrize(
        "data",
        [
            Dict[str, List[str]],
            Dict[str, List[Union[str, int]]],
        ],
    )
    @pytest.mark.parametrize(
        "to_replace,value",
        [
            (str, Union[float, str]),
            (str, str),
        ],
    )
    @pytest.mark.parametrize("compile_regex", [bool])
    @pytest.mark.parametrize("regex_kwarg", [bool])
    @pytest.mark.parametrize("inplace", [bool])
    def test_regex_replace_scalar(
        self,
        data: Dict[str, List[Union[str, int]]],
        to_replace: str,
        value: Union[float, str],
        compile_regex: bool,
        regex_kwarg: bool,
        inplace: bool,
    ) -> None: ...

    @pytest.mark.parametrize("regex", [bool])
    @pytest.mark.parametrize("value", [Union[int, str]])
    def test_replace_regex_dtype_frame(
        self,
        regex: bool,
        value: Union[int, str],
    ) -> None: ...

    def test_replace_with_value_also_being_replaced(self) -> None: ...

    def test_replace_categorical_no_replacement(self) -> None: ...

    def test_replace_object_splitting(self, using_infer_string: bool) -> None: ...