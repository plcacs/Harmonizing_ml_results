from __future__ import annotations

from datetime import datetime
import re
from typing import Any

import numpy as np
import pytest

import pandas as pd
from pandas import DataFrame, Index, Series, Timestamp, date_range
import pandas._testing as tm


@pytest.fixture
def mix_ab() -> dict[str, list[Any]]: ...

@pytest.fixture
def mix_abc() -> dict[str, list[Any]]: ...


class TestDataFrameReplace:
    def test_replace_inplace(self, datetime_frame: DataFrame, float_string_frame: DataFrame) -> None: ...

    @pytest.mark.parametrize(
        "to_replace,values,expected",
        [
            (
                ["\\s*\\.\\s*", "e|f|g"],
                [np.nan, "crap"],
                {"a": ["a", "b", np.nan, np.nan], "b": ["crap"] * 3 + ["h"], "c": ["h", "crap", "l", "o"]},
            ),
            (
                ["\\s*(\\.)\\s*", "(e|f|g)"],
                ["\\1\\1", "\\1_crap"],
                {"a": ["a", "b", "..", ".."], "b": ["e_crap", "f_crap", "g_crap", "h"], "c": ["h", "e_crap", "l", "o"]},
            ),
            (
                ["\\s*(\\.)\\s*", "e"],
                ["\\1\\1", "crap"],
                {"a": ["a", "b", "..", ".."], "b": ["crap", "f", "g", "h"], "c": ["h", "crap", "l", "o"]},
            ),
        ],
    )
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("use_value_regex_args", [True, False])
    def test_regex_replace_list_obj(
        self,
        to_replace: list[str],
        values: list[Any],
        expected: dict[str, list[Any]],
        inplace: bool,
        use_value_regex_args: bool,
    ) -> None: ...

    def test_regex_replace_list_mixed(self, mix_ab: dict[str, list[Any]]) -> None: ...
    def test_regex_replace_list_mixed_inplace(self, mix_ab: dict[str, list[Any]]) -> None: ...
    def test_regex_replace_dict_mixed(self, mix_abc: dict[str, list[Any]]) -> None: ...
    def test_regex_replace_dict_nested(self, mix_abc: dict[str, list[Any]]) -> None: ...
    def test_regex_replace_dict_nested_non_first_character(self, any_string_dtype: str, using_infer_string: bool) -> None: ...
    def test_regex_replace_dict_nested_gh4115(self) -> None: ...
    def test_regex_replace_list_to_scalar(self, mix_abc: dict[str, list[Any]]) -> None: ...
    def test_regex_replace_str_to_numeric(self, mix_abc: dict[str, list[Any]]) -> None: ...
    def test_regex_replace_regex_list_to_numeric(self, mix_abc: dict[str, list[Any]]) -> None: ...
    def test_regex_replace_series_of_regexes(self, mix_abc: dict[str, list[Any]]) -> None: ...
    def test_regex_replace_numeric_to_object_conversion(self, mix_abc: dict[str, list[Any]]) -> None: ...

    @pytest.mark.parametrize("to_replace", [{"": np.nan, ",": ""}, {",": "", "": np.nan}])
    def test_joint_simple_replace_and_regex_replace(self, to_replace: dict[str, Any]) -> None: ...

    @pytest.mark.parametrize("metachar", ["[]", "()", "\\d", "\\w", "\\s"])
    def test_replace_regex_metachar(self, metachar: str) -> None: ...

    @pytest.mark.parametrize(
        "data,to_replace,expected",
        [
            (["xax", "xbx"], {"a": "c", "b": "d"}, ["xcx", "xdx"]),
            (["d", "", ""], {"^\\s*$": pd.NA}, ["d", pd.NA, pd.NA]),
        ],
    )
    def test_regex_replace_string_types(
        self,
        data: list[Any],
        to_replace: dict[str, Any],
        expected: list[Any],
        frame_or_series: type,
        any_string_dtype: str,
        using_infer_string: bool,
        request: pytest.FixtureRequest,
    ) -> None: ...

    def test_replace(self, datetime_frame: DataFrame) -> None: ...
    def test_replace_list(self) -> None: ...
    def test_replace_with_empty_list(self, frame_or_series: type) -> None: ...
    def test_replace_series_dict(self) -> None: ...
    def test_replace_convert(self, any_string_dtype: str) -> None: ...
    def test_replace_mixed(self, float_string_frame: DataFrame) -> None: ...
    def test_replace_mixed_int_block_upcasting(self) -> None: ...
    def test_replace_mixed_int_block_splitting(self) -> None: ...
    def test_replace_mixed2(self) -> None: ...
    def test_replace_mixed3(self) -> None: ...
    def test_replace_nullable_int_with_string_doesnt_cast(self) -> None: ...

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
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
            (DataFrame({"ints": [1, 2, 3]}), 1, 0, DataFrame({"ints": [0, 2, 3]})),
            (DataFrame({"ints": [1, 2, 3]}, dtype=np.int32), 1, 0, DataFrame({"ints": [0, 2, 3]}, dtype=np.int32)),
            (DataFrame({"ints": [1, 2, 3]}, dtype=np.int16), 1, 0, DataFrame({"ints": [0, 2, 3]}, dtype=np.int16)),
            (DataFrame({"bools": [True, False, True]}), False, True, DataFrame({"bools": [True, True, True]})),
            (DataFrame({"complex": [1j, 2j, 3j]}), 1j, 0, DataFrame({"complex": [0j, 2j, 3j]})),
            (
                DataFrame({"datetime64": Index([datetime(2018, 5, 28), datetime(2018, 7, 28), datetime(2018, 5, 28)])}),
                datetime(2018, 5, 28),
                datetime(2018, 7, 28),
                DataFrame({"datetime64": Index([datetime(2018, 7, 28)] * 3)}),
            ),
            (
                DataFrame({"dt": [datetime(3017, 12, 20)], "str": ["foo"]}),
                "foo",
                "bar",
                DataFrame({"dt": [datetime(3017, 12, 20)], "str": ["bar"]}),
            ),
            (
                DataFrame({"A": date_range("20130101", periods=3, tz="US/Eastern"), "B": [0, np.nan, 2]}),
                Timestamp("20130102", tz="US/Eastern"),
                Timestamp("20130104", tz="US/Eastern"),
                DataFrame(
                    {
                        "A": pd.DatetimeIndex(
                            [
                                Timestamp("20130101", tz="US/Eastern"),
                                Timestamp("20130104", tz="US/Eastern"),
                                Timestamp("20130103", tz="US/Eastern"),
                            ]
                        ).as_unit("ns"),
                        "B": [0, np.nan, 2],
                    }
                ),
            ),
            (DataFrame([[1, 1.0], [2, 2.0]]), 1.0, 5, DataFrame([[5, 5.0], [2, 2.0]])),
            (DataFrame([[1, 1.0], [2, 2.0]]), 1, 5, DataFrame([[5, 5.0], [2, 2.0]])),
            (DataFrame([[1, 1.0], [2, 2.0]]), 1.0, 5.0, DataFrame([[5, 5.0], [2, 2.0]])),
            (DataFrame([[1, 1.0], [2, 2.0]]), 1, 5.0, DataFrame([[5, 5.0], [2, 2.0]])),
        ],
    )
    def test_replace_dtypes(self, frame: DataFrame, to_replace: Any, value: Any, expected: DataFrame) -> None: ...

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
    def test_replace_nested_dict_overlapping_keys_replace_int(self) -> None: ...
    def test_replace_nested_dict_overlapping_keys_replace_str(self) -> None: ...
    def test_replace_swapping_bug(self) -> None: ...
    def test_replace_datetimetz(self) -> None: ...
    def test_replace_with_empty_dictlike(self, mix_abc: dict[str, list[Any]]) -> None: ...

    @pytest.mark.parametrize(
        "df, to_replace, exp",
        [
            ({"col1": [1, 2, 3], "col2": [4, 5, 6]}, {4: 5, 5: 6, 6: 7}, {"col1": [1, 2, 3], "col2": [5, 6, 7]}),
            (
                {"col1": [1, 2, 3], "col2": ["4", "5", "6"]},
                {"4": "5", "5": "6", "6": "7"},
                {"col1": [1, 2, 3], "col2": ["5", "6", "7"]},
            ),
        ],
    )
    def test_replace_commutative(self, df: dict[str, list[Any]], to_replace: dict[Any, Any], exp: dict[str, list[Any]]) -> None: ...

    @pytest.mark.parametrize(
        "replacer",
        [Timestamp("20170827"), np.int8(1), np.int16(1), np.float32(1), np.float64(1)],
    )
    def test_replace_replacer_dtype(self, replacer: Any) -> None: ...

    def test_replace_after_convert_dtypes(self) -> None: ...
    def test_replace_invalid_to_replace(self) -> None: ...

    @pytest.mark.parametrize("dtype", ["float", "float64", "int64", "Int64", "boolean"])
    @pytest.mark.parametrize("value", [np.nan, pd.NA])
    def test_replace_no_replacement_dtypes(self, dtype: str, value: Any) -> None: ...

    @pytest.mark.parametrize("replacement", [np.nan, 5])
    def test_replace_with_duplicate_columns(self, replacement: Any) -> None: ...

    @pytest.mark.parametrize("value", [pd.Period("2020-01"), pd.Interval(0, 5)])
    def test_replace_ea_ignore_float(self, frame_or_series: type, value: Any) -> None: ...

    @pytest.mark.parametrize(
        "replace_dict, final_data",
        [({"a": 1, "b": 1}, [[2, 2], [2, 2]]), ({"a": 1, "b": 2}, [[2, 1], [2, 2]])],
    )
    def test_categorical_replace_with_dict(self, replace_dict: dict[str, int], final_data: list[list[int]]) -> None: ...

    def test_replace_value_category_type(self) -> None: ...
    def test_replace_dict_category_type(self) -> None: ...
    def test_replace_with_compiled_regex(self) -> None: ...
    def test_replace_intervals(self) -> None: ...
    def test_replace_unicode(self) -> None: ...
    def test_replace_bytes(self, frame_or_series: type) -> None: ...

    @pytest.mark.parametrize(
        "data, to_replace, value, expected",
        [
            ([1], [1.0], [0], [0]),
            ([1], [1], [0], [0]),
            ([1.0], [1.0], [0], [0.0]),
            ([1.0], [1], [0], [0.0]),
        ],
    )
    @pytest.mark.parametrize("box", [list, tuple, np.array])
    def test_replace_list_with_mixed_type(
        self,
        data: list[Any],
        to_replace: list[Any],
        value: list[Any],
        expected: list[Any],
        box: type,
        frame_or_series: type,
    ) -> None: ...

    @pytest.mark.parametrize("val", [2, np.nan, 2.0])
    def test_replace_value_none_dtype_numeric(self, val: Any) -> None: ...

    def test_replace_with_nil_na(self) -> None: ...


class TestDataFrameReplaceRegex:
    @pytest.mark.parametrize(
        "data",
        [{"a": list("ab.."), "b": list("efgh")}, {"a": list("ab.."), "b": list(range(4))}],
    )
    @pytest.mark.parametrize("to_replace,value", [("\\s*\\.\\s*", np.nan), ("\\s*(\\.)\\s*", "\\1\\1\\1")])
    @pytest.mark.parametrize("compile_regex", [True, False])
    @pytest.mark.parametrize("regex_kwarg", [True, False])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_regex_replace_scalar(
        self,
        data: dict[str, list[Any]],
        to_replace: str,
        value: Any,
        compile_regex: bool,
        regex_kwarg: bool,
        inplace: bool,
    ) -> None: ...

    @pytest.mark.parametrize("regex", [False, True])
    @pytest.mark.parametrize("value", [1, "1"])
    def test_replace_regex_dtype_frame(self, regex: bool, value: int | str) -> None: ...

    def test_replace_with_value_also_being_replaced(self) -> None: ...
    def test_replace_categorical_no_replacement(self) -> None: ...
    def test_replace_object_splitting(self, using_infer_string: bool) -> None: ...