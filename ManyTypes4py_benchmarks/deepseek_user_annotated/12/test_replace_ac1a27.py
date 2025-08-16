from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm


@pytest.fixture
def mix_ab() -> dict[str, list[Union[int, str]]:
    return {"a": list(range(4)), "b": list("ab..")}


@pytest.fixture
def mix_abc() -> dict[str, list[Union[float, str]]:
    return {"a": list(range(4)), "b": list("ab.."), "c": ["a", "b", np.nan, "d"]}


class TestDataFrameReplace:
    def test_replace_inplace(self, datetime_frame: DataFrame, float_string_frame: DataFrame) -> None:
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        tsframe = datetime_frame.copy()
        return_value = tsframe.replace(np.nan, 0, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))

        # mixed type
        mf = float_string_frame
        mf.iloc[5:20, mf.columns.get_loc("foo")] = np.nan
        mf.iloc[-10:, mf.columns.get_loc("A")] = np.nan

        result = float_string_frame.replace(np.nan, 0)
        expected = float_string_frame.copy()
        expected["foo"] = expected["foo"].astype(object)
        expected = expected.fillna(value=0)
        tm.assert_frame_equal(result, expected)

        tsframe = datetime_frame.copy()
        return_value = tsframe.replace([np.nan], [0], inplace=True)
        assert return_value is None
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))

    @pytest.mark.parametrize(
        "to_replace,values,expected",
        [
            # lists of regexes and values
            # list of [re1, re2, ..., reN] -> [v1, v2, ..., vN]
            (
                [r"\s*\.\s*", r"e|f|g"],
                [np.nan, "crap"],
                {
                    "a": ["a", "b", np.nan, np.nan],
                    "b": ["crap"] * 3 + ["h"],
                    "c": ["h", "crap", "l", "o"],
                },
            ),
            # list of [re1, re2, ..., reN] -> [re1, re2, .., reN]
            (
                [r"\s*(\.)\s*", r"(e|f|g)"],
                [r"\1\1", r"\1_crap"],
                {
                    "a": ["a", "b", "..", ".."],
                    "b": ["e_crap", "f_crap", "g_crap", "h"],
                    "c": ["h", "e_crap", "l", "o"],
                },
            ),
            # list of [re1, re2, ..., reN] -> [(re1 or v1), (re2 or v2), ..., (reN
            # or vN)]
            (
                [r"\s*(\.)\s*", r"e"],
                [r"\1\1", r"crap"],
                {
                    "a": ["a", "b", "..", ".."],
                    "b": ["crap", "f", "g", "h"],
                    "c": ["h", "crap", "l", "o"],
                },
            ),
        ],
    )
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("use_value_regex_args", [True, False])
    def test_regex_replace_list_obj(
        self,
        to_replace: List[str],
        values: List[Union[str, float]],
        expected: Dict[str, List[Union[str, float]]],
        inplace: bool,
        use_value_regex_args: bool,
    ) -> None:
        df = DataFrame({"a": list("ab.."), "b": list("efgh"), "c": list("helo")})

        if use_value_regex_args:
            result = df.replace(value=values, regex=to_replace, inplace=inplace)
        else:
            result = df.replace(to_replace, values, regex=True, inplace=inplace)

        if inplace:
            assert result is None
            result = df

        expected_df = DataFrame(expected)
        tm.assert_frame_equal(result, expected_df)

    def test_regex_replace_list_mixed(self, mix_ab: Dict[str, List[Union[int, str]]]) -> None:
        # mixed frame to make sure this doesn't break things
        dfmix = DataFrame(mix_ab)

        # lists of regexes and values
        # list of [re1, re2, ..., reN] -> [v1, v2, ..., vN]
        to_replace_res = [r"\s*\.\s*", r"a"]
        values = [np.nan, "crap"]
        mix2 = {"a": list(range(4)), "b": list("ab.."), "c": list("halo")}
        dfmix2 = DataFrame(mix2)
        res = dfmix2.replace(to_replace_res, values, regex=True)
        expec = DataFrame(
            {
                "a": mix2["a"],
                "b": ["crap", "b", np.nan, np.nan],
                "c": ["h", "crap", "l", "o"],
            }
        )
        tm.assert_frame_equal(res, expec)

        # list of [re1, re2, ..., reN] -> [re1, re2, .., reN]
        to_replace_res = [r"\s*(\.)\s*", r"(a|b)"]
        values = [r"\1\1", r"\1_crap"]
        res = dfmix.replace(to_replace_res, values, regex=True)
        expec = DataFrame({"a": mix_ab["a"], "b": ["a_crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

        # list of [re1, re2, ..., reN] -> [(re1 or v1), (re2 or v2), ..., (reN
        # or vN)]
        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.replace(to_replace_res, values, regex=True)
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.replace(regex=to_replace_res, value=values)
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

    def test_regex_replace_list_mixed_inplace(self, mix_ab: Dict[str, List[Union[int, str]]]) -> None:
        dfmix = DataFrame(mix_ab)
        # the same inplace
        # lists of regexes and values
        # list of [re1, re2, ..., reN] -> [v1, v2, ..., vN]
        to_replace_res = [r"\s*\.\s*", r"a"]
        values = [np.nan, "crap"]
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b", np.nan, np.nan]})
        tm.assert_frame_equal(res, expec)

        # list of [re1, re2, ..., reN] -> [re1, re2, .., reN]
        to_replace_res = [r"\s*(\.)\s*", r"(a|b)"]
        values = [r"\1\1", r"\1_crap"]
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({"a": mix_ab["a"], "b": ["a_crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

        # list of [re1, re2, ..., reN] -> [(re1 or v1), (re2 or v2), ..., (reN
        # or vN)]
        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        assert return_value is None
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.copy()
        return_value = res.replace(regex=to_replace_res, value=values, inplace=True)
        assert return_value is None
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)

    def test_regex_replace_dict_mixed(self, mix_abc: Dict[str, List[Union[float, str]]]) -> None:
        dfmix = DataFrame(mix_abc)

        # dicts
        # single dict {re1: v1}, search the whole frame
        # need test for this...

        # list of dicts {re1: v1, re2: v2, ..., re3: v3}, search the whole
        # frame
        res = dfmix.replace({"b": r"\s*\.\s*"}, {"b": np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(
            {"b": r"\s*\.\s*"}, {"b": np.nan}, inplace=True, regex=True
        )
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", np.nan, np.nan], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        # list of dicts {re1: re11, re2: re12, ..., reN: re1N}, search the
        # whole frame
        res = dfmix.replace({"b": r"\s*(\.)\s*"}, {"b": r"\1ty"}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(
            {"b": r"\s*(\.)\s*"}, {"b": r"\1ty"}, inplace=True, regex=True
        )
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", ".ty", ".ty"], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        res = dfmix.replace(regex={"b": r"\s*(\.)\s*"}, value={"b": r"\1ty"})
        res2 = dfmix.copy()
        return_value = res2.replace(
            regex={"b": r"\s*(\.)\s*"}, value={"b": r"\1ty"}, inplace=True
        )
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", ".ty", ".ty"], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        # scalar -> dict
        # to_replace regex, {value: value}
        expec = DataFrame(
            {"a": mix_abc["a"], "b": [np.nan, "b", ".", "."], "c": mix_abc["c"]}
        )
        res = dfmix.replace("a", {"b": np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace("a", {"b": np.nan}, regex=True, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        res = dfmix.replace("a", {"b": np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(regex="a", value={"b": np.nan}, inplace=True)
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": [np.nan, "b", ".", "."], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

    def test_regex_replace_dict_nested(self, mix_abc: Dict[str, List[Union[float, str]]]) -> None:
        # nested dicts will not work until this is implemented for Series
        dfmix = DataFrame(mix_abc)
        res = dfmix.replace({"b": {r"\s*\.\s*": np.nan}}, regex=True)
        res2 = dfmix.copy()
        res4 = dfmix.copy()
        return_value = res2.replace(
            {"b": {r"\s*\.\s*": np.nan}}, inplace=True, regex=True
        )
        assert return_value is None
        res3 = dfmix.replace(regex={"b": {r"\s*\.\s*": np.nan}})
        return_value = res4.replace(regex={"b": {r"\s*\.\s*": np.nan}}, inplace=True)
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", np.nan, np.nan], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)
        tm.assert_frame_equal(res4, expec)

    def test_regex_replace_dict_nested_non_first_character(
        self, any_string_dtype: str, using_infer_string: bool
    ) -> None:
        # GH 25259
        dtype = any_string_dtype
        df = DataFrame({"first": ["abc", "bca", "cab"]}, dtype=dtype)
        result = df.replace({"a": "."}, regex=True)
        expected = DataFrame({"first": [".bc", "bc.", "c.b"]}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_dict_nested_gh4115(self) -> None:
        df = DataFrame(
            {"Type": Series(["Q", "T", "Q", "Q", "T"], dtype=object), "tmp": 2}
        )
        expected = DataFrame({"Type": Series([0, 1, 0, 0, 1], dtype=object), "tmp": 2})
        result = df.replace({"Type": {"Q": 0, "T": 1}})
        tm.assert_frame_equal(result, expected)

    def test_regex_replace_list_to_scalar(self, mix_abc: Dict[str, List[Union[float, str]]]) -> None:
        df = DataFrame(mix_abc)
        expec = DataFrame(
            {
                "a": mix_abc["a"],
                "b": Series([np.nan] * 4, dtype="str"),
                "c": [np.nan, np.nan, np.nan, "d"],
            }
        )

        res = df.replace([r"\s*\.\s*", "a|b"], np.nan, regex=True)
        res2 = df.copy()
        res3 = df.copy()
        return_value = res2.replace(
            [r"\s*\.\s*", "a|b"], np.nan, regex=True, inplace=True
        )
        assert return_value is None
        return_value = res3.replace(
            regex=[r"\s*\.\s*", "a|b"], value=np.nan, inplace=True
        )
        assert return_value is None
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    def test_regex_replace_str_to_numeric(self, mix_abc: Dict[str, List[Union[float, str]]]) -> None:
        # what happens when you try to replace a numeric value with a regex?
        df = DataFrame(mix_abc)
        res = df.replace(r"\s*\.\s*", 0, regex=True)
        res2 = df.copy()
        return_value = res2.replace(r"\s*\.\s*", 0, inplace=True, regex=True)
        assert return_value is None
        res3 = df.copy()
        return_value = res3.replace(regex=r"\s*\.\s*", value=0, inplace=True)
        assert return_value is None
        expec = DataFrame({"a": mix_