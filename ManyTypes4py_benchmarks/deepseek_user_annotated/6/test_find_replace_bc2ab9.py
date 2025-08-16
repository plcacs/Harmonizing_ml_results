from datetime import datetime
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence, Tuple, Union

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import Series
from pandas import _testing as tm
from pandas.tests.strings import _convert_na_value, is_object_or_nan_string_dtype

# --------------------------------------------------------------------------------------
# str.contains
# --------------------------------------------------------------------------------------

def test_contains(any_string_dtype: str) -> None:
    values = np.array(
        ["foo", np.nan, "fooommm__foo", "mmm_", "foommm[_]+bar"], dtype=np.object_
    )
    values = Series(values, dtype=any_string_dtype)
    pat = "mmm[_]+"

    result = values.str.contains(pat)
    if any_string_dtype == "str":
        expected = Series([False, False, True, True, False], dtype=bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series(
            np.array([False, np.nan, True, True, False], dtype=np.object_),
            dtype=expected_dtype,
        )

    tm.assert_series_equal(result, expected)

    result = values.str.contains(pat, regex=False)
    if any_string_dtype == "str":
        expected = Series([False, False, False, False, True], dtype=bool)
    else:
        expected = Series(
            np.array([False, np.nan, False, False, True], dtype=np.object_),
            dtype=expected_dtype,
        )
    tm.assert_series_equal(result, expected)

    values = Series(
        np.array(["foo", "xyz", "fooommm__foo", "mmm_"], dtype=object),
        dtype=any_string_dtype,
    )
    result = values.str.contains(pat)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # case insensitive using regex
    values = Series(
        np.array(["Foo", "xYz", "fOOomMm__fOo", "MMM_"], dtype=object),
        dtype=any_string_dtype,
    )

    result = values.str.contains("FOO|mmm", case=False)
    expected = Series(np.array([True, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # case insensitive without regex
    result = values.str.contains("foo", regex=False, case=False)
    expected = Series(np.array([True, False, True, False]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # unicode
    values = Series(
        np.array(["foo", np.nan, "fooommm__foo", "mmm_"], dtype=np.object_),
        dtype=any_string_dtype,
    )
    pat = "mmm[_]+"

    result = values.str.contains(pat)
    if any_string_dtype == "str":
        expected = Series([False, False, True, True], dtype=bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series(
            np.array([False, np.nan, True, True], dtype=np.object_),
            dtype=expected_dtype,
        )
    tm.assert_series_equal(result, expected)

    result = values.str.contains(pat, na=False)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    values = Series(
        np.array(["foo", "xyz", "fooommm__foo", "mmm_"], dtype=np.object_),
        dtype=any_string_dtype,
    )
    result = values.str.contains(pat)
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

def test_contains_object_mixed() -> None:
    mixed = Series(
        np.array(
            ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
            dtype=object,
        )
    )
    result = mixed.str.contains("o")
    expected = Series(
        np.array(
            [False, np.nan, False, np.nan, np.nan, True, None, np.nan, np.nan],
            dtype=np.object_,
        )
    )
    tm.assert_series_equal(result, expected)

def test_contains_na_kwarg_for_object_category() -> None:
    # gh 22158

    # na for category
    values = Series(["a", "b", "c", "a", np.nan], dtype="category")
    result = values.str.contains("a", na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)

    result = values.str.contains("a", na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)

    # na for objects
    values = Series(["a", "b", "c", "a", np.nan])
    result = values.str.contains("a", na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)

    result = values.str.contains("a", na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    "na, expected",
    [
        (None, pd.NA),
        (True, True),
        (False, False),
        (0, False),
        (3, True),
        (np.nan, pd.NA),
    ],
)
@pytest.mark.parametrize("regex", [True, False])
def test_contains_na_kwarg_for_nullable_string_dtype(
    nullable_string_dtype: str, na: Any, expected: Any, regex: bool
) -> None:
    # https://github.com/pandas-dev/pandas/pull/41025#issuecomment-824062416

    values = Series(["a", "b", "c", "a", np.nan], dtype=nullable_string_dtype)

    msg = (
        "Allowing a non-bool 'na' in obj.str.contains is deprecated and "
        "will raise in a future version"
    )
    warn = None
    if not pd.isna(na) and not isinstance(na, bool):
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = values.str.contains("a", na=na, regex=regex)
    expected = Series([True, False, False, True, expected], dtype="boolean")
    tm.assert_series_equal(result, expected)

def test_contains_moar(any_string_dtype: str) -> None:
    # PR #1179
    s = Series(
        ["A", "B", "C", "Aaba", "Baca", "", np.nan, "CABA", "dog", "cat"],
        dtype=any_string_dtype,
    )

    result = s.str.contains("a")
    if any_string_dtype == "str":
        expected_dtype = bool
        na_value = False
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        na_value = np.nan
    expected = Series(
        [False, False, False, True, True, False, na_value, False, False, True],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("a", case=False)
    expected = Series(
        [True, False, False, True, True, False, na_value, True, False, True],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("Aa")
    expected = Series(
        [False, False, False, True, False, False, na_value, False, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("ba")
    expected = Series(
        [False, False, False, True, False, False, na_value, False, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("ba", case=False)
    expected = Series(
        [False, False, False, True, True, False, na_value, True, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

def test_contains_nan(any_string_dtype: str) -> None:
    # PR #14171
    s = Series([np.nan, np.nan, np.nan], dtype=any_string_dtype)

    result = s.str.contains("foo", na=False)
    expected_dtype = (
        np.bool_ if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected = Series([False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = s.str.contains("foo", na=True)
    expected = Series([True, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    msg = (
        "Allowing a non-bool 'na' in obj.str.contains is deprecated and "
        "will raise in a future version"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.str.contains("foo", na="foo")
    if any_string_dtype == "object":
        expected = Series(["foo", "foo", "foo"], dtype=np.object_)
    elif any_string_dtype.na_value is np.nan:
        expected = Series([True, True, True], dtype=np.bool_)
    else:
        expected = Series([True, True, True], dtype="boolean")
    tm.assert_series_equal(result, expected)

    result = s.str.contains("foo")
    if any_string_dtype == "str":
        expected = Series([False, False, False], dtype=bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected = Series([np.nan, np.nan, np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

# --------------------------------------------------------------------------------------
# str.startswith
# --------------------------------------------------------------------------------------

def test_startswith_endswith_validate_na(any_string_dtype: str) -> None:
    # GH#59615
    ser = Series(
        ["om", np.nan, "foo_nom", "nom", "bar_foo", np.nan, "foo"],
        dtype=any_string_dtype,
    )

    msg = "Allowing a non-bool 'na' in obj.str.startswith is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        ser.str.startswith("kapow", na="baz")
    msg = "Allowing a non-bool 'na' in obj.str.endswith is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        ser.str.endswith("bar", na="baz")

@pytest.mark.parametrize("pat", ["foo", ("foo", "baz")])
@pytest.mark.parametrize("dtype", ["object", "category"])
@pytest.mark.parametrize("null_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("na", [True, False])
def test_startswith(
    pat: Union[str, Tuple[str, str]],
    dtype: str,
    null_value: Any,
    na: bool,
    using_infer_string: bool,
) -> None:
    # add category dtype parametrizations for GH-36241
    values = Series(
        ["om", null_value, "foo_nom", "nom", "bar_foo", null_value, "foo"],
        dtype=dtype,
    )

    result = values.str.startswith(pat)
    exp = Series([False, np.nan, True, False, False, np.nan, True])
    if dtype == "object" and null_value is pd.NA:
        exp = exp.fillna(null_value)
    elif dtype == "object" and null_value is None:
        exp[exp.isna()] = None
    elif using_infer_string and dtype == "category":
        exp = exp.fillna(False).astype(bool)
    tm.assert_series_equal(result, exp)

    result = values.str.startswith(pat, na=na)
    exp = Series([False, na, True, False, False, na, True])
    tm.assert_series_equal(result, exp)

    # mixed
    mixed = np.array(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=np.object_,
    )
    rs = Series(mixed).str.startswith("f")
    xp = Series([False, np.nan, False, np.nan, np.nan, True, None, np.nan, np.nan])
    tm.assert_series_equal(rs, xp)

@pytest.mark.parametrize("na", [None, True, False])
def test_startswith_string_dtype(any_string_dtype: str, na: Optional[bool]) -> None:
    values = Series(
        ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
        dtype=any_string_dtype,
    )
    result = values.str.startswith("foo", na=na)

    expected_dtype = (
        (object if na is None else bool)
        if is_object_or_nan_string_dtype(any_string_dtype)
        else "boolean"
    )
    if any_string_dtype == "str":
        expected_dtype = bool
        if na is None:
            na = False
    exp = Series(
        [False, na, True, False, False, na, True, False, False], dtype=expected_dtype
    )
    tm.assert_series_equal(result, exp)

    result = values.str.startswith("rege.", na=na)
    exp = Series(
        [False, na, False, False, False, na, False, False, True], dtype=expected_dtype
    )
    tm.assert_series_equal(result, exp)

# --------------------------------------------------------------------------------------
# str.endswith
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("pat", ["foo", ("foo", "baz")])
@pytest.mark.parametrize("dtype", ["object", "category"])
@pytest.mark.parametrize("null_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("na", [True, False])
def test_endswith(
    pat: Union[str, Tuple[str, str]],
    dtype: str,
    null_value: Any,
    na: bool,
    using_infer_string: bool,
) -> None:
    # add category dtype parametrizations for GH-36241
    values = Series(
        ["om", null_value, "foo_nom", "nom", "bar_foo", null_value, "foo"],
        dtype=dtype,
    )

    result = values.str.endswith(pat)
    exp = Series([False, np.nan, False, False, True, np.nan, True])
    if dtype == "object" and null_value is pd.NA:
        exp = exp.fillna(null_value)
    elif dtype == "object" and null_value is None:
        exp[exp.isna()] = None
    elif using_infer_string and dtype == "category":
        exp = exp.fillna(False).astype(bool)
    tm.assert_series_equal(result, exp)

    result = values.str.endswith(pat, na=na)
    exp = Series([False, na, False, False, True, na, True])
    tm.assert_series_equal(result, exp)

    # mixed
    mixed = np.array(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    rs = Series(mixed).str.endswith("f")
    xp = Series([False, np.nan, False, np.nan, np.nan, False, None, np.nan, np.nan])
    tm.assert_series_equal(rs, xp)

@pytest.mark.parametrize("na", [None, True, False])
def test_endswith_string_dtype(any_string_dtype: str, na: Optional[bool]) -> None:
    values = Series(
        ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
        dtype=any_string_dtype,
    )
    result = values.str.endswith("foo", na=na)
    expected_dtype = (
        (object if na is None else bool)
        if is_object_or_nan_string_dtype(any_string_dtype)
        else "boolean"
    )
    if any_string_dtype == "str":
        expected_dtype = bool
        if na is None:
            na = False
    exp = Series(
        [False, na, False, False, True, na, True, False, False], dtype=expected_dtype
    )
    tm.assert_series_equal(result, exp)

    result = values.str.endswith("rege.", na=na)
    exp = Series(
        [False, na, False, False, False, na, False, False, True], dtype=expected_dtype
    )
    tm.assert_series_equal(result, exp)

# --------------------------------------------------------------------------------------
# str.replace
# --------------------------------------------------------------------------------------

def test_replace_dict_invalid(any_string_dtype: str) -> None:
    # GH 51914
