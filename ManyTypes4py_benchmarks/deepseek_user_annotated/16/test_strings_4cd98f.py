from datetime import datetime, timedelta
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pandas import DataFrame, Index, MultiIndex, Series
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import is_object_or_nan_string_dtype


@pytest.mark.parametrize("pattern", [0, True, Series(["foo", "bar"])])
def test_startswith_endswith_non_str_patterns(pattern: Union[int, bool, Series]) -> None:
    # GH3485
    ser = Series(["foo", "bar"])
    msg = f"expected a string or tuple, not {type(pattern).__name__}"
    with pytest.raises(TypeError, match=msg):
        ser.str.startswith(pattern)
    with pytest.raises(TypeError, match=msg):
        ser.str.endswith(pattern)


def test_iter_raises() -> None:
    # GH 54173
    ser = Series(["foo", "bar"])
    with pytest.raises(TypeError, match="'StringMethods' object is not iterable"):
        iter(ser.str)


def test_count(any_string_dtype: str) -> None:
    ser = Series(["foo", "foofoo", np.nan, "foooofooofommmfoo"], dtype=any_string_dtype)
    result = ser.str.count("f[o]+")
    expected_dtype = (
        np.float64 if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )
    expected = Series([1, 2, np.nan, 4], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_count_mixed_object() -> None:
    ser = Series(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    result = ser.str.count("a")
    expected = Series([1, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_repeat(any_string_dtype: str) -> None:
    ser = Series(["a", "b", np.nan, "c", np.nan, "d"], dtype=any_string_dtype)

    result = ser.str.repeat(3)
    expected = Series(
        ["aaa", "bbb", np.nan, "ccc", np.nan, "ddd"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.repeat([1, 2, 3, 4, 5, 6])
    expected = Series(
        ["a", "bb", np.nan, "cccc", np.nan, "dddddd"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_repeat_mixed_object() -> None:
    ser = Series(["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0])
    result = ser.str.repeat(3)
    expected = Series(
        ["aaa", np.nan, "bbb", np.nan, np.nan, "foofoofoo", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("arg, repeat", [[None, 4], ["b", None]])
def test_repeat_with_null(any_string_dtype: str, arg: Optional[str], repeat: Optional[int]) -> None:
    # GH: 31632
    ser = Series(["a", arg], dtype=any_string_dtype)
    result = ser.str.repeat([3, repeat])
    expected = Series(["aaa", None], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_empty_str_methods(any_string_dtype: str) -> None:
    empty_str = empty = Series(dtype=any_string_dtype)
    empty_inferred_str = Series(dtype="str")
    if is_object_or_nan_string_dtype(any_string_dtype):
        empty_int = Series(dtype="int64")
        empty_bool = Series(dtype=bool)
    else:
        empty_int = Series(dtype="Int64")
        empty_bool = Series(dtype="boolean")
    empty_object = Series(dtype=object)
    empty_bytes = Series(dtype=object)
    empty_df = DataFrame()

    # GH7241
    # (extract) on empty series

    tm.assert_series_equal(empty_str, empty.str.cat(empty))
    assert "" == empty.str.cat()
    tm.assert_series_equal(empty_str, empty.str.title())
    tm.assert_series_equal(empty_int, empty.str.count("a"))
    tm.assert_series_equal(empty_bool, empty.str.contains("a"))
    tm.assert_series_equal(empty_bool, empty.str.startswith("a"))
    tm.assert_series_equal(empty_bool, empty.str.endswith("a"))
    tm.assert_series_equal(empty_str, empty.str.lower())
    tm.assert_series_equal(empty_str, empty.str.upper())
    tm.assert_series_equal(empty_str, empty.str.replace("a", "b"))
    tm.assert_series_equal(empty_str, empty.str.repeat(3))
    tm.assert_series_equal(empty_bool, empty.str.match("^a"))
    tm.assert_frame_equal(
        DataFrame(columns=range(1), empty.str.extract("()", expand=True),
    )
    tm.assert_frame_equal(
        DataFrame(columns=range(2)), empty.str.extract("()()", expand=True),
    )
    tm.assert_series_equal(empty_str, empty.str.extract("()", expand=False))
    tm.assert_frame_equal(
        DataFrame(columns=range(2)), empty.str.extract("()()", expand=False),
    )
    tm.assert_frame_equal(empty_df.set_axis([], axis=1), empty.str.get_dummies())
    tm.assert_series_equal(empty_str, empty_str.str.join(""))
    tm.assert_series_equal(empty_int, empty.str.len())
    tm.assert_series_equal(empty_object, empty_str.str.findall("a"))
    tm.assert_series_equal(empty_int, empty.str.find("a"))
    tm.assert_series_equal(empty_int, empty.str.rfind("a"))
    tm.assert_series_equal(empty_str, empty.str.pad(42))
    tm.assert_series_equal(empty_str, empty.str.center(42))
    tm.assert_series_equal(empty_object, empty.str.split("a"))
    tm.assert_series_equal(empty_object, empty.str.rsplit("a"))
    tm.assert_series_equal(empty_object, empty.str.partition("a", expand=False))
    tm.assert_frame_equal(empty_df, empty.str.partition("a"))
    tm.assert_series_equal(empty_object, empty.str.rpartition("a", expand=False))
    tm.assert_frame_equal(empty_df, empty.str.rpartition("a"))
    tm.assert_series_equal(empty_str, empty.str.slice(stop=1))
    tm.assert_series_equal(empty_str, empty.str.slice(step=1))
    tm.assert_series_equal(empty_str, empty.str.strip())
    tm.assert_series_equal(empty_str, empty.str.lstrip())
    tm.assert_series_equal(empty_str, empty.str.rstrip())
    tm.assert_series_equal(empty_str, empty.str.wrap(42))
    tm.assert_series_equal(empty_str, empty.str.get(0))
    tm.assert_series_equal(empty_inferred_str, empty_bytes.str.decode("ascii"))
    tm.assert_series_equal(empty_bytes, empty.str.encode("ascii"))
    # ismethods should always return boolean (GH 29624)
    tm.assert_series_equal(empty_bool, empty.str.isalnum())
    tm.assert_series_equal(empty_bool, empty.str.isalpha())
    tm.assert_series_equal(empty_bool, empty.str.isascii())
    tm.assert_series_equal(empty_bool, empty.str.isdigit())
    tm.assert_series_equal(empty_bool, empty.str.isspace())
    tm.assert_series_equal(empty_bool, empty.str.islower())
    tm.assert_series_equal(empty_bool, empty.str.isupper())
    tm.assert_series_equal(empty_bool, empty.str.istitle())
    tm.assert_series_equal(empty_bool, empty.str.isnumeric())
    tm.assert_series_equal(empty_bool, empty.str.isdecimal())
    tm.assert_series_equal(empty_str, empty.str.capitalize())
    tm.assert_series_equal(empty_str, empty.str.swapcase())
    tm.assert_series_equal(empty_str, empty.str.normalize("NFC"))

    table = str.maketrans("a", "b")
    tm.assert_series_equal(empty_str, empty.str.translate(table))


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isascii", [True, True, True, True, True, True, True, True, True, True]),
        ("isalnum", [True, True, True, True, True, False, True, True, False, False]),
        ("isalpha", [True, True, True, False, False, False, True, False, False, False]),
        (
            "isdigit",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isnumeric",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isspace",
            [False, False, False, False, False, False, False, False, False, True],
        ),
        (
            "islower",
            [False, True, False, False, False, False, False, False, False, False],
        ),
        (
            "isupper",
            [True, False, False, False, True, False, True, False, False, False],
        ),
        (
            "istitle",
            [True, False, True, False, True, False, False, False, False, False],
        ),
    ],
)
def test_ismethods(method: str, expected: List[bool], any_string_dtype: str) -> None:
    ser = Series(
        ["A", "b", "Xy", "4", "3A", "", "TT", "55", "-", "  "], dtype=any_string_dtype
    )
    expected_dtype = (
        "bool" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected_series = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected_series)

    # compare with standard library
    expected_stdlib = [getattr(item, method)() for item in ser]
    assert list(result) == expected_stdlib

    # with missing value
    ser.iloc[[1, 2, 3, 4]] = np.nan
    result = getattr(ser.str, method)()
    if ser.dtype == "object":
        expected_series = expected_series.astype(object)
        expected_series.iloc[[1, 2, 3, 4]] = np.nan
    elif ser.dtype == "str":
        # NaN propagates as False
        expected_series.iloc[[1, 2, 3, 4]] = False
    else:
        # nullable dtypes propagate NaN
        expected_series.iloc[[1, 2, 3, 4]] = np.nan


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, True, True, False, True, True, False]),
        ("isdecimal", [False, True, False, False, False, True, False]),
    ],
)
def test_isnumeric_unicode(method: str, expected: List[bool], any_string_dtype: str) -> None:
    # 0x00bc: ¼ VULGAR FRACTION ONE QUARTER
    # 0x2605: ★ not number
    # 0x1378: ፸ ETHIOPIC NUMBER SEVENTY
    # 0xFF13: ３ Em 3  # noqa: RUF003
    ser = Series(
        ["A", "3", "¼", "★", "፸", "３", "four"],  # noqa: RUF001
        dtype=any_string_dtype,
    )
    expected_dtype = (
        "bool" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
    )
    expected_series = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected_series)

    # compare with standard library
    expected_stdlib = [getattr(item, method)() for item in ser]
    assert list(result) == expected_stdlib


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, np.nan, True, False, np.nan, True, False]),
        ("isdecimal", [False, np.nan, False, False, np.nan, True, False]),
    ],
)
def test_isnumeric_unicode_missing(method: str, expected: List[Union[bool, float]], any_string_dtype: str) -> None:
    values = ["A", np.nan, "¼", "★", np.nan, "３", "four"]  # noqa: RUF001
    ser = Series(values, dtype=any_string_dtype)
    if any_string_dtype == "str":
        # NaN propagates as False
        expected_series = Series(expected, dtype=object).fillna(False).astype(bool)
    else:
        expected_dtype = (
            "object" if is_object_or_nan_string_dtype(any_string_dtype) else "boolean"
        )
        expected_series = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected_series)


def test_spilt_join_roundtrip(any_string_dtype: str) -> None:
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    result = ser.str.split("_").str.join("_")
    expected = ser.astype(object)
    tm.assert_series_equal(result, expected)


def test_spilt_join_roundtrip_mixed_object() -> None:
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    result = ser.str.split("_").str.join("_")
    expected = Series(
        ["a_b", np.nan, "asdf_cas_asdf", np.nan, np.nan, "foo", None, np.nan, np.nan],
        dtype=object,
    )
    tm.assert_series_equal(result, expected)


def test_len(any_string_dtype: str) -> None:
    ser = Series(
        ["foo", "fooo", "fooooo", np.nan, "fooooooo", "foo\n", "あ"],
        dtype=any_string_dtype,
    )
    result = ser.str.len()
    expected_dtype = (
        "float64" if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )
    expected = Series([3, 4, 6, np.nan, 8, 4, 1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_len_mixed() -> None:
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    result = ser.str.len()
    expected = Series([3, np.nan, 13, np.nan, np.nan, 3, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method,sub,start,end,expected",
    [
        ("index", "EF", None, None, [4, 3, 1, 0]),
        ("rindex", "EF", None, None, [4, 5, 7, 4]),
        ("index", "EF", 3, None, [4, 3, 7, 4]),
        ("rindex", "EF", 3, None, [4, 5, 7, 4]),
        ("index", "E", 4, 8, [4, 5, 7, 4]),
        ("rindex", "E", 0, 5, [4, 3, 1, 4]),
    ],
)
def test_index(
    method: str,
    sub: str,
    start: Optional[int],
    end: Optional[int],
    index_or_series: Any,
    any_string_dtype: str,
    expected: List[int],
) -> None:
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    expected_dtype = (
        np.int64 if is_object_or_nan_string_dtype(any_string_dtype) else "Int64"
    )
    expected_obj = index_or_series(expected, dtype=expected_dtype)

    result = getattr(obj.str, method)(sub, start, end)

    if index_or_series is Series:
        tm.assert_series_equal(result, expected_obj)
    else:
        tm.assert_index_equal(result, expected_obj)

    # compare with standard library
    expected_stdlib = [getattr(item, method)(sub, start, end) for item in obj]
    assert list(result) == expected_stdlib


def test_index_not_found_raises(index_or_series: Any, any_string_dtype: str) -> None:
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
   