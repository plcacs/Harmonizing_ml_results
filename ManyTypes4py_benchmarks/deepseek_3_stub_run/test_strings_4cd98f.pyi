import sys
from typing import Any, Literal, overload
from typing_extensions import Self
import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.strings.accessor import StringMethods
from pandas._testing import assert_series_equal, assert_index_equal, assert_frame_equal

@pytest.mark.parametrize("pattern", [0, True, Series(["foo", "bar"])])
def test_startswith_endswith_non_str_patterns(pattern: int | bool | Series) -> None: ...

def test_iter_raises() -> None: ...

def test_count(any_string_dtype: str) -> None: ...

def test_count_mixed_object() -> None: ...

def test_repeat(any_string_dtype: str) -> None: ...

def test_repeat_mixed_object() -> None: ...

@pytest.mark.parametrize("arg, repeat", [[None, 4], ["b", None]])
def test_repeat_with_null(any_string_dtype: str, arg: str | None, repeat: int | None) -> None: ...

def test_empty_str_methods(any_string_dtype: str) -> None: ...

@pytest.mark.parametrize(
    "method, expected",
    [
        ("isascii", [True, True, True, True, True, True, True, True, True, True]),
        ("isalnum", [True, True, True, True, True, False, True, True, False, False]),
        ("isalpha", [True, True, True, False, False, False, True, False, False, False]),
        ("isdigit", [False, False, False, True, False, False, False, True, False, False]),
        ("isnumeric", [False, False, False, True, False, False, False, True, False, False]),
        ("isspace", [False, False, False, False, False, False, False, False, False, True]),
        ("islower", [False, True, False, False, False, False, False, False, False, False]),
        ("isupper", [True, False, False, False, True, False, True, False, False, False]),
        ("istitle", [True, False, True, False, True, False, False, False, False, False]),
    ],
)
def test_ismethods(method: str, expected: list[bool], any_string_dtype: str) -> None: ...

@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, True, True, False, True, True, False]),
        ("isdecimal", [False, True, False, False, False, True, False]),
    ],
)
def test_isnumeric_unicode(method: str, expected: list[bool], any_string_dtype: str) -> None: ...

@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, np.nan, True, False, np.nan, True, False]),
        ("isdecimal", [False, np.nan, False, False, np.nan, True, False]),
    ],
)
def test_isnumeric_unicode_missing(method: str, expected: list[bool | float], any_string_dtype: str) -> None: ...

def test_spilt_join_roundtrip(any_string_dtype: str) -> None: ...

def test_spilt_join_roundtrip_mixed_object() -> None: ...

def test_len(any_string_dtype: str) -> None: ...

def test_len_mixed() -> None: ...

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
    method: Literal["index", "rindex"],
    sub: str,
    start: int | None,
    end: int | None,
    index_or_series: type[Index] | type[Series],
    any_string_dtype: str,
    expected: list[int],
) -> None: ...

def test_index_not_found_raises(index_or_series: type[Index] | type[Series], any_string_dtype: str) -> None: ...

@pytest.mark.parametrize("method", ["index", "rindex"])
def test_index_wrong_type_raises(
    index_or_series: type[Index] | type[Series], any_string_dtype: str, method: Literal["index", "rindex"]
) -> None: ...

@pytest.mark.parametrize("method, exp", [["index", [1, 1, 0]], ["rindex", [3, 1, 2]]])
def test_index_missing(any_string_dtype: str, method: Literal["index", "rindex"], exp: list[int]) -> None: ...

def test_pipe_failures(any_string_dtype: str) -> None: ...

@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", "bar", np.nan, "baz"]),
        (0, 3, -1, ["", "", np.nan, ""]),
        (None, None, -1, ["owtoofaa", "owtrabaa", np.nan, "xuqzabaa"]),
        (None, 2, -1, ["owtoo", "owtra", np.nan, "xuqza"]),
        (3, 10, 2, ["oto", "ato", np.nan, "aqx"]),
        (3, 0, -1, ["ofa", "aba", np.nan, "aba"]),
    ],
)
def test_slice(start: int | None, stop: int | None, step: int | None, expected: list[str], any_string_dtype: str) -> None: ...

@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", np.nan, "bar", np.nan, np.nan, None, np.nan, np.nan]),
        (4, 1, -1, ["oof", np.nan, "rab", np.nan, np.nan, None, np.nan, np.nan]),
    ],
)
def test_slice_mixed_object(start: int, stop: int, step: int | None, expected: list[str | None]) -> None: ...

@pytest.mark.parametrize(
    "start,stop,repl,expected",
    [
        (2, 3, None, ["shrt", "a it longer", "evnlongerthanthat", "", np.nan]),
        (2, 3, "z", ["shzrt", "a zit longer", "evznlongerthanthat", "z", np.nan]),
        (2, 2, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (2, 1, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (-1, None, "z", ["shorz", "a bit longez", "evenlongerthanthaz", "z", np.nan]),
        (None, -2, "z", ["zrt", "zer", "zat", "z", np.nan]),
        (6, 8, "z", ["shortz", "a bit znger", "evenlozerthanthat", "z", np.nan]),
        (-10, 3, "z", ["zrt", "a zit longer", "evenlongzerthanthat", "z", np.nan]),
    ],
)
def test_slice_replace(
    start: int | None, stop: int | None, repl: str | None, expected: list[str], any_string_dtype: str
) -> None: ...

@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["aa", "bb", np.nan, "cc"]],
        ["lstrip", ["aa   ", "bb \n", np.nan, "cc  "]],
        ["rstrip", ["  aa", " bb", np.nan, "cc"]],
    ],
)
def test_strip_lstrip_rstrip(any_string_dtype: str, method: Literal["strip", "lstrip", "rstrip"], exp: list[str]) -> None: ...

@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["aa", np.nan, "bb"]],
        ["lstrip", ["aa  ", np.nan, "bb \t\n"]],
        ["rstrip", ["  aa", np.nan, " bb"]],
    ],
)
def test_strip_lstrip_rstrip_mixed_object(method: Literal["strip", "lstrip", "rstrip"], exp: list[str]) -> None: ...

@pytest.mark.parametrize(
    "method, exp",
    [
        ["strip", ["ABC", " BNSD", "LDFJH "]],
        ["lstrip", ["ABCxx", " BNSD", "LDFJH xx"]],
        ["rstrip", ["xxABC", "xx BNSD", "LDFJH "]],
    ],
)
def test_strip_lstrip_rstrip_args(any_string_dtype: str, method: Literal["strip", "lstrip", "rstrip"], exp: list[str]) -> None: ...

@pytest.mark.parametrize("prefix, expected", [("a", ["b", " b c", "bc"]), ("ab", ["", "a b c", "bc"])])
def test_removeprefix(any_string_dtype: str, prefix: str, expected: list[str]) -> None: ...

@pytest.mark.parametrize("suffix, expected", [("c", ["ab", "a b ", "b"]), ("bc", ["ab", "a b c", ""])])
def test_removesuffix(any_string_dtype: str, suffix: str, expected: list[str]) -> None: ...

def test_string_slice_get_syntax(any_string_dtype: str) -> None: ...

def test_string_slice_out_of_bounds_nested() -> None: ...

def test_string_slice_out_of_bounds(any_string_dtype: str) -> None: ...

def test_encode_decode(any_string_dtype: str) -> None: ...

def test_encode_errors_kwarg(any_string_dtype: str) -> None: ...

def test_decode_errors_kwarg() -> None: ...

@pytest.mark.parametrize(
    "form, expected",
    [
        ("NFKC", ["ABC", "ABC", "123", np.nan, "アイエ"]),
        ("NFC", ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"]),
    ],
)
def test_normalize(form: Literal["NFKC", "NFC"], expected: list[str], any_string_dtype: str) -> None: ...

def test_normalize_bad_arg_raises(any_string_dtype: str) -> None: ...

def test_normalize_index() -> None: ...

@pytest.mark.parametrize(
    "values,inferred_type",
    [
        (["a", "b"], "string"),
        (["a", "b", 1], "mixed-integer"),
        (["a", "b", 1.3], "mixed"),
        (["a", "b", 1.3, 1], "mixed-integer"),
        (["aa", datetime(2011, 1, 1)], "mixed"),
    ],
)
def test_index_str_accessor_visibility(
    values: list[Any], inferred_type: str, index_or_series: type[Index] | type[Series]
) -> None: ...

@pytest.mark.parametrize(
    "values,inferred_type",
    [
        ([1, np.nan], "floating"),
        ([datetime(2011, 1, 1)], "datetime64"),
        ([timedelta(1)], "timedelta64"),
    ],
)
def test_index_str_accessor_non_string_values_raises(
    values: list[Any], inferred_type: str, index_or_series: type[Index] | type[Series]
) -> None: ...

def test_index_str_accessor_multiindex_raises() -> None: ...

def test_str_accessor_no_new_attributes(any_string_dtype: str) -> None: ...

def test_cat_on_bytes_raises() -> None: ...

def test_str_accessor_in_apply_func() -> None: ...

def test_zfill() -> None: ...

def test_zfill_with_non_integer_argument() -> None: ...

def test_zfill_with_leading_sign() -> None: ...

def test_get_with_dict_label() -> None: ...

def test_series_str_decode() -> None: ...