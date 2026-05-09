from typing import Any, List, Optional, Union, Dict, Tuple, Callable, Iterable, TypeVar
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.strings.accessor import StringMethods

@pytest.mark.parametrize('pattern', [0, True, Series(['foo', 'bar'])])
def test_startswith_endswith_non_str_patterns(pattern: Union[int, bool, Series]) -> None:
    ...

def test_iter_raises() -> None:
    ...

def test_count(any_string_dtype: str) -> None:
    ...

def test_count_mixed_object() -> None:
    ...

def test_repeat(any_string_dtype: str) -> None:
    ...

def test_repeat_mixed_object() -> None:
    ...

@pytest.mark.parametrize('arg, repeat', [[None, 4], ['b', None]])
def test_repeat_with_null(any_string_dtype: str, arg: Optional[str], repeat: Optional[int]) -> None:
    ...

def test_empty_str_methods(any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('method, expected', [('isascii', [True]*10), ('isalnum', [True]*10)])
def test_ismethods(method: str, expected: List[bool], any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('method, expected', [('isnumeric', [False, True, True, False, True, True, False]), ('isdecimal', [False, True, False, False, False, True, False])])
def test_isnumeric_unicode(method: str, expected: List[bool], any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('method, expected', [('isnumeric', [False, np.nan, True, False, np.nan, True, False]), ('isdecimal', [False, np.nan, False, False, np.nan, True, False])])
def test_isnumeric_unicode_missing(method: str, expected: List[Union[bool, float]], any_string_dtype: str) -> None:
    ...

def test_spilt_join_roundtrip(any_string_dtype: str) -> None:
    ...

def test_spilt_join_roundtrip_mixed_object() -> None:
    ...

def test_len(any_string_dtype: str) -> None:
    ...

def test_len_mixed() -> None:
    ...

@pytest.mark.parametrize('method,sub,start,end,expected', [('index', 'EF', None, None, [4, 3, 1, 0])])
def test_index(method: str, sub: str, start: Optional[int], end: Optional[int], index_or_series: Union[Series, Index], any_string_dtype: str, expected: List[int]) -> None:
    ...

def test_index_not_found_raises(index_or_series: Union[Series, Index], any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('method', ['index', 'rindex'])
def test_index_wrong_type_raises(index_or_series: Union[Series, Index], any_string_dtype: str, method: str) -> None:
    ...

@pytest.mark.parametrize('method, exp', [('index', [1, 1, 0]), ('rindex', [3, 1, 2])])
def test_index_missing(any_string_dtype: str, method: str, exp: List[int]) -> None:
    ...

def test_pipe_failures(any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('start, stop, step, expected', [(2, 5, None, ['foo', 'bar', np.nan, 'baz'])])
def test_slice(start: int, stop: int, step: Optional[int], expected: List[str], any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('start, stop, step, expected', [(2, 5, None, ['foo', np.nan, 'bar', np.nan, np.nan, None, np.nan, np.nan])])
def test_slice_mixed_object(start: int, stop: int, step: Optional[int], expected: List[Union[str, float, None]]) -> None:
    ...

@pytest.mark.parametrize('start,stop,repl,expected', [(2, 3, None, ['shrt', 'a it longer', 'evnlongerthanthat', '', np.nan])])
def test_slice_replace(start: int, stop: int, repl: Optional[str], expected: List[str], any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('method, exp', [('strip', ['aa', 'bb', np.nan, 'cc']), ('lstrip', ['aa   ', 'bb \n', np.nan, 'cc  ']), ('rstrip', ['  aa', ' bb', np.nan, 'cc'])])
def test_strip_lstrip_rstrip(any_string_dtype: str, method: str, exp: List[Union[str, float]]) -> None:
    ...

@pytest.mark.parametrize('method, exp', [('strip', ['aa', np.nan, 'bb']), ('lstrip', ['aa  ', np.nan, 'bb \t\n']), ('rstrip', ['  aa', np.nan, ' bb'])])
def test_strip_lstrip_rstrip_mixed_object(method: str, exp: List[Union[str, float, None]]) -> None:
    ...

@pytest.mark.parametrize('method, exp', [('strip', ['ABC', ' BNSD', 'LDFJH ']), ('lstrip', ['ABCxx', ' BNSD', 'LDFJH xx']), ('rstrip', ['xxABC', 'xx BNSD', 'LDFJH '])])
def test_strip_lstrip_rstrip_args(any_string_dtype: str, method: str, exp: List[str]) -> None:
    ...

@pytest.mark.parametrize('prefix, expected', [('a', ['b', ' b c', 'bc']), ('ab', ['', 'a b c', 'bc'])])
def test_removeprefix(any_string_dtype: str, prefix: str, expected: List[str]) -> None:
    ...

@pytest.mark.parametrize('suffix, expected', [('c', ['ab', 'a b ', 'b']), ('bc', ['ab', 'a b c', ''])])
def test_removesuffix(any_string_dtype: str, suffix: str, expected: List[str]) -> None:
    ...

def test_string_slice_get_syntax(any_string_dtype: str) -> None:
    ...

def test_string_slice_out_of_bounds_nested() -> None:
    ...

def test_string_slice_out_of_bounds(any_string_dtype: str) -> None:
    ...

def test_encode_decode(any_string_dtype: str) -> None:
    ...

def test_encode_errors_kwarg(any_string_dtype: str) -> None:
    ...

def test_decode_errors_kwarg() -> None:
    ...

@pytest.mark.parametrize('form, expected', [('NFKC', ['ABC', 'ABC', '123', np.nan, 'アイエ']), ('NFC', ['ABC', 'ＡＢＣ', '１２３', np.nan, 'ｱｲｴ'])])
def test_normalize(form: str, expected: List[Union[str, float]], any_string_dtype: str) -> None:
    ...

def test_normalize_bad_arg_raises(any_string_dtype: str) -> None:
    ...

def test_normalize_index() -> None:
    ...

@pytest.mark.parametrize('values,inferred_type', [(['a', 'b'], 'string'), (['a', 'b', 1], 'mixed-integer')])
def test_index_str_accessor_visibility(values: List, inferred_type: str, index_or_series: Union[Series, Index]) -> None:
    ...

@pytest.mark.parametrize('values,inferred_type', [([1, np.nan], 'floating'), ([datetime(2011, 1, 1)], 'datetime64')])
def test_index_str_accessor_non_string_values_raises(values: List, inferred_type: str, index_or_series: Union[Series, Index]) -> None:
    ...

def test_index_str_accessor_multiindex_raises() -> None:
    ...

def test_str_accessor_no_new_attributes(any_string_dtype: str) -> None:
    ...

def test_cat_on_bytes_raises() -> None:
    ...

def test_str_accessor_in_apply_func() -> None:
    ...

def test_zfill() -> None:
    ...

def test_zfill_with_non_integer_argument() -> None:
    ...

def test_zfill_with_leading_sign() -> None:
    ...

def test_get_with_dict_label() -> None:
    ...

def test_series_str_decode() -> None:
    ...