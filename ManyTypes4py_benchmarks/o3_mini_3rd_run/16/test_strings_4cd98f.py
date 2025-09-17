#!/usr/bin/env python
from datetime import datetime, timedelta
from typing import Any, Callable, List, Optional, Union
import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import is_object_or_nan_string_dtype

@pytest.mark.parametrize('pattern', [0, True, Series(['foo', 'bar'])])
def test_startswith_endswith_non_str_patterns(pattern: Any) -> None:
    ser: Series = Series(['foo', 'bar'])
    msg: str = f'expected a string or tuple, not {type(pattern).__name__}'
    with pytest.raises(TypeError, match=msg):
        ser.str.startswith(pattern)
    with pytest.raises(TypeError, match=msg):
        ser.str.endswith(pattern)

def test_iter_raises() -> None:
    ser: Series = Series(['foo', 'bar'])
    with pytest.raises(TypeError, match="'StringMethods' object is not iterable"):
        iter(ser.str)

def test_count(any_string_dtype: str) -> None:
    ser: Series = Series(['foo', 'foofoo', np.nan, 'foooofooofommmfoo'], dtype=any_string_dtype)
    result: Series = ser.str.count('f[o]+')
    expected_dtype: Union[str, type] = np.float64 if is_object_or_nan_string_dtype(any_string_dtype) else 'Int64'
    expected: Series = Series([1, 2, np.nan, 4], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

def test_count_mixed_object() -> None:
    ser: Series = Series(['a', np.nan, 'b', True, datetime.today(), 'foo', None, 1, 2.0], dtype=object)
    result: Series = ser.str.count('a')
    expected: Series = Series([1, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

def test_repeat(any_string_dtype: str) -> None:
    ser: Series = Series(['a', 'b', np.nan, 'c', np.nan, 'd'], dtype=any_string_dtype)
    result: Series = ser.str.repeat(3)
    expected: Series = Series(['aaa', 'bbb', np.nan, 'ccc', np.nan, 'ddd'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    result = ser.str.repeat([1, 2, 3, 4, 5, 6])
    expected = Series(['a', 'bb', np.nan, 'cccc', np.nan, 'dddddd'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

def test_repeat_mixed_object() -> None:
    ser: Series = Series(['a', np.nan, 'b', True, datetime.today(), 'foo', None, 1, 2.0])
    result: Series = ser.str.repeat(3)
    expected: Series = Series(['aaa', np.nan, 'bbb', np.nan, np.nan, 'foofoofoo', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('arg, repeat', [[None, 4], ['b', None]])
def test_repeat_with_null(any_string_dtype: str, arg: Any, repeat: Any) -> None:
    ser: Series = Series(['a', arg], dtype=any_string_dtype)
    result: Series = ser.str.repeat([3, repeat])
    expected: Series = Series(['aaa', None], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

def test_empty_str_methods(any_string_dtype: str) -> None:
    empty_str: Series = Series(dtype=any_string_dtype)
    empty: Series = empty_str
    empty_inferred_str: Series = Series(dtype='str')
    if is_object_or_nan_string_dtype(any_string_dtype):
        empty_int: Series = Series(dtype='int64')
        empty_bool: Series = Series(dtype=bool)
    else:
        empty_int = Series(dtype='Int64')
        empty_bool = Series(dtype='boolean')
    empty_object: Series = Series(dtype=object)
    empty_bytes: Series = Series(dtype=object)
    empty_df: DataFrame = DataFrame()
    tm.assert_series_equal(empty_str, empty.str.cat(empty))
    assert '' == empty.str.cat()
    tm.assert_series_equal(empty_str, empty.str.title())
    tm.assert_series_equal(empty_int, empty.str.count('a'))
    tm.assert_series_equal(empty_bool, empty.str.contains('a'))
    tm.assert_series_equal(empty_bool, empty.str.startswith('a'))
    tm.assert_series_equal(empty_bool, empty.str.endswith('a'))
    tm.assert_series_equal(empty_str, empty.str.lower())
    tm.assert_series_equal(empty_str, empty.str.upper())
    tm.assert_series_equal(empty_str, empty.str.replace('a', 'b'))
    tm.assert_series_equal(empty_str, empty.str.repeat(3))
    tm.assert_series_equal(empty_bool, empty.str.match('^a'))
    tm.assert_frame_equal(DataFrame(columns=range(1), dtype=any_string_dtype), empty.str.extract('()', expand=True))
    tm.assert_frame_equal(DataFrame(columns=range(2), dtype=any_string_dtype), empty.str.extract('()()', expand=True))
    tm.assert_series_equal(empty_str, empty.str.extract('()', expand=False))
    tm.assert_frame_equal(DataFrame(columns=range(2), dtype=any_string_dtype), empty.str.extract('()()', expand=False))
    tm.assert_frame_equal(empty_df.set_axis([], axis=1), empty.str.get_dummies())
    tm.assert_series_equal(empty_str, empty_str.str.join(''))
    tm.assert_series_equal(empty_int, empty.str.len())
    tm.assert_series_equal(empty_object, empty_str.str.findall('a'))
    tm.assert_series_equal(empty_int, empty.str.find('a'))
    tm.assert_series_equal(empty_int, empty.str.rfind('a'))
    tm.assert_series_equal(empty_str, empty.str.pad(42))
    tm.assert_series_equal(empty_str, empty.str.center(42))
    tm.assert_series_equal(empty_object, empty.str.split('a'))
    tm.assert_series_equal(empty_object, empty.str.rsplit('a'))
    tm.assert_series_equal(empty_object, empty.str.partition('a', expand=False))
    tm.assert_frame_equal(empty_df, empty.str.partition('a'))
    tm.assert_series_equal(empty_object, empty.str.rpartition('a', expand=False))
    tm.assert_frame_equal(empty_df, empty.str.rpartition('a'))
    tm.assert_series_equal(empty_str, empty.str.slice(stop=1))
    tm.assert_series_equal(empty_str, empty.str.slice(step=1))
    tm.assert_series_equal(empty_str, empty.str.strip())
    tm.assert_series_equal(empty_str, empty.str.lstrip())
    tm.assert_series_equal(empty_str, empty.str.rstrip())
    tm.assert_series_equal(empty_str, empty.str.wrap(42))
    tm.assert_series_equal(empty_str, empty.str.get(0))
    tm.assert_series_equal(empty_inferred_str, empty_bytes.str.decode('ascii'))
    tm.assert_series_equal(empty_bytes, empty.str.encode('ascii'))
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
    tm.assert_series_equal(empty_str, empty.str.normalize('NFC'))
    table: dict = str.maketrans('a', 'b')
    tm.assert_series_equal(empty_str, empty.str.translate(table))

@pytest.mark.parametrize(
    'method, expected',
    [
        ('isascii', [True, True, True, True, True, True, True, True, True, True]),
        ('isalnum', [True, True, True, True, True, False, True, True, False, False]),
        ('isalpha', [True, True, True, False, False, False, True, False, False, False]),
        ('isdigit', [False, False, False, True, False, False, False, True, False, False]),
        ('isnumeric', [False, False, False, True, False, False, False, True, False, False]),
        ('isspace', [False, False, False, False, False, False, False, False, False, True]),
        ('islower', [False, True, False, False, False, False, False, False, False, False]),
        ('isupper', [True, False, False, False, True, False, True, False, False, False]),
        ('istitle', [True, False, True, False, True, False, False, False, False, False])
    ]
)
def test_ismethods(method: str, expected: List[bool], any_string_dtype: str) -> None:
    ser: Series = Series(['A', 'b', 'Xy', '4', '3A', '', 'TT', '55', '-', '  '], dtype=any_string_dtype)
    expected_dtype: str = 'bool' if is_object_or_nan_string_dtype(any_string_dtype) else 'boolean'
    expected_series: Series = Series(expected, dtype=expected_dtype)
    result: Series = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected_series)
    expected_stdlib: List[Any] = [getattr(item, method)() for item in ser]
    assert list(result) == expected_stdlib
    ser.iloc[[1, 2, 3, 4]] = np.nan
    result = getattr(ser.str, method)()
    if ser.dtype == 'object':
        expected_series = expected_series.astype(object)
        expected_series.iloc[[1, 2, 3, 4]] = np.nan
    elif ser.dtype == 'str':
        expected_series.iloc[[1, 2, 3, 4]] = False
    else:
        expected_series.iloc[[1, 2, 3, 4]] = np.nan

@pytest.mark.parametrize(
    'method, expected',
    [
        ('isnumeric', [False, True, True, False, True, True, False]),
        ('isdecimal', [False, True, False, False, False, True, False])
    ]
)
def test_isnumeric_unicode(method: str, expected: List[bool], any_string_dtype: str) -> None:
    ser: Series = Series(['A', '3', '¼', '★', '፸', '３', 'four'], dtype=any_string_dtype)
    expected_dtype: str = 'bool' if is_object_or_nan_string_dtype(any_string_dtype) else 'boolean'
    expected_series: Series = Series(expected, dtype=expected_dtype)
    result: Series = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected_series)
    expected_stdlib: List[Any] = [getattr(item, method)() for item in ser]
    assert list(result) == expected_stdlib

@pytest.mark.parametrize(
    'method, expected',
    [
        ('isnumeric', [False, np.nan, True, False, np.nan, True, False]),
        ('isdecimal', [False, np.nan, False, False, np.nan, True, False])
    ]
)
def test_isnumeric_unicode_missing(method: str, expected: List[Optional[Union[bool, float]]], any_string_dtype: str) -> None:
    values: List[Optional[Any]] = ['A', np.nan, '¼', '★', np.nan, '３', 'four']
    ser: Series = Series(values, dtype=any_string_dtype)
    if any_string_dtype == 'str':
        expected_series: Series = Series(expected, dtype=object).fillna(False).astype(bool)
    else:
        expected_dtype: Union[str, type] = 'object' if is_object_or_nan_string_dtype(any_string_dtype) else 'boolean'
        expected_series = Series(expected, dtype=expected_dtype)
    result: Series = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected_series)

def test_spilt_join_roundtrip(any_string_dtype: str) -> None:
    ser: Series = Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'], dtype=any_string_dtype)
    result: Series = ser.str.split('_').str.join('_')
    expected: Series = ser.astype(object)
    tm.assert_series_equal(result, expected)

def test_spilt_join_roundtrip_mixed_object() -> None:
    ser: Series = Series(['a_b', np.nan, 'asdf_cas_asdf', True, datetime.today(), 'foo', None, 1, 2.0])
    result: Series = ser.str.split('_').str.join('_')
    expected: Series = Series(['a_b', np.nan, 'asdf_cas_asdf', np.nan, np.nan, 'foo', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)

def test_len(any_string_dtype: str) -> None:
    ser: Series = Series(['foo', 'fooo', 'fooooo', np.nan, 'fooooooo', 'foo\n', 'あ'], dtype=any_string_dtype)
    result: Series = ser.str.len()
    expected_dtype: Union[str, type] = 'float64' if is_object_or_nan_string_dtype(any_string_dtype) else 'Int64'
    expected: Series = Series([3, 4, 6, np.nan, 8, 4, 1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

def test_len_mixed() -> None:
    ser: Series = Series(['a_b', np.nan, 'asdf_cas_asdf', True, datetime.today(), 'foo', None, 1, 2.0])
    result: Series = ser.str.len()
    expected: Series = Series([3, np.nan, 13, np.nan, np.nan, 3, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    'method,sub,start,end,expected',
    [
        ('index', 'EF', None, None, [4, 3, 1, 0]),
        ('rindex', 'EF', None, None, [4, 5, 7, 4]),
        ('index', 'EF', 3, None, [4, 3, 7, 4]),
        ('rindex', 'EF', 3, None, [4, 5, 7, 4]),
        ('index', 'E', 4, 8, [4, 5, 7, 4]),
        ('rindex', 'E', 0, 5, [4, 3, 1, 4])
    ]
)
def test_index(method: str, sub: str, start: Optional[int], end: Optional[int],
               index_or_series: Callable[..., Union[Series, Index]], any_string_dtype: str,
               expected: List[int]) -> None:
    obj: Union[Series, Index] = index_or_series(['ABCDEFG', 'BCDEFEF', 'DEFGHIJEF', 'EFGHEF'], dtype=any_string_dtype)
    expected_dtype: Union[str, type] = np.int64 if is_object_or_nan_string_dtype(any_string_dtype) else 'Int64'
    expected_series_or_index: Union[Series, Index] = index_or_series(expected, dtype=expected_dtype)
    result = getattr(obj.str, method)(sub, start, end)
    if isinstance(obj, Series):
        tm.assert_series_equal(result, expected_series_or_index)  # type: ignore
    else:
        tm.assert_index_equal(result, expected_series_or_index)  # type: ignore
    expected_stdlib: List[Any] = [getattr(item, method)(sub, start, end) for item in obj]
    assert list(result) == expected_stdlib

def test_index_not_found_raises(index_or_series: Callable[..., Union[Series, Index]], any_string_dtype: str) -> None:
    obj: Union[Series, Index] = index_or_series(['ABCDEFG', 'BCDEFEF', 'DEFGHIJEF', 'EFGHEF'], dtype=any_string_dtype)
    with pytest.raises(ValueError, match='substring not found'):
        obj.str.index('DE')

@pytest.mark.parametrize('method', ['index', 'rindex'])
def test_index_wrong_type_raises(index_or_series: Callable[..., Union[Series, Index]], any_string_dtype: str, method: str) -> None:
    obj: Union[Series, Index] = index_or_series([], dtype=any_string_dtype)
    msg: str = 'expected a string object, not int'
    with pytest.raises(TypeError, match=msg):
        getattr(obj.str, method)(0)

@pytest.mark.parametrize('method, exp', [['index', [1, 1, 0]], ['rindex', [3, 1, 2]]])
def test_index_missing(any_string_dtype: str, method: str, exp: List[int]) -> None:
    ser: Series = Series(['abcb', 'ab', 'bcbe', np.nan], dtype=any_string_dtype)
    expected_dtype: Union[str, type] = np.float64 if is_object_or_nan_string_dtype(any_string_dtype) else 'Int64'
    result: Series = getattr(ser.str, method)('b')
    expected: Series = Series(exp + [np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

def test_pipe_failures(any_string_dtype: str) -> None:
    ser: Series = Series(['A|B|C'], dtype=any_string_dtype)
    result: Series = ser.str.split('|')
    expected: Series = Series([['A', 'B', 'C']], dtype=object)
    tm.assert_series_equal(result, expected)
    result = ser.str.replace('|', ' ', regex=False)
    expected = Series(['A B C'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    'start, stop, step, expected',
    [
        (2, 5, None, ['foo', 'bar', np.nan, 'baz']),
        (0, 3, -1, ['', '', np.nan, '']),
        (None, None, -1, ['owtoofaa', 'owtrabaa', np.nan, 'xuqzabaa']),
        (None, 2, -1, ['owtoo', 'owtra', np.nan, 'xuqza']),
        (3, 10, 2, ['oto', 'ato', np.nan, 'aqx']),
        (3, 0, -1, ['ofa', 'aba', np.nan, 'aba'])
    ]
)
def test_slice(start: Optional[int], stop: Optional[int], step: Optional[int],
               expected: List[Any], any_string_dtype: str) -> None:
    ser: Series = Series(['aafootwo', 'aabartwo', np.nan, 'aabazqux'], dtype=any_string_dtype)
    result: Series = ser.str.slice(start, stop, step)
    expected_series: Series = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected_series)

@pytest.mark.parametrize(
    'start, stop, step, expected',
    [
        (2, 5, None, ['foo', np.nan, 'bar', np.nan, np.nan, None, np.nan, np.nan]),
        (4, 1, -1, ['oof', np.nan, 'rab', np.nan, np.nan, None, np.nan, np.nan])
    ]
)
def test_slice_mixed_object(start: Optional[int], stop: Optional[int], step: Optional[int],
                            expected: List[Any]) -> None:
    ser: Series = Series(['aafootwo', np.nan, 'aabartwo', True, datetime.today(), None, 1, 2.0])
    result: Series = ser.str.slice(start, stop, step)
    expected_series: Series = Series(expected, dtype=object)
    tm.assert_series_equal(result, expected_series)

@pytest.mark.parametrize(
    'start,stop,repl,expected',
    [
        (2, 3, None, ['shrt', 'a it longer', 'evnlongerthanthat', '', np.nan]),
        (2, 3, 'z', ['shzrt', 'a zit longer', 'evznlongerthanthat', 'z', np.nan]),
        (2, 2, 'z', ['shzort', 'a zbit longer', 'evzenlongerthanthat', 'z', np.nan]),
        (2, 1, 'z', ['shzort', 'a zbit longer', 'evzenlongerthanthat', 'z', np.nan]),
        (-1, None, 'z', ['shorz', 'a bit longez', 'evenlongerthanthaz', 'z', np.nan]),
        (None, -2, 'z', ['zrt', 'zer', 'zat', 'z', np.nan]),
        (6, 8, 'z', ['shortz', 'a bit znger', 'evenlozerthanthat', 'z', np.nan]),
        (-10, 3, 'z', ['zrt', 'a zit longer', 'evenlongzerthanthat', 'z', np.nan])
    ]
)
def test_slice_replace(start: Optional[int], stop: Optional[int], repl: Optional[str],
                       expected: List[Any], any_string_dtype: str) -> None:
    ser: Series = Series(['short', 'a bit longer', 'evenlongerthanthat', '', np.nan], dtype=any_string_dtype)
    expected_series: Series = Series(expected, dtype=any_string_dtype)
    result: Series = ser.str.slice_replace(start, stop, repl)
    tm.assert_series_equal(result, expected_series)

@pytest.mark.parametrize(
    'method, exp',
    [
        ('strip', ['aa', 'bb', np.nan, 'cc']),
        ('lstrip', ['aa   ', 'bb \n', np.nan, 'cc  ']),
        ('rstrip', ['  aa', ' bb', np.nan, 'cc'])
    ]
)
def test_strip_lstrip_rstrip(any_string_dtype: str, method: str, exp: List[Any]) -> None:
    ser: Series = Series(['  aa   ', ' bb \n', np.nan, 'cc  '], dtype=any_string_dtype)
    result: Series = getattr(ser.str, method)()
    expected: Series = Series(exp, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    'method, exp',
    [
        ('strip', ['aa', np.nan, 'bb']),
        ('lstrip', ['aa  ', np.nan, 'bb \t\n']),
        ('rstrip', ['  aa', np.nan, ' bb'])
    ]
)
def test_strip_lstrip_rstrip_mixed_object(method: str, exp: List[Any]) -> None:
    ser: Series = Series(['  aa  ', np.nan, ' bb \t\n', True, datetime.today(), None, 1, 2.0])
    result: Series = getattr(ser.str, method)()
    expected: Series = Series(exp + [np.nan, np.nan, None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    'method, exp',
    [
        ('strip', ['ABC', ' BNSD', 'LDFJH ']),
        ('lstrip', ['ABCxx', ' BNSD', 'LDFJH xx']),
        ('rstrip', ['xxABC', 'xx BNSD', 'LDFJH '])
    ]
)
def test_strip_lstrip_rstrip_args(any_string_dtype: str, method: str, exp: List[Any]) -> None:
    ser: Series = Series(['xxABCxx', 'xx BNSD', 'LDFJH xx'], dtype=any_string_dtype)
    result: Series = getattr(ser.str, method)('x')
    expected: Series = Series(exp, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('prefix, expected', [('a', ['b', ' b c', 'bc']), ('ab', ['', 'a b c', 'bc'])])
def test_removeprefix(any_string_dtype: str, prefix: str, expected: List[str]) -> None:
    ser: Series = Series(['ab', 'a b c', 'bc'], dtype=any_string_dtype)
    result: Series = ser.str.removeprefix(prefix)
    ser_expected: Series = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, ser_expected)

@pytest.mark.parametrize('suffix, expected', [('c', ['ab', 'a b ', 'b']), ('bc', ['ab', 'a b c', ''])])
def test_removesuffix(any_string_dtype: str, suffix: str, expected: List[str]) -> None:
    ser: Series = Series(['ab', 'a b c', 'bc'], dtype=any_string_dtype)
    result: Series = ser.str.removesuffix(suffix)
    ser_expected: Series = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, ser_expected)

def test_string_slice_get_syntax(any_string_dtype: str) -> None:
    ser: Series = Series(['YYY', 'B', 'C', 'YYYYYYbYYY', 'BYYYcYYY', np.nan, 'CYYYBYYY', 'dog', 'cYYYt'], dtype=any_string_dtype)
    result: Series = ser.str[0]
    expected: Series = ser.str.get(0)
    tm.assert_series_equal(result, expected)
    result = ser.str[:3]
    expected = ser.str.slice(stop=3)
    tm.assert_series_equal(result, expected)
    result = ser.str[2::-1]
    expected = ser.str.slice(start=2, step=-1)
    tm.assert_series_equal(result, expected)

def test_string_slice_out_of_bounds_nested() -> None:
    ser: Series = Series([(1, 2), (1,), (3, 4, 5)])
    result: Series = ser.str[1]
    expected: Series = Series([2, np.nan, 4])
    tm.assert_series_equal(result, expected)

def test_string_slice_out_of_bounds(any_string_dtype: str) -> None:
    ser: Series = Series(['foo', 'b', 'ba'], dtype=any_string_dtype)
    result: Series = ser.str[1]
    expected: Series = Series(['o', np.nan, 'a'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

def test_encode_decode(any_string_dtype: str) -> None:
    ser: Series = Series(['a', 'b', 'aä'], dtype=any_string_dtype).str.encode('utf-8')
    result: Series = ser.str.decode('utf-8')
    expected: Series = Series(['a', 'b', 'aä'], dtype='str')
    tm.assert_series_equal(result, expected)

def test_encode_errors_kwarg(any_string_dtype: str) -> None:
    ser: Series = Series(['a', 'b', 'a\x9d'], dtype=any_string_dtype)
    msg: str = "'charmap' codec can't encode character '\\\\x9d' in position 1: character maps to <undefined>"
    with pytest.raises(UnicodeEncodeError, match=msg):
        ser.str.encode('cp1252')
    result: Series = ser.str.encode('cp1252', 'ignore')
    expected: Series = ser.map(lambda x: x.encode('cp1252', 'ignore'))
    tm.assert_series_equal(result, expected)

def test_decode_errors_kwarg() -> None:
    ser: Series = Series([b'a', b'b', b'a\x9d'])
    msg: str = "'charmap' codec can't decode byte 0x9d in position 1: character maps to <undefined>"
    with pytest.raises(UnicodeDecodeError, match=msg):
        ser.str.decode('cp1252')
    result: Series = ser.str.decode('cp1252', 'ignore')
    expected: Series = ser.map(lambda x: x.decode('cp1252', 'ignore')).astype('str')
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize(
    'form, expected',
    [
        ('NFKC', ['ABC', 'ABC', '123', np.nan, 'アイエ']),
        ('NFC', ['ABC', 'ＡＢＣ', '１２３', np.nan, 'ｱｲｴ'])
    ]
)
def test_normalize(form: str, expected: List[Optional[str]], any_string_dtype: str) -> None:
    ser: Series = Series(['ABC', 'ＡＢＣ', '１２３', np.nan, 'ｱｲｴ'], index=['a', 'b', 'c', 'd', 'e'], dtype=any_string_dtype)
    expected_series: Series = Series(expected, index=['a', 'b', 'c', 'd', 'e'], dtype=any_string_dtype)
    result: Series = ser.str.normalize(form)
    tm.assert_series_equal(result, expected_series)

def test_normalize_bad_arg_raises(any_string_dtype: str) -> None:
    ser: Series = Series(['ABC', 'ＡＢＣ', '１２３', np.nan, 'ｱｲｴ'], index=['a', 'b', 'c', 'd', 'e'], dtype=any_string_dtype)
    with pytest.raises(ValueError, match='invalid normalization form'):
        ser.str.normalize('xxx')

def test_normalize_index() -> None:
    idx: Index = Index(['ＡＢＣ', '１２３', 'ｱｲｴ'])
    expected: Index = Index(['ABC', '123', 'アイエ'])
    result: Index = idx.str.normalize('NFKC')
    tm.assert_index_equal(result, expected)

@pytest.mark.parametrize(
    'values,inferred_type',
    [
        (['a', 'b'], 'string'),
        (['a', 'b', 1], 'mixed-integer'),
        (['a', 'b', 1.3], 'mixed'),
        (['a', 'b', 1.3, 1], 'mixed-integer'),
        (['aa', datetime(2011, 1, 1)], 'mixed')
    ]
)
def test_index_str_accessor_visibility(values: List[Any], inferred_type: str, index_or_series: Callable[[List[Any]], Union[Series, Index]]) -> None:
    obj: Union[Series, Index] = index_or_series(values)
    if isinstance(obj, Index):
        assert obj.inferred_type == inferred_type  # type: ignore
    assert isinstance(obj.str, StringMethods)

@pytest.mark.parametrize(
    'values,inferred_type',
    [
        ([1, np.nan], 'floating'),
        ([datetime(2011, 1, 1)], 'datetime64'),
        ([timedelta(1)], 'timedelta64')
    ]
)
def test_index_str_accessor_non_string_values_raises(values: List[Any], inferred_type: str, index_or_series: Callable[[List[Any]], Union[Series, Index]]) -> None:
    obj: Union[Series, Index] = index_or_series(values)
    if isinstance(obj, Index):
        assert obj.inferred_type == inferred_type  # type: ignore
    msg: str = 'Can only use .str accessor with string values'
    with pytest.raises(AttributeError, match=msg):
        obj.str

def test_index_str_accessor_multiindex_raises() -> None:
    idx: MultiIndex = MultiIndex.from_tuples([('a', 'b'), ('a', 'b')])
    assert idx.inferred_type == 'mixed'
    msg: str = 'Can only use .str accessor with Index, not MultiIndex'
    with pytest.raises(AttributeError, match=msg):
        idx.str

def test_str_accessor_no_new_attributes(any_string_dtype: str) -> None:
    ser: Series = Series(list('aabbcde'), dtype=any_string_dtype)
    with pytest.raises(AttributeError, match='You cannot add any new attribute'):
        ser.str.xlabel = 'a'

def test_cat_on_bytes_raises() -> None:
    lhs: Series = Series(np.array(list('abc'), 'S1').astype(object))
    rhs: Series = Series(np.array(list('def'), 'S1').astype(object))
    msg: str = "Cannot use .str.cat with values of inferred dtype 'bytes'"
    with pytest.raises(TypeError, match=msg):
        lhs.str.cat(rhs)

def test_str_accessor_in_apply_func() -> None:
    df: DataFrame = DataFrame(list(zip('abc', 'def')))
    expected: Series = Series(['A/D', 'B/E', 'C/F'])
    result: Series = df.apply(lambda f: '/'.join(f.str.upper()), axis=1)
    tm.assert_series_equal(result, expected)

def test_zfill() -> None:
    value: Series = Series(['-1', '1', '1000', 10, np.nan])
    expected: Series = Series(['-01', '001', '1000', np.nan, np.nan], dtype=object)
    tm.assert_series_equal(value.str.zfill(3), expected)
    value = Series(['-2', '+5'])
    expected = Series(['-0002', '+0005'])
    tm.assert_series_equal(value.str.zfill(5), expected)

def test_zfill_with_non_integer_argument() -> None:
    value: Series = Series(['-2', '+5'])
    wid: Any = 'a'
    msg: str = f'width must be of integer type, not {type(wid).__name__}'
    with pytest.raises(TypeError, match=msg):
        value.str.zfill(wid)

def test_zfill_with_leading_sign() -> None:
    value: Series = Series(['-cat', '-1', '+dog'])
    expected: Series = Series(['-0cat', '-0001', '+0dog'])
    tm.assert_series_equal(value.str.zfill(5), expected)

def test_get_with_dict_label() -> None:
    s: Series = Series([{'name': 'Hello', 'value': 'World'}, {'name': 'Goodbye', 'value': 'Planet'}, {'value': 'Sea'}])
    result: Series = s.str.get('name')
    expected: Series = Series(['Hello', 'Goodbye', None], dtype=object)
    tm.assert_series_equal(result, expected)
    result = s.str.get('value')
    expected = Series(['World', 'Planet', 'Sea'], dtype=object)
    tm.assert_series_equal(result, expected)

def test_series_str_decode() -> None:
    result: Series = Series([b'x', b'y']).str.decode(encoding='UTF-8', errors='strict')
    expected: Series = Series(['x', 'y'], dtype='str')
    tm.assert_series_equal(result, expected)
