from datetime import datetime, timedelta
import numpy as np
import pytest
from pandas import DataFrame, Index, MultiIndex, Series
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import is_object_or_nan_string_dtype

@pytest.mark.parametrize('pattern', [0, True, Series(['foo', 'bar'])])
def test_startswith_endswith_non_str_patterns(pattern: str):
    ser: Series = Series(['foo', 'bar'])
    msg: str = f'expected a string or tuple, not {type(pattern).__name__}'
    with pytest.raises(TypeError, match=msg):
        ser.str.startswith(pattern)
    with pytest.raises(TypeError, match=msg):
        ser.str.endswith(pattern)

def test_iter_raises():
    ser: Series = Series(['foo', 'bar'])
    with pytest.raises(TypeError, match="'StringMethods' object is not iterable"):
        iter(ser.str)

def test_count(any_string_dtype: str):
    ser: Series = Series(['foo', 'foofoo', np.nan, 'foooofooofommmfoo'], dtype=any_string_dtype)
    result: Series = ser.str.count('f[o]+')
    expected_dtype: str = np.float64 if is_object_or_nan_string_dtype(any_string_dtype) else 'Int64'
    expected: Series = Series([1, 2, np.nan, 4], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

def test_count_mixed_object():
    ser: Series = Series(['a', np.nan, 'b', True, datetime.today(), 'foo', None, 1, 2.0], dtype=object)
    result: Series = ser.str.count('a')
    expected: Series = Series([1, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

def test_repeat(any_string_dtype: str):
    ser: Series = Series(['a', 'b', np.nan, 'c', np.nan, 'd'], dtype=any_string_dtype)
    result: Series = ser.str.repeat(3)
    expected: Series = Series(['aaa', 'bbb', np.nan, 'ccc', np.nan, 'ddd'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
    result: Series = ser.str.repeat([1, 2, 3, 4, 5, 6])
    expected: Series = Series(['a', 'bb', np.nan, 'cccc', np.nan, 'dddddd'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

def test_repeat_mixed_object():
    ser: Series = Series(['a', np.nan, 'b', True, datetime.today(), 'foo', None, 1, 2.0])
    result: Series = ser.str.repeat(3)
    expected: Series = Series(['aaa', np.nan, 'bbb', np.nan, np.nan, 'foofoofoo', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('arg, repeat', [[None, 4], ['b', None]])
def test_repeat_with_null(any_string_dtype: str, arg: str, repeat: int):
    ser: Series = Series(['a', arg], dtype=any_string_dtype)
    result: Series = ser.str.repeat([3, repeat])
    expected: Series = Series(['aaa', None], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

def test_empty_str_methods(any_string_dtype: str):
    empty_str: Series = empty = Series(dtype=any_string_dtype)
    empty_inferred_str: Series = Series(dtype='str')
    if is_object_or_nan_string_dtype(any_string_dtype):
        empty_int: Series = Series(dtype='int64')
        empty_bool: Series = Series(dtype=bool)
    else:
        empty_int: Series = Series(dtype='Int64')
        empty_bool: Series = Series(dtype='boolean')
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
    table = str.maketrans('a', 'b')
    tm.assert_series_equal(empty_str, empty.str.translate(table))

@pytest.mark.parametrize('method, expected', [('isascii', [True, True, True, True, True, True, True, True, True, True]), ('isalnum', [True, True, True, True, True, False, True, True, False, False]), ('isalpha', [True, True, True, False, False, False, True, False, False, False]), ('isdigit', [False, False, False, True, False, False, False, True, False, False]), ('isnumeric', [False, False, False, True, False, False, False, True, False, False]), ('isspace', [False, False, False, False, False, False, False, False, False, True]), ('islower', [False, True, False, False, False, False, False, False, False, False]), ('isupper', [True, False, False, False, True, False, True, False, False, False]), ('istitle', [True, False, True, False, True, False, False, False, False, False])])
def test_ismethods(method: str, expected: List[bool], any_string_dtype: str):
    ser: Series = Series(['A', 'b', 'Xy', '4', '3A', '', 'TT', '55', '-', '  '], dtype=any_string_dtype)
    expected_dtype: str = 'bool' if is_object_or_nan_string_dtype(any_string_dtype) else 'boolean'
    expected: Series = Series(expected, dtype=expected_dtype)
    result: Series = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)
    expected_stdlib: List[bool] = [getattr(item, method)() for item in ser]
    assert list(result) == expected_stdlib
    ser.iloc[[1, 2, 3, 4]] = np.nan
    result: Series = getattr(ser.str, method)()
    if ser.dtype == 'object':
        expected: Series = expected.astype(object)
        expected.iloc[[1, 2, 3, 4]] = np.nan
    elif ser.dtype == 'str':
        expected.iloc[[1, 2, 3, 4]] = False
    else:
        expected.iloc[[1, 2, 3, 4]] = np.nan

@pytest.mark.parametrize('method, expected', [('isnumeric', [False, True, True, False, True, True, False]), ('isdecimal', [False, True, False, False, False, True, False])])
def test_isnumeric_unicode(method: str, expected: List[bool], any_string_dtype: str):
    ser: Series = Series(['A', '3', '¼', '★', '፸', '３', 'four'], dtype=any_string_dtype)
    expected_dtype: str = 'bool' if is_object_or_nan_string_dtype(any_string_dtype) else 'boolean'
    expected: Series = Series(expected, dtype=expected_dtype)
    result: Series = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)
    expected: List[bool] = [getattr(item, method)() for item in ser]
    assert list(result) == expected

@pytest.mark.parametrize('method, expected', [('isnumeric', [False, np.nan, True, False, np.nan, True, False]), ('isdecimal', [False, np.nan, False, False, np.nan, True, False])])
def test_isnumeric_unicode_missing(method: str, expected: List[bool], any_string_dtype: str):
    values: List[Union[str, float]] = ['A', np.nan, '¼', '★', np.nan, '３', 'four']
    ser: Series = Series(values, dtype=any_string_dtype)
    if any_string_dtype == 'str':
        expected: Series = Series(expected, dtype=object).fillna(False).astype(bool)
    else:
        expected_dtype: str = 'object' if is_object_or_nan_string_dtype(any_string_dtype) else 'boolean'
        expected: Series = Series(expected, dtype=expected_dtype)
    result: Series = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

def test_spilt_join_roundtrip(any_string_dtype: str):
    ser: Series = Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'], dtype=any_string_dtype)
    result: Series = ser.str.split('_').str.join('_')
    expected: Series = ser.astype(object)
    tm.assert_series_equal(result, expected)

def test_spilt_join_roundtrip_mixed_object():
    ser: Series = Series(['a_b', np.nan, 'asdf_cas_asdf', True, datetime.today(), 'foo', None, 1, 2.0])
    result: Series = ser.str.split('_').str.join('_')
    expected: Series = Series(['a_b', np.nan, 'asdf_cas_asdf', np.nan, np.nan, 'foo', None, np.nan, np.nan], dtype=object)
    tm.assert_series_equal(result, expected)

def test_len(any_string_dtype: str):
    ser: Series = Series(['foo', 'fooo', 'fooooo', np.nan, 'fooooooo', 'foo\n', 'あ'], dtype=any_string_dtype)
    result: Series = ser.str.len()
    expected_dtype: str = 'float64' if is_object_or_nan_string_dtype(any_string_dtype) else 'Int64'
    expected: Series = Series([3, 4, 6, np.nan, 8, 4, 1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

def test_len_mixed():
    ser: Series = Series(['a_b', np.nan, 'asdf_cas_asdf', True, datetime.today(), 'foo', None, 1, 2.0])
    result: Series = ser.str.len()
    expected: Series = Series([3, np.nan, 13, np.nan, np.nan, 3, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

@pytest.mark.parametrize('method,sub,start,end,expected', [('index', 'EF', None, None, [4, 3, 1, 0]), ('rindex', 'EF', None, None, [4, 5, 7, 4]), ('index', 'EF', 3, None, [4, 3, 7, 4]), ('rindex', 'EF', 3, None, [4, 5, 7, 4]), ('index', 'E', 4, 8, [4, 5, 7, 4]), ('rindex', 'E', 0, 5, [4, 3, 1, 4])])
def test_index(method: str, sub: str, start: Optional[int], end: Optional[int], index_or_series: Union[Index, Series], any_string_dtype: str, expected: List[int]):
    obj: Union[Index, Series] = index_or_series(['ABCDEFG', 'BCDEFEF', 'DEFGHIJEF', 'EFGHEF'], dtype=any_string_dtype)
    expected_dtype: Union[str, np.int64] = np.int64 if is_object_or_nan_string_dtype(any_string_dtype) else 'Int64'
    expected: Union[Index, Series] = index_or_series(expected, dtype=expected_dtype)
    result: Union[Index, Series] = getattr(obj.str, method)(sub, start, end)
    if index_or_series is Series:
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)
    expected: List[int] = [getattr(item, method)(sub, start, end) for item in obj]
    assert list(result) == expected

def test_index_not_found_raises(index_or_series: Union[Index, Series], any_string_dtype: str):
    obj: Union[Index, Series] = index_or_series(['ABCDEFG', 'BCDEFEF', 'DEFGHIJEF', 'EFGHEF'], dtype=any_string_dtype)
    with pytest.raises(ValueError, match='substring not found'):
        obj.str.index('DE')

@pytest.mark.parametrize('method', ['index', 'rindex'])
def test_index_wrong_type_raises(index_or_series: Union[Index, Series], any_string_dtype: str, method: str):
    obj: Union[Index, Series] = index_or_series([], dtype=any_string_dtype)
    msg: str = 'expected a string object, not int'
    with pytest.raises(TypeError, match=msg):
        getattr(obj.str, method)(0)

@pytest.mark.parametrize('method, exp', [['index', [1, 1, 0]], ['rindex', [3, 1, 2]]])
def test_index_missing(any_string_dtype: str, method: str, exp: List[int]):
    ser: Series = Series(['abcb', 'ab', 'bcbe', np.nan], dtype=any_string_dtype)
    expected_dtype: Union[str, np.float64] = np.float64 if is_object_or_nan_string_dtype(any_string_dtype) else 'Int64'
    result: Series = getattr(ser.str, method)('b')
    expected: Series = Series(exp + [np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

def test_pipe_failures(any_string_dtype: str):
    ser: Series = Series(['A|B|C'], dtype=any_string_dtype)
    result: Series = ser.str.split('|')
    expected: Series = Series([['A', 'B', 'C']], dtype=object)
    tm.assert_series_equal(result, expected)
    result: Series = ser.str.replace('|', ' ', regex=False)
    expected: Series = Series(['A B C'], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

def test_str_accessor_in_apply_func():
    df: DataFrame = DataFrame(zip('abc', 'def'))
    expected: Series = Series(['A/D', 'B/E', 'C/F'])
    result: Series = df.apply(lambda f: '/'.join(f.str.upper()), axis=1)
    tm.assert_series_equal(result, expected)

def test_zfill():
    value: Series = Series(['-1', '1', '1000', 10, np.nan])
    expected: Series = Series(['-01', '001', '1000', np.nan, np.nan], dtype=object)
    tm.assert_series_equal(value.str.zfill(3), expected)
    value: Series = Series(['-2', '+5'])
    expected: Series = Series(['-0002', '+0005'])
    tm.assert_series_equal(value.str.zfill(5), expected)

def test_zfill_with_non_integer_argument():
    value: Series = Series(['-2', '+5'])
    wid: str = 'a'
    msg: str = f'width must be of integer type, not {type(wid).__name__}'
    with pytest.raises(TypeError, match=msg):
        value.str.zfill(wid)

def test_zfill_with_leading_sign():
    value: Series = Series(['-cat', '-1', '+dog'])
    expected: Series = Series(['-0cat', '-0001', '+0dog'])
    tm.assert_series_equal(value.str.zfill(5), expected)

def test_get_with_dict_label():
    s: Series = Series([{'name': '