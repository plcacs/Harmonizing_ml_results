#!/usr/bin/env python3
import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import dateutil
import numpy as np
import pytest
import pandas._testing as tm
import pandas._libs.json as ujson
from pandas import DataFrame, DatetimeIndex, Index, NaT, PeriodIndex, Series, Timedelta, Timestamp, date_range
from pandas.compat import IS64


def _clean_dict(d: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Sanitize dictionary for JSON by converting all keys to strings.

    Parameters
    ----------
    d : dict
        The dictionary to convert.

    Returns
    -------
    cleaned_dict : dict
    """
    return {str(k): v for k, v in d.items()}


@pytest.fixture(params=[None, 'split', 'records', 'values', 'index'])
def orient(request: Any) -> Optional[str]:
    return request.param


class TestUltraJSONTests:
    @pytest.mark.skipif(not IS64, reason='not compliant on 32-bit, xref #15865')
    @pytest.mark.parametrize('value, double_precision', [
        ('1337.1337', 15), ('0.95', 1), ('0.94', 1), ('1.95', 1),
        ('-1.95', 1), ('0.995', 2), ('0.9995', 3), ('0.99999999999999944', 15)
    ])
    def test_encode_decimal(self, value: str, double_precision: int) -> None:
        sut: decimal.Decimal = decimal.Decimal(value)
        encoded: str = ujson.ujson_dumps(sut, double_precision=double_precision)
        decoded: Any = ujson.ujson_loads(encoded)
        assert decoded == value

    @pytest.mark.parametrize('ensure_ascii', [True, False])
    def test_encode_string_conversion(self, ensure_ascii: bool) -> None:
        string_input: str = 'A string \\ / \x08 \x0c \n \r \t </script> &'
        not_html_encoded: str = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
        html_encoded: str = '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'

        def helper(expected_output: str, **encode_kwargs: Any) -> None:
            output: str = ujson.ujson_dumps(string_input, ensure_ascii=ensure_ascii, **encode_kwargs)
            assert output == expected_output
            assert string_input == json.loads(output)
            assert string_input == ujson.ujson_loads(output)

        helper(not_html_encoded)
        helper(not_html_encoded, encode_html_chars=False)
        helper(html_encoded, encode_html_chars=True)

    @pytest.mark.parametrize('long_number', [-4342969734183514, -12345678901234.568, -528656961.4399388])
    def test_double_long_numbers(self, long_number: Union[int, float]) -> None:
        sut: Dict[str, Union[int, float]] = {'a': long_number}
        encoded: str = ujson.ujson_dumps(sut, double_precision=15)
        decoded: Any = ujson.ujson_loads(encoded)
        assert sut == decoded

    def test_encode_non_c_locale(self) -> None:
        lc_category = locale.LC_NUMERIC
        for new_locale in ('it_IT.UTF-8', 'Italian_Italy'):
            if tm.can_set_locale(new_locale, lc_category):
                with tm.set_locale(new_locale, lc_category):
                    assert ujson.ujson_loads(ujson.ujson_dumps(4.78e+60)) == 4.78e+60
                    assert ujson.ujson_loads('4.78', precise_float=True) == 4.78
                break

    def test_decimal_decode_test_precise(self) -> None:
        sut: Dict[str, float] = {'a': 4.56}
        encoded: str = ujson.ujson_dumps(sut)
        decoded: Any = ujson.ujson_loads(encoded, precise_float=True)
        assert sut == decoded

    def test_encode_double_tiny_exponential(self) -> None:
        num: float = 1e-40
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = 1e-100
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-45
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-145
        assert np.allclose(num, ujson.ujson_loads(ujson.ujson_dumps(num)))

    @pytest.mark.parametrize('unicode_key', ['key1', 'بن'])
    def test_encode_dict_with_unicode_keys(self, unicode_key: str) -> None:
        unicode_dict: Dict[str, str] = {unicode_key: 'value1'}
        assert unicode_dict == ujson.ujson_loads(ujson.ujson_dumps(unicode_dict))

    @pytest.mark.parametrize('double_input', [math.pi, -math.pi])
    def test_encode_double_conversion(self, double_input: float) -> None:
        output: str = ujson.ujson_dumps(double_input)
        assert round(double_input, 5) == round(json.loads(output), 5)
        assert round(double_input, 5) == round(ujson.ujson_loads(output), 5)

    def test_encode_with_decimal(self) -> None:
        decimal_input: float = 1.0
        output: str = ujson.ujson_dumps(decimal_input)
        assert output == '1.0'

    def test_encode_array_of_nested_arrays(self) -> None:
        nested_input: List[Any] = [[[[]]]] * 20
        output: str = ujson.ujson_dumps(nested_input)
        assert nested_input == json.loads(output)
        assert nested_input == ujson.ujson_loads(output)

    def test_encode_array_of_doubles(self) -> None:
        doubles_input: List[float] = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
        output: str = ujson.ujson_dumps(doubles_input)
        assert doubles_input == json.loads(output)
        assert doubles_input == ujson.ujson_loads(output)

    def test_double_precision(self) -> None:
        double_input: float = 30.012345678901234
        output: str = ujson.ujson_dumps(double_input, double_precision=15)
        assert double_input == json.loads(output)
        assert double_input == ujson.ujson_loads(output)
        for double_precision in (3, 9):
            output = ujson.ujson_dumps(double_input, double_precision=double_precision)
            rounded_input: float = round(double_input, double_precision)
            assert rounded_input == json.loads(output)
            assert rounded_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('invalid_val', [20, -1, '9', None])
    def test_invalid_double_precision(self, invalid_val: Union[int, None, str]) -> None:
        double_input: float = 30.123456789012344
        expected_exception: Any = ValueError if isinstance(invalid_val, int) else TypeError
        msg: str = ("Invalid value '.*' for option 'double_precision', max is '15'|an integer is required "
                    "\\(got type |object cannot be interpreted as an integer")
        with pytest.raises(expected_exception, match=msg):
            ujson.ujson_dumps(double_input, double_precision=invalid_val)

    def test_encode_string_conversion2(self) -> None:
        string_input: str = 'A string \\ / \x08 \x0c \n \r \t'
        output: str = ujson.ujson_dumps(string_input)
        assert string_input == json.loads(output)
        assert string_input == ujson.ujson_loads(output)
        assert output == '"A string \\\\ \\/ \\b \\f \\n \\r \\t"'

    @pytest.mark.parametrize('unicode_input', ['Räksmörgås اسامة بن محمد بن عوض بن لادن', 'æ\x97¥Ñ\x88'])
    def test_encode_unicode_conversion(self, unicode_input: str) -> None:
        enc: str = ujson.ujson_dumps(unicode_input)
        dec: Any = ujson.ujson_loads(enc)
        assert enc == json.dumps(unicode_input)
        assert dec == json.loads(enc)

    def test_encode_control_escaping(self) -> None:
        escaped_input: str = '\x19'
        enc: str = ujson.ujson_dumps(escaped_input)
        dec: Any = ujson.ujson_loads(enc)
        assert escaped_input == dec
        assert enc == json.dumps(escaped_input)

    def test_encode_unicode_surrogate_pair(self) -> None:
        surrogate_input: str = 'ð\x90\x8d\x86'
        enc: str = ujson.ujson_dumps(surrogate_input)
        dec: Any = ujson.ujson_loads(enc)
        assert enc == json.dumps(surrogate_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8(self) -> None:
        four_bytes_input: str = 'ð\x91\x80°TRAILINGNORMAL'
        enc: str = ujson.ujson_dumps(four_bytes_input)
        dec: Any = ujson.ujson_loads(enc)
        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8highest(self) -> None:
        four_bytes_input: str = 'ó¿¿¿TRAILINGNORMAL'
        enc: str = ujson.ujson_dumps(four_bytes_input)
        dec: Any = ujson.ujson_loads(enc)
        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_error(self) -> None:
        string: str = "'\udac0'"
        msg: str = ("'utf-8' codec can't encode character '\\\\udac0' in position 1: surrogates not allowed")
        with pytest.raises(UnicodeEncodeError, match=msg):
            ujson.ujson_dumps([string])

    def test_encode_array_in_array(self) -> None:
        arr_in_arr_input: List[Any] = [[[[]]]]
        output: str = ujson.ujson_dumps(arr_in_arr_input)
        assert arr_in_arr_input == json.loads(output)
        assert output == json.dumps(arr_in_arr_input)
        assert arr_in_arr_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('num_input', [31337, -31337, -9223372036854775808])
    def test_encode_num_conversion(self, num_input: int) -> None:
        output: str = ujson.ujson_dumps(num_input)
        assert num_input == json.loads(output)
        assert output == json.dumps(num_input)
        assert num_input == ujson.ujson_loads(output)

    def test_encode_list_conversion(self) -> None:
        list_input: List[int] = [1, 2, 3, 4]
        output: str = ujson.ujson_dumps(list_input)
        assert list_input == json.loads(output)
        assert list_input == ujson.ujson_loads(output)

    def test_encode_dict_conversion(self) -> None:
        dict_input: Dict[str, int] = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4}
        output: str = ujson.ujson_dumps(dict_input)
        assert dict_input == json.loads(output)
        assert dict_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('builtin_value', [None, True, False])
    def test_encode_builtin_values_conversion(self, builtin_value: Optional[bool]) -> None:
        output: str = ujson.ujson_dumps(builtin_value)
        assert builtin_value == json.loads(output)
        assert output == json.dumps(builtin_value)
        assert builtin_value == ujson.ujson_loads(output)

    def test_encode_datetime_conversion(self) -> None:
        datetime_input: datetime.datetime = datetime.datetime.fromtimestamp(time.time())
        output: str = ujson.ujson_dumps(datetime_input, date_unit='s')
        expected: int = calendar.timegm(datetime_input.utctimetuple())
        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    def test_encode_date_conversion(self) -> None:
        date_input: datetime.date = datetime.date.fromtimestamp(time.time())
        output: str = ujson.ujson_dumps(date_input, date_unit='s')
        tup: Tuple[int, int, int, int, int, int] = (date_input.year, date_input.month, date_input.day, 0, 0, 0)
        expected: int = calendar.timegm(tup)
        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    @pytest.mark.parametrize('test', [datetime.time(), datetime.time(1, 2, 3), datetime.time(10, 12, 15, 343243)])
    def test_encode_time_conversion_basic(self, test: datetime.time) -> None:
        output: str = ujson.ujson_dumps(test)
        expected: str = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_pytz(self) -> None:
        pytz = pytest.importorskip('pytz')
        test: datetime.time = datetime.time(10, 12, 15, 343243, pytz.utc)
        output: str = ujson.ujson_dumps(test)
        expected: str = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_dateutil(self) -> None:
        test: datetime.time = datetime.time(10, 12, 15, 343243, dateutil.tz.tzutc())
        output: str = ujson.ujson_dumps(test)
        expected: str = f'"{test.isoformat()}"'
        assert expected == output

    @pytest.mark.parametrize('decoded_input', [NaT, np.datetime64('NaT'), np.nan, np.inf, -np.inf])
    def test_encode_as_null(self, decoded_input: Any) -> None:
        assert ujson.ujson_dumps(decoded_input) == 'null', 'Expected null'

    def test_datetime_units(self) -> None:
        val: datetime.datetime = datetime.datetime(2013, 8, 17, 21, 17, 12, 215504)
        stamp: Timestamp = Timestamp(val).as_unit('ns')
        roundtrip: Any = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='s'))
        assert roundtrip == stamp._value // 10 ** 9
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='ms'))
        assert roundtrip == stamp._value // 10 ** 6
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='us'))
        assert roundtrip == stamp._value // 10 ** 3
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='ns'))
        assert roundtrip == stamp._value
        msg: str = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_dumps(val, date_unit='foo')

    def test_encode_to_utf8(self) -> None:
        unencoded: str = 'æ\x97¥Ñ\x88'
        enc: str = ujson.ujson_dumps(unencoded, ensure_ascii=False)
        dec: Any = ujson.ujson_loads(enc)
        assert enc == json.dumps(unencoded, ensure_ascii=False)
        assert dec == json.loads(enc)

    def test_decode_from_unicode(self) -> None:
        unicode_input: str = '{"obj": 31337}'
        dec1: Any = ujson.ujson_loads(unicode_input)
        dec2: Any = ujson.ujson_loads(str(unicode_input))
        assert dec1 == dec2

    def test_encode_recursion_max(self) -> None:
        class O2:
            member: int = 0

        class O1:
            member: int = 0

        decoded_input: Any = O1()
        decoded_input.member = O2()
        decoded_input.member.member = decoded_input
        with pytest.raises(OverflowError, match='Maximum recursion level reached'):
            ujson.ujson_dumps(decoded_input)

    def test_decode_jibberish(self) -> None:
        jibberish: str = 'fdsa sda v9sa fdsa'
        msg: str = "Unexpected character found when decoding 'false'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(jibberish)

    @pytest.mark.parametrize('broken_json', ['[', '{', ']', '}'])
    def test_decode_broken_json(self, broken_json: str) -> None:
        msg: str = 'Expected object or value'
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize('too_big_char', ['[', '{'])
    def test_decode_depth_too_big(self, too_big_char: str) -> None:
        with pytest.raises(ValueError, match='Reached object decoding depth limit'):
            ujson.ujson_loads(too_big_char * (1024 * 1024))

    @pytest.mark.parametrize('bad_string', ['"TESTING', '"TESTING\\"', 'tru', 'fa', 'n'])
    def test_decode_bad_string(self, bad_string: str) -> None:
        msg: str = 'Unexpected character found when decoding|Unmatched \'\'"\' when when decoding \'string\''
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(bad_string)

    @pytest.mark.parametrize('broken_json, err_msg', [
        ('{{1337:""}}', "Key name of object must be 'string' when decoding 'object'"),
        ('{{"key":"}', 'Unmatched \'\'"\' when when decoding \'string\''),
        ('[[[true', 'Unexpected character found when decoding array value (2)')
    ])
    def test_decode_broken_json_leak(self, broken_json: str, err_msg: str) -> None:
        for _ in range(1000):
            with pytest.raises(ValueError, match=re.escape(err_msg)):
                ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize('invalid_dict', ['{{{{31337}}}}', '{{{{"key":}}}}', '{{{{"key"}}}}'])
    def test_decode_invalid_dict(self, invalid_dict: str) -> None:
        msg: str = ("Key name of object must be 'string' when decoding 'object'|No ':' found when decoding object value|"
                    "Expected object or value")
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_dict)

    @pytest.mark.parametrize('numeric_int_as_str', ['31337', '-31337'])
    def test_decode_numeric_int(self, numeric_int_as_str: str) -> None:
        assert int(numeric_int_as_str) == ujson.ujson_loads(numeric_int_as_str)

    def test_encode_null_character(self) -> None:
        wrapped_input: str = '31337 \x00 1337'
        output: str = ujson.ujson_dumps(wrapped_input)
        assert wrapped_input == json.loads(output)
        assert output == json.dumps(wrapped_input)
        assert wrapped_input == ujson.ujson_loads(output)
        alone_input: str = '\x00'
        output = ujson.ujson_dumps(alone_input)
        assert alone_input == json.loads(output)
        assert output == json.dumps(alone_input)
        assert '"  \\u0000\\r\\n "' == ujson.ujson_dumps('  \x00\r\n ')

    def test_decode_null_character(self) -> None:
        wrapped_input: str = '"31337 \\u0000 31337"'
        assert ujson.ujson_loads(wrapped_input) == json.loads(wrapped_input)

    def test_encode_list_long_conversion(self) -> None:
        long_input: List[int] = [9223372036854775807, 9223372036854775807, 9223372036854775807,
                                 9223372036854775807, 9223372036854775807, 9223372036854775807]
        output: str = ujson.ujson_dumps(long_input)
        assert long_input == json.loads(output)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('long_input', [9223372036854775807, 18446744073709551615])
    def test_encode_long_conversion(self, long_input: int) -> None:
        output: str = ujson.ujson_dumps(long_input)
        assert long_input == json.loads(output)
        assert output == json.dumps(long_input)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('bigNum', [2 ** 64, -2 ** 63 - 1])
    def test_dumps_ints_larger_than_maxsize(self, bigNum: int) -> None:
        encoding: str = ujson.ujson_dumps(bigNum)
        assert str(bigNum) == encoding
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            _ = ujson.ujson_loads(encoding) == bigNum

    @pytest.mark.parametrize('int_exp', ['1337E40', '1.337E40', '1337E+9', '1.337e+40', '1.337E-4'])
    def test_decode_numeric_int_exp(self, int_exp: str) -> None:
        assert ujson.ujson_loads(int_exp) == json.loads(int_exp)

    def test_loads_non_str_bytes_raises(self) -> None:
        msg: str = "a bytes-like object is required, not 'NoneType'"
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_loads(None)

    @pytest.mark.parametrize('val', [3590016419, 2 ** 31, 2 ** 32, 2 ** 32 - 1])
    def test_decode_number_with_32bit_sign_bit(self, val: int) -> None:
        doc: str = f'{{"id": {val}}}'
        assert ujson.ujson_loads(doc)['id'] == val

    def test_encode_big_escape(self) -> None:
        for _ in range(10):
            base: bytes = 'å'.encode()
            escape_input: bytes = base * (1024 * 1024 * 2)
            ujson.ujson_dumps(escape_input)

    def test_decode_big_escape(self) -> None:
        for _ in range(10):
            base: bytes = 'å'.encode()
            quote: bytes = b'"'
            escape_input: bytes = quote + base * (1024 * 1024 * 2) + quote
            ujson.ujson_loads(escape_input)

    def test_to_dict(self) -> None:
        d: Dict[str, int] = {'key': 31337}

        class DictTest:
            def toDict(self) -> Dict[str, int]:
                return d

        o: Any = DictTest()
        output: str = ujson.ujson_dumps(o)
        dec: Any = ujson.ujson_loads(output)
        assert dec == d

    def test_default_handler(self) -> None:
        class _TestObject:
            def __init__(self, val: Any) -> None:
                self.val = val

            @property
            def recursive_attr(self) -> Any:
                return _TestObject('recursive_attr')

            def __str__(self) -> str:
                return str(self.val)

        msg: str = 'Maximum recursion level reached'
        with pytest.raises(OverflowError, match=msg):
            ujson.ujson_dumps(_TestObject('foo'))
        assert '"foo"' == ujson.ujson_dumps(_TestObject('foo'), default_handler=str)

        def my_handler(_: Any) -> str:
            return 'foobar'
        assert '"foobar"' == ujson.ujson_dumps(_TestObject('foo'), default_handler=my_handler)

        def my_handler_raises(_: Any) -> Any:
            raise TypeError('I raise for anything')
        with pytest.raises(TypeError, match='I raise for anything'):
            ujson.ujson_dumps(_TestObject('foo'), default_handler=my_handler_raises)

        def my_int_handler(_: Any) -> int:
            return 42
        assert ujson.ujson_loads(ujson.ujson_dumps(_TestObject('foo'), default_handler=my_int_handler)) == 42

        def my_obj_handler(_: Any) -> datetime.datetime:
            return datetime.datetime(2013, 2, 3)
        assert ujson.ujson_loads(ujson.ujson_dumps(datetime.datetime(2013, 2, 3))) == ujson.ujson_loads(
            ujson.ujson_dumps(_TestObject('foo'), default_handler=my_obj_handler))
        obj_list: List[Any] = [_TestObject('foo'), _TestObject('bar')]
        assert json.loads(json.dumps(obj_list, default=str)) == ujson.ujson_loads(ujson.ujson_dumps(obj_list, default_handler=str))

    def test_encode_object(self) -> None:
        class _TestObject:
            def __init__(self, a: Any, b: Any, _c: Any, d: Any) -> None:
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self) -> int:
                return 5

        test_object = _TestObject(a=1, b=2, _c=3, d=4)
        assert ujson.ujson_loads(ujson.ujson_dumps(test_object)) == {'a': 1, 'b': 2, 'd': 4}

    def test_ujson__name__(self) -> None:
        assert ujson.__name__ == 'pandas._libs.json'


class TestNumpyJSONTests:
    @pytest.mark.parametrize('bool_input', [True, False])
    def test_bool(self, bool_input: bool) -> None:
        b: bool = bool(bool_input)
        assert ujson.ujson_loads(ujson.ujson_dumps(b)) == b

    def test_bool_array(self) -> None:
        bool_array: np.ndarray = np.array([True, False, True, True, False, True, False, False], dtype=bool)
        output: np.ndarray = np.array(ujson.ujson_loads(ujson.ujson_dumps(bool_array)), dtype=bool)
        tm.assert_numpy_array_equal(bool_array, output)

    def test_int(self, any_int_numpy_dtype: Any) -> None:
        klass = np.dtype(any_int_numpy_dtype).type
        num = klass(1)
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_int_array(self, any_int_numpy_dtype: Any) -> None:
        arr: np.ndarray = np.arange(100, dtype=int)
        arr_input: np.ndarray = arr.astype(any_int_numpy_dtype)
        arr_output: np.ndarray = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr_input)), dtype=any_int_numpy_dtype)
        tm.assert_numpy_array_equal(arr_input, arr_output)

    def test_int_max(self, any_int_numpy_dtype: Any) -> None:
        if any_int_numpy_dtype in ('int64', 'uint64') and (not IS64):
            pytest.skip('Cannot test 64-bit integer on 32-bit platform')
        klass = np.dtype(any_int_numpy_dtype).type
        if any_int_numpy_dtype == 'uint64':
            num = np.iinfo('int64').max
        else:
            num = np.iinfo(any_int_numpy_dtype).max
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float(self, float_numpy_dtype: Any) -> None:
        klass = np.dtype(float_numpy_dtype).type
        num = klass(256.2013)
        assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num

    def test_float_array(self, float_numpy_dtype: Any) -> None:
        arr: np.ndarray = np.arange(12.5, 185.72, 1.7322, dtype=float)
        float_input: np.ndarray = arr.astype(float_numpy_dtype)
        float_output: np.ndarray = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr, double_precision=15)), dtype=float_numpy_dtype)
        tm.assert_almost_equal(float_input, float_output)

    def test_float_max(self, float_numpy_dtype: Any) -> None:
        klass = np.dtype(float_numpy_dtype).type
        num = klass(np.finfo(float_numpy_dtype).max / 10)
        tm.assert_almost_equal(klass(ujson.ujson_loads(ujson.ujson_dumps(num, double_precision=15))), num)

    def test_array_basic(self) -> None:
        arr: np.ndarray = np.arange(96)
        arr = arr.reshape((2, 2, 2, 2, 3, 2))
        tm.assert_numpy_array_equal(np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr)

    @pytest.mark.parametrize('shape', [(10, 10), (5, 5, 4), (100, 1)])
    def test_array_reshaped(self, shape: Tuple[int, ...]) -> None:
        arr: np.ndarray = np.arange(100)
        arr = arr.reshape(shape)
        tm.assert_numpy_array_equal(np.array(ujson.ujson_loads(ujson.ujson_dumps(arr))), arr)

    def test_array_list(self) -> None:
        arr_list: List[Any] = ['a', [], {}, {}, [], 42, 97.8, ['a', 'b'], {'key': 'val'}]
        arr: np.ndarray = np.array(arr_list, dtype=object)
        result: np.ndarray = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=object)
        tm.assert_numpy_array_equal(result, arr)

    def test_array_float(self) -> None:
        dtype = np.float32
        arr: np.ndarray = np.arange(100.202, 200.202, 1, dtype=dtype)
        arr = arr.reshape((5, 5, 4))
        arr_out: np.ndarray = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=dtype)
        tm.assert_almost_equal(arr, arr_out)

    def test_0d_array(self) -> None:
        msg: str = re.escape('array(1) (numpy-scalar) is not JSON serializable at the moment')
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.array(1))

    def test_array_long_double(self) -> None:
        msg = re.compile('1234.5.* \\(numpy-scalar\\) is not JSON serializable at the moment')
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_dumps(np.longdouble(1234.5))


class TestPandasJSONTests:
    def test_dataframe(self, orient: Optional[str]) -> None:
        dtype = np.int64
        df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6]], index=['a', 'b'], columns=['x', 'y', 'z'], dtype=dtype)
        encode_kwargs: Dict[str, Any] = {} if orient is None else {'orient': orient}
        assert (df.dtypes == dtype).all()
        output: Any = ujson.ujson_loads(ujson.ujson_dumps(df, **encode_kwargs))
        assert (df.dtypes == dtype).all()
        if orient == 'split':
            dec: Dict[str, Any] = _clean_dict(output)
            output = DataFrame(**dec)
        else:
            output = DataFrame(output)
        if orient == 'values':
            df.columns = [0, 1, 2]
            df.index = [0, 1]
        elif orient == 'records':
            df.index = [0, 1]
        elif orient == 'index':
            df = df.transpose()
        assert (df.dtypes == dtype).all()
        tm.assert_frame_equal(output, df)

    def test_dataframe_nested(self, orient: Optional[str]) -> None:
        df: DataFrame = DataFrame([[1, 2, 3], [4, 5, 6]], index=['a', 'b'], columns=['x', 'y', 'z'])
        nested: Dict[str, Any] = {'df1': df, 'df2': df.copy()}
        kwargs: Dict[str, Any] = {} if orient is None else {'orient': orient}
        exp: Dict[str, Any] = {'df1': ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)),
                               'df2': ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs))}
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_series(self, orient: Optional[str]) -> None:
        dtype = np.int64
        s: Series = Series([10, 20, 30, 40, 50, 60], name='series', index=[6, 7, 8, 9, 10, 15], dtype=dtype).sort_values()
        assert s.dtype == dtype
        encode_kwargs: Dict[str, Any] = {} if orient is None else {'orient': orient}
        output: Any = ujson.ujson_loads(ujson.ujson_dumps(s, **encode_kwargs))
        assert s.dtype == dtype
        if orient == 'split':
            dec: Dict[str, Any] = _clean_dict(output)
            output = Series(**dec)
        else:
            output = Series(output)
        if orient in (None, 'index'):
            s.name = None
            output = output.sort_values()
            s.index = ['6', '7', '8', '9', '10', '15']
        elif orient in ('records', 'values'):
            s.name = None
            s.index = [0, 1, 2, 3, 4, 5]
        assert s.dtype == dtype
        tm.assert_series_equal(output, s)

    def test_series_nested(self, orient: Optional[str]) -> None:
        s: Series = Series([10, 20, 30, 40, 50, 60], name='series', index=[6, 7, 8, 9, 10, 15]).sort_values()
        nested: Dict[str, Any] = {'s1': s, 's2': s.copy()}
        kwargs: Dict[str, Any] = {} if orient is None else {'orient': orient}
        exp: Dict[str, Any] = {'s1': ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)),
                               's2': ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs))}
        assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp

    def test_index(self) -> None:
        i: Index = Index([23, 45, 18, 98, 43, 11], name='index')
        output: Index = Index(ujson.ujson_loads(ujson.ujson_dumps(i)), name='index')
        tm.assert_index_equal(i, output)
        dec: Dict[str, Any] = _clean_dict(ujson.ujson_loads(ujson.ujson_dumps(i, orient='split')))
        output = Index(**dec)
        tm.assert_index_equal(i, output)
        assert i.name == output.name
        tm.assert_index_equal(i, output)
        assert i.name == output.name
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='values')), name='index')
        tm.assert_index_equal(i, output)
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='records')), name='index')
        tm.assert_index_equal(i, output)
        output = Index(ujson.ujson_loads(ujson.ujson_dumps(i, orient='index')), name='index')
        tm.assert_index_equal(i, output)

    def test_datetime_index(self) -> None:
        date_unit: str = 'ns'
        rng: DatetimeIndex = DatetimeIndex(list(date_range('1/1/2000', periods=20)), freq=None)
        encoded: str = ujson.ujson_dumps(rng, date_unit=date_unit)
        decoded: DatetimeIndex = DatetimeIndex(np.array(ujson.ujson_loads(encoded)))
        tm.assert_index_equal(rng, decoded)
        ts: Series = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
        decoded = Series(ujson.ujson_loads(ujson.ujson_dumps(ts, date_unit=date_unit)))
        idx_values: np.ndarray = decoded.index.values.astype(np.int64)
        decoded.index = DatetimeIndex(idx_values)
        tm.assert_series_equal(ts, decoded)

    @pytest.mark.parametrize('invalid_arr', ['[31337,]', '[,31337]', '[]]', '[,]'])
    def test_decode_invalid_array(self, invalid_arr: str) -> None:
        msg: str = 'Expected object or value|Trailing data|Unexpected character found when decoding array value'
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_arr)

    @pytest.mark.parametrize('arr', [[], [31337]])
    def test_decode_array(self, arr: List[Any]) -> None:
        assert arr == ujson.ujson_loads(str(arr))

    @pytest.mark.parametrize('extreme_num', [9223372036854775807, -9223372036854775808])
    def test_decode_extreme_numbers(self, extreme_num: int) -> None:
        assert extreme_num == ujson.ujson_loads(str(extreme_num))

    @pytest.mark.parametrize('too_extreme_num', [f'{2 ** 64}', f'{-2 ** 63 - 1}'])
    def test_decode_too_extreme_numbers(self, too_extreme_num: str) -> None:
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            ujson.ujson_loads(too_extreme_num)

    def test_decode_with_trailing_whitespaces(self) -> None:
        assert {} == ujson.ujson_loads('{}\n\t ')

    def test_decode_with_trailing_non_whitespaces(self) -> None:
        with pytest.raises(ValueError, match='Trailing data'):
            ujson.ujson_loads('{}\n\t a')

    @pytest.mark.parametrize('value', [f'{2 ** 64}', f'{-2 ** 63 - 1}'])
    def test_decode_array_with_big_int(self, value: str) -> None:
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            ujson.ujson_loads(value)

    @pytest.mark.parametrize('float_number', [1.1234567893, 1.234567893, 1.34567893, 1.4567893, 1.567893, 1.67893, 1.7893, 1.893, 1.3])
    @pytest.mark.parametrize('sign', [-1, 1])
    def test_decode_floating_point(self, sign: int, float_number: float) -> None:
        float_number *= sign
        tm.assert_almost_equal(float_number, ujson.ujson_loads(str(float_number)), rtol=1e-15)

    def test_encode_big_set(self) -> None:
        s: set = set(range(100000))
        ujson.ujson_dumps(s)

    def test_encode_empty_set(self) -> None:
        assert '[]' == ujson.ujson_dumps(set())

    def test_encode_set(self) -> None:
        s: set = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        enc: str = ujson.ujson_dumps(s)
        dec: Any = ujson.ujson_loads(enc)
        for v in dec:
            assert v in s

    @pytest.mark.parametrize('td', [
        Timedelta(days=366), Timedelta(days=-1), Timedelta(hours=13, minutes=5, seconds=5),
        Timedelta(hours=13, minutes=20, seconds=30), Timedelta(days=-1, nanoseconds=5),
        Timedelta(nanoseconds=1), Timedelta(microseconds=1, nanoseconds=1),
        Timedelta(milliseconds=1, microseconds=1, nanoseconds=1),
        Timedelta(milliseconds=999, microseconds=999, nanoseconds=999)
    ])
    def test_encode_timedelta_iso(self, td: Timedelta) -> None:
        result: str = ujson.ujson_dumps(td, iso_dates=True)
        expected: str = f'"{td.isoformat()}"'
        assert result == expected

    def test_encode_periodindex(self) -> None:
        p: PeriodIndex = PeriodIndex(['2022-04-06', '2022-04-07'], freq='D')
        df: DataFrame = DataFrame(index=p)
        assert df.to_json() == '{}'
