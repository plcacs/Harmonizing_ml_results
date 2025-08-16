import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
from typing import Any, Dict, List, Optional, Union, cast

import dateutil
import numpy as np
import pytest

import pandas._libs.json as ujson
from pandas.compat import IS64

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    NaT,
    PeriodIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm


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


@pytest.fixture(
    params=[None, "split", "records", "values", "index"]  # Column indexed by default.
)
def orient(request: pytest.FixtureRequest) -> Optional[str]:
    return cast(Optional[str], request.param)


class TestUltraJSONTests:
    @pytest.mark.skipif(not IS64, reason="not compliant on 32-bit, xref #15865")
    @pytest.mark.parametrize(
        "value, double_precision",
        [
            ("1337.1337", 15),
            ("0.95", 1),
            ("0.94", 1),
            ("1.95", 1),
            ("-1.95", 1),
            ("0.995", 2),
            ("0.9995", 3),
            ("0.99999999999999944", 15),
        ],
    )
    def test_encode_decimal(self, value: str, double_precision: int) -> None:
        sut = decimal.Decimal(value)
        encoded = ujson.ujson_dumps(sut, double_precision=double_precision)
        decoded = ujson.ujson_loads(encoded)
        assert decoded == value

    @pytest.mark.parametrize("ensure_ascii", [True, False])
    def test_encode_string_conversion(self, ensure_ascii: bool) -> None:
        string_input = "A string \\ / \b \f \n \r \t </script> &"
        not_html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
        html_encoded = (
            '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'
        )

        def helper(expected_output: str, **encode_kwargs: Any) -> None:
            output = ujson.ujson_dumps(
                string_input, ensure_ascii=ensure_ascii, **encode_kwargs
            )

            assert output == expected_output
            assert string_input == json.loads(output)
            assert string_input == ujson.ujson_loads(output)

        # Default behavior assumes encode_html_chars=False.
        helper(not_html_encoded)

        # Make sure explicit encode_html_chars=False works.
        helper(not_html_encoded, encode_html_chars=False)

        # Make sure explicit encode_html_chars=True does the encoding.
        helper(html_encoded, encode_html_chars=True)

    @pytest.mark.parametrize(
        "long_number", [-4342969734183514, -12345678901234.56789012, -528656961.4399388]
    )
    def test_double_long_numbers(self, long_number: float) -> None:
        sut = {"a": long_number}
        encoded = ujson.ujson_dumps(sut, double_precision=15)

        decoded = ujson.ujson_loads(encoded)
        assert sut == decoded

    def test_encode_non_c_locale(self) -> None:
        lc_category = locale.LC_NUMERIC

        # We just need one of these locales to work.
        for new_locale in ("it_IT.UTF-8", "Italian_Italy"):
            if tm.can_set_locale(new_locale, lc_category):
                with tm.set_locale(new_locale, lc_category):
                    assert ujson.ujson_loads(ujson.ujson_dumps(4.78e60)) == 4.78e60
                    assert ujson.ujson_loads("4.78", precise_float=True) == 4.78
                break

    def test_decimal_decode_test_precise(self) -> None:
        sut = {"a": 4.56}
        encoded = ujson.ujson_dumps(sut)
        decoded = ujson.ujson_loads(encoded, precise_float=True)
        assert sut == decoded

    def test_encode_double_tiny_exponential(self) -> None:
        num = 1e-40
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = 1e-100
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-45
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-145
        assert np.allclose(num, ujson.ujson_loads(ujson.ujson_dumps(num)))

    @pytest.mark.parametrize("unicode_key", ["key1", "بن"])
    def test_encode_dict_with_unicode_keys(self, unicode_key: str) -> None:
        unicode_dict = {unicode_key: "value1"}
        assert unicode_dict == ujson.ujson_loads(ujson.ujson_dumps(unicode_dict))

    @pytest.mark.parametrize("double_input", [math.pi, -math.pi])
    def test_encode_double_conversion(self, double_input: float) -> None:
        output = ujson.ujson_dumps(double_input)
        assert round(double_input, 5) == round(json.loads(output), 5)
        assert round(double_input, 5) == round(ujson.ujson_loads(output), 5)

    def test_encode_with_decimal(self) -> None:
        decimal_input = 1.0
        output = ujson.ujson_dumps(decimal_input)

        assert output == "1.0"

    def test_encode_array_of_nested_arrays(self) -> None:
        nested_input = [[[[]]]] * 20
        output = ujson.ujson_dumps(nested_input)

        assert nested_input == json.loads(output)
        assert nested_input == ujson.ujson_loads(output)

    def test_encode_array_of_doubles(self) -> None:
        doubles_input = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
        output = ujson.ujson_dumps(doubles_input)

        assert doubles_input == json.loads(output)
        assert doubles_input == ujson.ujson_loads(output)

    def test_double_precision(self) -> None:
        double_input = 30.012345678901234
        output = ujson.ujson_dumps(double_input, double_precision=15)

        assert double_input == json.loads(output)
        assert double_input == ujson.ujson_loads(output)

        for double_precision in (3, 9):
            output = ujson.ujson_dumps(double_input, double_precision=double_precision)
            rounded_input = round(double_input, double_precision)

            assert rounded_input == json.loads(output)
            assert rounded_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "invalid_val",
        [
            20,
            -1,
            "9",
            None,
        ],
    )
    def test_invalid_double_precision(self, invalid_val: Any) -> None:
        double_input = 30.12345678901234567890
        expected_exception = ValueError if isinstance(invalid_val, int) else TypeError
        msg = (
            r"Invalid value '.*' for option 'double_precision', max is '15'|"
            r"an integer is required \(got type |"
            r"object cannot be interpreted as an integer"
        )
        with pytest.raises(expected_exception, match=msg):
            ujson.ujson_dumps(double_input, double_precision=invalid_val)

    def test_encode_string_conversion2(self) -> None:
        string_input = "A string \\ / \b \f \n \r \t"
        output = ujson.ujson_dumps(string_input)

        assert string_input == json.loads(output)
        assert string_input == ujson.ujson_loads(output)
        assert output == '"A string \\\\ \\/ \\b \\f \\n \\r \\t"'

    @pytest.mark.parametrize(
        "unicode_input",
        ["Räksmörgås اسامة بن محمد بن عوض بن لادن", "\xe6\x97\xa5\xd1\x88"],
    )
    def test_encode_unicode_conversion(self, unicode_input: str) -> None:
        enc = ujson.ujson_dumps(unicode_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(unicode_input)
        assert dec == json.loads(enc)

    def test_encode_control_escaping(self) -> None:
        escaped_input = "\x19"
        enc = ujson.ujson_dumps(escaped_input)
        dec = ujson.ujson_loads(enc)

        assert escaped_input == dec
        assert enc == json.dumps(escaped_input)

    def test_encode_unicode_surrogate_pair(self) -> None:
        surrogate_input = "\xf0\x90\x8d\x86"
        enc = ujson.ujson_dumps(surrogate_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(surrogate_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8(self) -> None:
        four_bytes_input = "\xf0\x91\x80\xb0TRAILINGNORMAL"
        enc = ujson.ujson_dumps(four_bytes_input)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8highest(self) -> None:
        four_bytes_input = "\xf3\xbf\xbf\xbfTRAILINGNORMAL"
        enc = ujson.ujson_dumps(four_bytes_input)

        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_error(self) -> None:
        string = "'\udac0'"
        msg = (
            r"'utf-8' codec can't encode character '\\udac0' "
            r"in position 1: surrogates not allowed"
        )
        with pytest.raises(UnicodeEncodeError, match=msg):
            ujson.ujson_dumps([string])

    def test_encode_array_in_array(self) -> None:
        arr_in_arr_input = [[[[]]]]
        output = ujson.ujson_dumps(arr_in_arr_input)

        assert arr_in_arr_input == json.loads(output)
        assert output == json.dumps(arr_in_arr_input)
        assert arr_in_arr_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "num_input",
        [
            31337,
            -31337,  # Negative number.
            -9223372036854775808,  # Large negative number.
        ],
    )
    def test_encode_num_conversion(self, num_input: int) -> None:
        output = ujson.ujson_dumps(num_input)
        assert num_input == json.loads(output)
        assert output == json.dumps(num_input)
        assert num_input == ujson.ujson_loads(output)

    def test_encode_list_conversion(self) -> None:
        list_input = [1, 2, 3, 4]
        output = ujson.ujson_dumps(list_input)

        assert list_input == json.loads(output)
        assert list_input == ujson.ujson_loads(output)

    def test_encode_dict_conversion(self) -> None:
        dict_input = {"k1": 1, "k2": 2, "k3": 3, "k4": 4}
        output = ujson.ujson_dumps(dict_input)

        assert dict_input == json.loads(output)
        assert dict_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize("builtin_value", [None, True, False])
    def test_encode_builtin_values_conversion(self, builtin_value: Any) -> None:
        output = ujson.ujson_dumps(builtin_value)
        assert builtin_value == json.loads(output)
        assert output == json.dumps(builtin_value)
        assert builtin_value == ujson.ujson_loads(output)

    def test_encode_datetime_conversion(self) -> None:
        datetime_input = datetime.datetime.fromtimestamp(time.time())
        output = ujson.ujson_dumps(datetime_input, date_unit="s")
        expected = calendar.timegm(datetime_input.utctimetuple())

        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    def test_encode_date_conversion(self) -> None:
        date_input = datetime.date.fromtimestamp(time.time())
        output = ujson.ujson_dumps(date_input, date_unit="s")

        tup = (date_input.year, date_input.month, date_input.day, 0, 0, 0)
        expected = calendar.timegm(tup)

        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    @pytest.mark.parametrize(
        "test",
        [datetime.time(), datetime.time(1, 2, 3), datetime.time(10, 12, 15, 343243)],
    )
    def test_encode_time_conversion_basic(self, test: datetime.time) -> None:
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_pytz(self) -> None:
        # see gh-11473: to_json segfaults with timezone-aware datetimes
        pytz = pytest.importorskip("pytz")
        test = datetime.time(10, 12, 15, 343243, pytz.utc)
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_dateutil(self) -> None:
        # see gh-11473: to_json segfaults with timezone-aware datetimes
        test = datetime.time(10, 12, 15, 343243, dateutil.tz.tzutc())
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    @pytest.mark.parametrize(
        "decoded_input", [NaT, np.datetime64("NaT"), np.nan, np.inf, -np.inf]
    )
    def test_encode_as_null(self, decoded_input: Any) -> None:
        assert ujson.ujson_dumps(decoded_input) == "null", "Expected null"

    def test_datetime_units(self) -> None:
        val = datetime.datetime(2013, 8, 17, 21, 17, 12, 215504)
        stamp = Timestamp(val).as_unit("ns")

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="s"))
        assert roundtrip == stamp._value // 10**9

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="ms"))
        assert roundtrip == stamp._value // 10**6

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="us"))
        assert roundtrip == stamp._value // 10**3

        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit="ns"))
        assert roundtrip == stamp._value

        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_dumps(val, date_unit="foo")

    def test_encode_to_utf8(self) -> None:
        unencoded = "\xe6\x97\xa5\xd1\x88"

        enc = ujson.ujson_dumps(unencoded, ensure_ascii=False)
        dec = ujson.ujson_loads(enc)

        assert enc == json.dumps(unencoded, ensure_ascii=False)
        assert dec == json.loads(enc)

    def test_decode_from_unicode(self) -> None:
        unicode_input = '{"obj": 31337}'

        dec1 = ujson.ujson_loads(unicode_input)
        dec2 = ujson.ujson_loads(str(unicode_input))

        assert dec1 == dec2

    def test_encode_recursion_max(self) -> None:
        # 8 is the max recursion depth

        class O2:
            member = 0

        class O1:
            member = 0

        decoded_input = O1()
        decoded_input.member = O2()
        decoded_input.member.member = decoded_input

        with pytest.raises(OverflowError, match="Maximum recursion level reached"):
            ujson.ujson_dumps(decoded_input)

    def test_decode_jibberish(self) -> None:
        jibberish = "fdsa sda v9sa fdsa"
        msg = "Unexpected character found when decoding 'false'"
        with pytest.raises(ValueError, match=msg):
            ujson