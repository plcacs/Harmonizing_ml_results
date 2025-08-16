from datetime import datetime
import re
from dateutil.parser import parse as du_parse
from hypothesis import given
import numpy as np
import pytest
from pandas._libs.tslibs import parsing, strptime
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso
from pandas.compat import ISMUSL, WASM, is_platform_windows
import pandas.util._test_decorators as td
from pandas import Timestamp
import pandas._testing as tm
from pandas._testing._hypothesis import DATETIME_NO_TZ

def test_parsing_tzlocal_deprecated() -> None:
    ...

def test_parse_datetime_string_with_reso() -> None:
    ...

def test_parse_datetime_string_with_reso_nanosecond_reso() -> None:
    ...

def test_parse_datetime_string_with_reso_invalid_type() -> None:
    ...

def test_parse_time_quarter_with_dash() -> None:
    ...

def test_parse_time_quarter_with_dash_error() -> None:
    ...

def test_does_not_convert_mixed_integer() -> None:
    ...

def test_parsers_quarterly_with_freq_error() -> None:
    ...

def test_parsers_quarterly_with_freq() -> None:
    ...

def test_parsers_quarter_invalid() -> None:
    ...

def test_parsers_month_freq() -> None:
    ...

def test_guess_datetime_format_with_parseable_formats() -> None:
    ...

def test_guess_datetime_format_with_dayfirst() -> None:
    ...

def test_guess_datetime_format_with_locale_specific_formats() -> None:
    ...

def test_guess_datetime_format_invalid_inputs() -> None:
    ...

def test_guess_datetime_format_wrong_type_inputs() -> None:
    ...

def test_guess_datetime_format_no_padding() -> None:
    ...

def test_try_parse_dates() -> None:
    ...

def test_parse_datetime_string_with_reso_check_instance_type_raise_exception() -> None:
    ...

def test_is_iso_format() -> None:
    ...

def test_guess_datetime_format_f() -> None:
    ...

def _helper_hypothesis_delimited_date(call, date_string, **kwargs) -> Tuple[str, Any]:
    ...

@given(DATETIME_NO_TZ)
def test_hypothesis_delimited_date(request, date_format, dayfirst, delimiter, test_datetime) -> None:
    ...
