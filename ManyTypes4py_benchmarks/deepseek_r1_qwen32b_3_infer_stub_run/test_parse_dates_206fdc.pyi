"""
Stub file for test_parse_dates_206fdc.py
"""

from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    NaT,
    _testing as tm,
)
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv

pytestmark = pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
xfail_pyarrow = pytest.mark.usefixtures('pyarrow_xfail')
skip_pyarrow = pytest.mark.usefixtures('pyarrow_skip')

def test_date_col_as_index_col(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_nat_parse(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dates_implicit_first_col(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dates_string(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dates_column_list(all_parsers: pytest.FixtureRequest, parse_dates: Union[List[int], List[str]]) -> None:
    ...

def test_multi_index_parse_dates(all_parsers: pytest.FixtureRequest, index_col: List[int]) -> None:
    ...

def test_parse_tz_aware(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_read_with_parse_dates_scalar_non_bool(all_parsers: pytest.FixtureRequest, kwargs: Dict[str, Any]) -> None:
    ...

def test_read_with_parse_dates_invalid_type(all_parsers: pytest.FixtureRequest, parse_dates: Union[Tuple[int, ...], np.ndarray, Set[int]]) -> None:
    ...

def test_bad_date_parse(all_parsers: pytest.FixtureRequest, cache: bool, value: str) -> None:
    ...

def test_bad_date_parse_with_warning(all_parsers: pytest.FixtureRequest, cache: bool, value: str) -> None:
    ...

def test_parse_dates_empty_string(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dates_no_convert_thousands(all_parsers: pytest.FixtureRequest, data: str, kwargs: Dict[str, Any], expected: DataFrame) -> None:
    ...

def test_parse_date_column_with_empty_string(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_date_float(all_parsers: pytest.FixtureRequest, data: str, expected: List[Union[int, float]], parse_dates: bool) -> None:
    ...

def test_parse_timezone(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_invalid_parse_delimited_date(all_parsers: pytest.FixtureRequest, date_string: str) -> None:
    ...

def test_parse_delimited_date_swap_no_warning(all_parsers: pytest.FixtureRequest, date_string: str, dayfirst: bool, expected: datetime) -> None:
    ...

def test_parse_delimited_date_swap_with_warning(all_parsers: pytest.FixtureRequest, date_string: str, dayfirst: bool, expected: datetime) -> None:
    ...

def test_parse_multiple_delimited_dates_with_swap_warnings() -> None:
    ...

def test_missing_parse_dates_column_raises(all_parsers: pytest.FixtureRequest, names: Optional[List[str]], usecols: Optional[List[str]], parse_dates: Union[List[str], List[int]], missing_cols: str) -> None:
    ...

def test_date_parser_and_names(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_date_parser_multiindex_columns(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_date_parser_usecols_thousands(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_dayfirst_warnings() -> None:
    ...

def test_dayfirst_warnings_no_leading_zero(date_string: str, dayfirst: bool) -> None:
    ...

def test_infer_first_column_as_index(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_replace_nans_before_parsing_dates(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dates_and_string_dtype(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dot_separated_dates(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dates_dict_format(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dates_dict_format_index(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_parse_dates_arrow_engine(all_parsers: pytest.FixtureRequest) -> None:
    ...

def test_from_csv_with_mixed_offsets(all_parsers: pytest.FixtureRequest) -> None:
    ...