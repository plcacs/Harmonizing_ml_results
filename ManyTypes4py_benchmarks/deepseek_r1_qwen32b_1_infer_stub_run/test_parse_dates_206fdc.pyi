"""
Stub file for test_parse_dates_206fdc module
"""

from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, Timestamp
from pandas._testing import tm
from pytest import fixture, mark

@fixture
def all_parsers() -> Any:
    ...

@mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
@mark.usefixtures('pyarrow_xfail')
def test_date_col_as_index_col(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
def test_nat_parse(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_skip')
def test_parse_dates_implicit_first_col(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
def test_parse_dates_string(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
@pytest.mark.parametrize('parse_dates', [[0, 2], ['a', 'c']])
def test_parse_dates_column_list(all_parsers: Any, parse_dates: Union[List[int], List[str]]) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
@pytest.mark.parametrize('index_col', [[0, 1], [1, 0]])
def test_multi_index_parse_dates(all_parsers: Any, index_col: List[int]) -> None:
    ...

def test_parse_tz_aware(all_parsers: Any) -> None:
    ...

@pytest.mark.parametrize('kwargs', [{}, {'index_col': 'C'}])
def test_read_with_parse_dates_scalar_non_bool(all_parsers: Any, kwargs: Dict[str, Any]) -> None:
    ...

@pytest.mark.parametrize('parse_dates', [(1,), np.array([4, 5]), {1, 3}])
def test_read_with_parse_dates_invalid_type(all_parsers: Any, parse_dates: Union[Tuple[int, ...], np.ndarray, Set[int]]) -> None:
    ...

@pytest.mark.parametrize('value', ['nan', ''])
def test_bad_date_parse(all_parsers: Any, cache: Any, value: str) -> None:
    ...

@pytest.mark.parametrize('value', ['0'])
def test_bad_date_parse_with_warning(all_parsers: Any, cache: Any, value: str) -> None:
    ...

def test_parse_dates_empty_string(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
@pytest.mark.parametrize('data,kwargs,expected', [
    ('a\n04.15.2016', {'parse_dates': ['a']}, DataFrame([datetime(2016, 4, 15)], columns=['a'], dtype='M8[s]')),
    ('a\n04.15.2016', {'parse_dates': True, 'index_col': 0}, DataFrame(index=DatetimeIndex(['2016-04-15'], dtype='M8[s]', name='a'), columns=[])),
    ('a,b\n04.15.2016,09.16.2013', {'parse_dates': ['a', 'b']}, DataFrame([[datetime(2016, 4, 15), datetime(2013, 9, 16)]], dtype='M8[s]', columns=['a', 'b'])),
    ('a,b\n04.15.2016,09.16.2013', {'parse_dates': True, 'index_col': [0, 1]}, DataFrame(index=MultiIndex.from_tuples([(Timestamp(2016, 4, 15).as_unit('s'), Timestamp(2013, 9, 16).as_unit('s'))], names=['a', 'b']), columns=[]))
])
def test_parse_dates_no_convert_thousands(all_parsers: Any, data: str, kwargs: Dict[str, Any], expected: DataFrame) -> None:
    ...

def test_parse_date_column_with_empty_string(all_parsers: Any) -> None:
    ...

@pytest.mark.parametrize('data,expected', [
    ('a\n135217135789158401\n1352171357E+5', [135217135789158401, 135217135700000]),
    ('a\n99999999999\n123456789012345\n1234E+0', [99999999999, 123456789012345, 1234])
])
@pytest.mark.parametrize('parse_dates', [True, False])
def test_parse_date_float(all_parsers: Any, data: str, expected: List[int], parse_dates: bool) -> None:
    ...

def test_parse_timezone(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_skip')
@pytest.mark.parametrize('date_string', ['32/32/2019', '02/30/2019', '13/13/2019', '13/2019', 'a3/11/2018', '10/11/2o17'])
def test_invalid_parse_delimited_date(all_parsers: Any, date_string: str) -> None:
    ...

@pytest.mark.parametrize('date_string,dayfirst,expected', [
    ('13/02/2019', True, datetime(2019, 2, 13)),
    ('02/13/2019', False, datetime(2019, 2, 13)),
    ('04/02/2019', True, datetime(2019, 2, 4))
])
def test_parse_delimited_date_swap_no_warning(all_parsers: Any, date_string: str, dayfirst: bool, expected: datetime, request: Any) -> None:
    ...

@pytest.mark.parametrize('date_string,dayfirst,expected', [
    ('13/02/2019', False, datetime(2019, 2, 13)),
    ('02/13/2019', True, datetime(2019, 2, 13))
])
def test_parse_delimited_date_swap_with_warning(all_parsers: Any, date_string: str, dayfirst: bool, expected: datetime) -> None:
    ...

def test_parse_multiple_delimited_dates_with_swap_warnings() -> None:
    ...

@mark.usefixtures('pyarrow_skip')
@pytest.mark.parametrize('names, usecols, parse_dates, missing_cols', [
    (None, ['val'], ['date', 'time'], 'date, time'),
    (None, ['val'], [0, 'time'], 'time'),
    (['date1', 'time1', 'temperature'], None, ['date', 'time'], 'date, time'),
    (['date1', 'time1', 'temperature'], ['date1', 'temperature'], ['date1', 'time'], 'time')
])
def test_missing_parse_dates_column_raises(all_parsers: Any, names: Optional[List[str]], usecols: Optional[List[str]], parse_dates: Union[List[str], List[int]], missing_cols: str) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
def test_date_parser_and_names(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
def test_date_parser_multiindex_columns(all_parsers: Any) -> None:
    ...

def test_date_parser_usecols_thousands(all_parsers: Any) -> None:
    ...

def test_dayfirst_warnings() -> None:
    ...

@pytest.mark.parametrize('date_string, dayfirst', [
    pytest.param('31/1/2014', False, id='second date is single-digit'),
    pytest.param('1/31/2014', True, id='first date is single-digit')
])
def test_dayfirst_warnings_no_leading_zero(date_string: str, dayfirst: bool) -> None:
    ...

@mark.usefixtures('pyarrow_skip')
def test_infer_first_column_as_index(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
def test_replace_nans_before_parsing_dates(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
def test_parse_dates_and_string_dtype(all_parsers: Any) -> None:
    ...

def test_parse_dot_separated_dates(all_parsers: Any) -> None:
    ...

def test_parse_dates_dict_format(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
def test_parse_dates_dict_format_index(all_parsers: Any) -> None:
    ...

def test_parse_dates_arrow_engine(all_parsers: Any) -> None:
    ...

@mark.usefixtures('pyarrow_xfail')
def test_from_csv_with_mixed_offsets(all_parsers: Any) -> None:
    ...