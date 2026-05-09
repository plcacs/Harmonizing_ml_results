from datetime import datetime, timedelta, timezone
from io import StringIO
import numpy as np
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Index, MultiIndex, Series, Timestamp
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.tools.datetimes import start_caching_at
from pandas.io.parsers import read_csv
from typing import Any, Dict, List, Optional

@pytest.mark.filterwarnings('ignore:Passing a BlockManager to DataFrame:DeprecationWarning')
def test_date_col_as_index_col(all_parsers: Any) -> None:
    # ...

@pytest.mark.parametrize('parse_dates', [(1,), np.array([4, 5]), {1, 3}])
def test_read_with_parse_dates_invalid_type(all_parsers: Any, parse_dates: Any) -> None:
    # ...

def test_parse_date_column_with_empty_string(all_parsers: Any) -> None:
    # ...

@pytest.mark.parametrize('date_string', ['32/32/2019', '02/30/2019', '13/13/2019', '13/2019', 'a3/11/2018', '10/11/2o17'])
def test_invalid_parse_delimited_date(all_parsers: Any, date_string: str) -> None:
    # ...

@pytest.mark.parametrize('names, usecols, parse_dates, missing_cols', [(None, ['val'], ['date', 'time'], 'date, time'), (None, ['val'], [0, 'time'], 'time'), (['date1', 'time1', 'temperature'], None, ['date1', 'time'], 'time'), (['date1', 'time1', 'temperature'], ['date1', 'temperature'], ['date1', 'time'], 'time')])
def test_missing_parse_dates_column_raises(all_parsers: Any, names: Optional[List[str]], usecols: Optional[List[str]], parse_dates: Optional[List[str]], missing_cols: str) -> None:
    # ...

@xfail_pyarrow
def test_date_parser_and_names(all_parsers: Any) -> None:
    # ...

@xfail_pyarrow
def test_date_parser_multiindex_columns(all_parsers: Any) -> None:
    # ...

def test_date_parser_usecols_thousands(all_parsers: Any) -> None:
    # ...

def test_dayfirst_warnings() -> None:
    # ...

@pytest.mark.parametrize('date_string, dayfirst', [pytest.param('31/1/2014', False, id='second date is single-digit'), pytest.param('1/31/2014', True, id='first date is single-digit')])
def test_dayfirst_warnings_no_leading_zero(date_string: str, dayfirst: bool) -> None:
    # ...

@skip_pyarrow
def test_infer_first_column_as_index(all_parsers: Any) -> None:
    # ...

@xfail_pyarrow
def test_replace_nans_before_parsing_dates(all_parsers: Any) -> None:
    # ...

@xfail_pyarrow
def test_parse_dates_and_string_dtype(all_parsers: Any) -> None:
    # ...

def test_parse_dot_separated_dates(all_parsers: Any) -> None:
    # ...

def test_parse_dates_dict_format(all_parsers: Any) -> None:
    # ...

@xfail_pyarrow
def test_parse_dates_dict_format_index(all_parsers: Any) -> None:
    # ...

def test_parse_dates_arrow_engine(all_parsers: Any) -> None:
    # ...

@xfail_pyarrow
def test_from_csv_with_mixed_offsets(all_parsers: Any) -> None:
    # ...
