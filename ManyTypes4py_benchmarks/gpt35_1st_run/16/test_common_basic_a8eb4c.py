from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
from typing import List, Dict, Any
import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat import HAS_PYARROW
from pandas.errors import EmptyDataError, ParserError, ParserWarning
from pandas import DataFrame, Index, compat
import pandas._testing as tm

def test_read_csv_local(all_parsers: Any, csv1: str) -> None:
    ...

def test_1000_sep(all_parsers: Any) -> None:
    ...

def test_unnamed_columns(all_parsers: Any) -> None:
    ...

def test_csv_mixed_type(all_parsers: Any) -> None:
    ...

def test_read_csv_low_memory_no_rows_with_index(all_parsers: Any) -> None:
    ...

def test_read_csv_dataframe(all_parsers: Any, csv1: str) -> None:
    ...

def test_read_nrows(all_parsers: Any, nrows: int) -> None:
    ...

def test_read_nrows_bad(all_parsers: Any, nrows: Any) -> None:
    ...

def test_nrows_skipfooter_errors(all_parsers: Any) -> None:
    ...

def test_missing_trailing_delimiters(all_parsers: Any) -> None:
    ...

def test_skip_initial_space(all_parsers: Any) -> None:
    ...

def test_trailing_delimiters(all_parsers: Any) -> None:
    ...

def test_escapechar(all_parsers: Any) -> None:
    ...

def test_ignore_leading_whitespace(all_parsers: Any) -> None:
    ...

def test_skip_initial_space(all_parsers: Any) -> None:
    ...

def test_uneven_lines_with_usecols(all_parsers: Any, usecols: Any) -> None:
    ...

def test_read_empty_with_usecols(all_parsers: Any, data: str, kwargs: Dict[str, Any], expected: Any) -> None:
    ...

def test_trailing_spaces(all_parsers: Any, kwargs: Dict[str, Any], expected_data: Any) -> None:
    ...

def test_blank_lines_between_header_and_data_rows(all_parsers: Any, nrows: int) -> None:
    ...

def test_no_header_two_extra_columns(all_parsers: Any) -> None:
    ...

def test_dict_keys_as_names(all_parsers: Any) -> None:
    ...

def test_encoding_surrogatepass(all_parsers: Any) -> None:
    ...

def test_malformed_second_line(all_parsers: Any) -> None:
    ...

def test_short_single_line(all_parsers: Any) -> None:
    ...

def test_short_multi_line(all_parsers: Any) -> None:
    ...

def test_read_seek(all_parsers: Any) -> None:
    ...
