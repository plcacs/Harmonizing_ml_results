from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, float64, inf, linspace, nan, ndarray, random, where

# === Internal dependency: pandas ===
from pandas._config import get_option
from pandas._config import reset_option
from pandas._config import option_context
from pandas.core.api import NA
from pandas.core.api import Index
from pandas.core.api import MultiIndex
from pandas.core.api import NaT
from pandas.core.api import Period
from pandas.core.api import period_range
from pandas.core.api import Timestamp
from pandas.core.api import date_range
from pandas.core.api import to_datetime
from pandas.core.api import to_timedelta
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import concat
from pandas import api
from pandas.io.api import read_csv

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas.io.formats.format ===
class SeriesFormatter:
    def __init__(self, series, *, length=..., header=..., index=..., na_rep=..., name=..., float_format=..., dtype=..., max_rows=..., min_rows=...): ...
    def _get_footer(self): ...
class DataFrameFormatter:
    def __init__(self, frame, columns=..., col_space=..., header=..., index=..., na_rep=..., formatters=..., justify=..., float_format=..., sparsify=..., index_names=..., max_rows=..., min_rows=..., max_cols=..., show_dimensions=..., decimal=..., bold_rows=..., escape=...): ...
class _GenericArrayFormatter:
    def __init__(self, values, digits=..., formatter=..., na_rep=..., space=..., float_format=..., justify=..., decimal=..., quoting=..., fixed_width=..., leading_space=..., fallback_formatter=...): ...
    def get_result(self): ...
class FloatArrayFormatter(_GenericArrayFormatter):
    def __init__(self, *args, **kwargs): ...
class _Datetime64Formatter(_GenericArrayFormatter):
    def __init__(self, values, nat_rep=..., date_format=..., **kwargs): ...
def format_percentiles(percentiles): ...
class _Datetime64TZFormatter(_Datetime64Formatter): ...
class _Timedelta64Formatter(_GenericArrayFormatter): ...

# === Internal dependency: pandas.io.formats.printing ===
def pprint_thing(thing, _nest_lvl=..., escape_chars=..., default_escapes=..., quote_strings=..., max_seq_items=...): ...
def get_adjustment(): ...

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises, skip