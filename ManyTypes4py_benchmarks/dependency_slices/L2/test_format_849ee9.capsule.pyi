from typing import Any

# === Internal dependency: io ===
StringIO: Any

# === Third-party dependency: numpy ===
# Used symbols: arange, array, float64, inf, linspace, nan, ndarray, random, where

# === Internal dependency: pandas ===
# re-export: from pandas._config import get_option
# re-export: from pandas._config import reset_option
# re-export: from pandas._config import option_context
# re-export: from pandas.core.api import NA
# re-export: from pandas.core.api import Index
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import NaT
# re-export: from pandas.core.api import Period
# re-export: from pandas.core.api import period_range
# re-export: from pandas.core.api import Timestamp
# re-export: from pandas.core.api import date_range
# re-export: from pandas.core.api import to_datetime
# re-export: from pandas.core.api import to_timedelta
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame
# re-export: from pandas.core.reshape.api import concat
# re-export: from pandas import api
# re-export: from pandas.io.api import read_csv

# === Internal dependency: pandas._config ===
def using_string_dtype() -> bool: ...

# === Internal dependency: pandas.io.formats.format ===
class SeriesFormatter:
    def __init__(self, series: Series, *, length: bool | str = ..., header: bool = ..., index: bool = ..., na_rep: str = ..., name: bool = ..., float_format: str | None = ..., dtype: bool = ..., max_rows: int | None = ..., min_rows: int | None = ...) -> None: ...
    def _get_footer(self) -> str: ...
class DataFrameFormatter:
    def __init__(self, frame: DataFrame, columns: Axes | None = ..., col_space: ColspaceArgType | None = ..., header: bool | SequenceNotStr[str] = ..., index: bool = ..., na_rep: str = ..., formatters: FormattersType | None = ..., justify: str | None = ..., float_format: FloatFormatType | None = ..., sparsify: bool | None = ..., index_names: bool = ..., max_rows: int | None = ..., min_rows: int | None = ..., max_cols: int | None = ..., show_dimensions: bool | str = ..., decimal: str = ..., bold_rows: bool = ..., escape: bool = ...) -> None: ...
class _GenericArrayFormatter:
    def __init__(self, values: ArrayLike, digits: int = ..., formatter: Callable | None = ..., na_rep: str = ..., space: str | int = ..., float_format: FloatFormatType | None = ..., justify: str = ..., decimal: str = ..., quoting: int | None = ..., fixed_width: bool = ..., leading_space: bool | None = ..., fallback_formatter: Callable | None = ...) -> None: ...
    def get_result(self) -> list[str]: ...
class FloatArrayFormatter(_GenericArrayFormatter):
    def __init__(self, *args, **kwargs) -> None: ...
class _Datetime64Formatter(_GenericArrayFormatter):
    def __init__(self, values: DatetimeArray, nat_rep: str = ..., date_format: None = ..., **kwargs) -> None: ...
def format_percentiles(percentiles: np.ndarray | Sequence[float]) -> list[str]: ...
class _Datetime64TZFormatter(_Datetime64Formatter): ...
class _Timedelta64Formatter(_GenericArrayFormatter): ...

# === Internal dependency: pandas.io.formats.printing ===
def pprint_thing(thing: object, _nest_lvl: int = ..., escape_chars: EscapeChars | None = ..., default_escapes: bool = ..., quote_strings: bool = ..., max_seq_items: int | None = ...) -> str: ...
def get_adjustment() -> _TextAdjustment: ...

# === Third-party dependency: pytest ===
# Used symbols: importorskip, mark, raises, skip