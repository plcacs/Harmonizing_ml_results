"""
Internal module for formatting output data in csv, html, xml,
and latex files. This module also applies to display formatting.
"""
from __future__ import annotations
from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import TYPE_CHECKING, Any, Final, cast, Optional, Union, List, Dict, Tuple
import numpy as np
from pandas._config.config import get_option, set_option
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import NaT, Timedelta, Timestamp
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.dtypes.common import is_complex_dtype, is_float, is_integer, is_list_like, is_numeric_dtype, is_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, ExtensionDtype
from pandas.core.dtypes.missing import isna, notna
from pandas.core.arrays import Categorical, DatetimeArray, ExtensionArray, TimedeltaArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import Index, MultiIndex, PeriodIndex, ensure_index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.io.common import check_parent_directory, stringify_path
from pandas.io.formats import printing
if TYPE_CHECKING:
    from pandas._typing import ArrayLike, Axes, ColspaceArgType, ColspaceType, CompressionOptions, FilePath, FloatFormatType, FormattersType, IndexLabel, SequenceNotStr, StorageOptions, WriteBuffer
    from pandas import DataFrame, Series
common_docstring: str = "\n        Parameters\n        ----------\n        buf : str, Path or StringIO-like, optional, default None\n            Buffer to write to. If None, the output is returned as a string.\n        columns : array-like, optional, default None\n            The subset of columns to write. Writes all columns by default.\n        col_space : %(col_space_type)s, optional\n            %(col_space)s.\n        header : %(header_type)s, optional\n            %(header)s.\n        index : bool, optional, default True\n            Whether to print index (row) labels.\n        na_rep : str, optional, default 'NaN'\n            String representation of ``NaN`` to use.\n        formatters : list, tuple or dict of one-param. functions, optional\n            Formatter functions to apply to columns' elements by position or\n            name.\n            The result of each function must be a unicode string.\n            List/tuple must be of length equal to the number of columns.\n        float_format : one-parameter function, optional, default None\n            Formatter function to apply to columns' elements if they are\n            floats. This function must return a unicode string and will be\n            applied only to the non-``NaN`` elements, with ``NaN`` being\n            handled by ``na_rep``.\n        sparsify : bool, optional, default True\n            Set to False for a DataFrame with a hierarchical index to print\n            every multiindex key at each row.\n        index_names : bool, optional, default True\n            Prints the names of the indexes.\n        justify : str, default None\n            How to justify the column labels. If None uses the option from\n            the print configuration (controlled by set_option), 'right' out\n            of the box. Valid values are\n\n            * left\n            * right\n            * center\n            * justify\n            * justify-all\n            * start\n            * end\n            * inherit\n            * match-parent\n            * initial\n            * unset.\n        max_rows : int, optional\n            Maximum number of rows to display in the console.\n        max_cols : int, optional\n            Maximum number of columns to display in the console.\n        show_dimensions : bool, default False\n            Display DataFrame dimensions (number of rows by number of columns).\n        decimal : str, default '.'\n            Character recognized as decimal separator, e.g. ',' in Europe.\n    "
VALID_JUSTIFY_PARAMETERS: Tuple[str, ...] = ('left', 'right', 'center', 'justify', 'justify-all', 'start', 'end', 'inherit', 'match-parent', 'initial', 'unset')
return_docstring: str = '\n        Returns\n        -------\n        str or None\n            If buf is None, returns the result as a string. Otherwise returns\n            None.\n    '

class SeriesFormatter:
    """
    Implement the main logic of Series.to_string, which underlies
    Series.__repr__.
    """

    def __init__(self, series: Series, *, length: bool | str = True, header: bool = True, index: bool = True, na_rep: str = 'NaN', name: bool = False, float_format: Optional[str] = None, dtype: bool = True, max_rows: Optional[int] = None, min_rows: Optional[int] = None) -> None:
        self.series: Series = series
        self.buf: StringIO = StringIO()
        self.name: bool = name
        self.na_rep: str = na_rep
        self.header: bool = header
        self.length: bool | str = length
        self.index: bool = index
        self.max_rows: Optional[int] = max_rows
        self.min_rows: Optional[int] = min_rows
        if float_format is None:
            float_format = get_option('display.float_format')
        self.float_format: Optional[str] = float_format
        self.dtype: bool = dtype
        self.adj: Any = printing.get_adjustment()
        self.tr_row_num: Optional[int] = None
        self.tr_series: Series = series
        self.is_truncated_vertically: bool = False
        self._chk_truncate()

    def _chk_truncate(self) -> None:
        min_rows: Optional[int] = self.min_rows
        max_rows: Optional[int] = self.max_rows
        is_truncated_vertically: bool = bool(max_rows and len(self.series) > max_rows)
        series: Series = self.series
        if is_truncated_vertically:
            max_rows = cast(int, max_rows)
            if min_rows:
                max_rows = min(min_rows, max_rows)
            if max_rows == 1:
                row_num: int = max_rows
                series = series.iloc[:max_rows]
            else:
                row_num = max_rows // 2
                _len: int = len(series)
                _slice: np.ndarray = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
                series = series.iloc[_slice]
            self.tr_row_num = row_num
        else:
            self.tr_row_num = None
        self.tr_series = series
        self.is_truncated_vertically = is_truncated_vertically

    def _get_footer(self) -> str:
        name: Optional[Any] = self.series.name
        footer: str = ''
        index: Index = self.series.index
        if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)) and index.freq is not None:
            footer += f'Freq: {index.freqstr}'
        if self.name is not False and name is not None:
            if footer:
                footer += ', '
            series_name: str = printing.pprint_thing(name, escape_chars=('\t', '\r', '\n'))
            footer += f'Name: {series_name}'
        if self.length is True or (self.length == 'truncate' and self.is_truncated_vertically):
            if footer:
                footer += ', '
            footer += f'Length: {len(self.series)}'
        if self.dtype is not False and self.dtype is not None:
            dtype_name: Optional[str] = getattr(self.tr_series.dtype, 'name', None)
            if dtype_name:
                if footer:
                    footer += ', '
                footer += f'dtype: {printing.pprint_thing(dtype_name)}'
        if isinstance(self.tr_series.dtype, CategoricalDtype):
            level_info: str = self.tr_series._values._get_repr_footer()
            if footer:
                footer += '\n'
            footer += level_info
        return str(footer)

    def _get_formatted_values(self) -> List[str]:
        return format_array(self.tr_series._values, None, float_format=self.float_format, na_rep=self.na_rep, leading_space=self.index)

    def to_string(self) -> str:
        series: Series = self.tr_series
        footer: str = self._get_footer()
        if len(series) == 0:
            return f'{type(self.series).__name__}([], {footer})'
        index: Index = series.index
        have_header: bool = _has_names(index)
        fmt_index: List[str]
        if isinstance(index, MultiIndex):
            fmt_index = index._format_multi(include_names=True, sparsify=None)
            adj: Any = printing.get_adjustment()
            fmt_index = adj.adjoin(2, *fmt_index).split('\n')
        else:
            fmt_index = index._format_flat(include_name=True)
        fmt_values: List[str] = self._get_formatted_values()
        if self.is_truncated_vertically:
            n_header_rows: int = 0
            row_num: int = cast(int, self.tr_row_num)
            width: int = self.adj.len(fmt_values[row_num - 1])
            dot_str: str
            if width > 3:
                dot_str = '...'
            else:
                dot_str = '..'
            dot_str = self.adj.justify([dot_str], width, mode='center')[0]
            fmt_values.insert(row_num + n_header_rows, dot_str)
            fmt_index.insert(row_num + 1, '')
        result: str
        if self.index:
            result = self.adj.adjoin(3, *[fmt_index[1:], fmt_values])
        else:
            result = self.adj.adjoin(3, fmt_values)
        if self.header and have_header:
            result = fmt_index[0] + '\n' + result
        if footer:
            result += '\n' + footer
        return str(''.join(result))

def get_dataframe_repr_params() -> Dict[str, Any]:
    """Get the parameters used to repr(dataFrame) calls using DataFrame.to_string.

    Supplying these parameters to DataFrame.to_string is equivalent to calling
    ``repr(DataFrame)``. This is useful if you want to adjust the repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame([[1, 2], [3, 4]])
    >>> repr_params = pd.io.formats.format.get_dataframe_repr_params()
    >>> repr(df) == df.to_string(**repr_params)
    True
    """
    from pandas.io.formats import console
    line_width: Optional[int]
    if get_option('display.expand_frame_repr'):
        (line_width, _) = console.get_console_size()
    else:
        line_width = None
    return {'max_rows': get_option('display.max_rows'), 'min_rows': get_option('display.min_rows'), 'max_cols': get_option('display.max_columns'), 'max_colwidth': get_option('display.max_colwidth'), 'show_dimensions': get_option('display.show_dimensions'), 'line_width': line_width}

def get_series_repr_params() -> Dict[str, Any]:
    """Get the parameters used to repr(Series) calls using Series.to_string.

    Supplying these parameters to Series.to_string is equivalent to calling
    ``repr(series)``. This is useful if you want to adjust the series repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> ser = pd.Series([1, 2, 3, 4])
    >>> repr_params = pd.io.formats.format.get_series_repr_params()
    >>> repr(ser) == ser.to_string(**repr_params)
    True
    """
    (width, height) = get_terminal_size()
    max_rows_opt: int = get_option('display.max_rows')
    max_rows: int = height if max_rows_opt == 0 else max_rows_opt
    min_rows: int = height if max_rows_opt == 0 else get_option('display.min_rows')
    return {'name': True, 'dtype': True, 'min_rows': min_rows, 'max_rows': max_rows, 'length': get_option('display.show_dimensions')}

class DataFrameFormatter:
    """
    Class for processing dataframe formatting options and data.

    Used by DataFrame.to_string, which backs DataFrame.__repr__.
    """
    __doc__ = __doc__ if __doc__ else ''
    __doc__ += common_docstring + return_docstring

    def __init__(self, frame: DataFrame, columns: Optional[Axes] = None, col_space: Optional[ColspaceArgType] = None, header: bool | SequenceNotStr[str] = True, index: bool = True, na_rep: str = 'NaN', formatters: Optional[FormattersType] = None, justify: Optional[str] = None, float_format: Optional[FloatFormatType] = None, sparsify: Optional[bool] = None, index_names: bool = True, max_rows: Optional[int] = None, min_rows: Optional[int] = None, max_cols: Optional[int] = None, show_dimensions: bool | str = False, decimal: str = '.', bold_rows: bool = False, escape: bool = True) -> None:
        self.frame: DataFrame = frame
        self.columns: Index = self._initialize_columns(columns)
        self.col_space: ColspaceType = self._initialize_colspace(col_space)
        self.header: bool | SequenceNotStr[str] = header
        self.index: bool = index
        self.na_rep: str = na_rep
        self.formatters: FormattersType = self._initialize_formatters(formatters)
        self.justify: str = self._initialize_justify(justify)
        self.float_format: Optional[FloatFormatType] = float_format
        self.sparsify: bool = self._initialize_sparsify(sparsify)
        self.show_index_names: bool = index_names
        self.decimal: str = decimal
        self.bold_rows: bool = bold_rows
        self.escape: bool = escape
        self.max_rows: Optional[int] = max_rows
        self.min_rows: Optional[int] = min_rows
        self.max_cols: Optional[int] = max_cols
        self.show_dimensions: bool | str = show_dimensions
        self.max_cols_fitted: Optional[int] = self._calc_max_cols_fitted()
        self.max_rows_fitted: Optional[int] = self._calc_max_rows_fitted()
        self.tr_frame: DataFrame = self.frame
        self.tr_col_num: int = 0
        self.tr_row_num: Optional[int] = None
        self.truncate()
        self.adj: Any = printing.get_adjustment()

    def get_strcols(self) -> List[List[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        strcols: List[List[str]] = self._get_strcols_without_index()
        if self.index:
            str_index: List[str] = self._get_formatted_index(self.tr_frame)
            strcols.insert(0, str_index)
        return strcols

    @property
    def should_show_dimensions(self) -> bool:
        return self.show_dimensions is True or (self.show_dimensions == 'truncate' and self.is_truncated)

    @property
    def is_truncated(self) -> bool:
        return bool(self.is_truncated_horizontally or self.is_truncated_vertically)

    @property
    def is_truncated_horizontally(self) -> bool:
        return bool(self.max_cols_fitted and len(self.columns) > self.max_cols_fitted)

    @property
    def is_truncated_vertically(self) -> bool:
        return bool(self.max_rows_fitted and len(self.frame) > self.max_rows_fitted)

    @property
    def dimensions_info(self) -> str:
        return f'\n\n[{len(self.frame)} rows x {len(self.frame.columns)} columns]'

    @property
    def has_index_names(self) -> bool:
        return _has_names(self.frame.index)

    @property
    def has_column_names(self) -> bool:
        return _has_names(self.frame.columns)

    @property
    def show_row_idx_names(self) -> bool:
        return all((self.has_index_names, self.index, self.show_index_names))

    @property
    def show_col_idx_names(self) -> bool:
        return all((self.has_column_names, self.show_index_names, self.header))

    @property
    def max_rows_displayed(self) -> int:
        return min(self.max_rows or len(self.frame), len(self.frame))

    def _initialize_sparsify(self, sparsify: Optional[bool]) -> bool:
        if sparsify is None:
            return get_option('display.multi_sparse')
        return cast(bool, sparsify)

    def _initialize_formatters(self, formatters: Optional[FormattersType]) -> FormattersType:
        if formatters is None:
            return {}
        elif len(self.frame.columns) == len(formatters) or isinstance(formatters, dict):
            return cast(FormattersType, formatters)
        else:
            raise ValueError(f'Formatters length({len(formatters)}) should match DataFrame number of columns({len(self.frame.columns)})')

    def _initialize_justify(self, justify: Optional[str]) -> str:
        if justify is None:
            return get_option('display.colheader_justify')
        else:
            return cast(str, justify)

    def _initialize_columns(self, columns: Optional[Axes]) -> Index:
        if columns is not