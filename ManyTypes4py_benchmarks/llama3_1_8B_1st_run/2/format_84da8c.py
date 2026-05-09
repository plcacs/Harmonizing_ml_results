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
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from pandas._config.config import get_option, set_option
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import NaT, Timedelta, Timestamp
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.nattype import NaT
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

common_docstring = "\n        Parameters\n        ----------\n        buf : str, Path or StringIO-like, optional, default None\n            Buffer to write to. If None, the output is returned as a string.\n        columns : array-like, optional, default None\n            The subset of columns to write. Writes all columns by default.\n        col_space : %(col_space_type)s, optional\n            %(col_space)s.\n        header : %(header_type)s, optional\n            %(header)s.\n        index : bool, optional, default True\n            Whether to print index (row) labels.\n        na_rep : str, optional, default 'NaN'\n            String representation of ``NaN`` to use.\n        formatters : list, tuple or dict of one-param. functions, optional\n            Formatter functions to apply to columns' elements by position or\n            name.\n            The result of each function must be a unicode string.\n            List/tuple must be of length equal to the number of columns.\n        float_format : one-parameter function, optional, default None\n            Formatter function to apply to columns' elements if they are\n            floats. This function must return a unicode string and will be\n            applied only to the non-``NaN`` elements, with ``NaN`` being\n            handled by ``na_rep``.\n        sparsify : bool, optional, default True\n            Set to False for a DataFrame with a hierarchical index to print\n            every multiindex key at each row.\n        index_names : bool, optional, default True\n            Prints the names of the indexes.\n        justify : str, default None\n            How to justify the column labels. If None uses the option from\n            the print configuration (controlled by set_option), 'right' out\n            of the box. Valid values are\n\n            * left\n            * right\n            * center\n            * justify\n            * justify-all\n            * start\n            * end\n            * inherit\n            * match-parent\n            * initial\n            * unset.\n        max_rows : int, optional\n            Maximum number of rows to display in the console.\n        max_cols : int, optional\n            Maximum number of columns to display in the console.\n        show_dimensions : bool, default False\n            Display DataFrame dimensions (number of rows by number of columns).\n        decimal : str, default '.'\n            Character recognized as decimal separator, e.g. ',' in Europe.\n    "
VALID_JUSTIFY_PARAMETERS = ('left', 'right', 'center', 'justify', 'justify-all', 'start', 'end', 'inherit', 'match-parent', 'initial', 'unset')
return_docstring = '\n        Returns\n        -------\n        str or None\n            If buf is None, returns the result as a string. Otherwise returns\n            None.\n    '

class SeriesFormatter:
    """
    Implement the main logic of Series.to_string, which underlies
    Series.__repr__.
    """

    def __init__(self, series: Series, *, length: bool = True, header: bool = True, index: bool = True, na_rep: str = 'NaN', name: bool = False, float_format: FloatFormatType = None, dtype: bool = True, max_rows: int | None = None, min_rows: int | None = None) -> None:
        self.series: Series = series
        self.buf: StringIO = StringIO()
        self.name: bool = name
        self.na_rep: str = na_rep
        self.header: bool = header
        self.length: bool = length
        self.index: bool = index
        self.max_rows: int | None = max_rows
        self.min_rows: int | None = min_rows
        if float_format is None:
            float_format: FloatFormatType = get_option('display.float_format')
        self.float_format: FloatFormatType = float_format
        self.dtype: bool = dtype
        self.adj: printing.Adjustment = printing.get_adjustment()
        self._chk_truncate()

    def _chk_truncate(self) -> None:
        min_rows: int | None = self.min_rows
        max_rows: int | None = self.max_rows
        is_truncated_vertically: bool = max_rows and len(self.series) > max_rows
        series: Series = self.series
        if is_truncated_vertically:
            max_rows: int = cast(int, max_rows)
            if min_rows:
                max_rows = min(min_rows, max_rows)
            if max_rows == 1:
                row_num: int = max_rows
                series = series.iloc[:max_rows]
            else:
                row_num: int = max_rows // 2
                _len: int = len(series)
                _slice: np.ndarray = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
                series = series.iloc[_slice]
            self.tr_row_num: int = row_num
        else:
            self.tr_row_num: int | None = None
        self.tr_series: Series = series
        self.is_truncated_vertically: bool = is_truncated_vertically

    def _get_footer(self) -> str:
        name: str | None = self.series.name
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
            dtype_name: str | None = getattr(self.tr_series.dtype, 'name', None)
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

    def _get_formatted_values(self) -> list[str]:
        return format_array(self.tr_series._values, None, float_format=self.float_format, na_rep=self.na_rep, leading_space=self.index)

    def to_string(self) -> str:
        series: Series = self.tr_series
        footer: str = self._get_footer()
        if len(series) == 0:
            return f'{type(self.series).__name__}([], {footer})'
        index: Index = series.index
        have_header: bool = _has_names(index)
        if isinstance(index, MultiIndex):
            fmt_index: list[str] = index._format_multi(include_names=True, sparsify=None)
            adj: printing.Adjustment = printing.get_adjustment()
            fmt_index: list[str] = adj.adjoin(2, *fmt_index).split('\n')
        else:
            fmt_index: list[str] = index._format_flat(include_name=True)
        fmt_values: list[str] = self._get_formatted_values()
        if self.is_truncated_vertically:
            n_header_rows: int = 0
            row_num: int = self.tr_row_num
            row_num: int = cast(int, row_num)
            width: int = self.adj.len(fmt_values[row_num - 1])
            if width > 3:
                dot_str: str = '...'
            else:
                dot_str: str = '..'
            dot_str: str = self.adj.justify([dot_str], width, mode='center')[0]
            fmt_values.insert(row_num + n_header_rows, dot_str)
            fmt_index.insert(row_num + 1, '')
        if self.index:
            result: str = self.adj.adjoin(3, *[fmt_index[1:], fmt_values])
        else:
            result: str = self.adj.adjoin(3, fmt_values)
        if self.header and have_header:
            result: str = fmt_index[0] + '\n' + result
        if footer:
            result += '\n' + footer
        return str(''.join(result))

def get_dataframe_repr_params() -> dict[str, Any]:
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
    if get_option('display.expand_frame_repr'):
        line_width, _ = console.get_console_size()
    else:
        line_width: int | None = None
    return {'max_rows': get_option('display.max_rows'), 'min_rows': get_option('display.min_rows'), 'max_cols': get_option('display.max_columns'), 'max_colwidth': get_option('display.max_colwidth'), 'show_dimensions': get_option('display.show_dimensions'), 'line_width': line_width}

def get_series_repr_params() -> dict[str, Any]:
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
    width, height = get_terminal_size()
    max_rows_opt: int = get_option('display.max_rows')
    max_rows: int = height if max_rows_opt == 0 else max_rows_opt
    min_rows: int = height if max_rows_opt == 0 else get_option('display.min_rows')
    return {'name': True, 'dtype': True, 'min_rows': min_rows, 'max_rows': max_rows, 'length': get_option('display.show_dimensions')}

class DataFrameFormatter:
    """
    Class for processing dataframe formatting options and data.

    Used by DataFrame.to_string, which backs DataFrame.__repr__.
    """

    def __init__(self, frame: DataFrame, columns: Sequence[str] | None = None, col_space: ColspaceType | None = None, header: bool = True, index: bool = True, na_rep: str = 'NaN', formatters: FormattersType | None = None, justify: str | None = None, float_format: FloatFormatType | None = None, sparsify: bool | None = None, index_names: bool = True, max_rows: int | None = None, min_rows: int | None = None, max_cols: int | None = None, show_dimensions: bool = False, decimal: str = '.', bold_rows: bool = False, escape: bool = True) -> None:
        self.frame: DataFrame = frame
        self.columns: Index = self._initialize_columns(columns)
        self.col_space: ColspaceType = self._initialize_colspace(col_space)
        self.header: bool = header
        self.index: bool = index
        self.na_rep: str = na_rep
        self.formatters: FormattersType = self._initialize_formatters(formatters)
        self.justify: str = self._initialize_justify(justify)
        self.float_format: FloatFormatType | None = float_format
        self.sparsify: bool = self._initialize_sparsify(sparsify)
        self.show_index_names: bool = index_names
        self.decimal: str = decimal
        self.bold_rows: bool = bold_rows
        self.escape: bool = escape
        self.max_rows: int | None = max_rows
        self.min_rows: int | None = min_rows
        self.max_cols: int | None = max_cols
        self.show_dimensions: bool = show_dimensions
        self.max_cols_fitted: int = self._calc_max_cols_fitted()
        self.max_rows_fitted: int = self._calc_max_rows_fitted()
        self.tr_frame: DataFrame = self.frame
        self.truncate()
        self.adj: printing.Adjustment = printing.get_adjustment()

    def get_strcols(self) -> list[list[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        strcols: list[list[str]] = self._get_strcols_without_index()
        if self.index:
            str_index: list[str] = self._get_formatted_index(self.tr_frame)
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

    def _initialize_sparsify(self, sparsify: bool | None) -> bool:
        if sparsify is None:
            return get_option('display.multi_sparse')
        return sparsify

    def _initialize_formatters(self, formatters: FormattersType | None) -> FormattersType:
        if formatters is None:
            return {}
        elif len(self.frame.columns) == len(formatters) or isinstance(formatters, dict):
            return formatters
        else:
            raise ValueError(f'Formatters length({len(formatters)}) should match DataFrame number of columns({len(self.frame.columns)})')

    def _initialize_justify(self, justify: str | None) -> str:
        if justify is None:
            return get_option('display.colheader_justify')
        else:
            return justify

    def _initialize_columns(self, columns: Sequence[str] | None) -> Index:
        if columns is not None:
            cols: Index = ensure_index(columns)
            self.frame = self.frame[cols]
            return cols
        else:
            return self.frame.columns

    def _initialize_colspace(self, col_space: ColspaceType | None) -> ColspaceType:
        if col_space is None:
            result: ColspaceType = {}
        elif isinstance(col_space, (int, str)):
            result: ColspaceType = {'': col_space}
            result.update({column: col_space for column in self.frame.columns})
        elif isinstance(col_space, Mapping):
            for column in col_space.keys():
                if column not in self.frame.columns and column != '':
                    raise ValueError(f'Col_space is defined for an unknown column: {column}')
            result: ColspaceType = col_space
        else:
            if len(self.frame.columns) != len(col_space):
                raise ValueError(f'Col_space length({len(col_space)}) should match DataFrame number of columns({len(self.frame.columns)})')
            result: ColspaceType = dict(zip(self.frame.columns, col_space))
        return result

    def _calc_max_cols_fitted(self) -> int:
        """Number of columns fitting the screen."""
        if not self._is_in_terminal():
            return self.max_cols
        width, _ = get_terminal_size()
        if self._is_screen_narrow(width):
            return width
        else:
            return self.max_cols

    def _calc_max_rows_fitted(self) -> int:
        """Number of rows with data fitting the screen."""
        if self._is_in_terminal():
            _, height = get_terminal_size()
            if self.max_rows == 0:
                return height - self._get_number_of_auxiliary_rows()
            if self._is_screen_short(height):
                max_rows: int = height
            else:
                max_rows: int = self.max_rows
        else:
            max_rows: int = self.max_rows
        return self._adjust_max_rows(max_rows)

    def _adjust_max_rows(self, max_rows: int) -> int:
        """Adjust max_rows using display logic.

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options

        GH #37359
        """
        if max_rows:
            if len(self.frame) > max_rows and self.min_rows:
                max_rows = min(self.min_rows, max_rows)
        return max_rows

    def _is_in_terminal(self) -> bool:
        """Check if the output is to be shown in terminal."""
        return bool(self.max_cols == 0 or self.max_rows == 0)

    def _is_screen_narrow(self, max_width: int) -> bool:
        return bool(self.max_cols == 0 and len(self.frame.columns) > max_width)

    def _is_screen_short(self, max_height: int) -> bool:
        return bool(self.max_rows == 0 and len(self.frame) > max_height)

    def _get_number_of_auxiliary_rows(self) -> int:
        """Get number of rows occupied by prompt, dots and dimension info."""
        dot_row: int = 1
        prompt_row: int = 1
        num_rows: int = dot_row + prompt_row
        if self.show_dimensions:
            num_rows += len(self.dimensions_info.splitlines())
        if self.header:
            num_rows += 1
        return num_rows

    def truncate(self) -> None:
        """
        Check whether the frame should be truncated. If so, slice the frame up.
        """
        if self.is_truncated_horizontally:
            self._truncate_horizontally()
        if self.is_truncated_vertically:
            self._truncate_vertically()

    def _truncate_horizontally(self) -> None:
        """Remove columns, which are not to be displayed and adjust formatters.

        Attributes affected:
            - tr_frame
            - formatters
            - tr_col_num
        """
        assert self.max_cols_fitted is not None
        col_num: int = self.max_cols_fitted // 2
        if col_num >= 1:
            _len: int = len(self.tr_frame.columns)
            _slice: np.ndarray = np.hstack([np.arange(col_num), np.arange(_len - col_num, _len)])
            self.tr_frame = self.tr_frame.iloc[:, _slice]
            if isinstance(self.formatters, (list, tuple)):
                self.formatters = [*self.formatters[:col_num], *self.formatters[-col_num:]]
        else:
            col_num: int = cast(int, self.max_cols)
            self.tr_frame = self.tr_frame.iloc[:, :col_num]
        self.tr_col_num: int = col_num

    def _truncate_vertically(self) -> None:
        """Remove rows, which are not to be displayed.

        Attributes affected:
            - tr_frame
            - tr_row_num
        """
        assert self.max_rows_fitted is not None
        row_num: int = self.max_rows_fitted // 2
        if row_num >= 1:
            _len: int = len(self.tr_frame)
            _slice: np.ndarray = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
            self.tr_frame = self.tr_frame.iloc[_slice]
        else:
            row_num: int = cast(int, self.max_rows)
            self.tr_frame = self.tr_frame.iloc[:row_num, :]
        self.tr_row_num: int = row_num

    def _get_strcols_without_index(self) -> list[list[str]]:
        strcols: list[list[str]] = []
        if not is_list_like(self.header) and (not self.header):
            for i, c in enumerate(self.tr_frame):
                fmt_values: list[str] = self.format_col(i)
                fmt_values: list[str] = _make_fixed_width(strings=fmt_values, justify=self.justify, minimum=int(self.col_space.get(c, 0)), adj=self.adj)
                strcols.append(fmt_values)
            return strcols
        if is_list_like(self.header):
            self.header: list[str] = cast(list[str], self.header)
            if len(self.header) != len(self.columns):
                raise ValueError(f'Writing {len(self.columns)} cols but got {len(self.header)} aliases')
            str_columns: list[list[str]] = [[label] for label in self.header]
        else:
            str_columns: list[list[str]] = self._get_formatted_column_labels(self.tr_frame)
        if self.show_row_idx_names:
            for x in str_columns:
                x.append('')
        for i, c in enumerate(self.tr_frame):
            cheader: list[str] = str_columns[i]
            header_colwidth: int = max(int(self.col_space.get(c, 0)), *(self.adj.len(x) for x in cheader))
            fmt_values: list[str] = self.format_col(i)
            fmt_values: list[str] = _make_fixed_width(fmt_values, self.justify, minimum=header_colwidth, adj=self.adj)
            max_len: int = max(*(self.adj.len(x) for x in fmt_values), header_colwidth)
            cheader: list[str] = self.adj.justify(cheader, max_len, mode=self.justify)
            strcols.append(cheader + fmt_values)
        return strcols

    def format_col(self, i: int) -> list[str]:
        frame: DataFrame = self.tr_frame
        formatter: Callable[[Any], str] | None = self._get_formatter(i)
        return format_array(frame.iloc[:, i]._values, formatter, float_format=self.float_format, na_rep=self.na_rep, space=self.col_space.get(frame.columns[i]), decimal=self.decimal, leading_space=self.index)

    def _get_formatter(self, i: int) -> Callable[[Any], str] | None:
        if isinstance(self.formatters, (list, tuple)):
            if is_integer(i):
                i: int = cast(int, i)
                return self.formatters[i]
            else:
                return None
        else:
            if is_integer(i) and i not in self.columns:
                i: str = self.columns[i]
            return self.formatters.get(i, None)

    def _get_formatted_column_labels(self, frame: DataFrame) -> list[list[str]]:
        from pandas.core.indexes.multi import sparsify_labels
        columns: Index = frame.columns
        if isinstance(columns, MultiIndex):
            fmt_columns: list[list[str]] = columns._format_multi(sparsify=False, include_names=False)
            if self.sparsify and len(fmt_columns):
                fmt_columns: list[list[str]] = sparsify_labels(fmt_columns)
            str_columns: list[list[str]] = [list(x) for x in zip(*fmt_columns)]
        else:
            fmt_columns: list[str] = columns._format_flat(include_name=False)
            str_columns: list[list[str]] = [[' ' + x if not self._get_formatter(i) and is_numeric_dtype(dtype) else x] for i, (x, dtype) in enumerate(zip(fmt_columns, self.frame.dtypes))]
        return str_columns

    def _get_formatted_index(self, frame: DataFrame) -> list[str]:
        col_space: dict[str, int] = {k: cast(int, v) for k, v in self.col_space.items()}
        index: Index = frame.index
        columns: Index = frame.columns
        fmt: Callable[[Any], str] | None = self._get_formatter('__index__')
        if isinstance(index, MultiIndex):
            fmt_index: list[str] = index._format_multi(sparsify=self.sparsify, include_names=self.show_row_idx_names, formatter=fmt)
        else:
            fmt_index: list[str] = [index._format_flat(include_name=self.show_row_idx_names, formatter=fmt)]
        fmt_index: list[str] = [tuple(_make_fixed_width(list(x), justify='left', minimum=col_space.get('', 0), adj=self.adj)) for x in fmt_index]
        adjoined: str = self.adj.adjoin(1, *fmt_index).split('\n')
        if self.show_col_idx_names:
            col_header: list[str] = [str(x) for x in self._get_column_name_list()]
        else:
            col_header: list[str] = [''] * columns.nlevels
        if self.header:
            return col_header + adjoined
        else:
            return adjoined

    def _get_column_name_list(self) -> list[str]:
        names: list[str] = []
        columns: Index = self.frame.columns
        if isinstance(columns, MultiIndex):
            names.extend(('' if name is None else name for name in columns.names))
        else:
            names.append('' if columns.name is None else columns.name)
        return names

class DataFrameRenderer:
    """Class for creating dataframe output in multiple formats.

    Called in pandas.core.generic.NDFrame:
        - to_csv
        - to_latex

    Called in pandas.DataFrame:
        - to_html
        - to_string

    Parameters
    ----------
    fmt : DataFrameFormatter
        Formatter with the formatting options.
    """

    def __init__(self, fmt: DataFrameFormatter) -> None:
        self.fmt: DataFrameFormatter = fmt

    def to_html(self, buf: str | Path | StringIO | None, encoding: str | None, classes: str | list[str] | None, notebook: bool, border: int | bool, table_id: str | None, render_links: bool) -> str:
        """
        Render a DataFrame to a html table.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding : str, default “utf-8”
            Set character encoding.
        classes : str or list-like
            classes to include in the `class` attribute of the opening
            ``<table>`` tag, in addition to the default "dataframe".
        notebook : {True, False}, optional, default False
            Whether the generated HTML is for IPython Notebook.
        border : int or bool
            When an integer value is provided, it sets the border attribute in
            the opening tag, specifying the thickness of the border.
            If ``False`` or ``0`` is passed, the border attribute will not
            be present in the ``<table>`` tag.
            The default value for this parameter is governed by
            ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        """
        from pandas.io.formats.html import HTMLFormatter, NotebookFormatter
        Klass: type[HTMLFormatter] = NotebookFormatter if notebook else HTMLFormatter
        html_formatter: HTMLFormatter = Klass(self.fmt, classes=classes, border=border, table_id=table_id, render_links=render_links)
        string: str = html_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_string(self, buf: str | Path | StringIO | None, encoding: str | None, line_width: int | None) -> str:
        """
        Render a DataFrame to a console-friendly tabular output.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding: str, default “utf-8”
            Set character encoding.
        line_width : int, optional
            Width to wrap a line in characters.
        """
        from pandas.io.formats.string import StringFormatter
        string_formatter: StringFormatter = StringFormatter(self.fmt, line_width=line_width)
        string: str = string_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_csv(self, path_or_buf: str | Path | StringIO | None, encoding: str | None, sep: str, columns: Sequence[str] | None, index_label: str | None, mode: str, compression: CompressionOptions, quoting: int, quotechar: str, lineterminator: str | None, chunksize: int | None, date_format: str | None, doublequote: bool, escapechar: str | None, errors: str, storage_options: StorageOptions | None) -> None:
        """
        Render dataframe as comma-separated file.
        """
        from pandas.io.formats.csvs import CSVFormatter
        if path_or_buf is None:
            created_buffer: bool = True
            path_or_buf: StringIO = StringIO()
        else:
            created_buffer: bool = False
        csv_formatter: CSVFormatter = CSVFormatter(path_or_buf=path_or_buf, lineterminator=lineterminator, sep=sep, encoding=encoding, errors=errors, compression=compression, quoting=quoting, cols=columns, index_label=index_label, mode=mode, chunksize=chunksize, quotechar=quotechar, date_format=date_format, doublequote=doublequote, escapechar=escapechar, storage_options=storage_options, formatter=self.fmt)
        csv_formatter.save()
        if created_buffer:
            assert isinstance(path_or_buf, StringIO)
            content: str = path_or_buf.getvalue()
            path_or_buf.close()
            return content
        return None

def save_to_buffer(string: str, buf: str | Path | StringIO | None, encoding: str | None) -> str | None:
    """
    Perform serialization. Write to buf or return as string if buf is None.
    """
    with _get_buffer(buf, encoding=encoding) as fd:
        fd.write(string)
        if buf is None:
            return fd.getvalue()
        return None

@contextmanager
def _get_buffer(buf: str | Path | StringIO | None, encoding: str | None) -> StringIO | file:
    """
    Context manager to open, yield and close buffer for filenames or Path-like
    objects, otherwise yield buf unchanged.
    """
    if buf is not None:
        buf: str = stringify_path(buf)
    else:
        buf: StringIO = StringIO()
    if encoding is None:
        encoding: str = 'utf-8'
    elif not isinstance(buf, str):
        raise ValueError('buf is not a file name and encoding is specified.')
    if hasattr(buf, 'write'):
        yield buf
    elif isinstance(buf, str):
        check_parent_directory(str(buf))
        with open(buf, 'w', encoding=encoding, newline='') as f:
            yield f
    else:
        raise TypeError('buf is not a file name and it has no write method')

def format_array(values: np.ndarray | ExtensionArray, formatter: Callable[[Any], str] | None, float_format: FloatFormatType | None, na_rep: str, digits: int | None, space: int | None, justify: str, decimal: str, leading_space: bool, quoting: int | None, fallback_formatter: Callable[[Any], str] | None) -> list[str]:
    """
    Format an array for printing.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    formatter
    float_format
    na_rep
    digits
    space
    justify
    decimal
    leading_space : bool, optional, default True
        Whether the array should be formatted with a leading space.
        When an array as a column of a Series or DataFrame, we do want
        the leading space to pad between columns.

        When formatting an Index subclass
        (e.g. IntervalIndex._get_values_for_csv), we don't want the
        leading space since it should be left-aligned.
    fallback_formatter

    Returns
    -------
    List[str]
    """
    if lib.is_np_dtype(values.dtype, 'M'):
        fmt_klass: type[_Datetime64Formatter] = _Datetime64Formatter
        values: DatetimeArray = cast(DatetimeArray, values)
    elif isinstance(values.dtype, DatetimeTZDtype):
        fmt_klass: type[_Datetime64TZFormatter] = _Datetime64TZFormatter
        values: DatetimeArray = cast(DatetimeArray, values)
    elif lib.is_np_dtype(values.dtype, 'm'):
        fmt_klass: type[_Timedelta64Formatter] = _Timedelta64Formatter
        values: TimedeltaArray = cast(TimedeltaArray, values)
    elif isinstance(values.dtype, ExtensionDtype):
        fmt_klass: type[_ExtensionArrayFormatter] = _ExtensionArrayFormatter
    elif lib.is_np_dtype(values.dtype, 'fc'):
        fmt_klass: type[FloatArrayFormatter] = FloatArrayFormatter
    elif lib.is_np_dtype(values.dtype, 'iu'):
        fmt_klass: type[_IntArrayFormatter] = _IntArrayFormatter
    else:
        fmt_klass: type[_GenericArrayFormatter] = _GenericArrayFormatter
    if space is None:
        space: int = 12
    if float_format is None:
        float_format: FloatFormatType = get_option('display.float_format')
    if digits is None:
        digits: int = get_option('display.precision')
    fmt_obj: _GenericArrayFormatter = fmt_klass(values, digits=digits, na_rep=na_rep, float_format=float_format, formatter=formatter, space=space, justify=justify, decimal=decimal, leading_space=leading_space, quoting=quoting, fallback_formatter=fallback_formatter)
    return fmt_obj.get_result()

class _GenericArrayFormatter:
    def __init__(self, values: np.ndarray | ExtensionArray, digits: int, formatter: Callable[[Any], str] | None, na_rep: str, space: int, float_format: FloatFormatType | None, justify: str, decimal: str, quoting: int | None, fixed_width: bool, leading_space: bool, fallback_formatter: Callable[[Any], str] | None) -> None:
        self.values: np.ndarray | ExtensionArray = values
        self.digits: int = digits
        self.na_rep: str = na_rep
        self.space: int = space
        self.formatter: Callable[[Any], str] | None = formatter
        self.float_format: FloatFormatType | None = float_format
        self.justify: str = justify
        self.decimal: str = decimal
        self.quoting: int | None = quoting
        self.fixed_width: bool = fixed_width
        self.leading_space: bool = leading_space
        self.fallback_formatter: Callable[[Any], str] | None = fallback_formatter

    def get_result(self) -> list[str]:
        fmt_values: list[str] = self._format_strings()
        return _make_fixed_width(fmt_values, self.justify)

    def _format_strings(self) -> list[str]:
        if self.float_format is None:
            float_format: FloatFormatType = get_option('display.float_format')
            if float_format is None:
                precision: int = get_option('display.precision')
                float_format: FloatFormatType = lambda x: _trim_zeros_single_float(f'{x: .{precision:d}f}')
        else:
            float_format: FloatFormatType = self.float_format
        if self.formatter is not None:
            formatter: Callable[[Any], str] = self.formatter
        elif self.fallback_formatter is not None:
            formatter: Callable[[Any], str] = self.fallback_formatter
        else:
            quote_strings: bool = self.quoting is not None and self.quoting != QUOTE_NONE
            formatter: Callable[[Any], str] = partial(printing.pprint_thing, escape_chars=('\t', '\r', '\n'), quote_strings=quote_strings)

        def _format(x: Any) -> str:
            if self.na_rep is not None and is_scalar(x) and isna(x):
                if x is None:
                    return 'None'
                elif x is NA:
                    return str(NA)
                elif x is NaT or isinstance(x, (np.datetime64, np.timedelta64)):
                    return 'NaT'
                return self.na_rep
            elif isinstance(x, PandasObject):
                return str(x)
            elif isinstance(x, StringDtype):
                return repr(x)
            else:
                return str(formatter(x))
        vals: np.ndarray | ExtensionArray = self.values
        if not isinstance(vals, np.ndarray):
            raise TypeError('ExtensionArray formatting should use _ExtensionArrayFormatter')
        inferred: np.ndarray = lib.map_infer(vals, is_float)
        is_float_type: np.ndarray = inferred & np.all(notna(vals), axis=tuple(range(1, len(vals.shape))))
        leading_space: bool = self.leading_space
        if leading_space is None:
            leading_space: bool = is_float_type.any()
        fmt_values: list[str] = []
        for i, v in enumerate(vals):
            if (not is_float_type[i] or self.formatter is not None) and leading_space:
                fmt_values.append(f' {_format(v)}')
            elif is_float_type[i]:
                fmt_values.append(float_format(v))
            else:
                if leading_space is False:
                    tpl: str = '{v}'
                else:
                    tpl: str = ' {v}'
                fmt_values.append(tpl.format(v=_format(v)))
        return fmt_values

class FloatArrayFormatter(_GenericArrayFormatter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.float_format is not None and self.formatter is None:
            self.fixed_width: bool = False
            if callable(self.float_format):
                self.formatter: Callable[[Any], str] = self.float_format
                self.float_format: FloatFormatType | None = None

    def _value_formatter(self, float_format: FloatFormatType | None, threshold: float | None) -> Callable[[Any], str]:
        """Returns a function to be applied on each value to format it"""
        if float_format is None:
            float_format: FloatFormatType = self.float_format
        if float_format:

            def base_formatter(v: Any) -> str:
                assert float_format is not None
                return float_format(value=v) if notna(v) else self.na_rep
        else:

            def base_formatter(v: Any) -> str:
                return str(v) if notna(v) else self.na_rep
        if self.decimal != '.':

            def decimal_formatter(v: Any) -> str:
                return base_formatter(v).replace('.', self.decimal, 1)
        else:
            decimal_formatter: Callable[[Any], str] = base_formatter
        if threshold is None:
            return decimal_formatter

        def formatter(value: Any) -> str:
            if notna(value):
                if abs(value) > threshold:
                    return decimal_formatter(value)
                else:
                    return decimal_formatter(0.0)
            else:
                return self.na_rep
        return formatter

    def get_result_as_array(self) -> np.ndarray:
        """
        Returns the float values converted into strings using
        the parameters given at initialisation, as a numpy array
        """

        def format_with_na_rep(values: np.ndarray, formatter: Callable[[Any], str], na_rep: str) -> np.ndarray:
            mask: np.ndarray = isna(values)
            formatted: np.ndarray = np.array([formatter(val) if not m else na_rep for val, m in zip(values.ravel(), mask.ravel())]).reshape(values.shape)
            return formatted

        def format_complex_with_na_rep(values: np.ndarray, formatter: Callable[[Any], str], na_rep: str) -> np.ndarray:
            real_values: np.ndarray = np.real(values).ravel()
            imag_values: np.ndarray = np.imag(values).ravel()
            real_mask, imag_mask: tuple[np.ndarray] = (isna(real_values), isna(imag_values))
            formatted_lst: list[str] = []
            for val, real_val, imag_val, re_isna, im_isna in zip(values.ravel(), real_values, imag_values, real_mask, imag_mask):
                if not re_isna and (not im_isna):
                    formatted_lst.append(formatter(val))
                elif not re_isna:
                    formatted_lst.append(f'{formatter(real_val)}+{na_rep}j')
                elif not im_isna:
                    imag_formatted: str = formatter(imag_val).strip()
                    if imag_formatted.startswith('-'):
                        formatted_lst.append(f'{na_rep}{imag_formatted}j')
                    else:
                        formatted_lst.append(f'{na_rep}+{imag_formatted}j')
                else:
                    formatted_lst.append(f'{na_rep}+{na_rep}j')
            return np.array(formatted_lst).reshape(values.shape)
        if self.formatter is not None:
            return format_with_na_rep(self.values, self.formatter, self.na_rep)
        if self.fixed_width:
            threshold: float | None = get_option('display.chop_threshold')
        else:
            threshold: float | None = None

        def format_values_with(float_format: FloatFormatType) -> np.ndarray:
            formatter: Callable[[Any], str] = self._value_formatter(float_format, threshold)
            na_rep: str = ' ' + self.na_rep if self.justify == 'left' else self.na_rep
            values: np.ndarray = self.values
            is_complex: bool = is_complex_dtype(values)
            if is_complex:
                values: np.ndarray = format_complex_with_na_rep(values, formatter, na_rep)
            else:
                values: np.ndarray = format_with_na_rep(values, formatter, na_rep)
            if self.fixed_width:
                if is_complex:
                    result: np.ndarray = _trim_zeros_complex(values, self.decimal)
                else:
                    result: np.ndarray = _trim_zeros_float(values, self.decimal)
                return np.asarray(result, dtype='object')
            return values
        if self.float_format is None:
            if self.fixed_width:
                if self.leading_space is True:
                    fmt_str: str = '{value: .{digits:d}f}'
                else:
                    fmt_str: str = '{value:.{digits:d}f}'
                float_format: FloatFormatType = partial(fmt_str.format, digits=self.digits)
            else:
                float_format: FloatFormatType = self.float_format
        else:
            float_format: FloatFormatType = lambda value: self.float_format % value
        formatted_values: np.ndarray = format_values_with(float_format)
        if not self.fixed_width:
            return formatted_values
        if len(formatted_values) > 0:
            maxlen: int = max((len(x) for x in formatted_values))
            too_long: bool = maxlen > self.digits + 6
        else:
            too_long: bool = False
        abs_vals: np.ndarray = np.abs(self.values)
        has_large_values: bool = (abs_vals > 1000000.0).any()
        has_small_values: bool = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
        if has_small_values or (too_long and has_large_values):
            if self.leading_space is True:
                fmt_str: str = '{value: .{digits:d}e}'
            else:
                fmt_str: str = '{value:.{digits:d}e}'
            float_format: FloatFormatType = partial(fmt_str.format, digits=self.digits)
            formatted_values: np.ndarray = format_values_with(float_format)
        return formatted_values

    def _format_strings(self) -> list[str]:
        return list(self.get_result_as_array())

class _IntArrayFormatter(_GenericArrayFormatter):
    def _format_strings(self) -> list[str]:
        if self.leading_space is False:
            formatter_str: Callable[[Any], str] = lambda x: f'{x:d}'.format(x=x)
        else:
            formatter_str: Callable[[Any], str] = lambda x: f'{x: d}'.format(x=x)
        formatter: Callable[[Any], str] = self.formatter or formatter_str
        fmt_values: list[str] = [formatter(x) for x in self.values]
        return fmt_values

class _Datetime64Formatter(_GenericArrayFormatter):
    def __init__(self, values: np.ndarray | ExtensionArray, nat_rep: str = 'NaT', date_format: str | None = None, **kwargs) -> None:
        super().__init__(values, **kwargs)
        self.nat_rep: str = nat_rep
        self.date_format: str | None = date_format

    def _format_strings(self) -> list[str]:
        """we by definition have DO NOT have a TZ"""
        values: np.ndarray | ExtensionArray = self.values
        if self.formatter is not None:
            return [self.formatter(x) for x in values]
        fmt_values: list[str] = values._format_native_types(na_rep=self.nat_rep, date_format=self.date_format)
        return fmt_values.tolist()

class _ExtensionArrayFormatter(_GenericArrayFormatter):
    def _format_strings(self) -> list[str]:
        values: np.ndarray | ExtensionArray = self.values
        formatter: Callable[[Any], str] | None = self.formatter
        fallback_formatter: Callable[[Any], str] | None = None
        if formatter is None:
            fallback_formatter: Callable[[Any], str] | None = values._formatter(boxed=True)
        if isinstance(values, Categorical):
            array: np.ndarray = values._internal_get_values()
        else:
            array: np.ndarray = np.asarray(values, dtype=object)
        fmt_values: list[str] = format_array(array, formatter, float_format=self.float_format, na_rep=self.na_rep, digits=self.digits, space=self.space, justify=self.justify, decimal=self.decimal, leading_space=self.leading_space, quoting=self.quoting, fallback_formatter=fallback_formatter)
        return fmt_values

def format_percentiles(percentiles: list[float]) -> list[str]:
    """
    Outputs rounded and formatted percentiles.

    Parameters
    ----------
    percentiles : list-like, containing floats from interval [0,1]

    Returns
    -------
    formatted : list of strings

    Notes
    -----
    Rounding precision is chosen so that: (1) if any two elements of
    ``percentiles`` differ, they remain different after rounding
    (2) no entry is *rounded* to 0% or 100%.
    Any non-integer is always rounded to at least 1 decimal place.

    Examples
    --------
    Keeps all entries different after rounding:

    >>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
    ['1.999%', '2.001%', '50%', '66.667%', '99.99%']

    No element is rounded to 0% or 100% (unless already equal to it).
    Duplicates are allowed:

    >>> format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])
    ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']
    """
    percentiles: np.ndarray = np.asarray(percentiles)
    if not is_numeric_dtype(percentiles) or not np.all(percentiles >= 0) or (not np.all(percentiles <= 1)):
        raise ValueError('percentiles should all be in the interval [0,1]')
    percentiles: np.ndarray = 100 * percentiles
    prec: int = get_precision(percentiles)
    percentiles_round_type: np.ndarray = percentiles.round(prec).astype(int)
    int_idx: np.ndarray = np.isclose(percentiles_round_type, percentiles)
    if np.all(int_idx):
        out: np.ndarray = percentiles_round_type.astype(str)
        return [i + '%' for i in out]
    unique_pcts: np.ndarray = np.unique(percentiles)
    prec: int = get_precision(unique_pcts)
    out: np.ndarray = np.empty_like(percentiles, dtype=object)
    out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)
    out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
    return [i + '%' for i in out]

def get_precision(array: np.ndarray) -> int:
    to_begin: float | None = array[0] if array[0] > 0 else None
    to_end: float | None = 100 - array[-1] if array[-1] < 100 else None
    diff: np.ndarray = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
    diff: np.ndarray = abs(diff)
    prec: int = -np.floor(np.log10(np.min(diff))).astype(int)
    prec: int = max(1, prec)
    return prec

def _format_datetime64(x: np.datetime64, nat_rep: str = 'NaT') -> str:
    if x is NaT:
        return nat_rep
    return str(x)

def _format_datetime64_dateonly(x: np.datetime64, nat_rep: str = 'NaT', date_format: str | None = None) -> str:
    if isinstance(x, NaTType):
        return nat_rep
    if date_format:
        return x.strftime(date_format)
    else:
        return x._date_repr

def get_format_datetime64(is_dates_only: bool, nat_rep: str = 'NaT', date_format: str | None = None) -> Callable[[np.datetime64], str]:
    """Return a formatter callable taking a datetime64 as input and providing
    a string as output"""
    if is_dates_only:
        return lambda x: _format_datetime64_dateonly(x, nat_rep=nat_rep, date_format=date_format)
    else:
        return lambda x: _format_datetime64(x, nat_rep=nat_rep)

class _Datetime64TZFormatter(_Datetime64Formatter):
    def _format_strings(self) -> list[str]:
        """we by definition have a TZ"""
        ido: bool = self.values._is_dates_only
        values: np.ndarray | ExtensionArray = self.values.astype(object)
        formatter: Callable[[Any], str] = self.formatter or get_format_datetime64(ido, date_format=self.date_format)
        fmt_values: list[str] = [formatter(x) for x in values]
        return fmt_values

class _Timedelta64Formatter(_GenericArrayFormatter):
    def __init__(self, values: np.ndarray | ExtensionArray, nat_rep: str = 'NaT', **kwargs) -> None:
        super().__init__(values, **kwargs)
        self.nat_rep: str = nat_rep

    def _format_strings(self) -> list[str]:
        formatter: Callable[[Any], str] = self.formatter or get_format_timedelta64(self.values, nat_rep=self.nat_rep, box=False)
        return [formatter(x) for x in self.values]

def get_format_timedelta64(values: np.ndarray | ExtensionArray, nat_rep: str = 'NaT', box: bool = False) -> Callable[[Any], str]:
    """
    Return a formatter function for a range of timedeltas.
    These will all have the same format argument

    If box, then show the return in quotes
    """
    even_days: bool = values._is_dates_only
    if even_days:
        format: str | None = None
    else:
        format: str = 'long'

    def _formatter(x: Any) -> str:
        if x is None or (is_scalar(x) and isna(x)):
            return nat_rep
        if not isinstance(x, Timedelta):
            x: Timedelta = Timedelta(x)
        result: str = x._repr_base(format=format)
        if box:
            result: str = f"'{result}'"
        return result
    return _formatter

def _make_fixed_width(strings: list[str], justify: str, minimum: int | None, adj: printing.Adjustment | None) -> list[str]:
    if len(strings) == 0 or justify == 'all':
        return strings
    if adj is None:
        adjustment: printing.Adjustment = printing.get_adjustment()
    else:
        adjustment: printing.Adjustment = adj
    max_len: int = max((adjustment.len(x) for x in strings))
    if minimum is not None:
        max_len: int = max(minimum, max_len)
    conf_max: int | None = get_option('display.max_colwidth')
    if conf_max is not None and max_len > conf_max:
        max_len: int = conf_max

    def just(x: str) -> str:
        if conf_max is not None:
            if (conf_max > 3) & (adjustment.len(x) > max_len):
                x: str = x[:max_len - 3] + '...'
        return x
    strings: list[str] = [just(x) for x in strings]
    result: list[str] = adjustment.justify(strings, max_len, mode=justify)
    return result

def _trim_zeros_complex(str_complexes: list[str], decimal: str = '.') -> list[str]:
    """
    Separates the real and imaginary parts from the complex number, and
    executes the _trim_zeros_float method on each of those.
    """
    real_part: list[str] = []
    imag_part: list[str] = []
    for x in str_complexes:
        trimmed: list[str] = re.split('(?<!e)([j+-])', x)
        real_part.append(''.join(trimmed[:-4]))
        imag_part.append(''.join(trimmed[-4:-2]))
    n: int = len(str_complexes)
    padded_parts: list[str] = _trim_zeros_float(real_part + imag_part, decimal)
    if len(padded_parts) == 0:
        return []
    padded_length: int = max((len(part) for part in padded_parts)) - 1
    padded: list[str] = [real_pt + imag_pt[0] + f'{imag_pt[1:]:>{padded_length}}' + 'j' for real_pt, imag_pt in zip(padded_parts[:n], padded_parts[n:])]
    return padded

def _trim_zeros_single_float(str_float: str) -> str:
    """
    Trims trailing zeros after a decimal point,
    leaving just one if necessary.
    """
    str_float: str = str_float.rstrip('0')
    if str_float.endswith('.'):
        str_float += '0'
    return str_float

def _trim_zeros_float(str_floats: list[str], decimal: str = '.') -> list[str]:
    """
    Trims the maximum number of trailing zeros equally from
    all numbers containing decimals, leaving just one if
    necessary.
    """
    trimmed: list[str] = str_floats
    number_regex: re.Pattern[str] = re.compile(f'^\\s*[\\+-]?[0-9]+\\{decimal}[0-9]*$')

    def is_number_with_decimal(x: str) -> bool:
        return re.match(number_regex, x) is not None

    def should_trim(values: list[str]) -> bool:
        """
        Determine if an array of strings should be trimmed.

        Returns True if all numbers containing decimals (defined by the
        above regular expression) within the array end in a zero, otherwise
        returns False.
        """
        numbers: list[str] = [x for x in values if is_number_with_decimal(x)]
        return len(numbers) > 0 and all((x.endswith('0') for x in numbers))
    while should_trim(trimmed):
        trimmed: list[str] = [x[:-1] if is_number_with_decimal(x) else x for x in trimmed]
    result: list[str] = [x + '0' if is_number_with_decimal(x) and x.endswith(decimal) else x for x in trimmed]
    return result

def _has_names(index: Index) -> bool:
    if isinstance(index, MultiIndex):
        return com.any_not_none(*index.names)
    else:
        return index.name is not None

class EngFormatter:
    """
    Formats float values according to engineering format.

    Based on matplotlib.ticker.EngFormatter
    """
    ENG_PREFIXES: dict[int, str] = {-24: 'y', -21: 'z', -18: 'a', -15: 'f', -12: 'p', -9: 'n', -6: 'u', -3: 'm', 0: '', 3: 'k', 6: 'M', 9: 'G', 12: 'T', 15: 'P', 18: 'E', 21: 'Z', 24: 'Y'}

    def __init__(self, accuracy: int | None, use_eng_prefix: bool) -> None:
        self.accuracy: int | None = accuracy
        self.use_eng_prefix: bool = use_eng_prefix

    def __call__(self, num: Any) -> str:
        """
        Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:
        >>> format_eng = EngFormatter(accuracy=0, use_eng_prefix=True)
        >>> format_eng(0)
        ' 0'
        >>> format_eng = EngFormatter(accuracy=1, use_eng_prefix=True)
        >>> format_eng(1_000_000)
        ' 1.0M'
        >>> format_eng = EngFormatter(accuracy=2, use_eng_prefix=False)
        >>> format_eng("-1e-6")
        '-1.00E-06'

        @param num: the value to represent
        @type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        @return: engineering formatted string
        """
        dnum: Decimal = Decimal(str(num))
        if Decimal.is_nan(dnum):
            return 'NaN'
        if Decimal.is_infinite(dnum):
            return 'inf'
        sign: int = 1
        if dnum < 0:
            sign: int = -1
            dnum: Decimal = -dnum
        if dnum != 0:
            pow10: Decimal = Decimal(int(math.floor(dnum.log10() / 3) * 3))
        else:
            pow10: Decimal = Decimal(0)
        pow10: Decimal = pow10.min(max(self.ENG_PREFIXES.keys()))
        pow10: Decimal = pow10.max(min(self.ENG_PREFIXES.keys()))
        int_pow10: int = int(pow10)
        if self.use_eng_prefix:
            prefix: str = self.ENG_PREFIXES[int_pow10]
        elif int_pow10 < 0:
            prefix: str = f'E-{-int_pow10:02d}'
        else:
            prefix: str = f'E+{int_pow10:02d}'
        mant: Decimal = sign * dnum / 10 ** pow10
        if self.accuracy is None:
            format_str: str = '{mant: g}{prefix}'
        else:
            format_str: str = f'{{mant: .{self.accuracy:d}f}}{{prefix}}'
        formatted: str = format_str.format(mant=mant, prefix=prefix)
        return formatted

def set_eng_float_format(accuracy: int | None = 3, use_eng_prefix: bool = False) -> None:
    """
    Format float representation in DataFrame with SI notation.

    Sets the floating-point display format for ``DataFrame`` objects using engineering
    notation (SI units), allowing easier readability of values across wide ranges.

    Parameters
    ----------
    accuracy : int, default 3
        Number of decimal digits after the floating point.
    use_eng_prefix : bool, default False
        Whether to represent a value with SI prefixes.

    Returns
    -------
    None
        This method does not return a value. it updates the global display format
        for floats in DataFrames.

    See Also
    --------
    set_option : Set the value of the specified option or options.
    reset_option : Reset one or more options to their default value.

    Examples
    --------
    >>> df = pd.DataFrame([1e-9, 1e-3, 1, 1e3, 1e6])
    >>> df
                  0
    0  1.000000e-09
    1  1.000000e-03
    2  1.000000e+00
    3  1.000000e+03
    4  1.000000e+06

    >>> pd.set_eng_float_format(accuracy=1)
    >>> df
             0
    0  1.0E-09
    1  1.0E-03
    2  1.0E+00
    3  1.0E+03
    4  1.0E+06

    >>> pd.set_eng_float_format(use_eng_prefix=True)
    >>> df
            0
    0  1.000n
    1  1.000m
    2   1.000
    3  1.000k
    4  1.000M

    >>> pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)
    >>> df
          0
    0  1.0n
    1  1.0m
    2   1.0
    3  1.0k
    4  1.0M

    >>> pd.set_option("display.float_format", None)  # unset option
    """
    set_option('display.float_format', EngFormatter(accuracy, use_eng_prefix))

def get_level_lengths(levels: list[list[str]], sentinel: str = '') -> list[dict[int, int]]:
    """
    For each index in each level the function returns lengths of indexes.

    Parameters
    ----------
    levels : list of lists
        List of values on for level.
    sentinel : string, optional
        Value which states that no new index starts on there.

    Returns
    -------
    Returns list of maps. For each level returns map of indexes (key is index
    in row and value is length of index).
    """
    if len(levels) == 0:
        return []
    control: list[bool] = [True] * len(levels[0])
    result: list[dict[int, int]] = []
    for level in levels:
        last_index: int = 0
        lengths: dict[int, int] = {}
        for i, key in enumerate(level):
            if control[i] and key == sentinel:
                pass
            else:
                control[i] = False
                lengths[last_index] = i - last_index
                last_index = i
        lengths[last_index] = len(level) - last_index
        result.append(lengths)
    return result

def buffer_put_lines(buf: StringIO | file, lines: list[str]) -> None:
    """
    Appends lines to a buffer.

    Parameters
    ----------
    buf
        The buffer to write to
    lines
        The lines to append.
    """
    if any((isinstance(x, str) for x in lines)):
        lines: list[str] = [str(x) for x in lines]
    buf.write('\n'.join(lines))
