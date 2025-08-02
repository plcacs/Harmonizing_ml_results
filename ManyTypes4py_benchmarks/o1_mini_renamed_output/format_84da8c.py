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
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    cast,
    Callable,
    Dict,
    Generator,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import numpy as np
from pandas._config.config import get_option, set_option
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import NaT, Timedelta, Timestamp
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.dtypes.common import (
    is_complex_dtype,
    is_float,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import CategoricalDtype, DatetimeTZDtype, ExtensionDtype
from pandas.core.dtypes.missing import isna, notna
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    ExtensionArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import Index, MultiIndex, PeriodIndex, ensure_index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.io.common import check_parent_directory, stringify_path
from pandas.io.formats import printing

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Axes,
        ColspaceArgType,
        ColspaceType,
        CompressionOptions,
        FilePath,
        FloatFormatType,
        FormattersType,
        IndexLabel,
        SequenceNotStr,
        StorageOptions,
        WriteBuffer,
    )
    from pandas import DataFrame, Series

common_docstring: str = """
        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        columns : array-like, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : %(col_space_type)s, optional
            %(col_space)s.
        header : %(header_type)s, optional
            %(header)s.
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of ``NaN`` to use.
        formatters : list, tuple or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List/tuple must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. This function must return a unicode string and will be
            applied only to the non-``NaN`` elements, with ``NaN`` being
            handled by ``na_rep``.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
    """

VALID_JUSTIFY_PARAMETERS: Tuple[str, ...] = (
    'left',
    'right',
    'center',
    'justify',
    'justify-all',
    'start',
    'end',
    'inherit',
    'match-parent',
    'initial',
    'unset',
)

return_docstring: str = """
        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns
            None.
    """


class SeriesFormatter:
    """
    Implement the main logic of Series.to_string, which underlies
    Series.__repr__.
    """

    def __init__(
        self,
        series: Series,
        *,
        length: bool = True,
        header: bool = True,
        index: bool = True,
        na_rep: str = 'NaN',
        name: bool = False,
        float_format: Optional[FloatFormatType] = None,
        dtype: bool = True,
        max_rows: Optional[int] = None,
        min_rows: Optional[int] = None,
    ) -> None:
        self.series = series
        self.buf: StringIO = StringIO()
        self.name = name
        self.na_rep = na_rep
        self.header = header
        self.length = length
        self.index = index
        self.max_rows = max_rows
        self.min_rows = min_rows
        if float_format is None:
            float_format = get_option('display.float_format')
        self.float_format = float_format
        self.dtype = dtype
        self.adj = printing.get_adjustment()
        self._chk_truncate()

    def func_0gwmn6ms(self) -> None:
        min_rows = self.min_rows
        max_rows = self.max_rows
        is_truncated_vertically: bool = max_rows is not None and len(self.series) > max_rows
        series: Series = self.series
        if is_truncated_vertically:
            max_rows = cast(int, max_rows)
            if min_rows:
                max_rows = min(min_rows, max_rows)
            if max_rows == 1:
                row_num = max_rows
                series = series.iloc[:max_rows]
            else:
                row_num = max_rows // 2
                _len = len(series)
                _slice = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
                series = series.iloc[_slice]
            self.tr_row_num: Optional[int] = row_num
        else:
            self.tr_row_num = None
        self.tr_series: Series = series
        self.is_truncated_vertically: bool = is_truncated_vertically

    def func_cz838sg5(self) -> str:
        name = self.series.name
        footer = ''
        index = self.series.index
        if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)) and index.freq is not None:
            footer += f'Freq: {index.freqstr}'
        if self.name is not False and name is not None:
            if footer:
                footer += ', '
            series_name = printing.pprint_thing(name, escape_chars=('\t', '\r', '\n'))
            footer += f'Name: {series_name}'
        if (self.length is True or (self.length == 'truncate' and self.is_truncated_vertically)):
            if footer:
                footer += ', '
            footer += f'Length: {len(self.series)}'
        if self.dtype is not False and self.dtype is not None:
            dtype_name = getattr(self.tr_series.dtype, 'name', None)
            if dtype_name:
                if footer:
                    footer += ', '
                footer += f'dtype: {printing.pprint_thing(dtype_name)}'
        if isinstance(self.tr_series.dtype, CategoricalDtype):
            level_info = self.tr_series._values._get_repr_footer()
            if footer:
                footer += '\n'
            footer += level_info
        return str(footer)

    def func_5ezi3eu7(self) -> List[str]:
        return format_array(
            self.tr_series._values,
            None,
            float_format=self.float_format,
            na_rep=self.na_rep,
            leading_space=self.index,
        )

    def func_5yr670s4(self) -> str:
        series = self.tr_series
        footer = self._get_footer()
        if len(series) == 0:
            return f'{type(self.series).__name__}([], {footer})'
        index = series.index
        have_header = _has_names(index)
        if isinstance(index, MultiIndex):
            fmt_index = index._format_multi(include_names=True, sparsify=None)
            adj = printing.get_adjustment()
            fmt_index = adj.adjoin(2, *fmt_index).split('\n')
        else:
            fmt_index = index._format_flat(include_name=True)
        fmt_values = self._get_formatted_values()
        if self.is_truncated_vertically:
            n_header_rows = 0
            row_num = self.tr_row_num
            row_num = cast(int, row_num)
            width = self.adj.len(fmt_values[row_num - 1])
            if width > 3:
                dot_str = '...'
            else:
                dot_str = '..'
            dot_str = self.adj.justify([dot_str], width, mode='center')[0]
            fmt_values.insert(row_num + n_header_rows, dot_str)
            fmt_index.insert(row_num + 1, '')
        if self.index:
            result = self.adj.adjoin(3, *[fmt_index[1:], fmt_values])
        else:
            result = self.adj.adjoin(3, fmt_values)
        if self.header and have_header:
            result = fmt_index[0] + '\n' + result
        if footer:
            result += '\n' + footer
        return str(''.join(result))


def func_bjzg1kv7() -> Dict[str, Optional[int]]:
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
        line_width = None
    return {
        'max_rows': get_option('display.max_rows'),
        'min_rows': get_option('display.min_rows'),
        'max_cols': get_option('display.max_columns'),
        'max_colwidth': get_option('display.max_colwidth'),
        'show_dimensions': get_option('display.show_dimensions'),
        'line_width': line_width,
    }


def func_hjsfxo5e() -> Dict[str, Optional[int]]:
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
    max_rows_opt = get_option('display.max_rows')
    if max_rows_opt == 0:
        max_rows = height
    else:
        max_rows = max_rows_opt
    if max_rows_opt == 0:
        min_rows = height
    else:
        min_rows = get_option('display.min_rows')
    return {
        'name': True,
        'dtype': True,
        'min_rows': min_rows,
        'max_rows': max_rows,
        'length': get_option('display.show_dimensions'),
    }


class DataFrameFormatter:
    """
    Class for processing dataframe formatting options and data.

    Used by DataFrame.to_string, which backs DataFrame.__repr__.
    """
    __doc__ = __doc__ if __doc__ else ''
    __doc__ += common_docstring + return_docstring

    def __init__(
        self,
        frame: DataFrame,
        columns: Optional[Sequence[Hashable]] = None,
        col_space: Optional[ColspaceArgType] = None,
        header: bool = True,
        index: bool = True,
        na_rep: str = 'NaN',
        formatters: Optional[FormattersType] = None,
        justify: Optional[str] = None,
        float_format: Optional[FloatFormatType] = None,
        sparsify: Optional[bool] = None,
        index_names: bool = True,
        max_rows: Optional[int] = None,
        min_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        show_dimensions: bool = False,
        decimal: str = '.',
        bold_rows: bool = False,
        escape: bool = True,
    ) -> None:
        self.frame = frame
        self.columns: Index = self._initialize_columns(columns)
        self.col_space: Dict[Hashable, Union[int, str]] = self._initialize_colspace(col_space)
        self.header = header
        self.index = index
        self.na_rep = na_rep
        self.formatters: Union[List[Callable[[Any], str]], Tuple[Callable[[Any], str], ...], Dict[Any, Callable[[Any], str]]] = self._initialize_formatters(formatters)
        self.justify: Optional[str] = self._initialize_justify(justify)
        self.float_format: Optional[FloatFormatType] = float_format
        self.sparsify: Optional[bool] = self._initialize_sparsify(sparsify)
        self.show_index_names: bool = index_names
        self.decimal: str = decimal
        self.bold_rows: bool = bold_rows
        self.escape: bool = escape
        self.max_rows: Optional[int] = max_rows
        self.min_rows: Optional[int] = min_rows
        self.max_cols: Optional[int] = max_cols
        self.show_dimensions: bool = show_dimensions
        self.max_cols_fitted: Optional[int] = self._calc_max_cols_fitted()
        self.max_rows_fitted: Optional[int] = self._calc_max_rows_fitted()
        self.tr_frame: DataFrame = self.frame
        self.truncate()
        self.adj = printing.get_adjustment()

    def func_o1om5zuu(self) -> List[List[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        strcols: List[List[str]] = self._get_strcols_without_index()
        if self.index:
            str_index: List[str] = self._get_formatted_index(self.tr_frame)
            strcols.insert(0, str_index)
        return strcols

    @property
    def func_b8fn61bx(self) -> bool:
        return self.show_dimensions is True or (
            self.show_dimensions == 'truncate' and self.is_truncated
        )

    @property
    def func_ghxs29dn(self) -> bool:
        return bool(self.is_truncated_horizontally or self.is_truncated_vertically)

    @property
    def func_tidwiqa2(self) -> bool:
        return bool(self.max_cols_fitted is not None and len(self.columns) > self.max_cols_fitted)

    @property
    def func_iwd0l1t0(self) -> bool:
        return bool(self.max_rows_fitted is not None and len(self.frame) > self.max_rows_fitted)

    @property
    def func_y20x7w83(self) -> str:
        return f'\n\n[{len(self.frame)} rows x {len(self.frame.columns)} columns]'

    @property
    def func_ggzgxdi4(self) -> bool:
        return _has_names(self.frame.index)

    @property
    def func_jvfttnus(self) -> bool:
        return _has_names(self.frame.columns)

    @property
    def func_utzdrdkb(self) -> bool:
        return all((self.has_index_names, self.index, self.show_index_names))

    @property
    def func_yxv05zmy(self) -> bool:
        return all((self.has_column_names, self.show_index_names, self.header))

    @property
    def func_m2qxx6dz(self) -> int:
        return min(self.max_rows or len(self.frame), len(self.frame))

    def func_mrgzxv6a(self, sparsify: Optional[bool]) -> bool:
        if sparsify is None:
            return get_option('display.multi_sparse')
        return sparsify

    def func_8nodjc50(self, formatters: Optional[FormattersType]) -> Union[List[Callable[[Any], str]], Tuple[Callable[[Any], str], ...], Dict[Any, Callable[[Any], str]]]:
        if formatters is None:
            return {}
        elif len(self.frame.columns) == len(formatters) or isinstance(formatters, dict):
            return formatters
        else:
            raise ValueError(
                f'Formatters length({len(formatters)}) should match DataFrame number of columns({len(self.frame.columns)})'
            )

    def func_ax4d69wb(self, justify: Optional[str]) -> Optional[str]:
        if justify is None:
            return get_option('display.colheader_justify')
        else:
            return justify

    def func_v6cyufe1(self, columns: Optional[Sequence[Hashable]]) -> Index:
        if columns is not None:
            cols = ensure_index(columns)
            self.frame = self.frame[cols]
            return cols
        else:
            return self.frame.columns

    def func_hp1fgkky(self, col_space: Optional[ColspaceArgType]) -> Dict[Hashable, Union[int, str]]:
        if col_space is None:
            result: Dict[Hashable, Union[int, str]] = {}
        elif isinstance(col_space, (int, str)):
            result = {'': col_space}
            result.update({column: col_space for column in self.frame.columns})
        elif isinstance(col_space, Mapping):
            for column in col_space.keys():
                if column not in self.frame.columns and column != '':
                    raise ValueError(
                        f'Col_space is defined for an unknown column: {column}'
                    )
            result = col_space
        else:
            if len(self.frame.columns) != len(col_space):
                raise ValueError(
                    f'Col_space length({len(col_space)}) should match DataFrame number of columns({len(self.frame.columns)})'
                )
            result = dict(zip(self.frame.columns, col_space))
        return result

    def func_hwrgx2mp(self) -> Optional[int]:
        """Number of columns fitting the screen."""
        if not self._is_in_terminal():
            return self.max_cols
        width, _ = get_terminal_size()
        if self._is_screen_narrow(width):
            return width
        else:
            return self.max_cols

    def func_i1agfsgo(self) -> Optional[int]:
        """Number of rows with data fitting the screen."""
        if self._is_in_terminal():
            _, height = get_terminal_size()
            if self.max_rows == 0:
                return height - self._get_number_of_auxiliary_rows()
            if self._is_screen_short(height):
                max_rows = height
            else:
                max_rows = self.max_rows
        else:
            max_rows = self.max_rows
        return self._adjust_max_rows(max_rows)

    def func_gjofhusp(self, max_rows: Optional[int]) -> Optional[int]:
        """Adjust max_rows using display logic.

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options

        GH #37359
        """
        if max_rows:
            if len(self.frame) > max_rows and self.min_rows:
                max_rows = min(self.min_rows, max_rows)
        return max_rows

    def func_nwfsj2vz(self) -> bool:
        """Check if the output is to be shown in terminal."""
        return bool(self.max_cols == 0 or self.max_rows == 0)

    def func_poqep1o6(self, max_width: int) -> bool:
        return bool(self.max_cols == 0 and len(self.frame.columns) > max_width)

    def func_6a97hq7j(self, max_height: int) -> bool:
        return bool(self.max_rows == 0 and len(self.frame) > max_height)

    def func_59e6cwb6(self) -> int:
        """Get number of rows occupied by prompt, dots and dimension info."""
        dot_row = 1
        prompt_row = 1
        num_rows = dot_row + prompt_row
        if self.show_dimensions:
            num_rows += len(self.dimensions_info.splitlines())
        if self.header:
            num_rows += 1
        return num_rows

    def func_f1uz4zhc(self) -> None:
        """
        Check whether the frame should be truncated. If so, slice the frame up.
        """
        if self.is_truncated_horizontally:
            self._truncate_horizontally()
        if self.is_truncated_vertically:
            self._truncate_vertically()

    def func_xurv6etb(self) -> None:
        """Remove columns, which are not to be displayed and adjust formatters.

        Attributes affected:
            - tr_frame
            - formatters
            - tr_col_num
        """
        assert self.max_cols_fitted is not None
        col_num = self.max_cols_fitted // 2
        if col_num >= 1:
            _len = len(self.tr_frame.columns)
            _slice = np.hstack([np.arange(col_num), np.arange(_len - col_num, _len)])
            self.tr_frame = self.tr_frame.iloc[:, _slice]
            if isinstance(self.formatters, (list, tuple)):
                self.formatters = [*self.formatters[:col_num], *self.formatters[-col_num:]]
        else:
            col_num = cast(int, self.max_cols)
            self.tr_frame = self.tr_frame.iloc[:, :col_num]
        self.tr_col_num: int = col_num

    def func_5niwluec(self) -> None:
        """Remove rows, which are not to be displayed.

        Attributes affected:
            - tr_frame
            - tr_row_num
        """
        assert self.max_rows_fitted is not None
        row_num = self.max_rows_fitted // 2
        if row_num >= 1:
            _len = len(self.tr_frame)
            _slice = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
            self.tr_frame = self.tr_frame.iloc[_slice]
        else:
            row_num = cast(int, self.max_rows)
            self.tr_frame = self.tr_frame.iloc[:row_num, :]
        self.tr_row_num: int = row_num

    def func_mltaf07l(self) -> List[List[str]]:
        strcols: List[List[str]] = []
        if not is_list_like(self.header) and not self.header:
            for i, c in enumerate(self.tr_frame):
                fmt_values = self.format_col(i)
                fmt_values = _make_fixed_width(
                    strings=fmt_values,
                    justify=self.justify,
                    minimum=int(self.col_space.get(c, 0)),
                    adj=self.adj,
                )
                strcols.append(fmt_values)
            return strcols
        if is_list_like(self.header):
            self.header = cast(List[str], self.header)
            if len(self.header) != len(self.columns):
                raise ValueError(
                    f'Writing {len(self.columns)} cols but got {len(self.header)} aliases'
                )
            str_columns: List[List[str]] = [[label] for label in self.header]
        else:
            str_columns = self._get_formatted_column_labels(self.tr_frame)
        if self.show_row_idx_names:
            for x in str_columns:
                x.append('')
        for i, c in enumerate(self.tr_frame):
            cheader = str_columns[i]
            header_colwidth = max(
                int(self.col_space.get(c, 0)),
                *(self.adj.len(x) for x in cheader),
            )
            fmt_values = self.format_col(i)
            fmt_values = _make_fixed_width(
                fmt_values, self.justify, minimum=header_colwidth, adj=self.adj
            )
            max_len = max(
                *(self.adj.len(x) for x in fmt_values),
                header_colwidth,
            )
            cheader = self.adj.justify(cheader, max_len, mode=self.justify)
            strcols.append(cheader + fmt_values)
        return strcols

    def func_otvhkgm3(self, i: int) -> List[str]:
        frame = self.tr_frame
        formatter: Optional[Callable[[Any], str]] = self._get_formatter(i)
        return format_array(
            frame.iloc[:, i]._values,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            space=self.col_space.get(frame.columns[i]),
            decimal=self.decimal,
            leading_space=self.index,
        )

    def func_9g5icflh(self, i: int) -> Optional[Callable[[Any], str]]:
        if isinstance(self.formatters, (list, tuple)):
            if is_integer(i):
                i = cast(int, i)
                return self.formatters[i]
            else:
                return None
        else:
            if is_integer(i) and i not in self.columns:
                i = self.columns[i]
            return self.formatters.get(i, None)

    def func_j51egvu7(self, frame: DataFrame) -> List[List[str]]:
        from pandas.core.indexes.multi import sparsify_labels
        columns = frame.columns
        if isinstance(columns, MultiIndex):
            fmt_columns = columns._format_multi(sparsify=False, include_names=False)
            if self.sparsify and len(fmt_columns):
                fmt_columns = sparsify_labels(fmt_columns)
            str_columns: List[List[str]] = [list(x) for x in zip(*fmt_columns)]
        else:
            fmt_columns = columns._format_flat(include_name=False)
            str_columns = [
                [' ' + x if not self._get_formatter(i) and is_numeric_dtype(dtype) else x]
                for i, (x, dtype) in enumerate(zip(fmt_columns, self.frame.dtypes))
            ]
        return str_columns

    def func_qmll54ra(self, frame: DataFrame) -> List[List[str]]:
        col_space: Dict[Hashable, int] = {k: cast(int, v) for k, v in self.col_space.items()}
        index = frame.index
        columns = frame.columns
        fmt = self._get_formatter('__index__')
        if isinstance(index, MultiIndex):
            fmt_index = index._format_multi(
                sparsify=self.sparsify,
                include_names=self.show_row_idx_names,
                formatter=fmt
            )
        else:
            fmt_index = [index._format_flat(include_name=self.show_row_idx_names, formatter=fmt)]
        fmt_index = [
            tuple(
                _make_fixed_width(
                    list(x),
                    justify='left',
                    minimum=col_space.get('', 0),
                    adj=self.adj
                )
            ) for x in fmt_index
        ]
        adjoined = self.adj.adjoin(1, *fmt_index).split('\n')
        if self.show_col_idx_names:
            col_header = [str(x) for x in self._get_column_name_list()]
        else:
            col_header = [''] * columns.nlevels
        if self.header:
            return col_header + adjoined
        else:
            return adjoined

    def func_1ma8ly75(self) -> List[str]:
        names: List[str] = []
        columns = self.frame.columns
        if isinstance(columns, MultiIndex):
            names.extend('' if name is None else name for name in columns.names)
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
        self.fmt = fmt

    def func_dgwc1wwl(
        self,
        buf: Optional[Union[str, WriteBuffer[str]]] = None,
        encoding: Optional[str] = None,
        classes: Optional[Union[str, Sequence[str]]] = None,
        notebook: bool = False,
        border: Optional[Union[int, bool]] = None,
        table_id: Optional[str] = None,
        render_links: bool = False,
    ) -> Optional[str]:
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

        Klass: Callable[..., Any] = NotebookFormatter if notebook else HTMLFormatter
        html_formatter = Klass(
            self.fmt,
            classes=classes,
            border=border,
            table_id=table_id,
            render_links=render_links,
        )
        string: str = html_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def func_5yr670s4(
        self,
        buf: Optional[Union[str, WriteBuffer[str]]] = None,
        encoding: Optional[str] = None,
        line_width: Optional[int] = None,
    ) -> Optional[str]:
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

        string_formatter = StringFormatter(self.fmt, line_width=line_width)
        string = string_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def func_szac8qka(
        self,
        path_or_buf: Optional[Union[str, WriteBuffer[str]]] = None,
        encoding: Optional[str] = None,
        sep: str = ',',
        columns: Optional[Sequence[Hashable]] = None,
        index_label: Optional[Union[IndexLabel, Sequence[IndexLabel]]] = None,
        mode: str = 'w',
        compression: Optional[Union[str, CompressionOptions]] = 'infer',
        quoting: Optional[int] = None,
        quotechar: str = '"',
        lineterminator: Optional[str] = None,
        chunksize: Optional[int] = None,
        date_format: Optional[str] = None,
        doublequote: bool = True,
        escapechar: Optional[str] = None,
        errors: str = 'strict',
        storage_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Render dataframe as comma-separated file.
        """
        from pandas.io.formats.csvs import CSVFormatter

        if path_or_buf is None:
            created_buffer = True
            path_or_buf = StringIO()
        else:
            created_buffer = False
        csv_formatter = CSVFormatter(
            path_or_buf=path_or_buf,
            lineterminator=lineterminator,
            sep=sep,
            encoding=encoding,
            errors=errors,
            compression=compression,
            quoting=quoting,
            cols=columns,
            index_label=index_label,
            mode=mode,
            chunksize=chunksize,
            quotechar=quotechar,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            storage_options=storage_options,
            formatter=self.fmt,
        )
        csv_formatter.save()
        if created_buffer:
            assert isinstance(path_or_buf, StringIO)
            content: str = path_or_buf.getvalue()
            path_or_buf.close()
            return content
        return None


def func_9l51yfnw(string: str, buf: Optional[Union[str, WriteBuffer[str]]] = None, encoding: Optional[str] = None) -> Optional[str]:
    """
    Perform serialization. Write to buf or return as string if buf is None.
    """
    with _get_buffer(buf, encoding=encoding) as fd:
        fd.write(string)
        if buf is None:
            return fd.getvalue()
        return None


@contextmanager
def func_9oh5cun7(buf: Optional[Union[str, WriteBuffer[str]]], encoding: Optional[str] = None) -> Generator[WriteBuffer[str], None, None]:
    """
    Context manager to open, yield and close buffer for filenames or Path-like
    objects, otherwise yield buf unchanged.
    """
    if buf is not None:
        buf = stringify_path(buf)
    else:
        buf = StringIO()
    if encoding is None:
        encoding = 'utf-8'
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


def func_pb8pwp13(
    values: Union[np.ndarray, ExtensionArray],
    formatter: Optional[Callable[[Any], str]],
    float_format: Optional[FloatFormatType] = None,
    na_rep: str = 'NaN',
    digits: Optional[int] = None,
    space: Optional[int] = None,
    justify: str = 'right',
    decimal: str = '.',
    leading_space: bool = True,
    quoting: Optional[int] = None,
    fallback_formatter: Optional[Callable[[Any], str]] = None,
) -> List[str]:
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
        fmt_klass = _Datetime64Formatter
        values = cast(DatetimeArray, values)
    elif isinstance(values.dtype, DatetimeTZDtype):
        fmt_klass = _Datetime64TZFormatter
        values = cast(DatetimeArray, values)
    elif lib.is_np_dtype(values.dtype, 'm'):
        fmt_klass = _Timedelta64Formatter
        values = cast(TimedeltaArray, values)
    elif isinstance(values.dtype, ExtensionDtype):
        fmt_klass = _ExtensionArrayFormatter
    elif lib.is_np_dtype(values.dtype, 'fc'):
        fmt_klass = FloatArrayFormatter
    elif lib.is_np_dtype(values.dtype, 'iu'):
        fmt_klass = _IntArrayFormatter
    else:
        fmt_klass = _GenericArrayFormatter
    if space is None:
        space = 12
    if float_format is None:
        float_format = get_option('display.float_format')
    if digits is None:
        digits = get_option('display.precision')
    fmt_obj = fmt_klass(
        values,
        digits=digits,
        na_rep=na_rep,
        float_format=float_format,
        formatter=formatter,
        space=space,
        justify=justify,
        decimal=decimal,
        leading_space=leading_space,
        quoting=quoting,
        fallback_formatter=fallback_formatter,
    )
    return fmt_obj.get_result()


class _GenericArrayFormatter:
    def __init__(
        self,
        values: np.ndarray,
        digits: int = 7,
        formatter: Optional[Callable[[Any], str]] = None,
        na_rep: str = 'NaN',
        space: int = 12,
        float_format: Optional[FloatFormatType] = None,
        justify: str = 'right',
        decimal: str = '.',
        quoting: Optional[int] = None,
        fixed_width: bool = True,
        leading_space: bool = True,
        fallback_formatter: Optional[Callable[[Any], str]] = None,
    ) -> None:
        self.values = values
        self.digits = digits
        self.na_rep = na_rep
        self.space = space
        self.formatter = formatter
        self.float_format = float_format
        self.justify = justify
        self.decimal = decimal
        self.quoting = quoting
        self.fixed_width = fixed_width
        self.leading_space = leading_space
        self.fallback_formatter = fallback_formatter

    def func_bp9fyxu5(self) -> List[str]:
        fmt_values: List[str] = self._format_strings()
        return _make_fixed_width(fmt_values, self.justify)

    def func_ru5dduen(self) -> List[str]:
        if self.float_format is None:
            float_format = get_option('display.float_format')
            if float_format is None:
                precision = get_option('display.precision')
                float_format = lambda x: _trim_zeros_single_float(f'{x: .{precision:d}f}')
        else:
            float_format = self.float_format
        if self.formatter is not None:
            formatter = self.formatter
        elif self.fallback_formatter is not None:
            formatter = self.fallback_formatter
        else:
            quote_strings = (self.quoting is not None and self.quoting != QUOTE_NONE)
            formatter = partial(printing.pprint_thing, escape_chars=('\t', '\r', '\n'), quote_strings=quote_strings)

        def func_880s3b3j(x: Any) -> str:
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

        vals = self.values
        if not isinstance(vals, np.ndarray):
            raise TypeError('ExtensionArray formatting should use _ExtensionArrayFormatter')
        inferred = lib.map_infer(vals, is_float)
        is_float_type = inferred & np.all(~isna(vals), axis=tuple(range(1, len(vals.shape))))
        leading_space = self.leading_space
        if leading_space is None:
            leading_space = is_float_type.any()
        fmt_values: List[str] = []
        for i, v in enumerate(vals):
            if (not is_float_type[i] or self.formatter is not None) and leading_space:
                fmt_values.append(f' {func_880s3b3j(v)}')
            elif is_float_type[i]:
                fmt_values.append(float_format(v))
            else:
                if leading_space is False:
                    tpl = '{v}'
                else:
                    tpl = ' {v}'
                fmt_values.append(tpl.format(v=func_880s3b3j(v)))
        return fmt_values

    def func_bp9fyxu5(self) -> List[str]:
        fmt_values: List[str] = self._format_strings()
        return _make_fixed_width(fmt_values, self.justify)

    def get_result(self) -> List[str]:
        return self.func_ru5dduen()


class FloatArrayFormatter(_GenericArrayFormatter):
    def __init__(
        self,
        values: np.ndarray,
        digits: int = 7,
        formatter: Optional[Callable[[Any], str]] = None,
        na_rep: str = 'NaN',
        space: int = 12,
        float_format: Optional[FloatFormatType] = None,
        justify: str = 'right',
        decimal: str = '.',
        quoting: Optional[int] = None,
        fixed_width: bool = True,
        leading_space: bool = True,
        fallback_formatter: Optional[Callable[[Any], str]] = None,
    ) -> None:
        super().__init__(
            values,
            digits,
            formatter,
            na_rep,
            space,
            float_format,
            justify,
            decimal,
            quoting,
            fixed_width,
            leading_space,
            fallback_formatter,
        )
        if self.float_format is not None and self.formatter is None:
            self.fixed_width = False
            if callable(self.float_format):
                self.formatter = self.float_format
                self.float_format = None

    def func_xzb5g3g5(
        self,
        float_format: Optional[FloatFormatType] = None,
        threshold: Optional[float] = None,
    ) -> Callable[[Any], str]:
        """Returns a function to be applied on each value to format it"""
        if float_format is None:
            float_format = self.float_format
        if float_format:

            def func_lvxrror6(v: Any) -> str:
                assert float_format is not None
                return float_format(value=v) if notna(v) else self.na_rep
        else:

            def func_lvxrror6(v: Any) -> str:
                return str(v) if notna(v) else self.na_rep

        if self.decimal != '.':

            def func_3bcrpldi(v: Any) -> str:
                return func_lvxrror6(v).replace('.', self.decimal, 1)
        else:
            decimal_formatter = func_lvxrror6

        if threshold is None:
            return decimal_formatter

        def func_9bn2bjyh(value: Any) -> str:
            if notna(value):
                if abs(value) > threshold:
                    return func_3bcrpldi(value)
                else:
                    return func_3bcrpldi(0.0)
            else:
                return self.na_rep

        return func_9bn2bjyh

    def func_gnlz7b48(self) -> List[str]:
        """
        Returns the float values converted into strings using
        the parameters given at initialisation, as a numpy array
        """
        def func_4nyiqg6j(values: np.ndarray, formatter: Callable[[Any], str], na_rep: str) -> List[str]:
            mask = isna(values)
            formatted = [
                (func_9bn2bjyh(val) if not m else na_rep)
                for val, m in zip(values.ravel(), mask.ravel())
            ]
            return formatted

        def func_aobrursl(values: np.ndarray, formatter: Callable[[Any], str], na_rep: str) -> np.ndarray:
            real_values = np.real(values).ravel()
            imag_values = np.imag(values).ravel()
            real_mask, imag_mask = isna(real_values), isna(imag_values)
            formatted_lst: List[str] = []
            for val, real_val, imag_val, re_isna, im_isna in zip(
                values.ravel(), real_values, imag_values, real_mask, imag_mask
            ):
                if not re_isna and not im_isna:
                    formatted_lst.append(func_9bn2bjyh(val))
                elif not re_isna:
                    formatted_lst.append(f'{func_9bn2bjyh(real_val)}+{na_rep}j')
                elif not im_isna:
                    imag_formatted = func_9bn2bjyh(imag_val).strip()
                    if imag_formatted.startswith('-'):
                        formatted_lst.append(f'{na_rep}{imag_formatted}j')
                    else:
                        formatted_lst.append(f'{na_rep}+{imag_formatted}j')
                else:
                    formatted_lst.append(f'{na_rep}+{na_rep}j')
            return np.array(formatted_lst).reshape(values.shape)

        if self.formatter is not None:
            return func_4nyiqg6j(self.values, self.formatter, self.na_rep)
        if self.fixed_width:
            threshold = get_option('display.chop_threshold')
        else:
            threshold = None

        def func_73nce3in(float_format: Callable[[Any], str]) -> List[str]:
            formatter = self._value_formatter(float_format, threshold)
            na_rep = (' ' + self.na_rep if self.justify == 'left' else self.na_rep)
            values = self.values
            is_complex = is_complex_dtype(values)
            if is_complex:
                values = func_aobrursl(values, formatter, na_rep)
            else:
                values = func_4nyiqg6j(values, formatter, na_rep)
            if self.fixed_width:
                if is_complex:
                    result = _trim_zeros_complex(values, self.decimal)
                else:
                    result = _trim_zeros_float(values, self.decimal)
                return np.asarray(result, dtype='object')
            return values

        if self.float_format is None:
            if self.fixed_width:
                if self.leading_space is True:
                    fmt_str = '{value: .{digits:d}f}'
                else:
                    fmt_str = '{value:.{digits:d}f}'
                float_format = partial(fmt_str.format, digits=self.digits)
            else:
                float_format = self.float_format
        else:
            float_format = lambda value: self.float_format % value
        formatted_values = func_73nce3in(float_format)
        if not self.fixed_width:
            return formatted_values
        if len(formatted_values) > 0:
            maxlen = max(len(x) for x in formatted_values)
            too_long = maxlen > self.digits + 6
        else:
            too_long = False
        abs_vals = np.abs(self.values)
        has_large_values = (abs_vals > 1000000.0).any()
        has_small_values = ((abs_vals < 10 ** -self.digits) & (abs_vals > 0)).any()
        if has_small_values or (too_long and has_large_values):
            if self.leading_space is True:
                fmt_str = '{value: .{digits:d}e}'
            else:
                fmt_str = '{value:.{digits:d}e}'
            float_format = partial(fmt_str.format, digits=self.digits)
            formatted_values = func_73nce3in(float_format)
        return formatted_values

    def get_result_as_array(self) -> List[str]:
        return list(self.func_ru5dduen())


class _IntArrayFormatter(_GenericArrayFormatter):
    def func_ru5dduen(self) -> List[str]:
        if self.leading_space is False:
            formatter_str = lambda x: f'{x:d}'.format(x=x)
        else:
            formatter_str = lambda x: f'{x: d}'.format(x=x)
        formatter = self.formatter or formatter_str
        fmt_values: List[str] = [formatter(x) for x in self.values]
        return fmt_values


class _Datetime64Formatter(_GenericArrayFormatter):
    def __init__(
        self,
        values: np.ndarray,
        nat_rep: str = 'NaT',
        date_format: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep
        self.date_format = date_format

    def func_ru5dduen(self) -> List[str]:
        """we by definition have DO NOT have a TZ"""
        values = self.values
        if self.formatter is not None:
            return [self.formatter(x) for x in values]
        fmt_values: List[str] = values._format_native_types(
            na_rep=self.nat_rep, date_format=self.date_format
        )
        return fmt_values.tolist()


class _ExtensionArrayFormatter(_GenericArrayFormatter):
    def func_ru5dduen(self) -> List[str]:
        values = self.values
        formatter = self.formatter
        fallback_formatter = None
        if formatter is None:
            fallback_formatter = values._formatter(boxed=True)
        if isinstance(values, Categorical):
            array = values._internal_get_values()
        else:
            array = np.asarray(values, dtype=object)
        fmt_values = func_pb8pwp13(
            array,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            digits=self.digits,
            space=self.space,
            justify=self.justify,
            decimal=self.decimal,
            leading_space=self.leading_space,
            quoting=self.quoting,
            fallback_formatter=fallback_formatter,
        )
        return fmt_values


def func_du42aufb(percentiles: Iterable[float]) -> List[str]:
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
    percentiles = np.asarray(percentiles)
    if not is_numeric_dtype(percentiles) or not np.all(percentiles >= 0) or not np.all(percentiles <= 1):
        raise ValueError('percentiles should all be in the interval [0,1]')
    percentiles = 100 * percentiles
    prec = get_precision(percentiles)
    percentiles_round_type = percentiles.round(prec).astype(int)
    int_idx = np.isclose(percentiles_round_type, percentiles)
    if np.all(int_idx):
        out = percentiles_round_type.astype(str)
        return [(i + '%') for i in out]
    unique_pcts = np.unique(percentiles)
    prec = get_precision(unique_pcts)
    out: np.ndarray = np.empty_like(percentiles, dtype=object)
    out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)
    out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
    return [(i + '%') for i in out]


def func_lpneod21(array: Iterable[float]) -> int:
    to_begin = array[0] if array[0] > 0 else None
    to_end = 100 - array[-1] if array[-1] < 100 else None
    diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
    diff = abs(diff)
    prec = -int(np.floor(np.log10(np.min(diff))))
    prec = max(1, prec)
    return prec


def func_a3fvzihw(x: Any, nat_rep: str = 'NaT') -> str:
    if x is NaT:
        return nat_rep
    return str(x)


def func_nprjy8dg(x: Any, nat_rep: str = 'NaT', date_format: Optional[str] = None) -> str:
    if isinstance(x, NaTType):
        return nat_rep
    if date_format:
        return x.strftime(date_format)
    else:
        return x._date_repr


def func_3q5ocjsx(is_dates_only: bool, nat_rep: str = 'NaT', date_format: Optional[str] = None) -> Callable[[Any], str]:
    """Return a formatter callable taking a datetime64 as input and providing
    a string as output"""
    if is_dates_only:
        return lambda x: func_nprjy8dg(x, nat_rep=nat_rep, date_format=date_format)
    else:
        return lambda x: func_a3fvzihw(x, nat_rep=nat_rep)


class _Datetime64TZFormatter(_Datetime64Formatter):
    def func_ru5dduen(self) -> List[str]:
        """we by definition have a TZ"""
        ido = self.values._is_dates_only
        values = self.values.astype(object)
        formatter = self.formatter or func_3q5ocjsx(ido, date_format=self.date_format)
        fmt_values = [self.formatter(x) for x in values]
        return fmt_values


class _Timedelta64Formatter(_GenericArrayFormatter):
    def __init__(
        self,
        values: np.ndarray,
        nat_rep: str = 'NaT',
        **kwargs: Any
    ) -> None:
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep

    def func_ru5dduen(self) -> List[str]:
        formatter = self.formatter or get_format_timedelta64(
            self.values, nat_rep=self.nat_rep, box=False
        )
        return [formatter(x) for x in self.values]


def func_fkapyx2a(values: np.ndarray, nat_rep: str = 'NaT', box: bool = False) -> Callable[[Any], str]:
    """
    Return a formatter function for a range of timedeltas.
    These will all have the same format argument

    If box, then show the return in quotes
    """
    even_days = values._is_dates_only
    if even_days:
        format: Optional[str] = None
    else:
        format = 'long'

    def _formatter(x: Any) -> str:
        if x is None or (is_scalar(x) and isna(x)):
            return nat_rep
        if not isinstance(x, Timedelta):
            x = Timedelta(x)
        result = x._repr_base(format=format)
        if box:
            result = f"'{result}'"
        return result

    return _formatter


def func_5nbmjd8u(strings: List[str], justify: str = 'right', minimum: Optional[int] = None, adj: Optional[Any] = None) -> List[str]:
    if len(strings) == 0 or justify == 'all':
        return strings
    if adj is None:
        adjustment = printing.get_adjustment()
    else:
        adjustment = adj
    max_len = max(adjustment.len(x) for x in strings)
    if minimum is not None:
        max_len = max(minimum, max_len)
    conf_max = get_option('display.max_colwidth')
    if conf_max is not None and max_len > conf_max:
        max_len = conf_max

    def func_jep7se3p(x: str) -> str:
        if conf_max is not None:
            if (conf_max > 3) and (adjustment.len(x) > max_len):
                x = x[:max_len - 3] + '...'
        return x

    strings = [func_jep7se3p(x) for x in strings]
    result = adjustment.justify(strings, max_len, mode=justify)
    return result


def func_uwerv52s(str_complexes: List[str], decimal: str = '.') -> List[str]:
    """
    Separates the real and imaginary parts from the complex number, and
    executes the _trim_zeros_float method on each of those.
    """
    real_part: List[str] = []
    imag_part: List[str] = []
    for x in str_complexes:
        trimmed = re.split('(?<!e)([j+-])', x)
        real_part.append(''.join(trimmed[:-4]))
        imag_part.append(''.join(trimmed[-4:-2]))
    n = len(str_complexes)
    padded_parts = _trim_zeros_float(real_part + imag_part, decimal)
    if len(padded_parts) == 0:
        return []
    padded_length = max(len(part) for part in padded_parts) - 1
    padded = [
        real_pt + imag_pt[0] + f'{imag_pt[1:]:>{padded_length}}j'
        for real_pt, imag_pt in zip(padded_parts[:n], padded_parts[n:])
    ]
    return padded


def func_raykbgoj(str_float: str) -> str:
    """
    Trims trailing zeros after a decimal point,
    leaving just one if necessary.
    """
    str_float = str_float.rstrip('0')
    if str_float.endswith('.'):
        str_float += '0'
    return str_float


def func_9drd4gv6(str_floats: List[str], decimal: str = '.') -> List[str]:
    """
    Trims the maximum number of trailing zeros equally from
    all numbers containing decimals, leaving just one if
    necessary.
    """
    trimmed = str_floats.copy()
    number_regex = re.compile(f'^\\s*[\\+-]?[0-9]+\\{decimal}[0-9]*$')

    def func_d638a7or(x: str) -> bool:
        return re.match(number_regex, x) is not None

    def func_ojcit083(values: List[str]) -> bool:
        """
        Determine if an array of strings should be trimmed.

        Returns True if all numbers containing decimals (defined by the
        above regular expression) within the array end in a zero, otherwise
        returns False.
        """
        numbers = [x for x in values if func_d638a7or(x)]
        return len(numbers) > 0 and all(x.endswith('0') for x in numbers)

    while func_ojcit083(trimmed):
        trimmed = [(x[:-1] if func_d638a7or(x) else x) for x in trimmed]
    result = [
        (x + '0' if func_d638a7or(x) and x.endswith(decimal) else x)
        for x in trimmed
    ]
    return result


def func_ueab8sa1(index: Index) -> bool:
    if isinstance(index, MultiIndex):
        return com.any_not_none(*index.names)
    else:
        return index.name is not None


class EngFormatter:
    """
    Formats float values according to engineering format.

    Based on matplotlib.ticker.EngFormatter
    """
    ENG_PREFIXES: Final[Dict[int, str]] = {
        (-24): 'y',
        (-21): 'z',
        (-18): 'a',
        (-15): 'f',
        (-12): 'p',
        (-9): 'n',
        (-6): 'u',
        (-3): 'm',
        (0): '',
        (3): 'k',
        (6): 'M',
        (9): 'G',
        (12): 'T',
        (15): 'P',
        (18): 'E',
        (21): 'Z',
        (24): 'Y',
    }

    def __init__(self, accuracy: Optional[int] = None, use_eng_prefix: bool = False) -> None:
        self.accuracy = accuracy
        self.use_eng_prefix = use_eng_prefix

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
        dnum = Decimal(str(num))
        if dnum.is_nan():
            return 'NaN'
        if dnum.is_infinite():
            return 'inf'
        sign = 1
        if dnum < 0:
            sign = -1
            dnum = -dnum
        if dnum != 0:
            pow10 = Decimal(int(math.floor(dnum.log10() / 3) * 3))
        else:
            pow10 = Decimal(0)
        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
        int_pow10 = int(pow10)
        if self.use_eng_prefix:
            prefix = self.ENG_PREFIXES[int_pow10]
        elif int_pow10 < 0:
            prefix = f'E-{-int_pow10:02d}'
        else:
            prefix = f'E+{int_pow10:02d}'
        mant = sign * dnum / 10 ** pow10
        if self.accuracy is None:
            format_str = '{mant: g}{prefix}'
        else:
            format_str = f'{{mant: .{self.accuracy:d}f}}{{prefix}}'
        formatted = format_str.format(mant=mant, prefix=prefix)
        return formatted


def func_hz3nudet(accuracy: int = 3, use_eng_prefix: bool = False) -> None:
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


def func_dac6kc49(levels: List[List[Any]], sentinel: str = '') -> List[Dict[int, int]]:
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
    List[Dict[int, int]]
        Returns list of maps. For each level returns map of indexes (key is index
        in row and value is length of index).
    """
    if len(levels) == 0:
        return []
    control: List[bool] = [True] * len(levels[0])
    result: List[Dict[int, int]] = []
    for level in levels:
        last_index = 0
        lengths: Dict[int, int] = {}
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


def func_qq93mab9(buf: WriteBuffer[str], lines: Iterable[str]) -> None:
    """
    Appends lines to a buffer.

    Parameters
    ----------
    buf
        The buffer to write to
    lines
        The lines to append.
    """
    if any(isinstance(x, str) for x in lines):
        lines = [str(x) for x in lines]
    buf.write('\n'.join(lines))


def func_hz3nudet(accuracy: int = 3, use_eng_prefix: bool = False) -> None:
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
