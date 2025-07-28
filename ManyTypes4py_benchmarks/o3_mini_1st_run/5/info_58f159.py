from __future__ import annotations
from abc import ABC, abstractmethod
import sys
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Optional, Union, List, Mapping, Sequence, Iterator, Tuple
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pandas._typing import Dtype, WriteBuffer
    from pandas import DataFrame, Index, Series

frame_max_cols_sub: str = dedent(
    '    max_cols : int, optional\n'
    '        When to switch from the verbose to the truncated output. If the\n'
    '        DataFrame has more than `max_cols` columns, the truncated output\n'
    '        is used. By default, the setting in\n'
    '        ``pandas.options.display.max_info_columns`` is used.'
)
show_counts_sub: str = dedent(
    '    show_counts : bool, optional\n'
    '        Whether to show the non-null counts. By default, this is shown\n'
    '        only if the DataFrame is smaller than\n'
    '        ``pandas.options.display.max_info_rows`` and\n'
    '        ``pandas.options.display.max_info_columns``. A value of True always\n'
    '        shows the counts, and False never shows the counts.'
)
frame_examples_sub: str = dedent(
    '    >>> int_values = [1, 2, 3, 4, 5]\n'
    '    >>> text_values = [\'alpha\', \'beta\', \'gamma\', \'delta\', \'epsilon\']\n'
    '    >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]\n'
    '    >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,\n'
    '    ...                   "float_col": float_values})\n'
    '    >>> df\n'
    '        int_col text_col  float_col\n'
    '    0        1    alpha       0.00\n'
    '    1        2     beta       0.25\n'
    '    2        3    gamma       0.50\n'
    '    3        4    delta       0.75\n'
    '    4        5  epsilon       1.00\n'
    '\n'
    '    Prints information of all columns:\n'
    '\n'
    '    >>> df.info(verbose=True)\n'
    '    <class \'pandas.DataFrame\'>\n'
    '    RangeIndex: 5 entries, 0 to 4\n'
    '    Data columns (total 3 columns):\n'
    '     #   Column     Non-Null Count  Dtype\n'
    '    ---  ------     --------------  -----\n'
    '     0   int_col    5 non-null      int64\n'
    '     1   text_col   5 non-null      object\n'
    '     2   float_col  5 non-null      float64\n'
    '    dtypes: float64(1), int64(1), object(1)\n'
    '    memory usage: 248.0+ bytes\n'
    '\n'
    '    Prints a summary of columns count and its dtypes but not per column\n'
    '    information:\n'
    '\n'
    '    >>> df.info(verbose=False)\n'
    '    <class \'pandas.DataFrame\'>\n'
    '    RangeIndex: 5 entries, 0 to 4\n'
    '    Columns: 3 entries, int_col to float_col\n'
    '    dtypes: float64(1), int64(1), object(1)\n'
    '    memory usage: 248.0+ bytes\n'
    '\n'
    '    Pipe output of DataFrame.info to buffer instead of sys.stdout, get\n'
    '    buffer content and writes to a text file:\n'
    '\n'
    '    >>> import io\n'
    '    >>> buffer = io.StringIO()\n'
    '    >>> df.info(buf=buffer)\n'
    '    >>> s = buffer.getvalue()\n'
    '    >>> with open("df_info.txt", "w",\n'
    '    ...           encoding="utf-8") as f:  # doctest: +SKIP\n'
    '    ...     f.write(s)\n'
    '    260\n'
    '\n'
    '    The `memory_usage` parameter allows deep introspection mode, specially\n'
    '    useful for big DataFrames and fine-tune memory optimization:\n'
    '\n'
    '    >>> random_strings_array = np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)\n'
    '    >>> df = pd.DataFrame({\n'
    '    ...     \'column_1\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6),\n'
    '    ...     \'column_2\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6),\n'
    '    ...     \'column_3\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)\n'
    '    ... })\n'
    '    >>> df.info()\n'
    '    <class \'pandas.DataFrame\'>\n'
    '    RangeIndex: 1000000 entries, 0 to 999999\n'
    '    Data columns (total 3 columns):\n'
    '     #   Column    Non-Null Count    Dtype\n'
    '    ---  ------    --------------    -----\n'
    '     0   column_1  1000000 non-null  object\n'
    '     1   column_2  1000000 non-null  object\n'
    '     2   column_3  1000000 non-null  object\n'
    '    dtypes: object(3)\n'
    '    memory usage: 22.9+ MB\n'
    '\n'
    '    >>> df.info(memory_usage=\'deep\')\n'
    '    <class \'pandas.DataFrame\'>\n'
    '    RangeIndex: 1000000 entries, 0 to 999999\n'
    '    Data columns (total 3 columns):\n'
    '     #   Column    Non-Null Count    Dtype\n'
    '    ---  ------    --------------    -----\n'
    '     0   column_1  1000000 non-null  object\n'
    '     1   column_2  1000000 non-null  object\n'
    '     2   column_3  1000000 non-null  object\n'
    '    dtypes: object(3)\n'
    '    memory usage: 165.9 MB'
)
frame_see_also_sub: str = dedent(
    '    DataFrame.describe: Generate descriptive statistics of DataFrame\n'
    '        columns.\n'
    '    DataFrame.memory_usage: Memory usage of DataFrame columns.'
)
frame_sub_kwargs: dict[str, Any] = {
    'klass': 'DataFrame',
    'type_sub': ' and columns',
    'max_cols_sub': frame_max_cols_sub,
    'show_counts_sub': show_counts_sub,
    'examples_sub': frame_examples_sub,
    'see_also_sub': frame_see_also_sub,
    'version_added_sub': ''
}
series_examples_sub: str = dedent(
    '    >>> int_values = [1, 2, 3, 4, 5]\n'
    '    >>> text_values = [\'alpha\', \'beta\', \'gamma\', \'delta\', \'epsilon\']\n'
    '    >>> s = pd.Series(text_values, index=int_values)\n'
    '    >>> s.info()\n'
    '    <class \'pandas.Series\'>\n'
    '    Index: 5 entries, 1 to 5\n'
    '    Series name: None\n'
    '    Non-Null Count  Dtype\n'
    '    --------------  -----\n'
    '    5 non-null      object\n'
    '    dtypes: object(1)\n'
    '    memory usage: 80.0+ bytes\n'
    '\n'
    '    Prints a summary excluding information about its values:\n'
    '\n'
    '    >>> s.info(verbose=False)\n'
    '    <class \'pandas.Series\'>\n'
    '    Index: 5 entries, 1 to 5\n'
    '    dtypes: object(1)\n'
    '    memory usage: 80.0+ bytes\n'
    '\n'
    '    Pipe output of Series.info to buffer instead of sys.stdout, get\n'
    '    buffer content and writes to a text file:\n'
    '\n'
    '    >>> import io\n'
    '    >>> buffer = io.StringIO()\n'
    '    >>> s.info(buf=buffer)\n'
    '    >>> s = buffer.getvalue()\n'
    '    >>> with open("df_info.txt", "w",\n'
    '    ...           encoding="utf-8") as f:  # doctest: +SKIP\n'
    '    ...     f.write(s)\n'
    '    260\n'
    '\n'
    '    The `memory_usage` parameter allows deep introspection mode, specially\n'
    '    useful for big Series and fine-tune memory optimization:\n'
    '\n'
    '    >>> random_strings_array = np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)\n'
    '    >>> s = pd.Series(np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6))\n'
    '    >>> s.info()\n'
    '    <class \'pandas.Series\'>\n'
    '    RangeIndex: 1000000 entries, 0 to 999999\n'
    '    Series name: None\n'
    '    Non-Null Count    Dtype\n'
    '    --------------    -----\n'
    '    1000000 non-null  object\n'
    '    dtypes: object(1)\n'
    '    memory usage: 7.6+ MB\n'
    '\n'
    '    >>> s.info(memory_usage=\'deep\')\n'
    '    <class \'pandas.Series\'>\n'
    '    RangeIndex: 1000000 entries, 0 to 999999\n'
    '    Series name: None\n'
    '    Non-Null Count    Dtype\n'
    '    --------------    -----\n'
    '    1000000 non-null  object\n'
    '    dtypes: object(1)\n'
    '    memory usage: 55.3 MB'
)
series_see_also_sub: str = dedent(
    '    Series.describe: Generate descriptive statistics of Series.\n'
    '    Series.memory_usage: Memory usage of Series.'
)
series_max_cols_sub: str = dedent(
    '    max_cols : int, optional\n'
    '        Unused, exists only for compatibility with DataFrame.info.'
)
series_sub_kwargs: dict[str, Any] = {
    'klass': 'Series',
    'type_sub': '',
    'max_cols_sub': series_max_cols_sub,
    'show_counts_sub': show_counts_sub,
    'examples_sub': series_examples_sub,
    'see_also_sub': series_see_also_sub,
    'version_added_sub': '\n.. versionadded:: 1.4.0\n'
}
INFO_DOCSTRING: str = dedent(
    '\n'
    '    Print a concise summary of a {klass}.\n'
    '\n'
    '    This method prints information about a {klass} including\n'
    '    the index dtype{type_sub}, non-null values and memory usage.\n'
    '    {version_added_sub}\n'
    '    Parameters\n'
    '    ----------\n'
    '    verbose : bool, optional\n'
    '        Whether to print the full summary. By default, the setting in\n'
    '        ``pandas.options.display.max_info_columns`` is followed.\n'
    '    buf : writable buffer, defaults to sys.stdout\n'
    '        Where to send the output. By default, the output is printed to\n'
    '        sys.stdout. Pass a writable buffer if you need to further process\n'
    '        the output.\n'
    '    {max_cols_sub}\n'
    '    memory_usage : bool, str, optional\n'
    '        Specifies whether total memory usage of the {klass}\n'
    '        elements (including the index) should be displayed. By default,\n'
    '        this follows the ``pandas.options.display.memory_usage`` setting.\n'
    '\n'
    '        True always show memory usage. False never shows memory usage.\n'
    '        A value of \'deep\' is equivalent to "True with deep introspection".\n'
    '        Memory usage is shown in human-readable units (base-2\n'
    '        representation). Without deep introspection a memory estimation is\n'
    '        made based in column dtype and number of rows assuming values\n'
    '        consume the same memory amount for corresponding dtypes. With deep\n'
    '        memory introspection, a real memory usage calculation is performed\n'
    '        at the cost of computational resources. See the\n'
    '        :ref:`Frequently Asked Questions <df-memory-usage>` for more\n'
    '        details.\n'
    '    {show_counts_sub}\n'
    '\n'
    '    Returns\n'
    '    -------\n'
    '    None\n'
    '        This method prints a summary of a {klass} and returns None.\n'
    '\n'
    '    See Also\n'
    '    --------\n'
    '    {see_also_sub}\n'
    '\n'
    '    Examples\n'
    '    --------\n'
    '    {examples_sub}\n'
    '    '
)


def _put_str(s: Any, space: int) -> str:
    """
    Make string of specified length, padding to the right if necessary.

    Parameters
    ----------
    s : Union[str, Dtype]
        String to be formatted.
    space : int
        Length to force string to be of.

    Returns
    -------
    str
        String coerced to given length.

    Examples
    --------
    >>> pd.io.formats.info._put_str("panda", 6)
    'panda '
    >>> pd.io.formats.info._put_str("panda", 4)
    'pand'
    """
    return str(s)[:space].ljust(space)


def _sizeof_fmt(num: int, size_qualifier: str) -> str:
    """
    Return size in human readable format.

    Parameters
    ----------
    num : int
        Size in bytes.
    size_qualifier : str
        Either empty, or '+' (if lower bound).

    Returns
    -------
    str
        Size in human readable format.

    Examples
    --------
    >>> _sizeof_fmt(23028, "")
    '22.5 KB'

    >>> _sizeof_fmt(23028, "+")
    '22.5+ KB'
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return f'{num:3.1f}{size_qualifier} {x}'
        num /= 1024.0
    return f'{num:3.1f}{size_qualifier} PB'


def _initialize_memory_usage(memory_usage: Optional[Union[bool, str]] = None) -> Union[bool, str]:
    """Get memory usage based on inputs and display options."""
    if memory_usage is None:
        memory_usage = get_option('display.memory_usage')
    return memory_usage


class _BaseInfo(ABC):
    """
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    memory_usage : bool or str, optional
        If "deep", introspect the data deeply by interrogating object dtypes
        for system-level memory consumption, and include it in the returned
        values.
    """

    @property
    @abstractmethod
    def dtypes(self) -> Any:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[Any, int]:
        """Mapping dtype - number of counts."""

    @property
    @abstractmethod
    def non_null_counts(self) -> Any:
        """Sequence of non-null counts for all columns or column (if series)."""

    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int:
        """
        Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """

    @property
    def memory_usage_string(self) -> str:
        """Memory usage in a form of human readable string."""
        return f'{_sizeof_fmt(self.memory_usage_bytes, self.size_qualifier)}\n'

    @property
    def size_qualifier(self) -> str:
        size_qualifier: str = ''
        if self.memory_usage:
            if self.memory_usage != 'deep':
                if 'object' in self.dtype_counts or self.data.index._is_memory_usage_qualified:
                    size_qualifier = '+'
        return size_qualifier

    @abstractmethod
    def render(self, *, buf: Optional[Any], max_cols: Optional[int], verbose: Optional[bool], show_counts: Optional[bool]) -> None:
        pass


class DataFrameInfo(_BaseInfo):
    """
    Class storing dataframe-specific info.
    """

    def __init__(self, data: DataFrame, memory_usage: Optional[Union[bool, str]] = None) -> None:
        self.data: DataFrame = data
        self.memory_usage: Union[bool, str] = _initialize_memory_usage(memory_usage)

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self) -> Any:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        return self.data.dtypes

    @property
    def ids(self) -> Any:
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
        return self.data.columns

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return len(self.ids)

    @property
    def non_null_counts(self) -> Any:
        """Sequence of non-null counts for all columns or column (if series)."""
        return self.data.count()

    @property
    def memory_usage_bytes(self) -> int:
        deep: bool = self.memory_usage == 'deep'
        return self.data.memory_usage(index=True, deep=deep).sum()

    def render(self, *, buf: Optional[Any], max_cols: Optional[int], verbose: Optional[bool], show_counts: Optional[bool]) -> None:
        printer: _DataFrameInfoPrinter = _DataFrameInfoPrinter(info=self, max_cols=max_cols, verbose=verbose, show_counts=show_counts)
        printer.to_buffer(buf)


class SeriesInfo(_BaseInfo):
    """
    Class storing series-specific info.
    """

    def __init__(self, data: Series, memory_usage: Optional[Union[bool, str]] = None) -> None:
        self.data: Series = data
        self.memory_usage: Union[bool, str] = _initialize_memory_usage(memory_usage)

    def render(self, *, buf: Optional[Any] = None, max_cols: Optional[int] = None, verbose: Optional[bool] = None, show_counts: Optional[bool] = None) -> None:
        if max_cols is not None:
            raise ValueError('Argument `max_cols` can only be passed in DataFrame.info, not Series.info')
        printer: _SeriesInfoPrinter = _SeriesInfoPrinter(info=self, verbose=verbose, show_counts=show_counts)
        printer.to_buffer(buf)

    @property
    def non_null_counts(self) -> List[Any]:
        return [self.data.count()]

    @property
    def dtypes(self) -> List[Any]:
        return [self.data.dtypes]

    @property
    def dtype_counts(self) -> Mapping[Any, int]:
        from pandas.core.frame import DataFrame
        return _get_dataframe_dtype_counts(DataFrame(self.data))

    @property
    def memory_usage_bytes(self) -> int:
        """Memory usage in bytes.

        Returns
        -------
        memory_usage_bytes : int
            Object's total memory usage in bytes.
        """
        deep: bool = self.memory_usage == 'deep'
        return self.data.memory_usage(index=True, deep=deep)


class _InfoPrinterAbstract:
    """
    Class for printing dataframe or series info.
    """

    def to_buffer(self, buf: Optional[Any] = None) -> None:
        """Save dataframe info into buffer."""
        table_builder: _TableBuilderAbstract = self._create_table_builder()
        lines: List[str] = table_builder.get_lines()
        if buf is None:
            buf = sys.stdout
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self) -> _TableBuilderAbstract:
        """Create instance of table builder."""


class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(self, info: DataFrameInfo, max_cols: Optional[int] = None, verbose: Optional[bool] = None, show_counts: Optional[bool] = None) -> None:
        self.info: DataFrameInfo = info
        self.data: DataFrame = info.data
        self.verbose: Optional[bool] = verbose
        self.max_cols: int = self._initialize_max_cols(max_cols)
        self.show_counts: bool = self._initialize_show_counts(show_counts)

    @property
    def max_rows(self) -> int:
        """Maximum info rows to be displayed."""
        return get_option('display.max_info_rows')

    @property
    def exceeds_info_cols(self) -> bool:
        """Check if number of columns to be summarized does not exceed maximum."""
        return bool(self.col_count > self.max_cols)

    @property
    def exceeds_info_rows(self) -> bool:
        """Check if number of rows to be summarized does not exceed maximum."""
        return bool(len(self.data) > self.max_rows)

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return self.info.col_count

    def _initialize_max_cols(self, max_cols: Optional[int]) -> int:
        if max_cols is None:
            return get_option('display.max_info_columns')
        return max_cols

    def _initialize_show_counts(self, show_counts: Optional[bool]) -> bool:
        if show_counts is None:
            return bool(not self.exceeds_info_cols and (not self.exceeds_info_rows))
        else:
            return show_counts

    def _create_table_builder(self) -> _TableBuilderAbstract:
        """
        Create instance of table builder based on verbosity and display settings.
        """
        if self.verbose:
            return _DataFrameTableBuilderVerbose(info=self.info, with_counts=self.show_counts)
        elif self.verbose is False:
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        elif self.exceeds_info_cols:
            return _DataFrameTableBuilderNonVerbose(info=self.info)
        else:
            return _DataFrameTableBuilderVerbose(info=self.info, with_counts=self.show_counts)


class _SeriesInfoPrinter(_InfoPrinterAbstract):
    """Class for printing series info.

    Parameters
    ----------
    info : SeriesInfo
        Instance of SeriesInfo.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    def __init__(self, info: SeriesInfo, verbose: Optional[bool] = None, show_counts: Optional[bool] = None) -> None:
        self.info: SeriesInfo = info
        self.data: Series = info.data
        self.verbose: Optional[bool] = verbose
        self.show_counts: bool = self._initialize_show_counts(show_counts)

    def _create_table_builder(self) -> _TableBuilderAbstract:
        """
        Create instance of table builder based on verbosity.
        """
        if self.verbose or self.verbose is None:
            return _SeriesTableBuilderVerbose(info=self.info, with_counts=self.show_counts)
        else:
            return _SeriesTableBuilderNonVerbose(info=self.info)

    def _initialize_show_counts(self, show_counts: Optional[bool]) -> bool:
        if show_counts is None:
            return True
        else:
            return show_counts


class _TableBuilderAbstract(ABC):
    """
    Abstract builder for info table.
    """

    @abstractmethod
    def get_lines(self) -> List[str]:
        """Product in a form of list of lines (strings)."""

    @property
    def data(self) -> Any:
        return self.info.data

    @property
    def dtypes(self) -> Any:
        """Dtypes of each of the DataFrame's columns."""
        return self.info.dtypes

    @property
    def dtype_counts(self) -> Mapping[Any, int]:
        """Mapping dtype - number of counts."""
        return self.info.dtype_counts

    @property
    def display_memory_usage(self) -> bool:
        """Whether to display memory usage."""
        return bool(self.info.memory_usage)

    @property
    def memory_usage_string(self) -> str:
        """Memory usage string with proper size qualifier."""
        return self.info.memory_usage_string

    @property
    def non_null_counts(self) -> Any:
        return self.info.non_null_counts

    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
        self._lines.append(str(type(self.data)))

    def add_index_range_line(self) -> None:
        """Add line with range of indices to the table."""
        self._lines.append(self.data.index._summary())

    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""
        collected_dtypes: List[str] = [f'{key}({val:d})' for key, val in sorted(self.dtype_counts.items())]
        self._lines.append(f'dtypes: {", ".join(collected_dtypes)}')


class _DataFrameTableBuilder(_TableBuilderAbstract):
    """
    Abstract builder for dataframe info table.

    Parameters
    ----------
    info : DataFrameInfo.
        Instance of DataFrameInfo.
    """

    def __init__(self, *, info: DataFrameInfo) -> None:
        self.info: DataFrameInfo = info

    def get_lines(self) -> List[str]:
        self._lines: List[str] = []
        if self.col_count == 0:
            self._fill_empty_info()
        else:
            self._fill_non_empty_info()
        return self._lines

    def _fill_empty_info(self) -> None:
        """Add lines to the info table, pertaining to empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self._lines.append(f'Empty {type(self.data).__name__}\n')

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""

    @property
    def data(self) -> DataFrame:
        """DataFrame."""
        return self.info.data

    @property
    def ids(self) -> Any:
        """Dataframe columns."""
        return self.info.ids

    @property
    def col_count(self) -> int:
        """Number of dataframe columns to be summarized."""
        return self.info.col_count

    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        self._lines.append(f'memory usage: {self.memory_usage_string}')


class _DataFrameTableBuilderNonVerbose(_DataFrameTableBuilder):
    """
    Dataframe info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_columns_summary_line(self) -> None:
        self._lines.append(self.ids._summary(name='Columns'))


class _TableBuilderVerboseMixin(_TableBuilderAbstract):
    """
    Mixin for verbose info output.
    """
    SPACING: str = ' ' * 2

    @property
    @abstractmethod
    def headers(self) -> List[str]:
        """Headers names of the columns in verbose table."""

    @property
    def header_column_widths(self) -> List[int]:
        """Widths of header columns (only titles)."""
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> List[int]:
        """Get widths of columns containing both headers and actual content."""
        body_column_widths: List[int] = self._get_body_column_widths()
        return [max(header, body) for header, body in zip(self.header_column_widths, body_column_widths)]

    def _get_body_column_widths(self) -> List[int]:
        """Get widths of table content columns."""
        strcols: List[Tuple[str, ...]] = list(zip(*self.strrows))
        return [max((len(x) for x in col)) for col in strcols]

    def _gen_rows(self) -> Iterator[Tuple[str, ...]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
        if self.with_counts:
            return self._gen_rows_with_counts()
        else:
            return self._gen_rows_without_counts()

    @abstractmethod
    def _gen_rows_with_counts(self) -> Iterator[Tuple[str, ...]]:
        """Iterator with string representation of body data with counts."""

    @abstractmethod
    def _gen_rows_without_counts(self) -> Iterator[Tuple[str, ...]]:
        """Iterator with string representation of body data without counts."""

    def add_header_line(self) -> None:
        header_line: str = self.SPACING.join([_put_str(header, col_width) for header, col_width in zip(self.headers, self.gross_column_widths)])
        self._lines.append(header_line)

    def add_separator_line(self) -> None:
        separator_line: str = self.SPACING.join([_put_str('-' * header_colwidth, gross_colwidth) for header_colwidth, gross_colwidth in zip(self.header_column_widths, self.gross_column_widths)])
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        for row in self.strrows:
            body_line: str = self.SPACING.join([_put_str(col, gross_colwidth) for col, gross_colwidth in zip(row, self.gross_column_widths)])
            self._lines.append(body_line)

    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
        for count in self.non_null_counts:
            yield f'{count} non-null'

    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
        for dtype in self.dtypes:
            yield pprint_thing(dtype)


class _DataFrameTableBuilderVerbose(_DataFrameTableBuilder, _TableBuilderVerboseMixin):
    """
    Dataframe info table builder for verbose output.
    """

    def __init__(self, *, info: DataFrameInfo, with_counts: bool) -> None:
        self.info: DataFrameInfo = info
        self.with_counts: bool = with_counts
        self.strrows: List[Tuple[str, ...]] = list(self._gen_rows())
        self.gross_column_widths: List[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_columns_summary_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    @property
    def headers(self) -> List[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return [' # ', 'Column', 'Non-Null Count', 'Dtype']
        return [' # ', 'Column', 'Dtype']

    def add_columns_summary_line(self) -> None:
        self._lines.append(f'Data columns (total {self.col_count} columns):')

    def _gen_rows_without_counts(self) -> Iterator[Tuple[str, ...]]:
        """Iterator with string representation of body data without counts."""
        return zip(self._gen_line_numbers(), self._gen_columns(), self._gen_dtypes())

    def _gen_rows_with_counts(self) -> Iterator[Tuple[str, ...]]:
        """Iterator with string representation of body data with counts."""
        return zip(self._gen_line_numbers(), self._gen_columns(), self._gen_non_null_counts(), self._gen_dtypes())

    def _gen_line_numbers(self) -> Iterator[str]:
        """Iterator with string representation of column numbers."""
        for i, _ in enumerate(self.ids):
            yield f' {i}'

    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""
        for col in self.ids:
            yield pprint_thing(col)


class _SeriesTableBuilder(_TableBuilderAbstract):
    """
    Abstract builder for series info table.

    Parameters
    ----------
    info : SeriesInfo.
        Instance of SeriesInfo.
    """

    def __init__(self, *, info: SeriesInfo) -> None:
        self.info: SeriesInfo = info

    def get_lines(self) -> List[str]:
        self._lines: List[str] = []
        self._fill_non_empty_info()
        return self._lines

    @property
    def data(self) -> Series:
        """Series."""
        return self.info.data

    def add_memory_usage_line(self) -> None:
        """Add line containing memory usage."""
        self._lines.append(f'memory usage: {self.memory_usage_string}')

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""


class _SeriesTableBuilderNonVerbose(_SeriesTableBuilder):
    """
    Series info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()


class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    """
    Series info table builder for verbose output.
    """

    def __init__(self, *, info: SeriesInfo, with_counts: bool) -> None:
        self.info: SeriesInfo = info
        self.with_counts: bool = with_counts
        self.strrows: List[Tuple[str, ...]] = list(self._gen_rows())
        self.gross_column_widths: List[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_object_type_line()
        self.add_index_range_line()
        self.add_series_name_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()
        if self.display_memory_usage:
            self.add_memory_usage_line()

    def add_series_name_line(self) -> None:
        self._lines.append(f'Series name: {self.data.name}')

    @property
    def headers(self) -> List[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return ['Non-Null Count', 'Dtype']
        return ['Dtype']

    def _gen_rows_without_counts(self) -> Iterator[Tuple[str, ...]]:
        """Iterator with string representation of body data without counts."""
        # Wrap single element into a tuple for consistency.
        return ((dtype,) for dtype in self._gen_dtypes())
    
    def _gen_rows_with_counts(self) -> Iterator[Tuple[str, ...]]:
        """Iterator with string representation of body data with counts."""
        return zip(self._gen_non_null_counts(), self._gen_dtypes())


def _get_dataframe_dtype_counts(df: DataFrame) -> Mapping[str, int]:
    """
    Create mapping between datatypes and their number of occurrences.
    """
    return df.dtypes.value_counts().groupby(lambda x: x.name).sum()  # type: ignore
