from __future__ import annotations
from abc import ABC, abstractmethod
import sys
from textwrap import dedent
from typing import TYPE_CHECKING, Optional, Union, List, Dict, Any, Tuple
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pandas._typing import Dtype, WriteBuffer
    from pandas import DataFrame, Index, Series

frame_max_cols_sub: str = dedent('    max_cols : int, optional\n        When to switch from the verbose to the truncated output. If the\n        DataFrame has more than `max_cols` columns, the truncated output\n        is used. By default, the setting in\n        ``pandas.options.display.max_info_columns`` is used.')
show_counts_sub: str = dedent('    show_counts : bool, optional\n        Whether to show the non-null counts. By default, this is shown\n        only if the DataFrame is smaller than\n        ``pandas.options.display.max_info_rows`` and\n        ``pandas.options.display.max_info_columns``. A value of True always\n        shows the counts, and False never shows the counts.')
frame_examples_sub: str = dedent('    >>> int_values = [1, 2, 3, 4, 5]\n    >>> text_values = [\'alpha\', \'beta\', \'gamma\', \'delta\', \'epsilon\']\n    >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]\n    >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,\n    ...                   "float_col": float_values})\n    >>> df\n        int_col text_col  float_col\n    0        1    alpha       0.00\n    1        2     beta       0.25\n    2        3    gamma       0.50\n    3        4    delta       0.75\n    4        5  epsilon       1.00\n\n    Prints information of all columns:\n\n    >>> df.info(verbose=True)\n    <class \'pandas.DataFrame\'>\n    RangeIndex: 5 entries, 0 to 4\n    Data columns (total 3 columns):\n     #   Column     Non-Null Count  Dtype\n    ---  ------     --------------  -----\n     0   int_col    5 non-null      int64\n     1   text_col   5 non-null      object\n     2   float_col  5 non-null      float64\n    dtypes: float64(1), int64(1), object(1)\n    memory usage: 248.0+ bytes\n\n    Prints a summary of columns count and its dtypes but not per column\n    information:\n\n    >>> df.info(verbose=False)\n    <class \'pandas.DataFrame\'>\n    RangeIndex: 5 entries, 0 to 4\n    Columns: 3 entries, int_col to float_col\n    dtypes: float64(1), int64(1), object(1)\n    memory usage: 248.0+ bytes\n\n    Pipe output of DataFrame.info to buffer instead of sys.stdout, get\n    buffer content and writes to a text file:\n\n    >>> import io\n    >>> buffer = io.StringIO()\n    >>> df.info(buf=buffer)\n    >>> s = buffer.getvalue()\n    >>> with open("df_info.txt", "w",\n    ...           encoding="utf-8") as f:  # doctest: +SKIP\n    ...     f.write(s)\n    260\n\n    The `memory_usage` parameter allows deep introspection mode, specially\n    useful for big DataFrames and fine-tune memory optimization:\n\n    >>> random_strings_array = np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)\n    >>> df = pd.DataFrame({\n    ...     \'column_1\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6),\n    ...     \'column_2\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6),\n    ...     \'column_3\': np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)\n    ... })\n    >>> df.info()\n    <class \'pandas.DataFrame\'>\n    RangeIndex: 1000000 entries, 0 to 999999\n    Data columns (total 3 columns):\n     #   Column    Non-Null Count    Dtype\n    ---  ------    --------------    -----\n     0   column_1  1000000 non-null  object\n     1   column_2  1000000 non-null  object\n     2   column_3  1000000 non-null  object\n    dtypes: object(3)\n    memory usage: 22.9+ MB\n\n    >>> df.info(memory_usage=\'deep\')\n    <class \'pandas.DataFrame\'>\n    RangeIndex: 1000000 entries, 0 to 999999\n    Data columns (total 3 columns):\n     #   Column    Non-Null Count    Dtype\n    ---  ------    --------------    -----\n     0   column_1  1000000 non-null  object\n     1   column_2  1000000 non-null  object\n     2   column_3  1000000 non-null  object\n    dtypes: object(3)\n    memory usage: 165.9 MB')
frame_see_also_sub: str = dedent('    DataFrame.describe: Generate descriptive statistics of DataFrame\n        columns.\n    DataFrame.memory_usage: Memory usage of DataFrame columns.')
frame_sub_kwargs: Dict[str, str] = {'klass': 'DataFrame', 'type_sub': ' and columns', 'max_cols_sub': frame_max_cols_sub, 'show_counts_sub': show_counts_sub, 'examples_sub': frame_examples_sub, 'see_also_sub': frame_see_also_sub, 'version_added_sub': ''}
series_examples_sub: str = dedent('    >>> int_values = [1, 2, 3, 4, 5]\n    >>> text_values = [\'alpha\', \'beta\', \'gamma\', \'delta\', \'epsilon\']\n    >>> s = pd.Series(text_values, index=int_values)\n    >>> s.info()\n    <class \'pandas.Series\'>\n    Index: 5 entries, 1 to 5\n    Series name: None\n    Non-Null Count  Dtype\n    --------------  -----\n    5 non-null      object\n    dtypes: object(1)\n    memory usage: 80.0+ bytes\n\n    Prints a summary excluding information about its values:\n\n    >>> s.info(verbose=False)\n    <class \'pandas.Series\'>\n    Index: 5 entries, 1 to 5\n    dtypes: object(1)\n    memory usage: 80.0+ bytes\n\n    Pipe output of Series.info to buffer instead of sys.stdout, get\n    buffer content and writes to a text file:\n\n    >>> import io\n    >>> buffer = io.StringIO()\n    >>> s.info(buf=buffer)\n    >>> s = buffer.getvalue()\n    >>> with open("df_info.txt", "w",\n    ...           encoding="utf-8") as f:  # doctest: +SKIP\n    ...     f.write(s)\n    260\n\n    The `memory_usage` parameter allows deep introspection mode, specially\n    useful for big Series and fine-tune memory optimization:\n\n    >>> random_strings_array = np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6)\n    >>> s = pd.Series(np.random.choice([\'a\', \'b\', \'c\'], 10 ** 6))\n    >>> s.info()\n    <class \'pandas.Series\'>\n    RangeIndex: 1000000 entries, 0 to 999999\n    Series name: None\n    Non-Null Count    Dtype\n    --------------    -----\n    1000000 non-null  object\n    dtypes: object(1)\n    memory usage: 7.6+ MB\n\n    >>> s.info(memory_usage=\'deep\')\n    <class \'pandas.Series\'>\n    RangeIndex: 1000000 entries, 0 to 999999\n    Series name: None\n    Non-Null Count    Dtype\n    --------------    -----\n    1000000 non-null  object\n    dtypes: object(1)\n    memory usage: 55.3 MB')
series_see_also_sub: str = dedent('    Series.describe: Generate descriptive statistics of Series.\n    Series.memory_usage: Memory usage of Series.')
series_max_cols_sub: str = dedent('    max_cols : int, optional\n        Unused, exists only for compatibility with DataFrame.info.')
series_sub_kwargs: Dict[str, str] = {'klass': 'Series', 'type_sub': '', 'max_cols_sub': series_max_cols_sub, 'show_counts_sub': show_counts_sub, 'examples_sub': series_examples_sub, 'see_also_sub': series_see_also_sub, 'version_added_sub': '\n.. versionadded:: 1.4.0\n'}
INFO_DOCSTRING: str = dedent('\n    Print a concise summary of a {klass}.\n\n    This method prints information about a {klass} including\n    the index dtype{type_sub}, non-null values and memory usage.\n    {version_added_sub}\n    Parameters\n    ----------\n    verbose : bool, optional\n        Whether to print the full summary. By default, the setting in\n        ``pandas.options.display.max_info_columns`` is followed.\n    buf : writable buffer, defaults to sys.stdout\n        Where to send the output. By default, the output is printed to\n        sys.stdout. Pass a writable buffer if you need to further process\n        the output.\n    {max_cols_sub}\n    memory_usage : bool, str, optional\n        Specifies whether total memory usage of the {klass}\n        elements (including the index) should be displayed. By default,\n        this follows the ``pandas.options.display.memory_usage`` setting.\n\n        True always show memory usage. False never shows memory usage.\n        A value of \'deep\' is equivalent to "True with deep introspection".\n        Memory usage is shown in human-readable units (base-2\n        representation). Without deep introspection a memory estimation is\n        made based in column dtype and number of rows assuming values\n        consume the same memory amount for corresponding dtypes. With deep\n        memory introspection, a real memory usage calculation is performed\n        at the cost of computational resources. See the\n        :ref:`Frequently Asked Questions <df-memory-usage>` for more\n        details.\n    {show_counts_sub}\n\n    Returns\n    -------\n    None\n        This method prints a summary of a {klass} and returns None.\n\n    See Also\n    --------\n    {see_also_sub}\n\n    Examples\n    --------\n    {examples_sub}\n    ')

def _put_str(s: Union[str, Any], space: int) -> str:
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
    data: Union['DataFrame', 'Series']
    memory_usage: Union[bool, str]

    @property
    @abstractmethod
    def dtypes(self) -> 'Iterable[Dtype]':
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def dtype_counts(self) -> 'Mapping[str, int]':
        """Mapping dtype - number of counts."""

    @property
    @abstractmethod
    def non_null_counts(self) -> Union[List[int], 'Series']:
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
        size_qualifier = ''
        if self.memory_usage:
            if self.memory_usage != 'deep':
                if 'object' in self.dtype_counts or self.data.index._is_memory_usage_qualified:
                    size_qualifier = '+'
        return size_qualifier

    @abstractmethod
    def render(self, *, buf: Optional['WriteBuffer[str]'], max_cols: Optional[int], verbose: Optional[bool], show_counts: Optional[bool]) -> None:
        pass

class DataFrameInfo(_BaseInfo):
    """
    Class storing dataframe-specific info.
    """

    def __init__(self, data: 'DataFrame', memory_usage: Optional[Union[bool, str]] = None) -> None:
        self.data: 'DataFrame' = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    @property
    def dtype_counts(self) -> 'Mapping[str, int]':
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self) -> 'Iterable[Dtype]':
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        return self.data.dtypes

    @property
    def ids(self) -> 'Index':
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
    def non_null_counts(self) -> 'Series':
        """Sequence of non-null counts for all columns or column (if series)."""
        return self.data.count()

    @property
    def memory_usage_bytes(self) -> int:
        deep = self.memory_usage == 'deep'
        return self.data.memory_usage(index=True, deep=deep).sum()

    def render(self, *, buf: Optional['WriteBuffer[str]'], max_cols: Optional[int], verbose: Optional[bool], show_counts: Optional[bool]) -> None:
        printer = _DataFrameInfoPrinter(info=self, max_cols=max_cols, verbose=verbose, show_counts=show_counts)
        printer.to_buffer(buf)

class SeriesInfo(_BaseInfo):
    """
    Class storing series-specific info.
    """

    def __init__(self, data: 'Series', memory_usage: Optional[Union[bool, str]] = None) -> None:
        self.data: 'Series' = data
        self.memory_usage = _initialize_memory_usage(memory_usage)

    def render(self, *, buf: Optional['WriteBuffer[str]'] = None, max_cols: Optional[int] = None, verbose: Optional[bool] = None, show_counts: Optional[bool] = None) -> None:
        if max_cols is not None:
            raise ValueError('Argument `max_cols` can only be passed in DataFrame.info, not Series.info')
        printer = _SeriesInfoPrinter(info=self, verbose=verbose, show_counts=show_counts)
        printer.to_buffer(buf)

    @property
    def non_null_counts(self)