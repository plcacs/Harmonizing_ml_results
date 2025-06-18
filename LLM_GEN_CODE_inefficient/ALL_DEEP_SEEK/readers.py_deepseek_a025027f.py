from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
import csv
import sys
from textwrap import fill
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypedDict,
    overload,
    Union,
    Optional,
    Sequence,
    Mapping,
    Callable,
    Dict,
    List,
    Set,
    Tuple,
    Iterator,
    TypeVar,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import (
    AbstractMethodError,
    ParserWarning,
)
from pandas.util._decorators import (
    Appender,
    set_module,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_file_like,
    is_float,
    is_integer,
    is_list_like,
    pandas_dtype,
)

from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs

from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper
from pandas.io.parsers.base_parser import (
    ParserBase,
    is_index_col,
    parser_defaults,
)
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
from pandas.io.parsers.python_parser import (
    FixedWidthFieldParser,
    PythonParser,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterable,
        Mapping,
        Sequence,
    )
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        CSVEngine,
        DtypeArg,
        DtypeBackend,
        FilePath,
        HashableT,
        IndexLabel,
        ReadCsvBuffer,
        Self,
        StorageOptions,
        Unpack,
        UsecolsArgType,
    )

    class _read_shared(TypedDict, Generic[HashableT], total=False):
        sep: str | None | lib.NoDefault
        delimiter: str | None | lib.NoDefault
        header: int | Sequence[int] | None | Literal["infer"]
        names: Sequence[Hashable] | None | lib.NoDefault
        index_col: IndexLabel | Literal[False] | None
        usecols: UsecolsArgType
        dtype: DtypeArg | None
        engine: CSVEngine | None
        converters: Mapping[HashableT, Callable] | None
        true_values: list | None
        false_values: list | None
        skipinitialspace: bool
        skiprows: list[int] | int | Callable[[Hashable], bool] | None
        skipfooter: int
        nrows: int | None
        na_values: (
            Hashable | Iterable[Hashable] | Mapping[Hashable, Iterable[Hashable]] | None
        )
        keep_default_na: bool
        na_filter: bool
        skip_blank_lines: bool
        parse_dates: bool | Sequence[Hashable] | None
        date_format: str | dict[Hashable, str] | None
        dayfirst: bool
        cache_dates: bool
        compression: CompressionOptions
        thousands: str | None
        decimal: str
        lineterminator: str | None
        quotechar: str
        quoting: int
        doublequote: bool
        escapechar: str | None
        comment: str | None
        encoding: str | None
        encoding_errors: str | None
        dialect: str | csv.Dialect | None
        on_bad_lines: str
        low_memory: bool
        memory_map: bool
        float_precision: Literal["high", "legacy", "round_trip"] | None
        storage_options: StorageOptions | None
        dtype_backend: DtypeBackend | lib.NoDefault
else:
    _read_shared = dict


_doc_read_csv_and_table = r"""
{summary}

Also supports optionally iterating or breaking of the file
into chunks.

Additional help can be found in the online docs for
`IO Tools <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

Parameters
----------
filepath_or_buffer : str, path object or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is
    expected. A local file could be: file://localhost/path/to/table.csv.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method, such as
    a file handle (e.g. via builtin ``open`` function) or ``StringIO``.
sep : str, default {_default_sep}
    Character or regex pattern to treat as the delimiter. If ``sep=None``, the
    C engine cannot automatically detect
    the separator, but the Python parsing engine can, meaning the latter will
    be used and automatically detect the separator from only the first valid
    row of the file by Python's builtin sniffer tool, ``csv.Sniffer``.
    In addition, separators longer than 1 character and different from
    ``'\s+'`` will be interpreted as regular expressions and will also force
    the use of the Python parsing engine. Note that regex delimiters are prone
    to ignoring quoted data. Regex example: ``'\r\t'``.
delimiter : str, optional
    Alias for ``sep``.
header : int, Sequence of int, 'infer' or None, default 'infer'
    Row number(s) containing column labels and marking the start of the
    data (zero-indexed). Default behavior is to infer the column names: if no ``names``
    are passed the behavior is identical to ``header=0`` and column
    names are inferred from the first line of the file, if column
    names are passed explicitly to ``names`` then the behavior is identical to
    ``header=None``. Explicitly pass ``header=0`` to be able to
    replace existing names. The header can be a list of integers that
    specify row locations for a :class:`~pandas.MultiIndex` on the columns
    e.g. ``[0, 1, 3]``. Intervening rows that are not specified will be
    skipped (e.g. 2 in this example is skipped). Note that this
    parameter ignores commented lines and empty lines if
    ``skip_blank_lines=True``, so ``header=0`` denotes the first line of
    data rather than the first line of the file.

    When inferred from the file contents, headers are kept distinct from
    each other by renaming duplicate names with a numeric suffix of the form
    ``".{{count}}"`` starting from 1, e.g. ``"foo"`` and ``"foo.1"``.
    Empty headers are named ``"Unnamed: {{i}}"`` or ``"Unnamed: {{i}}_level_{{level}}"``
    in the case of MultiIndex columns.
names : Sequence of Hashable, optional
    Sequence of column labels to apply. If the file contains a header row,
    then you should explicitly pass ``header=0`` to override the column names.
    Duplicates in this list are not allowed.
index_col : Hashable, Sequence of Hashable or False, optional
  Column(s) to use as row label(s), denoted either by column labels or column
  indices.  If a sequence of labels or indices is given, :class:`~pandas.MultiIndex`
  will be formed for the row labels.

  Note: ``index_col=False`` can be used to force pandas to *not* use the first
  column as the index, e.g., when you have a malformed file with delimiters at
  the end of each line.
usecols : Sequence of Hashable or Callable, optional
    Subset of columns to select, denoted either by column labels or column indices.
    If list-like, all elements must either
    be positional (i.e. integer indices into the document columns) or strings
    that correspond to column names provided either by the user in ``names`` or
    inferred from the document header row(s). If ``names`` are given, the document
    header row(s) are not taken into account. For example, a valid list-like
    ``usecols`` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
    Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.
    To instantiate a :class:`~pandas.DataFrame` from ``data`` with element order
    preserved use ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]``
    for columns in ``['foo', 'bar']`` order or
    ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``
    for ``['bar', 'foo']`` order.

    If callable, the callable function will be evaluated against the column
    names, returning names where the callable function evaluates to ``True``. An
    example of a valid callable argument would be ``lambda x: x.upper() in
    ['AAA', 'BBB', 'DDD']``. Using this parameter results in much faster
    parsing time and lower memory usage.
dtype : dtype or dict of {{Hashable : dtype}}, optional
    Data type(s) to apply to either the whole dataset or individual columns.
    E.g., ``{{'a': np.float64, 'b': np.int32, 'c': 'Int64'}}``
    Use ``str`` or ``object`` together with suitable ``na_values`` settings
    to preserve and not interpret ``dtype``.
    If ``converters`` are specified, they will be applied INSTEAD
    of ``dtype`` conversion.

    .. versionadded:: 1.5.0

        Support for ``defaultdict`` was added. Specify a ``defaultdict`` as input where
        the default determines the ``dtype`` of the columns which are not explicitly
        listed.
engine : {{'c', 'python', 'pyarrow'}}, optional
    Parser engine to use. The C and pyarrow engines are faster, while the python engine
    is currently more feature-complete. Multithreading is currently only supported by
    the pyarrow engine.

    .. versionadded:: 1.4.0

        The 'pyarrow' engine was added as an *experimental* engine, and some features
        are unsupported, or may not work correctly, with this engine.
converters : dict of {{Hashable : Callable}}, optional
    Functions for converting values in specified columns. Keys can either
    be column labels or column indices.
true_values : list, optional
    Values to consider as ``True`` in addition to case-insensitive variants of 'True'.
false_values : list, optional
    Values to consider as ``False`` in addition to case-insensitive variants of 'False'.
skipinitialspace : bool, default False
    Skip spaces after delimiter.
skiprows : int, list of int or Callable, optional
    Line numbers to skip (0-indexed) or number of lines to skip (``int``)
    at the start of the file.

    If callable, the callable function will be evaluated against the row
    indices, returning ``True`` if the row should be skipped and ``False`` otherwise.
    An example of a valid callable argument would be ``lambda x: x in [0, 2]``.
skipfooter : int, default 0
    Number of lines at bottom of file to skip (Unsupported with ``engine='c'``).
nrows : int, optional
    Number of rows of file to read. Useful for reading pieces of large files.
    Refers to the number of data rows in the returned DataFrame, excluding:

    * The header row containing column names.
    * Rows before the header row, if ``header=1`` or larger.

    Example usage:

    * To read the first 999,999 (non-header) rows:
      ``read_csv(..., nrows=999999)``

    * To read rows 1,000,000 through 1,999,999:
      ``read_csv(..., skiprows=1000000, nrows=999999)``
na_values : Hashable, Iterable of Hashable or dict of {{Hashable : Iterable}}, optional
    Additional strings to recognize as ``NA``/``NaN``. If ``dict`` passed, specific
    per-column ``NA`` values.  By default the following values are interpreted as
    ``NaN``: "{na_values_str}".
keep_default_na : bool, default True
    Whether or not to include the default ``NaN`` values when parsing the data.
    Depending on whether ``na_values`` is passed in, the behavior is as follows:

    * If ``keep_default_na`` is ``True``, and ``na_values`` are specified, ``na_values``
      is appended to the default ``NaN`` values used for parsing.
    * If ``keep_default_na`` is ``True``, and ``na_values`` are not specified, only
      the default ``NaN`` values are used for parsing.
    * If ``keep_default_na`` is ``False``, and ``na_values`` are specified, only
      the ``NaN`` values specified ``na_values`` are used for parsing.
    * If ``keep_default_na`` is ``False``, and ``na_values`` are not specified, no
      strings will be parsed as ``NaN``.

    Note that if ``na_filter`` is passed in as ``False``, the ``keep_default_na`` and
    ``na_values`` parameters will be ignored.
na_filter : bool, default True
    Detect missing value markers (empty strings and the value of ``na_values``). In
    data without any ``NA`` values, passing ``na_filter=False`` can improve the
    performance of reading a large file.
skip_blank_lines : bool, default True
    If ``True``, skip over blank lines rather than interpreting as ``NaN`` values.
parse_dates : bool, None, list of Hashable, default None
    The behavior is as follows:

    * ``bool``. If ``True`` -> try parsing the index.
    * ``None``. Behaves like ``True`` if ``date_format`` is specified.
    * ``list`` of ``int`` or names. e.g. If ``[1, 2, 3]`` -> try parsing columns 1, 2, 3
      each as a separate date column.

    If a column or index cannot be represented as an array of ``datetime``,
    say because of an unparsable value or a mixture of timezones, the column
    or index will be returned unaltered as an ``object`` data type. For
    non-standard ``datetime`` parsing, use :func:`~pandas.to_datetime` after
    :func:`~pandas.read_csv`.

    Note: A fast-path exists for iso8601-formatted dates.
date_format : str or dict of column -> format, optional
    Format to use for parsing dates and/or times when used in conjunction with ``parse_dates``.
    The strftime to parse time, e.g. :const:`"%d/%m/%Y"`. See
    `strftime documentation
    <https://docs.python.org/3/library/datetime.html
    #strftime-and-strptime-behavior>`_ for more information on choices, though
    note that :const:`"%f"` will parse all the way up to nanoseconds.
    You can also pass:

    - "ISO8601", to parse any `ISO8601 <https://en.wikipedia.org/wiki/ISO_8601>`_
      time string (not necessarily in exactly the same format);
    - "mixed", to infer the format for each element individually. This is risky,
      and you should probably use it along with `dayfirst`.

    .. versionadded:: 2.0.0
dayfirst : bool, default False
    DD/MM format dates, international and European format.
cache_dates : bool, default True
    If ``True``, use a cache of unique, converted dates to apply the ``datetime``
    conversion. May produce significant speed-up when parsing duplicate
    date strings, especially ones with timezone offsets.

iterator : bool, default False
    Return ``TextFileReader`` object for iteration or getting chunks with
    ``get_chunk()``.
chunksize : int, optional
    Number of lines to read from the file per chunk. Passing a value will cause the
    function to return a ``TextFileReader`` object for iteration.
    See the `IO Tools docs
    <https://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking>`_
    for more information on ``iterator`` and ``chunksize``.

{decompression_options}

    .. versionchanged:: 1.4.0 Zstandard support.

thousands : str (length 1), optional
    Character acting as the thousands separator in numerical values.
decimal : str (length 1), default '.'
    Character to recognize as decimal point (e.g., use ',' for European data).
lineterminator : str (length 1), optional
    Character used to denote a line break. Only valid with C parser.
quotechar : str (length 1), optional
    Character used to denote the start and end of a quoted item. Quoted
    items can include the ``delimiter`` and it will be ignored.
quoting : {{0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL, 2 or csv.QUOTE_NONNUMERIC, 3 or csv.QUOTE_NONE}}, default csv.QUOTE_MINIMAL
    Control field quoting behavior per ``csv.QUOTE_*`` constants. Default is
    ``csv.QUOTE_MINIMAL`` (i.e., 0) which implies that only fields containing special
    characters are quoted (e.g., characters defined in ``quotechar``, ``delimiter``,
    or ``lineterminator``.
doublequote : bool, default True
   When ``quotechar`` is specified and ``quoting`` is not ``QUOTE_NONE``, indicate
   whether or not to interpret two consecutive ``quotechar`` elements INSIDE a
   field as a single ``quotechar`` element.
escapechar : str (length 1), optional
    Character used to escape other characters.
comment : str (length 1), optional
    Character indicating that the remainder of line should not be parsed.
    If found at the beginning
    of a line, the line will be ignored altogether. This parameter must be a
    single character. Like empty lines (as long as ``skip_blank_lines=True``),
    fully commented lines are ignored by the parameter ``header`` but not by
    ``skiprows``. For example, if ``comment='#'``, parsing
    ``#empty\\na,b,c\\n1,2,3`` with ``header=0`` will result in ``'a,b,c'`` being
    treated as the header.
encoding : str, optional, default 'utf-8'
    Encoding to use for UTF when reading/writing (ex. ``'utf-8'``). `List of Python
    standard encodings
    <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .

encoding_errors : str, optional, default 'strict'
    How encoding errors are treated. `List of possible values
    <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

    .. versionadded:: 1.3.0

dialect : str or csv.Dialect, optional
    If provided, this parameter will override values (default or not) for the
    following parameters: ``delimiter``, ``doublequote``, ``escapechar``,
    ``skipinitialspace``, ``quotechar``, and ``quoting``. If it is necessary to
    override values, a ``ParserWarning`` will be issued. See ``csv.Dialect``
    documentation for more details.
on_bad_lines : {{'error', 'warn', 'skip'}} or Callable, default 'error'
    Specifies what to do upon encountering a bad line (a line with too many fields).
    Allowed values are:

    - ``'error'``, raise an Exception when a bad line is encountered.
    - ``'warn'``, raise a warning when a bad line is encountered and skip that line.
    - ``'skip'``, skip bad lines without raising or warning when they are encountered.
    - Callable, function that will process a single bad line.
        - With ``engine='python'``, function with signature
          ``(bad_line: list[str]) -> list[str] | None``.
          ``bad_line`` is a list of strings split by the ``sep``.
          If the function returns ``None``, the bad line will be ignored.
          If the function returns a new ``list`` of strings with more elements than
          expected, a ``ParserWarning`` will be emitted while dropping extra elements.
        - With ``engine='pyarrow'``, function with signature
          as described in pyarrow documentation: `invalid_row_handler
          <https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html
          #pyarrow.csv.ParseOptions.invalid_row_handler>`_.

    .. versionadded:: 1.3.0

    .. versionadded:: 1.4.0

        Callable

    .. versionchanged:: 2.2.0

        Callable for ``engine='pyarrow'``

low_memory : bool, default True
    Internally process the file in chunks, resulting in lower memory use
    while parsing, but possibly mixed type inference.  To ensure no mixed
    types either set ``False``, or specify the type with the ``dtype`` parameter.
    Note that the entire file is read into a single :class:`~pandas.DataFrame`
    regardless, use the ``chunksize`` or ``iterator`` parameter to return the data in
    chunks. (Only valid with C parser).
memory_map : bool, default False
    If a filepath is provided for ``filepath_or_buffer``, map the file object
    directly onto memory and access the data directly from there. Using this
    option can improve performance because there is no longer any I/O overhead.
float_precision : {{'high', 'legacy', 'round_trip'}}, optional
    Specifies which converter the C engine should use for floating-point
    values. The options are ``None`` or ``'high'`` for the ordinary converter,
    ``'legacy'`` for the original lower precision pandas converter, and
    ``'round_trip'`` for the round-trip converter.

{storage_options}

dtype_backend : {{'numpy_nullable', 'pyarrow'}}
    Back-end data type applied to the resultant :class:`DataFrame`
    (still experimental). If not specified, the default behavior
    is to not use nullable data types. If specified, the behavior
    is as follows:

    * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
    * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype` :class:`DataFrame`

    .. versionadded:: 2.0

Returns
-------
DataFrame or TextFileReader
    A comma-separated values (csv) file is returned as two-dimensional
    data structure with labeled axes.

See Also
--------
DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
{see_also_func_name} : {see_also_func_summary}
read_fwf : Read a table of fixed-width formatted lines into DataFrame.

Examples
--------
>>> pd.{func_name}('data.csv')  # doctest: +SKIP
   Name  Value
0   foo      1
1   bar      2
2  #baz      3

Index and header can be specified via the `index_col` and `header` arguments.

>>> pd.{func_name}('data.csv', header=None)  # doctest: +SKIP
      0      1
0  Name  Value
1   foo      1
2   bar      2
3  #baz      3

>>> pd.{func_name}('data.csv', index_col='Value')  # doctest: +SKIP
       Name
Value
1       foo
2       bar
3      #baz

Column types are inferred but can be explicitly specified using the dtype argument.

>>> pd.{func_name}('data.csv', dtype={{'Value': float}})  # doctest: +SKIP
   Name  Value
0   foo    1.0
1   bar    2.0
2  #baz    3.0

True, False, and NA values, and thousands separators have defaults,
but can be explicitly specified, too. Supply the values you would like
as strings or lists of strings!

>>> pd.{func_name}('data.csv', na_values=['foo', 'bar'])  # doctest: +SKIP
   Name  Value
0   NaN      1
1   NaN      2
2  #baz      3

Comment lines in the input file can be skipped using the `comment` argument.

>>> pd.{func_name}('data.csv', comment='#')  # doctest: +SKIP
  Name  Value
0  foo      1
1  bar      2

By default, columns with dates will be read as ``object`` rather than  ``datetime``.

>>> df = pd.{func_name}('tmp.csv')  # doctest: +SKIP

>>> df  # doctest: +SKIP
   col 1       col 2            col 3
0     10  10/04/2018  Sun 15 Jan 2023
1     20  15/04/2018  Fri 12 May 2023

>>> df.dtypes  # doctest: +SKIP
col 1     int64
col 2    object
col 3    object
dtype: object

Specific columns can be parsed as dates by using the `parse_dates` and
`date_format` arguments.

>>> df = pd.{func_name}(
...     'tmp.csv',
...     parse_dates=[1, 2],
...     date_format={{'col 2': '%d/%m/%Y', 'col 3': '%a %d %b %Y'}},
... )  # doctest: +SKIP

>>> df.dtypes  # doctest: +SKIP
col 1             int64
col 2    datetime64[ns]
col 3    datetime64[ns]
dtype: object
"""  # noqa: E501


class _C_Parser_Defaults(TypedDict):
    na_filter: Literal[True]
    low_memory: Literal[True]
    memory_map: Literal[False]
    float_precision: None


_c_parser_defaults: _C_Parser_Defaults = {
    "na_filter": True,
    "low_memory": True,
    "memory_map": False,
    "float_precision": None,
}


class _Fwf_Defaults(TypedDict):
    colspecs: Literal["infer"]
    infer_nrows: Literal[100]
    widths: None


_fwf_defaults: _Fwf_Defaults = {"colspecs": "infer", "infer_nrows": 100, "widths": None}
_c_unsupported = {"skipfooter"}
_python_unsupported = {"low_memory", "float_precision"}
_pyarrow_unsupported = {
    "skipfooter",
    "float_precision",
    "chunksize",
    "comment",
    "nrows",
    "thousands",
    "memory_map",
    "dialect",
    "quoting",
    "lineterminator",
    "converters",
    "iterator",
    "dayfirst",
    "skipinitialspace",
    "low_memory",
}


@overload
def validate_integer(name: str, val: None, min_val: int = ...) -> None: ...


@overload
def validate_integer(name: str, val: float, min_val: int = ...) -> int: ...


@overload
def validate_integer(name: str, val: int | None, min_val: int = ...) -> int | None: ...


def validate_integer(
    name: str, val: int | float | None, min_val: int = 0
) -> int | None:
    """
    Checks whether the 'name' parameter for parsing is either
    an integer OR float that can SAFELY be cast to an integer
    without losing accuracy. Raises a ValueError if that is
    not the case.

    Parameters
    ----------
    name : str
        Parameter name (used for error reporting)
    val : int or float
        The value to check
    min_val : int
        Minimum allowed value (val < min_val will result in a ValueError)
    """
    if val is None:
        return val

    msg = f"'{name:s}' must be an integer >={min_val:d}"
    if is_float(val):
        if int(val) != val:
            raise ValueError(msg)
        val = int(val)
    elif not (is_integer(val) and val >= min_val):
        raise ValueError(msg)

    return int(val)


def _validate_names(names: Sequence[Hashable] | None) -> None:
    """
    Raise ValueError if the `names` parameter contains duplicates or has an
    invalid data type.

    Parameters
    ----------
    names : array-like or None
        An array containing a list of the names used for the output DataFrame.

    Raises
    ------
    ValueError
        If names are not unique or are not ordered (e.g. set).
    """
    if names is not None:
        if len(names) != len(set(names)):
            raise ValueError("Duplicate names are not allowed.")
        if not (
            is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView)
        ):
            raise ValueError("Names should be an ordered collection.")


def _read(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], kwds
) -> DataFrame | TextFileReader:
    """Generic reader of line files."""
    # if we pass a date_format and parse_dates=False, we should not parse the
    # dates GH#44366
    if kwds.get("parse_dates", None) is None:
        if kwds.get("date_format", None) is None:
            kwds["parse_dates"] = False
        else:
            kwds["parse_dates"] = True

    # Extract some of the arguments (pass chunksize on).
    iterator = kwds.get("iterator", False)
    chunksize = kwds.get("chunksize", None)

    # Check type of encoding_errors
    errors = kwds.get("encoding_errors", "strict")
    if not isinstance(errors, str):
        raise ValueError(
            f"encoding_errors must be a string, got {type(errors).__name__}"
        )

    if kwds.get("engine") == "pyarrow":
        if iterator:
            raise ValueError(
                "The 'iterator' option is not supported with the 'pyarrow' engine"
            )

        if chunksize is not None:
            raise ValueError(
                "The 'chunksize' option is not supported with the 'pyarrow' engine"
            )
    else:
        chunksize = validate_integer("chunksize", chunksize, 1)

    nrows = kwds.get("nrows", None)

    # Check for duplicates in names.
    _validate_names(kwds.get("names", None))

    # Create the parser.
    parser = TextFileReader(filepath_or_buffer, **kwds)

    if chunksize or iterator:
        return parser

    with parser:
        return parser.read(nrows)


@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> TextFileReader: ...


@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: bool = ...,
    chunksize: int,
    **kwds: Unpack[_read_shared[HashableT]],
) -> TextFileReader: ...


@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> DataFrame: ...


@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    iterator: bool = ...,
    chunksize: int | None = ...,
    **kwds: Unpack[_read_shared[HashableT]],
) -> DataFrame | TextFileReader: ...


@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_csv",
        summary="Read a comma-separated values (csv) file into DataFrame.",
        see_also_func_name="read_table",
        see_also_func_summary="Read general delimited file into DataFrame.",
        na_values_str=fill(
            '", "'.join(sorted(STR_NA_VALUES)), 70, subsequent_indent="    "
        ),
        _default_sep="','",
        storage_options=_shared_docs["storage_options"],
        decompression_options=_shared_docs["decompression_options"]
        % "filepath_or_buffer",
    )
)
@set_module("pandas")
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[HashableT, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] | None = None,
    date_format: str | dict[Hashable, str] | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression,