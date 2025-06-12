"""
Module contains tools for processing files into DataFrames or other objects

GH#48849 provides a convenient way of deprecating keyword arguments
"""
from __future__ import annotations
from collections import abc, defaultdict
import csv
import sys
from textwrap import fill
from typing import IO, TYPE_CHECKING, Any, Generic, Literal, TypedDict, overload
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import AbstractMethodError, ParserWarning
from pandas.util._decorators import Appender, set_module
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_file_like, is_float, is_integer, is_list_like, pandas_dtype
from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import IOHandles, get_handle, stringify_path, validate_header_arg
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper
from pandas.io.parsers.base_parser import ParserBase, is_index_col, parser_defaults
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
from pandas.io.parsers.python_parser import FixedWidthFieldParser, PythonParser
if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
    from types import TracebackType
    from pandas._typing import CompressionOptions, CSVEngine, DtypeArg, DtypeBackend, FilePath, HashableT, IndexLabel, ReadCsvBuffer, Self, StorageOptions, Unpack, UsecolsArgType

    class _read_shared(TypedDict, Generic[HashableT], total=False):
        pass
else:
    _read_shared = dict
_doc_read_csv_and_table = '\n{summary}\n\nAlso supports optionally iterating or breaking of the file\ninto chunks.\n\nAdditional help can be found in the online docs for\n`IO Tools <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.\n\nParameters\n----------\nfilepath_or_buffer : str, path object or file-like object\n    Any valid string path is acceptable. The string could be a URL. Valid\n    URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is\n    expected. A local file could be: file://localhost/path/to/table.csv.\n\n    If you want to pass in a path object, pandas accepts any ``os.PathLike``.\n\n    By file-like object, we refer to objects with a ``read()`` method, such as\n    a file handle (e.g. via builtin ``open`` function) or ``StringIO``.\nsep : str, default {_default_sep}\n    Character or regex pattern to treat as the delimiter. If ``sep=None``, the\n    C engine cannot automatically detect\n    the separator, but the Python parsing engine can, meaning the latter will\n    be used and automatically detect the separator from only the first valid\n    row of the file by Python\'s builtin sniffer tool, ``csv.Sniffer``.\n    In addition, separators longer than 1 character and different from\n    ``\'\\s+\'`` will be interpreted as regular expressions and will also force\n    the use of the Python parsing engine. Note that regex delimiters are prone\n    to ignoring quoted data. Regex example: ``\'\\r\\t\'``.\ndelimiter : str, optional\n    Alias for ``sep``.\nheader : int, Sequence of int, \'infer\' or None, default \'infer\'\n    Row number(s) containing column labels and marking the start of the\n    data (zero-indexed). Default behavior is to infer the column names: if no ``names``\n    are passed the behavior is identical to ``header=0`` and column\n    names are inferred from the first line of the file, if column\n    names are passed explicitly to ``names`` then the behavior is identical to\n    ``header=None``. Explicitly pass ``header=0`` to be able to\n    replace existing names. The header can be a list of integers that\n    specify row locations for a :class:`~pandas.MultiIndex` on the columns\n    e.g. ``[0, 1, 3]``. Intervening rows that are not specified will be\n    skipped (e.g. 2 in this example is skipped). Note that this\n    parameter ignores commented lines and empty lines if\n    ``skip_blank_lines=True``, so ``header=0`` denotes the first line of\n    data rather than the first line of the file.\n\n    When inferred from the file contents, headers are kept distinct from\n    each other by renaming duplicate names with a numeric suffix of the form\n    ``".{{count}}"`` starting from 1, e.g. ``"foo"`` and ``"foo.1"``.\n    Empty headers are named ``"Unnamed: {{i}}"`` or ``"Unnamed: {{i}}_level_{{level}}"``\n    in the case of MultiIndex columns.\nnames : Sequence of Hashable, optional\n    Sequence of column labels to apply. If the file contains a header row,\n    then you should explicitly pass ``header=0`` to override the column names.\n    Duplicates in this list are not allowed.\nindex_col : Hashable, Sequence of Hashable or False, optional\n  Column(s) to use as row label(s), denoted either by column labels or column\n  indices.  If a sequence of labels or indices is given, :class:`~pandas.MultiIndex`\n  will be formed for the row labels.\n\n  Note: ``index_col=False`` can be used to force pandas to *not* use the first\n  column as the index, e.g., when you have a malformed file with delimiters at\n  the end of each line.\nusecols : Sequence of Hashable or Callable, optional\n    Subset of columns to select, denoted either by column labels or column indices.\n    If list-like, all elements must either\n    be positional (i.e. integer indices into the document columns) or strings\n    that correspond to column names provided either by the user in ``names`` or\n    inferred from the document header row(s). If ``names`` are given, the document\n    header row(s) are not taken into account. For example, a valid list-like\n    ``usecols`` parameter would be ``[0, 1, 2]`` or ``[\'foo\', \'bar\', \'baz\']``.\n    Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.\n    To instantiate a :class:`~pandas.DataFrame` from ``data`` with element order\n    preserved use ``pd.read_csv(data, usecols=[\'foo\', \'bar\'])[[\'foo\', \'bar\']]``\n    for columns in ``[\'foo\', \'bar\']`` order or\n    ``pd.read_csv(data, usecols=[\'foo\', \'bar\'])[[\'bar\', \'foo\']]``\n    for ``[\'bar\', \'foo\']`` order.\n\n    If callable, the callable function will be evaluated against the column\n    names, returning names where the callable function evaluates to ``True``. An\n    example of a valid callable argument would be ``lambda x: x.upper() in\n    [\'AAA\', \'BBB\', \'DDD\']``. Using this parameter results in much faster\n    parsing time and lower memory usage.\ndtype : dtype or dict of {{Hashable : dtype}}, optional\n    Data type(s) to apply to either the whole dataset or individual columns.\n    E.g., ``{{\'a\': np.float64, \'b\': np.int32, \'c\': \'Int64\'}}``\n    Use ``str`` or ``object`` together with suitable ``na_values`` settings\n    to preserve and not interpret ``dtype``.\n    If ``converters`` are specified, they will be applied INSTEAD\n    of ``dtype`` conversion.\n\n    .. versionadded:: 1.5.0\n\n        Support for ``defaultdict`` was added. Specify a ``defaultdict`` as input where\n        the default determines the ``dtype`` of the columns which are not explicitly\n        listed.\nengine : {{\'c\', \'python\', \'pyarrow\'}}, optional\n    Parser engine to use. The C and pyarrow engines are faster, while the python engine\n    is currently more feature-complete. Multithreading is currently only supported by\n    the pyarrow engine.\n\n    .. versionadded:: 1.4.0\n\n        The \'pyarrow\' engine was added as an *experimental* engine, and some features\n        are unsupported, or may not work correctly, with this engine.\nconverters : dict of {{Hashable : Callable}}, optional\n    Functions for converting values in specified columns. Keys can either\n    be column labels or column indices.\ntrue_values : list, optional\n    Values to consider as ``True`` in addition to case-insensitive variants of \'True\'.\nfalse_values : list, optional\n    Values to consider as ``False`` in addition to case-insensitive variants of \'False\'.\nskipinitialspace : bool, default False\n    Skip spaces after delimiter.\nskiprows : int, list of int or Callable, optional\n    Line numbers to skip (0-indexed) or number of lines to skip (``int``)\n    at the start of the file.\n\n    If callable, the callable function will be evaluated against the row\n    indices, returning ``True`` if the row should be skipped and ``False`` otherwise.\n    An example of a valid callable argument would be ``lambda x: x in [0, 2]``.\nskipfooter : int, default 0\n    Number of lines at bottom of file to skip (Unsupported with ``engine=\'c\'``).\nnrows : int, optional\n    Number of rows of file to read. Useful for reading pieces of large files.\n    Refers to the number of data rows in the returned DataFrame, excluding:\n\n    * The header row containing column names.\n    * Rows before the header row, if ``header=1`` or larger.\n\n    Example usage:\n\n    * To read the first 999,999 (non-header) rows:\n      ``read_csv(..., nrows=999999)``\n\n    * To read rows 1,000,000 through 1,999,999:\n      ``read_csv(..., skiprows=1000000, nrows=999999)``\nna_values : Hashable, Iterable of Hashable or dict of {{Hashable : Iterable}}, optional\n    Additional strings to recognize as ``NA``/``NaN``. If ``dict`` passed, specific\n    per-column ``NA`` values.  By default the following values are interpreted as\n    ``NaN``: "{na_values_str}".\nkeep_default_na : bool, default True\n    Whether or not to include the default ``NaN`` values when parsing the data.\n    Depending on whether ``na_values`` is passed in, the behavior is as follows:\n\n    * If ``keep_default_na`` is ``True``, and ``na_values`` are specified, ``na_values``\n      is appended to the default ``NaN`` values used for parsing.\n    * If ``keep_default_na`` is ``True``, and ``na_values`` are not specified, only\n      the default ``NaN`` values are used for parsing.\n    * If ``keep_default_na`` is ``False``, and ``na_values`` are specified, only\n      the ``NaN`` values specified ``na_values`` are used for parsing.\n    * If ``keep_default_na`` is ``False``, and ``na_values`` are not specified, no\n      strings will be parsed as ``NaN``.\n\n    Note that if ``na_filter`` is passed in as ``False``, the ``keep_default_na`` and\n    ``na_values`` parameters will be ignored.\nna_filter : bool, default True\n    Detect missing value markers (empty strings and the value of ``na_values``). In\n    data without any ``NA`` values, passing ``na_filter=False`` can improve the\n    performance of reading a large file.\nskip_blank_lines : bool, default True\n    If ``True``, skip over blank lines rather than interpreting as ``NaN`` values.\nparse_dates : bool, None, list of Hashable, default None\n    The behavior is as follows:\n\n    * ``bool``. If ``True`` -> try parsing the index.\n    * ``None``. Behaves like ``True`` if ``date_format`` is specified.\n    * ``list`` of ``int`` or names. e.g. If ``[1, 2, 3]`` -> try parsing columns 1, 2, 3\n      each as a separate date column.\n\n    If a column or index cannot be represented as an array of ``datetime``,\n    say because of an unparsable value or a mixture of timezones, the column\n    or index will be returned unaltered as an ``object`` data type. For\n    non-standard ``datetime`` parsing, use :func:`~pandas.to_datetime` after\n    :func:`~pandas.read_csv`.\n\n    Note: A fast-path exists for iso8601-formatted dates.\ndate_format : str or dict of column -> format, optional\n    Format to use for parsing dates and/or times when used in conjunction with ``parse_dates``.\n    The strftime to parse time, e.g. :const:`"%d/%m/%Y"`. See\n    `strftime documentation\n    <https://docs.python.org/3/library/datetime.html\n    #strftime-and-strptime-behavior>`_ for more information on choices, though\n    note that :const:`"%f"` will parse all the way up to nanoseconds.\n    You can also pass:\n\n    - "ISO8601", to parse any `ISO8601 <https://en.wikipedia.org/wiki/ISO_8601>`_\n      time string (not necessarily in exactly the same format);\n    - "mixed", to infer the format for each element individually. This is risky,\n      and you should probably use it along with `dayfirst`.\n\n    .. versionadded:: 2.0.0\ndayfirst : bool, default False\n    DD/MM format dates, international and European format.\ncache_dates : bool, default True\n    If ``True``, use a cache of unique, converted dates to apply the ``datetime``\n    conversion. May produce significant speed-up when parsing duplicate\n    date strings, especially ones with timezone offsets.\n\niterator : bool, default False\n    Return ``TextFileReader`` object for iteration or getting chunks with\n    ``get_chunk()``.\nchunksize : int, optional\n    Number of lines to read from the file per chunk. Passing a value will cause the\n    function to return a ``TextFileReader`` object for iteration.\n    See the `IO Tools docs\n    <https://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking>`_\n    for more information on ``iterator`` and ``chunksize``.\n\n{decompression_options}\n\n    .. versionchanged:: 1.4.0 Zstandard support.\n\nthousands : str (length 1), optional\n    Character acting as the thousands separator in numerical values.\ndecimal : str (length 1), default \'.\'\n    Character to recognize as decimal point (e.g., use \',\' for European data).\nlineterminator : str (length 1), optional\n    Character used to denote a line break. Only valid with C parser.\nquotechar : str (length 1), optional\n    Character used to denote the start and end of a quoted item. Quoted\n    items can include the ``delimiter`` and it will be ignored.\nquoting : {{0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL, 2 or csv.QUOTE_NONNUMERIC, 3 or csv.QUOTE_NONE}}, default csv.QUOTE_MINIMAL\n    Control field quoting behavior per ``csv.QUOTE_*`` constants. Default is\n    ``csv.QUOTE_MINIMAL`` (i.e., 0) which implies that only fields containing special\n    characters are quoted (e.g., characters defined in ``quotechar``, ``delimiter``,\n    or ``lineterminator``.\ndoublequote : bool, default True\n   When ``quotechar`` is specified and ``quoting`` is not ``QUOTE_NONE``, indicate\n   whether or not to interpret two consecutive ``quotechar`` elements INSIDE a\n   field as a single ``quotechar`` element.\nescapechar : str (length 1), optional\n    Character used to escape other characters.\ncomment : str (length 1), optional\n    Character indicating that the remainder of line should not be parsed.\n    If found at the beginning\n    of a line, the line will be ignored altogether. This parameter must be a\n    single character. Like empty lines (as long as ``skip_blank_lines=True``),\n    fully commented lines are ignored by the parameter ``header`` but not by\n    ``skiprows``. For example, if ``comment=\'#\'``, parsing\n    ``#empty\\\\na,b,c\\\\n1,2,3`` with ``header=0`` will result in ``\'a,b,c\'`` being\n    treated as the header.\nencoding : str, optional, default \'utf-8\'\n    Encoding to use for UTF when reading/writing (ex. ``\'utf-8\'``). `List of Python\n    standard encodings\n    <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .\n\nencoding_errors : str, optional, default \'strict\'\n    How encoding errors are treated. `List of possible values\n    <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .\n\n    .. versionadded:: 1.3.0\n\ndialect : str or csv.Dialect, optional\n    If provided, this parameter will override values (default or not) for the\n    following parameters: ``delimiter``, ``doublequote``, ``escapechar``,\n    ``skipinitialspace``, ``quotechar``, and ``quoting``. If it is necessary to\n    override values, a ``ParserWarning`` will be issued. See ``csv.Dialect``\n    documentation for more details.\non_bad_lines : {{\'error\', \'warn\', \'skip\'}} or Callable, default \'error\'\n    Specifies what to do upon encountering a bad line (a line with too many fields).\n    Allowed values are:\n\n    - ``\'error\'``, raise an Exception when a bad line is encountered.\n    - ``\'warn\'``, raise a warning when a bad line is encountered and skip that line.\n    - ``\'skip\'``, skip bad lines without raising or warning when they are encountered.\n    - Callable, function that will process a single bad line.\n        - With ``engine=\'python\'``, function with signature\n          ``(bad_line: list[str]) -> list[str] | None``.\n          ``bad_line`` is a list of strings split by the ``sep``.\n          If the function returns ``None``, the bad line will be ignored.\n          If the function returns a new ``list`` of strings with more elements than\n          expected, a ``ParserWarning`` will be emitted while dropping extra elements.\n        - With ``engine=\'pyarrow\'``, function with signature\n          as described in pyarrow documentation: `invalid_row_handler\n          <https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html\n          #pyarrow.csv.ParseOptions.invalid_row_handler>`_.\n\n    .. versionadded:: 1.3.0\n\n    .. versionadded:: 1.4.0\n\n        Callable\n\n    .. versionchanged:: 2.2.0\n\n        Callable for ``engine=\'pyarrow\'``\n\nlow_memory : bool, default True\n    Internally process the file in chunks, resulting in lower memory use\n    while parsing, but possibly mixed type inference.  To ensure no mixed\n    types either set ``False``, or specify the type with the ``dtype`` parameter.\n    Note that the entire file is read into a single :class:`~pandas.DataFrame`\n    regardless, use the ``chunksize`` or ``iterator`` parameter to return the data in\n    chunks. (Only valid with C parser).\nmemory_map : bool, default False\n    If a filepath is provided for ``filepath_or_buffer``, map the file object\n    directly onto memory and access the data directly from there. Using this\n    option can improve performance because there is no longer any I/O overhead.\nfloat_precision : {{\'high\', \'legacy\', \'round_trip\'}}, optional\n    Specifies which converter the C engine should use for floating-point\n    values. The options are ``None`` or ``\'high\'`` for the ordinary converter,\n    ``\'legacy\'`` for the original lower precision pandas converter, and\n    ``\'round_trip\'`` for the round-trip converter.\n\n{storage_options}\n\ndtype_backend : {{\'numpy_nullable\', \'pyarrow\'}}\n    Back-end data type applied to the resultant :class:`DataFrame`\n    (still experimental). If not specified, the default behavior\n    is to not use nullable data types. If specified, the behavior\n    is as follows:\n\n    * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`\n    * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype` :class:`DataFrame`\n\n    .. versionadded:: 2.0\n\nReturns\n-------\nDataFrame or TextFileReader\n    A comma-separated values (csv) file is returned as two-dimensional\n    data structure with labeled axes.\n\nSee Also\n--------\nDataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.\n{see_also_func_name} : {see_also_func_summary}\nread_fwf : Read a table of fixed-width formatted lines into DataFrame.\n\nExamples\n--------\n>>> pd.{func_name}(\'data.csv\')  # doctest: +SKIP\n   Name  Value\n0   foo      1\n1   bar      2\n2  #baz      3\n\nIndex and header can be specified via the `index_col` and `header` arguments.\n\n>>> pd.{func_name}(\'data.csv\', header=None)  # doctest: +SKIP\n      0      1\n0  Name  Value\n1   foo      1\n2   bar      2\n3  #baz      3\n\n>>> pd.{func_name}(\'data.csv\', index_col=\'Value\')  # doctest: +SKIP\n       Name\nValue\n1       foo\n2       bar\n3      #baz\n\nColumn types are inferred but can be explicitly specified using the dtype argument.\n\n>>> pd.{func_name}(\'data.csv\', dtype={{\'Value\': float}})  # doctest: +SKIP\n   Name  Value\n0   foo    1.0\n1   bar    2.0\n2  #baz    3.0\n\nTrue, False, and NA values, and thousands separators have defaults,\nbut can be explicitly specified, too. Supply the values you would like\nas strings or lists of strings!\n\n>>> pd.{func_name}(\'data.csv\', na_values=[\'foo\', \'bar\'])  # doctest: +SKIP\n   Name  Value\n0   NaN      1\n1   NaN      2\n2  #baz      3\n\nComment lines in the input file can be skipped using the `comment` argument.\n\n>>> pd.{func_name}(\'data.csv\', comment=\'#\')  # doctest: +SKIP\n  Name  Value\n0  foo      1\n1  bar      2\n\nBy default, columns with dates will be read as ``object`` rather than  ``datetime``.\n\n>>> df = pd.{func_name}(\'tmp.csv\')  # doctest: +SKIP\n\n>>> df  # doctest: +SKIP\n   col 1       col 2            col 3\n0     10  10/04/2018  Sun 15 Jan 2023\n1     20  15/04/2018  Fri 12 May 2023\n\n>>> df.dtypes  # doctest: +SKIP\ncol 1     int64\ncol 2    object\ncol 3    object\ndtype: object\n\nSpecific columns can be parsed as dates by using the `parse_dates` and\n`date_format` arguments.\n\n>>> df = pd.{func_name}(\n...     \'tmp.csv\',\n...     parse_dates=[1, 2],\n...     date_format={{\'col 2\': \'%d/%m/%Y\', \'col 3\': \'%a %d %b %Y\'}},\n... )  # doctest: +SKIP\n\n>>> df.dtypes  # doctest: +SKIP\ncol 1             int64\ncol 2    datetime64[ns]\ncol 3    datetime64[ns]\ndtype: object\n'

class _C_Parser_Defaults(TypedDict):
    pass
_c_parser_defaults = {'na_filter': True, 'low_memory': True, 'memory_map': False, 'float_precision': None}

class _Fwf_Defaults(TypedDict):
    pass
_fwf_defaults = {'colspecs': 'infer', 'infer_nrows': 100, 'widths': None}
_c_unsupported = {'skipfooter'}
_python_unsupported = {'low_memory', 'float_precision'}
_pyarrow_unsupported = {'skipfooter', 'float_precision', 'chunksize', 'comment', 'nrows', 'thousands', 'memory_map', 'dialect', 'quoting', 'lineterminator', 'converters', 'iterator', 'dayfirst', 'skipinitialspace', 'low_memory'}

@overload
def validate_integer(name, val, min_val=...):
    ...

@overload
def validate_integer(name, val, min_val=...):
    ...

@overload
def validate_integer(name, val, min_val=...):
    ...

def validate_integer(name, val, min_val=0):
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

def _validate_names(names):
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
            raise ValueError('Duplicate names are not allowed.')
        if not (is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView)):
            raise ValueError('Names should be an ordered collection.')

def _read(filepath_or_buffer, kwds):
    """Generic reader of line files."""
    if kwds.get('parse_dates', None) is None:
        if kwds.get('date_format', None) is None:
            kwds['parse_dates'] = False
        else:
            kwds['parse_dates'] = True
    iterator = kwds.get('iterator', False)
    chunksize = kwds.get('chunksize', None)
    errors = kwds.get('encoding_errors', 'strict')
    if not isinstance(errors, str):
        raise ValueError(f'encoding_errors must be a string, got {type(errors).__name__}')
    if kwds.get('engine') == 'pyarrow':
        if iterator:
            raise ValueError("The 'iterator' option is not supported with the 'pyarrow' engine")
        if chunksize is not None:
            raise ValueError("The 'chunksize' option is not supported with the 'pyarrow' engine")
    else:
        chunksize = validate_integer('chunksize', chunksize, 1)
    nrows = kwds.get('nrows', None)
    _validate_names(kwds.get('names', None))
    parser = TextFileReader(filepath_or_buffer, **kwds)
    if chunksize or iterator:
        return parser
    with parser:
        return parser.read(nrows)

@overload
def read_csv(filepath_or_buffer, *, iterator, chunksize=..., **kwds):
    ...

@overload
def read_csv(filepath_or_buffer, *, iterator=..., chunksize, **kwds):
    ...

@overload
def read_csv(filepath_or_buffer, *, iterator=..., chunksize=..., **kwds):
    ...

@overload
def read_csv(filepath_or_buffer, *, iterator=..., chunksize=..., **kwds):
    ...

@Appender(_doc_read_csv_and_table.format(func_name='read_csv', summary='Read a comma-separated values (csv) file into DataFrame.', see_also_func_name='read_table', see_also_func_summary='Read general delimited file into DataFrame.', na_values_str=fill('", "'.join(sorted(STR_NA_VALUES)), 70, subsequent_indent='    '), _default_sep="','", storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'filepath_or_buffer'))
@set_module('pandas')
def read_csv(filepath_or_buffer, *, sep=lib.no_default, delimiter=None, header='infer', names=lib.no_default, index_col=None, usecols=None, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, skip_blank_lines=True, parse_dates=None, date_format=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None, quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True, escapechar=None, comment=None, encoding=None, encoding_errors='strict', dialect=None, on_bad_lines='error', low_memory=_c_parser_defaults['low_memory'], memory_map=False, float_precision=None, storage_options=None, dtype_backend=lib.no_default):
    kwds = locals().copy()
    del kwds['filepath_or_buffer']
    del kwds['sep']
    kwds_defaults = _refine_defaults_read(dialect, delimiter, engine, sep, on_bad_lines, names, defaults={'delimiter': ','}, dtype_backend=dtype_backend)
    kwds.update(kwds_defaults)
    return _read(filepath_or_buffer, kwds)

@overload
def read_table(filepath_or_buffer, *, iterator, chunksize=..., **kwds):
    ...

@overload
def read_table(filepath_or_buffer, *, iterator=..., chunksize, **kwds):
    ...

@overload
def read_table(filepath_or_buffer, *, iterator=..., chunksize=..., **kwds):
    ...

@overload
def read_table(filepath_or_buffer, *, iterator=..., chunksize=..., **kwds):
    ...

@Appender(_doc_read_csv_and_table.format(func_name='read_table', summary='Read general delimited file into DataFrame.', see_also_func_name='read_csv', see_also_func_summary='Read a comma-separated values (csv) file into DataFrame.', na_values_str=fill('", "'.join(sorted(STR_NA_VALUES)), 70, subsequent_indent='    '), _default_sep="'\\\\t' (tab-stop)", storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'filepath_or_buffer'))
@set_module('pandas')
def read_table(filepath_or_buffer, *, sep=lib.no_default, delimiter=None, header='infer', names=lib.no_default, index_col=None, usecols=None, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, skipfooter=0, nrows=None, na_values=None, keep_default_na=True, na_filter=True, skip_blank_lines=True, parse_dates=None, date_format=None, dayfirst=False, cache_dates=True, iterator=False, chunksize=None, compression='infer', thousands=None, decimal='.', lineterminator=None, quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True, escapechar=None, comment=None, encoding=None, encoding_errors='strict', dialect=None, on_bad_lines='error', low_memory=_c_parser_defaults['low_memory'], memory_map=False, float_precision=None, storage_options=None, dtype_backend=lib.no_default):
    kwds = locals().copy()
    del kwds['filepath_or_buffer']
    del kwds['sep']
    kwds_defaults = _refine_defaults_read(dialect, delimiter, engine, sep, on_bad_lines, names, defaults={'delimiter': '\t'}, dtype_backend=dtype_backend)
    kwds.update(kwds_defaults)
    return _read(filepath_or_buffer, kwds)

@overload
def read_fwf(filepath_or_buffer, *, colspecs=..., widths=..., infer_nrows=..., iterator, chunksize=..., **kwds):
    ...

@overload
def read_fwf(filepath_or_buffer, *, colspecs=..., widths=..., infer_nrows=..., iterator=..., chunksize, **kwds):
    ...

@overload
def read_fwf(filepath_or_buffer, *, colspecs=..., widths=..., infer_nrows=..., iterator=..., chunksize=..., **kwds):
    ...

@set_module('pandas')
def read_fwf(filepath_or_buffer, *, colspecs='infer', widths=None, infer_nrows=100, iterator=False, chunksize=None, **kwds):
    """
    Read a table of fixed-width formatted lines into DataFrame.

    Also supports optionally iterating or breaking of the file
    into chunks.

    Additional help can be found in the `online docs for IO Tools
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a text ``read()`` function.The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.csv``.
    colspecs : list of tuple (int, int) or 'infer'. optional
        A list of tuples giving the extents of the fixed-width
        fields of each line as half-open intervals (i.e.,  [from, to] ).
        String value 'infer' can be used to instruct the parser to try
        detecting the column specifications from the first 100 rows of
        the data which are not being skipped via skiprows (default='infer').
    widths : list of int, optional
        A list of field widths which can be used instead of 'colspecs' if
        the intervals are contiguous.
    infer_nrows : int, default 100
        The number of rows to consider when letting the parser determine the
        `colspecs`.
    iterator : bool, default False
        Return ``TextFileReader`` object for iteration or getting chunks with
        ``get_chunk()``.
    chunksize : int, optional
        Number of lines to read from the file per chunk.
    **kwds : optional
        Optional keyword arguments can be passed to ``TextFileReader``.

    Returns
    -------
    DataFrame or TextFileReader
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.

    See Also
    --------
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.

    Examples
    --------
    >>> pd.read_fwf("data.csv")  # doctest: +SKIP
    """
    if colspecs is None and widths is None:
        raise ValueError('Must specify either colspecs or widths')
    if colspecs not in (None, 'infer') and widths is not None:
        raise ValueError("You must specify only one of 'widths' and 'colspecs'")
    if widths is not None:
        colspecs, col = ([], 0)
        for w in widths:
            colspecs.append((col, col + w))
            col += w
    assert colspecs is not None
    names = kwds.get('names')
    if names is not None and names is not lib.no_default:
        if len(names) != len(colspecs) and colspecs != 'infer':
            len_index = 0
            if kwds.get('index_col') is not None:
                index_col = kwds.get('index_col')
                if index_col is not False:
                    if not is_list_like(index_col):
                        len_index = 1
                    else:
                        assert index_col is not lib.no_default
                        len_index = len(index_col)
            if kwds.get('usecols') is None and len(names) + len_index != len(colspecs):
                raise ValueError('Length of colspecs must match length of names')
    check_dtype_backend(kwds.setdefault('dtype_backend', lib.no_default))
    return _read(filepath_or_buffer, kwds | {'colspecs': colspecs, 'infer_nrows': infer_nrows, 'engine': 'python-fwf', 'iterator': iterator, 'chunksize': chunksize})

class TextFileReader(abc.Iterator):
    """

    Passed dialect overrides any of the related parser options

    """

    def __init__(self, f, engine=None, **kwds):
        if engine is not None:
            engine_specified = True
        else:
            engine = 'python'
            engine_specified = False
        self.engine = engine
        self._engine_specified = kwds.get('engine_specified', engine_specified)
        _validate_skipfooter(kwds)
        dialect = _extract_dialect(kwds)
        if dialect is not None:
            if engine == 'pyarrow':
                raise ValueError("The 'dialect' option is not supported with the 'pyarrow' engine")
            kwds = _merge_with_dialect_properties(dialect, kwds)
        if kwds.get('header', 'infer') == 'infer':
            kwds['header'] = 0 if kwds.get('names') is None else None
        self.orig_options = kwds
        self._currow = 0
        options = self._get_options_with_defaults(engine)
        options['storage_options'] = kwds.get('storage_options', None)
        self.chunksize = options.pop('chunksize', None)
        self.nrows = options.pop('nrows', None)
        self._check_file_or_buffer(f, engine)
        self.options, self.engine = self._clean_options(options, engine)
        if 'has_index_names' in kwds:
            self.options['has_index_names'] = kwds['has_index_names']
        self.handles = None
        self._engine = self._make_engine(f, self.engine)

    def close(self):
        if self.handles is not None:
            self.handles.close()
        self._engine.close()

    def _get_options_with_defaults(self, engine):
        kwds = self.orig_options
        options = {}
        for argname, default in parser_defaults.items():
            value = kwds.get(argname, default)
            if engine == 'pyarrow' and argname in _pyarrow_unsupported and (value != default) and (value != getattr(value, 'value', default)):
                raise ValueError(f"The {argname!r} option is not supported with the 'pyarrow' engine")
            options[argname] = value
        for argname, default in _c_parser_defaults.items():
            if argname in kwds:
                value = kwds[argname]
                if engine != 'c' and value != default:
                    if 'python' in engine and argname not in _python_unsupported:
                        pass
                    elif 'pyarrow' in engine and argname not in _pyarrow_unsupported:
                        pass
                    else:
                        raise ValueError(f'The {argname!r} option is not supported with the {engine!r} engine')
            else:
                value = default
            options[argname] = value
        if engine == 'python-fwf':
            for argname, default in _fwf_defaults.items():
                options[argname] = kwds.get(argname, default)
        return options

    def _check_file_or_buffer(self, f, engine):
        if is_file_like(f) and engine != 'c' and (not hasattr(f, '__iter__')):
            raise ValueError("The 'python' engine cannot iterate through this file buffer.")
        if hasattr(f, 'encoding'):
            file_encoding = f.encoding
            orig_reader_enc = self.orig_options.get('encoding', None)
            any_none = file_encoding is None or orig_reader_enc is None
            if file_encoding != orig_reader_enc and (not any_none):
                file_path = getattr(f, 'name', None)
                raise ValueError(f'The specified reader encoding {orig_reader_enc} is different from the encoding {file_encoding} of file {file_path}.')

    def _clean_options(self, options, engine):
        result = options.copy()
        fallback_reason = None
        if engine == 'c':
            if options['skipfooter'] > 0:
                fallback_reason = "the 'c' engine does not support skipfooter"
                engine = 'python'
        sep = options['delimiter']
        if sep is not None and len(sep) > 1:
            if engine == 'c' and sep == '\\s+':
                result['delim_whitespace'] = True
                del result['delimiter']
            elif engine not in ('python', 'python-fwf'):
                fallback_reason = f"the '{engine}' engine does not support regex separators (separators > 1 char and different from '\\s+' are interpreted as regex)"
                engine = 'python'
        elif sep is not None:
            encodeable = True
            encoding = sys.getfilesystemencoding() or 'utf-8'
            try:
                if len(sep.encode(encoding)) > 1:
                    encodeable = False
            except UnicodeDecodeError:
                encodeable = False
            if not encodeable and engine not in ('python', 'python-fwf'):
                fallback_reason = f"the separator encoded in {encoding} is > 1 char long, and the '{engine}' engine does not support such separators"
                engine = 'python'
        quotechar = options['quotechar']
        if quotechar is not None and isinstance(quotechar, (str, bytes)):
            if len(quotechar) == 1 and ord(quotechar) > 127 and (engine not in ('python', 'python-fwf')):
                fallback_reason = f"ord(quotechar) > 127, meaning the quotechar is larger than one byte, and the '{engine}' engine does not support such quotechars"
                engine = 'python'
        if fallback_reason and self._engine_specified:
            raise ValueError(fallback_reason)
        if engine == 'c':
            for arg in _c_unsupported:
                del result[arg]
        if 'python' in engine:
            for arg in _python_unsupported:
                if fallback_reason and result[arg] != _c_parser_defaults.get(arg):
                    raise ValueError(f"Falling back to the 'python' engine because {fallback_reason}, but this causes {arg!r} to be ignored as it is not supported by the 'python' engine.")
                del result[arg]
        if fallback_reason:
            warnings.warn(f"Falling back to the 'python' engine because {fallback_reason}; you can avoid this warning by specifying engine='python'.", ParserWarning, stacklevel=find_stack_level())
        index_col = options['index_col']
        names = options['names']
        converters = options['converters']
        na_values = options['na_values']
        skiprows = options['skiprows']
        validate_header_arg(options['header'])
        if index_col is True:
            raise ValueError("The value of index_col couldn't be 'True'")
        if is_index_col(index_col):
            if not isinstance(index_col, (list, tuple, np.ndarray)):
                index_col = [index_col]
        result['index_col'] = index_col
        names = list(names) if names is not None else names
        if converters is not None:
            if not isinstance(converters, dict):
                raise TypeError(f'Type converters must be a dict or subclass, input was a {type(converters).__name__}')
        else:
            converters = {}
        keep_default_na = options['keep_default_na']
        floatify = engine != 'pyarrow'
        na_values, na_fvalues = _clean_na_values(na_values, keep_default_na, floatify=floatify)
        if engine == 'pyarrow':
            if not is_integer(skiprows) and skiprows is not None:
                raise ValueError("skiprows argument must be an integer when using engine='pyarrow'")
        else:
            if is_integer(skiprows):
                skiprows = range(skiprows)
            if skiprows is None:
                skiprows = set()
            elif not callable(skiprows):
                skiprows = set(skiprows)
        result['names'] = names
        result['converters'] = converters
        result['na_values'] = na_values
        result['na_fvalues'] = na_fvalues
        result['skiprows'] = skiprows
        return (result, engine)

    def __next__(self):
        try:
            return self.get_chunk()
        except StopIteration:
            self.close()
            raise

    def _make_engine(self, f, engine='c'):
        mapping = {'c': CParserWrapper, 'python': PythonParser, 'pyarrow': ArrowParserWrapper, 'python-fwf': FixedWidthFieldParser}
        if engine not in mapping:
            raise ValueError(f'Unknown engine: {engine} (valid options are {mapping.keys()})')
        if not isinstance(f, list):
            is_text = True
            mode = 'r'
            if engine == 'pyarrow':
                is_text = False
                mode = 'rb'
            elif engine == 'c' and self.options.get('encoding', 'utf-8') == 'utf-8' and isinstance(stringify_path(f), str):
                is_text = False
                if 'b' not in mode:
                    mode += 'b'
            self.handles = get_handle(f, mode, encoding=self.options.get('encoding', None), compression=self.options.get('compression', None), memory_map=self.options.get('memory_map', False), is_text=is_text, errors=self.options.get('encoding_errors', 'strict'), storage_options=self.options.get('storage_options', None))
            assert self.handles is not None
            f = self.handles.handle
        elif engine != 'python':
            msg = f'Invalid file path or buffer object type: {type(f)}'
            raise ValueError(msg)
        try:
            return mapping[engine](f, **self.options)
        except Exception:
            if self.handles is not None:
                self.handles.close()
            raise

    def _failover_to_python(self):
        raise AbstractMethodError(self)

    def read(self, nrows=None):
        if self.engine == 'pyarrow':
            try:
                df = self._engine.read()
            except Exception:
                self.close()
                raise
        else:
            nrows = validate_integer('nrows', nrows)
            try:
                index, columns, col_dict = self._engine.read(nrows)
            except Exception:
                self.close()
                raise
            if index is None:
                if col_dict:
                    new_rows = len(next(iter(col_dict.values())))
                    index = RangeIndex(self._currow, self._currow + new_rows)
                else:
                    new_rows = 0
            else:
                new_rows = len(index)
            if hasattr(self, 'orig_options'):
                dtype_arg = self.orig_options.get('dtype', None)
            else:
                dtype_arg = None
            if isinstance(dtype_arg, dict):
                dtype = defaultdict(lambda: None)
                dtype.update(dtype_arg)
            elif dtype_arg is not None and pandas_dtype(dtype_arg) in (np.str_, np.object_):
                dtype = defaultdict(lambda: dtype_arg)
            else:
                dtype = None
            if dtype is not None:
                new_col_dict = {}
                for k, v in col_dict.items():
                    d = dtype[k] if pandas_dtype(dtype[k]) in (np.str_, np.object_) else None
                    new_col_dict[k] = Series(v, index=index, dtype=d, copy=False)
            else:
                new_col_dict = col_dict
            df = DataFrame(new_col_dict, columns=columns, index=index, copy=False)
            self._currow += new_rows
        return df

    def get_chunk(self, size=None):
        if size is None:
            size = self.chunksize
        if self.nrows is not None:
            if self._currow >= self.nrows:
                raise StopIteration
            if size is None:
                size = self.nrows - self._currow
            else:
                size = min(size, self.nrows - self._currow)
        return self.read(nrows=size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def TextParser(*args, **kwds):
    """
    Converts lists of lists/tuples into DataFrames with proper type inference
    and optional (e.g. string to datetime) conversion. Also enables iterating
    lazily over chunks of large files

    Parameters
    ----------
    data : file-like object or list
    delimiter : separator character to use
    dialect : str or csv.Dialect instance, optional
        Ignored if delimiter is longer than 1 character
    names : sequence, default
    header : int, default 0
        Row to use to parse column labels. Defaults to the first row. Prior
        rows will be discarded
    index_col : int or list, optional
        Column or columns to use as the (possibly hierarchical) index
    has_index_names: bool, default False
        True if the cols defined in index_col have an index name and are
        not in the header.
    na_values : scalar, str, list-like, or dict, optional
        Additional strings to recognize as NA/NaN.
    keep_default_na : bool, default True
    thousands : str, optional
        Thousands separator
    comment : str, optional
        Comment out remainder of line
    parse_dates : bool, default False
    date_format : str or dict of column -> format, default ``None``

        .. versionadded:: 2.0.0
    skiprows : list of integers
        Row numbers to skip
    skipfooter : int
        Number of line at bottom of file to skip
    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the cell (not column) content, and return the
        transformed content.
    encoding : str, optional
        Encoding to use for UTF when reading/writing (ex. 'utf-8')
    float_precision : str, optional
        Specifies which converter the C engine should use for floating-point
        values. The options are `None` or `high` for the ordinary converter,
        `legacy` for the original lower precision pandas converter, and
        `round_trip` for the round-trip converter.
    """
    kwds['engine'] = 'python'
    return TextFileReader(*args, **kwds)

def _clean_na_values(na_values, keep_default_na=True, floatify=True):
    if na_values is None:
        if keep_default_na:
            na_values = STR_NA_VALUES
        else:
            na_values = set()
        na_fvalues = set()
    elif isinstance(na_values, dict):
        old_na_values = na_values.copy()
        na_values = {}
        for k, v in old_na_values.items():
            if not is_list_like(v):
                v = [v]
            if keep_default_na:
                v = set(v) | STR_NA_VALUES
            na_values[k] = _stringify_na_values(v, floatify)
        na_fvalues = {k: _floatify_na_values(v) for k, v in na_values.items()}
    else:
        if not is_list_like(na_values):
            na_values = [na_values]
        na_values = _stringify_na_values(na_values, floatify)
        if keep_default_na:
            na_values = na_values | STR_NA_VALUES
        na_fvalues = _floatify_na_values(na_values)
    return (na_values, na_fvalues)

def _floatify_na_values(na_values):
    result = set()
    for v in na_values:
        try:
            v = float(v)
            if not np.isnan(v):
                result.add(v)
        except (TypeError, ValueError, OverflowError):
            pass
    return result

def _stringify_na_values(na_values, floatify):
    """return a stringified and numeric for these values"""
    result = []
    for x in na_values:
        result.append(str(x))
        result.append(x)
        try:
            v = float(x)
            if v == int(v):
                v = int(v)
                result.append(f'{v}.0')
                result.append(str(v))
            if floatify:
                result.append(v)
        except (TypeError, ValueError, OverflowError):
            pass
        if floatify:
            try:
                result.append(int(x))
            except (TypeError, ValueError, OverflowError):
                pass
    return set(result)

def _refine_defaults_read(dialect, delimiter, engine, sep, on_bad_lines, names, defaults, dtype_backend):
    """Validate/refine default values of input parameters of read_csv, read_table.

    Parameters
    ----------
    dialect : str or csv.Dialect
        If provided, this parameter will override values (default or not) for the
        following parameters: `delimiter`, `doublequote`, `escapechar`,
        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to
        override values, a ParserWarning will be issued. See csv.Dialect
        documentation for more details.
    delimiter : str or object
        Alias for sep.
    engine : {{'c', 'python'}}
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    sep : str or object
        A delimiter provided by the user (str) or a sentinel value, i.e.
        pandas._libs.lib.no_default.
    on_bad_lines : str, callable
        An option for handling bad lines or a sentinel value(None).
    names : array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass ``header=0`` to override the column names.
        Duplicates in this list are not allowed.
    defaults: dict
        Default values of input parameters.

    Returns
    -------
    kwds : dict
        Input parameters with correct values.
    """
    delim_default = defaults['delimiter']
    kwds = {}
    if dialect is not None:
        kwds['sep_override'] = delimiter is None and (sep is lib.no_default or sep == delim_default)
    if delimiter and sep is not lib.no_default:
        raise ValueError('Specified a sep and a delimiter; you can only specify one.')
    kwds['names'] = None if names is lib.no_default else names
    if delimiter is None:
        delimiter = sep
    if delimiter == '\n':
        raise ValueError('Specified \\n as separator or delimiter. This forces the python engine which does not accept a line terminator. Hence it is not allowed to use the line terminator as separator.')
    if delimiter is lib.no_default:
        kwds['delimiter'] = delim_default
    else:
        kwds['delimiter'] = delimiter
    if engine is not None:
        kwds['engine_specified'] = True
    else:
        kwds['engine'] = 'c'
        kwds['engine_specified'] = False
    if on_bad_lines == 'error':
        kwds['on_bad_lines'] = ParserBase.BadLineHandleMethod.ERROR
    elif on_bad_lines == 'warn':
        kwds['on_bad_lines'] = ParserBase.BadLineHandleMethod.WARN
    elif on_bad_lines == 'skip':
        kwds['on_bad_lines'] = ParserBase.BadLineHandleMethod.SKIP
    elif callable(on_bad_lines):
        if engine not in ['python', 'pyarrow']:
            raise ValueError("on_bad_line can only be a callable function if engine='python' or 'pyarrow'")
        kwds['on_bad_lines'] = on_bad_lines
    else:
        raise ValueError(f'Argument {on_bad_lines} is invalid for on_bad_lines')
    check_dtype_backend(dtype_backend)
    kwds['dtype_backend'] = dtype_backend
    return kwds

def _extract_dialect(kwds):
    """
    Extract concrete csv dialect instance.

    Returns
    -------
    csv.Dialect or None
    """
    if kwds.get('dialect') is None:
        return None
    dialect = kwds['dialect']
    if dialect in csv.list_dialects():
        dialect = csv.get_dialect(dialect)
    _validate_dialect(dialect)
    return dialect
MANDATORY_DIALECT_ATTRS = ('delimiter', 'doublequote', 'escapechar', 'skipinitialspace', 'quotechar', 'quoting')

def _validate_dialect(dialect):
    """
    Validate csv dialect instance.

    Raises
    ------
    ValueError
        If incorrect dialect is provided.
    """
    for param in MANDATORY_DIALECT_ATTRS:
        if not hasattr(dialect, param):
            raise ValueError(f'Invalid dialect {dialect} provided')

def _merge_with_dialect_properties(dialect, defaults):
    """
    Merge default kwargs in TextFileReader with dialect parameters.

    Parameters
    ----------
    dialect : csv.Dialect
        Concrete csv dialect. See csv.Dialect documentation for more details.
    defaults : dict
        Keyword arguments passed to TextFileReader.

    Returns
    -------
    kwds : dict
        Updated keyword arguments, merged with dialect parameters.
    """
    kwds = defaults.copy()
    for param in MANDATORY_DIALECT_ATTRS:
        dialect_val = getattr(dialect, param)
        parser_default = parser_defaults[param]
        provided = kwds.get(param, parser_default)
        conflict_msgs = []
        if provided not in (parser_default, dialect_val):
            msg = f"Conflicting values for '{param}': '{provided}' was provided, but the dialect specifies '{dialect_val}'. Using the dialect-specified value."
            if not (param == 'delimiter' and kwds.pop('sep_override', False)):
                conflict_msgs.append(msg)
        if conflict_msgs:
            warnings.warn('\n\n'.join(conflict_msgs), ParserWarning, stacklevel=find_stack_level())
        kwds[param] = dialect_val
    return kwds

def _validate_skipfooter(kwds):
    """
    Check whether skipfooter is compatible with other kwargs in TextFileReader.

    Parameters
    ----------
    kwds : dict
        Keyword arguments passed to TextFileReader.

    Raises
    ------
    ValueError
        If skipfooter is not compatible with other parameters.
    """
    if kwds.get('skipfooter'):
        if kwds.get('iterator') or kwds.get('chunksize'):
            raise ValueError("'skipfooter' not supported for iteration")
        if kwds.get('nrows'):
            raise ValueError("'skipfooter' not supported with 'nrows'")