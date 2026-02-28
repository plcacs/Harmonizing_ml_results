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

class _C_Parser_Defaults(TypedDict):
    pass
_c_parser_defaults: _C_Parser_Defaults = {'na_filter': True, 'low_memory': True, 'memory_map': False, 'float_precision': None}

class _Fwf_Defaults(TypedDict):
    pass
_fwf_defaults: _Fwf_Defaults = {'colspecs': 'infer', 'infer_nrows': 100, 'widths': None}

@overload
def validate_integer(name: str, val: int | float, min_val: int | None = ...) -> int:
    ...

@overload
def validate_integer(name: str, val: int | float, min_val: int | None = ...) -> int:
    ...

@overload
def validate_integer(name: str, val: int | float, min_val: int | None = ...) -> int:
    ...

def validate_integer(name: str, val: int | float, min_val: int | None = 0) -> int:
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

def _validate_names(names: Sequence[Any] | None) -> None:
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

def _read(filepath_or_buffer: IO[Any], kwds: _read_shared) -> TextFileReader | DataFrame:
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
def read_csv(filepath_or_buffer: IO[Any], *, iterator: bool, chunksize: int | None = ..., **kwds: _read_shared) -> TextFileReader:
    ...

@overload
def read_csv(filepath_or_buffer: IO[Any], *, iterator: bool | None = ..., chunksize: int, **kwds: _read_shared) -> TextFileReader:
    ...

@overload
def read_csv(filepath_or_buffer: IO[Any], *, iterator: bool | None = ..., chunksize: int | None = ..., **kwds: _read_shared) -> TextFileReader:
    ...

@overload
def read_csv(filepath_or_buffer: IO[Any], *, iterator: bool | None = ..., chunksize: int | None = ..., **kwds: _read_shared) -> TextFileReader:
    ...

@Appender(_doc_read_csv_and_table.format(func_name='read_csv', summary='Read a comma-separated values (csv) file into DataFrame.', see_also_func_name='read_table', see_also_func_summary='Read general delimited file into DataFrame.', na_values_str=fill('", "'.join(sorted(STR_NA_VALUES)), 70, subsequent_indent='    '), _default_sep="','", storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'filepath_or_buffer'))
@set_module('pandas')
def read_csv(filepath_or_buffer: IO[Any], *, sep: str | None = lib.no_default, delimiter: str | None = None, header: int | Sequence[int] | None = 'infer', names: Sequence[Any] | None = lib.no_default, index_col: Hashable | Sequence[Hashable] | None = None, usecols: Sequence[Hashable] | Callable[[Hashable], bool] | None = None, dtype: DtypeArg | None = None, engine: CSVEngine | None = None, converters: Mapping[Hashable, Callable[[Any], Any]] | None = None, true_values: Sequence[Any] | None = None, false_values: Sequence[Any] | None = None, skipinitialspace: bool = False, skiprows: Sequence[int] | Callable[[int], bool] | None = None, skipfooter: int = 0, nrows: int | None = None, na_values: Hashable | Sequence[Any] | Mapping[Hashable, Sequence[Any]] | None = None, keep_default_na: bool = True, na_filter: bool = True, skip_blank_lines: bool = True, parse_dates: bool | Sequence[int] | None = None, date_format: str | Mapping[Hashable, str] | None = None, dayfirst: bool = False, cache_dates: bool = True, iterator: bool = False, chunksize: int | None = None, compression: CompressionOptions = 'infer', thousands: str | None = None, decimal: str = '.', lineterminator: str | None = None, quotechar: str = '"', quoting: int = csv.QUOTE_MINIMAL, doublequote: bool = True, escapechar: str | None = None, comment: str | None = None, encoding: str | None = 'utf-8', encoding_errors: str = 'strict', dialect: str | csv.Dialect | None = None, on_bad_lines: Literal['error', 'warn', 'skip'] | Callable[[Sequence[str]], Sequence[str] | None] = 'error', low_memory: bool = _c_parser_defaults['low_memory'], memory_map: bool = False, float_precision: Literal['high', 'legacy', 'round_trip'] | None = None, storage_options: StorageOptions | None = None, dtype_backend: DtypeBackend | None = lib.no_default) -> DataFrame | TextFileReader:
    kwds = locals().copy()
    del kwds['filepath_or_buffer']
    del kwds['sep']
    kwds_defaults = _refine_defaults_read(dialect, delimiter, engine, sep, on_bad_lines, names, defaults={'delimiter': ','}, dtype_backend=dtype_backend)
    kwds.update(kwds_defaults)
    return _read(filepath_or_buffer, kwds)

@overload
def read_table(filepath_or_buffer: IO[Any], *, iterator: bool, chunksize: int | None = ..., **kwds: _read_shared) -> TextFileReader:
    ...

@overload
def read_table(filepath_or_buffer: IO[Any], *, iterator: bool | None = ..., chunksize: int, **kwds: _read_shared) -> TextFileReader:
    ...

@overload
def read_table(filepath_or_buffer: IO[Any], *, iterator: bool | None = ..., chunksize: int | None = ..., **kwds: _read_shared) -> TextFileReader:
    ...

@overload
def read_table(filepath_or_buffer: IO[Any], *, iterator: bool | None = ..., chunksize: int | None = ..., **kwds: _read_shared) -> TextFileReader:
    ...

@Appender(_doc_read_csv_and_table.format(func_name='read_table', summary='Read general delimited file into DataFrame.', see_also_func_name='read_csv', see_also_func_summary='Read a comma-separated values (csv) file into DataFrame.', na_values_str=fill('", "'.join(sorted(STR_NA_VALUES)), 70, subsequent_indent='    '), _default_sep="'\\\\t' (tab-stop)", storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'filepath_or_buffer'))
@set_module('pandas')
def read_table(filepath_or_buffer: IO[Any], *, sep: str | None = lib.no_default, delimiter: str | None = None, header: int | Sequence[int] | None = 'infer', names: Sequence[Any] | None = lib.no_default, index_col: Hashable | Sequence[Hashable] | None = None, usecols: Sequence[Hashable] | Callable[[Hashable], bool] | None = None, dtype: DtypeArg | None = None, engine: CSVEngine | None = None, converters: Mapping[Hashable, Callable[[Any], Any]] | None = None, true_values: Sequence[Any] | None = None, false_values: Sequence[Any] | None = None, skipinitialspace: bool = False, skiprows: Sequence[int] | Callable[[int], bool] | None = None, skipfooter: int = 0, nrows: int | None = None, na_values: Hashable | Sequence[Any] | Mapping[Hashable, Sequence[Any]] | None = None, keep_default_na: bool = True, na_filter: bool = True, skip_blank_lines: bool = True, parse_dates: bool | Sequence[int] | None = None, date_format: str | Mapping[Hashable, str] | None = None, dayfirst: bool = False, cache_dates: bool = True, iterator: bool = False, chunksize: int | None = None, compression: CompressionOptions = 'infer', thousands: str | None = None, decimal: str = '.', lineterminator: str | None = None, quotechar: str = '"', quoting: int = csv.QUOTE_MINIMAL, doublequote: bool = True, escapechar: str | None = None, comment: str | None = None, encoding: str | None = 'utf-8', encoding_errors: str = 'strict', dialect: str | csv.Dialect | None = None, on_bad_lines: Literal['error', 'warn', 'skip'] | Callable[[Sequence[str]], Sequence[str] | None] = 'error', low_memory: bool = _c_parser_defaults['low_memory'], memory_map: bool = False, float_precision: Literal['high', 'legacy', 'round_trip'] | None = None, storage_options: StorageOptions | None = None, dtype_backend: DtypeBackend | None = lib.no_default) -> DataFrame | TextFileReader:
    kwds = locals().copy()
    del kwds['filepath_or_buffer']
    del kwds['sep']
    kwds_defaults = _refine_defaults_read(dialect, delimiter, engine, sep, on_bad_lines, names, defaults={'delimiter': '\t'}, dtype_backend=dtype_backend)
    kwds.update(kws_defaults)
    return _read(filepath_or_buffer, kwds)

@overload
def read_fwf(filepath_or_buffer: IO[Any], *, colspecs: Sequence[tuple[int, int]] | None = ..., widths: Sequence[int] | None = ..., infer_nrows: int = 100, iterator: bool, chunksize: int | None = ..., **kwds: _read_shared) -> TextFileReader:
    ...

@overload
def read_fwf(filepath_or_buffer: IO[Any], *, colspecs: Sequence[tuple[int, int]] | None = ..., widths: Sequence[int] | None = ..., infer_nrows: int = 100, iterator: bool | None = ..., chunksize: int, **kwds: _read_shared) -> TextFileReader:
    ...

@overload
def read_fwf(filepath_or_buffer: IO[Any], *, colspecs: Sequence[tuple[int, int]] | None = ..., widths: Sequence[int] | None = ..., infer_nrows: int = 100, iterator: bool | None = ..., chunksize: int | None = ..., **kwds: _read_shared) -> TextFileReader:
    ...

@set_module('pandas')
def read_fwf(filepath_or_buffer: IO[Any], *, colspecs: Sequence[tuple[int, int]] | None = ..., widths: Sequence[int] | None = ..., infer_nrows: int = 100, iterator: bool = False, chunksize: int | None = None, **kwds: _read_shared) -> DataFrame | TextFileReader:
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

    def __init__(self, f: IO[Any], engine: CSVEngine | None = None, **kwds: _read_shared):
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

    def close(self) -> None:
        if self.handles is not None:
            self.handles.close()
        self._engine.close()

    def _get_options_with_defaults(self, engine: CSVEngine | None) -> dict[str, Any]:
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

    def _check_file_or_buffer(self, f: IO[Any], engine: CSVEngine | None) -> None:
        if is_file_like(f) and engine != 'c' and (not hasattr(f, '__iter__')):
            raise ValueError("The 'python' engine cannot iterate through this file buffer.")
        if hasattr(f, 'encoding'):
            file_encoding = f.encoding
            orig_reader_enc = self.orig_options.get('encoding', None)
            any_none = file_encoding is None or orig_reader_enc is None
            if file_encoding != orig_reader_enc and (not any_none):
                file_path = getattr(f, 'name', None)
                raise ValueError(f'The specified reader encoding {orig_reader_enc} is different from the encoding {file_encoding} of file {file_path}.')

    def _clean_options(self, options: dict[str, Any], engine: CSVEngine | None) -> tuple[dict[str, Any], CSVEngine]:
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

    def __next__(self) -> DataFrame:
        try:
            return self.get_chunk()
        except StopIteration:
            self.close()
            raise

    def _make_engine(self, f: IO[Any], engine: CSVEngine | None = 'c') -> ParserBase:
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

    def _failover_to_python(self) -> None:
        raise AbstractMethodError(self)

    def read(self, nrows: int | None = None) -> DataFrame:
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

    def get_chunk(self, size: int | None = None) -> DataFrame:
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

    def __enter__(self) -> TextFileReader:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.close()

def TextParser(*args: Any, **kwds: _read_shared) -> TextFileReader:
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

def _clean_na_values(na_values: Hashable | Sequence[Any] | Mapping[Hashable, Sequence[Any]] | None, keep_default_na: bool, floatify: bool) -> tuple[Hashable | Sequence[Any], Hashable | Sequence[Any]]:
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

def _floatify_na_values(na_values: Hashable | Sequence[Any]) -> Hashable | Sequence[Any]:
    result = set()
    for v in na_values:
        try:
            v = float(v)
            if not np.isnan(v):
                result.add(v)
        except (TypeError, ValueError, OverflowError):
            pass
    return result

def _stringify_na_values(na_values: Hashable | Sequence[Any], floatify: bool) -> Hashable | Sequence[Any]:
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

def _refine_defaults_read(dialect: str | csv.Dialect | None, delimiter: str | None, engine: CSVEngine | None, sep: str | None, on_bad_lines: str | Callable[[Sequence[str]], Sequence[str] | None], names: Sequence[Any] | None, defaults: dict[str, Any], dtype_backend: DtypeBackend | None) -> dict[str, Any]:
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

def _extract_dialect(kwds: dict[str, Any]) -> csv.Dialect | None:
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

def _validate_dialect(dialect: csv.Dialect) -> None:
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

def _merge_with_dialect_properties(dialect: csv.Dialect, defaults: dict[str, Any]) -> dict[str, Any]:
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

def _validate_skipfooter(kwds: dict[str, Any]) -> None:
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
