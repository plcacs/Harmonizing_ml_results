from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import datetime
from decimal import Decimal
from functools import partial
import os
from textwrap import fill
from typing import IO, Optional, Any, Generic, Literal, TypeVar, Union, cast, overload, Dict, List, Sequence as TypingSequence
import warnings
import zipfile

from pandas._config import config
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import get_version, import_optional_dependency
from pandas.errors import EmptyDataError
from pandas.util._decorators import Appender, doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import is_bool, is_decimal, is_file_like, is_float, is_integer, is_list_like
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util.version import Version
from pandas.io.common import IOHandles, get_handle, stringify_path, validate_header_arg
from pandas.io.excel._util import fill_mi_header, get_default_engine, get_writer, maybe_convert_usecols, pop_header_name
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from types import TracebackType
    from pandas._typing import DtypeArg, DtypeBackend, ExcelWriterIfSheetExists, FilePath, HashableT, IntStrT, ReadBuffer, Self, SequenceNotStr, StorageOptions, WriteExcelBuffer

# Type alias for Excel sheet name which can be str or int.
SheetName = Union[str, int]
SheetNameList = List[SheetName]

_read_excel_doc: str = (
    "\nRead an Excel file into a ``pandas`` ``DataFrame``.\n\n"
    "Supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt` file extensions\n"
    "read from a local filesystem or URL. Supports an option to read\n"
    "a single sheet or a list of sheets.\n\n"
    "Parameters\n"
    "----------\n"
    "io : str, ExcelFile, xlrd.Book, path object, or file-like object\n"
    "    Any valid string path is acceptable. The string could be a URL. Valid\n"
    "    URL schemes include http, ftp, s3, and file. For file URLs, a host is\n"
    "    expected. A local file could be: ``file://localhost/path/to/table.xlsx``.\n\n"
    "    If you want to pass in a path object, pandas accepts any ``os.PathLike``.\n\n"
    "    By file-like object, we refer to objects with a ``read()`` method,\n"
    "    such as a file handle (e.g. via builtin ``open`` function)\n"
    "    or ``StringIO``.\n\n"
    "    .. deprecated:: 2.1.0\n"
    "        Passing byte strings is deprecated. To read from a\n"
    "        byte string, wrap it in a ``BytesIO`` object.\n"
    "sheet_name : str, int, list, or None, default 0\n"
    "    Strings are used for sheet names. Integers are used in zero-indexed\n"
    "    sheet positions (chart sheets do not count as a sheet position).\n"
    "    Lists of strings/integers are used to request multiple sheets.\n"
    "    When ``None``, will return a dictionary containing DataFrames for each sheet.\n\n"
    "    Available cases:\n\n"
    "    * Defaults to ``0``: 1st sheet as a `DataFrame`\n"
    "    * ``1``: 2nd sheet as a `DataFrame`\n"
    "    * ``\"Sheet1\"``: Load sheet with name \"Sheet1\"\n"
    "    * ``[0, 1, \"Sheet5\"]``: Load first, second and sheet named \"Sheet5\"\n"
    "      as a dict of `DataFrame`\n"
    "    * ``None``: Returns a dictionary containing DataFrames for each sheet..\n\n"
    "header : int, list of int, default 0\n"
    "    Row (0-indexed) to use for the column labels of the parsed\n"
    "    DataFrame. If a list of integers is passed those row positions will\n"
    "    be combined into a ``MultiIndex``. Use None if there is no header.\n"
    "names : array-like, default None\n"
    "    List of column names to use. If file contains no header row,\n"
    "    then you should explicitly pass header=None.\n"
    "index_col : int, str, list of int, default None\n"
    "    Column (0-indexed) to use as the row labels of the DataFrame.\n"
    "    Pass None if there is no such column.  If a list is passed,\n"
    "    those columns will be combined into a ``MultiIndex``.  If a\n"
    "    subset of data is selected with ``usecols``, index_col\n"
    "    is based on the subset.\n\n"
    "    Missing values will be forward filled to allow roundtripping with\n"
    "    ``to_excel`` for ``merged_cells=True``. To avoid forward filling the\n"
    "    missing values use ``set_index`` after reading the data instead of\n"
    "    ``index_col``.\n"
    "usecols : str, list-like, or callable, default None\n"
    "    * If None, then parse all columns.\n"
    "    * If str, then indicates comma separated list of Excel column letters\n"
    "      and column ranges (e.g. \"A:E\" or \"A,C,E:F\"). Ranges are inclusive of\n"
    "      both sides.\n"
    "    * If list of int, then indicates list of column numbers to be parsed\n"
    "      (0-indexed).\n"
    "    * If list of string, then indicates list of column names to be parsed.\n"
    "    * If callable, then evaluate each column name against it and parse the\n"
    "      column if the callable returns ``True``.\n\n"
    "    Returns a subset of the columns according to behavior above.\n"
    "dtype : Type name or dict of column -> type, default None\n"
    "    Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32}}\n"
    "    Use ``object`` to preserve data as stored in Excel and not interpret dtype,\n"
    "    which will necessarily result in ``object`` dtype.\n"
    "    If converters are specified, they will be applied INSTEAD\n"
    "    of dtype conversion.\n"
    "    If you use ``None``, it will infer the dtype of each column based on the data.\n"
    "engine : {{'openpyxl', 'calamine', 'odf', 'pyxlsb', 'xlrd'}}, default None\n"
    "    If io is not a buffer or path, this must be set to identify io.\n"
    "    Engine compatibility :\n\n"
    "    - ``openpyxl`` supports newer Excel file formats.\n"
    "    - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)\n"
    "      and OpenDocument (.ods) file formats.\n"
    "    - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).\n"
    "    - ``pyxlsb`` supports Binary Excel files.\n"
    "    - ``xlrd`` supports old-style Excel files (.xls).\n\n"
    "    When ``engine=None``, the following logic will be used to determine the engine:\n\n"
    "    - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),\n"
    "      then `odf <https://pypi.org/project/odfpy/>`_ will be used.\n"
    "    - Otherwise if ``path_or_buffer`` is an xls format, ``xlrd`` will be used.\n"
    "    - Otherwise if ``path_or_buffer`` is in xlsb format, ``pyxlsb`` will be used.\n"
    "    - Otherwise ``openpyxl`` will be used.\n"
    "converters : dict, default None\n"
    "    Dict of functions for converting values in certain columns. Keys can\n"
    "    either be integers or column labels, values are functions that take one\n"
    "    input argument, the Excel cell content, and return the transformed\n"
    "    content.\n"
    "true_values : list, default None\n"
    "    Values to consider as True.\n"
    "false_values : list, default None\n"
    "    Values to consider as False.\n"
    "skiprows : list-like, int, or callable, optional\n"
    "    Line numbers to skip (0-indexed) or number of lines to skip (int) at the\n"
    "    start of the file. If callable, the callable function will be evaluated\n"
    "    against the row indices, returning True if the row should be skipped and\n"
    "    False otherwise. An example of a valid callable argument would be ``lambda\n"
    "    x: x in [0, 2]``.\n"
    "nrows : int, default None\n"
    "    Number of rows to parse.\n"
    "na_values : scalar, str, list-like, or dict, default None\n"
    "    Additional strings to recognize as NA/NaN. If dict passed, specific\n"
    "    per-column NA values. By default the following values are interpreted\n"
    "    as NaN: '" + fill("', '".join(sorted(STR_NA_VALUES)), 70, subsequent_indent='    ') + "'.\n"
    "keep_default_na : bool, default True\n"
    "    Whether or not to include the default NaN values when parsing the data.\n"
    "    Depending on whether ``na_values`` is passed in, the behavior is as follows:\n\n"
    "    * If ``keep_default_na`` is True, and ``na_values`` are specified,\n"
    "      ``na_values`` is appended to the default NaN values used for parsing.\n"
    "    * If ``keep_default_na`` is True, and ``na_values`` are not specified, only\n"
    "      the default NaN values are used for parsing.\n"
    "    * If ``keep_default_na`` is False, and ``na_values`` are specified, only\n"
    "      the NaN values specified ``na_values`` are used for parsing.\n"
    "    * If ``keep_default_na`` is False, and ``na_values`` are not specified, no\n"
    "      strings will be parsed as NaN.\n\n"
    "    Note that if `na_filter` is passed in as False, the ``keep_default_na`` and\n"
    "    ``na_values`` parameters will be ignored.\n"
    "na_filter : bool, default True\n"
    "    Detect missing value markers (empty strings and the value of na_values). In\n"
    "    data without any NAs, passing ``na_filter=False`` can improve the\n"
    "    performance of reading a large file.\n"
    "verbose : bool, default False\n"
    "    Indicate number of NA values placed in non-numeric columns.\n"
    "parse_dates : bool, list-like, or dict, default False\n"
    "    The behavior is as follows:\n\n"
    "    * ``bool``. If True -> try parsing the index.\n"
    "    * ``list`` of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3\n"
    "      each as a separate date column.\n"
    "    * ``list`` of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as\n"
    "      a single date column.\n"
    "    * ``dict``, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call\n"
    "      result 'foo'\n\n"
    "    If a column or index contains an unparsable date, the entire column or\n"
    "    index will be returned unaltered as an object data type. If you don`t want to\n"
    "    parse some cells as date just change their type in Excel to \"Text\".\n"
    "    For non-standard datetime parsing, use ``pd.to_datetime`` after ``pd.read_excel``.\n\n"
    "    Note: A fast-path exists for iso8601-formatted dates.\n"
    "date_format : str or dict of column -> format, default ``None``\n"
    "   If used in conjunction with ``parse_dates``, will parse dates according to this\n"
    "   format. For anything more complex,\n"
    "   please read in as ``object`` and then apply :func:`to_datetime` as-needed.\n\n"
    "   .. versionadded:: 2.0.0\n"
    "thousands : str, default None\n"
    "    Thousands separator for parsing string columns to numeric.  Note that\n"
    "    this parameter is only necessary for columns stored as TEXT in Excel,\n"
    "    any numeric columns will automatically be parsed, regardless of display\n"
    "    format.\n"
    "decimal : str, default '.'\n"
    "    Character to recognize as decimal point for parsing string columns to numeric.\n"
    "    Note that this parameter is only necessary for columns stored as TEXT in Excel,\n"
    "    any numeric columns will automatically be parsed, regardless of display\n"
    "    format.(e.g. use ',' for European data).\n\n"
    "    .. versionadded:: 1.4.0\n\n"
    "comment : str, default None\n"
    "    Comments out remainder of line. Pass a character or characters to this\n"
    "    argument to indicate comments in the input file. Any data between the\n"
    "    comment string and the end of the current line is ignored.\n"
    "skipfooter : int, default 0\n"
    "    Rows at the end to skip (0-indexed).\n"
    "{storage_options}\n\n"
    "dtype_backend : {{'numpy_nullable', 'pyarrow'}}\n"
    "    Back-end data type applied to the resultant :class:`DataFrame`\n"
    "    (still experimental). If not specified, the default behavior\n"
    "    is to not use nullable data types. If specified, the behavior\n"
    "    is as follows:\n\n"
    "    * ``\"numpy_nullable\"``: returns nullable-dtype-backed :class:`DataFrame`\n"
    "    * ``\"pyarrow\"``: returns pyarrow-backed nullable\n"
    "      :class:`ArrowDtype` :class:`DataFrame`\n\n"
    "    .. versionadded:: 2.0\n\n"
    "engine_kwargs : dict, optional\n"
    "    Arbitrary keyword arguments passed to excel engine.\n\n"
    "Returns\n"
    "-------\n"
    "DataFrame or dict of DataFrames\n"
    "    DataFrame from the passed in Excel file. See notes in sheet_name\n"
    "    argument for more information on when a dict of DataFrames is returned.\n\n"
    "See Also\n"
    "--------\n"
    "DataFrame.to_excel : Write DataFrame to an Excel file.\n"
    "DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.\n"
    "read_csv : Read a comma-separated values (csv) file into DataFrame.\n"
    "read_fwf : Read a table of fixed-width formatted lines into DataFrame.\n\n"
    "Notes\n"
    "-----\n"
    "For specific information on the methods used for each Excel engine, refer to the pandas\n"
    ":ref:`user guide <io.excel_reader>`\n\n"
    "Examples\n"
    "--------\n"
    "The file can be read using the file name as string or an open file object:\n\n"
    ">>> pd.read_excel('tmp.xlsx', index_col=0)  # doctest: +SKIP\n"
    "       Name  Value\n"
    "0   string1      1\n"
    "1   string2      2\n"
    "2  #Comment      3\n\n"
    ">>> pd.read_excel(open('tmp.xlsx', 'rb'),\n"
    "...               sheet_name='Sheet3')  # doctest: +SKIP\n"
    "   Unnamed: 0      Name  Value\n"
    "0           0   string1      1\n"
    "1           1   string2      2\n"
    "2           2  #Comment      3\n\n"
    "Index and header can be specified via the `index_col` and `header` arguments\n\n"
    ">>> pd.read_excel('tmp.xlsx', index_col=None, header=None)  # doctest: +SKIP\n"
    "     0         1      2\n"
    "0  NaN      Name  Value\n"
    "1  0.0   string1      1\n"
    "2  1.0   string2      2\n"
    "3  2.0  #Comment      3\n\n"
    "Column types are inferred but can be explicitly specified\n\n"
    ">>> pd.read_excel('tmp.xlsx', index_col=0,\n"
    "...               dtype={'Name': str, 'Value': float})  # doctest: +SKIP\n"
    "       Name  Value\n"
    "0   string1    1.0\n"
    "1   string2    2.0\n"
    "2  #Comment    3.0\n\n"
    "True, False, and NA values, and thousands separators have defaults,\n"
    "but can be explicitly specified, too. Supply the values you would like\n"
    "as strings or lists of strings!\n\n"
    ">>> pd.read_excel('tmp.xlsx', index_col=0,\n"
    "...               na_values=['string1', 'string2'])  # doctest: +SKIP\n"
    "       Name  Value\n"
    "0       NaN      1\n"
    "1       NaN      2\n"
    "2  #Comment      3\n\n"
    "Comment lines in the excel input file can be skipped using the\n"
    "``comment`` kwarg.\n\n"
    ">>> pd.read_excel('tmp.xlsx', index_col=0, comment='#')  # doctest: +SKIP\n"
    "      Name  Value\n"
    "0  string1    1.0\n"
    "1  string2    2.0\n"
    "2     None    NaN\n"
)

@overload
def read_excel(io: Union[str, os.PathLike[Any], ExcelFile, IO[Any]],
               sheet_name: Optional[Union[SheetName, SheetNameList]] = ...,
               *,
               header: Optional[Union[int, List[int]]] = ...,
               names: Optional[Iterable[Any]] = ...,
               index_col: Optional[Union[int, str, List[Union[int, str]]]] = ...,
               usecols: Optional[Union[str, List[Any], Callable[[Any], bool]]] = ...,
               dtype: Optional[DtypeArg] = ...,
               engine: Optional[str] = ...,
               converters: Optional[Dict[Any, Callable[[Any], Any]]] = ...,
               true_values: Optional[List[Any]] = ...,
               false_values: Optional[List[Any]] = ...,
               skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = ...,
               nrows: Optional[int] = ...,
               na_values: Optional[Union[str, List[str], Dict[Any, Any]]] = ...,
               keep_default_na: bool = ...,
               na_filter: bool = ...,
               verbose: bool = ...,
               parse_dates: Union[bool, List[Any], Dict[Any, Any]] = ...,
               date_format: Optional[Union[str, Dict[Any, str]]] = ...,
               thousands: Optional[str] = ...,
               decimal: str = ...,
               comment: Optional[str] = ...,
               skipfooter: int = ...,
               storage_options: Optional[Any] = ...,
               dtype_backend: Union[str, lib.no_default] = ...) -> Union[DataFrame, Dict[Union[int, str], DataFrame]]:
    ...

@overload
def read_excel(io: Union[str, os.PathLike[Any], ExcelFile, IO[Any]],
               sheet_name: Union[SheetName, SheetNameList],
               *,
               header: Optional[Union[int, List[int]]] = ...,
               names: Optional[Iterable[Any]] = ...,
               index_col: Optional[Union[int, str, List[Union[int, str]]]] = ...,
               usecols: Optional[Union[str, List[Any], Callable[[Any], bool]]] = ...,
               dtype: Optional[DtypeArg] = ...,
               engine: Optional[str] = ...,
               converters: Optional[Dict[Any, Callable[[Any], Any]]] = ...,
               true_values: Optional[List[Any]] = ...,
               false_values: Optional[List[Any]] = ...,
               skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = ...,
               nrows: Optional[int] = ...,
               na_values: Optional[Union[str, List[str], Dict[Any, Any]]] = ...,
               keep_default_na: bool = ...,
               na_filter: bool = ...,
               verbose: bool = ...,
               parse_dates: Union[bool, List[Any], Dict[Any, Any]] = ...,
               date_format: Optional[Union[str, Dict[Any, str]]] = ...,
               thousands: Optional[str] = ...,
               decimal: str = ...,
               comment: Optional[str] = ...,
               skipfooter: int = ...,
               storage_options: Optional[Any] = ...,
               dtype_backend: Union[str, lib.no_default] = ...) -> Union[DataFrame, Dict[Union[int, str], DataFrame]]:
    ...

@doc(storage_options=_shared_docs['storage_options'])
@Appender(_read_excel_doc)
def read_excel(io: Union[str, os.PathLike[Any], ExcelFile, IO[Any]],
               sheet_name: Optional[Union[SheetName, SheetNameList]] = 0,
               *,
               header: Optional[Union[int, List[int]]] = 0,
               names: Optional[Iterable[Any]] = None,
               index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
               usecols: Optional[Union[str, List[Any], Callable[[Any], bool]]] = None,
               dtype: Optional[DtypeArg] = None,
               engine: Optional[str] = None,
               converters: Optional[Dict[Any, Callable[[Any], Any]]] = None,
               true_values: Optional[List[Any]] = None,
               false_values: Optional[List[Any]] = None,
               skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
               nrows: Optional[int] = None,
               na_values: Optional[Union[str, List[str], Dict[Any, Any]]] = None,
               keep_default_na: bool = True,
               na_filter: bool = True,
               verbose: bool = False,
               parse_dates: Union[bool, List[Any], Dict[Any, Any]] = False,
               date_format: Optional[Union[str, Dict[Any, str]]] = None,
               thousands: Optional[str] = None,
               decimal: str = '.',
               comment: Optional[str] = None,
               skipfooter: int = 0,
               storage_options: Optional[Any] = None,
               dtype_backend: Union[str, lib.no_default] = lib.no_default,
               engine_kwargs: Optional[Dict[str, Any]] = None) -> Union[DataFrame, Dict[Union[int, str], DataFrame]]:
    check_dtype_backend(dtype_backend)
    should_close: bool = False
    if engine_kwargs is None:
        engine_kwargs = {}
    from pandas.io.excel import ExcelFile  # Ensure using the ExcelFile defined below
    if not isinstance(io, ExcelFile):
        should_close = True
        io = ExcelFile(io, storage_options=storage_options, engine=engine, engine_kwargs=engine_kwargs)
    elif engine and engine != io.engine:
        raise ValueError('Engine should not be specified when passing an ExcelFile - ExcelFile already has the engine set')
    try:
        data: Union[DataFrame, Dict[Union[int, str], DataFrame]] = io.parse(sheet_name=sheet_name, header=header, names=names, index_col=index_col,
                                                                                 usecols=usecols, dtype=dtype, converters=converters,
                                                                                 true_values=true_values, false_values=false_values,
                                                                                 skiprows=skiprows, nrows=nrows, na_values=na_values,
                                                                                 keep_default_na=keep_default_na, na_filter=na_filter,
                                                                                 verbose=verbose, parse_dates=parse_dates, date_format=date_format,
                                                                                 thousands=thousands, decimal=decimal, comment=comment,
                                                                                 skipfooter=skipfooter, dtype_backend=dtype_backend)
    finally:
        if should_close:
            io.close()
    return data

_WorkbookT = TypeVar("_WorkbookT")

class BaseExcelReader(Generic[_WorkbookT]):
    def __init__(self, filepath_or_buffer: Any, storage_options: Optional[Any] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        self.handles: IOHandles = IOHandles(handle=filepath_or_buffer, compression={'method': None})
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            self.handles = get_handle(filepath_or_buffer, 'rb', storage_options=storage_options, is_text=False)
        if isinstance(self.handles.handle, self._workbook_class):
            self.book: _WorkbookT = self.handles.handle
        elif hasattr(self.handles.handle, 'read'):
            self.handles.handle.seek(0)
            try:
                self.book = self.load_workbook(self.handles.handle, engine_kwargs)
            except Exception:
                self.close()
                raise
        else:
            raise ValueError('Must explicitly set engine if not passing in buffer or path for io.')

    @property
    def _workbook_class(self) -> Any:
        raise NotImplementedError

    def load_workbook(self, filepath_or_buffer: IO[Any], engine_kwargs: Dict[str, Any]) -> _WorkbookT:
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self, 'book'):
            if hasattr(self.book, 'close'):
                self.book.close()
            elif hasattr(self.book, 'release_resources'):
                self.book.release_resources()
        self.handles.close()

    @property
    def sheet_names(self) -> List[str]:
        raise NotImplementedError

    def get_sheet_by_name(self, name: str) -> Any:
        raise NotImplementedError

    def get_sheet_by_index(self, index: int) -> Any:
        raise NotImplementedError

    def get_sheet_data(self, sheet: Any, rows: Optional[int] = None) -> Any:
        raise NotImplementedError

    def raise_if_bad_sheet_by_index(self, index: int) -> None:
        n_sheets: int = len(self.sheet_names)
        if index >= n_sheets:
            raise ValueError(f'Worksheet index {index} is invalid, {n_sheets} worksheets found')

    def raise_if_bad_sheet_by_name(self, name: str) -> None:
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    def _check_skiprows_func(self, skiprows: Callable[[int], bool], rows_to_use: int) -> int:
        i: int = 0
        rows_used_so_far: int = 0
        while rows_used_so_far < rows_to_use:
            if not skiprows(i):
                rows_used_so_far += 1
            i += 1
        return i

    def _calc_rows(self, header: Optional[Union[int, List[int]]], index_col: Optional[Union[int, str, List[Union[int, str]]]], skiprows: Optional[Union[int, List[int], Callable[[int], bool]]], nrows: Optional[int]) -> Optional[int]:
        if nrows is None:
            return None
        if header is None:
            header_rows: int = 1
        elif is_integer(header):
            header_rows = 1 + cast(int, header)
        else:
            header = cast(TypingSequence[int], header)
            header_rows = 1 + header[-1]
        if is_list_like(header) and index_col is not None:
            header = cast(TypingSequence[int], header)
            if len(header) > 1:
                header_rows += 1
        if skiprows is None:
            return header_rows + nrows
        if is_integer(skiprows):
            skiprows = cast(int, skiprows)
            return header_rows + nrows + skiprows
        if is_list_like(skiprows):
            def f(skiprows_list: List[int], x: int) -> bool:
                return x in skiprows_list
            skiprows_list = cast(List[int], skiprows)
            return self._check_skiprows_func(partial(f, skiprows_list), header_rows + nrows)
        if callable(skiprows):
            return self._check_skiprows_func(skiprows, header_rows + nrows)
        return None

    def parse(self,
              sheet_name: Optional[Union[SheetName, SheetNameList]] = 0,
              header: Optional[Union[int, List[int]]] = 0,
              names: Optional[Iterable[Any]] = None,
              index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
              usecols: Optional[Union[str, List[Any], Callable[[Any], bool]]] = None,
              dtype: Optional[DtypeArg] = None,
              true_values: Optional[List[Any]] = None,
              false_values: Optional[List[Any]] = None,
              skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
              nrows: Optional[int] = None,
              na_values: Optional[Union[str, List[str], Dict[Any, Any]]] = None,
              verbose: bool = False,
              parse_dates: Union[bool, List[Any], Dict[Any, Any]] = False,
              date_format: Optional[Union[str, Dict[Any, str]]] = None,
              thousands: Optional[str] = None,
              decimal: str = '.',
              comment: Optional[str] = None,
              skipfooter: int = 0,
              dtype_backend: Union[str, lib.no_default] = lib.no_default,
              **kwds: Any) -> Union[DataFrame, Dict[Union[int, str], DataFrame]]:
        validate_header_arg(header)
        validate_integer("nrows", nrows)
        ret_dict: bool = False
        if isinstance(sheet_name, list):
            sheets: SheetNameList = sheet_name
            ret_dict = True
        elif sheet_name is None:
            sheets = self.sheet_names  # type: ignore
            ret_dict = True
        elif isinstance(sheet_name, str):
            sheets = [sheet_name]
        else:
            sheets = [sheet_name]
        sheets = cast(List[Union[int, str]], list(dict.fromkeys(sheets).keys()))
        output: Dict[Union[int, str], DataFrame] = {}
        last_sheetname: Optional[Union[int, str]] = None
        for asheetname in sheets:
            last_sheetname = asheetname
            if verbose:
                print(f"Reading sheet {asheetname}")
            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:
                sheet = self.get_sheet_by_index(asheetname)
            file_rows_needed: Optional[int] = self._calc_rows(header, index_col, skiprows, nrows)
            data = self.get_sheet_data(sheet, file_rows_needed)
            if hasattr(sheet, "close"):
                sheet.close()
            usecols = maybe_convert_usecols(usecols)
            if not data:
                output[asheetname] = DataFrame()
                continue
            output = self._parse_sheet(data=data, output=output, asheetname=asheetname, header=header, names=names,
                                        index_col=index_col, usecols=usecols, dtype=dtype, skiprows=skiprows, nrows=nrows,
                                        true_values=true_values, false_values=false_values, na_values=na_values,
                                        parse_dates=parse_dates, date_format=date_format, thousands=thousands,
                                        decimal=decimal, comment=comment, skipfooter=skipfooter, dtype_backend=dtype_backend,
                                        **kwds)
        if last_sheetname is None:
            raise ValueError("Sheet name is an empty list")
        if ret_dict:
            return output
        else:
            return output[last_sheetname]

    def _parse_sheet(self,
                     data: Any,
                     output: Dict[Union[int, str], DataFrame],
                     asheetname: Optional[Union[int, str]] = None,
                     header: Optional[Union[int, List[int]]] = 0,
                     names: Optional[Iterable[Any]] = None,
                     index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
                     usecols: Optional[Union[str, List[Any], Callable[[Any], bool]]] = None,
                     dtype: Optional[DtypeArg] = None,
                     skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
                     nrows: Optional[int] = None,
                     true_values: Optional[List[Any]] = None,
                     false_values: Optional[List[Any]] = None,
                     na_values: Optional[Union[str, List[str], Dict[Any, Any]]] = None,
                     parse_dates: Union[bool, List[Any], Dict[Any, Any]] = False,
                     date_format: Optional[Union[str, Dict[Any, str]]] = None,
                     thousands: Optional[str] = None,
                     decimal: str = ".",
                     comment: Optional[str] = None,
                     skipfooter: int = 0,
                     dtype_backend: Union[str, lib.no_default] = lib.no_default,
                     **kwds: Any) -> Dict[Union[int, str], DataFrame]:
        is_list_header: bool = False
        is_len_one_list_header: bool = False
        if is_list_like(header):
            assert isinstance(header, list)
            is_list_header = True
            if len(header) == 1:
                is_len_one_list_header = True
        if is_len_one_list_header:
            header = cast(List[int], header)[0]
        header_names: Optional[List[Any]] = None
        if header is not None and is_list_like(header):
            assert isinstance(header, list)
            header_names = []
            control_row: List[bool] = [True] * len(data[0])
            for row in cast(List[int], header):
                if is_integer(skiprows):
                    assert isinstance(skiprows, int)
                    row += skiprows
                if row > len(data) - 1:
                    raise ValueError(f"header index {row} exceeds maximum index {len(data) - 1} of data.")
                data[row], control_row = fill_mi_header(data[row], control_row)
                if index_col is not None:
                    header_name, _ = pop_header_name(data[row], index_col)
                    header_names.append(header_name)
        has_index_names: bool = False
        if is_list_header and (not is_len_one_list_header) and (index_col is not None):
            if isinstance(index_col, int):
                index_col_set = {index_col}
            else:
                assert isinstance(index_col, list)
                index_col_set = set(index_col)
            assert isinstance(header, list)
            if len(header) < len(data):
                potential_index_names = data[len(header)]
                has_index_names = all((x == "" or x is None for i, x in enumerate(potential_index_names) if not control_row[i] and i not in index_col_set))
        if is_list_like(index_col):
            if header is None:
                offset = 0
            elif isinstance(header, int):
                offset = 1 + header
            else:
                offset = 1 + max(header)
            if has_index_names:
                offset += 1
            if offset < len(data):
                assert isinstance(index_col, list)
                for col in index_col:
                    last = data[offset][col]
                    for row in range(offset + 1, len(data)):
                        if data[row][col] == "" or data[row][col] is None:
                            data[row][col] = last
                        else:
                            last = data[row][col]
        try:
            parser = TextParser(data, names=names, header=header, index_col=index_col, has_index_names=has_index_names,
                                dtype=dtype, true_values=true_values, false_values=false_values, skiprows=skiprows,
                                nrows=nrows, na_values=na_values, skip_blank_lines=False, parse_dates=parse_dates,
                                date_format=date_format, thousands=thousands, decimal=decimal, comment=comment,
                                skipfooter=skipfooter, usecols=usecols, dtype_backend=dtype_backend, **kwds)
            output[asheetname] = parser.read(nrows=nrows)
            if header_names:
                output[asheetname].columns = output[asheetname].columns.set_names(header_names)
        except EmptyDataError:
            output[asheetname] = DataFrame()
        except Exception as err:
            err.args = (f"{err.args[0]} (sheet: {asheetname})", *err.args[1:])
            raise err
        return output

@doc(storage_options=_shared_docs['storage_options'])
class ExcelWriter(Generic[_WorkbookT]):
    def __new__(cls, path: Union[str, IO[bytes]], engine: Optional[str] = None,
                date_format: Optional[str] = None, datetime_format: Optional[str] = None,
                mode: Literal['w', 'a'] = 'w', storage_options: Optional[Any] = None,
                if_sheet_exists: Optional[Literal['error', 'new', 'replace', 'overlay']] = None,
                engine_kwargs: Optional[Dict[str, Any]] = None) -> ExcelWriter:
        if cls is ExcelWriter:
            if engine is None or (isinstance(engine, str) and engine == "auto"):
                if isinstance(path, str):
                    ext = os.path.splitext(path)[-1][1:]
                else:
                    ext = "xlsx"
                try:
                    engine = config.get_option(f"io.excel.{ext}.writer")
                    if engine == "auto":
                        engine = get_default_engine(ext, mode="writer")
                except KeyError as err:
                    raise ValueError(f"No engine for filetype: '{ext}'") from err
            assert engine is not None
            cls = get_writer(engine)
        return object.__new__(cls)

    _path: Optional[Any] = None

    @property
    def supported_extensions(self) -> List[str]:
        return self._supported_extensions

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def sheets(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @property
    def book(self) -> Any:
        raise NotImplementedError

    def _write_cells(self, cells: Any, sheet_name: Optional[str] = None, startrow: int = 0,
                     startcol: int = 0, freeze_panes: Optional[TypingSequence[int]] = None) -> None:
        raise NotImplementedError

    def _save(self) -> None:
        raise NotImplementedError

    def __init__(self, path: Union[str, IO[bytes]], engine: Optional[str] = None,
                 date_format: Optional[str] = None, datetime_format: Optional[str] = None,
                 mode: Literal['w', 'a'] = 'w', storage_options: Optional[Any] = None,
                 if_sheet_exists: Optional[Literal['error', 'new', 'replace', 'overlay']] = None,
                 engine_kwargs: Optional[Dict[str, Any]] = None) -> None:
        if isinstance(path, str):
            ext = os.path.splitext(path)[-1]
            self.check_extension(ext)
        if "b" not in mode:
            mode += "b"
        mode = mode.replace("a", "r+")
        if if_sheet_exists not in (None, "error", "new", "replace", "overlay"):
            raise ValueError(f"'{if_sheet_exists}' is not valid for if_sheet_exists. Valid options are 'error', 'new', 'replace' and 'overlay'.")
        if if_sheet_exists and "r+" not in mode:
            raise ValueError("if_sheet_exists is only valid in append mode (mode='a')")
        if if_sheet_exists is None:
            if_sheet_exists = "error"
        self._if_sheet_exists: Literal["error", "new", "replace", "overlay"] = if_sheet_exists
        self._handles = IOHandles(cast(IO[bytes], path), compression={'compression': None})
        if not isinstance(path, ExcelWriter):
            self._handles = get_handle(path, mode, storage_options=storage_options, is_text=False)
        self._cur_sheet: Optional[str] = None
        if date_format is None:
            self._date_format: str = "YYYY-MM-DD"
        else:
            self._date_format = date_format
        if datetime_format is None:
            self._datetime_format: str = "YYYY-MM-DD HH:MM:SS"
        else:
            self._datetime_format = datetime_format
        self._mode: str = mode

    @property
    def date_format(self) -> str:
        return self._date_format

    @property
    def datetime_format(self) -> str:
        return self._datetime_format

    @property
    def if_sheet_exists(self) -> Literal["error", "new", "replace", "overlay"]:
        return self._if_sheet_exists

    def __fspath__(self) -> str:
        return getattr(self._handles.handle, "name", "")

    def _get_sheet_name(self, sheet_name: Optional[str]) -> str:
        if sheet_name is None:
            sheet_name = self._cur_sheet
        if sheet_name is None:
            raise ValueError("Must pass explicit sheet_name or set _cur_sheet property")
        return sheet_name

    def _value_with_fmt(self, val: Any) -> tuple[Any, Optional[str]]:
        fmt: Optional[str] = None
        if is_integer(val):
            val = int(val)
        elif is_float(val):
            val = float(val)
        elif is_bool(val):
            val = bool(val)
        elif is_decimal(val):
            val = Decimal(val)
        elif isinstance(val, datetime.datetime):
            fmt = self._datetime_format
        elif isinstance(val, datetime.date):
            fmt = self._date_format
        elif isinstance(val, datetime.timedelta):
            val = val.total_seconds() / 86400
            fmt = "0"
        else:
            val = str(val)
            if len(val) > 32767:
                warnings.warn(f"Cell contents too long ({len(val)}), truncated to 32767 characters", UserWarning, stacklevel=find_stack_level())
        return (val, fmt)

    @classmethod
    def check_extension(cls, ext: str) -> bool:
        if ext.startswith("."):
            ext = ext[1:]
        if not any((ext in extension for extension in cls._supported_extensions)):
            raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")
        return True

    def __enter__(self) -> ExcelWriter:
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()

    def close(self) -> None:
        self._save()
        self._handles.close()

XLS_SIGNATURES: tuple[bytes, ...] = (b"\t\x00\x04\x00\x07\x00\x10\x00",
                                      b"\t\x02\x06\x00\x00\x00\x10\x00",
                                      b"\t\x04\x06\x00\x00\x00\x10\x00",
                                      b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
ZIP_SIGNATURE: bytes = b"PK\x03\x04"
PEEK_SIZE: int = max(map(len, XLS_SIGNATURES + (ZIP_SIGNATURE,)))

@doc(storage_options=_shared_docs['storage_options'])
def inspect_excel_format(content_or_path: Union[str, IO[Any]], storage_options: Optional[Any] = None) -> Optional[str]:
    with get_handle(content_or_path, "rb", storage_options=storage_options, is_text=False) as handle:
        stream: IO[Any] = handle.handle
        stream.seek(0)
        buf: bytes = stream.read(PEEK_SIZE)
        if buf is None:
            raise ValueError("stream is empty")
        assert isinstance(buf, bytes)
        peek: bytes = buf
        stream.seek(0)
        if any((peek.startswith(sig) for sig in XLS_SIGNATURES)):
            return "xls"
        elif not peek.startswith(ZIP_SIGNATURE):
            return None
        with zipfile.ZipFile(stream) as zf:
            component_names: set[str] = {name.replace("\\", "/").lower() for name in zf.namelist()}
        if "xl/workbook.xml" in component_names:
            return "xlsx"
        if "xl/workbook.bin" in component_names:
            return "xlsb"
        if "content.xml" in component_names:
            return "ods"
        return "zip"

@doc(storage_options=_shared_docs['storage_options'])
class ExcelFile:
    from pandas.io.excel._calamine import CalamineReader
    from pandas.io.excel._odfreader import ODFReader
    from pandas.io.excel._openpyxl import OpenpyxlReader
    from pandas.io.excel._pyxlsb import PyxlsbReader
    from pandas.io.excel._xlrd import XlrdReader
    _engines: Dict[str, Any] = {'xlrd': XlrdReader,
                                'openpyxl': OpenpyxlReader,
                                'odf': ODFReader,
                                'pyxlsb': PyxlsbReader,
                                'calamine': CalamineReader}

    def __init__(self, path_or_buffer: Union[str, bytes, os.PathLike[Any], IO[Any]], engine: Optional[str] = None,
                 storage_options: Optional[Any] = None, engine_kwargs: Optional[Dict[str, Any]] = None) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        if engine is not None and engine not in self._engines:
            raise ValueError(f"Unknown engine: {engine}")
        self._io: str = stringify_path(path_or_buffer)
        if engine is None:
            ext: Optional[str] = None
            if not isinstance(path_or_buffer, (str, os.PathLike, ExcelFile)) and (not is_file_like(path_or_buffer)):
                if import_optional_dependency("xlrd", errors="ignore") is None:
                    xlrd_version: Optional[str] = None
                else:
                    import xlrd  # type: ignore
                    xlrd_version = Version(get_version(xlrd))
                if xlrd_version is not None and isinstance(path_or_buffer, xlrd.Book):
                    ext = "xls"
            if ext is None:
                ext = inspect_excel_format(content_or_path=path_or_buffer, storage_options=storage_options)
                if ext is None:
                    raise ValueError("Excel file format cannot be determined, you must specify an engine manually.")
            engine = config.get_option(f"io.excel.{ext}.reader")
            if engine == "auto":
                engine = get_default_engine(ext, mode="reader")
        assert engine is not None
        self.engine: str = engine
        self.storage_options: Optional[Any] = storage_options
        self._reader: Any = self._engines[engine](self._io, storage_options=storage_options, engine_kwargs=engine_kwargs)

    def __fspath__(self) -> str:
        return self._io

    def parse(self,
              sheet_name: Optional[Union[SheetName, SheetNameList]] = 0,
              header: Optional[Union[int, List[int]]] = 0,
              names: Optional[Iterable[Any]] = None,
              index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
              converters: Optional[Dict[Any, Callable[[Any], Any]]] = None,
              true_values: Optional[List[Any]] = None,
              false_values: Optional[List[Any]] = None,
              skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
              nrows: Optional[int] = None,
              na_values: Optional[Union[str, List[str], Dict[Any, Any]]] = None,
              parse_dates: Union[bool, List[Any], Dict[Any, Any]] = False,
              date_format: Optional[Union[str, Dict[Any, str]]] = None,
              thousands: Optional[str] = None,
              comment: Optional[str] = None,
              skipfooter: int = 0,
              dtype_backend: Union[str, lib.no_default] = lib.no_default,
              **kwds: Any) -> Union[DataFrame, Dict[Union[int, str], DataFrame]]:
        return self._reader.parse(sheet_name=sheet_name, header=header, names=names, index_col=index_col,
                                  usecols=None, converters=converters, true_values=true_values,
                                  false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values,
                                  parse_dates=parse_dates, date_format=date_format, thousands=thousands,
                                  comment=comment, skipfooter=skipfooter, dtype_backend=dtype_backend, **kwds)

    @property
    def book(self) -> Any:
        return self._reader.book

    @property
    def sheet_names(self) -> List[str]:
        return self._reader.sheet_names

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> ExcelFile:
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()