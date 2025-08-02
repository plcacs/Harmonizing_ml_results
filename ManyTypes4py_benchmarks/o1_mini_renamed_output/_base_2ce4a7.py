from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import datetime
from decimal import Decimal
from functools import partial
import os
from textwrap import fill
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
    Optional,
    List,
    Dict,
    Tuple,
)
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
from pandas.core.dtypes.common import (
    is_bool,
    is_decimal,
    is_file_like,
    is_float,
    is_integer,
    is_list_like,
)
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util.version import Version
from pandas.io.common import IOHandles, get_handle, stringify_path, validate_header_arg
from pandas.io.excel._util import (
    fill_mi_header,
    get_default_engine,
    get_writer,
    maybe_convert_usecols,
    pop_header_name,
)
from pandas.io.parsers import TextParser
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from types import TracebackType
    from pandas._typing import (
        DtypeArg,
        DtypeBackend,
        ExcelWriterIfSheetExists,
        FilePath,
        HashableT,
        IntStrT,
        ReadBuffer,
        Self,
        SequenceNotStr,
        StorageOptions,
        WriteExcelBuffer,
    )

_read_excel_doc = """
Read an Excel file into a ``pandas`` ``DataFrame``.

Supports `xls`, `xlsx`, `xlsm`, `xlsb`, `odf`, `ods` and `odt` file extensions
read from a local filesystem or URL. Supports an option to read
a single sheet or a list of sheets.

Parameters
----------
io : str, ExcelFile, xlrd.Book, path object, or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, and file. For file URLs, a host is
    expected. A local file could be: ``file://localhost/path/to/table.xlsx``.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method,
    such as a file handle (e.g. via builtin ``open`` function)
    or ``StringIO``.

    .. deprecated:: 2.1.0
        Passing byte strings is deprecated. To read from a
        byte string, wrap it in a ``BytesIO`` object.
sheet_name : str, int, list, or None, default 0
    Strings are used for sheet names. Integers are used in zero-indexed
    sheet positions (chart sheets do not count as a sheet position).
    Lists of strings/integers are used to request multiple sheets.
    When ``None``, will return a dictionary containing DataFrames for each sheet.

    Available cases:

    * Defaults to ``0``: 1st sheet as a `DataFrame`
    * ``1``: 2nd sheet as a `DataFrame`
    * ``"Sheet1"``: Load sheet with name "Sheet1"
    * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5"
      as a dict of `DataFrame`
    * ``None``: Returns a dictionary containing DataFrames for each sheet..

header : int, list of int, default 0
    Row (0-indexed) to use for the column labels of the parsed
    DataFrame. If a list of integers is passed those row positions will
    be combined into a ``MultiIndex``. Use None if there is no header.
names : array-like, default None
    List of column names to use. If file contains no header row,
    then you should explicitly pass header=None.
index_col : int, str, list of int, default None
    Column (0-indexed) to use as the row labels of the DataFrame.
    Pass None if there is no such column.  If a list is passed,
    those columns will be combined into a ``MultiIndex``.  If a
    subset of data is selected with ``usecols``, index_col
    is based on the subset.

    Missing values will be forward filled to allow roundtripping with
    ``to_excel`` for ``merged_cells=True``. To avoid forward filling the
    missing values use ``set_index`` after reading the data instead of
    ``index_col``.
usecols : str, list-like, or callable, default None
    * If None, then parse all columns.
    * If str, then indicates comma separated list of Excel column letters
      and column ranges (e.g. "A:E" or "A,C,E:F"). Ranges are inclusive of
      both sides.
    * If list of int, then indicates list of column numbers to be parsed
      (0-indexed).
    * If list of string, then indicates list of column names to be parsed.
    * If callable, then evaluate each column name against it and parse the
      column if the callable returns ``True``.

    Returns a subset of the columns according to behavior above.
dtype : Type name or dict of column -> type, default None
    Data type for data or columns. E.g. {{'a': np.float64, 'b': np.int32}}
    Use ``object`` to preserve data as stored in Excel and not interpret dtype,
    which will necessarily result in ``object`` dtype.
    If converters are specified, they will be applied INSTEAD
    of dtype conversion.
    If you use ``None``, it will infer the dtype of each column based on the data.
engine : {{'openpyxl', 'calamine', 'odf', 'pyxlsb', 'xlrd'}}, default None
    If io is not a buffer or path, this must be set to identify io.
    Engine compatibility :

    - ``openpyxl`` supports newer Excel file formats.
    - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)
      and OpenDocument (.ods) file formats.
    - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
    - ``pyxlsb`` supports Binary Excel files.
    - ``xlrd`` supports old-style Excel files (.xls).

    When ``engine=None``, the following logic will be used to determine the engine:

    - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
      then `odf <https://pypi.org/project/odfpy/>`_ will be used.
    - Otherwise if ``path_or_buffer`` is an xls format, ``xlrd`` will be used.
    - Otherwise if ``path_or_buffer`` is in xlsb format, ``pyxlsb`` will be used.
    - Otherwise ``openpyxl`` will be used.
converters : dict, default None
    Dict of functions for converting values in certain columns. Keys can
    either be integers or column labels, values are functions that take one
    input argument, the Excel cell content, and return the transformed
    content.
true_values : list, default None
    Values to consider as True.
false_values : list, default None
    Values to consider as False.
skiprows : list-like, int, or callable, optional
    Line numbers to skip (0-indexed) or number of lines to skip (int) at the
    start of the file. If callable, the callable function will be evaluated
    against the row indices, returning True if the row should be skipped and
    False otherwise. An example of a valid callable argument would be ``lambda
    x: x in [0, 2]``.
nrows : int, default None
    Number of rows to parse.
na_values : scalar, str, list-like, or dict, default None
    Additional strings to recognize as NA/NaN. If dict passed, specific
    per-column NA values. By default the following values are interpreted
    as NaN: '""" + fill("', '".join(sorted(STR_NA_VALUES)), 70, subsequent_indent='    ') + """'.
keep_default_na : bool, default True
    Whether or not to include the default NaN values when parsing the data.
    Depending on whether ``na_values`` is passed in, the behavior is as follows:

    * If ``keep_default_na`` is True, and ``na_values`` are specified,
      ``na_values`` is appended to the default NaN values used for parsing.
    * If ``keep_default_na`` is True, and ``na_values`` are not specified, only
      the default NaN values are used for parsing.
    * If ``keep_default_na`` is False, and ``na_values`` are specified, only
      the NaN values specified ``na_values`` are used for parsing.
    * If ``keep_default_na`` is False, and ``na_values`` are not specified, no
      strings will be parsed as NaN.

    Note that if `na_filter` is passed in as False, the ``keep_default_na`` and
    ``na_values`` parameters will be ignored.
na_filter : bool, default True
    Detect missing value markers (empty strings and the value of na_values). In
    data without any NAs, passing ``na_filter=False`` can improve the
    performance of reading a large file.
verbose : bool, default False
    Indicate number of NA values placed in non-numeric columns.
parse_dates : bool, list-like, or dict, default False
    The behavior is as follows:

    * ``bool``. If True -> try parsing the index.
    * ``list`` of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
      each as a separate date column.
    * ``list`` of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
      a single date column.
    * ``dict``, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call
      result 'foo'

    If a column or index contains an unparsable date, the entire column or
    index will be returned unaltered as an object data type. If you don`t want to
    parse some cells as date just change their type in Excel to "Text".
    For non-standard datetime parsing, use ``pd.to_datetime`` after ``pd.read_excel``.

    Note: A fast-path exists for iso8601-formatted dates.
date_format : str or dict of column -> format, default ``None``
   If used in conjunction with ``parse_dates``, will parse dates according to this
   format. For anything more complex,
   please read in as ``object`` and then apply :func:`to_datetime` as-needed.

   .. versionadded:: 2.0.0
thousands : str, default None
    Thousands separator for parsing string columns to numeric.  Note that
    this parameter is only necessary for columns stored as TEXT in Excel,
    any numeric columns will automatically be parsed, regardless of display
    format.
decimal : str, default '.'
    Character to recognize as decimal point for parsing string columns to numeric.
    Note that this parameter is only necessary for columns stored as TEXT in Excel,
    any numeric columns will automatically be parsed, regardless of display
    format.(e.g. use ',' for European data).

    .. versionadded:: 1.4.0

comment : str, default None
    Comments out remainder of line. Pass a character or characters to this
    argument to indicate comments in the input file. Any data between the
    comment string and the end of the current line is ignored.
skipfooter : int, default 0
    Rows at the end to skip (0-indexed).
{storage_options}

dtype_backend : {{'numpy_nullable', 'pyarrow'}}
    Back-end data type applied to the resultant :class:`DataFrame`
    (still experimental). If not specified, the default behavior
    is to not use nullable data types. If specified, the behavior
    is as follows:

    * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
    * ``"pyarrow"``: returns pyarrow-backed nullable
      :class:`ArrowDtype` :class:`DataFrame`

    .. versionadded:: 2.0

engine_kwargs : dict, optional
    Arbitrary keyword arguments passed to excel engine.

Returns
-------
DataFrame or dict of DataFrames
    DataFrame from the passed in Excel file. See notes in sheet_name
    argument for more information on when a dict of DataFrames is returned.

See Also
--------
DataFrame.to_excel : Write DataFrame to an Excel file.
DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
read_csv : Read a comma-separated values (csv) file into DataFrame.
read_fwf : Read a table of fixed-width formatted lines into DataFrame.

Notes
-----
For specific information on the methods used for each Excel engine, refer to the pandas
:ref:`user guide <io.excel_reader>`

Examples
--------
The file can be read using the file name as string or an open file object:

>>> pd.read_excel('tmp.xlsx', index_col=0)  # doctest: +SKIP
       Name  Value
0   string1      1
1   string2      2
2  #Comment      3

>>> pd.read_excel(open('tmp.xlsx', 'rb'),
...               sheet_name='Sheet3')  # doctest: +SKIP
   Unnamed: 0      Name  Value
0           0   string1      1
1           1   string2      2
2           2  #Comment      3

Index and header can be specified via the `index_col` and `header` arguments

>>> pd.read_excel('tmp.xlsx', index_col=None, header=None)  # doctest: +SKIP
     0         1      2
0  NaN      Name  Value
1  0.0   string1      1
2  1.0   string2      2
3  2.0  #Comment      3

Column types are inferred but can be explicitly specified

>>> pd.read_excel('tmp.xlsx', index_col=0,
...               dtype={{'Name': str, 'Value': float}})  # doctest: +SKIP
       Name  Value
0   string1    1.0
1   string2    2.0
2  #Comment    3.0

True, False, and NA values, and thousands separators have defaults,
but can be explicitly specified, too. Supply the values you would like
as strings or lists of strings!

>>> pd.read_excel('tmp.xlsx', index_col=0,
...               na_values=['string1', 'string2'])  # doctest: +SKIP
       Name  Value
0       NaN      1
1       NaN      2
2  #Comment      3

Comment lines in the excel input file can be skipped using the
``comment`` kwarg.

>>> pd.read_excel('tmp.xlsx', index_col=0, comment='#')  # doctest: +SKIP
      Name  Value
0  string1    1.0
1  string2    2.0
2     None    NaN
"""


@overload
def func_i6b8zsnw(
    io: Union[str, ExcelFile, Any],
    sheet_name: Optional[Union[str, int, List[Union[str, int]]]] = 0,
    *,
    header: Optional[Union[int, List[int]]] = 0,
    names: Optional[List[Any]] = None,
    index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
    usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
    dtype: Optional[Union[DtypeArg, Dict[str, DtypeArg]]] = None,
    engine: Optional[str] = None,
    converters: Optional[Dict[Union[int, str], Callable[[Any], Any]]] = None,
    true_values: Optional[List[Any]] = None,
    false_values: Optional[List[Any]] = None,
    skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
    nrows: Optional[int] = None,
    na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: Optional[Union[bool, List[Union[int, str]], List[List[int]], Dict[str, List[int]]]] = False,
    date_format: Optional[Union[str, Dict[str, str]]] = None,
    thousands: Optional[str] = None,
    decimal: str = ".",
    comment: Optional[str] = None,
    skipfooter: int = 0,
    storage_options: Optional[StorageOptions] = None,
    dtype_backend: Union[DtypeBackend, lib.NoDefault] = lib.no_default,
) -> Union[DataFrame, Dict[str, DataFrame]]:
    ...


@overload
def func_i6b8zsnw(
    io: Union[str, ExcelFile, Any],
    sheet_name: Union[str, int, List[Union[str, int]]],
    *,
    header: Optional[Union[int, List[int]]] = 0,
    names: Optional[List[Any]] = None,
    index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
    usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
    dtype: Optional[Union[DtypeArg, Dict[str, DtypeArg]]] = None,
    engine: Optional[str] = None,
    converters: Optional[Dict[Union[int, str], Callable[[Any], Any]]] = None,
    true_values: Optional[List[Any]] = None,
    false_values: Optional[List[Any]] = None,
    skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
    nrows: Optional[int] = None,
    na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: Optional[Union[bool, List[Union[int, str]], List[List[int]], Dict[str, List[int]]]] = False,
    date_format: Optional[Union[str, Dict[str, str]]] = None,
    thousands: Optional[str] = None,
    decimal: str = ".",
    comment: Optional[str] = None,
    skipfooter: int = 0,
    storage_options: Optional[StorageOptions] = None,
    dtype_backend: Union[DtypeBackend, lib.NoDefault] = lib.no_default,
) -> Union[DataFrame, Dict[str, DataFrame]]:
    ...


@doc(storage_options=_shared_docs['storage_options'])
@Appender(_read_excel_doc)
def func_i6b8zsnw(
    io: Union[str, ExcelFile, Any],
    sheet_name: Optional[Union[str, int, List[Union[str, int]]]] = 0,
    *,
    header: Optional[Union[int, List[int]]] = 0,
    names: Optional[List[Any]] = None,
    index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
    usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
    dtype: Optional[Union[DtypeArg, Dict[str, DtypeArg]]] = None,
    engine: Optional[str] = None,
    converters: Optional[Dict[Union[int, str], Callable[[Any], Any]]] = None,
    true_values: Optional[List[Any]] = None,
    false_values: Optional[List[Any]] = None,
    skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
    nrows: Optional[int] = None,
    na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: Optional[Union[bool, List[Union[int, str]], List[List[int]], Dict[str, List[int]]]] = False,
    date_format: Optional[Union[str, Dict[str, str]]] = None,
    thousands: Optional[str] = None,
    decimal: str = ".",
    comment: Optional[str] = None,
    skipfooter: int = 0,
    storage_options: Optional[StorageOptions] = None,
    dtype_backend: Union[DtypeBackend, lib.NoDefault] = lib.no_default,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[DataFrame, Dict[str, DataFrame]]:
    check_dtype_backend(dtype_backend)
    should_close: bool = False
    if engine_kwargs is None:
        engine_kwargs = {}
    if not isinstance(io, ExcelFile):
        should_close = True
        io = ExcelFile(io, storage_options=storage_options, engine=engine, engine_kwargs=engine_kwargs)
    elif engine and engine != io.engine:
        raise ValueError(
            'Engine should not be specified when passing an ExcelFile - ExcelFile already has the engine set'
        )
    try:
        data = io.parse(
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            dtype=dtype,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            verbose=verbose,
            parse_dates=parse_dates,
            date_format=date_format,
            thousands=thousands,
            decimal=decimal,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
        )
    finally:
        if should_close:
            io.close()
    return data


_WorkbookT = TypeVar("_WorkbookT")


class BaseExcelReader(Generic[_WorkbookT]):
    def __init__(
        self,
        filepath_or_buffer: Union[str, bytes, IO[Any], ExcelFile, _WorkbookT],
        storage_options: Optional[StorageOptions] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        self.handles: IOHandles = IOHandles(handle=filepath_or_buffer, compression={"method": None})
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            self.handles = get_handle(
                filepath_or_buffer, "rb", storage_options=storage_options, is_text=False
            )
        if isinstance(self.handles.handle, self._workbook_class):
            self.book: _WorkbookT = self.handles.handle
        elif hasattr(self.handles.handle, "read"):
            self.handles.handle.seek(0)
            try:
                self.book = self.load_workbook(self.handles.handle, engine_kwargs)
            except Exception:
                self.close()
                raise
        else:
            raise ValueError(
                "Must explicitly set engine if not passing in buffer or path for io."
            )

    @property
    def func_5ofthm91(self) -> Any:
        raise NotImplementedError

    def func_8hidufb5(
        self, filepath_or_buffer: Union[str, bytes, IO[Any], ExcelFile, _WorkbookT], engine_kwargs: Dict[str, Any]
    ) -> Any:
        raise NotImplementedError

    def func_c5u0hzi0(self) -> None:
        if hasattr(self, "book"):
            if hasattr(self.book, "close"):
                self.book.close()
            elif hasattr(self.book, "release_resources"):
                self.book.release_resources()
        self.handles.close()

    @property
    def func_uffdua8n(self) -> List[str]:
        raise NotImplementedError

    def func_6m9s7wsm(self, name: str) -> None:
        raise NotImplementedError

    def func_livfsxsu(self, index: int) -> None:
        raise NotImplementedError

    def func_byv4k48w(
        self, sheet: Any, rows: Optional[int] = None
    ) -> Any:
        raise NotImplementedError

    def func_m1w8lrf9(self, index: int) -> None:
        n_sheets: int = len(self.sheet_names)
        if index >= n_sheets:
            raise ValueError(
                f"Worksheet index {index} is invalid, {n_sheets} worksheets found"
            )

    def func_uoftjbr0(self, name: str) -> None:
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    def func_5g3vofz6(
        self, skiprows: Callable[[int], bool], rows_to_use: int
    ) -> int:
        """
        Determine how many file rows are required to obtain `nrows` data
        rows when `skiprows` is a function.

        Parameters
        ----------
        skiprows : function
            The function passed to read_excel by the user.
        rows_to_use : int
            The number of rows that will be needed for the header and
            the data.

        Returns
        -------
        int
        """
        i: int = 0
        rows_used_so_far: int = 0
        while rows_used_so_far < rows_to_use:
            if not skiprows(i):
                rows_used_so_far += 1
            i += 1
        return i

    def func_e4e8rlwc(
        self,
        header: Optional[Union[int, List[int]]],
        index_col: Optional[Union[int, str, List[Union[int, str]]]],
        skiprows: Optional[Union[int, List[int], Callable[[int], bool]]],
        nrows: Optional[int],
    ) -> Optional[int]:
        """
        If nrows specified, find the number of rows needed from the
        file, otherwise return None.

        Parameters
        ----------
        header : int, list of int, or None
            See read_excel docstring.
        index_col : int, str, list of int, or None
            See read_excel docstring.
        skiprows : list-like, int, callable, or None
            See read_excel docstring.
        nrows : int or None
            See read_excel docstring.

        Returns
        -------
        int or None
        """
        if nrows is None:
            return None
        if header is None:
            header_rows: int = 1
        elif is_integer(header):
            header = cast(int, header)
            header_rows = 1 + header
        else:
            header = cast(List[int], header)
            header_rows = 1 + header[-1]
        if is_list_like(header) and index_col is not None:
            header = cast(List[int], header)
            if len(header) > 1:
                header_rows += 1
        if skiprows is None:
            return header_rows + nrows
        if is_integer(skiprows):
            skiprows = cast(int, skiprows)
            return header_rows + nrows + skiprows
        if is_list_like(skiprows):

            def func_xh1k7oqb(x: int) -> bool:
                return x in skiprows

            skiprows_func: Callable[[int], bool] = partial(func_xh1k7oqb, x=0)
            return self._check_skiprows_func(skiprows_func, header_rows + nrows)
        if callable(skiprows):
            return self._check_skiprows_func(skiprows, header_rows + nrows)
        return None

    def func_watsf5es(
        self,
        sheet_name: Union[str, int, List[Union[str, int]], None] = 0,
        header: Optional[Union[int, List[int]]] = 0,
        names: Optional[List[Any]] = None,
        index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
        usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
        dtype: Optional[Union[DtypeArg, Dict[str, DtypeArg]]] = None,
        true_values: Optional[List[Any]] = None,
        false_values: Optional[List[Any]] = None,
        skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
        nrows: Optional[int] = None,
        na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
        verbose: bool = False,
        parse_dates: Optional[Union[bool, List[Union[int, str]], List[List[int]], Dict[str, List[int]]]] = False,
        date_format: Optional[Union[str, Dict[str, str]]] = None,
        thousands: Optional[str] = None,
        decimal: str = ".",
        comment: Optional[str] = None,
        skipfooter: int = 0,
        dtype_backend: Union[DtypeBackend, lib.NoDefault] = lib.no_default,
        **kwds: Any,
    ) -> Union[DataFrame, Dict[str, DataFrame]]:
        validate_header_arg(header)
        validate_integer("nrows", nrows)
        ret_dict: bool = False
        if isinstance(sheet_name, list):
            sheets: List[Union[str, int]] = sheet_name
            ret_dict = True
        elif sheet_name is None:
            sheets = self.sheet_names
            ret_dict = True
        elif isinstance(sheet_name, str):
            sheets = [sheet_name]
        else:
            sheets = [sheet_name]
        sheets = cast(List[Union[str, int]], list(dict.fromkeys(sheets).keys()))
        output: Union[DataFrame, Dict[str, DataFrame]] = {}
        last_sheetname: Optional[Union[str, int]] = None
        for asheetname in sheets:
            last_sheetname = asheetname
            if verbose:
                print(f"Reading sheet {asheetname}")
            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:
                sheet = self.get_sheet_by_index(asheetname)
            file_rows_needed: Optional[int] = self._calc_rows(header, index_col, skiprows, nrows)
            data: Any = self.get_sheet_data(sheet, file_rows_needed)
            if hasattr(sheet, "close"):
                sheet.close()
            usecols_converted = maybe_convert_usecols(usecols)
            if not data:
                output[asheetname] = DataFrame()
                continue
            output = self._parse_sheet(
                data=data,
                output=output,
                asheetname=asheetname,
                header=header,
                names=names,
                index_col=index_col,
                usecols=usecols_converted,
                dtype=dtype,
                skiprows=skiprows,
                nrows=nrows,
                true_values=true_values,
                false_values=false_values,
                na_values=na_values,
                parse_dates=parse_dates,
                date_format=date_format,
                thousands=thousands,
                decimal=decimal,
                comment=comment,
                skipfooter=skipfooter,
                dtype_backend=dtype_backend,
                **kwds,
            )
        if last_sheetname is None:
            raise ValueError("Sheet name is an empty list")
        if ret_dict:
            return output
        else:
            return cast(DataFrame, output[last_sheetname])

    def func_10r5ffs7(
        self,
        data: Any,
        output: Union[DataFrame, Dict[str, DataFrame]],
        asheetname: Union[str, int],
        header: Optional[Union[int, List[int]]],
        names: Optional[List[Any]],
        index_col: Optional[Union[int, str, List[Union[int, str]]]],
        usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]],
        dtype: Optional[Union[DtypeArg, Dict[str, DtypeArg]]],
        skiprows: Optional[Union[int, List[int], Callable[[int], bool]]],
        nrows: Optional[int],
        true_values: Optional[List[Any]],
        false_values: Optional[List[Any]],
        na_values: Optional[Union[Any, List[Any], Dict[str, Any]]],
        parse_dates: Optional[Union[bool, List[Union[int, str]], List[List[int]], Dict[str, List[int]]]],
        date_format: Optional[Union[str, Dict[str, str]]],
        thousands: Optional[str],
        decimal: str,
        comment: Optional[str],
        skipfooter: int,
        dtype_backend: Union[DtypeBackend, lib.NoDefault],
        **kwds: Any,
    ) -> Union[DataFrame, Dict[str, DataFrame]]:
        is_list_header: bool = False
        is_len_one_list_header: bool = False
        if is_list_like(header):
            assert isinstance(header, Sequence)
            is_list_header = True
            if len(header) == 1:
                is_len_one_list_header = True
        if is_len_one_list_header:
            header = cast(List[int], header)[0]
        header_names: Optional[List[Any]] = None
        if header is not None and is_list_like(header):
            assert isinstance(header, Sequence)
            header_names = []
            control_row: List[bool] = [True] * len(data[0])
            for row in cast(List[int], header):
                if is_integer(skiprows):
                    assert isinstance(skiprows, int)
                    row += skiprows
                if row > len(data) - 1:
                    raise ValueError(
                        f"header index {row} exceeds maximum index {len(data) - 1} of data."
                    )
                data[row], control_row = fill_mi_header(data[row], control_row)
                if index_col is not None:
                    header_name, _ = pop_header_name(data[row], index_col)
                    header_names.append(header_name)
        has_index_names: bool = False
        if is_list_header and not is_len_one_list_header and index_col is not None:
            if isinstance(index_col, int):
                index_col_set: set = {index_col}
            else:
                assert isinstance(index_col, Sequence)
                index_col_set = set(index_col)
            assert isinstance(header, Sequence)
            if len(header) < len(data):
                potential_index_names = data[len(header)]
                has_index_names = all(
                    x == "" or x is None for i, x in enumerate(potential_index_names) if not control_row[i] and i not in index_col_set
                )
        if is_list_like(index_col):
            if header is None:
                offset = 0
            elif isinstance(header, int):
                offset = 1 + header
            else:
                offset = 1 + max(cast(List[int], header))
            if is_list_like(header) and index_col is not None and len(header) > 1:
                offset += 1
            if offset < len(data):
                assert isinstance(index_col, Sequence)
                for col in index_col:
                    last = data[offset][col]
                    for row in range(offset + 1, len(data)):
                        if data[row][col] == "" or data[row][col] is None:
                            data[row][col] = last
                        else:
                            last = data[row][col]
        try:
            parser = TextParser(
                data,
                names=names,
                header=header,
                index_col=index_col,
                has_index_names=has_index_names,
                dtype=dtype,
                true_values=true_values,
                false_values=false_values,
                skiprows=skiprows,
                nrows=nrows,
                na_values=na_values,
                skip_blank_lines=False,
                parse_dates=parse_dates,
                date_format=date_format,
                thousands=thousands,
                decimal=decimal,
                comment=comment,
                skipfooter=skipfooter,
                usecols=usecols,
                dtype_backend=dtype_backend,
                **kwds,
            )
            output[asheetname] = parser.read(nrows=nrows)
            if header_names:
                output[asheetname].columns = cast(DataFrame, output[asheetname]).columns.set_names(header_names)
        except EmptyDataError:
            output[asheetname] = DataFrame()
        except Exception as err:
            err.args = (f"{err.args[0]} (sheet: {asheetname})",) + err.args[1:]
            raise err
        return output


@doc(storage_options=_shared_docs['storage_options'])
class ExcelWriter(Generic[_WorkbookT]):
    """
    Class for writing DataFrame objects into excel sheets.

    Default is to use:

    * `xlsxwriter <https://pypi.org/project/XlsxWriter/>`__ for xlsx files if xlsxwriter
      is installed otherwise `openpyxl <https://pypi.org/project/openpyxl/>`__
    * `odf <https://pypi.org/project/odfpy/>`__ for ods files

    See :meth:`DataFrame.to_excel` for typical usage.

    The writer should be used as a context manager. Otherwise, call `close()` to save
    and close any opened file handles.

    Parameters
    ----------
    path : str or typing.BinaryIO
        Path to xls or xlsx or ods file.
    engine : str (optional)
        Engine to use for writing. If None, defaults to
        ``io.excel.<extension>.writer``.  NOTE: can only be passed as a keyword
        argument.
    date_format : str, default None
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
    datetime_format : str, default None
        Format string for datetime objects written into Excel files.
        (e.g. 'YYYY-MM-DD HH:MM:SS').
    mode : {{'w', 'a'}}, default 'w'
        File mode to use (write or append). Append does not work with fsspec URLs.
    {storage_options}

    if_sheet_exists : {{'error', 'new', 'replace', 'overlay'}}, default 'error'
        How to behave when trying to write to a sheet that already
        exists (append mode only).

        * error: raise a ValueError.
        * new: Create a new sheet, with a name determined by the engine.
        * replace: Delete the contents of the sheet before writing to it.
        * overlay: Write contents to the existing sheet without first removing,
          but possibly over top of, the existing contents.

        .. versionadded:: 1.3.0

        .. versionchanged:: 1.4.0

           Added ``overlay`` option

    engine_kwargs : dict, optional
        Keyword arguments to be passed into the engine. These will be passed to
        the following functions of the respective engines:

        * xlsxwriter: ``xlsxwriter.Workbook(file, **engine_kwargs)``
        * openpyxl (write mode): ``openpyxl.Workbook(**engine_kwargs)``
        * openpyxl (append mode): ``openpyxl.load_workbook(file, **engine_kwargs)``
        * odf: ``odf.opendocument.OpenDocumentSpreadsheet(**engine_kwargs)``

        .. versionadded:: 1.3.0

    See Also
    --------
    read_excel : Read an Excel sheet values (xlsx) file into DataFrame.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Notes
    -----
    For compatibility with CSV writers, ExcelWriter serializes lists
    and dicts to strings before writing.

    Examples
    --------
    Default usage:

    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    To write to separate sheets in a single file:

    >>> df1 = pd.DataFrame([["AAA", "BBB"]], columns=["Spam", "Egg"])  # doctest: +SKIP
    >>> df2 = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with pd.ExcelWriter("path_to_file.xlsx") as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP
    ...     df2.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP

    You can set the date format or datetime format:

    >>> from datetime import date, datetime  # doctest: +SKIP
    >>> df = pd.DataFrame(
    ...     [
    ...         [date(2014, 1, 31), date(1999, 9, 24)],
    ...         [datetime(1998, 5, 26, 23, 33, 4), datetime(2014, 2, 28, 13, 5, 13)],
    ...     ],
    ...     index=["Date", "Datetime"],
    ...     columns=["X", "Y"],
    ... )  # doctest: +SKIP
    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     date_format="YYYY-MM-DD",
    ...     datetime_format="YYYY-MM-DD HH:MM:SS",
    ... ) as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    You can also append to an existing Excel file:

    >>> with pd.ExcelWriter("path_to_file.xlsx", mode="a", engine="openpyxl") as writer:
    ...     df.to_excel(writer, sheet_name="Sheet3")  # doctest: +SKIP

    Here, the `if_sheet_exists` parameter can be set to replace a sheet if it
    already exists:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     mode="a",
    ...     engine="openpyxl",
    ...     if_sheet_exists="replace",
    ... ) as writer:
    ...     df.to_excel(writer, sheet_name="Sheet1")  # doctest: +SKIP

    You can also write multiple DataFrames to a single sheet. Note that the
    ``if_sheet_exists`` parameter needs to be set to ``overlay``:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     mode="a",
    ...     engine="openpyxl",
    ...     if_sheet_exists="overlay",
    ... ) as writer:
    ...     df1.to_excel(writer, sheet_name="Sheet1")
    ...     df2.to_excel(writer, sheet_name="Sheet1", startcol=3)  # doctest: +SKIP

    You can store Excel file in RAM:

    >>> import io
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])
    >>> buffer = io.BytesIO()
    >>> with pd.ExcelWriter(buffer) as writer:
    ...     df.to_excel(writer)

    You can pack Excel file into zip archive:

    >>> import zipfile  # doctest: +SKIP
    >>> df = pd.DataFrame([["ABC", "XYZ"]], columns=["Foo", "Bar"])  # doctest: +SKIP
    >>> with zipfile.ZipFile("path_to_file.zip", "w") as zf:
    ...     with zf.open("filename.xlsx", "w") as buffer:
    ...         with pd.ExcelWriter(buffer) as writer:
    ...             df.to_excel(writer)  # doctest: +SKIP

    You can specify additional arguments to the underlying engine:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     engine="xlsxwriter",
    ...     engine_kwargs={{"options": {{"nan_inf_to_errors": True}}}},
    ... ) as writer:
    ...     df.to_excel(writer)  # doctest: +SKIP

    In append mode, ``engine_kwargs`` are passed through to
    openpyxl's ``load_workbook``:

    >>> with pd.ExcelWriter(
    ...     "path_to_file.xlsx",
    ...     engine="openpyxl",
    ...     mode="a",
    ...     engine_kwargs={{"keep_vba": True}},
    ... ) as writer:
    ...     df.to_excel(writer, sheet_name="Sheet2")  # doctest: +SKIP
    """

    def __new__(
        cls,
        path: Union[str, IO[bytes]],
        engine: Optional[str] = None,
        date_format: Optional[str] = None,
        datetime_format: Optional[str] = None,
        mode: Literal["w", "a"] = "w",
        storage_options: Optional[StorageOptions] = None,
        if_sheet_exists: Optional[ExcelWriterIfSheetExists] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ExcelWriter:
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

    _path: Optional[str] = None

    @property
    def func_65mvfn5p(self) -> List[str]:
        """Extensions that writer engine supports."""
        return self._supported_extensions

    @property
    def func_z699qf8f(self) -> str:
        """Name of engine."""
        return self._engine

    @property
    def func_bpxpalwh(self) -> Dict[str, Any]:
        """Mapping of sheet names to sheet objects."""
        raise NotImplementedError

    @property
    def func_869ucxly(self) -> Any:
        """
        Book instance. Class type will depend on the engine used.

        This attribute can be used to access engine-specific features.
        """
        raise NotImplementedError

    def func_eol2pkau(
        self,
        cells: Callable[[], Iterable[Any]],
        sheet_name: Optional[str] = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Write given formatted cells into Excel an excel sheet

        Parameters
        ----------
        cells : generator
            cell of formatted data to save to Excel sheet
        sheet_name : str, default None
            Name of Excel sheet, if None, then use self.cur_sheet
        startrow : upper left cell row to dump data frame
        startcol : upper left cell column to dump data frame
        freeze_panes: int tuple of length 2
            contains the bottom-most row and right-most column to freeze
        """
        raise NotImplementedError

    def func_wrh57rd5(self) -> None:
        """
        Save workbook to disk.
        """
        raise NotImplementedError

    def __init__(
        self,
        path: Union[str, IO[bytes]],
        engine: Optional[str] = None,
        date_format: Optional[str] = None,
        datetime_format: Optional[str] = None,
        mode: Literal["w", "a"] = "w",
        storage_options: Optional[StorageOptions] = None,
        if_sheet_exists: Optional[ExcelWriterIfSheetExists] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(path, str):
            ext = os.path.splitext(path)[-1]
            self.check_extension(ext)
        if "b" not in mode:
            mode += "b"
        mode = mode.replace("a", "r+")
        if if_sheet_exists not in (None, "error", "new", "replace", "overlay"):
            raise ValueError(
                f"'{if_sheet_exists}' is not valid for if_sheet_exists. Valid options are 'error', 'new', 'replace' and 'overlay'."
            )
        if if_sheet_exists and "r+" not in mode:
            raise ValueError(
                "if_sheet_exists is only valid in append mode (mode='a')"
            )
        if if_sheet_exists is None:
            if_sheet_exists = "error"
        self._if_sheet_exists: ExcelWriterIfSheetExists = if_sheet_exists
        self._handles: IOHandles = IOHandles(cast(IO[bytes], path), compression={"compression": None})
        if not isinstance(path, ExcelWriter):
            self._handles = get_handle(
                path, mode, storage_options=storage_options, is_text=False
            )
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
    def func_5fs57wkg(self) -> str:
        """
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
        """
        return self._date_format

    @property
    def func_4lo7frcp(self) -> str:
        """
        Format string for dates written into Excel files (e.g. 'YYYY-MM-DD').
        """
        return self._datetime_format

    @property
    def func_vy26sh3r(self) -> ExcelWriterIfSheetExists:
        """
        How to behave when writing to a sheet that already exists in append mode.
        """
        return self._if_sheet_exists

    def __fspath__(self) -> str:
        return getattr(self._handles.handle, "name", "")

    def func_wkwo4yvx(self, sheet_name: Optional[str] = None) -> str:
        if sheet_name is None:
            sheet_name = self._cur_sheet
        if sheet_name is None:
            raise ValueError(
                "Must pass explicit sheet_name or set _cur_sheet property"
            )
        return sheet_name

    def func_9opqdeyx(
        self, val: Any
    ) -> Tuple[Union[int, float, bool, Decimal, str], Optional[str]]:
        """
        Convert numpy types to Python types for the Excel writers.

        Parameters
        ----------
        val : object
            Value to be written into cells

        Returns
        -------
        Tuple with the first element being the converted value and the second
            being an optional format
        """
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
                warnings.warn(
                    f"Cell contents too long ({len(val)}), truncated to 32767 characters",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
        return val, fmt

    @classmethod
    def func_96vm8cbj(cls, ext: str) -> bool:
        """
        checks that path's extension against the Writer's supported
        extensions.  If it isn't supported, raises UnsupportedFiletypeError.
        """
        if ext.startswith("."):
            ext = ext[1:]
        if not any(ext in extension for extension in cls._supported_extensions):
            raise ValueError(
                f"Invalid extension for engine '{cls.engine}': '{ext}'"
            )
        return True

    def __enter__(self) -> ExcelWriter:
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self.close()

    def func_c5u0hzi0(self) -> None:
        """synonym for save, to make it more file-like"""
        self._save()
        self._handles.close()


XLS_SIGNATURES: Tuple[bytes, ...] = (
    b"\t\x00\x04\x00\x07\x00\x10\x00",
    b"\t\x02\x06\x00\x00\x00\x10\x00",
    b"\t\x04\x06\x00\x00\x00\x10\x00",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
)
ZIP_SIGNATURE: bytes = b"PK\x03\x04"
PEEK_SIZE: int = max(map(len, XLS_SIGNATURES + (ZIP_SIGNATURE,)))


@doc(storage_options=_shared_docs['storage_options'])
def func_ftqu2y27(
    content_or_path: Union[str, IO[Any]],
    storage_options: Optional[StorageOptions] = None,
) -> Optional[str]:
    """
    Inspect the path or content of an excel file and get its format.

    Adopted from xlrd: https://github.com/python-excel/xlrd.

    Parameters
    ----------
    content_or_path : str or file-like object
        Path to file or content of file to inspect. May be a URL.
    {storage_options}

    Returns
    -------
    str or None
        Format of file if it can be determined.

    Raises
    ------
    ValueError
        If resulting stream is empty.
    BadZipFile
        If resulting stream does not have an XLS signature and is not a valid zipfile.
    """
    with get_handle(
        content_or_path,
        "rb",
        storage_options=storage_options,
        is_text=False,
    ) as handle:
        stream = handle.handle
        stream.seek(0)
        buf: bytes = stream.read(PEEK_SIZE)
        if buf is None:
            raise ValueError("stream is empty")
        peek: bytes = buf
        stream.seek(0)
        if any(peek.startswith(sig) for sig in XLS_SIGNATURES):
            return "xls"
        elif not peek.startswith(ZIP_SIGNATURE):
            return None
        with zipfile.ZipFile(stream) as zf:
            component_names = {name.replace("\\", "/").lower() for name in zf.namelist()}
        if "xl/workbook.xml" in component_names:
            return "xlsx"
        if "xl/workbook.bin" in component_names:
            return "xlsb"
        if "content.xml" in component_names:
            return "ods"
        return "zip"


@doc(storage_options=_shared_docs['storage_options'])
class ExcelFile:
    """
    Class for parsing tabular Excel sheets into DataFrame objects.

    See read_excel for more documentation.

    Parameters
    ----------
    path_or_buffer : str, bytes, pathlib.Path,
        A file-like object, xlrd workbook or openpyxl workbook.
        If a string or path object, expected to be a path to a
        .xls, .xlsx, .xlsb, .xlsm, .odf, .ods, or .odt file.
    engine : str, default None
        If io is not a buffer or path, this must be set to identify io.
        Supported engines: ``xlrd``, ``openpyxl``, ``odf``, ``pyxlsb``, ``calamine``
        Engine compatibility :

        - ``xlrd`` supports old-style Excel files (.xls).
        - ``openpyxl`` supports newer Excel file formats.
        - ``odf`` supports OpenDocument file formats (.odf, .ods, .odt).
        - ``pyxlsb`` supports Binary Excel files.
        - ``calamine`` supports Excel (.xls, .xlsx, .xlsm, .xlsb)
          and OpenDocument (.ods) file formats.

        .. versionchanged:: 1.2.0

           The engine `xlrd <https://xlrd.readthedocs.io/en/latest/>`_
           now only supports old-style ``.xls`` files.
           When ``engine=None``, the following logic will be
           used to determine the engine:

           - If ``path_or_buffer`` is an OpenDocument format (.odf, .ods, .odt),
             then `odf <https://pypi.org/project/odfpy/>`_ will be used.
           - Otherwise if ``path_or_buffer`` is an xls format,
             ``xlrd`` will be used.
           - Otherwise if ``path_or_buffer`` is in xlsb format,
             `pyxlsb <https://pypi.org/project/pyxlsb/>`_ will be used.

        .. versionadded:: 1.3.0

           - Otherwise if `openpyxl <https://pypi.org/project/openpyxl/>`_ is installed,
             then ``openpyxl`` will be used.
           - Otherwise if ``xlrd >= 2.0`` is installed, a ``ValueError`` will be raised.

        .. warning::

           Please do not report issues when using ``xlrd`` to read ``.xlsx`` files.
           This is not supported, switch to using ``openpyxl`` instead.
    {storage_options}
    engine_kwargs : dict, optional
        Arbitrary keyword arguments passed to excel engine.

    See Also
    --------
    DataFrame.to_excel : Write DataFrame to an Excel file.
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_fwf : Read a table of fixed-width formatted lines into DataFrame.

    Examples
    --------
    >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
    >>> with pd.ExcelFile("myfile.xls") as xls:  # doctest: +SKIP
    ...     df1 = pd.read_excel(xls, "Sheet1")  # doctest: +SKIP
    """

    from pandas.io.excel._calamine import CalamineReader
    from pandas.io.excel._odfreader import ODFReader
    from pandas.io.excel._openpyxl import OpenpyxlReader
    from pandas.io.excel._pyxlsb import PyxlsbReader
    from pandas.io.excel._xlrd import XlrdReader

    _engines: Dict[str, Any] = {
        "xlrd": XlrdReader,
        "openpyxl": OpenpyxlReader,
        "odf": ODFReader,
        "pyxlsb": PyxlsbReader,
        "calamine": CalamineReader,
    }

    def __init__(
        self,
        path_or_buffer: Union[str, bytes, os.PathLike, ExcelFile, IO[Any], Any],
        engine: Optional[str] = None,
        storage_options: Optional[StorageOptions] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        if engine is not None and engine not in self._engines:
            raise ValueError(f"Unknown engine: {engine}")
        self._io: Union[str, Any] = stringify_path(path_or_buffer)
        if engine is None:
            ext: Optional[str] = None
            if not isinstance(
                path_or_buffer, (str, os.PathLike, ExcelFile)
            ) and not is_file_like(path_or_buffer):
                if import_optional_dependency("xlrd", errors="ignore") is None:
                    xlrd_version: Optional[Version] = None
                else:
                    import xlrd

                    xlrd_version = Version(get_version(xlrd))
                if xlrd_version is not None and isinstance(path_or_buffer, xlrd.Book):
                    ext = "xls"
            if ext is None:
                ext = func_ftqu2y27(
                    content_or_path=path_or_buffer, storage_options=storage_options
                )
                if ext is None:
                    raise ValueError(
                        "Excel file format cannot be determined, you must specify an engine manually."
                    )
            engine = config.get_option(f"io.excel.{ext}.reader")
            if engine == "auto":
                engine = get_default_engine(ext, mode="reader")
        assert engine is not None
        self.engine: str = engine
        self.storage_options: Optional[StorageOptions] = storage_options
        self._reader = self._engines[engine](
            self._io, storage_options=storage_options, engine_kwargs=engine_kwargs
        )

    def __fspath__(self) -> str:
        return self._io

    def func_watsf5es(
        self,
        sheet_name: Optional[Union[str, int, List[Union[str, int]]]] = 0,
        header: Optional[Union[int, List[int]]] = 0,
        names: Optional[List[Any]] = None,
        index_col: Optional[Union[int, str, List[Union[int, str]]]] = None,
        usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
        converters: Optional[Dict[Union[int, str], Callable[[Any], Any]]] = None,
        true_values: Optional[List[Any]] = None,
        false_values: Optional[List[Any]] = None,
        skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
        nrows: Optional[int] = None,
        na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
        parse_dates: Optional[Union[bool, List[Union[int, str]], List[List[int]], Dict[str, List[int]]]] = False,
        date_format: Optional[Union[str, Dict[str, str]]] = None,
        thousands: Optional[str] = None,
        comment: Optional[str] = None,
        skipfooter: int = 0,
        dtype_backend: Union[DtypeBackend, lib.NoDefault] = lib.no_default,
        **kwds: Any,
    ) -> Union[DataFrame, Dict[str, DataFrame]]:
        """
        Parse specified sheet(s) into a DataFrame.

        Equivalent to read_excel(ExcelFile, ...)  See the read_excel
        docstring for more info on accepted parameters.

        Parameters
        ----------
        sheet_name : str, int, list, or None, default 0
            Strings are used for sheet names. Integers are used in zero-indexed
            sheet positions (chart sheets do not count as a sheet position).
            Lists of strings/integers are used to request multiple sheets.
            When ``None``, will return a dictionary containing DataFrames for
            each sheet.
        header : int, list of int, default 0
            Row (0-indexed) to use for the column labels of the parsed
            DataFrame. If a list of integers is passed those row positions will
            be combined into a ``MultiIndex``. Use None if there is no header.
        names : array-like, default None
            List of column names to use. If file contains no header row,
            then you should explicitly pass header=None.
        index_col : int, str, list of int, default None
            Column (0-indexed) to use as the row labels of the DataFrame.
            Pass None if there is no such column.  If a list is passed,
            those columns will be combined into a ``MultiIndex``.  If a
            subset of data is selected with ``usecols``, index_col
            is based on the subset.

            Missing values will be forward filled to allow roundtripping with
            ``to_excel`` for ``merged_cells=True``. To avoid forward filling the
            missing values use ``set_index`` after reading the data instead of
            ``index_col``.
        usecols : str, list-like, or callable, default None
            * If None, then parse all columns.
            * If str, then indicates comma separated list of Excel column letters
              and column ranges (e.g. "A:E" or "A,C,E:F"). Ranges are inclusive of
              both sides.
            * If list of int, then indicates list of column numbers to be parsed
              (0-indexed).
            * If list of string, then indicates list of column names to be parsed.
            * If callable, then evaluate each column name against it and parse the
              column if the callable returns ``True``.

            Returns a subset of the columns according to behavior above.
        converters : dict, default None
            Dict of functions for converting values in certain columns. Keys can
            either be integers or column labels, values are functions that take one
            input argument, the Excel cell content, and return the transformed
            content.
        true_values : list, default None
            Values to consider as True.
        false_values : list, default None
            Values to consider as False.
        skiprows : list-like, int, or callable, optional
            Line numbers to skip (0-indexed) or number of lines to skip (int) at the
            start of the file. If callable, the callable function will be evaluated
            against the row indices, returning True if the row should be skipped and
            False otherwise. An example of a valid callable argument would be ``lambda
            x: x in [0, 2]``.
        nrows : int, default None
            Number of rows to parse.
        na_values : scalar, str, list-like, or dict, default None
            Additional strings to recognize as NA/NaN. If dict passed, specific
            per-column NA values.
        parse_dates : bool, list-like, or dict, default False
            The behavior is as follows:

            * ``bool``. If True -> try parsing the index.
            * ``list`` of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
              each as a separate date column.
            * ``list`` of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and
              parse as a single date column.
            * ``dict``, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call
              result 'foo'

            If a column or index contains an unparsable date, the entire column or
            index will be returned unaltered as an object data type. If you
            don`t want to parse some cells as date just change their type
            in Excel to "Text".For non-standard datetime parsing, use
            ``pd.to_datetime`` after ``pd.read_excel``.
        date_format : str or dict of column -> format, default ``None``
           If used in conjunction with ``parse_dates``, will parse dates
           according to this format. For anything more complex,
           please read in as ``object`` and then apply :func:`to_datetime` as-needed.
        thousands : str, default None
            Thousands separator for parsing string columns to numeric.  Note that
            this parameter is only necessary for columns stored as TEXT in Excel,
            any numeric columns will automatically be parsed, regardless of display
            format.
        comment : str, default None
            Comments out remainder of line. Pass a character or characters to this
            argument to indicate comments in the input file. Any data between the
            comment string and the end of the current line is ignored.
        skipfooter : int, default 0
            Rows at the end to skip (0-indexed).
        dtype_backend : {{'numpy_nullable', 'pyarrow'}}
            Back-end data type applied to the resultant :class:`DataFrame`
            (still experimental). If not specified, the default behavior
            is to not use nullable data types. If specified, the behavior
            is as follows:

            * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
            * ``"pyarrow"``: returns pyarrow-backed nullable
              :class:`ArrowDtype` :class:`DataFrame`

            .. versionadded:: 2.0
        **kwds : dict, optional
            Arbitrary keyword arguments passed to excel engine.

        Returns
        -------
        DataFrame or dict of DataFrames
            DataFrame from the passed in Excel file.

        See Also
        --------
        read_excel : Read an Excel sheet values (xlsx) file into DataFrame.
        read_csv : Read a comma-separated values (csv) file into DataFrame.
        read_fwf : Read a table of fixed-width formatted lines into DataFrame.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
        >>> df.to_excel("myfile.xlsx")  # doctest: +SKIP
        >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
        >>> file.parse()  # doctest: +SKIP
        """
        return self._reader.parse(
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            parse_dates=parse_dates,
            date_format=date_format,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            dtype_backend=dtype_backend,
            **kwds,
        )

    @property
    def func_869ucxly(self) -> Any:
        """
        Gets the Excel workbook.

        Workbook is the top-level container for all document information.

        Returns
        -------
        Excel Workbook
            The workbook object of the type defined by the engine being used.

        See Also
        --------
        read_excel : Read an Excel file into a pandas DataFrame.

        Examples
        --------
        >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
        >>> file.book  # doctest: +SKIP
        <openpyxl.workbook.workbook.Workbook object at 0x11eb5ad70>
        >>> file.book.path  # doctest: +SKIP
        '/xl/workbook.xml'
        >>> file.book.active  # doctest: +SKIP
        <openpyxl.worksheet._read_only.ReadOnlyWorksheet object at 0x11eb5b370>
        >>> file.book.sheetnames  # doctest: +SKIP
        ['Sheet1', 'Sheet2']
        """
        return self._reader.book

    @property
    def func_uffdua8n(self) -> List[str]:
        """
        Names of the sheets in the document.

        This is particularly useful for loading a specific sheet into a DataFrame when
        you do not know the sheet names beforehand.

        Returns
        -------
        list of str
            List of sheet names in the document.

        See Also
        --------
        ExcelFile.parse : Parse a sheet into a DataFrame.
        read_excel : Read an Excel file into a pandas DataFrame. If you know the sheet
            names, it may be easier to specify them directly to read_excel.

        Examples
        --------
        >>> file = pd.ExcelFile("myfile.xlsx")  # doctest: +SKIP
        >>> file.sheet_names  # doctest: +SKIP
        ["Sheet1", "Sheet2"]
        """
        return self._reader.sheet_names

    def func_c5u0hzi0(self) -> None:
        """close io if necessary"""
        self._reader.close()

    def __enter__(self) -> ExcelFile:
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self.close()
