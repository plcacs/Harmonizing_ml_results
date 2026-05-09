from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import datetime
from decimal import Decimal
from functools import partial
import os
from textwrap import fill
from typing import IO, TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union, cast, overload
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

_WorkbookT = TypeVar('_WorkbookT')
DtypeBackendT = Literal['numpy_nullable', 'pyarrow']

@overload
def read_excel(io: FilePath | ReadBuffer, sheet_name: Union[int, str, list[int | str], None] = ..., *, 
               header: Union[int, list[int], None] = ..., names: list[str] = ..., index_col: Union[int, str, list[int | str], None] = ..., 
               usecols: Union[str, list[int | str], Callable[[str], bool], None] = ..., dtype: Union[DtypeArg, None] = ..., 
               engine: str = ..., converters: dict[Hashable, Callable[[str], Any], None] = ..., true_values: list[str, None] = ..., 
               false_values: list[str, None] = ..., skiprows: Union[int, list[int], Callable[[int], bool], None] = ..., 
               nrows: int = ..., na_values: Union[str, list[str], dict[Hashable, str], None] = ..., keep_default_na: bool = ..., 
               na_filter: bool = ..., verbose: bool = ..., parse_dates: Union[bool, list[int | str], list[list[int | str]], dict[str, list[int | str]], None] = ..., 
               date_format: str = ..., thousands: str = ..., decimal: str = ..., comment: str = ..., skipfooter: int = ..., 
               storage_options: StorageOptions = ..., dtype_backend: DtypeBackendT = ..., engine_kwargs: dict[str, Any] = ...) -> DataFrame | dict[str, DataFrame]:
    ...

@overload
def read_excel(io: FilePath | ReadBuffer, sheet_name: Union[int, str, list[int | str], None], *, 
               header: Union[int, list[int], None] = ..., names: list[str] = ..., index_col: Union[int, str, list[int | str], None] = ..., 
               usecols: Union[str, list[int | str], Callable[[str], bool], None] = ..., dtype: Union[DtypeArg, None] = ..., 
               engine: str = ..., converters: dict[Hashable, Callable[[str], Any], None] = ..., true_values: list[str, None] = ..., 
               false_values: list[str, None] = ..., skiprows: Union[int, list[int], Callable[[int], bool], None] = ..., 
               nrows: int = ..., na_values: Union[str, list[str], dict[Hashable, str], None] = ..., keep_default_na: bool = ..., 
               na_filter: bool = ..., verbose: bool = ..., parse_dates: Union[bool, list[int | str], list[list[int | str]], dict[str, list[int | str]], None] = ..., 
               date_format: str = ..., thousands: str = ..., decimal: str = ..., comment: str = ..., skipfooter: int = ..., 
               storage_options: StorageOptions = ..., dtype_backend: DtypeBackendT = ..., engine_kwargs: dict[str, Any] = ...) -> DataFrame | dict[str, DataFrame]:
    ...

@doc(storage_options=_shared_docs['storage_options'])
@Appender(_read_excel_doc)
def read_excel(io: FilePath | ReadBuffer, sheet_name: Union[int, str, list[int | str], None] = 0, *, 
               header: Union[int, list[int], None] = 0, names: list[str] = None, index_col: Union[int, str, list[int | str], None] = None, 
               usecols: Union[str, list[int | str], Callable[[str], bool], None] = None, dtype: Union[DtypeArg, None] = None, 
               engine: str = None, converters: dict[Hashable, Callable[[str], Any], None] = None, true_values: list[str, None] = None, 
               false_values: list[str, None] = None, skiprows: Union[int, list[int], Callable[[int], bool], None] = None, 
               nrows: int = None, na_values: Union[str, list[str], dict[Hashable, str], None] = None, keep_default_na: bool = True, 
               na_filter: bool = True, verbose: bool = False, parse_dates: Union[bool, list[int | str], list[list[int | str]], dict[str, list[int | str]], None] = False, 
               date_format: str = None, thousands: str = None, decimal: str = '.', comment: str = None, skipfooter: int = 0, 
               storage_options: StorageOptions = None, dtype_backend: DtypeBackendT = lib.no_default, engine_kwargs: dict[str, Any] = None) -> DataFrame | dict[str, DataFrame]:
    check_dtype_backend(dtype_backend)
    should_close = False
    if engine_kwargs is None:
        engine_kwargs = {}
    if not isinstance(io, ExcelFile):
        should_close = True
        io = ExcelFile(io, storage_options=storage_options, engine=engine, engine_kwargs=engine_kwargs)
    elif engine and engine != io.engine:
        raise ValueError('Engine should not be specified when passing an ExcelFile - ExcelFile already has the engine set')
    try:
        data = io.parse(sheet_name=sheet_name, header=header, names=names, index_col=index_col, usecols=usecols, dtype=dtype, converters=converters, true_values=true_values, false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values, keep_default_na=keep_default_na, na_filter=na_filter, verbose=verbose, parse_dates=parse_dates, date_format=date_format, thousands=thousands, decimal=decimal, comment=comment, skipfooter=skipfooter, dtype_backend=dtype_backend)
    finally:
        if should_close:
            io.close()
    return data

class BaseExcelReader(Generic[_WorkbookT]):
    def __init__(self, filepath_or_buffer: FilePath | ReadBuffer, storage_options: StorageOptions = None, engine_kwargs: dict[str, Any] = None) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        self.handles = IOHandles(handle=filepath_or_buffer, compression={'method': None})
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            self.handles = get_handle(filepath_or_buffer, 'rb', storage_options=storage_options, is_text=False)
        if isinstance(self.handles.handle, self._workbook_class):
            self.book = self.handles.handle
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
    def _workbook_class(self) -> type[_WorkbookT]:
        raise NotImplementedError

    def load_workbook(self, filepath_or_buffer: FilePath | ReadBuffer, engine_kwargs: dict[str, Any]) -> _WorkbookT:
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self, 'book'):
            if hasattr(self.book, 'close'):
                self.book.close()
            elif hasattr(self.book, 'release_resources'):
                self.book.release_resources()
        self.handles.close()

    @property
    def sheet_names(self) -> list[str]:
        raise NotImplementedError

    def get_sheet_by_name(self, name: str) -> _WorkbookT:
        raise NotImplementedError

    def get_sheet_by_index(self, index: int) -> _WorkbookT:
        raise NotImplementedError

    def get_sheet_data(self, sheet: _WorkbookT, rows: int = None) -> list[list[Any]]:
        raise NotImplementedError

    def raise_if_bad_sheet_by_index(self, index: int) -> None:
        n_sheets = len(self.sheet_names)
        if index >= n_sheets:
            raise ValueError(f'Worksheet index {index} is invalid, {n_sheets} worksheets found')

    def raise_if_bad_sheet_by_name(self, name: str) -> None:
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    def _check_skiprows_func(self, skiprows: Callable[[int], bool], rows_to_use: int) -> int:
        i = 0
        rows_used_so_far = 0
        while rows_used_so_far < rows_to_use:
            if not skiprows(i):
                rows_used_so_far += 1
            i += 1
        return i

    def _calc_rows(self, header: Union[int, list[int], None], index_col: Union[int, str, list[int | str], None], skiprows: Union[int, list[int], Callable[[int], bool], None], nrows: int) -> int | None:
        if nrows is None:
            return None
        if header is None:
            header_rows = 1
        elif is_integer(header):
            header = cast(int, header)
            header_rows = 1 + header
        else:
            header = cast(Sequence, header)
            header_rows = 1 + header[-1]
        if is_list_like(header) and index_col is not None:
            header = cast(Sequence, header)
            if len(header) > 1:
                header_rows += 1
        if skiprows is None:
            return header_rows + nrows
        if is_integer(skiprows):
            skiprows = cast(int, skiprows)
            return header_rows + nrows + skiprows
        if is_list_like(skiprows):
            def f(skiprows: list[int], x: int) -> bool:
                return x in skiprows
            skiprows = cast(Sequence, skiprows)
            return self._check_skiprows_func(partial(f, skiprows), header_rows + nrows)
        if callable(skiprows):
            return self._check_skiprows_func(skiprows, header_rows + nrows)
        return None

    def parse(self, sheet_name: Union[int, str, list[int | str], None] = 0, header: Union[int, list[int], None] = 0, names: list[str] = None, index_col: Union[int, str, list[int | str], None] = None, usecols: Union[str, list[int | str], Callable[[str], bool], None] = None, dtype: Union[DtypeArg, None] = None, true_values: list[str, None] = None, false_values: list[str, None] = None, skiprows: Union[int, list[int], Callable[[int], bool], None] = None, nrows: int = None, na_values: Union[str, list[str], dict[Hashable, str], None] = None, verbose: bool = False, parse_dates: Union[bool, list[int | str], list[list[int | str]], dict[str, list[int | str]], None] = False, date_format: str = None, thousands: str = None, decimal: str = '.', comment: str = None, skipfooter: int = 0, dtype_backend: DtypeBackendT = lib.no_default, **kwds: Any) -> dict[str, DataFrame]:
        validate_header_arg(header)
        validate_integer('nrows', nrows)
        ret_dict = False
        if isinstance(sheet_name, list):
            sheets = sheet_name
            ret_dict = True
        elif sheet_name is None:
            sheets = self.sheet_names
            ret_dict = True
        elif isinstance(sheet_name, str):
            sheets = [sheet_name]
        else:
            sheets = [sheet_name]
        sheets = cast(Union[list[int], list[str]], list(dict.fromkeys(sheets).keys()))
        output = {}
        last_sheetname = None
        for asheetname in sheets:
            last_sheetname = asheetname
            if verbose:
                print(f'Reading sheet {asheetname}')
            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:
                sheet = self.get_sheet_by_index(asheetname)
            file_rows_needed = self._calc_rows(header, index_col, skiprows, nrows)
            data = self.get_sheet_data(sheet, file_rows_needed)
            if hasattr(sheet, 'close'):
                sheet.close()
            usecols = maybe_convert_usecols(usecols)
            if not data:
                output[asheetname] = DataFrame()
                continue
            output = self._parse_sheet(data=data, output=output, asheetname=asheetname, header=header, names=names, index_col=index_col, usecols=usecols, dtype=dtype, skiprows=skiprows, nrows=nrows, true_values=true_values, false_values=false_values, na_values=na_values, parse_dates=parse_dates, date_format=date_format, thousands=thousands, decimal=decimal, comment=comment, skipfooter=skipfooter, dtype_backend=dtype_backend, **kwds)
        if last_sheetname is None:
            raise ValueError('Sheet name is an empty list')
        if ret_dict:
            return output
        else:
            return output[last_sheetname]

    def _parse_sheet(self, data: list[list[Any]], output: dict[str, DataFrame], asheetname: str = None, header: Union[int, list[int], None] = 0, names: list[str] = None, index_col: Union[int, str, list[int | str], None] = None, usecols: Union[str, list[int | str], Callable[[str], bool], None] = None, dtype: Union[DtypeArg, None] = None, skiprows: Union[int, list[int], Callable[[int], bool], None] = None, nrows: int = None, true_values: list[str, None] = None, false_values: list[str, None] = None, na_values: Union[str, list[str], dict[Hashable, str], None] = None, parse_dates: Union[bool, list[int | str], list[list[int | str]], dict[str, list[int | str]], None] = False, date_format: str = None, thousands: str = None, decimal: str = '.', comment: str = None, skipfooter: int = 0, dtype_backend: DtypeBackendT = lib.no_default, **kwds: Any) -> dict[str, DataFrame]:
        is_list_header = False
        is_len_one_list_header = False
        if is_list_like(header):
            assert isinstance(header, Sequence)
            is_list_header = True
            if len(header) == 1:
                is_len_one_list_header = True
        if is_len_one_list_header:
            header = cast(Sequence[int], header)[0]
        header_names = None
        if header is not None and is_list_like(header):
            assert isinstance(header, Sequence)
            header_names = []
            control_row = [True] * len(data[0])
            for row in header:
                if is_integer(skiprows):
                    assert isinstance(skiprows, int)
                    row += skiprows
                if row > len(data) - 1:
                    raise ValueError(f'header index {row} exceeds maximum index {len(data) - 1} of data.')
                data[row], control_row = fill_mi_header(data[row], control_row)
                if index_col is not None:
                    header_name, _ = pop_header_name(data[row], index_col)
                    header_names.append(header_name)
        has_index_names = False
        if is_list_header and (not is_len_one_list_header) and (index_col is not None):
            if isinstance(index_col, int):
                index_col_set = {index_col}
            else:
                assert isinstance(index_col, Sequence)
                index_col_set = set(index_col)
            assert isinstance(header, Sequence)
            if len(header) < len(data):
                potential_index_names = data[len(header)]
                has_index_names = all((x == '' or x is None for i, x in enumerate(potential_index_names) if not control_row[i] and i not in index_col_set))
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
                assert isinstance(index_col, Sequence)
                for col in index_col:
                    last = data[offset][col]
                    for row in range(offset + 1, len(data)):
                        if data[row][col] == '' or data[row][col] is None:
                            data[row][col] = last
                        else:
                            last = data[row][col]
        try:
            parser = TextParser(data, names=names, header=header, index_col=index_col, has_index_names=has_index_names, dtype=dtype, true_values=true_values, false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values, skip_blank_lines=False, parse_dates=parse_dates, date_format=date_format, thousands=thousands, decimal=decimal, comment=comment, skipfooter=skipfooter, usecols=usecols, dtype_backend=dtype_backend, **kwds)
            output[asheetname] = parser.read(nrows=nrows)
            if header_names:
                output[asheetname].columns = output[asheetname].columns.set_names(header_names)
        except EmptyDataError:
            output[asheetname] = DataFrame()
        except Exception as err:
            err.args = (f'{err.args[0]} (sheet: {asheetname})', *err.args[1:])
            raise err
        return output

@doc(storage_options=_shared_docs['storage_options'])
class ExcelWriter(Generic[_WorkbookT]):
    def __new__(cls, path: FilePath | WriteExcelBuffer, engine: str = None, date_format: str = None, datetime_format: str = None, mode: Literal['w', 'a'] = 'w', storage_options: StorageOptions = None, if_sheet_exists: ExcelWriterIfSheetExists = None, engine_kwargs: dict[str, Any] = None) -> ExcelWriter[_WorkbookT]:
        if cls is ExcelWriter:
            if engine is None or (isinstance(engine, str) and engine == 'auto'):
                if isinstance(path, str):
                    ext = os.path.splitext(path)[-1][1:]
                else:
                    ext = 'xlsx'
                try:
                    engine = config.get_option(f'io.excel.{ext}.writer')
                    if engine == 'auto':
                        engine = get_default_engine(ext, mode='writer')
                except KeyError as err:
                    raise ValueError(f"No engine for filetype: '{ext}'") from err
            assert engine is not None
            cls = get_writer(engine)
        return object.__new__(cls)

    _path = None

    @property
    def supported_extensions(self) -> list[str]:
        return self._supported_extensions

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def sheets(self) -> dict[str, _WorkbookT]:
        raise NotImplementedError

    @property
    def book(self) -> _WorkbookT:
        raise NotImplementedError

    def _write_cells(self, cells: Iterable[tuple[Any, Any]], sheet_name: str = None, startrow: int = 0, startcol: int = 0, freeze_panes: tuple[int, int] = None) -> None:
        raise NotImplementedError

    def _save(self) -> None:
        raise NotImplementedError

    def __init__(self, path: FilePath | WriteExcelBuffer, engine: str = None, date_format: str = None, datetime_format: str = None, mode: Literal['w', 'a'] = 'w', storage_options: StorageOptions = None, if_sheet_exists: ExcelWriterIfSheetExists = None, engine_kwargs: dict[str, Any] = None) -> None:
        if isinstance(path, str):
            ext = os.path.splitext(path)[-1]
            self.check_extension(ext)
        if 'b' not in mode:
            mode += 'b'
        mode = mode.replace('a', 'r+')
        if if_sheet_exists not in (None, 'error', 'new', 'replace', 'overlay'):
            raise ValueError(f"'{if_sheet_exists}' is not valid for if_sheet_exists. Valid options are 'error', 'new', 'replace' and 'overlay'.")
        if if_sheet_exists and 'r+' not in mode:
            raise ValueError("if_sheet_exists is only valid in append mode (mode='a')")
        if if_sheet_exists is None:
            if_sheet_exists = 'error'
        self._if_sheet_exists = if_sheet_exists
        self._handles = IOHandles(cast(IO[bytes], path), compression={'compression': None})
        if not isinstance(path, ExcelWriter):
            self._handles = get_handle(path, mode, storage_options=storage_options, is_text=False)
        self._cur_sheet = None
        if date_format is None:
            self._date_format = 'YYYY-MM-DD'
        else:
            self._date_format = date_format
        if datetime_format is None:
            self._datetime_format = 'YYYY-MM-DD HH:MM:SS'
        else:
            self._datetime_format = datetime_format
        self._mode = mode

    @property
    def date_format(self) -> str:
        return self._date_format

    @property
    def datetime_format(self) -> str:
        return self._datetime_format

    @property
    def if_sheet_exists(self) -> ExcelWriterIfSheetExists:
        return self._if_sheet_exists

    def __fspath__(self) -> str:
        return getattr(self._handles.handle, 'name', '')

    def _get_sheet_name(self, sheet_name: str = None) -> str:
        if sheet_name is None:
            sheet_name = self._cur_sheet
        if sheet_name is None:
            raise ValueError('Must pass explicit sheet_name or set _cur_sheet property')
        return sheet_name

    def _value_with_fmt(self, val: Any) -> tuple[Any, str | None]:
        fmt = None
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
            fmt = '0'
        else:
            val = str(val)
            if len(val) > 32767:
                warnings.warn(f'Cell contents too long ({len(val)}), truncated to 32767 characters', UserWarning, stacklevel=find_stack_level())
        return (val, fmt)

    @classmethod
    def check_extension(cls, ext: str) -> None:
        if ext.startswith('.'):
            ext = ext[1:]
        if not any((ext in extension for extension in cls._supported_extensions)):
            raise ValueError(f"Invalid extension for engine '{cls.engine}': '{ext}'")
        return None

    def __enter__(self) -> ExcelWriter[_WorkbookT]:
        return self

    def __exit__(self, exc_type: type[BaseException], exc_value: BaseException, traceback: TracebackType) -> None:
        self.close()

    def close(self) -> None:
        self._save()
        self._handles.close()

XLS_SIGNATURES = (b'\t\x00\x04\x00\x07\x00\x10\x00', b'\t\x02\x06\x00\x00\x00\x10\x00', b'\t\x04\x06\x00\x00\x00\x10\x00', b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1')
ZIP_SIGNATURE = b'PK\x03\x04'
PEEK_SIZE = max(map(len, XLS_SIGNATURES + (ZIP_SIGNATURE,)))

@doc(storage_options=_shared_docs['storage_options'])
def inspect_excel_format(content_or_path: FilePath | ReadBuffer, storage_options: StorageOptions = None) -> str | None:
    with get_handle(content_or_path, 'rb', storage_options=storage_options, is_text=False) as handle:
        stream = handle.handle
        stream.seek(0)
        buf = stream.read(PEEK_SIZE)
        if buf is None:
            raise ValueError('stream is empty')
        assert isinstance(buf, bytes)
        peek = buf
        stream.seek(0)
        if any((peek.startswith(sig) for sig in XLS_SIGNATURES)):
            return 'xls'
        elif not peek.startswith(ZIP_SIGNATURE):
            return None
        with zipfile.ZipFile(stream) as zf:
            component_names = {name.replace('\\', '/').lower() for name in zf.namelist()}
        if 'xl/workbook.xml' in component_names:
            return 'xlsx'
        if 'xl/workbook.bin' in component_names:
            return 'xlsb'
        if 'content.xml' in component_names:
            return 'ods'
        return 'zip'

@doc(storage_options=_shared_docs['storage_options'])
class ExcelFile:
    def __init__(self, path_or_buffer: FilePath | ReadBuffer, engine: str = None, storage_options: StorageOptions = None, engine_kwargs: dict[str, Any] = None) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}
        if engine is not None and engine not in self._engines:
            raise ValueError(f'Unknown engine: {engine}')
        self._io = stringify_path(path_or_buffer)
        if engine is None:
            ext = None
            if not isinstance(path_or_buffer, (str, os.PathLike, ExcelFile)) and (not is_file_like(path_or_buffer)):
                if import_optional_dependency('xlrd', errors='ignore') is None:
                    xlrd_version = None
                else:
                    import xlrd
                    xlrd_version = Version(get_version(xlrd))
                if xlrd_version is not None and isinstance(path_or_buffer, xlrd.Book):
                    ext = 'xls'
            if ext is None:
                ext = inspect_excel_format(content_or_path=path_or_buffer, storage_options=storage_options)
                if ext is None:
                    raise ValueError('Excel file format cannot be determined, you must specify an engine manually.')
            engine = config.get_option(f'io.excel.{ext}.reader')
            if engine == 'auto':
                engine = get_default_engine(ext, mode='reader')
        assert engine is not None
        self.engine = engine
        self.storage_options = storage_options
        self._reader = self._engines[engine](self._io, storage_options=storage_options, engine_kwargs=engine_kwargs)

    def __fspath__(self) -> str:
        return self._io

    def parse(self, sheet_name: Union[int, str, list[int | str], None] = 0, header: Union[int, list[int], None] = 0, names: list[str] = None, index_col: Union[int, str, list[int | str], None] = None, usecols: Union[str, list[int | str], Callable[[str], bool], None] = None, converters: dict[Hashable, Callable[[str], Any], None] = None, true_values: list[str, None] = None, false_values: list[str, None] = None, skiprows: Union[int, list[int], Callable[[int], bool], None] = None, nrows: int = None, na_values: Union[str, list[str], dict[Hashable, str], None] = None, parse_dates: Union[bool, list[int | str], list[list[int | str]], dict[str, list[int | str]], None] = False, date_format: str = None, thousands: str = None, comment: str = None, skipfooter: int = 0, dtype_backend: DtypeBackendT = lib.no_default, **kwds: Any) -> DataFrame | dict[str, DataFrame]:
        return self._reader.parse(sheet_name=sheet_name, header=header, names=names, index_col=index_col, usecols=usecols, converters=converters, true_values=true_values, false_values=false_values, skiprows=skiprows, nrows=nrows, na_values=na_values, parse_dates=parse_dates, date_format=date_format, thousands=thousands, comment=comment, skipfooter=skipfooter, dtype_backend=dtype_backend, **kwds)

    @property
    def book(self) -> _WorkbookT:
        return self._reader.book

    @property
    def sheet_names(self) -> list[str]:
        return self._reader.sheet_names

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> ExcelFile:
        return self

    def __exit__(self, exc_type: type[BaseException], exc_value: BaseException, traceback: TracebackType) -> None:
        self.close()
