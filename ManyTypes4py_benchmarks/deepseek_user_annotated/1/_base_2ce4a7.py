from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
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
)
import warnings
import zipfile

from pandas._config import config

from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.compat._optional import (
    get_version,
    import_optional_dependency,
)
from pandas.errors import EmptyDataError
from pandas.util._decorators import (
    Appender,
    doc,
)
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

from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)
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

_read_excel_doc = (
    """
Read an Excel file into a ``pandas`` ``DataFrame``.
[Rest of the docstring...]
"""
)

@overload
def read_excel(
    io: str | ExcelFile | Any,
    sheet_name: str | int = ...,
    *,
    header: int | Sequence[int] | None = ...,
    names: SequenceNotStr[Hashable] | range | None = ...,
    index_col: int | str | Sequence[int] | None = ...,
    usecols: int | str | Sequence[int] | Sequence[str] | Callable[[HashableT], bool] | None = ...,
    dtype: DtypeArg | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = ...,
    converters: dict[str, Callable] | dict[int, Callable] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = ...,
    nrows: int | None = ...,
    na_values: Any = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame: ...

@overload
def read_excel(
    io: str | ExcelFile | Any,
    sheet_name: list[IntStrT] | None,
    *,
    header: int | Sequence[int] | None = ...,
    names: SequenceNotStr[Hashable] | range | None = ...,
    index_col: int | str | Sequence[int] | None = ...,
    usecols: int | str | Sequence[int] | Sequence[str] | Callable[[HashableT], bool] | None = ...,
    dtype: DtypeArg | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = ...,
    converters: dict[str, Callable] | dict[int, Callable] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = ...,
    nrows: int | None = ...,
    na_values: Any = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: list | dict | bool = ...,
    date_format: dict[Hashable, str] | str | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> dict[IntStrT, DataFrame]: ...

@doc(storage_options=_shared_docs["storage_options"])
@Appender(_read_excel_doc)
def read_excel(
    io: str | ExcelFile | Any,
    sheet_name: str | int | list[IntStrT] | None = 0,
    *,
    header: int | Sequence[int] | None = 0,
    names: SequenceNotStr[Hashable] | range | None = None,
    index_col: int | str | Sequence[int] | None = None,
    usecols: int | str | Sequence[int] | Sequence[str] | Callable[[HashableT], bool] | None = None,
    dtype: DtypeArg | None = None,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = None,
    converters: dict[str, Callable] | dict[int, Callable] | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
    nrows: int | None = None,
    na_values: Any = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: list | dict | bool = False,
    date_format: dict[Hashable, str] | str | None = None,
    thousands: str | None = None,
    decimal: str = ".",
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    engine_kwargs: dict | None = None,
) -> DataFrame | dict[IntStrT, DataFrame]:
    check_dtype_backend(dtype_backend)
    should_close = False
    if engine_kwargs is None:
        engine_kwargs = {}

    if not isinstance(io, ExcelFile):
        should_close = True
        io = ExcelFile(
            io,
            storage_options=storage_options,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )
    elif engine and engine != io.engine:
        raise ValueError(
            "Engine should not be specified when passing "
            "an ExcelFile - ExcelFile already has the engine set"
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
    book: _WorkbookT

    def __init__(
        self,
        filepath_or_buffer: FilePath | WriteExcelBuffer | ExcelFile,
        storage_options: StorageOptions | None = None,
        engine_kwargs: dict | None = None,
    ) -> None:
        if engine_kwargs is None:
            engine_kwargs = {}

        self.handles = IOHandles(
            handle=filepath_or_buffer, compression={"method": None}
        )
        if not isinstance(filepath_or_buffer, (ExcelFile, self._workbook_class)):
            self.handles = get_handle(
                filepath_or_buffer, "rb", storage_options=storage_options, is_text=False
            )

        if isinstance(self.handles.handle, self._workbook_class):
            self.book = self.handles.handle
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
    def _workbook_class(self) -> type[_WorkbookT]:
        raise NotImplementedError

    def load_workbook(self, filepath_or_buffer: FilePath | WriteExcelBuffer, engine_kwargs: dict) -> _WorkbookT:
        raise NotImplementedError

    def close(self) -> None:
        if hasattr(self, "book"):
            if hasattr(self.book, "close"):
                self.book.close()
            elif hasattr(self.book, "release_resources"):
                self.book.release_resources()
        self.handles.close()

    @property
    def sheet_names(self) -> list[str]:
        raise NotImplementedError

    def get_sheet_by_name(self, name: str) -> Any:
        raise NotImplementedError

    def get_sheet_by_index(self, index: int) -> Any:
        raise NotImplementedError

    def get_sheet_data(self, sheet: Any, rows: int | None = None) -> list[list[Any]]:
        raise NotImplementedError

    def raise_if_bad_sheet_by_index(self, index: int) -> None:
        n_sheets = len(self.sheet_names)
        if index >= n_sheets:
            raise ValueError(
                f"Worksheet index {index} is invalid, {n_sheets} worksheets found"
            )

    def raise_if_bad_sheet_by_name(self, name: str) -> None:
        if name not in self.sheet_names:
            raise ValueError(f"Worksheet named '{name}' not found")

    def _check_skiprows_func(
        self,
        skiprows: Callable,
        rows_to_use: int,
    ) -> int:
        i = 0
        rows_used_so_far = 0
        while rows_used_so_far < rows_to_use:
            if not skiprows(i):
                rows_used_so_far += 1
            i += 1
        return i

    def _calc_rows(
        self,
        header: int | Sequence[int] | None,
        index_col: int | Sequence[int] | None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None,
        nrows: int | None,
    ) -> int | None:
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
            def f(skiprows: Sequence, x: int) -> bool:
                return x in skiprows
            skiprows = cast(Sequence, skiprows)
            return self._check_skiprows_func(partial(f, skiprows), header_rows + nrows)
        if callable(skiprows):
            return self._check_skiprows_func(
                skiprows,
                header_rows + nrows,
            )
        return None

    def parse(
        self,
        sheet_name: str | int | list[int] | list[str] | None = 0,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols: Any = None,
        dtype: DtypeArg | None = None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        na_values: Any = None,
        verbose: bool = False,
        parse_dates: list | dict | bool = False,
        date_format: dict[Hashable, str] | str | None = None,
        thousands: str | None = None,
        decimal: str = ".",
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds: Any,
    ) -> DataFrame | dict[str, DataFrame] | dict[int, DataFrame]:
        validate_header_arg(header)
        validate_integer("nrows", nrows)

        ret_dict = False

        sheets: list[int] | list[str]
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

        output: dict[str, DataFrame] | dict[int, DataFrame] = {}

        last_sheetname = None
        for asheetname in sheets:
            last_sheetname = asheetname
            if verbose:
                print(f"Reading sheet {asheetname}")

            if isinstance(asheetname, str):
                sheet = self.get_sheet_by_name(asheetname)
            else:
                sheet = self.get_sheet_by_index(asheetname)

            file_rows_needed = self._calc_rows(header, index_col, skiprows, nrows)
            data = self.get_sheet_data(sheet, file_rows_needed)
            if hasattr(sheet, "close"):
                sheet.close()
            usecols = maybe_convert_usecols(usecols)

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
                usecols=usecols,
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
            return output[last_sheetname]

    def _parse_sheet(
        self,
        data: list[list[Any]],
        output: dict[str, DataFrame] | dict[int, DataFrame],
        asheetname: str | int | None = None,
        header: int | Sequence[int] | None = 0,
        names: SequenceNotStr[Hashable] | range | None = None,
        index_col: int | Sequence[int] | None = None,
        usecols: Any = None,
        dtype: DtypeArg | None = None,
        skiprows: Sequence[int] | int | Callable[[int], object] | None = None,
        nrows: int | None = None,
        true_values: Iterable[Hashable] | None = None,
        false_values: Iterable[Hashable] | None = None,
        na_values: Any = None,
        parse_dates: list | dict | bool = False,
        date_format: dict[Hashable, str] | str | None = None,
        thousands: str | None = None,
        decimal: str = ".",
        comment: str | None = None,
        skipfooter: int = 0,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        **kwds: Any,
    ) -> dict[str, DataFrame] | dict[int, DataFrame]:
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
        if header is not None and is_list_like(