from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import datetime
from decimal import Decimal
from functools import partial
import os
from textwrap import fill
from typing import IO, TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union, cast, overload, Optional, Dict, List, Tuple, Set
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

_read_excel_doc: str = """
Read an Excel file into a ``pandas`` ``DataFrame``.
...
"""

@overload
def func_i6b8zsnw(
    io: Union[str, ExcelFile, Any],
    sheet_name: Union[str, int, List[Union[str, int]], ...,
    header: Union[int, List[int]] = ...,
    names: Optional[List[str]] = ...,
    index_col: Optional[Union[int, str, List[int]]] = ...,
    usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = ...,
    dtype: Optional[Dict[str, Any]] = ...,
    engine: Optional[str] = ...,
    converters: Optional[Dict[Union[int, str], Callable[[Any], Any]]] = ...,
    true_values: Optional[List[Any]] = ...,
    false_values: Optional[List[Any]] = ...,
    skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = ...,
    nrows: Optional[int] = ...,
    na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: Union[bool, List[int], List[List[int]], Dict[str, List[int]]] = ...,
    date_format: Optional[Union[str, Dict[str, str]]] = ...,
    thousands: Optional[str] = ...,
    decimal: str = ...,
    comment: Optional[str] = ...,
    skipfooter: int = ...,
    storage_options: Optional[Dict[str, Any]] = ...,
    dtype_backend: Optional[str] = ...,
) -> Union[DataFrame, Dict[str, DataFrame]]: ...

@overload
def func_i6b8zsnw(
    io: Union[str, ExcelFile, Any],
    sheet_name: Union[str, int, List[Union[str, int]]],
    *, header: Union[int, List[int]] = ...,
    names: Optional[List[str]] = ...,
    index_col: Optional[Union[int, str, List[int]]] = ...,
    usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = ...,
    dtype: Optional[Dict[str, Any]] = ...,
    engine: Optional[str] = ...,
    converters: Optional[Dict[Union[int, str], Callable[[Any], Any]]] = ...,
    true_values: Optional[List[Any]] = ...,
    false_values: Optional[List[Any]] = ...,
    skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = ...,
    nrows: Optional[int] = ...,
    na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: Union[bool, List[int], List[List[int]], Dict[str, List[int]]] = ...,
    date_format: Optional[Union[str, Dict[str, str]]] = ...,
    thousands: Optional[str] = ...,
    decimal: str = ...,
    comment: Optional[str] = ...,
    skipfooter: int = ...,
    storage_options: Optional[Dict[str, Any]] = ...,
    dtype_backend: Optional[str] = ...,
) -> Union[DataFrame, Dict[str, DataFrame]]: ...

@doc(storage_options=_shared_docs['storage_options'])
@Appender(_read_excel_doc)
def func_i6b8zsnw(
    io: Union[str, ExcelFile, Any],
    sheet_name: Union[str, int, List[Union[str, int]], None] = 0,
    *, header: Union[int, List[int]] = 0,
    names: Optional[List[str]] = None,
    index_col: Optional[Union[int, str, List[int]]] = None,
    usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
    dtype: Optional[Dict[str, Any]] = None,
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
    parse_dates: Union[bool, List[int], List[List[int]], Dict[str, List[int]]] = False,
    date_format: Optional[Union[str, Dict[str, str]]] = None,
    thousands: Optional[str] = None,
    decimal: str = '.',
    comment: Optional[str] = None,
    skipfooter: int = 0,
    storage_options: Optional[Dict[str, Any]] = None,
    dtype_backend: Optional[str] = lib.no_default,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[DataFrame, Dict[str, DataFrame]]:
    ...

_WorkbookT = TypeVar('_WorkbookT')

class BaseExcelReader(Generic[_WorkbookT]):
    def __init__(
        self,
        filepath_or_buffer: Union[str, ExcelFile, Any],
        storage_options: Optional[Dict[str, Any]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    @property
    def func_5ofthm91(self) -> Any:
        raise NotImplementedError

    def func_8hidufb5(self, filepath_or_buffer: Any, engine_kwargs: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def func_c5u0hzi0(self) -> None:
        ...

    @property
    def func_uffdua8n(self) -> List[str]:
        raise NotImplementedError

    def func_6m9s7wsm(self, name: str) -> Any:
        raise NotImplementedError

    def func_livfsxsu(self, index: int) -> Any:
        raise NotImplementedError

    def func_byv4k48w(self, sheet: Any, rows: Optional[int] = None) -> List[List[Any]]:
        raise NotImplementedError

    def func_m1w8lrf9(self, index: int) -> None:
        ...

    def func_uoftjbr0(self, name: str) -> None:
        ...

    def func_5g3vofz6(self, skiprows: Callable[[int], bool], rows_to_use: int) -> int:
        ...

    def func_e4e8rlwc(
        self,
        header: Optional[Union[int, List[int]]],
        index_col: Optional[Union[int, str, List[int]]],
        skiprows: Optional[Union[int, List[int], Callable[[int], bool]]],
        nrows: Optional[int],
    ) -> Optional[int]:
        ...

    def func_watsf5es(
        self,
        sheet_name: Union[str, int, List[Union[str, int]], None] = 0,
        header: Union[int, List[int]] = 0,
        names: Optional[List[str]] = None,
        index_col: Optional[Union[int, str, List[int]]] = None,
        usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        true_values: Optional[List[Any]] = None,
        false_values: Optional[List[Any]] = None,
        skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
        nrows: Optional[int] = None,
        na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
        verbose: bool = False,
        parse_dates: Union[bool, List[int], List[List[int]], Dict[str, List[int]]] = False,
        date_format: Optional[Union[str, Dict[str, str]]] = None,
        thousands: Optional[str] = None,
        decimal: str = '.',
        comment: Optional[str] = None,
        skipfooter: int = 0,
        dtype_backend: Optional[str] = lib.no_default,
        **kwds: Any,
    ) -> Union[DataFrame, Dict[str, DataFrame]]:
        ...

    def func_10r5ffs7(
        self,
        data: List[List[Any]],
        output: Dict[str, DataFrame],
        asheetname: Optional[str] = None,
        header: Union[int, List[int]] = 0,
        names: Optional[List[str]] = None,
        index_col: Optional[Union[int, str, List[int]]] = None,
        usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
        nrows: Optional[int] = None,
        true_values: Optional[List[Any]] = None,
        false_values: Optional[List[Any]] = None,
        na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
        parse_dates: Union[bool, List[int], List[List[int]], Dict[str, List[int]]] = False,
        date_format: Optional[Union[str, Dict[str, str]]] = None,
        thousands: Optional[str] = None,
        decimal: str = '.',
        comment: Optional[str] = None,
        skipfooter: int = 0,
        dtype_backend: Optional[str] = lib.no_default,
        **kwds: Any,
    ) -> Dict[str, DataFrame]:
        ...

@doc(storage_options=_shared_docs['storage_options'])
class ExcelWriter(Generic[_WorkbookT]):
    def __new__(
        cls,
        path: Union[str, IO[bytes]],
        engine: Optional[str] = None,
        date_format: Optional[str] = None,
        datetime_format: Optional[str] = None,
        mode: str = 'w',
        storage_options: Optional[Dict[str, Any]] = None,
        if_sheet_exists: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...

    def __init__(
        self,
        path: Union[str, IO[bytes]],
        engine: Optional[str] = None,
        date_format: Optional[str] = None,
        datetime_format: Optional[str] = None,
        mode: str = 'w',
        storage_options: Optional[Dict[str, Any]] = None,
        if_sheet_exists: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    @property
    def func_65mvfn5p(self) -> List[str]:
        ...

    @property
    def func_z699qf8f(self) -> str:
        ...

    @property
    def func_bpxpalwh(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def func_869ucxly(self) -> Any:
        raise NotImplementedError

    def func_eol2pkau(
        self,
        cells: Iterable[Tuple[int, int, Any, Optional[str]]],
        sheet_name: Optional[str] = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: Optional[Tuple[int, int]] = None,
    ) -> None:
        raise NotImplementedError

    def func_wrh57rd5(self) -> None:
        raise NotImplementedError

    @property
    def func_5fs57wkg(self) -> str:
        ...

    @property
    def func_4lo7frcp(self) -> str:
        ...

    @property
    def func_vy26sh3r(self) -> str:
        ...

    def __fspath__(self) -> str:
        ...

    def func_wkwo4yvx(self, sheet_name: Optional[str]) -> str:
        ...

    def func_9opqdeyx(self, val: Any) -> Tuple[Any, Optional[str]]:
        ...

    @classmethod
    def func_96vm8cbj(cls, ext: str) -> bool:
        ...

    def __enter__(self) -> ExcelWriter:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        self.close()

    def func_c5u0hzi0(self) -> None:
        ...

XLS_SIGNATURES: Tuple[bytes, ...] = (b'\t\x00\x04\x00\x07\x00\x10\x00', ...)
ZIP_SIGNATURE: bytes = b'PK\x03\x04'
PEEK_SIZE: int = max(map(len, XLS_SIGNATURES + (ZIP_SIGNATURE,)))

@doc(storage_options=_shared_docs['storage_options'])
def func_ftqu2y27(
    content_or_path: Union[str, IO[bytes]],
    storage_options: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    ...

@doc(storage_options=_shared_docs['storage_options'])
class ExcelFile:
    _engines: Dict[str, Any] = {...}

    def __init__(
        self,
        path_or_buffer: Union[str, IO[bytes], Any],
        engine: Optional[str] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    def __fspath__(self) -> str:
        ...

    def func_watsf5es(
        self,
        sheet_name: Union[str, int, List[Union[str, int]], None] = 0,
        header: Union[int, List[int]] = 0,
        names: Optional[List[str]] = None,
        index_col: Optional[Union[int, str, List[int]]] = None,
        usecols: Optional[Union[str, List[int], List[str], Callable[[str], bool]]] = None,
        converters: Optional[Dict[Union[int, str], Callable[[Any], Any]]] = None,
        true_values: Optional[List[Any]] = None,
        false_values: Optional[List[Any]] = None,
        skiprows: Optional[Union[int, List[int], Callable[[int], bool]]] = None,
        nrows: Optional[int] = None,
        na_values: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
        parse_dates: Union[bool, List[int], List[List[int]], Dict[str, List[int]]] = False,
        date_format: Optional[Union[str, Dict[str, str]]] = None,
        thousands: Optional[str] = None,
        comment: Optional[str] = None,
        skipfooter: int = 0,
        dtype_backend: Optional[str] = lib.no_default,
        **kwds: Any,
    ) -> Union[DataFrame, Dict[str, DataFrame]]:
        ...

    @property
    def func_869ucxly(self) -> Any:
        ...

    @property
    def func_uffdua8n(self) -> List[str]:
        ...

    def func_c5u0hzi0(self) -> None:
        ...

    def __enter__(self) -> ExcelFile:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        self.close()
