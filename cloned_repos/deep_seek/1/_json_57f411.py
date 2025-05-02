from __future__ import annotations
from abc import ABC, abstractmethod
from collections import abc
from itertools import islice
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, final, overload, Optional, Union, Dict, List, Tuple, Set, Iterable, Iterator, Sequence, cast
import numpy as np
from pandas._libs import lib
from pandas._libs.json import ujson_dumps, ujson_loads
from pandas._libs.tslibs import iNaT
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import ensure_str, is_string_dtype
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import DataFrame, Index, MultiIndex, Series, isna, notna, to_datetime
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import IOHandles, dedup_names, get_handle, is_potential_multi_index, stringify_path
from pandas.io.json._normalize import convert_to_line_delimits
from pandas.io.json._table_schema import build_table_schema, parse_table_schema, set_default_names
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping
    from types import TracebackType
    from pandas._typing import (
        CompressionOptions, DtypeArg, DtypeBackend, FilePath, IndexLabel, 
        JSONEngine, JSONSerializable, ReadBuffer, Self, StorageOptions, WriteBuffer
    )
    from pandas.core.generic import NDFrame

FrameSeriesStrT = TypeVar('FrameSeriesStrT', bound=Literal['frame', 'series'])

@overload
def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | None,
    obj: Union[Series, DataFrame],
    orient: Optional[str] = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Optional[Callable[[Any], Any]] = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: Optional[bool] = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: str = ...
) -> None: ...

@overload
def to_json(
    path_or_buf: None,
    obj: Union[Series, DataFrame],
    orient: Optional[str] = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Optional[Callable[[Any], Any]] = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: Optional[bool] = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: str = ...
) -> str: ...

def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | None,
    obj: Union[Series, DataFrame],
    orient: Optional[str] = None,
    date_format: str = 'epoch',
    double_precision: int = 10,
    force_ascii: bool = True,
    date_unit: str = 'ms',
    default_handler: Optional[Callable[[Any], Any]] = None,
    lines: bool = False,
    compression: CompressionOptions = 'infer',
    index: Optional[bool] = None,
    indent: int = 0,
    storage_options: Optional[StorageOptions] = None,
    mode: str = 'w'
) -> Optional[str]:
    if orient in ['records', 'values'] and index is True:
        raise ValueError("'index=True' is only valid when 'orient' is 'split', 'table', 'index', or 'columns'.")
    elif orient in ['index', 'columns'] and index is False:
        raise ValueError("'index=False' is only valid when 'orient' is 'split', 'table', 'records', or 'values'.")
    elif index is None:
        index = True
    if lines and orient != 'records':
        raise ValueError("'lines' keyword only valid when 'orient' is records")
    if mode not in ['a', 'w']:
        msg = f"mode={mode} is not a valid option.Only 'w' and 'a' are currently supported."
        raise ValueError(msg)
    if mode == 'a' and (not lines or orient != 'records'):
        msg = "mode='a' (append) is only supported when lines is True and orient is 'records'"
        raise ValueError(msg)
    if orient == 'table' and isinstance(obj, Series):
        obj = obj.to_frame(name=obj.name or 'values')
    if orient == 'table' and isinstance(obj, DataFrame):
        writer = JSONTableWriter
    elif isinstance(obj, Series):
        writer = SeriesWriter
    elif isinstance(obj, DataFrame):
        writer = FrameWriter
    else:
        raise NotImplementedError("'obj' should be a Series or a DataFrame")
    s = writer(obj, orient=orient, date_format=date_format, double_precision=double_precision, ensure_ascii=force_ascii, date_unit=date_unit, default_handler=default_handler, index=index, indent=indent).write()
    if lines:
        s = convert_to_line_delimits(s)
    if path_or_buf is not None:
        with get_handle(path_or_buf, mode, compression=compression, storage_options=storage_options) as handles:
            handles.handle.write(s)
    else:
        return s
    return None

class Writer(ABC):
    def __init__(
        self,
        obj: Union[Series, DataFrame],
        orient: Optional[str],
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Optional[Callable[[Any], Any]] = None,
        indent: int = 0
    ) -> None:
        self.obj = obj
        if orient is None:
            orient = self._default_orient
        self.orient = orient
        self.date_format = date_format
        self.double_precision = double_precision
        self.ensure_ascii = ensure_ascii
        self.date_unit = date_unit
        self.default_handler = default_handler
        self.index = index
        self.indent = indent
        self._format_axes()

    def _format_axes(self) -> None:
        raise AbstractMethodError(self)

    def write(self) -> str:
        iso_dates = self.date_format == 'iso'
        return ujson_dumps(
            self.obj_to_write,
            orient=self.orient,
            double_precision=self.double_precision,
            ensure_ascii=self.ensure_ascii,
            date_unit=self.date_unit,
            iso_dates=iso_dates,
            default_handler=self.default_handler,
            indent=self.indent
        )

    @property
    @abstractmethod
    def obj_to_write(self) -> Any:
        """Object to write in JSON format."""

class SeriesWriter(Writer):
    _default_orient = 'index'

    @property
    def obj_to_write(self) -> Any:
        if not self.index and self.orient == 'split':
            return {'name': self.obj.name, 'data': self.obj.values}
        else:
            return self.obj

    def _format_axes(self) -> None:
        if not self.obj.index.is_unique and self.orient == 'index':
            raise ValueError(f"Series index must be unique for orient='{self.orient}'")

class FrameWriter(Writer):
    _default_orient = 'columns'

    @property
    def obj_to_write(self) -> Any:
        if not self.index and self.orient == 'split':
            obj_to_write = self.obj.to_dict(orient='split')
            del obj_to_write['index']
        else:
            obj_to_write = self.obj
        return obj_to_write

    def _format_axes(self) -> None:
        if not self.obj.index.is_unique and self.orient in ('index', 'columns'):
            raise ValueError(f"DataFrame index must be unique for orient='{self.orient}'.")
        if not self.obj.columns.is_unique and self.orient in ('index', 'columns', 'records'):
            raise ValueError(f"DataFrame columns must be unique for orient='{self.orient}'.")

class JSONTableWriter(FrameWriter):
    _default_orient = 'records'

    def __init__(
        self,
        obj: DataFrame,
        orient: Optional[str],
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Optional[Callable[[Any], Any]] = None,
        indent: int = 0
    ) -> None:
        super().__init__(
            obj, orient, date_format, double_precision, ensure_ascii,
            date_unit, index, default_handler=default_handler, indent=indent
        )
        if date_format != 'iso':
            msg = f"Trying to write with `orient='table'` and `date_format='{date_format}'`. Table Schema requires dates to be formatted with `date_format='iso'`"
            raise ValueError(msg)
        self.schema = build_table_schema(obj, index=self.index)
        if self.index:
            obj = set_default_names(obj)
        if obj.ndim == 2 and isinstance(obj.columns, MultiIndex):
            raise NotImplementedError("orient='table' is not supported for MultiIndex columns")
        if obj.ndim == 1 and obj.name in set(obj.index.names) or len(obj.columns.intersection(obj.index.names)):
            msg = 'Overlapping names between the index and columns'
            raise ValueError(msg)
        timedeltas = obj.select_dtypes(include=['timedelta']).columns
        copied = False
        if len(timedeltas):
            obj = obj.copy()
            copied = True
            obj[timedeltas] = obj[timedeltas].map(lambda x: x.isoformat())
        if not self.index:
            self.obj = obj.reset_index(drop=True)
        else:
            if isinstance(obj.index.dtype, PeriodDtype):
                if not copied:
                    obj = obj.copy(deep=False)
                obj.index = obj.index.to_timestamp()
            self.obj = obj.reset_index(drop=False)
        self.date_format = 'iso'
        self.orient = 'records'
        self.index = index

    @property
    def obj_to_write(self) -> Dict[str, Any]:
        return {'schema': self.schema, 'data': self.obj}

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str],
    *,
    orient: Optional[str] = ...,
    typ: Literal['frame'] = ...,
    dtype: Optional[DtypeArg] = ...,
    convert_axes: Optional[bool] = ...,
    convert_dates: Union[bool, List[str]] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: str = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: Optional[int] = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend = ...,
    engine: JSONEngine = ...
) -> DataFrame: ...

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str],
    *,
    orient: Optional[str] = ...,
    typ: Literal['series'],
    dtype: Optional[DtypeArg] = ...,
    convert_axes: Optional[bool] = ...,
    convert_dates: Union[bool, List[str]] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: str = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: Optional[int] = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend = ...,
    engine: JSONEngine = ...
) -> Series: ...

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str],
    *,
    orient: Optional[str] = ...,
    typ: Literal['frame'] = ...,
    dtype: Optional[DtypeArg] = ...,
    convert_axes: Optional[bool] = ...,
    convert_dates: Union[bool, List[str]] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: str = ...,
    lines: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: Optional[int] = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend = ...,
    engine: JSONEngine = ...
) -> JsonReader[Literal['frame']]: ...

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str],
    *,
    orient: Optional[str] = ...,
    typ: Literal['series'],
    dtype: Optional[DtypeArg] = ...,
    convert_axes: Optional[bool] = ...,
    convert_dates: Union[bool, List[str]] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: str = ...,
    lines: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: Optional[int] = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend = ...,
    engine: JSONEngine = ...
) -> JsonReader[Literal['series']]: ...

@doc(storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'path_or_buf')
def read_json(
    path_or_buf: FilePath | ReadBuffer[str],
    *,
    orient: Optional[str] = None,
    typ: Literal['frame', 'series'] = 'frame',
    dtype: Optional[DtypeArg] = None,
    convert_axes: Optional[bool] = None,
    convert_dates: Union[bool, List[str]] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: Optional[str] = None,
    encoding: Optional[str] = None,
    encoding_errors: str = 'strict',
    lines: bool = False,
    chunksize: Optional[int] = None,
    compression: CompressionOptions = 'infer',
    nrows: Optional[int] = None,
    storage_options: Optional[StorageOptions] = None,
    dtype_backend: DtypeBackend = lib.no_default,
    engine: JSONEngine = 'ujson'
) -> Union[DataFrame, Series, JsonReader[FrameSeriesStrT]]:
    if orient == 'table' and dtype:
        raise ValueError("cannot pass both dtype and orient='table'")
    if orient == 'table' and convert_axes:
        raise ValueError("cannot pass both convert_axes and orient='table'")
    check_dtype_backend(dtype_backend)
    if dtype is None and orient != 'table':
        dtype = True
    if convert_axes is None and orient != 'table':
        convert_axes = True
    json_reader = JsonReader(
        path_or_buf,
        orient=orient,
        typ=typ,
        dtype=dtype,
        convert_axes=convert_axes,
        convert_dates=convert_dates,
        keep_default_dates=keep_default_dates,
        precise_float=precise_float,
        date_unit=date_unit,
        encoding=encoding,
        lines=lines,
        chunksize=chunksize,
        compression=compression,
        nrows=nrows,
        storage_options=storage_options,
        encoding_errors=encoding_errors,
        dtype_backend=dtype_backend,
        engine=engine
    )
    if chunksize:
        return json_reader
    else:
        return json_reader.read()

class JsonReader(abc.Iterator, Generic[FrameSeriesStrT]):
    def __init__(
        self,
        filepath_or_buffer: FilePath | ReadBuffer[str],
        orient: Optional[str],
        typ: Literal['frame', 'series'],
        dtype: Optional[DtypeArg],
        convert_axes: bool,
        convert_dates: Union[bool, List[str]],
        keep_default_dates: bool,
        precise_float: bool,
        date_unit: Optional[str],
        encoding: Optional[str],
        lines: bool,
        chunksize: Optional[int],
        compression: CompressionOptions,
        nrows: Optional[int],
        storage_options: Optional[StorageOptions] = None,
        encoding_errors: str = 'strict',
        dtype_backend: DtypeBackend = lib.no_default,
        engine: JSONEngine = 'ujson'
    ) -> None:
        self.orient = orient
        self.typ = typ
        self.dtype = dtype
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.keep_default_dates = keep_default_dates
        self.precise_float = precise_float
        self.date_unit = date_unit
        self.encoding = encoding
        self.engine = engine
        self.compression = compression
        self.storage_options = storage_options
        self.lines = lines
        self.chunksize = chunksize
        self.nrows_seen = 0
        self.nrows = nrows
        self.encoding_errors = encoding_errors
        self.handles: Optional[IOHandles] = None
        self.dtype_backend = dtype_backend
        if self.engine not in {'pyarrow', 'ujson'}:
            raise ValueError(f'The engine