from __future__ import annotations
from abc import ABC, abstractmethod
from collections import abc
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    overload,
    Tuple,
    TypeVar,
)
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
    from types import TracebackType

FrameSeriesStrT = TypeVar('FrameSeriesStrT', bound=Literal['frame', 'series'])


@overload
def to_json(
    path_or_buf: Optional[Any],
    obj: DataFrame | Series,
    orient: Optional[str] = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Optional[Any] = ...,
    lines: bool = ...,
    compression: Any = ...,
    index: Optional[bool] = ...,
    indent: int = ...,
    storage_options: Optional[Any] = ...,
    mode: str = ...,
) -> None:
    ...


@overload
def to_json(
    path_or_buf: Optional[Any],
    obj: DataFrame | Series,
    orient: Optional[str] = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Optional[Any] = ...,
    lines: bool = ...,
    compression: Any = ...,
    index: Optional[bool] = ...,
    indent: int = ...,
    storage_options: Optional[Any] = ...,
    mode: str = ...,
) -> Optional[str]:
    ...


def to_json(
    path_or_buf: Optional[Any],
    obj: DataFrame | Series,
    orient: Optional[str] = None,
    date_format: str = 'epoch',
    double_precision: int = 10,
    force_ascii: bool = True,
    date_unit: str = 'ms',
    default_handler: Optional[Any] = None,
    lines: bool = False,
    compression: Any = 'infer',
    index: Optional[bool] = None,
    indent: int = 0,
    storage_options: Optional[Any] = None,
    mode: str = 'w',
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
        writer: type[JSONTableWriter] = JSONTableWriter
    elif isinstance(obj, Series):
        writer = SeriesWriter
    elif isinstance(obj, DataFrame):
        writer = FrameWriter
    else:
        raise NotImplementedError("'obj' should be a Series or a DataFrame")
    s: str = writer(
        obj,
        orient=orient,
        date_format=date_format,
        double_precision=double_precision,
        ensure_ascii=force_ascii,
        date_unit=date_unit,
        default_handler=default_handler,
        index=index,
        indent=indent,
    ).write()
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
        obj: DataFrame | Series,
        orient: Optional[str],
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Optional[Any] = None,
        indent: int = 0,
    ) -> None:
        self.obj = obj
        if orient is None:
            orient = self._default_orient  # type: ignore
        self.orient: str = orient
        self.date_format: str = date_format
        self.double_precision: int = double_precision
        self.ensure_ascii: bool = ensure_ascii
        self.date_unit: str = date_unit
        self.default_handler: Optional[Any] = default_handler
        self.index: bool = index
        self.indent: int = indent
        self._format_axes()

    def _format_axes(self) -> None:
        raise AbstractMethodError(self)

    def write(self) -> str:
        iso_dates: bool = self.date_format == 'iso'
        return ujson_dumps(
            self.obj_to_write,
            orient=self.orient,
            double_precision=self.double_precision,
            ensure_ascii=self.ensure_ascii,
            date_unit=self.date_unit,
            iso_dates=iso_dates,
            default_handler=self.default_handler,
            indent=self.indent,
        )

    @property
    @abstractmethod
    def obj_to_write(self) -> Any:
        """Object to write in JSON format."""


class SeriesWriter(Writer):
    _default_orient: Literal['index'] = 'index'

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
    _default_orient: Literal['columns'] = 'columns'

    @property
    def obj_to_write(self) -> Any:
        if not self.index and self.orient == 'split':
            obj_to_write: Any = self.obj.to_dict(orient='split')
            del obj_to_write['index']
        else:
            obj_to_write = self.obj
        return obj_to_write

    def _format_axes(self) -> None:
        """
        Try to format axes if they are datelike.
        """
        if not self.obj.index.is_unique and self.orient in ('index', 'columns'):
            raise ValueError(f"DataFrame index must be unique for orient='{self.orient}'.")
        if not self.obj.columns.is_unique and self.orient in ('index', 'columns', 'records'):
            raise ValueError(f"DataFrame columns must be unique for orient='{self.orient}'.")


class JSONTableWriter(FrameWriter):
    _default_orient: Literal['records'] = 'records'

    def __init__(
        self,
        obj: DataFrame,
        orient: Optional[str],
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Optional[Any] = None,
        indent: int = 0,
    ) -> None:
        """
        Adds a `schema` attribute with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
        super().__init__(
            obj,
            orient,
            date_format,
            double_precision,
            ensure_ascii,
            date_unit,
            index,
            default_handler=default_handler,
            indent=indent,
        )
        if date_format != 'iso':
            msg = f"Trying to write with `orient='table'` and `date_format='{date_format}'`. Table Schema requires dates to be formatted with `date_format='iso'`"
            raise ValueError(msg)
        self.schema: Any = build_table_schema(obj, index=self.index)
        if self.index:
            obj = set_default_names(obj)
        if obj.ndim == 2 and isinstance(obj.columns, MultiIndex):
            raise NotImplementedError("orient='table' is not supported for MultiIndex columns")
        if obj.ndim == 1 and obj.name in set(obj.index.names) or len(obj.columns.intersection(obj.index.names)):
            msg = 'Overlapping names between the index and columns'
            raise ValueError(msg)
        timedeltas = obj.select_dtypes(include=['timedelta']).columns
        copied: bool = False
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
    def obj_to_write(self) -> Any:
        return {'schema': self.schema, 'data': self.obj}


@overload
def read_json(
    path_or_buf: Any,
    *,
    orient: Optional[str] = ...,
    typ: Literal['frame', 'series'] = ...,
    dtype: Optional[Union[bool, Dict[Any, Any]]] = ...,
    convert_axes: Optional[bool] = ...,
    convert_dates: Union[bool, List[str]] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: str = ...,
    lines: bool = ...,
    chunksize: Optional[int],
    compression: Any = ...,
    nrows: Optional[int] = ...,
    storage_options: Optional[Any] = ...,
    dtype_backend: Any = ...,
    engine: Literal['ujson', 'pyarrow'] = ...,
) -> DataFrame | Series:
    ...


@overload
def read_json(
    path_or_buf: Any,
    *,
    orient: Optional[str] = ...,
    typ: Literal['frame', 'series'],
    dtype: Optional[Union[bool, Dict[Any, Any]]] = ...,
    convert_axes: Optional[bool] = ...,
    convert_dates: Union[bool, List[str]] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: str = ...,
    lines: bool = ...,
    chunksize: Optional[int],
    compression: Any = ...,
    nrows: Optional[int] = ...,
    storage_options: Optional[Any] = ...,
    dtype_backend: Any = ...,
    engine: Literal['ujson', 'pyarrow'] = ...,
) -> JsonReader[FrameSeriesStrT]:
    ...


@doc(storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'path_or_buf')
def read_json(
    path_or_buf: Any,
    *,
    orient: Optional[str] = None,
    typ: Literal['frame', 'series'] = 'frame',
    dtype: Optional[Union[bool, Dict[Any, Any]]] = None,
    convert_axes: Optional[bool] = None,
    convert_dates: Union[bool, List[str]] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: Optional[str] = None,
    encoding: Optional[str] = None,
    encoding_errors: str = 'strict',
    lines: bool = False,
    chunksize: Optional[int] = None,
    compression: Any = 'infer',
    nrows: Optional[int] = None,
    storage_options: Optional[Any] = None,
    dtype_backend: Any = lib.no_default,
    engine: Literal['ujson', 'pyarrow'] = 'ujson',
) -> DataFrame | Series | JsonReader[FrameSeriesStrT]:
    if orient == 'table' and dtype:
        raise ValueError("cannot pass both dtype and orient='table'")
    if orient == 'table' and convert_axes:
        raise ValueError("cannot pass both convert_axes and orient='table'")
    check_dtype_backend(dtype_backend)
    if dtype is None and orient != 'table':
        dtype = True
    if convert_axes is None and orient != 'table':
        convert_axes = True
    json_reader: JsonReader[FrameSeriesStrT] = JsonReader(
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
        engine=engine,
    )
    if chunksize:
        return json_reader
    else:
        return json_reader.read()


class JsonReader(abc.Iterator, Generic[FrameSeriesStrT]):
    """
    JsonReader provides an interface for reading in a JSON file.

    If initialized with ``lines=True`` and ``chunksize``, can be iterated over
    ``chunksize`` lines at a time. Otherwise, calling ``read`` reads in the
    whole document.
    """
    def __init__(
        self,
        filepath_or_buffer: Any,
        orient: Optional[str],
        typ: Literal['frame', 'series'],
        dtype: Optional[Union[bool, Dict[Any, Any]]],
        convert_axes: Optional[bool],
        convert_dates: Union[bool, List[str]],
        keep_default_dates: bool,
        precise_float: bool,
        date_unit: Optional[str],
        encoding: Optional[str],
        lines: bool,
        chunksize: Optional[int],
        compression: Any,
        nrows: Optional[int],
        storage_options: Optional[Any] = None,
        encoding_errors: str = 'strict',
        dtype_backend: Any = lib.no_default,
        engine: Literal['ujson', 'pyarrow'] = 'ujson',
    ) -> None:
        self.orient: Optional[str] = orient
        self.typ: Literal['frame', 'series'] = typ
        self.dtype: Optional[Union[bool, Dict[Any, Any]]] = dtype
        self.convert_axes: Optional[bool] = convert_axes
        self.convert_dates: Union[bool, List[str]] = convert_dates
        self.keep_default_dates: bool = keep_default_dates
        self.precise_float: bool = precise_float
        self.date_unit: Optional[str] = date_unit
        self.encoding: Optional[str] = encoding
        self.engine: Literal['ujson', 'pyarrow'] = engine
        self.compression: Any = compression
        self.storage_options: Optional[Any] = storage_options
        self.lines: bool = lines
        self.chunksize: Optional[int] = chunksize
        self.nrows_seen: int = 0
        self.nrows: Optional[int] = nrows
        self.encoding_errors: str = encoding_errors
        self.handles: Optional[IOHandles] = None
        self.dtype_backend: Any = dtype_backend
        if self.engine not in {'pyarrow', 'ujson'}:
            raise ValueError(f'The engine type {self.engine} is currently not supported.')
        if self.chunksize is not None:
            self.chunksize = validate_integer('chunksize', self.chunksize, 1)
            if not self.lines:
                raise ValueError('chunksize can only be passed if lines=True')
            if self.engine == 'pyarrow':
                raise ValueError("currently pyarrow engine doesn't support chunksize parameter")
        if self.nrows is not None:
            self.nrows = validate_integer('nrows', self.nrows, 0)
            if not self.lines:
                raise ValueError('nrows can only be passed if lines=True')
        if self.engine == 'pyarrow':
            if not self.lines:
                raise ValueError('currently pyarrow engine only supports the line-delimited JSON format')
            self.data: Any = filepath_or_buffer
        elif self.engine == 'ujson':
            data = self._get_data_from_filepath(filepath_or_buffer)
            if not (self.chunksize or self.nrows):
                with self:
                    self.data = data.read()
            else:
                self.data = data

    def _get_data_from_filepath(self, filepath_or_buffer: Any) -> Any:
        """
        The function read_json accepts three input types:
            1. filepath (string-like)
            2. file-like object (e.g. open file object, StringIO)
        """
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        try:
            self.handles = get_handle(
                filepath_or_buffer,
                'r',
                encoding=self.encoding,
                compression=self.compression,
                storage_options=self.storage_options,
                errors=self.encoding_errors,
            )
        except OSError as err:
            raise FileNotFoundError(f'File {filepath_or_buffer} does not exist') from err
        return self.handles.handle

    def _combine_lines(self, lines: List[str]) -> str:
        """
        Combines a list of JSON objects into one JSON object.
        """
        return f'[{",".join([line for line in (line.strip() for line in lines) if line])}]'

    @overload
    def read(self) -> DataFrame | Series:
        ...

    @overload
    def read(self) -> DataFrame | Series:
        ...

    @overload
    def read(self) -> DataFrame | Series:
        ...

    def read(self) -> DataFrame | Series:
        """
        Read the whole JSON input into a pandas object.
        """
        with self:
            if self.engine == 'pyarrow':
                pyarrow_json = import_optional_dependency('pyarrow.json')
                pa_table = pyarrow_json.read_json(self.data)
                return arrow_table_to_pandas(pa_table, dtype_backend=self.dtype_backend)
            elif self.engine == 'ujson':
                if self.lines:
                    if self.chunksize:
                        obj = concat(self)
                    elif self.nrows:
                        lines_list = list(islice(self.data, self.nrows))
                        lines_json = self._combine_lines(lines_list)
                        obj = self._get_object_parser(lines_json)
                    else:
                        data_str = ensure_str(self.data)
                        data_lines = data_str.split('\n')
                        obj = self._get_object_parser(self._combine_lines(data_lines))
                else:
                    obj = self._get_object_parser(self.data)
                if self.dtype_backend is not lib.no_default:
                    return obj.convert_dtypes(infer_objects=False, dtype_backend=self.dtype_backend)
                else:
                    return obj

    def _get_object_parser(self, json: str) -> DataFrame | Series:
        """
        Parses a json document into a pandas object.
        """
        kwargs: Dict[str, Any] = {
            'orient': self.orient,
            'dtype': self.dtype,
            'convert_axes': self.convert_axes,
            'convert_dates': self.convert_dates,
            'keep_default_dates': self.keep_default_dates,
            'precise_float': self.precise_float,
            'date_unit': self.date_unit,
            'dtype_backend': self.dtype_backend,
        }
        if self.typ == 'frame':
            from pandas.io.json._table_schema import parse_table_schema  # local import if needed
            return FrameParser(json, **kwargs).parse()
        elif self.typ == 'series':
            if not isinstance(self.dtype, bool):
                kwargs['dtype'] = self.dtype
            return SeriesParser(json, **kwargs).parse()
        else:
            raise ValueError(f"typ={self.typ!r} must be 'frame' or 'series'.")

    def close(self) -> None:
        """
        If we opened a stream earlier, in _get_data_from_filepath, we should
        close it.

        If an open stream or file was passed, we leave it open.
        """
        if self.handles is not None:
            self.handles.close()

    def __iter__(self) -> JsonReader[FrameSeriesStrT]:
        return self

    @overload
    def __next__(self) -> DataFrame | Series:
        ...

    @overload
    def __next__(self) -> DataFrame | Series:
        ...

    @overload
    def __next__(self) -> DataFrame | Series:
        ...

    def __next__(self) -> DataFrame | Series:
        if self.nrows and self.nrows_seen >= self.nrows:
            self.close()
            raise StopIteration
        lines = list(islice(self.data, self.chunksize))
        if not lines:
            self.close()
            raise StopIteration
        try:
            lines_json: str = self._combine_lines(lines)
            obj: DataFrame | Series = self._get_object_parser(lines_json)
            obj.index = range(self.nrows_seen, self.nrows_seen + len(obj))
            self.nrows_seen += len(obj)
        except Exception as ex:
            self.close()
            raise ex
        if self.dtype_backend is not lib.no_default:
            return obj.convert_dtypes(infer_objects=False, dtype_backend=self.dtype_backend)
        else:
            return obj

    def __enter__(self) -> JsonReader[FrameSeriesStrT]:
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> Optional[bool]:
        self.close()
        return None


class Parser:
    _STAMP_UNITS: Tuple[str, ...] = ('s', 'ms', 'us', 'ns')
    _MIN_STAMPS: Dict[str, int] = {'s': 31536000, 'ms': 31536000000, 'us': 31536000000000, 'ns': 31536000000000000}

    def __init__(
        self,
        json: str,
        orient: Optional[str],
        dtype: Optional[Union[bool, Dict[Any, Any]]] = None,
        convert_axes: bool = True,
        convert_dates: bool = True,
        keep_default_dates: bool = False,
        precise_float: bool = False,
        date_unit: Optional[str] = None,
        dtype_backend: Any = lib.no_default,
    ) -> None:
        self.json: str = json
        if orient is None:
            orient = self._default_orient  # type: ignore
        self.orient: str = orient
        self.dtype: Optional[Union[bool, Dict[Any, Any]]] = dtype
        if date_unit is not None:
            date_unit = date_unit.lower()
            if date_unit not in self._STAMP_UNITS:
                raise ValueError(f'date_unit must be one of {self._STAMP_UNITS}')
            self.min_stamp: int = self._MIN_STAMPS[date_unit]
        else:
            self.min_stamp = self._MIN_STAMPS['s']
        self.precise_float: bool = precise_float
        self.convert_axes: bool = convert_axes
        self.convert_dates: bool = convert_dates
        self.date_unit: Optional[str] = date_unit
        self.keep_default_dates: bool = keep_default_dates
        self.dtype_backend: Any = dtype_backend

    def check_keys_split(self, decoded: Dict[Any, Any]) -> None:
        """
        Checks that dict has only the appropriate keys for orient='split'.
        """
        bad_keys = set(decoded.keys()).difference(set(self._split_keys))  # type: ignore
        if bad_keys:
            bad_keys_joined = ', '.join(bad_keys)
            raise ValueError(f'JSON data had unexpected key(s): {bad_keys_joined}')

    def parse(self) -> DataFrame | Series:
        obj: DataFrame | Series = self._parse()
        if self.convert_axes:
            obj = self._convert_axes(obj)
        obj = self._try_convert_types(obj)
        return obj

    def _parse(self) -> Any:
        raise AbstractMethodError(self)

    def _convert_axes(self, obj: DataFrame) -> DataFrame:
        """
        Try to convert axes.
        """
        for axis_name in obj._AXIS_ORDERS:
            ax = obj._get_axis(axis_name)
            ser = Series(ax, dtype=ax.dtype, copy=False)
            new_ser, result = self._try_convert_data(name=axis_name, data=ser, use_dtypes=False, convert_dates=True, is_axis=True)
            if result:
                new_axis: Index = Index(new_ser, dtype=new_ser.dtype, copy=False)
                setattr(obj, axis_name, new_axis)
        return obj

    def _try_convert_types(self, obj: DataFrame | Series) -> DataFrame | Series:
        raise AbstractMethodError(self)

    def _try_convert_data(
        self,
        name: str,
        data: Series,
        use_dtypes: bool = True,
        convert_dates: bool = True,
        is_axis: bool = False,
    ) -> Tuple[Series, bool]:
        """
        Try to parse a Series into a column by inferring dtype.
        """
        org_data: Series = data
        if use_dtypes:
            if not self.dtype:
                if all(notna(data)):
                    return (data, False)
                filled = data.fillna(np.nan)
                return (filled, True)
            elif self.dtype is True:
                pass
            elif not _should_convert_dates(convert_dates, self.keep_default_dates, name):
                dtype_val = self.dtype.get(name) if isinstance(self.dtype, dict) else self.dtype
                if dtype_val is not None:
                    try:
                        return (data.astype(dtype_val), True)
                    except (TypeError, ValueError):
                        return (data, False)
        if convert_dates:
            new_data = self._try_convert_to_date(data)
            if new_data is not data:
                return (new_data, True)
        converted = False
        if self.dtype_backend is not lib.no_default and (not is_axis):
            return (data, True)
        elif is_string_dtype(data.dtype):
            try:
                data = data.astype('float64')
                converted = True
            except (TypeError, ValueError):
                pass
        if data.dtype.kind == 'f' and data.dtype != 'float64':
            try:
                data = data.astype('float64')
                converted = True
            except (TypeError, ValueError):
                pass
        if len(data) and data.dtype in ('float', 'object'):
            try:
                new_data = org_data.astype('int64')
                if (new_data == data).all():
                    data = new_data
                    converted = True
            except (TypeError, ValueError, OverflowError):
                pass
        if data.dtype == 'int' and data.dtype != 'int64':
            try:
                data = data.astype('int64')
                converted = True
            except (TypeError, ValueError):
                pass
        if name == 'index' and len(data):
            if self.orient == 'split':
                return (data, False)
        return (data, converted)

    def _try_convert_to_date(self, data: Series) -> Series:
        """
        Try to parse a ndarray like into a date column.

        Try to coerce object in epoch/iso formats and integer/float in epoch
        formats.
        """
        if not len(data):
            return data
        new_data: Series = data
        if new_data.dtype == 'string':
            new_data = new_data.astype(object)
        if new_data.dtype == 'object':
            try:
                new_data = data.astype('int64')
            except OverflowError:
                return data
            except (TypeError, ValueError):
                pass
        if issubclass(new_data.dtype.type, np.number):
            in_range = isna(new_data._values) | (new_data > self.min_stamp) | (new_data._values == iNaT)
            if not in_range.all():
                return data
        date_units: Tuple[str, ...] = (self.date_unit,) if self.date_unit else self._STAMP_UNITS
        for date_unit in date_units:
            try:
                return to_datetime(new_data, errors='raise', unit=date_unit)
            except (ValueError, OverflowError, TypeError):
                continue
        return data


class SeriesParser(Parser):
    _default_orient: Literal['index'] = 'index'
    _split_keys: Tuple[str, str, str] = ('name', 'index', 'data')

    def _parse(self) -> Series:
        data = ujson_loads(self.json, precise_float=self.precise_float)
        if self.orient == 'split':
            decoded: Dict[str, Any] = {str(k): v for k, v in data.items()}
            self.check_keys_split(decoded)
            return Series(**decoded)
        else:
            return Series(data)

    def _try_convert_types(self, obj: Series) -> Series:
        obj, _ = self._try_convert_data('data', obj, convert_dates=self.convert_dates)
        return obj


class FrameParser(Parser):
    _default_orient: Literal['columns'] = 'columns'
    _split_keys: Tuple[str, str, str] = ('columns', 'index', 'data')

    def _parse(self) -> DataFrame:
        json_str: str = self.json
        orient: Optional[str] = self.orient
        if orient == 'split':
            decoded: Dict[str, Any] = {str(k): v for k, v in ujson_loads(json_str, precise_float=self.precise_float).items()}
            self.check_keys_split(decoded)
            orig_names = [tuple(col) if isinstance(col, list) else col for col in decoded['columns']]
            decoded['columns'] = dedup_names(orig_names, is_potential_multi_index(orig_names, None))
            return DataFrame(dtype=None, **decoded)
        elif orient == 'index':
            return DataFrame.from_dict(ujson_loads(json_str, precise_float=self.precise_float), dtype=None, orient='index')
        elif orient == 'table':
            return parse_table_schema(json_str, precise_float=self.precise_float)
        else:
            return DataFrame(ujson_loads(json_str, precise_float=self.precise_float), dtype=None)

    def _try_convert_types(self, obj: DataFrame) -> DataFrame:
        arrays: List[Any] = []
        for col_label, series in obj.items():
            result, _ = self._try_convert_data(col_label, series, convert_dates=_should_convert_dates(self.convert_dates, keep_default_dates=self.keep_default_dates, col=col_label))
            arrays.append(result.array)
        return DataFrame._from_arrays(arrays, obj.columns, obj.index, verify_integrity=False)


def _should_convert_dates(convert_dates: Union[bool, List[str]], keep_default_dates: bool, col: Any) -> bool:
    """
    Return bool whether a DataFrame column should be cast to datetime.
    """
    if convert_dates is False:
        return False
    elif not isinstance(convert_dates, bool) and col in set(convert_dates):
        return True
    elif not keep_default_dates:
        return False
    elif not isinstance(col, str):
        return False
    col_lower = col.lower()
    if col_lower.endswith(('_at', '_time')) or col_lower in {'modified', 'date', 'datetime'} or col_lower.startswith('timestamp'):
        return True
    return False