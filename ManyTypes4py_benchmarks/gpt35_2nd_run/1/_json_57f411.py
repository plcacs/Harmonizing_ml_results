from __future__ import annotations
from abc import ABC, abstractmethod
from collections import abc
from itertools import islice
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, final, overload
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
    from pandas._typing import CompressionOptions, DtypeArg, DtypeBackend, FilePath, IndexLabel, JSONEngine, JSONSerializable, ReadBuffer, Self, StorageOptions, WriteBuffer
    from pandas.core.generic import NDFrame
FrameSeriesStrT = TypeVar('FrameSeriesStrT', bound=Literal['frame', 'series'])

@overload
def to_json(path_or_buf, obj, orient=..., date_format=..., double_precision=..., force_ascii=..., date_unit=..., default_handler=..., lines=..., compression=..., index=..., indent=..., storage_options=..., mode=...):
    ...

@overload
def to_json(path_or_buf, obj, orient=..., date_format=..., double_precision=..., force_ascii=..., date_unit=..., default_handler=..., lines=..., compression=..., index=..., indent=..., storage_options=..., mode=...):
    ...

def to_json(path_or_buf, obj, orient=None, date_format='epoch', double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=None, indent=0, storage_options=None, mode='w'):
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

    def __init__(self, obj, orient, date_format, double_precision, ensure_ascii, date_unit, index, default_handler=None, indent=0):
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

    def _format_axes(self):
        raise AbstractMethodError(self)

    def write(self):
        iso_dates = self.date_format == 'iso'
        return ujson_dumps(self.obj_to_write, orient=self.orient, double_precision=self.double_precision, ensure_ascii=self.ensure_ascii, date_unit=self.date_unit, iso_dates=iso_dates, default_handler=self.default_handler, indent=self.indent)

    @property
    @abstractmethod
    def obj_to_write(self):
        """Object to write in JSON format."""

class SeriesWriter(Writer):
    _default_orient = 'index'

    @property
    def obj_to_write(self):
        if not self.index and self.orient == 'split':
            return {'name': self.obj.name, 'data': self.obj.values}
        else:
            return self.obj

    def _format_axes(self):
        if not self.obj.index.is_unique and self.orient == 'index':
            raise ValueError(f"Series index must be unique for orient='{self.orient}'")

class FrameWriter(Writer):
    _default_orient = 'columns'

    @property
    def obj_to_write(self):
        if not self.index and self.orient == 'split':
            obj_to_write = self.obj.to_dict(orient='split')
            del obj_to_write['index']
        else:
            obj_to_write = self.obj
        return obj_to_write

    def _format_axes(self):
        """
        Try to format axes if they are datelike.
        """
        if not self.obj.index.is_unique and self.orient in ('index', 'columns'):
            raise ValueError(f"DataFrame index must be unique for orient='{self.orient}'.")
        if not self.obj.columns.is_unique and self.orient in ('index', 'columns', 'records'):
            raise ValueError(f"DataFrame columns must be unique for orient='{self.orient}'.")

class JSONTableWriter(FrameWriter):
    _default_orient = 'records'

    def __init__(self, obj, orient, date_format, double_precision, ensure_ascii, date_unit, index, default_handler=None, indent=0):
        """
        Adds a `schema` attribute with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
        super().__init__(obj, orient, date_format, double_precision, ensure_ascii, date_unit, index, default_handler=default_handler, indent=indent)
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
    def obj_to_write(self):
        return {'schema': self.schema, 'data': self.obj}

@overload
def read_json(path_or_buf, *, orient=..., typ=..., dtype=..., convert_axes=..., convert_dates=..., keep_default_dates=..., precise_float=..., date_unit=..., encoding=..., encoding_errors=..., lines=..., chunksize, compression=..., nrows=..., storage_options=..., dtype_backend=..., engine=...):
    ...

@overload
def read_json(path_or_buf, *, orient=..., typ, dtype=..., convert_axes=..., convert_dates=..., keep_default_dates=..., precise_float=..., date_unit=..., encoding=..., encoding_errors=..., lines=..., chunksize, compression=..., nrows=..., storage_options=..., dtype_backend=..., engine=...):
    ...

@overload
def read_json(path_or_buf, *, orient=..., typ, dtype=..., convert_axes=..., convert_dates=..., keep_default_dates=..., precise_float=..., date_unit=..., encoding=..., encoding_errors=..., lines=..., chunksize=..., compression=..., nrows=..., storage_options=..., dtype_backend=..., engine=...):
    ...

@overload
def read_json(path_or_buf, *, orient=..., typ=..., dtype=..., convert_axes=..., convert_dates=..., keep_default_dates=..., precise_float=..., date_unit=..., encoding=..., encoding_errors=..., lines=..., chunksize=..., compression=..., nrows=..., storage_options=..., dtype_backend=..., engine=...):
    ...

@doc(storage_options=_shared_docs['storage_options'], decompression_options=_shared_docs['decompression_options'] % 'path_or_buf')
def read_json(path_or_buf, *, orient=None, typ='frame', dtype=None, convert_axes=None, convert_dates=True, keep_default_dates=True, precise_float=False, date_unit=None, encoding=None, encoding_errors='strict', lines=False, chunksize=None, compression='infer', nrows=None, storage_options=None, dtype_backend=lib.no_default, engine='ujson'):
    """
    Convert a JSON string to pandas object.

    This method reads JSON files or JSON-like data and converts them into pandas
    objects. It supports a variety of input formats, including line-delimited JSON,
    compressed files, and various data representations (table, records, index-based,
    etc.). When `chunksize` is specified, an iterator is returned instead of loading
    the entire data into memory.

    Parameters
    ----------
    path_or_buf : a str path, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.json``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.

        .. deprecated:: 2.1.0
            Passing json literal strings is deprecated.

    orient : str, optional
        Indication of expected JSON string format.
        Compatible JSON strings can be produced by ``to_json()`` with a
        corresponding orient value.
        The set of possible orients is:

        - ``'split'`` : dict like
          ``{{index -> [index], columns -> [columns], data -> [values]}}``
        - ``'records'`` : list like
          ``[{{column -> value}}, ... , {{column -> value}}]``
        - ``'index'`` : dict like ``{{index -> {{column -> value}}}}``
        - ``'columns'`` : dict like ``{{column -> {{index -> value}}}}``
        - ``'values'`` : just the values array
        - ``'table'`` : dict like ``{{'schema': {{schema}}, 'data': {{data}}}}``

        The allowed and default values depend on the value
        of the `typ` parameter.

        * when ``typ == 'series'``,

          - allowed orients are ``{{'split','records','index'}}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.

        * when ``typ == 'frame'``,

          - allowed orients are ``{{'split','records','index',
            'columns','values', 'table'}}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and
            ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``,
            ``'columns'``, and ``'records'``.

    typ : {{'frame', 'series'}}, default 'frame'
        The type of object to recover.

    dtype : bool or dict, default None
        If True, infer dtypes; if a dict of column to dtype, then use those;
        if False, then don't infer dtypes at all, applies only to the data.

        For all ``orient`` values except ``'table'``, default is True.

    convert_axes : bool, default None
        Try to convert the axes to the proper dtypes.

        For all ``orient`` values except ``'table'``, default is True.

    convert_dates : bool or list of str, default True
        If True then default datelike columns may be converted (depending on
        keep_default_dates).
        If False, no dates will be converted.
        If a list of column names, then those columns will be converted and
        default datelike columns may also be converted (depending on
        keep_default_dates).

    keep_default_dates : bool, default True
        If parsing dates (convert_dates is not False), then try to parse the
        default datelike columns.
        A column label is datelike if

        * it ends with ``'_at'``,

        * it ends with ``'_time'``,

        * it begins with ``'timestamp'``,

        * it is ``'modified'``, or

        * it is ``'date'``.

    precise_float : bool, default False
        Set to enable usage of higher precision (strtod) function when
        decoding string to double values. Default (False) is to use fast but
        less precise builtin functionality.

    date_unit : str, default None
        The timestamp unit to detect if converting dates. The default behaviour
        is to try and detect the correct precision, but if this is not desired
        then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
        milliseconds, microseconds or nanoseconds respectively.

    encoding : str, default is 'utf-8'
        The encoding to use to decode py3 bytes.

    encoding_errors : str, optional, default "strict"
        How encoding errors are treated. `List of possible values
        <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

        .. versionadded:: 1.3.0

    lines : bool, default False
        Read the file as a json object per line.

    chunksize : int, optional
        Return JsonReader object for iteration.
        See the `line-delimited json docs
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json>`_
        for more information on ``chunksize``.
        This can only be passed if `lines=True`.
        If this is None, the file will be read into memory all at once.
    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    nrows : int, optional
        The number of lines from the line-delimited jsonfile that has to be read.
        This can only be passed if `lines=True`.
        If this is None, all the rows will be returned.

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

    engine : {{"ujson", "pyarrow"}}, default "ujson"
        Parser engine to use. The ``"pyarrow"`` engine is only available when
        ``lines=True``.

        .. versionadded:: 2.0

    Returns
    -------
    Series, DataFrame, or pandas.api.typing.JsonReader
        A JsonReader is returned when ``chunksize`` is not ``0`` or ``None``.
        Otherwise, the type returned depends on the value of ``typ``.

    See Also
    --------
    DataFrame.to_json : Convert a DataFrame to a JSON string.
    Series.to_json : Convert a Series to a JSON string.
    json_normalize : Normalize semi-structured JSON data into a flat table.

    Notes
    -----
    Specific to ``orient='table'``, if a :class:`DataFrame` with a literal
    :class:`Index` name of `index` gets written with :func:`to_json`, the
    subsequent read operation will incorrectly set the :class:`Index` name to
    ``None``. This is because `index` is also used by :func:`DataFrame.to_json`
    to denote a missing :class:`Index` name, and the subsequent
    :func:`read_json` operation cannot distinguish between the two. The same
    limitation is encountered with a :class:`MultiIndex` and any names
    beginning with ``'level_'``.

    Examples
    --------
    >>> from io import StringIO
    >>> df = pd.DataFrame([['a', 'b'], ['c', 'd']],
    ...                   index=['row 1', 'row 2'],
    ...                   columns=['col 1', 'col 2'])

    Encoding/decoding a Dataframe using ``'split'``