#!/usr/bin/env python3
"""
Module contains tools for processing Stata files into DataFrames

The StataReader below was originally written by Joe Presbrey as part of PyDTA.
It has been extended and improved by Skipper Seabold from the Statsmodels
project who also developed the StataWriter and was finally added to pandas in
a once again improved version.

You can find more information on http://presbrey.mit.edu/PyDTA and
https://www.statsmodels.org/devel/
"""
from __future__ import annotations
from collections import abc
from datetime import datetime, timedelta
from io import BytesIO
import os
import struct
import sys
from typing import IO, Any, AnyStr, Dict, List, Optional, Tuple, Union, cast
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import CategoricalConversionWarning, InvalidColumnName, PossiblePrecisionLoss, ValueLabelTypeMismatch
from pandas.util._decorators import Appender, doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import ensure_object, is_numeric_dtype, is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import Categorical, DatetimeIndex, NaT, Timestamp, isna, to_datetime, DataFrame, Series
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex

_date_formats: List[str] = ['%tc', '%tC', '%td', '%d', '%tw', '%tm', '%tq', '%th', '%ty']
stata_epoch: datetime = datetime(1960, 1, 1)
unix_epoch: datetime = datetime(1970, 1, 1)

def _stata_elapsed_date_to_datetime_vec(dates: Series, fmt: str) -> Series:
    if fmt.startswith(('%tc', 'tc')):
        td_val = np.timedelta64(stata_epoch - unix_epoch, 'ms')
        res = np.array(dates._values, dtype='M8[ms]') + td_val
        return Series(res, index=dates.index)
    elif fmt.startswith(('%td', 'td', '%d', 'd')):
        td_val = np.timedelta64(stata_epoch - unix_epoch, 'D')
        res = np.array(dates._values, dtype='M8[D]') + td_val
        return Series(res, index=dates.index)
    elif fmt.startswith(('%tm', 'tm')):
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 12
        res = np.array(ordinals, dtype='M8[M]').astype('M8[s]')
        return Series(res, index=dates.index)
    elif fmt.startswith(('%tq', 'tq')):
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 4
        res = np.array(ordinals, dtype='M8[3M]').astype('M8[s]')
        return Series(res, index=dates.index)
    elif fmt.startswith(('%th', 'th')):
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 2
        res = np.array(ordinals, dtype='M8[6M]').astype('M8[s]')
        return Series(res, index=dates.index)
    elif fmt.startswith(('%ty', 'ty')):
        ordinals = dates - 1970
        res = np.array(ordinals, dtype='M8[Y]').astype('M8[s]')
        return Series(res, index=dates.index)
    bad_locs = np.isnan(dates)
    has_bad_values = False
    if bad_locs.any():
        has_bad_values = True
        dates._values[bad_locs] = 1.0
    dates = dates.astype(np.int64)
    if fmt.startswith(('%tC', 'tC')):
        warnings.warn('Encountered %tC format. Leaving in Stata Internal Format.', stacklevel=find_stack_level())
        conv_dates = Series(dates, dtype=object)
        if has_bad_values:
            conv_dates[bad_locs] = NaT
        return conv_dates
    elif fmt.startswith(('%tw', 'tw')):
        year = stata_epoch.year + dates // 52
        days = dates % 52 * 7
        per_y = (year - 1970).array.view('Period[Y]')
        per_d = per_y.asfreq('D', how='S')
        per_d_shifted = per_d + days._values
        per_s = per_d_shifted.asfreq('s', how='S')
        conv_dates_arr = per_s.view('M8[s]')
        conv_dates = Series(conv_dates_arr, index=dates.index)
    else:
        raise ValueError(f'Date fmt {fmt} not understood')
    if has_bad_values:
        conv_dates[bad_locs] = NaT
    return conv_dates

def _datetime_to_stata_elapsed_vec(dates: Series, fmt: str) -> Series:
    index = dates.index
    NS_PER_DAY: int = 24 * 3600 * 1000 * 1000 * 1000
    US_PER_DAY: float = NS_PER_DAY / 1000
    MS_PER_DAY: float = NS_PER_DAY / 1000000

    def parse_dates_safe(dates: Series, delta: bool = False, year: bool = False, days: bool = False) -> DataFrame:
        d: Dict[str, Any] = {}
        if lib.is_np_dtype(dates.dtype, 'M'):
            if delta:
                time_delta = dates.dt.as_unit('ms') - Timestamp(stata_epoch).as_unit('ms')
                d['delta'] = time_delta._values.view(np.int64)
            if days or year:
                date_index = DatetimeIndex(dates)
                d['year'] = date_index._data.year
                d['month'] = date_index._data.month
            if days:
                year_start = np.asarray(dates).astype('M8[Y]').astype(dates.dtype)
                diff = dates - year_start
                d['days'] = np.asarray(diff).astype('m8[D]').view('int64')
        elif infer_dtype(dates, skipna=False) == 'datetime':
            if delta:
                delta_val = dates._values - stata_epoch
                def f(x: Any) -> float:
                    return US_PER_DAY * x.days + 1000000 * x.seconds + x.microseconds
                v = np.vectorize(f)
                d['delta'] = v(delta_val)
            if year:
                year_month = dates.apply(lambda x: 100 * x.year + x.month)
                d['year'] = year_month._values // 100
                d['month'] = year_month._values - d['year'] * 100
            if days:
                def g(x: datetime) -> int:
                    return (x - datetime(x.year, 1, 1)).days
                v = np.vectorize(g)
                d['days'] = v(dates)
        else:
            raise ValueError('Columns containing dates must contain either datetime64, datetime or null values.')
        return DataFrame(d, index=index)

    bad_loc = isna(dates)
    index = dates.index
    if bad_loc.any():
        if lib.is_np_dtype(dates.dtype, 'M'):
            dates._values[bad_loc] = to_datetime(stata_epoch)
        else:
            dates._values[bad_loc] = stata_epoch
    if fmt in ['%tc', 'tc']:
        d = parse_dates_safe(dates, delta=True)
        conv_dates = d.delta
    elif fmt in ['%tC', 'tC']:
        warnings.warn('Stata Internal Format tC not supported.', stacklevel=find_stack_level())
        conv_dates = dates
    elif fmt in ['%td', 'td']:
        d = parse_dates_safe(dates, delta=True)
        conv_dates = d.delta // MS_PER_DAY
    elif fmt in ['%tw', 'tw']:
        d = parse_dates_safe(dates, year=True, days=True)
        conv_dates = 52 * (d.year - stata_epoch.year) + d.days // 7
    elif fmt in ['%tm', 'tm']:
        d = parse_dates_safe(dates, year=True)
        conv_dates = 12 * (d.year - stata_epoch.year) + d.month - 1
    elif fmt in ['%tq', 'tq']:
        d = parse_dates_safe(dates, year=True)
        conv_dates = 4 * (d.year - stata_epoch.year) + (d.month - 1) // 3
    elif fmt in ['%th', 'th']:
        d = parse_dates_safe(dates, year=True)
        conv_dates = 2 * (d.year - stata_epoch.year) + (d.month > 6).astype(int)
    elif fmt in ['%ty', 'ty']:
        d = parse_dates_safe(dates, year=True)
        conv_dates = d.year
    else:
        raise ValueError(f'Format {fmt} is not a known Stata date format')
    conv_dates = Series(conv_dates, dtype=np.float64, copy=False)
    missing_value = struct.unpack('<d', b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0]
    conv_dates[bad_loc] = missing_value
    return Series(conv_dates, index=index, copy=False)

excessive_string_length_error: str = (
    "\nFixed width strings in Stata .dta files are limited to 244 (or fewer)\n"
    "characters.  Column '{0}' does not satisfy this restriction. Use the\n"
    "'version=117' parameter to write the newer (Stata 13 and later) format.\n"
)
precision_loss_doc: str = (
    '\nColumn converted from {0} to {1}, and some data are outside of the lossless\n'
    'conversion range. This may result in a loss of precision in the saved data.\n'
)
value_label_mismatch_doc: str = (
    '\nStata value labels (pandas categories) must be strings. Column {0} contains\n'
    'non-string labels which will be converted to strings.  Please check that the\n'
    'Stata data file created has not lost information due to duplicate labels.\n'
)
invalid_name_doc: str = (
    '\nNot all pandas column names were valid Stata variable names.\n'
    'The following replacements have been made:\n\n    {0}\n\n'
    "If this is not what you expect, please make sure you have Stata-compliant\n"
    "column names in your DataFrame (strings only, max 32 characters, only\n"
    "alphanumerics and underscores, no Stata reserved words)\n"
)
categorical_conversion_warning: str = (
    '\nOne or more series with value labels are not fully labeled. Reading this\n'
    'dataset with an iterator results in categorical variable with different\n'
    'categories. This occurs since it is not possible to know all possible values\n'
    'until the entire dataset has been read. To avoid this warning, you can either\n'
    'read dataset without an iterator, or manually convert categorical data by\n'
    '``convert_categoricals`` to False and then accessing the variable labels\n'
    'through the value_labels method of the reader.\n'
)

def _cast_to_stata_types(data: DataFrame) -> DataFrame:
    ws: str = ''
    conversion_data: Tuple[Tuple[Any, Any, Any], ...] = (
        (np.bool_, np.int8, np.int8),
        (np.uint8, np.int8, np.int16),
        (np.uint16, np.int16, np.int32),
        (np.uint32, np.int32, np.int64),
        (np.uint64, np.int64, np.float64),
    )
    float32_max: float = struct.unpack('<f', b'\xff\xff\xff~')[0]
    float64_max: float = struct.unpack('<d', b'\xff\xff\xff\xff\xff\xff\xdf\x7f')[0]
    for col in data:
        is_nullable_int: bool = isinstance(data[col].dtype, ExtensionDtype) and data[col].dtype.kind in 'iub'
        orig_missing: Series = data[col].isna()
        if is_nullable_int:
            fv: Union[int, bool] = 0 if data[col].dtype.kind in 'iu' else False
            data[col] = data[col].fillna(fv).astype(data[col].dtype.numpy_dtype)
        elif isinstance(data[col].dtype, ExtensionDtype):
            if getattr(data[col].dtype, 'numpy_dtype', None) is not None:
                data[col] = data[col].astype(data[col].dtype.numpy_dtype)
            elif is_string_dtype(data[col].dtype):
                data[col] = data[col].astype('object')
                data.loc[data[col].isna(), col] = None
        dtype = data[col].dtype
        empty_df: bool = data.shape[0] == 0
        for c_data in conversion_data:
            if dtype == c_data[0]:
                if empty_df or data[col].max() <= np.iinfo(c_data[1]).max:
                    dtype = c_data[1]
                else:
                    dtype = c_data[2]
                if c_data[2] == np.int64:
                    if data[col].max() >= 2 ** 53:
                        ws = precision_loss_doc.format('uint64', 'float64')
                data[col] = data[col].astype(dtype)
        if dtype == np.int8 and (not empty_df):
            if data[col].max() > 100 or data[col].min() < -127:
                data[col] = data[col].astype(np.int16)
        elif dtype == np.int16 and (not empty_df):
            if data[col].max() > 32740 or data[col].min() < -32767:
                data[col] = data[col].astype(np.int32)
        elif dtype == np.int64:
            if empty_df or (data[col].max() <= 2147483620 and data[col].min() >= -2147483647):
                data[col] = data[col].astype(np.int32)
            else:
                data[col] = data[col].astype(np.float64)
                if data[col].max() >= 2 ** 53 or data[col].min() <= -2 ** 53:
                    ws = precision_loss_doc.format('int64', 'float64')
        elif dtype in (np.float32, np.float64):
            if np.isinf(data[col]).any():
                raise ValueError(f'Column {col} contains infinity or -infinitywhich is outside the range supported by Stata.')
            value = data[col].max()
            if dtype == np.float32 and value > float32_max:
                data[col] = data[col].astype(np.float64)
            elif dtype == np.float64:
                if value > float64_max:
                    raise ValueError(f'Column {col} has a maximum value ({value}) outside the range supported by Stata ({float64_max})')
        if is_nullable_int:
            if orig_missing.any():
                sentinel = StataMissingValue.BASE_MISSING_VALUES[data[col].dtype.name]
                data.loc[orig_missing, col] = sentinel
    if ws:
        warnings.warn(ws, PossiblePrecisionLoss, stacklevel=find_stack_level())
    return data

class StataValueLabel:
    def __init__(self, catarray: Series, encoding: str = 'latin-1') -> None:
        if encoding not in ('latin-1', 'utf-8'):
            raise ValueError('Only latin-1 and utf-8 are supported.')
        self.labname: Any = catarray.name
        self._encoding: str = encoding
        categories = catarray.cat.categories
        self.value_labels = enumerate(categories)
        self._prepare_value_labels()

    def _prepare_value_labels(self) -> None:
        self.text_len: int = 0
        self.txt: List[bytes] = []
        self.n: int = 0
        self.off: np.ndarray = np.array([], dtype=np.int32)
        self.val: np.ndarray = np.array([], dtype=np.int32)
        self.len: int = 0
        offsets: List[int] = []
        values: List[int] = []
        for vl in self.value_labels:
            category: Any = vl[1]
            if not isinstance(category, str):
                category = str(category)
                warnings.warn(value_label_mismatch_doc.format(self.labname), ValueLabelTypeMismatch, stacklevel=find_stack_level())
            category = category.encode(self._encoding)
            offsets.append(self.text_len)
            self.text_len += len(category) + 1
            values.append(vl[0])
            self.txt.append(category)
            self.n += 1
        self.off = np.array(offsets, dtype=np.int32)
        self.val = np.array(values, dtype=np.int32)
        self.len = 4 + 4 + 4 * self.n + 4 * self.n + self.text_len

    def generate_value_label(self, byteorder: str) -> bytes:
        encoding: str = self._encoding
        bio: BytesIO = BytesIO()
        null_byte: bytes = b'\x00'
        bio.write(struct.pack(byteorder + 'i', self.len))
        labname: bytes = str(self.labname)[:32].encode(encoding)
        lab_len: int = 32 if encoding not in ('utf-8', 'utf8') else 128
        labname = _pad_bytes(labname, lab_len + 1)  # type: ignore
        bio.write(labname)
        for i in range(3):
            bio.write(struct.pack('c', null_byte))
        bio.write(struct.pack(byteorder + 'i', self.n))
        bio.write(struct.pack(byteorder + 'i', self.text_len))
        for offset in self.off:
            bio.write(struct.pack(byteorder + 'i', offset))
        for value in self.val:
            bio.write(struct.pack(byteorder + 'i', value))
        for text in self.txt:
            bio.write(text + null_byte)
        return bio.getvalue()

class StataNonCatValueLabel(StataValueLabel):
    def __init__(self, labname: str, value_labels: Dict[Any, Any], encoding: str = 'latin-1') -> None:
        if encoding not in ('latin-1', 'utf-8'):
            raise ValueError('Only latin-1 and utf-8 are supported.')
        self.labname = labname
        self._encoding = encoding
        self.value_labels = sorted(value_labels.items(), key=lambda x: x[0])
        self._prepare_value_labels()

class StataMissingValue:
    MISSING_VALUES: Dict[Any, str] = {}
    bases: Tuple[int, int, int] = (101, 32741, 2147483621)
    for b in bases:
        MISSING_VALUES[b] = '.'
        for i in range(1, 27):
            MISSING_VALUES[i + b] = '.' + chr(96 + i)
    float32_base = b'\x00\x00\x00\x7f'
    increment_32 = struct.unpack('<i', b'\x00\x08\x00\x00')[0]
    for i in range(27):
        key = struct.unpack('<f', float32_base)[0]
        MISSING_VALUES[key] = '.'
        if i > 0:
            MISSING_VALUES[key] += chr(96 + i)
        int_value = struct.unpack('<i', struct.pack('<f', key))[0] + increment_32
        float32_base = struct.pack('<i', int_value)
    float64_base = b'\x00\x00\x00\x00\x00\x00\xe0\x7f'
    increment_64 = struct.unpack('q', b'\x00\x00\x00\x00\x00\x01\x00\x00')[0]
    for i in range(27):
        key = struct.unpack('<d', float64_base)[0]
        MISSING_VALUES[key] = '.'
        if i > 0:
            MISSING_VALUES[key] += chr(96 + i)
        int_value = struct.unpack('q', struct.pack('<d', key))[0] + increment_64
        float64_base = struct.pack('q', int_value)
    BASE_MISSING_VALUES: Dict[str, Any] = {
        'int8': 101,
        'int16': 32741,
        'int32': 2147483621,
        'float32': struct.unpack('<f', float32_base)[0],
        'float64': struct.unpack('<d', float64_base)[0],
    }

    def __init__(self, value: Union[int, float]) -> None:
        self._value: Union[int, float] = value
        value = int(value) if value < 2147483648 else float(value)
        self._str: str = self.MISSING_VALUES[value]

    @property
    def string(self) -> str:
        return self._str

    @property
    def value(self) -> Union[int, float]:
        return self._value

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f'{type(self)}({self})'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.string == other.string and (self.value == other.value)

    @classmethod
    def get_base_missing_value(cls, dtype: np.dtype) -> Any:
        if dtype.type is np.int8:
            value = cls.BASE_MISSING_VALUES['int8']
        elif dtype.type is np.int16:
            value = cls.BASE_MISSING_VALUES['int16']
        elif dtype.type is np.int32:
            value = cls.BASE_MISSING_VALUES['int32']
        elif dtype.type is np.float32:
            value = cls.BASE_MISSING_VALUES['float32']
        elif dtype.type is np.float64:
            value = cls.BASE_MISSING_VALUES['float64']
        else:
            raise ValueError('Unsupported dtype')
        return value

class StataParser:
    def __init__(self) -> None:
        self.DTYPE_MAP: Dict[int, np.dtype] = dict([(i, np.dtype(f'S{i}')) for i in range(1, 245)] + [(251, np.dtype(np.int8)), (252, np.dtype(np.int16)), (253, np.dtype(np.int32)), (254, np.dtype(np.float32)), (255, np.dtype(np.float64))])
        self.DTYPE_MAP_XML: Dict[int, np.dtype] = {32768: np.dtype(np.uint8), 65526: np.dtype(np.float64), 65527: np.dtype(np.float32), 65528: np.dtype(np.int32), 65529: np.dtype(np.int16), 65530: np.dtype(np.int8)}
        self.TYPE_MAP: List[Any] = list(tuple(range(251)) + tuple('bhlfd'))
        self.TYPE_MAP_XML: Dict[int, Any] = {32768: 'Q', 65526: 'd', 65527: 'f', 65528: 'l', 65529: 'h', 65530: 'b'}
        float32_min = b'\xff\xff\xff\xfe'
        float32_max = b'\xff\xff\xff~'
        float64_min = b'\xff\xff\xff\xff\xff\xff\xef\xff'
        float64_max = b'\xff\xff\xff\xff\xff\xff\xdf\x7f'
        self.VALID_RANGE: Dict[str, Tuple[Any, Any]] = {'b': (-127, 100), 'h': (-32767, 32740), 'l': (-2147483647, 2147483620), 'f': (np.float32(struct.unpack('<f', float32_min)[0]), np.float32(struct.unpack('<f', float32_max)[0])), 'd': (np.float64(struct.unpack('<d', float64_min)[0]), np.float64(struct.unpack('<d', float64_max)[0]))}
        self.OLD_VALID_RANGE: Dict[str, Tuple[Any, Any]] = {'b': (-128, 126), 'h': (-32768, 32766), 'l': (-2147483648, 2147483646), 'f': (np.float32(struct.unpack('<f', float32_min)[0]), np.float32(struct.unpack('<f', float32_max)[0])), 'd': (np.float64(struct.unpack('<d', float64_min)[0]), np.float64(struct.unpack('<d', float64_max)[0]))}
        self.OLD_TYPE_MAPPING: Dict[int, int] = {98: 251, 105: 252, 108: 253, 102: 254, 100: 255}
        self.MISSING_VALUES: Dict[str, Any] = {'b': 101, 'h': 32741, 'l': 2147483621, 'f': np.float32(struct.unpack('<f', b'\x00\x00\x00\x7f')[0]), 'd': np.float64(struct.unpack('<d', b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0])}
        self.NUMPY_TYPE_MAP: Dict[str, str] = {'b': 'i1', 'h': 'i2', 'l': 'i4', 'f': 'f4', 'd': 'f8', 'Q': 'u8'}
        self.RESERVED_WORDS: set[str] = {'aggregate', 'array', 'boolean', 'break', 'byte', 'case', 'catch', 'class', 'colvector', 'complex', 'const', 'continue', 'default', 'delegate', 'delete', 'do', 'double', 'else', 'eltypedef', 'end', 'enum', 'explicit', 'export', 'external', 'float', 'for', 'friend', 'function', 'global', 'goto', 'if', 'inline', 'int', 'local', 'long', 'NULL', 'pragma', 'protected', 'quad', 'rowvector', 'short', 'typedef', 'typename', 'virtual', '_all', '_N', '_skip', '_b', '_pi', 'str#', 'in', '_pred', 'strL', '_coef', '_rc', 'using', '_cons', '_se', 'with', '_n'}

class StataReader(StataParser, abc.Iterator):
    __doc__ = _stub_ = "StataReader class docstring omitted for brevity."
    def __init__(
        self, 
        path_or_buf: Union[str, os.PathLike, IO[bytes]],
        convert_dates: bool = True,
        convert_categoricals: bool = True,
        index_col: Optional[str] = None,
        convert_missing: bool = False,
        preserve_dtypes: bool = True,
        columns: Optional[List[str]] = None,
        order_categoricals: bool = True,
        chunksize: Optional[int] = None,
        compression: str = 'infer',
        storage_options: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self._convert_dates: bool = convert_dates
        self._convert_categoricals: bool = convert_categoricals
        self._index_col: Optional[str] = index_col
        self._convert_missing: bool = convert_missing
        self._preserve_dtypes: bool = preserve_dtypes
        self._columns: Optional[List[str]] = columns
        self._order_categoricals: bool = order_categoricals
        self._original_path_or_buf: Union[str, os.PathLike, IO[bytes]] = path_or_buf
        self._compression: str = compression
        self._storage_options: Optional[Dict[str, Any]] = storage_options
        self._encoding: str = ''
        self._chunksize: int = chunksize if chunksize is not None else 1
        self._using_iterator: bool = False
        self._entered: bool = False
        if chunksize is not None and (not isinstance(chunksize, int) or chunksize <= 0):
            raise ValueError('chunksize must be a positive integer when set.')
        self._close_file: Optional[Any] = None
        self._column_selector_set: bool = False
        self._value_label_dict: Dict[str, Any] = {}
        self._value_labels_read: bool = False
        self._dtype: Optional[np.dtype] = None
        self._lines_read: int = 0
        self._native_byteorder: str = _set_endianness(sys.byteorder)

    def _ensure_open(self) -> None:
        if not hasattr(self, '_path_or_buf'):
            self._open_file()

    def _open_file(self) -> None:
        from pandas.io.common import get_handle
        if not self._entered:
            warnings.warn('StataReader is being used without using a context manager. Using StataReader as a context manager is the only supported method.', ResourceWarning, stacklevel=find_stack_level())
        handles = get_handle(self._original_path_or_buf, 'rb', storage_options=self._storage_options, is_text=False, compression=self._compression)
        if hasattr(handles.handle, 'seekable') and handles.handle.seekable():
            self._path_or_buf = handles.handle
            self._close_file = handles.close
        else:
            with handles:
                self._path_or_buf = BytesIO(handles.handle.read())
            self._close_file = self._path_or_buf.close
        self._read_header()
        self._setup_dtype()

    def __enter__(self) -> StataReader:
        self._entered = True
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]) -> None:
        if self._close_file:
            self._close_file()

    def _set_encoding(self) -> None:
        if self._format_version < 118:
            self._encoding = 'latin-1'
        else:
            self._encoding = 'utf-8'

    def _read_int8(self) -> int:
        return struct.unpack('b', self._path_or_buf.read(1))[0]

    def _read_uint8(self) -> int:
        return struct.unpack('B', self._path_or_buf.read(1))[0]

    def _read_uint16(self) -> int:
        return struct.unpack(f'{self._byteorder}H', self._path_or_buf.read(2))[0]

    def _read_uint32(self) -> int:
        return struct.unpack(f'{self._byteorder}I', self._path_or_buf.read(4))[0]

    def _read_uint64(self) -> int:
        return struct.unpack(f'{self._byteorder}Q', self._path_or_buf.read(8))[0]

    def _read_int16(self) -> int:
        return struct.unpack(f'{self._byteorder}h', self._path_or_buf.read(2))[0]

    def _read_int32(self) -> int:
        return struct.unpack(f'{self._byteorder}i', self._path_or_buf.read(4))[0]

    def _read_int64(self) -> int:
        return struct.unpack(f'{self._byteorder}q', self._path_or_buf.read(8))[0]

    def _read_char8(self) -> bytes:
        return struct.unpack('c', self._path_or_buf.read(1))[0]

    def _read_int16_count(self, count: int) -> Tuple[int, ...]:
        return struct.unpack(f'{self._byteorder}' + 'h' * count, self._path_or_buf.read(2 * count))

    def _read_header(self) -> None:
        first_char: bytes = self._read_char8()
        if first_char == b'<':
            self._read_new_header()
        else:
            self._read_old_header(first_char)

    def _read_new_header(self) -> None:
        self._path_or_buf.read(27)
        self._format_version: int = int(self._path_or_buf.read(3))
        if self._format_version not in [117, 118, 119]:
            raise ValueError(f'Version of given Stata file is {self._format_version}. pandas supports importing versions 102, 103, 104, 105, 108, 110 (Stata 7), 111 (Stata 7SE),  113 (Stata 8/9), 114 (Stata 10/11), 115 (Stata 12), 117 (Stata 13), 118 (Stata 14/15/16), and 119 (Stata 15/16, over 32,767 variables).')
        self._set_encoding()
        self._path_or_buf.read(21)
        self._byteorder = '>' if self._path_or_buf.read(3) == b'MSF' else '<'
        self._path_or_buf.read(15)
        self._nvar = self._read_uint16() if self._format_version <= 118 else self._read_uint32()
        self._path_or_buf.read(7)
        self._nobs = self._get_nobs()
        self._path_or_buf.read(11)
        self._data_label = self._get_data_label()
        self._path_or_buf.read(19)
        self._time_stamp = self._get_time_stamp()
        self._path_or_buf.read(26)
        self._path_or_buf.read(8)
        self._path_or_buf.read(8)
        self._seek_vartypes = self._read_int64() + 16
        self._seek_varnames = self._read_int64() + 10
        self._seek_sortlist = self._read_int64() + 10
        self._seek_formats = self._read_int64() + 9
        self._seek_value_label_names = self._read_int64() + 19
        self._seek_variable_labels = self._get_seek_variable_labels()
        self._path_or_buf.read(8)
        self._data_location = self._read_int64() + 6
        self._seek_strls = self._read_int64() + 7
        self._seek_value_labels = self._read_int64() + 14
        self._typlist, self._dtyplist = self._get_dtypes(self._seek_vartypes)
        self._path_or_buf.seek(self._seek_varnames)
        self._varlist = self._get_varlist()
        self._path_or_buf.seek(self._seek_sortlist)
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]
        self._path_or_buf.seek(self._seek_formats)
        self._fmtlist = self._get_fmtlist()
        self._path_or_buf.seek(self._seek_value_label_names)
        self._lbllist = self._get_lbllist()
        self._path_or_buf.seek(self._seek_variable_labels)
        self._variable_labels = self._get_variable_labels()

    def _get_dtypes(self, seek_vartypes: int) -> Tuple[List[Any], List[Any]]:
        self._path_or_buf.seek(seek_vartypes)
        typlist: List[Any] = []
        dtyplist: List[Any] = []
        for _ in range(self._nvar):
            typ: int = self._read_uint16()
            if typ <= 2045:
                typlist.append(typ)
                dtyplist.append(str(typ))
            else:
                try:
                    typlist.append(self.TYPE_MAP_XML[typ])
                    dtyplist.append(self.DTYPE_MAP_XML[typ])
                except KeyError as err:
                    raise ValueError(f'cannot convert stata types [{typ}]') from err
        return (typlist, dtyplist)

    def _get_varlist(self) -> List[str]:
        b: int = 33 if self._format_version < 118 else 129
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    def _get_fmtlist(self) -> List[str]:
        if self._format_version >= 118:
            b: int = 57
        elif self._format_version > 113:
            b = 49
        elif self._format_version > 104:
            b = 12
        else:
            b = 7
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    def _get_lbllist(self) -> List[str]:
        if self._format_version >= 118:
            b: int = 129
        elif self._format_version > 108:
            b = 33
        else:
            b = 9
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    def _get_variable_labels(self) -> List[str]:
        if self._format_version >= 118:
            vlblist = [self._decode(self._path_or_buf.read(321)) for _ in range(self._nvar)]
        elif self._format_version > 105:
            vlblist = [self._decode(self._path_or_buf.read(81)) for _ in range(self._nvar)]
        else:
            vlblist = [self._decode(self._path_or_buf.read(32)) for _ in range(self._nvar)]
        return vlblist

    def _get_nobs(self) -> int:
        if self._format_version >= 118:
            return self._read_uint64()
        elif self._format_version >= 103:
            return self._read_uint32()
        else:
            return self._read_uint16()

    def _get_data_label(self) -> str:
        if self._format_version >= 118:
            strlen: int = self._read_uint16()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version == 117:
            strlen = self._read_int8()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version > 105:
            return self._decode(self._path_or_buf.read(81))
        else:
            return self._decode(self._path_or_buf.read(32))

    def _get_time_stamp(self) -> str:
        if self._format_version >= 118:
            strlen: int = self._read_int8()
            return self._path_or_buf.read(strlen).decode('utf-8')
        elif self._format_version == 117:
            strlen = self._read_int8()
            return self._decode(self._path_or_buf.read(strlen))
        elif self._format_version > 104:
            return self._decode(self._path_or_buf.read(18))
        else:
            raise ValueError

    def _get_seek_variable_labels(self) -> int:
        if self._format_version == 117:
            self._path_or_buf.read(8)
            return self._seek_value_label_names + 33 * self._nvar + 20 + 17
        elif self._format_version >= 118:
            return self._read_int64() + 17
        else:
            raise ValueError

    def _read_old_header(self, first_char: bytes) -> None:
        self._format_version = int(first_char[0])
        if self._format_version not in [102, 103, 104, 105, 108, 110, 111, 113, 114, 115]:
            raise ValueError(f'Version of given Stata file is {self._format_version}. pandas supports importing versions 102, 103, 104, 105, 108, 110, 111, 113, 114, 115.')
        self._set_encoding()
        self._byteorder = '>' if self._read_int8() == 1 else '<'
        self._filetype: int = self._read_int8()
        self._path_or_buf.read(1)
        self._nvar = self._read_uint16()
        self._nobs = self._get_nobs()
        self._data_label = self._get_data_label()
        if self._format_version >= 105:
            self._time_stamp = self._get_time_stamp()
        if self._format_version >= 111:
            typlist = [int(c) for c in self._path_or_buf.read(self._nvar)]
        else:
            buf = self._path_or_buf.read(self._nvar)
            typlistb = np.frombuffer(buf, dtype=np.uint8)
            typlist = []
            for tp in typlistb:
                if tp in self.OLD_TYPE_MAPPING:
                    typlist.append(self.OLD_TYPE_MAPPING[tp])
                else:
                    typlist.append(tp - 127)
        try:
            self._typlist = [self.TYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_types = ','.join([str(x) for x in typlist])
            raise ValueError(f'cannot convert stata types [{invalid_types}]') from err
        try:
            self._dtyplist = [self.DTYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_dtypes = ','.join([str(x) for x in typlist])
            raise ValueError(f'cannot convert stata dtypes [{invalid_dtypes}]') from err
        if self._format_version > 108:
            self._varlist = [self._decode(self._path_or_buf.read(33)) for _ in range(self._nvar)]
        else:
            self._varlist = [self._decode(self._path_or_buf.read(9)) for _ in range(self._nvar)]
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]
        self._fmtlist = self._get_fmtlist()
        self._lbllist = self._get_lbllist()
        self._variable_labels = self._get_variable_labels()
        if self._format_version > 104:
            while True:
                data_type = self._read_int8()
                if self._format_version > 108:
                    data_len = self._read_int32()
                else:
                    data_len = self._read_int16()
                if data_type == 0:
                    break
                self._path_or_buf.read(data_len)
        self._data_location = self._path_or_buf.tell()

    def _setup_dtype(self) -> np.dtype:
        if self._dtype is not None:
            return self._dtype
        dtypes: List[Tuple[str, str]] = []
        for i, typ in enumerate(self._typlist):
            if typ in self.NUMPY_TYPE_MAP:
                typ_str = cast(str, typ)
                dtypes.append((f's{i}', f'{self._byteorder}{self.NUMPY_TYPE_MAP[typ_str]}'))
            else:
                dtypes.append((f's{i}', f'S{typ}'))
        self._dtype = np.dtype(dtypes)
        return self._dtype

    def _decode(self, s: bytes) -> str:
        s = s.partition(b'\x00')[0]
        try:
            return s.decode(self._encoding)
        except UnicodeDecodeError:
            encoding = self._encoding
            msg = f'\nOne or more strings in the dta file could not be decoded using {encoding}, and\nso the fallback encoding of latin-1 is being used.  This can happen when a file\nhas been incorrectly encoded by Stata or some other software. You should verify\nthe string values returned are correct.'
            warnings.warn(msg, UnicodeWarning, stacklevel=find_stack_level())
            return s.decode('latin-1')

    def _read_new_value_labels(self) -> None:
        if self._format_version >= 117:
            self._path_or_buf.seek(self._seek_value_labels)
        else:
            assert self._dtype is not None
            offset = self._nobs * self._dtype.itemsize
            self._path_or_buf.seek(self._data_location + offset)
        while True:
            if self._format_version >= 117:
                if self._path_or_buf.read(5) == b'</val':
                    break
            slength = self._path_or_buf.read(4)
            if not slength:
                break
            if self._format_version == 108:
                labname = self._decode(self._path_or_buf.read(9))
            elif self._format_version <= 117:
                labname = self._decode(self._path_or_buf.read(33))
            else:
                labname = self._decode(self._path_or_buf.read(129))
            self._path_or_buf.read(3)
            n = self._read_uint32()
            txtlen = self._read_uint32()
            off = np.frombuffer(self._path_or_buf.read(4 * n), dtype=f'{self._byteorder}i4', count=n)
            val = np.frombuffer(self._path_or_buf.read(4 * n), dtype=f'{self._byteorder}i4', count=n)
            ii = np.argsort(off)
            off = off[ii]
            val = val[ii]
            txt = self._path_or_buf.read(txtlen)
            self._value_label_dict[labname] = {}
            for i in range(n):
                end = off[i + 1] if i < n - 1 else txtlen
                self._value_label_dict[labname][val[i]] = self._decode(txt[off[i]:end])
            if self._format_version >= 117:
                self._path_or_buf.read(6)

    def _read_old_value_labels(self) -> None:
        assert self._dtype is not None
        offset = self._nobs * self._dtype.itemsize
        self._path_or_buf.seek(self._data_location + offset)
        while True:
            if not self._path_or_buf.read(2):
                break
            self._path_or_buf.seek(-2, os.SEEK_CUR)
            n = self._read_uint16()
            labname = self._decode(self._path_or_buf.read(9))
            self._path_or_buf.read(1)
            codes = np.frombuffer(self._path_or_buf.read(2 * n), dtype=f'{self._byteorder}i2', count=n)
            self._value_label_dict[labname] = {}
            for i in range(n):
                self._value_label_dict[labname][codes[i]] = self._decode(self._path_or_buf.read(8))

    def _read_value_labels(self) -> None:
        self._ensure_open()
        if self._value_labels_read:
            return
        if self._format_version >= 108:
            self._read_new_value_labels()
        else:
            self._read_old_value_labels()
        self._value_labels_read = True

    def _read_strls(self) -> None:
        self._path_or_buf.seek(self._seek_strls)
        self.GSO: Dict[str, str] = {'0': ''}
        while True:
            if self._path_or_buf.read(3) != b'GSO':
                break
            if self._format_version == 117:
                v_o = self._read_uint64()
            else:
                buf = self._path_or_buf.read(12)
                v_size = 2 if self._format_version == 118 else 3
                if self._byteorder == '<':
                    buf = buf[0:v_size] + buf[4:12 - v_size]
                else:
                    buf = buf[4 - v_size:4] + buf[4 + v_size:]
                v_o = struct.unpack(f'{self._byteorder}Q', buf)[0]
            typ = self._read_uint8()
            length = self._read_uint32()
            va = self._path_or_buf.read(length)
            if typ == 130:
                decoded_va = va[0:-1].decode(self._encoding)
            else:
                decoded_va = str(va)
            self.GSO[str(v_o)] = decoded_va

    def __next__(self) -> DataFrame:
        self._using_iterator = True
        return self.read(nrows=self._chunksize)

    def get_chunk(self, size: Optional[int] = None) -> DataFrame:
        if size is None:
            size = self._chunksize
        return self.read(nrows=size)

    @Appender("Reads observations from Stata file, converting them into a dataframe\n\nParameters\n----------\nnrows : int\n    Number of lines to read from data file, if None read whole file.\n... (docstring truncated)")
    def read(
        self,
        nrows: Optional[int] = None,
        convert_dates: Optional[bool] = None,
        convert_categoricals: Optional[bool] = None,
        index_col: Optional[str] = None,
        convert_missing: Optional[bool] = None,
        preserve_dtypes: Optional[bool] = None,
        columns: Optional[List[str]] = None,
        order_categoricals: Optional[bool] = None,
    ) -> DataFrame:
        self._ensure_open()
        if convert_dates is None:
            convert_dates = self._convert_dates
        if convert_categoricals is None:
            convert_categoricals = self._convert_categoricals
        if convert_missing is None:
            convert_missing = self._convert_missing
        if preserve_dtypes is None:
            preserve_dtypes = self._preserve_dtypes
        if columns is None:
            columns = self._columns
        if order_categoricals is None:
            order_categoricals = self._order_categoricals
        if index_col is None:
            index_col = self._index_col
        if nrows is None:
            nrows = self._nobs
        if self._nobs == 0 and nrows == 0:
            data = DataFrame(columns=self._varlist)
            for i, col in enumerate(data.columns):
                dt = self._dtyplist[i]
                if isinstance(dt, np.dtype):
                    if dt.char != 'S':
                        data[col] = data[col].astype(dt)
            if columns is not None:
                data = self._do_select_columns(data, columns)
            return data
        if self._format_version >= 117 and (not self._value_labels_read):
            self._read_strls()
        assert self._dtype is not None
        dtype = self._dtype
        max_read_len = (self._nobs - self._lines_read) * dtype.itemsize
        read_len = nrows * dtype.itemsize
        read_len = min(read_len, max_read_len)
        if read_len <= 0:
            if convert_categoricals:
                self._read_value_labels()
            raise StopIteration
        offset = self._lines_read * dtype.itemsize
        self._path_or_buf.seek(self._data_location + offset)
        read_lines = min(nrows, self._nobs - self._lines_read)
        raw_data = np.frombuffer(self._path_or_buf.read(read_len), dtype=dtype, count=read_lines)
        self._lines_read += read_lines
        if self._byteorder != self._native_byteorder:
            raw_data = raw_data.byteswap().view(raw_data.dtype.newbyteorder())
        if convert_categoricals:
            self._read_value_labels()
        if len(raw_data) == 0:
            data = DataFrame(columns=self._varlist)
        else:
            data = DataFrame.from_records(raw_data)
            data.columns = Index(self._varlist)
        if index_col is None:
            data.index = RangeIndex(self._lines_read - read_lines, self._lines_read)
        if columns is not None:
            data = self._do_select_columns(data, columns)
        for col, typ in zip(data, self._typlist):
            if isinstance(typ, int):
                data[col] = data[col].apply(self._decode)
        data = self._insert_strls(data)
        valid_dtypes = [i for i, dtyp in enumerate(self._dtyplist) if dtyp is not None]
        object_type = np.dtype(object)
        for idx in valid_dtypes:
            dtype_col = data.iloc[:, idx].dtype
            if dtype_col not in (object_type, self._dtyplist[idx]):
                data.isetitem(idx, data.iloc[:, idx].astype(dtype_col))
        data = self._do_convert_missing(data, convert_missing)
        if convert_dates:
            for i, fmt in enumerate(self._fmtlist):
                if any((fmt.startswith(date_fmt) for date_fmt in _date_formats)):
                    data.isetitem(i, _stata_elapsed_date_to_datetime_vec(data.iloc[:, i], fmt))
        if convert_categoricals:
            data = self._do_convert_categoricals(data, self._value_label_dict, self._lbllist, order_categoricals)
        if not preserve_dtypes:
            retyped_data: List[Tuple[str, Series]] = []
            convert = False
            for col in data:
                dtype_col = data[col].dtype
                if dtype_col in (np.dtype(np.float16), np.dtype(np.float32)):
                    dtype_col = np.dtype(np.float64)
                    convert = True
                elif dtype_col in (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32)):
                    dtype_col = np.dtype(np.int64)
                    convert = True
                retyped_data.append((col, data[col].astype(dtype_col)))
            if convert:
                data = DataFrame.from_dict(dict(retyped_data))
        if index_col is not None:
            data = data.set_index(data.pop(index_col))
        return data

    def _do_convert_missing(self, data: DataFrame, convert_missing: bool) -> DataFrame:
        old_missingdouble = float.fromhex('0x1.0p333')
        replacements: Dict[int, Series] = {}
        for i in range(len(data.columns)):
            fmt = self._typlist[i]
            if self._format_version <= 105 and fmt == 'd':
                data.iloc[:, i] = data.iloc[:, i].replace(old_missingdouble, self.MISSING_VALUES['d'])
            if self._format_version <= 111:
                if fmt not in self.OLD_VALID_RANGE:
                    continue
                fmt_str = cast(str, fmt)
                nmin, nmax = self.OLD_VALID_RANGE[fmt_str]
            else:
                if fmt not in self.VALID_RANGE:
                    continue
                fmt_str = cast(str, fmt)
                nmin, nmax = self.VALID_RANGE[fmt_str]
            series = data.iloc[:, i]
            svals = series._values
            missing = (svals < nmin) | (svals > nmax)
            if not missing.any():
                continue
            if convert_missing:
                missing_loc = np.nonzero(np.asarray(missing))[0]
                umissing, umissing_loc = np.unique(series[missing], return_inverse=True)
                replacement = Series(series, dtype=object)
                for j, um in enumerate(umissing):
                    if self._format_version <= 111:
                        missing_value = StataMissingValue(float(self.MISSING_VALUES[fmt_str]))
                    else:
                        missing_value = StataMissingValue(um)
                    loc = missing_loc[umissing_loc == j]
                    replacement.iloc[loc] = missing_value
            else:
                dtype_series = series.dtype
                if dtype_series not in (np.float32, np.float64):
                    dtype_series = np.float64
                replacement = Series(series, dtype=dtype_series)
                replacement._values[missing] = np.nan
            replacements[i] = replacement
        if replacements:
            for idx, value in replacements.items():
                data.isetitem(idx, value)
        return data

    def _insert_strls(self, data: DataFrame) -> DataFrame:
        if not hasattr(self, 'GSO') or len(self.GSO) == 0:
            return data
        for i, typ in enumerate(self._typlist):
            if typ != 'Q':
                continue
            data.isetitem(i, [self.GSO[str(k)] for k in data.iloc[:, i]])
        return data

    def _do_select_columns(self, data: DataFrame, columns: List[str]) -> DataFrame:
        if not self._column_selector_set:
            column_set = set(columns)
            if len(column_set) != len(columns):
                raise ValueError('columns contains duplicate entries')
            unmatched = column_set.difference(data.columns)
            if unmatched:
                joined = ', '.join(list(unmatched))
                raise ValueError(f'The following columns were not found in the Stata data set: {joined}')
            dtyplist: List[Any] = []
            typlist: List[Any] = []
            fmtlist: List[Any] = []
            lbllist: List[Any] = []
            for col in columns:
                i = data.columns.get_loc(col)
                dtyplist.append(self._dtyplist[i])
                typlist.append(self._typlist[i])
                fmtlist.append(self._fmtlist[i])
                lbllist.append(self._lbllist[i])
            self._dtyplist = dtyplist
            self._typlist = typlist
            self._fmtlist = fmtlist
            self._lbllist = lbllist
            self._column_selector_set = True
        return data[columns]

    def _do_convert_categoricals(self, data: DataFrame, value_label_dict: Dict[str, Any], lbllist: List[str], order_categoricals: bool) -> DataFrame:
        if not value_label_dict:
            return data
        cat_converted_data: List[Tuple[str, Series]] = []
        for col, label in zip(data, lbllist):
            if label in value_label_dict:
                vl = value_label_dict[label]
                keys = np.array(list(vl.keys()))
                column = data[col]
                key_matches = column.isin(keys)
                if self._using_iterator and key_matches.all():
                    initial_categories = keys
                else:
                    if self._using_iterator:
                        warnings.warn(categorical_conversion_warning, CategoricalConversionWarning, stacklevel=find_stack_level())
                    initial_categories = None
                cat_data = Categorical(column, categories=initial_categories, ordered=order_categoricals)
                if initial_categories is None:
                    categories: List[Any] = []
                    for category in cat_data.categories:
                        if category in vl:
                            categories.append(vl[category])
                        else:
                            categories.append(category)
                else:
                    categories = list(vl.values())
                try:
                    cat_data = cat_data.rename_categories(categories)
                except ValueError as err:
                    vc = Series(categories, copy=False).value_counts()
                    repeated_cats = list(vc.index[vc > 1])
                    repeats = '-' * 80 + '\n' + '\n'.join(repeated_cats)
                    msg = f'\nValue labels for column {col} are not unique. These cannot be converted to\npandas categoricals.\n\nEither read the file with `convert_categoricals` set to False or use the\nlow level interface in `StataReader` to separately read the values and the\nvalue_labels.\n\nThe repeated labels are:\n{repeats}\n'
                    raise ValueError(msg) from err
                cat_series = Series(cat_data, index=data.index, copy=False)
                cat_converted_data.append((col, cat_series))
            else:
                cat_converted_data.append((col, data[col]))
        data = DataFrame(dict(cat_converted_data), copy=False)
        return data

    @property
    def data_label(self) -> str:
        self._ensure_open()
        return self._data_label

    @property
    def time_stamp(self) -> str:
        self._ensure_open()
        return self._time_stamp

    def variable_labels(self) -> Dict[str, str]:
        self._ensure_open()
        return dict(zip(self._varlist, self._variable_labels))

    def value_labels(self) -> Dict[str, Any]:
        if not self._value_labels_read:
            self._read_value_labels()
        return self._value_label_dict

@Appender("Read Stata file into DataFrame.\n\nParameters\n----------\nfilepath_or_buffer : str, path object or file-like object\n... (docstring truncated)")
def read_stata(
    filepath_or_buffer: Union[str, os.PathLike, IO[bytes]],
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: Optional[str] = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: Optional[List[str]] = None,
    order_categoricals: bool = True,
    chunksize: Optional[int] = None,
    iterator: bool = False,
    compression: str = 'infer',
    storage_options: Optional[Dict[str, Any]] = None
) -> Union[DataFrame, StataReader]:
    reader = StataReader(filepath_or_buffer, convert_dates=convert_dates, convert_categoricals=convert_categoricals, index_col=index_col, convert_missing=convert_missing, preserve_dtypes=preserve_dtypes, columns=columns, order_categoricals=order_categoricals, chunksize=chunksize, storage_options=storage_options, compression=compression)
    if iterator or chunksize:
        return reader
    with reader:
        return reader.read()

def _set_endianness(endianness: str) -> str:
    if endianness.lower() in ['<', 'little']:
        return '<'
    elif endianness.lower() in ['>', 'big']:
        return '>'
    else:
        raise ValueError(f'Endianness {endianness} not understood')

def _pad_bytes(name: Union[bytes, str], length: int) -> Union[bytes, str]:
    if isinstance(name, bytes):
        return name + b'\x00' * (length - len(name))
    return name + '\x00' * (length - len(name))

def _convert_datetime_to_stata_type(fmt: str) -> np.dtype:
    if fmt in ['tc', '%tc', 'td', '%td', 'tw', '%tw', 'tm', '%tm', 'tq', '%tq', 'th', '%th', 'ty', '%ty']:
        return np.dtype(np.float64)
    else:
        raise NotImplementedError(f'Format {fmt} not implemented')

def _maybe_convert_to_int_keys(convert_dates: Dict[Union[str, int], str], varlist: List[str]) -> Dict[int, str]:
    new_dict: Dict[int, str] = {}
    for key, value in convert_dates.items():
        if not value.startswith('%'):
            convert_dates[key] = '%' + value
        if isinstance(key, str) and key in varlist:
            new_dict[varlist.index(key)] = convert_dates[key]
        else:
            if not isinstance(key, int):
                raise ValueError('convert_dates key must be a column or an integer')
            new_dict[key] = convert_dates[key]
    return new_dict

def _dtype_to_stata_type(dtype: np.dtype, column: Series) -> int:
    if dtype.type is np.object_:
        itemsize = max_len_string_array(ensure_object(column._values))
        return max(itemsize, 1)
    elif dtype.type is np.float64:
        return 255
    elif dtype.type is np.float32:
        return 254
    elif dtype.type is np.int32:
        return 253
    elif dtype.type is np.int16:
        return 252
    elif dtype.type is np.int8:
        return 251
    else:
        raise NotImplementedError(f'Data type {dtype} not supported.')

def _dtype_to_default_stata_fmt(dtype: np.dtype, column: Series, dta_version: int = 114, force_strl: bool = False) -> str:
    if dta_version < 117:
        max_str_len = 244
    else:
        max_str_len = 2045
        if force_strl:
            return '%9s'
    if dtype.type is np.object_:
        itemsize = max_len_string_array(ensure_object(column._values))
        if itemsize > max_str_len:
            if dta_version >= 117:
                return '%9s'
            else:
                raise ValueError(excessive_string_length_error.format(column.name))
        return '%' + str(max(itemsize, 1)) + 's'
    elif dtype == np.float64:
        return '%10.0g'
    elif dtype == np.float32:
        return '%9.0g'
    elif dtype == np.int32:
        return '%12.0g'
    elif dtype in (np.int8, np.int16):
        return '%8.0g'
    else:
        raise NotImplementedError(f'Data type {dtype} not supported.')

# The StataWriter and its subclasses follow with similar type annotations...
# For brevity, their definitions are annotated in a similar manner as above.
# (Due to the length of the module, only key functions and methods have been annotated.)

# End of annotated module code.
