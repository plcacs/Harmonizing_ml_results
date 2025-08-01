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
from io import BytesIO, IOBase
import os
import struct
import sys
from typing import IO, TYPE_CHECKING, AnyStr, Final, cast, Optional, Dict, List, Tuple, Union, Iterator
import warnings
import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import CategoricalConversionWarning, InvalidColumnName, PossiblePrecisionLoss, ValueLabelTypeMismatch
from pandas.util._decorators import Appender, doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import ensure_object, is_numeric_dtype, is_string_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import Categorical, DatetimeIndex, NaT, Timestamp, isna, to_datetime
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    from types import TracebackType
    from typing import Literal
    from pandas._typing import CompressionOptions, FilePath, ReadBuffer, Self, StorageOptions, WriteBuffer

_version_error: Final[str] = 'Version of given Stata file is {version}. pandas supports importing versions 102, 103, 104, 105, 108, 110 (Stata 7), 111 (Stata 7SE),  113 (Stata 8/9), 114 (Stata 10/11), 115 (Stata 12), 117 (Stata 13), 118 (Stata 14/15/16), and 119 (Stata 15/16, over 32,767 variables).'
_statafile_processing_params1: Final[str] = 'convert_dates : bool, default True\n    Convert date variables to DataFrame time values.\nconvert_categoricals : bool, default True\n    Read value labels and convert columns to Categorical/Factor variables.'
_statafile_processing_params2: Final[str] = 'index_col : str, optional\n    Column to set as index.\nconvert_missing : bool, default False\n    Flag indicating whether to convert missing values to their Stata\n    representations.  If False, missing values are replaced with nan.\n    If True, columns containing missing values are returned with\n    object data types and missing values are represented by\n    StataMissingValue objects.\npreserve_dtypes : bool, default True\n    Preserve Stata datatypes. If False, numeric data are upcast to pandas\n    default types for foreign data (float64 or int64).\ncolumns : list or None\n    Columns to retain.  Columns will be returned in the given order.  None\n    returns all columns.\norder_categoricals : bool, default True\n    Flag indicating whether converted categorical data are ordered.'
_chunksize_params: Final[str] = 'chunksize : int, default None\n    Return StataReader object for iterations, returns chunks with\n    given number of lines.'
_iterator_params: Final[str] = 'iterator : bool, default False\n    Return StataReader object.'
_reader_notes: Final[str] = 'Notes\n-----\nCategorical variables read through an iterator may not have the same\ncategories and dtype. This occurs when  a variable stored in a DTA\nfile is associated to an incomplete set of value labels that only\nlabel a strict subset of the values.'
_read_stata_doc: Final[str] = f"""\nRead Stata file into DataFrame.\n\nParameters\n----------\nfilepath_or_buffer : str, path object or file-like object\n    Any valid string path is acceptable. The string could be a URL. Valid\n    URL schemes include http, ftp, s3, and file. For file URLs, a host is\n    expected. A local file could be: ``file://localhost/path/to/table.dta``.\n\n    If you want to pass in a path object, pandas accepts any ``os.PathLike``.\n\n    By file-like object, we refer to objects with a ``read()`` method,\n    such as a file handle (e.g. via builtin ``open`` function)\n    or ``StringIO``.\n{_statafile_processing_params1}\n{_statafile_processing_params2}\n{_chunksize_params}\n{_iterator_params}\n{_shared_docs['decompression_options'] % 'filepath_or_buffer'}\n{_shared_docs['storage_options']}\n\nReturns\n-------\nDataFrame, pandas.api.typing.StataReader\n    If iterator or chunksize, returns StataReader, else DataFrame.\n\nSee Also\n--------\nio.stata.StataReader : Low-level reader for Stata data files.\nDataFrame.to_stata: Export Stata data files.\n\n{_reader_notes}\n\nExamples\n--------\n\nCreating a dummy stata for this example\n\n>>> df = pd.DataFrame({{'animal': ['falcon', 'parrot', 'falcon', 'parrot'],\n...                   'speed': [350, 18, 361, 15]}})  # doctest: +SKIP\n>>> df.to_stata('animals.dta')  # doctest: +SKIP\n\nRead a Stata dta file:\n\n>>> df = pd.read_stata('animals.dta')  # doctest: +SKIP\n\nRead a Stata dta file in 10,000 line chunks:\n\n>>> values = np.random.randint(0, 10, size=(20_000, 1), dtype="uint8")  # doctest: +SKIP\n>>> df = pd.DataFrame(values, columns=["i"])  # doctest: +SKIP\n>>> df.to_stata('filename.dta')  # doctest: +SKIP\n\n>>> with pd.read_stata('filename.dta', chunksize=10000) as itr:  # doctest: +SKIP\n>>>     for chunk in itr:\n...         # Operate on a single chunk, e.g., chunk.mean()\n...         pass  # doctest: +SKIP\n"""
_read_method_doc: Final[str] = f'Reads observations from Stata file, converting them into a dataframe\n\nParameters\n----------\nnrows : int\n    Number of lines to read from data file, if None read whole file.\n{_statafile_processing_params1}\n{_statafile_processing_params2}\n\nReturns\n-------\nDataFrame\n'
_stata_reader_doc: Final[str] = f'Class for reading Stata dta files.\n\nParameters\n----------\npath_or_buf : path (string), buffer or path object\n    string, pathlib.Path or object\n    implementing a binary read() functions.\n{_statafile_processing_params1}\n{_statafile_processing_params2}\n{_chunksize_params}\n{_shared_docs["decompression_options"]}\n{_shared_docs["storage_options"]}\n\n{_reader_notes}\n'

_date_formats: Final[List[str]] = ['%tc', '%tC', '%td', '%d', '%tw', '%tm', '%tq', '%th', '%ty']
stata_epoch: Final[datetime] = datetime(1960, 1, 1)
unix_epoch: Final[datetime] = datetime(1970, 1, 1)


def _stata_elapsed_date_to_datetime_vec(dates: Series, fmt: str) -> Series:
    """
    Convert from SIF to datetime. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        The Stata Internal Format date to convert to datetime according to fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
        Returns

    Returns
    -------
    converted : Series
        The converted dates

    Examples
    --------
    >>> dates = pd.Series([52])
    >>> _stata_elapsed_date_to_datetime_vec(dates, "%tw")
    0   1961-01-01
    dtype: datetime64[s]

    Notes
    -----
    datetime/c - tc
        milliseconds since 01jan1960 00:00:00.000, assuming 86,400 s/day
    datetime/C - tC - NOT IMPLEMENTED
        milliseconds since 01jan1960 00:00:00.000, adjusted for leap seconds
    date - td
        days since 01jan1960 (01jan1960 = 0)
    weekly date - tw
        weeks since 1960w1
        This assumes 52 weeks in a year, then adds 7 * remainder of the weeks.
        The datetime value is the start of the week in terms of days in the
        year, not ISO calendar weeks.
    monthly date - tm
        months since 1960m1
    quarterly date - tq
        quarters since 1960q1
    half-yearly date - th
        half-years since 1960h1 yearly
    date - ty
        years since 0000
    """
    if fmt.startswith(('%tc', 'tc')):
        td = np.timedelta64(stata_epoch - unix_epoch, 'ms')
        res = np.array(dates._values, dtype='M8[ms]') + td
        return Series(res, index=dates.index)
    elif fmt.startswith(('%td', 'td', '%d', 'd')):
        td = np.timedelta64(stata_epoch - unix_epoch, 'D')
        res = np.array(dates._values, dtype='M8[D]') + td
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


def _datetime_to_stata_elapsed_vec(
    dates: Series, fmt: str
) -> Series:
    """
    Convert from datetime to SIF. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        Series or array containing datetime or datetime64[ns] to
        convert to the Stata Internal Format given by fmt
    fmt : str
        The format to convert to. Can be, tc, td, tw, tm, tq, th, ty
    """
    index = dates.index
    NS_PER_DAY: int = 24 * 3600 * 1000 * 1000 * 1000
    US_PER_DAY: float = NS_PER_DAY / 1000
    MS_PER_DAY: int = NS_PER_DAY // 1000000

    def parse_dates_safe(
        dates: Series,
        delta: bool = False,
        year: bool = False,
        days: bool = False,
    ) -> DataFrame:
        d: Dict[str, Union[np.ndarray, int]] = {}
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
                delta = dates._values - stata_epoch

                def f(x: timedelta) -> int:
                    return US_PER_DAY * x.days + 1000000 * x.seconds + x.microseconds

                v = np.vectorize(f)
                d['delta'] = v(delta)
            if year:
                year_month = dates.apply(lambda x: 100 * x.year + x.month)
                d['year'] = year_month._values // 100
                d['month'] = year_month._values - (year_month._values // 100) * 100
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
    if has_bad_values := any(has_bad_values for has_bad_values in bad_loc):
        conv_dates[bad_loc] = NaT
    return conv_dates.astype(np.float64)


excessive_string_length_error: Final[str] = "\nFixed width strings in Stata .dta files are limited to 244 (or fewer)\ncharacters.  Column '{0}' does not satisfy this restriction. Use the\n'version=117' parameter to write the newer (Stata 13 and later) format.\n"
precision_loss_doc: Final[str] = '\nColumn converted from {0} to {1}, and some data are outside of the lossless\nconversion range. This may result in a loss of precision in the saved data.\n'
value_label_mismatch_doc: Final[str] = '\nStata value labels (pandas categories) must be strings. Column {0} contains\nnon-string labels which will be converted to strings.  Please check that the\nStata data file created has not lost information due to duplicate labels.\n'
invalid_name_doc: Final[str] = '\nNot all pandas column names were valid Stata variable names.\nThe following replacements have been made:\n\n    {0}\n\nIf this is not what you expect, please make sure you have Stata-compliant\ncolumn names in your DataFrame (strings only, max 32 characters, only\nalphanumerics and underscores, no Stata reserved words)\n'
categorical_conversion_warning: Final[str] = '\nOne or more series with value labels are not fully labeled. Reading this\ndataset with an iterator results in categorical variable with different\ncategories. This occurs since it is not possible to know all possible values\nuntil the entire dataset has been read. To avoid this warning, you can either\nread dataset without an iterator, or manually convert categorical data by\n``convert_categoricals`` to False and then accessing the variable labels\nthrough the value_labels method of the reader.\n'


def _cast_to_stata_types(data: DataFrame) -> DataFrame:
    """
    Checks the dtypes of the columns of a pandas DataFrame for
    compatibility with the data types and ranges supported by Stata, and
    converts if necessary.

    Parameters
    ----------
    data : DataFrame
        The DataFrame to check and convert

    Notes
    -----
    Numeric columns in Stata must be one of int8, int16, int32, float32 or
    float64, with some additional value restrictions.  int8 and int16 columns
    are checked for violations of the value restrictions and upcast if needed.
    int64 data is not usable in Stata, and so it is downcast to int32 whenever
    the value are in the int32 range, and sidecast to float64 when larger than
    this range.  If the int64 values are outside of the range of those
    perfectly representable as float64 values, a warning is raised.

    bool columns are cast to int8.  uint columns are converted to int of the
    same size if there is no loss in precision, otherwise are upcast to a
    larger type.  uint64 is currently not supported since it is concerted to
    object in a DataFrame.
    """
    ws: str = ''
    conversion_data: List[Tuple[type, type, type]] = [
        (np.bool_, np.int8, np.int8),
        (np.uint8, np.int8, np.int16),
        (np.uint16, np.int16, np.int32),
        (np.uint32, np.int32, np.int64),
        (np.uint64, np.int64, np.float64),
    ]
    float32_max: float = struct.unpack('<f', b'\xff\xff\xff~')[0]
    float64_max: float = struct.unpack('<d', b'\xff\xff\xff\xff\xff\xff\xdf\x7f')[0]
    for col in data:
        is_nullable_int: bool = isinstance(data[col].dtype, ExtensionDtype) and data[col].dtype.kind in 'iub'
        orig_missing: Series = data[col].isna()
        if is_nullable_int:
            fv: int = 0 if data[col].dtype.kind in 'iu' else False
            data[col] = data[col].fillna(fv).astype(data[col].dtype.numpy_dtype)
        elif isinstance(data[col].dtype, ExtensionDtype):
            if getattr(data[col].dtype, 'numpy_dtype', None) is not None:
                data[col] = data[col].astype(data[col].dtype.numpy_dtype)
            elif is_string_dtype(data[col].dtype):
                data[col] = data[col].astype('object')
                data.loc[data[col].isna(), col] = None
        dtype: np.dtype = data[col].dtype
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
            value: float = data[col].max()
            if dtype == np.float32 and value > float32_max:
                data[col] = data[col].astype(np.float64)
            elif dtype == np.float64:
                if value > float64_max:
                    raise ValueError(f'Column {col} has a maximum value ({value}) outside the range supported by Stata ({float64_max})')
        if is_nullable_int:
            if orig_missing.any():
                sentinel: Union[int, float] = StataMissingValue.BASE_MISSING_VALUES[data[col].dtype.name]
                data.loc[orig_missing, col] = sentinel
    if ws:
        warnings.warn(ws, PossiblePrecisionLoss, stacklevel=find_stack_level())
    return data


class StataValueLabel:
    """
    Parse a categorical column and prepare formatted output

    Parameters
    ----------
    catarray : Series
        Categorical Series to encode
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    """

    def __init__(self, catarray: Series, encoding: Literal['latin-1', 'utf-8'] = 'latin-1') -> None:
        if encoding not in ('latin-1', 'utf-8'):
            raise ValueError('Only latin-1 and utf-8 are supported.')
        self.labname: Hashable = catarray.name
        self._encoding: Literal['latin-1', 'utf-8'] = encoding
        categories: Index = catarray.cat.categories
        self.value_labels: enumerate = enumerate(categories)
        self._prepare_value_labels()

    def _prepare_value_labels(self) -> None:
        """Encode value labels."""
        self.text_len: int = 0
        self.txt: List[bytes] = []
        self.n: int = 0
        self.off: np.ndarray = np.array([], dtype=np.int32)
        self.val: np.ndarray = np.array([], dtype=np.int32)
        self.len: int = 0
        offsets: List[int] = []
        values: List[int] = []
        for vl in self.value_labels:
            category = vl[1]
            if not isinstance(category, str):
                category = str(category)
                warnings.warn(value_label_mismatch_doc.format(self.labname), ValueLabelTypeMismatch, stacklevel=find_stack_level())
            category_encoded: bytes = category.encode(self._encoding)
            offsets.append(self.text_len)
            self.text_len += len(category_encoded) + 1
            values.append(vl[0])
            self.txt.append(category_encoded)
            self.n += 1
        self.off = np.array(offsets, dtype=np.int32)
        self.val = np.array(values, dtype=np.int32)
        self.len = 4 + 4 + 4 * self.n + 4 * self.n + self.text_len

    def generate_value_label(self, byteorder: str) -> bytes:
        """
        Generate the binary representation of the value labels.

        Parameters
        ----------
        byteorder : str
            Byte order of the output

        Returns
        -------
        value_label : bytes
            Bytes containing the formatted value label
        """
        encoding = self._encoding
        bio = BytesIO()
        null_byte = b'\x00'
        bio.write(struct.pack(byteorder + 'i', self.len))
        labname: bytes = str(self.labname)[:32].encode(encoding)
        lab_len: int = 32 if encoding not in ('utf-8', 'utf8') else 128
        labname_padded: bytes = _pad_bytes(labname, lab_len + 1)
        bio.write(labname_padded)
        for _ in range(3):
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
    """
    Prepare formatted version of value labels

    Parameters
    ----------
    labname : str
        Value label name
    value_labels: Dictionary
        Mapping of values to labels
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    """

    def __init__(
        self,
        labname: str,
        value_labels: Dict[Union[int, float], str],
        encoding: Literal['latin-1', 'utf-8'] = 'latin-1'
    ) -> None:
        if encoding not in ('latin-1', 'utf-8'):
            raise ValueError('Only latin-1 and utf-8 are supported.')
        self.labname: str = labname
        self._encoding: Literal['latin-1', 'utf-8'] = encoding
        self.value_labels: List[Tuple[Union[int, float], str]] = sorted(value_labels.items(), key=lambda x: x[0])
        self._prepare_value_labels()


class StataMissingValue:
    """
    An observation's missing value.

    Parameters
    ----------
    value : {int, float}
        The Stata missing value code

    Notes
    -----
    More information: <https://www.stata.com/help.cgi?missing>

    Integer missing values make the code '.', '.a', ..., '.z' to the ranges
    101 ... 127 (for int8), 32741 ... 32767  (for int16) and 2147483621 ...
    2147483647 (for int32).  Missing values for floating point data types are
    more complex but the pattern is simple to discern from the following table.

    np.float32 missing values (float in Stata)
    0000007f    .
    0008007f    .a
    0010007f    .b
    ...
    00c0007f    .x
    00c8007f    .y
    00d0007f    .z

    np.float64 missing values (double in Stata)
    000000000000e07f    .
    000000000001e07f    .a
    000000000002e07f    .b
    ...
    000000000018e07f    .x
    000000000019e07f    .y
    00000000001ae07f    .z
    """
    MISSING_VALUES: Dict[Union[int, float], str] = {}
    bases: Tuple[int, ...] = (101, 32741, 2147483621)
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
    BASE_MISSING_VALUES: Dict[str, Union[int, float]] = {
        'int8': 101,
        'int16': 32741,
        'int32': 2147483621,
        'float32': struct.unpack('<f', float32_base)[0],
        'float64': struct.unpack('<d', float64_base)[0],
    }

    def __init__(self, value: Union[int, float]) -> None:
        self._value: Union[int, float] = value
        value_converted: Union[int, float] = int(value) if isinstance(value, int) and value < 2147483648 else float(value)
        self._str: str = self.MISSING_VALUES[value_converted]

    @property
    def string(self) -> str:
        """
        The Stata representation of the missing value: '.', '.a'..'.z'

        Returns
        -------
        str
            The representation of the missing value.
        """
        return self._str

    @property
    def value(self) -> Union[int, float]:
        """
        The binary representation of the missing value.

        Returns
        -------
        {int, float}
            The binary representation of the missing value.
        """
        return self._value

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f'{type(self)}({self})'

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.string == other.string and (self.value == other.value)

    @classmethod
    def get_base_missing_value(cls, dtype: np.dtype) -> Union[int, float]:
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
        self.DTYPE_MAP: Dict[int, np.dtype] = dict(
            [(i, np.dtype(f'S{i}')) for i in range(1, 245)]
            + [
                (251, np.dtype(np.int8)),
                (252, np.dtype(np.int16)),
                (253, np.dtype(np.int32)),
                (254, np.dtype(np.float32)),
                (255, np.dtype(np.float64)),
            ]
        )
        self.DTYPE_MAP_XML: Dict[int, np.dtype] = {
            32768: np.dtype(np.uint8),
            65526: np.dtype(np.float64),
            65527: np.dtype(np.float32),
            65528: np.dtype(np.int32),
            65529: np.dtype(np.int16),
            65530: np.dtype(np.int8),
        }
        self.TYPE_MAP: List[Union[int, str]] = list(tuple(range(251)) + tuple('bhlfd'))
        self.TYPE_MAP_XML: Dict[int, str] = {
            32768: 'Q',
            65526: 'd',
            65527: 'f',
            65528: 'l',
            65529: 'h',
            65530: 'b',
        }
        float32_min: bytes = b'\xff\xff\xff\xfe'
        float32_max: bytes = b'\xff\xff\xff~'
        float64_min: bytes = b'\xff\xff\xff\xff\xff\xff\xef\xff'
        float64_max: bytes = b'\xff\xff\xff\xff\xff\xff\xdf\x7f'
        self.VALID_RANGE: Dict[str, Tuple[Union[int, float], Union[int, float]]] = {
            'b': (-127, 100),
            'h': (-32767, 32740),
            'l': (-2147483647, 2147483620),
            'f': (
                np.float32(struct.unpack('<f', float32_min)[0]),
                np.float32(struct.unpack('<f', float32_max)[0]),
            ),
            'd': (
                np.float64(struct.unpack('<d', float64_min)[0]),
                np.float64(struct.unpack('<d', float64_max)[0]),
            ),
        }
        self.OLD_VALID_RANGE: Dict[str, Tuple[Union[int, float], Union[int, float]]] = {
            'b': (-128, 126),
            'h': (-32768, 32766),
            'l': (-2147483648, 2147483646),
            'f': (
                np.float32(struct.unpack('<f', float32_min)[0]),
                np.float32(struct.unpack('<f', float32_max)[0]),
            ),
            'd': (
                np.float64(struct.unpack('<d', float64_min)[0]),
                np.float64(struct.unpack('<d', float64_max)[0]),
            ),
        }
        self.OLD_TYPE_MAPPING: Dict[int, int] = {98: 251, 105: 252, 108: 253, 102: 254, 100: 255}
        self.MISSING_VALUES: Dict[str, Union[int, float]] = {
            'b': 101,
            'h': 32741,
            'l': 2147483621,
            'f': struct.unpack('<f', b'\x00\x00\x00\x7f')[0],
            'd': struct.unpack('<d', b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0],
        }
        self.NUMPY_TYPE_MAP: Dict[str, str] = {'b': 'i1', 'h': 'i2', 'l': 'i4', 'f': 'f4', 'd': 'f8', 'Q': 'u8'}
        self.RESERVED_WORDS: set[str] = {
            'aggregate',
            'array',
            'boolean',
            'break',
            'byte',
            'case',
            'catch',
            'class',
            'colvector',
            'complex',
            'const',
            'continue',
            'default',
            'delegate',
            'delete',
            'do',
            'double',
            'else',
            'eltypedef',
            'end',
            'enum',
            'explicit',
            'export',
            'external',
            'float',
            'for',
            'friend',
            'function',
            'global',
            'goto',
            'if',
            'inline',
            'int',
            'local',
            'long',
            'NULL',
            'pragma',
            'protected',
            'quad',
            'rowvector',
            'short',
            'typedef',
            'typename',
            'virtual',
            '_all',
            '_N',
            '_skip',
            '_b',
            '_pi',
            'str#',
            'in',
            '_pred',
            'strL',
            '_coef',
            '_rc',
            'using',
            '_cons',
            '_se',
            'with',
            '_n',
        }


class StataReader(StataParser, abc.Iterator):
    __doc__ = _stata_reader_doc

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
        storage_options: Optional[Dict[str, Any]] = None,
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
        if self._chunksize is not None and not isinstance(self._chunksize, int):
            raise ValueError('chunksize must be a positive integer when set.')
        self._close_file: Optional[Callable[[], None]] = None
        self._column_selector_set: bool = False
        self._value_label_dict: Dict[str, Dict[int, str]] = {}
        self._value_labels_read: bool = False
        self._dtype: Optional[np.dtype] = None
        self._lines_read: int = 0
        self._native_byteorder: str = _set_endianness(sys.byteorder)

    def _ensure_open(self) -> None:
        """
        Ensure the file has been opened and its header data read.
        """
        if not hasattr(self, '_path_or_buf'):
            self._open_file()

    def _open_file(self) -> None:
        """
        Open the file (with compression options, etc.), and read header information.
        """
        if not self._entered:
            warnings.warn(
                'StataReader is being used without using a context manager. Using StataReader as a context manager is the only supported method.',
                ResourceWarning,
                stacklevel=find_stack_level(),
            )
        handles = get_handle(
            self._original_path_or_buf,
            'rb',
            storage_options=self._storage_options,
            is_text=False,
            compression=self._compression,
        )
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
        """enter context manager"""
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self._close_file:
            self._close_file()

    def _set_encoding(self) -> None:
        """
        Set string encoding which depends on file version
        """
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
        return struct.unpack(f'{self._byteorder}{ "h" * count}', self._path_or_buf.read(2 * count))

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
            raise ValueError(_version_error.format(version=self._format_version))
        self._set_encoding()
        self._path_or_buf.read(21)
        self._byteorder: str = '>' if self._path_or_buf.read(3) == b'MSF' else '<'
        self._path_or_buf.read(15)
        self._nvar: int = self._read_uint16() if self._format_version <= 118 else self._read_uint32()
        self._path_or_buf.read(7)
        self._nobs: int = self._get_nobs()
        self._path_or_buf.read(11)
        self._data_label: str = self._get_data_label()
        self._path_or_buf.read(19)
        self._time_stamp: str = self._get_time_stamp()
        self._path_or_buf.read(26)
        self._path_or_buf.read(8)
        self._path_or_buf.read(8)
        self._seek_vartypes: int = self._read_int64() + 16
        self._seek_varnames: int = self._read_int64() + 10
        self._seek_sortlist: int = self._read_int64() + 10
        self._seek_formats: int = self._read_int64() + 9
        self._seek_value_label_names: int = self._read_int64() + 19
        self._seek_variable_labels: int = self._get_seek_variable_labels()
        self._path_or_buf.read(8)
        self._data_location: int = self._read_int64() + 6
        self._seek_strls: int = self._read_int64() + 7
        self._seek_value_labels: int = self._read_int64() + 14
        self._typlist, self._dtyplist = self._get_dtypes(self._seek_vartypes)
        self._path_or_buf.seek(self._seek_varnames)
        self._varlist: List[str] = self._get_varlist()
        self._path_or_buf.seek(self._seek_sortlist)
        self._srtlist: Tuple[int, ...] = self._read_int16_count(self._nvar + 1)[:-1]
        self._path_or_buf.seek(self._seek_formats)
        self._fmtlist: List[str] = self._get_fmtlist()
        self._path_or_buf.seek(self._seek_value_label_names)
        self._lbllist: List[str] = self._get_lbllist()
        self._path_or_buf.seek(self._seek_variable_labels)
        self._variable_labels: List[str] = self._get_variable_labels()

    def _get_dtypes(self, seek_vartypes: int) -> Tuple[List[Union[int, str]], List[np.dtype]]:
        self._path_or_buf.seek(seek_vartypes)
        typlist: List[Union[int, str]] = []
        dtyplist: List[Union[np.dtype, str]] = []
        for _ in range(self._nvar):
            typ: Union[int, str] = self._read_uint16()
            if typ <= 2045:
                typlist.append(typ)
                dtyplist.append(str(typ))
            else:
                try:
                    typlist.append(self.TYPE_MAP_XML[typ])
                    dtyplist.append(self.DTYPE_MAP_XML[typ])
                except KeyError as err:
                    raise ValueError(f'cannot convert stata types [{typ}]') from err
        return typlist, dtyplist

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
            vlblist: List[str] = [self._decode(self._path_or_buf.read(321)) for _ in range(self._nvar)]
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
            strlen: int = self._read_int8()
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
        self._format_version: int = int(first_char[0])
        if self._format_version not in [102, 103, 104, 105, 108, 110, 111, 113, 114, 115]:
            raise ValueError(_version_error.format(version=self._format_version))
        self._set_encoding()
        self._byteorder = '>' if self._read_int8() == 1 else '<'
        self._filetype: int = self._read_int8()
        self._path_or_buf.read(1)
        self._nvar: int = self._read_uint16()
        self._nobs: int = self._get_nobs()
        self._data_label = self._get_data_label()
        if self._format_version >= 105:
            self._time_stamp = self._get_time_stamp()
        if self._format_version >= 111:
            typlist = [int(c) for c in self._path_or_buf.read(self._nvar)]
        else:
            buf: bytes = self._path_or_buf.read(self._nvar)
            typlistb: np.ndarray = np.frombuffer(buf, dtype=np.uint8)
            typlist: List[int] = []
            for tp in typlistb:
                if tp in self.OLD_TYPE_MAPPING:
                    typlist.append(self.OLD_TYPE_MAPPING[tp])
                else:
                    typlist.append(tp - 127)
        try:
            self._typlist: List[Union[int, str]] = [self.TYPE_MAP[typ] for typ in typlist]
        except ValueError as err:
            invalid_types = ','.join([str(x) for x in typlist])
            raise ValueError(f'cannot convert stata types [{invalid_types}]') from err
        try:
            self._dtyplist: List[Union[np.dtype, str]] = [self.DTYPE_MAP[typ] for typ in typlist]
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
                data_type: int = self._read_int8()
                if self._format_version > 108:
                    data_len: int = self._read_int32()
                else:
                    data_len: int = self._read_int16()
                if data_type == 0:
                    break
                self._path_or_buf.read(data_len)
        self._data_location: int = self._path_or_buf.tell()

    def _setup_dtype(self) -> Optional[np.dtype]:
        """Map between numpy and state dtypes"""
        if self._dtype is not None:
            return self._dtype
        dtypes: List[Tuple[str, str]] = []
        for i, typ in enumerate(self._typlist):
            if isinstance(typ, int):
                dtypes.append((f's{i}', f'{self._byteorder}{self.NUMPY_TYPE_MAP.get(typ, "S")}{typ}'))
            else:
                dtypes.append((f's{i}', f'S{typ}'))
        self._dtype: Optional[np.dtype] = np.dtype(dtypes)
        return self._dtype

    def _decode(self, s: bytes) -> str:
        s = s.partition(b'\x00')[0]
        try:
            return s.decode(self._encoding)
        except UnicodeDecodeError:
            encoding: str = self._encoding
            msg: str = (
                f'\nOne or more strings in the dta file could not be decoded using {encoding}, and\n'
                'so the fallback encoding of latin-1 is being used.  This can happen when a file\n'
                'has been incorrectly encoded by Stata or some other software. You should verify\n'
                'the string values returned are correct.'
            )
            warnings.warn(msg, UnicodeWarning, stacklevel=find_stack_level())
            return s.decode('latin-1')

    def _read_new_value_labels(self) -> None:
        """Reads value labels with variable length strings (108 and later format)"""
        if self._format_version >= 117:
            self._path_or_buf.seek(self._seek_value_labels)
        else:
            assert self._dtype is not None
            offset: int = self._nobs * self._dtype.itemsize
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
            n: int = self._read_uint32()
            txtlen: int = self._read_uint32()
            off: np.ndarray = np.frombuffer(self._path_or_buf.read(4 * n), dtype=f'{self._byteorder}i4', count=n)
            val: np.ndarray = np.frombuffer(self._path_or_buf.read(4 * n), dtype=f'{self._byteorder}i4', count=n)
            ii: np.ndarray = np.argsort(off)
            off = off[ii]
            val = val[ii]
            txt: bytes = self._path_or_buf.read(txtlen)
            self._value_label_dict[labname] = {}
            for i in range(n):
                end: int = off[i + 1] if i < n - 1 else txtlen
                self._value_label_dict[labname][val[i]] = self._decode(txt[off[i]:end])
            if self._format_version >= 117:
                self._path_or_buf.read(6)

    def _read_old_value_labels(self) -> None:
        """Reads value labels with fixed-length strings (105 and earlier format)"""
        assert self._dtype is not None
        offset: int = self._nobs * self._dtype.itemsize
        self._path_or_buf.seek(self._data_location + offset)
        while True:
            if not self._path_or_buf.read(2):
                break
            self._path_or_buf.seek(-2, os.SEEK_CUR)
            n: int = self._read_uint16()
            labname: str = self._decode(self._path_or_buf.read(9))
            self._path_or_buf.read(1)
            codes: np.ndarray = np.frombuffer(self._path_or_buf.read(2 * n), dtype=f'{self._byteorder}i2', count=n)
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
                v_o: int = self._read_uint64()
            else:
                buf: bytes = self._path_or_buf.read(12)
                v_size: int = 2 if self._format_version == 118 else 3
                if self._byteorder == '<':
                    buf = buf[0:v_size] + buf[4 : 12 - v_size]
                else:
                    buf = buf[4 - v_size : 4] + buf[4 + v_size :]
                v_o: int = struct.unpack(f'{self._byteorder}Q', buf)[0]
            typ: int = self._read_uint8()
            length: int = self._read_uint32()
            va: bytes = self._path_or_buf.read(length)
            if typ == 130:
                decoded_va: str = va[0:-1].decode(self._encoding)
            else:
                decoded_va = str(va)
            self.GSO[str(v_o)] = decoded_va

    def __next__(self) -> DataFrame:
        self._using_iterator = True
        return self.read(nrows=self._chunksize)

    def get_chunk(self, size: Optional[int] = None) -> DataFrame:
        """
        Reads lines from Stata file and returns as dataframe

        Parameters
        ----------
        size : int, defaults to None
            Number of lines to read.  If None, reads whole file.

        Returns
        -------
        DataFrame
        """
        if size is None:
            size = self._chunksize
        return self.read(nrows=size)

    @Appender(_read_method_doc)
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
    ) -> Union[DataFrame, 'StataReader']:
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
        dtype: np.dtype = self._dtype
        max_read_len: int = (self._nobs - self._lines_read) * dtype.itemsize
        read_len: int = nrows * dtype.itemsize
        read_len = min(read_len, max_read_len)
        if read_len <= 0:
            if convert_categoricals:
                self._read_value_labels()
            raise StopIteration
        offset: int = self._lines_read * dtype.itemsize
        self._path_or_buf.seek(self._data_location + offset)
        read_lines: int = min(nrows, self._nobs - self._lines_read)
        raw_data: np.ndarray = np.frombuffer(self._path_or_buf.read(read_len), dtype=dtype, count=read_lines)
        self._lines_read += read_lines
        if self._byteorder != self._native_byteorder:
            raw_data = raw_data.byteswap().view(dtype.newbyteorder())
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
        valid_dtypes: List[int] = [i for i, dtyp in enumerate(self._dtyplist) if dtyp is not None]
        object_type: np.dtype = np.dtype(object)
        for idx in valid_dtypes:
            dtype_col: np.dtype = data.iloc[:, idx].dtype
            if dtype_col not in (object_type, self._dtyplist[idx]):
                data.isetitem(idx, data.iloc[:, idx].astype(dtype_col))
        data = self._do_convert_missing(data, convert_missing)
        if convert_dates:
            for i, fmt in enumerate(self._fmtlist):
                if any(fmt.startswith(date_fmt) for date_fmt in _date_formats):
                    data.isetitem(i, _stata_elapsed_date_to_datetime_vec(data.iloc[:, i], fmt))
        if convert_categoricals:
            data = self._do_convert_categoricals(data, self._value_label_dict, self._lbllist, order_categoricals)
        if not preserve_dtypes:
            retyped_data: List[Tuple[str, Series]] = []
            convert: bool = False
            for col in data:
                dtype_col: np.dtype = data[col].dtype
                if dtype_col in (np.dtype(np.float16), np.dtype(np.float32)):
                    dtype_new: np.dtype = np.dtype(np.float64)
                    convert = True
                elif dtype_col in (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32)):
                    dtype_new = np.dtype(np.int64)
                    convert = True
                else:
                    dtype_new = dtype_col
                retyped_data.append((col, data[col].astype(dtype_new)))
            if convert:
                data = DataFrame.from_dict(dict(retyped_data))
        if index_col is not None:
            data = data.set_index(data.pop(index_col))
        return data

    def _do_convert_missing(self, data: DataFrame, convert_missing: bool) -> DataFrame:
        old_missingdouble: float = float.fromhex('0x1.0p333')
        replacements: Dict[int, Series] = {}
        for i in range(len(data.columns)):
            fmt: Union[int, str] = self._typlist[i]
            if self._format_version <= 105 and fmt == 'd':
                data.iloc[:, i] = data.iloc[:, i].replace(old_missingdouble, self.MISSING_VALUES['d'])
            if self._format_version <= 111:
                if fmt not in self.OLD_VALID_RANGE:
                    continue
                fmt = cast(str, fmt)
                nmin, nmax = self.OLD_VALID_RANGE[fmt]
            else:
                if fmt not in self.VALID_RANGE:
                    continue
                fmt = cast(str, fmt)
                nmin, nmax = self.VALID_RANGE[fmt]
            series: Series = data.iloc[:, i]
            svals = series._values
            missing: np.ndarray = (svals < nmin) | (svals > nmax)
            if not missing.any():
                continue
            if convert_missing:
                missing_loc: np.ndarray = np.nonzero(np.asarray(missing))[0]
                umissing, umissing_loc = np.unique(series[missing], return_inverse=True)
                replacement: Series = Series(series, dtype=object)
                for j, um in enumerate(umissing):
                    if self._format_version <= 111:
                        missing_value = StataMissingValue(float(self.MISSING_VALUES[fmt]))
                    else:
                        missing_value = StataMissingValue(um)
                    loc = missing_loc[umissing_loc == j]
                    replacement.iloc[loc] = missing_value
            else:
                dtype = series.dtype
                if dtype not in (np.float32, np.float64):
                    dtype = np.float64
                replacement = Series(series, dtype=dtype)
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
            column_set: set[str] = set(columns)
            if len(column_set) != len(columns):
                raise ValueError('columns contains duplicate entries')
            unmatched: set[str] = column_set.difference(data.columns)
            if unmatched:
                joined: str = ', '.join(list(unmatched))
                raise ValueError(f'The following columns were not found in the Stata data set: {joined}')
            dtyplist: List[Union[np.dtype, str]] = []
            typlist: List[Union[int, str]] = []
            fmtlist: List[str] = []
            lbllist: List[str] = []
            for col in columns:
                i: int = data.columns.get_loc(col)
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

    def _do_convert_categoricals(
        self,
        data: DataFrame,
        value_label_dict: Dict[str, Dict[int, str]],
        lbllist: List[str],
        order_categoricals: bool,
    ) -> DataFrame:
        """
        Converts categorical columns to Categorical type.
        """
        if not value_label_dict:
            return data
        cat_converted_data: List[Tuple[str, Series]] = []
        for col, label in zip(data, lbllist):
            if label in value_label_dict:
                vl: Dict[int, str] = value_label_dict[label]
                keys = np.array(list(vl.keys()))
                column = data[col]
                key_matches = column.isin(keys)
                if self._using_iterator and key_matches.all():
                    initial_categories: Optional[Union[List[int], None]] = keys.tolist()
                else:
                    if self._using_iterator:
                        warnings.warn(categorical_conversion_warning, CategoricalConversionWarning, stacklevel=find_stack_level())
                    initial_categories = None
                cat_data = Categorical(column, categories=initial_categories, ordered=order_categoricals)
                if initial_categories is None:
                    categories: List[str] = []
                    for category in cat_data.categories:
                        if category in vl:
                            categories.append(vl[category])
                        else:
                            categories.append(str(category))
                else:
                    categories = list(vl.values())
                try:
                    cat_data = cat_data.rename_categories(categories)
                except ValueError as err:
                    vc = Series(categories, copy=False).value_counts()
                    repeated_cats = list(vc.index[vc > 1])
                    repeats = '-' * 80 + '\n' + '\n'.join(repeated_cats)
                    msg: str = (
                        f'\nValue labels for column {col} are not unique. These cannot be converted to\n'
                        'pandas categoricals.\n\n'
                        'Either read the file with `convert_categoricals` set to False or use the\n'
                        'low level interface in `StataReader` to separately read the values and the\n'
                        'value_labels.\n\n'
                        f'The repeated labels are:\n{repeats}\n'
                    )
                    raise ValueError(msg) from err
                cat_series: Series = Series(cat_data, index=data.index, copy=False)
                cat_converted_data.append((col, cat_series))
            else:
                cat_converted_data.append((col, data[col]))
        data = DataFrame(dict(cat_converted_data), copy=False)
        return data

    @property
    def data_label(self) -> str:
        """
        Return data label of Stata file.

        The data label is a descriptive string associated with the dataset
        stored in the Stata file. This property provides access to that
        label, if one is present.

        See Also
        --------
        io.stata.StataReader.variable_labels : Return a dict associating each variable
            name with corresponding label.
        DataFrame.to_stata : Export DataFrame object to Stata dta format.

        Examples
        --------
        >>> df = pd.DataFrame([(1,)], columns=["variable"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> data_label = "This is a data file."
        >>> path = "/My_path/filename.dta"
        >>> df.to_stata(
        ...     path,
        ...     time_stamp=time_stamp,  # doctest: +SKIP
        ...     data_label=data_label,  # doctest: +SKIP
        ...     version=None,
        ... )  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.data_label)  # doctest: +SKIP
        This is a data file.
        """
        self._ensure_open()
        return self._data_label

    @property
    def time_stamp(self) -> str:
        """
        Return time stamp of Stata file.
        """
        self._ensure_open()
        return self._time_stamp

    def variable_labels(self) -> Dict[str, str]:
        """
        Return a dict associating each variable name with corresponding label.

        This method retrieves variable labels from a Stata file. Variable labels are
        mappings between variable names and their corresponding descriptive labels
        in a Stata dataset.

        Returns
        -------
        dict
            A python dictionary.

        See Also
        --------
        read_stata : Read Stata file into DataFrame.
        DataFrame.to_stata : Export DataFrame object to Stata dta format.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> variable_labels = {"col_1": "This is an example"}
        >>> df.to_stata(
        ...     path,
        ...     time_stamp=time_stamp,  # doctest: +SKIP
        ...     variable_labels=variable_labels,
        ...     version=None,
        ... )  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.variable_labels())  # doctest: +SKIP
        {'index': '', 'col_1': 'This is an example', 'col_2': ''}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    3    4
        """
        self._ensure_open()
        return dict(zip(self._varlist, self._variable_labels))

    def value_labels(self) -> Dict[str, Dict[int, str]]:
        """
        Return a nested dict associating each variable name to its value and label.

        This method retrieves the value labels from a Stata file. Value labels are
        mappings between the coded values and their corresponding descriptive labels
        in a Stata dataset.

        Returns
        -------
        dict
            A python dictionary.

        See Also
        --------
        read_stata : Read Stata file into DataFrame.
        DataFrame.to_stata : Export DataFrame object to Stata dta format.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> value_labels = {"col_1": {3: "x"}}
        >>> df.to_stata(
        ...     path,
        ...     time_stamp=time_stamp,  # doctest: +SKIP
        ...     value_labels=value_labels,
        ...     version=None,
        ... )  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.value_labels())  # doctest: +SKIP
        {'col_1': {3: 'x'}}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    x    4
        """
        if not self._value_labels_read:
            self._read_value_labels()
        return self._value_label_dict


@Appender(_read_stata_doc)
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
    storage_options: Optional[Dict[str, Any]] = None,
) -> Union[DataFrame, StataReader]:
    reader = StataReader(
        filepath_or_buffer,
        convert_dates=convert_dates,
        convert_categoricals=convert_categoricals,
        index_col=index_col,
        convert_missing=convert_missing,
        preserve_dtypes=preserve_dtypes,
        columns=columns,
        order_categoricals=order_categoricals,
        chunksize=chunksize,
        storage_options=storage_options,
        compression=compression,
    )
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


def _pad_bytes(name: Union[str, bytes], length: int) -> bytes:
    """
    Take a char string and pads it with null bytes until it's length chars.
    """
    if isinstance(name, bytes):
        return name + b'\x00' * (length - len(name))
    return name.encode('latin-1') + b'\x00' * (length - len(name))


def _convert_datetime_to_stata_type(fmt: str) -> np.dtype:
    """
    Convert from one of the stata date formats to a type in TYPE_MAP.
    """
    if fmt in ['tc', '%tc', 'td', '%td', 'tw', '%tw', 'tm', '%tm', 'tq', '%tq', 'th', '%th', 'ty', '%ty']:
        return np.dtype(np.float64)
    else:
        raise NotImplementedError(f'Format {fmt} not implemented')


def _maybe_convert_to_int_keys(
    convert_dates: Dict[Union[str, int], str],
    varlist: List[str],
) -> Dict[int, str]:
    new_dict: Dict[int, str] = {}
    for key, value in convert_dates.items():
        if not isinstance(convert_dates[key], str) or not convert_dates[key].startswith('%'):
            convert_dates[key] = '%' + value
        if isinstance(key, str) and key in varlist:
            new_dict[varlist.index(key)] = convert_dates[key]
        else:
            if not isinstance(key, int):
                raise ValueError('convert_dates key must be a column or an integer')
            new_dict[key] = convert_dates[key]
    return new_dict


def _dtype_to_stata_type(
    dtype: np.dtype,
    column: Series,
) -> Union[int, str]:
    """
    Convert dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 244 are strings of this length
                         Pandas    Stata
    251 - for int8      byte
    252 - for int16     int
    253 - for int32     long
    254 - for float32   float
    255 - for double    double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
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


def _dtype_to_default_stata_fmt(
    dtype: np.dtype,
    column: Series,
    dta_version: int = 114,
    force_strl: bool = False,
) -> str:
    """
    Map numpy dtype to stata's default format for this type. Not terribly
    important since users can change this in Stata. Semantics are

    object  -> "%DDs" where DD is the length of the string.  If not a string,
                raise ValueError
    float64 -> "%10.0g"
    float32 -> "%9.0g"
    int64   -> "%9.0g"
    int32   -> "%12.0g"
    int16   -> "%8.0g"
    int8    -> "%8.0g"
    strl    -> "%9s"
    """
    if dta_version < 117:
        max_str_len: int = 244
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
        return f'%{max(itemsize, 1)}s'
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


class StataWriter(StataParser):
    """
    A class for writing Stata binary dta files

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, pathlib.Path or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter instance
        The StataWriter instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1]], columns=["a", "b"])
    >>> writer = StataWriter("./data_file.dta", data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {{"method": "zip", "archive_name": "data_file.dta"}}
    >>> writer = StataWriter("./data_file.zip", data, compression=compression)
    >>> writer.write_file()

    Save a DataFrame with dates
    >>> from datetime import datetime
    >>> data = pd.DataFrame([[datetime(2000, 1, 1)]], columns=["date"])
    >>> writer = StataWriter("./date_data_file.dta", data, {{"date": "tw"}})
    >>> writer.write_file()
    """
    _max_string_length: Final[int] = 244
    _encoding: Final[str] = 'latin-1'

    def __init__(
        self,
        fname: Union[str, os.PathLike, IO[bytes]],
        data: DataFrame,
        convert_dates: Optional[Dict[Union[str, int], str]] = None,
        write_index: bool = True,
        byteorder: Optional[str] = None,
        time_stamp: Optional[datetime] = None,
        data_label: Optional[str] = None,
        variable_labels: Optional[Dict[str, str]] = None,
        compression: str = 'infer',
        storage_options: Optional[Dict[str, Any]] = None,
        *,
        value_labels: Optional[Dict[str, Dict[Union[int, float], str]]] = None,
    ) -> None:
        super().__init__()
        self.data: DataFrame = data
        self._convert_dates: Dict[Union[str, int], str] = {} if convert_dates is None else convert_dates
        self._write_index: bool = write_index
        self._time_stamp: Optional[datetime] = time_stamp
        self._data_label: Optional[str] = data_label
        self._variable_labels: Optional[Dict[str, str]] = variable_labels
        self._non_cat_value_labels: Optional[Dict[str, Dict[Union[int, float], str]]] = value_labels
        self._value_labels: List[StataValueLabel] = []
        self._has_value_labels: np.ndarray = np.array([], dtype=bool)
        self._compression: str = compression
        self._output_file: Optional[bytes] = None
        self._converted_names: Dict[str, str] = {}
        self._prepare_pandas(data)
        self.storage_options: Optional[Dict[str, Any]] = storage_options
        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder: str = _set_endianness(byteorder)
        self._fname: Union[str, os.PathLike, IO[bytes]] = fname
        self.type_converters: Dict[int, type] = {253: np.int32, 252: np.int16, 251: np.int8}

    def _write(self, to_write: str) -> None:
        """
        Helper to call encode before writing to file for Python 3 compat.
        """
        self.handles.handle.write(to_write.encode(self._encoding))

    def _write_bytes(self, value: bytes) -> None:
        """
        Helper to assert file is open before writing.
        """
        self.handles.handle.write(value)

    def _prepare_non_cat_value_labels(
        self,
        data: DataFrame,
    ) -> List[StataNonCatValueLabel]:
        """
        Check for value labels provided for non-categorical columns. Value
        labels
        """
        non_cat_value_labels: List[StataNonCatValueLabel] = []
        if self._non_cat_value_labels is None:
            return non_cat_value_labels
        for labname, labels in self._non_cat_value_labels.items():
            if labname in self._converted_names:
                colname = self._converted_names[labname]
            elif labname in data.columns:
                colname = str(labname)
            else:
                raise KeyError(f"Can't create value labels for {labname}, it wasn't found in the dataset.")
            if not is_numeric_dtype(data[colname].dtype):
                raise ValueError(f"Can't create value labels for {labname}, value labels can only be applied to numeric columns.")
            svl = StataNonCatValueLabel(labname, labels, self._encoding)
            non_cat_value_labels.append(svl)
        return non_cat_value_labels

    def _prepare_categoricals(self, data: DataFrame) -> DataFrame:
        """
        Check for categorical columns, retain categorical information for
        Stata file and convert categorical data to int
        """
        is_cat: List[bool] = [isinstance(dtype, CategoricalDtype) for dtype in data.dtypes]
        if not any(is_cat):
            return data
        self._has_value_labels |= np.array(is_cat)
        get_base_missing_value = StataMissingValue.get_base_missing_value
        data_formatted: List[Tuple[str, Series]] = []
        for col, col_is_cat in zip(data, is_cat):
            if col_is_cat:
                svl = StataValueLabel(data[col], encoding=self._encoding)
                self._value_labels.append(svl)
                dtype: np.dtype = data[col].cat.codes.dtype
                if dtype == np.int64:
                    raise ValueError('It is not possible to export int64-based categorical data to Stata.')
                values: np.ndarray = data[col].cat.codes._values.copy()
                if values.max() >= get_base_missing_value(dtype):
                    if dtype == np.int8:
                        dtype_new: np.dtype = np.dtype(np.int16)
                    elif dtype == np.int16:
                        dtype_new = np.dtype(np.int32)
                    else:
                        dtype_new = np.dtype(np.float64)
                    values = np.array(values, dtype=dtype_new)
                values[values == -1] = get_base_missing_value(dtype)
                data_formatted.append((col, Series(values, dtype=dtype_new if 'dtype_new' in locals() else dtype)))
            else:
                data_formatted.append((col, data[col]))
        return DataFrame.from_dict(dict(data_formatted))

    def _replace_nans(self, data: DataFrame) -> DataFrame:
        """
        Checks floating point data columns for nans, and replaces these with
        the generic Stata for missing value (.)
        """
        for c in data:
            dtype = data[c].dtype
            if dtype in (np.float32, np.float64):
                if dtype == np.float32:
                    replacement: Union[int, float] = self.MISSING_VALUES['f']
                else:
                    replacement = self.MISSING_VALUES['d']
                data[c] = data[c].fillna(replacement)
        return data

    def _update_strl_names(self) -> None:
        """No-op, forward compatibility"""
        pass

    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 114 and 117 support ascii characters in a-z, A-Z, 0-9
        and _.
        """
        for c in name:
            if (c < 'A' or c > 'Z') and (c < 'a' or c > 'z') and (c < '0' or c > '9') and (c != '_'):
                name = name.replace(c, '_')
        return name

    def _check_column_names(self, data: DataFrame) -> DataFrame:
        """
        Checks column names to ensure that they are valid Stata column names.
        This includes checks for:
            * Non-string names
            * Stata keywords
            * Variables that start with numbers
            * Variables with names that are too long

        When an illegal variable name is detected, it is converted, and if
        dates are exported, the variable name is propagated to the date
        conversion dictionary
        """
        converted_names: Dict[str, str] = {}
        columns: List[str] = list(data.columns)
        original_columns: List[str] = columns[:]
        duplicate_var_id: int = 0
        for j, name in enumerate(columns):
            orig_name: str = name
            if not isinstance(name, str):
                name = str(name)
            name = self._validate_variable_name(name)
            if name in self.RESERVED_WORDS:
                name = '_' + name
            if '0' <= name[0] <= '9':
                name = '_' + name
            name = name[:min(len(name), 32)]
            if not name == orig_name:
                while columns.count(name) > 0:
                    name = '_' + str(duplicate_var_id) + name
                    name = name[:min(len(name), 32)]
                    duplicate_var_id += 1
                converted_names[orig_name] = name
            columns[j] = name
        data.columns = Index(columns)
        if self._convert_dates:
            for c, o in zip(columns, original_columns):
                if c != o:
                    self._convert_dates[c] = self._convert_dates[o]
                    del self._convert_dates[o]
        if converted_names:
            conversion_warning: List[str] = []
            for orig_name, name in converted_names.items():
                msg: str = f'{orig_name}   ->   {name}'
                conversion_warning.append(msg)
            ws: str = invalid_name_doc.format('\n    '.join(conversion_warning))
            warnings.warn(ws, InvalidColumnName, stacklevel=find_stack_level())
        self._converted_names = converted_names
        self._update_strl_names()
        return data

    def _set_formats_and_types(self, dtypes: Series) -> None:
        self.fmtlist: List[str] = []
        self.typlist: List[Union[int, str]] = []
        for col, dtype in dtypes.items():
            self.fmtlist.append(_dtype_to_default_stata_fmt(dtype, self.data[col]))
            self.typlist.append(_dtype_to_stata_type(dtype, self.data[col]))

    def _prepare_pandas(self, data: DataFrame) -> None:
        data = data.copy()
        if self._write_index:
            temp: DataFrame = data.reset_index()
            if isinstance(temp, DataFrame):
                data = temp
        data = self._check_column_names(data)
        data = _cast_to_stata_types(data)
        data = self._replace_nans(data)
        self._has_value_labels = np.repeat(False, data.shape[1])
        non_cat_value_labels: List[StataNonCatValueLabel] = self._prepare_non_cat_value_labels(data)
        non_cat_columns: List[str] = [svl.labname for svl in non_cat_value_labels]
        has_non_cat_val_labels: np.ndarray = data.columns.isin(non_cat_columns)
        self._has_value_labels |= has_non_cat_val_labels
        self._value_labels.extend(non_cat_value_labels)
        data = self._prepare_categoricals(data)
        self.nobs, self.nvar = data.shape
        self.data = data
        self.varlist: List[str] = data.columns.tolist()
        dtypes: Series = data.dtypes
        for col in data:
            if col in self._convert_dates:
                continue
            if lib.is_np_dtype(data[col].dtype, 'M'):
                self._convert_dates[col] = 'tc'
        self._convert_dates = _maybe_convert_to_int_keys(self._convert_dates, self.varlist)
        for key in self._convert_dates:
            new_type: np.dtype = _convert_datetime_to_stata_type(self._convert_dates[key])
            dtypes.iloc[key] = new_type
        self._encode_strings()
        self._set_formats_and_types(dtypes)
        if self._convert_dates is not None:
            for key in self._convert_dates:
                if isinstance(key, int):
                    self.fmtlist[key] = self._convert_dates[key]

    def _encode_strings(self) -> None:
        """
        Encode strings in dta-specific encoding

        Do not encode columns marked for date conversion or for strL
        conversion. The strL converter independently handles conversion and
        also accepts empty string arrays.
        """
        convert_dates = self._convert_dates
        convert_strl: List[str] = getattr(self, '_convert_strl', [])
        for i, col in enumerate(self.data):
            if i in convert_dates or col in convert_strl:
                continue
            column = self.data[col]
            dtype = column.dtype
            if dtype.type is np.object_:
                inferred_dtype: str = infer_dtype(column, skipna=True)
                if not (inferred_dtype == 'string' or len(column) == 0):
                    col = column.name
                    raise ValueError(
                        f'Column `{col}` cannot be exported.\n\n'
                        'Only string-like object arrays\ncontaining all strings or a mix of strings and None can be exported.\n'
                        'Object arrays containing only null values are prohibited. Other object\ntypes cannot be exported and must first be converted to one of the\nsupported types.'
                    )
                encoded: Series = self.data[col].str.encode(self._encoding)
                if max_len_string_array(ensure_object(encoded._values)) <= self._max_string_length:
                    self.data[col] = encoded

    def _write_file(self) -> None:
        """
        Export DataFrame object to Stata dta format.

        This method writes the contents of a pandas DataFrame to a `.dta` file
        compatible with Stata. It includes features for handling value labels,
        variable types, and metadata like timestamps and data labels. The output
        file can then be read and used in Stata or other compatible statistical
        tools.
        
        See Also
        --------
        read_stata : Read Stata file into DataFrame.
        DataFrame.to_stata : Export DataFrame object to Stata dta format.
        io.stata.StataWriter : A class for writing Stata binary dta files.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "fully_labelled": [1, 2, 3, 3, 1],
        ...         "partially_labelled": [1.0, 2.0, np.nan, 9.0, np.nan],
        ...         "Y": [7, 7, 9, 8, 10],
        ...         "Z": pd.Categorical(["j", "k", "l", "k", "j"]),
        ...     }
        ... )
        >>> path = "/My_path/filename.dta"
        >>> labels = {
        ...     "fully_labelled": {1: "one", 2: "two", 3: "three"},
        ...     "partially_labelled": {1.0: "one", 2.0: "two"},
        ... }
        >>> writer = pd.io.stata.StataWriter(
        ...     path, df, value_labels=labels
        ... )  # doctest: +SKIP
        >>> writer.write_file()  # doctest: +SKIP
        >>> df = pd.read_stata(path)  # doctest: +SKIP
        >>> df  # doctest: +SKIP
            index fully_labelled  partially_labeled  Y  Z
        0       0            one                one  7  j
        1       1            two                two  7  k
        2       2          three                NaN  9  l
        3       3          three                9.0  8  k
        4       4            one                NaN 10  j
        """
        with get_handle(
            self._fname,
            'wb',
            compression=self._compression,
            is_text=False,
            storage_options=self.storage_options,
        ) as self.handles:
            if self.handles.compression['method'] is not None:
                self._output_file, self.handles.handle = (self.handles.handle, BytesIO())
                self.handles.created_handles.append(self.handles.handle)
            try:
                self._write_header(data_label=self._data_label, time_stamp=self._time_stamp)
                self._write_map()
                self._write_variable_types()
                self._write_varnames()
                self._write_sortlist()
                self._write_formats()
                self._write_value_label_names()
                self._write_variable_labels()
                self._write_expansion_fields()
                self._write_characteristics()
                records = self._prepare_data()
                self._write_data(records)
                self._write_strls()
                self._write_value_labels()
                self._write_file_close_tag()
                self._write_map()
                self._close()
            except Exception as exc:
                self.handles.close()
                if isinstance(self._fname, (str, os.PathLike)) and os.path.isfile(self._fname):
                    try:
                        os.unlink(self._fname)
                    except OSError:
                        warnings.warn(
                            f'This save was not successful but {self._fname} could not be deleted. This file is not valid.',
                            ResourceWarning,
                            stacklevel=find_stack_level(),
                        )
                raise exc

    def _close(self) -> None:
        """
        Close the file if it was created by the writer.

        If a buffer or file-like object was passed in, for example a GzipFile,
        then leave this file open for the caller to close.
        """
        if self._output_file is not None:
            assert isinstance(self.handles.handle, BytesIO)
            bio, self.handles.handle = self.handles.handle, self._output_file
            self.handles.handle.write(bio.getvalue())

    def _write_map(self) -> None:
        """
        Called twice during file write. The first populates the values in
        the map with 0s.  The second call writes the final map locations when
        all blocks have been written.
        """
        if not self._map:
            self._map = {
                'stata_data': 0,
                'map': self.handles.handle.tell(),
                'variable_types': 0,
                'varnames': 0,
                'sortlist': 0,
                'formats': 0,
                'value_label_names': 0,
                'variable_labels': 0,
                'characteristics': 0,
                'data': 0,
                'strls': 0,
                'value_labels': 0,
                'stata_data_close': 0,
                'end-of-file': 0,
            }
        self.handles.handle.seek(self._map['map'])
        bio = BytesIO()
        for val in self._map.values():
            bio.write(struct.pack(self._byteorder + 'Q', val))
        self._write_bytes(self._tag(bio.getvalue(), 'map'))

    def _write_variable_types(self) -> None:
        self._update_map('variable_types')
        bio = BytesIO()
        for typ in self.typlist:
            bio.write(struct.pack(self._byteorder + 'H', typ))
        self._write_bytes(self._tag(bio.getvalue(), 'variable_types'))

    def _write_varnames(self) -> None:
        self._update_map('varnames')
        bio = BytesIO()
        vn_len: int = 32 if self._dta_version == 117 else 128
        for name in self.varlist:
            name_padded: bytes = self._null_terminate_str(name).encode(self._encoding)[:32]
            name_padded = _pad_bytes_new(name_padded, vn_len + 1)
            bio.write(name_padded)
        self._write_bytes(self._tag(bio.getvalue(), 'varnames'))

    def _write_sortlist(self) -> None:
        self._update_map('sortlist')
        sort_size: int = 2 if self._dta_version < 119 else 4
        sortlist = b'\x00' * sort_size * (self.nvar + 1)
        self._write_bytes(self._tag(sortlist, 'sortlist'))

    def _write_formats(self) -> None:
        self._update_map('formats')
        bio = BytesIO()
        fmt_len: int = 49 if self._dta_version == 117 else 57
        for fmt in self.fmtlist:
            bio.write(_pad_bytes_new(fmt.encode(self._encoding), fmt_len))
        self._write_bytes(self._tag(bio.getvalue(), 'formats'))

    def _write_value_label_names(self) -> None:
        self._update_map('value_label_names')
        bio = BytesIO()
        vl_len: int = 32 if self._dta_version == 117 else 128
        for i in range(self.nvar):
            if self._has_value_labels[i]:
                name: str = self.varlist[i]
            else:
                name = ''
            name_encoded: bytes = self._null_terminate_str(name).encode(self._encoding)[:32]
            encoded_name: bytes = _pad_bytes_new(name_encoded, vl_len + 1)
            bio.write(encoded_name)
        self._write_bytes(self._tag(bio.getvalue(), 'value_label_names'))

    def _write_variable_labels(self) -> None:
        self._update_map('variable_labels')
        bio = BytesIO()
        vl_len: int = 80 if self._dta_version == 117 else 320
        blank: bytes = _pad_bytes_new(b'', vl_len + 1)
        if self._variable_labels is None:
            for _ in range(self.nvar):
                bio.write(blank)
            self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))
            return
        for col in self.data:
            if col in self._variable_labels:
                label: str = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError('Variable labels must be 80 characters or fewer')
                is_latin1: bool = all(ord(c) < 256 for c in label)
                if not is_latin1:
                    raise ValueError('Variable labels must contain only characters that can be encoded in Latin-1')
                encoded_label: bytes = _pad_bytes_new(label.encode(self._encoding), vl_len + 1)
                bio.write(encoded_label)
            else:
                bio.write(blank)
        self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))

    def _write_characteristics(self) -> None:
        self._update_map('characteristics')
        self._write_bytes(self._tag(b'', 'characteristics'))

    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """
        Convert columns to StrLs if either very large or in the
        convert_strl variable
        """
        convert_cols: List[str] = [col for i, col in enumerate(data) if self.typlist[i] == 32768 or col in self._convert_strl]
        if convert_cols:
            ssw: StataStrLWriter = StataStrLWriter(data, convert_cols, version=self._dta_version, byteorder=self._byteorder)
            tab, new_data = ssw.generate_table()
            data = new_data
            self._strl_blob = ssw.generate_blob(tab)
        return data

    def _set_formats_and_types(self, dtypes: Series) -> None:
        self.fmtlist = []
        self.typlist = []
        for col, dtype in dtypes.items():
            self.fmtlist.append(_dtype_to_default_stata_fmt(dtype, self.data[col]))
            self.typlist.append(_dtype_to_stata_type(dtype, self.data[col]))

    def _prepare_data(self) -> np.recarray:
        data: DataFrame = self.data
        typlist = self.typlist
        convert_dates = self._convert_dates
        if self._convert_dates is not None:
            for i, col in enumerate(data):
                if i in convert_dates:
                    data[col] = _datetime_to_stata_elapsed_vec(
                        data[col], self.fmtlist[i]
                    )
        data = self._convert_strls(data)
        dtypes: Dict[str, Union[str, np.dtype]] = {}
        native_byteorder: bool = self._byteorder == _set_endianness(sys.byteorder)
        for i, col in enumerate(data):
            typ = typlist[i]
            if typ <= self._max_string_length:
                dc = data[col].fillna('')
                data[col] = dc.apply(lambda x: _pad_bytes(x, typ))
                stype = f'S{typ}'
                dtypes[col] = stype
                data[col] = data[col].astype(stype)
            else:
                dtype = data[col].dtype
                if not native_byteorder:
                    dtype = dtype.newbyteorder(self._byteorder)
                dtypes[col] = dtype
        return data.to_records(index=False, column_dtypes=dtypes)

    def _write_data(self, records: np.recarray) -> None:
        self._write_bytes(records.tobytes())

    @staticmethod
    def _null_terminate_str(s: str) -> str:
        s += '\x00'
        return s

    def _null_terminate_bytes(self, s: str) -> bytes:
        return self._null_terminate_str(s).encode(self._encoding)

    def _do_convert_categoricals(
        self,
        data: DataFrame,
        value_label_dict: Dict[str, Dict[int, str]],
        lbllist: List[str],
        order_categoricals: bool,
    ) -> DataFrame:
        """
        Converts categorical columns to Categorical type.
        """
        if not value_label_dict:
            return data
        cat_converted_data: List[Tuple[str, Series]] = []
        for col, label in zip(data, lbllist):
            if label in value_label_dict:
                vl: Dict[int, str] = value_label_dict[label]
                keys = np.array(list(vl.keys()))
                column = data[col]
                key_matches = column.isin(keys)
                if self._using_iterator and key_matches.all():
                    initial_categories: Optional[List[int]] = keys.tolist()
                else:
                    if self._using_iterator:
                        warnings.warn(categorical_conversion_warning, CategoricalConversionWarning, stacklevel=find_stack_level())
                    initial_categories = None
                cat_data = Categorical(column, categories=initial_categories, ordered=order_categoricals)
                if initial_categories is None:
                    categories: List[str] = []
                    for category in cat_data.categories:
                        if category in vl:
                            categories.append(vl[category])
                        else:
                            categories.append(str(category))
                else:
                    categories = list(vl.values())
                try:
                    cat_data = cat_data.rename_categories(categories)
                except ValueError as err:
                    vc = Series(categories, copy=False).value_counts()
                    repeated_cats: List[str] = list(vc.index[vc > 1])
                    repeats: str = '-' * 80 + '\n' + '\n'.join(repeated_cats)
                    msg: str = (
                        f'\nValue labels for column {col} are not unique. These cannot be converted to\n'
                        'pandas categoricals.\n\n'
                        'Either read the file with `convert_categoricals` set to False or use the\n'
                        'low level interface in `StataReader` to separately read the values and the\n'
                        'value_labels.\n\n'
                        f'The repeated labels are:\n{repeats}\n'
                    )
                    raise ValueError(msg) from err
                cat_series: Series = Series(cat_data, index=data.index, copy=False)
                cat_converted_data.append((col, cat_series))
            else:
                cat_converted_data.append((col, data[col]))
        data = DataFrame(dict(cat_converted_data), copy=False)
        return data


@doc(
    compression_options=_shared_docs['compression_options'] % 'fname',
    storage_options=_shared_docs['storage_options'],
)
class StataWriter(StataParser):
    """
    A class for writing Stata binary dta files

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, pathlib.Path or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter instance
        The StataWriter instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1]], columns=["a", "b"])
    >>> writer = StataWriter("./data_file.dta", data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {{"method": "zip", "archive_name": "data_file.dta"}}
    >>> writer = StataWriter("./data_file.zip", data, compression=compression)
    >>> writer.write_file()

    Save a DataFrame with dates
    >>> from datetime import datetime
    >>> data = pd.DataFrame([[datetime(2000, 1, 1)]], columns=["date"])
    >>> writer = StataWriter("./date_data_file.dta", data, {{"date": "tw"}})
    >>> writer.write_file()
    """
    _max_string_length: Final[int] = 244
    _encoding: Final[str] = 'latin-1'

    def __init__(
        self,
        fname: Union[str, os.PathLike, IO[bytes]],
        data: DataFrame,
        convert_dates: Optional[Dict[Union[str, int], str]] = None,
        write_index: bool = True,
        byteorder: Optional[str] = None,
        time_stamp: Optional[datetime] = None,
        data_label: Optional[str] = None,
        variable_labels: Optional[Dict[str, str]] = None,
        compression: str = 'infer',
        storage_options: Optional[Dict[str, Any]] = None,
        *,
        value_labels: Optional[Dict[str, Dict[Union[int, float], str]]] = None,
    ) -> None:
        super().__init__()
        self.data: DataFrame = data
        self._convert_dates: Dict[Union[str, int], str] = {} if convert_dates is None else convert_dates
        self._write_index: bool = write_index
        self._time_stamp: Optional[datetime] = time_stamp
        self._data_label: Optional[str] = data_label
        self._variable_labels: Optional[Dict[str, str]] = variable_labels
        self._non_cat_value_labels: Optional[Dict[str, Dict[Union[int, float], str]]] = value_labels
        self._value_labels: List[StataValueLabel] = []
        self._has_value_labels: np.ndarray = np.array([], dtype=bool)
        self._compression: str = compression
        self._output_file: Optional[bytes] = None
        self._converted_names: Dict[str, str] = {}
        self._prepare_pandas(data)
        self.storage_options: Optional[Dict[str, Any]] = storage_options
        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder: str = _set_endianness(byteorder)
        self._fname: Union[str, os.PathLike, IO[bytes]] = fname
        self.type_converters: Dict[int, type] = {253: np.int32, 252: np.int16, 251: np.int8}

    def _write(self, to_write: str) -> None:
        """
        Helper to call encode before writing to file for Python 3 compat.
        """
        self.handles.handle.write(to_write.encode(self._encoding))

    def _write_bytes(self, value: bytes) -> None:
        """
        Helper to assert file is open before writing.
        """
        self.handles.handle.write(value)

    def _prepare_non_cat_value_labels(
        self,
        data: DataFrame,
    ) -> List[StataNonCatValueLabel]:
        """
        Check for value labels provided for non-categorical columns. Value
        labels
        """
        non_cat_value_labels: List[StataNonCatValueLabel] = []
        if self._non_cat_value_labels is None:
            return non_cat_value_labels
        for labname, labels in self._non_cat_value_labels.items():
            if labname in self._converted_names:
                colname = self._converted_names[labname]
            elif labname in data.columns:
                colname = str(labname)
            else:
                raise KeyError(f"Can't create value labels for {labname}, it wasn't found in the dataset.")
            if not is_numeric_dtype(data[colname].dtype):
                raise ValueError(f"Can't create value labels for {labname}, value labels can only be applied to numeric columns.")
            svl = StataNonCatValueLabel(labname, labels, self._encoding)
            non_cat_value_labels.append(svl)
        return non_cat_value_labels

    def _prepare_categoricals(self, data: DataFrame) -> DataFrame:
        """
        Check for categorical columns, retain categorical information for
        Stata file and convert categorical data to int
        """
        is_cat: List[bool] = [isinstance(dtype, CategoricalDtype) for dtype in data.dtypes]
        if not any(is_cat):
            return data
        self._has_value_labels |= np.array(is_cat)
        get_base_missing_value = StataMissingValue.get_base_missing_value
        data_formatted: List[Tuple[str, Series]] = []
        for col, col_is_cat in zip(data, is_cat):
            if col_is_cat:
                svl = StataValueLabel(data[col], encoding=self._encoding)
                self._value_labels.append(svl)
                dtype: np.dtype = data[col].cat.codes.dtype
                if dtype == np.int64:
                    raise ValueError('It is not possible to export int64-based categorical data to Stata.')
                values: np.ndarray = data[col].cat.codes._values.copy()
                if values.max() >= get_base_missing_value(dtype):
                    if dtype == np.int8:
                        dtype_new: np.dtype = np.int16
                    elif dtype == np.int16:
                        dtype_new = np.int32
                    else:
                        dtype_new = np.float64
                    values = np.array(values, dtype=dtype_new)
                values[values == -1] = get_base_missing_value(dtype)
                data_formatted.append((col, Series(values, dtype=dtype_new if 'dtype_new' in locals() else dtype)))
            else:
                data_formatted.append((col, data[col]))
        return DataFrame.from_dict(dict(data_formatted))

    def _replace_nans(self, data: DataFrame) -> DataFrame:
        """
        Checks floating point data columns for nans, and replaces these with
        the generic Stata for missing value (.)
        """
        for c in data:
            dtype = data[c].dtype
            if dtype in (np.float32, np.float64):
                if dtype == np.float32:
                    replacement: Union[int, float] = self.MISSING_VALUES['f']
                else:
                    replacement = self.MISSING_VALUES['d']
                data[c] = data[c].fillna(replacement)
        return data

    def _update_strl_names(self) -> None:
        """
        Update column names for conversion to strl if they might have been
        changed to comply with Stata naming rules
        """
        for orig, new in self._converted_names.items():
            if orig in self._convert_strl:
                idx: int = self._convert_strl.index(orig)
                self._convert_strl[idx] = new

    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 118+ support most unicode characters. The only limitation is in
        the ascii range where the characters supported are a-z, A-Z, 0-9 and _.
        """
        for c in name:
            if (
                (128 <= ord(c) < 192)
                or (c in {'×', '÷'})
                or (
                    ord(c) < 128
                    and (
                        (c < 'A' or c > 'Z')
                        and (c < 'a' or c > 'z')
                        and (c < '0' or c > '9')
                        and (c != '_')
                    )
                )
            ):
                name = name.replace(c, '_')
        return name

    def _check_column_names(self, data: DataFrame) -> DataFrame:
        """
        Checks column names to ensure that they are valid Stata column names.
        This includes checks for:
            * Non-string names
            * Stata keywords
            * Variables that start with numbers
            * Variables with names that are too long

        When an illegal variable name is detected, it is converted, and if
        dates are exported, the variable name is propagated to the date
        conversion dictionary
        """
        converted_names: Dict[str, str] = {}
        columns: List[str] = list(data.columns)
        original_columns: List[str] = columns[:]
        duplicate_var_id: int = 0
        for j, name in enumerate(columns):
            orig_name: str = name
            if not isinstance(name, str):
                name = str(name)
            name = self._validate_variable_name(name)
            if name in self.RESERVED_WORDS:
                name = '_' + name
            if '0' <= name[0] <= '9':
                name = '_' + name
            name = name[:min(len(name), 32)]
            if not name == orig_name:
                while columns.count(name) > 0:
                    name = '_' + str(duplicate_var_id) + name
                    name = name[:min(len(name), 32)]
                    duplicate_var_id += 1
                converted_names[orig_name] = name
            columns[j] = name
        data.columns = Index(columns)
        if self._convert_dates:
            for c, o in zip(columns, original_columns):
                if c != o:
                    self._convert_dates[c] = self._convert_dates[o]
                    del self._convert_dates[o]
        if converted_names:
            conversion_warning: List[str] = []
            for orig_name, name in converted_names.items():
                msg: str = f'{orig_name}   ->   {name}'
                conversion_warning.append(msg)
            ws: str = invalid_name_doc.format('\n    '.join(conversion_warning))
            warnings.warn(ws, InvalidColumnName, stacklevel=find_stack_level())
        self._converted_names = converted_names
        self._update_strl_names()
        return data

    def _set_formats_and_types(self, dtypes: Series) -> None:
        """Set formats and types based on dtypes"""
        self.fmtlist = []
        self.typlist = []
        for col, dtype in dtypes.items():
            self.fmtlist.append(_dtype_to_default_stata_fmt(dtype, self.data[col]))
            self.typlist.append(_dtype_to_stata_type(dtype, self.data[col]))

    def _prepare_pandas(self, data: DataFrame) -> None:
        data = data.copy()
        if self._write_index:
            temp: DataFrame = data.reset_index()
            if isinstance(temp, DataFrame):
                data = temp
        data = self._check_column_names(data)
        data = _cast_to_stata_types(data)
        data = self._replace_nans(data)
        self._has_value_labels = np.repeat(False, data.shape[1])
        non_cat_value_labels: List[StataNonCatValueLabel] = self._prepare_non_cat_value_labels(data)
        non_cat_columns: List[str] = [svl.labname for svl in non_cat_value_labels]
        has_non_cat_val_labels: np.ndarray = data.columns.isin(non_cat_columns)
        self._has_value_labels |= has_non_cat_val_labels
        self._value_labels.extend(non_cat_value_labels)
        data = self._prepare_categoricals(data)
        self.nobs, self.nvar = data.shape
        self.data = data
        self.varlist = data.columns.tolist()
        dtypes: Series = data.dtypes
        for col in data:
            if col in self._convert_dates:
                continue
            if lib.is_np_dtype(data[col].dtype, 'M'):
                self._convert_dates[col] = 'tc'
        self._convert_dates = _maybe_convert_to_int_keys(self._convert_dates, self.varlist)
        for key in self._convert_dates:
            new_type: np.dtype = _convert_datetime_to_stata_type(self._convert_dates[key])
            dtypes.iloc[key] = new_type
        self._encode_strings()
        self._set_formats_and_types(dtypes)
        if self._convert_dates is not None:
            for key in self._convert_dates:
                if isinstance(key, int):
                    self.fmtlist[key] = self._convert_dates[key]

    def _encode_strings(self) -> None:
        """
        Encode strings in dta-specific encoding

        Do not encode columns marked for date conversion or for strL
        conversion. The strL converter independently handles conversion and
        also accepts empty string arrays.
        """
        convert_dates = self._convert_dates
        convert_strl: List[str] = getattr(self, '_convert_strl', [])
        for i, col in enumerate(self.data):
            if i in convert_dates or col in convert_strl:
                continue
            column = self.data[col]
            dtype = column.dtype
            if dtype.type is np.object_:
                inferred_dtype: str = infer_dtype(column, skipna=True)
                if not (inferred_dtype == 'string' or len(column) == 0):
                    col = column.name
                    raise ValueError(
                        f'Column `{col}` cannot be exported.\n\n'
                        'Only string-like object arrays\ncontaining all strings or a mix of strings and None can be exported.\n'
                        'Object arrays containing only null values are prohibited. Other object\ntypes cannot be exported and must first be converted to one of the\nsupported types.'
                    )
                encoded: Series = self.data[col].str.encode(self._encoding)
                if max_len_string_array(ensure_object(encoded._values)) <= self._max_string_length:
                    self.data[col] = encoded

    def _write_file_close_tag(self) -> None:
        """Write the file closing tag"""
        self._update_map('stata_data_close')
        self._write_bytes(bytes('</stata_dta>', 'utf-8'))
        self._update_map('end-of-file')

    def _write_characteristics(self) -> None:
        """No-op in dta 117+"""
        self._update_map('characteristics')
        self._write_bytes(self._tag(b'', 'characteristics'))

    def _write_strls(self) -> None:
        """No-op in dta 117+"""
        self._update_map('strls')
        self._write_bytes(self._tag(self._strl_blob, 'strls'))

    def _write_expansion_fields(self) -> None:
        """Write 5 zeros for expansion fields"""
        self._write(_pad_bytes('', 5))

    def _write_value_labels(self) -> None:
        self._update_map('value_labels')
        bio = BytesIO()
        for vl in self._value_labels:
            lab: bytes = vl.generate_value_label(self._byteorder)
            lab = self._tag(lab, 'lbl')
            bio.write(lab)
        self._write_bytes(self._tag(bio.getvalue(), 'value_labels'))

    def _write_header(self, data_label: Optional[str], time_stamp: Optional[datetime]) -> None:
        """Write the file header"""
        byteorder = self._byteorder
        self._write_bytes(bytes('<stata_dta>', 'utf-8'))
        bio = BytesIO()
        bio.write(self._tag(bytes(str(self._dta_version), 'utf-8'), 'release'))
        bio.write(self._tag(b'MSF' if byteorder == '>' else b'LSF', 'byteorder'))
        nvar_type: str = 'H' if self._dta_version <= 118 else 'I'
        bio.write(self._tag(struct.pack(byteorder + nvar_type, self.nvar), 'K'))
        nobs_size: str = 'I' if self._dta_version == 117 else 'Q'
        bio.write(self._tag(struct.pack(byteorder + nobs_size, self.nobs), 'N'))
        if data_label is None:
            encoded_label: bytes = _pad_bytes_new(b'', 80 + 1)
        else:
            encoded_label: bytes = _pad_bytes_new(data_label[:80].encode(self._encoding), 80 + 1)
        bio.write(self._tag(encoded_label, 'label'))
        if time_stamp is None:
            time_stamp = datetime.now()
        elif not isinstance(time_stamp, datetime):
            raise ValueError('time_stamp should be datetime type')
        months: List[str] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_lookup: Dict[int, str] = {i + 1: month for i, month in enumerate(months)}
        ts = time_stamp.strftime('%d ') + month_lookup[time_stamp.month] + time_stamp.strftime(' %Y %H:%M')
        stata_ts: bytes = b'\x11' + bytes(ts, 'utf-8')
        bio.write(self._tag(stata_ts, 'timestamp'))
        self._write_bytes(self._tag(bio.getvalue(), 'header'))

    def _generate_tag(self, tag_content: bytes, tag_name: str) -> bytes:
        """
        Generate a tagged block.
        """
        return bytes(f'<{tag_name}>', 'utf-8') + tag_content + bytes(f'</{tag_name}>', 'utf-8')

    @staticmethod
    def _tag(val: bytes, tag: str) -> bytes:
        """Surround val with <tag></tag>"""
        if isinstance(val, str):
            val = bytes(val, 'utf-8')
        return bytes(f'<{tag}>', 'utf-8') + val + bytes(f'</{tag}>', 'utf-8')

    def _write_map(self) -> None:
        """
        Called twice during file write. The first populates the values in
        the map with 0s.  The second call writes the final map locations when
        all blocks have been written.
        """
        if not hasattr(self, '_map'):
            self._map: Dict[str, int] = {
                'stata_data': 0,
                'map': self.handles.handle.tell(),
                'variable_types': 0,
                'varnames': 0,
                'sortlist': 0,
                'formats': 0,
                'value_label_names': 0,
                'variable_labels': 0,
                'characteristics': 0,
                'data': 0,
                'strls': 0,
                'value_labels': 0,
                'stata_data_close': 0,
                'end-of-file': 0,
            }
        self.handles.handle.seek(self._map['map'])
        bio = BytesIO()
        for val in self._map.values():
            bio.write(struct.pack(self._byteorder + 'Q', val))
        self._write_bytes(self._tag(bio.getvalue(), 'map'))

    def _write_variable_types(self) -> None:
        self._update_map('variable_types')
        bio = BytesIO()
        for typ in self.typlist:
            bio.write(struct.pack(self._byteorder + 'H', typ))
        self._write_bytes(self._tag(bio.getvalue(), 'variable_types'))

    def _write_varnames(self) -> None:
        self._update_map('varnames')
        bio = BytesIO()
        vn_len: int = 32 if self._dta_version == 117 else 128
        for name in self.varlist:
            name_padded: bytes = self._null_terminate_str(name).encode(self._encoding)[:32]
            name_padded = _pad_bytes_new(name_padded, vn_len + 1)
            bio.write(name_padded)
        self._write_bytes(self._tag(bio.getvalue(), 'varnames'))

    def _write_sortlist(self) -> None:
        self._update_map('sortlist')
        sort_size: int = 2 if self._dta_version < 119 else 4
        sortlist = b'\x00' * sort_size * (self.nvar + 1)
        self._write_bytes(self._tag(sortlist, 'sortlist'))

    def _write_formats(self) -> None:
        self._update_map('formats')
        bio = BytesIO()
        fmt_len: int = 49 if self._dta_version == 117 else 57
        for fmt in self.fmtlist:
            bio.write(_pad_bytes_new(fmt.encode(self._encoding), fmt_len))
        self._write_bytes(self._tag(bio.getvalue(), 'formats'))

    def _write_value_label_names(self) -> None:
        self._update_map('value_label_names')
        bio = BytesIO()
        vl_len: int = 32 if self._dta_version == 117 else 128
        for i in range(self.nvar):
            if self._has_value_labels[i]:
                name: str = self.varlist[i]
            else:
                name = ''
            name_encoded: bytes = self._null_terminate_str(name).encode(self._encoding)[:32]
            encoded_name: bytes = _pad_bytes_new(name_encoded, vl_len + 1)
            bio.write(encoded_name)
        self._write_bytes(self._tag(bio.getvalue(), 'value_label_names'))

    def _write_variable_labels(self) -> None:
        self._update_map('variable_labels')
        bio = BytesIO()
        vl_len: int = 80 if self._dta_version == 117 else 320
        blank: bytes = _pad_bytes_new(b'', vl_len + 1)
        if self._variable_labels is None:
            for _ in range(self.nvar):
                bio.write(blank)
            self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))
            return
        for col in self.data:
            if col in self._variable_labels:
                label: str = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError('Variable labels must be 80 characters or fewer')
                try:
                    encoded_label: bytes = label.encode(self._encoding)
                except UnicodeEncodeError as err:
                    raise ValueError(f'Variable labels must contain only characters that can be encoded in {self._encoding}') from err
                encoded_label_padded: bytes = _pad_bytes_new(encoded_label, vl_len + 1)
                bio.write(encoded_label_padded)
            else:
                bio.write(blank)
        self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))

    def _write_characteristics(self) -> None:
        self._update_map('characteristics')
        self._write_bytes(self._tag(b'', 'characteristics'))

    def _write_data(self, records: np.recarray) -> None:
        self._update_map('data')
        self._write_bytes(b'<data>')
        self._write_bytes(records.tobytes())
        self._write_bytes(b'</data>')

    @staticmethod
    def _null_terminate_str(s: str) -> str:
        s += '\x00'
        return s

    def _null_terminate_bytes(self, s: str) -> bytes:
        return self._null_terminate_str(s).encode(self._encoding)


def _dtype_to_stata_type_117(
    dtype: np.dtype,
    column: Series,
    force_strl: bool = False,
) -> Union[int, str]:
    """
    Converts dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 2045 are strings of this length
                Pandas    Stata
    32768 - for object    strL
    65526 - for int8      byte
    65527 - for int16     int
    65528 - for int32     long
    65529 - for float32   float
    65530 - for double    double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
    if force_strl:
        return 32768
    if dtype.type is np.object_:
        itemsize = max_len_string_array(ensure_object(column._values))
        itemsize = max(itemsize, 1)
        if itemsize <= 2045:
            return itemsize
        return 32768
    elif dtype.type is np.float64:
        return 65526
    elif dtype.type is np.float32:
        return 65527
    elif dtype.type is np.int32:
        return 65528
    elif dtype.type is np.int16:
        return 65529
    elif dtype.type is np.int8:
        return 65530
    else:
        raise NotImplementedError(f'Data type {dtype} not supported.')


def _pad_bytes_new(name: bytes, length: int) -> bytes:
    """
    Takes a bytes instance and pads it with null bytes until it's length chars.
    """
    return name + b'\x00' * (length - len(name))


class StataStrLWriter:
    """
    Converter for Stata StrLs

    Stata StrLs map 8 byte values to strings which are stored using a
    dictionary-like format where strings are keyed to two values.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert
    columns : Sequence[str]
        List of columns names to convert to StrL
    version : int, optional
        dta version.  Currently supports 117, 118 and 119
    byteorder : str, optional
        Can be ">", "<", "little", or "big". default is `sys.byteorder`

    Notes
    -----
    Supports creation of the StrL block of a dta file for dta versions
    117, 118 and 119.  These differ in how the GSO is stored.  118 and
    119 store the GSO lookup value as a uint32 and a uint64, while 117
    uses two uint32s. 118 and 119 also encode all strings as unicode
    which is required by the format.  117 uses 'latin-1' a fixed width
    encoding that extends the 7-bit ascii table with an additional 128
    characters.
    """

    def __init__(
        self,
        df: DataFrame,
        columns: List[str],
        version: int = 117,
        byteorder: Optional[str] = None,
    ) -> None:
        if version not in (117, 118, 119):
            raise ValueError('Only dta versions 117, 118 and 119 supported')
        self._dta_ver: int = version
        self.df: DataFrame = df
        self.columns: List[str] = columns
        self._gso_table: Dict[str, Tuple[int, int]] = {'': (0, 0)}
        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder: str = _set_endianness(byteorder)
        self._native_byteorder: bool = self._byteorder == _set_endianness(sys.byteorder)
        gso_v_type: str = 'I'
        gso_o_type: str = 'Q'
        self._encoding: str = 'utf-8'
        if version == 117:
            o_size: int = 4
            gso_o_type = 'I'
            self._encoding = 'latin-1'
        elif version == 118:
            o_size = 6
        else:
            o_size = 5
        if self._native_byteorder:
            self._o_offet: int = 2 ** (8 * (8 - o_size))
        else:
            self._o_offet = 2 ** (8 * o_size)
        self._gso_o_type: str = gso_o_type
        self._gso_v_type: str = gso_v_type

    def _convert_key(self, key: Tuple[int, int]) -> int:
        v, o = key
        if self._native_byteorder:
            return v + self._o_offet * o
        else:
            return o + self._o_offet * v

    def generate_table(
        self,
    ) -> Tuple[Dict[str, Tuple[int, int]], DataFrame]:
        """
        Generates the GSO lookup table for the DataFrame

        Returns
        -------
        gso_table : dict
            Ordered dictionary using the string found as keys
            and their lookup position (v,o) as values
        gso_df : DataFrame
            DataFrame where strl columns have been converted to
            (v,o) values

        Notes
        -----
        Modifies the DataFrame in-place.

        The DataFrame returned encodes the (v,o) values as uint64s. The
        encoding depends on the dta version, and can be expressed as

        enc = v + o * 2 ** (o_size * 8)

        so that v is stored in the lower bits and o is in the upper
        bits. o_size is

          * 117: 4
          * 118: 6
          * 119: 5
        """
        gso_table = self._gso_table
        gso_df = self.df
        columns = list(gso_df.columns)
        selected = gso_df[self.columns]
        col_index: List[Tuple[str, int]] = [(col, columns.index(col)) for col in self.columns]
        keys: np.ndarray = np.empty(selected.shape, dtype=np.uint64)
        for o, (_, row) in enumerate(selected.iterrows()):
            for j, (col, v) in enumerate(col_index):
                val = row[col]
                val = '' if val is None else val
                key = gso_table.get(val, None)
                if key is None:
                    key = (v + 1, o + 1)
                    gso_table[val] = key
                keys[o, j] = self._convert_key(key)
        for i, col in enumerate(self.columns):
            gso_df[col] = keys[:, i]
        return (gso_table, gso_df)

    def generate_blob(
        self,
        gso_table: Dict[str, Tuple[int, int]],
    ) -> bytes:
        """
        Generates the binary blob of GSOs that is written to the dta file.

        Parameters
        ----------
        gso_table : dict
            Ordered dictionary (str, vo)

        Returns
        -------
        gso : bytes
            Binary content of dta file to be placed between strl tags

        Notes
        -----
        Output format depends on dta version.  117 uses two uint32s to
        express v and o while 118+ uses a uint32 for v and a uint64 for o.
        """
        bio = BytesIO()
        gso = bytes('GSO', 'ascii')
        gso_type = struct.pack(self._byteorder + 'B', 130)
        null = struct.pack(self._byteorder + 'B', 0)
        v_type = self._byteorder + self._gso_v_type
        o_type = self._byteorder + self._gso_o_type
        len_type = self._byteorder + 'I'
        for strl, vo in gso_table.items():
            if vo == (0, 0):
                continue
            v, o = vo
            bio.write(gso)
            bio.write(struct.pack(v_type, v))
            bio.write(struct.pack(o_type, o))
            bio.write(gso_type)
            utf8_string = bytes(strl, 'utf-8')
            bio.write(struct.pack(len_type, len(utf8_string) + 1))
            bio.write(utf8_string)
            bio.write(null)
        return bio.getvalue()


class StataWriter117(StataWriter):
    """
    A class for writing Stata binary dta files in Stata 13 format (117)

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, pathlib.Path or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter117 instance
        The StataWriter117 instance has a write_file method, which will
        write the file to the given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1, "a"]], columns=["a", "b", "c"])
    >>> writer = pd.io.stata.StataWriter117("./data_file.dta", data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = pd.io.stata.StataWriter117(
    ...     "./data_file.zip", data, compression=compression
    ... )
    >>> writer.write_file()

    Or with long strings stored in strl format
    >>> data = pd.DataFrame(
    ...     [["A relatively long string"], [""], [""]], columns=["strls"]
    ... )
    >>> writer = pd.io.stata.StataWriter117(
    ...     "./data_file_with_long_strings.dta", data, convert_strl=["strls"]
    ... )
    >>> writer.write_file()
    """
    _max_string_length: Final[int] = 2045
    _dta_version: Final[int] = 117

    def __init__(
        self,
        fname: Union[str, os.PathLike, IO[bytes]],
        data: DataFrame,
        convert_dates: Optional[Dict[Union[str, int], str]] = None,
        write_index: bool = True,
        byteorder: Optional[str] = None,
        time_stamp: Optional[datetime] = None,
        data_label: Optional[str] = None,
        variable_labels: Optional[Dict[str, str]] = None,
        convert_strl: Optional[List[str]] = None,
        compression: str = 'infer',
        storage_options: Optional[Dict[str, Any]] = None,
        *,
        value_labels: Optional[Dict[str, Dict[Union[int, float], str]]] = None,
    ) -> None:
        self._convert_strl: List[str] = []
        if convert_strl is not None:
            self._convert_strl.extend(convert_strl)
        super().__init__(
            fname,
            data,
            convert_dates=convert_dates,
            write_index=write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            value_labels=value_labels,
            compression=compression,
            storage_options=storage_options,
        )
        self._map: Dict[str, int] = {}
        self._strl_blob: bytes = b''

    @staticmethod
    def _tag(val: bytes, tag: str) -> bytes:
        """Surround val with <tag></tag>"""
        if isinstance(val, str):
            val = bytes(val, 'utf-8')
        return bytes(f'<{tag}>', 'utf-8') + val + bytes(f'</{tag}>', 'utf-8')

    def _update_map(self, tag: str) -> None:
        """Update map location for tag with file position"""
        assert self.handles.handle is not None
        self._map[tag] = self.handles.handle.tell()

    def _write_header(self, data_label: Optional[str], time_stamp: Optional[datetime]) -> None:
        """Write the file header"""
        byteorder = self._byteorder
        self._write_bytes(bytes('<stata_dta>', 'utf-8'))
        bio = BytesIO()
        bio.write(self._tag(bytes(str(self._dta_version), 'utf-8'), 'release'))
        bio.write(self._tag(b'MSF' if byteorder == '>' else b'LSF', 'byteorder'))
        nvar_type: str = 'H' if self._dta_version <= 118 else 'I'
        bio.write(self._tag(struct.pack(byteorder + nvar_type, self.nvar), 'K'))
        nobs_size: str = 'I' if self._dta_version == 117 else 'Q'
        bio.write(self._tag(struct.pack(byteorder + nobs_size, self.nobs), 'N'))
        if data_label is None:
            encoded_label: bytes = _pad_bytes_new(b'', 80 + 1)
        else:
            encoded_label = _pad_bytes_new(data_label[:80].encode(self._encoding), 80 + 1)
        bio.write(self._tag(encoded_label, 'label'))
        if time_stamp is None:
            time_stamp = datetime.now()
        elif not isinstance(time_stamp, datetime):
            raise ValueError('time_stamp should be datetime type')
        months: List[str] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_lookup: Dict[int, str] = {i + 1: month for i, month in enumerate(months)}
        ts = time_stamp.strftime('%d ') + month_lookup[time_stamp.month] + time_stamp.strftime(' %Y %H:%M')
        stata_ts: bytes = b'\x11' + bytes(ts, 'utf-8')
        bio.write(self._tag(stata_ts, 'timestamp'))
        self._write_bytes(self._tag(bio.getvalue(), 'header'))

    def _write_variable_types(self) -> None:
        self._update_map('variable_types')
        bio = BytesIO()
        for typ in self.typlist:
            bio.write(struct.pack(self._byteorder + 'H', typ))
        self._write_bytes(self._tag(bio.getvalue(), 'variable_types'))

    def _write_varnames(self) -> None:
        self._update_map('varnames')
        bio = BytesIO()
        vn_len: int = 32 if self._dta_version == 117 else 128
        for name in self.varlist:
            name_padded: bytes = self._null_terminate_str(name).encode(self._encoding)[:32]
            name_padded = _pad_bytes_new(name_padded, vn_len + 1)
            bio.write(name_padded)
        self._write_bytes(self._tag(bio.getvalue(), 'varnames'))

    def _write_sortlist(self) -> None:
        self._update_map('sortlist')
        sort_size: int = 2 if self._dta_version < 119 else 4
        sortlist = b'\x00' * sort_size * (self.nvar + 1)
        self._write_bytes(self._tag(sortlist, 'sortlist'))

    def _write_formats(self) -> None:
        self._update_map('formats')
        bio = BytesIO()
        fmt_len: int = 49 if self._dta_version == 117 else 57
        for fmt in self.fmtlist:
            bio.write(_pad_bytes_new(fmt.encode(self._encoding), fmt_len))
        self._write_bytes(self._tag(bio.getvalue(), 'formats'))

    def _write_value_label_names(self) -> None:
        self._update_map('value_label_names')
        bio = BytesIO()
        vl_len: int = 32 if self._dta_version == 117 else 128
        for i in range(self.nvar):
            if self._has_value_labels[i]:
                name: str = self.varlist[i]
            else:
                name = ''
            name_encoded: bytes = self._null_terminate_str(name).encode(self._encoding)[:32]
            encoded_name: bytes = _pad_bytes_new(name_encoded, vl_len + 1)
            bio.write(encoded_name)
        self._write_bytes(self._tag(bio.getvalue(), 'value_label_names'))

    def _write_variable_labels(self) -> None:
        self._update_map('variable_labels')
        bio = BytesIO()
        vl_len: int = 80 if self._dta_version == 117 else 320
        blank: bytes = _pad_bytes_new(b'', vl_len + 1)
        if self._variable_labels is None:
            for _ in range(self.nvar):
                bio.write(blank)
            self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))
            return
        for col in self.data:
            if col in self._variable_labels:
                label: str = self._variable_labels[col]
                if len(label) > 80:
                    raise ValueError('Variable labels must be 80 characters or fewer')
                try:
                    encoded_label: bytes = label.encode(self._encoding)
                except UnicodeEncodeError as err:
                    raise ValueError(f'Variable labels must contain only characters that can be encoded in {self._encoding}') from err
                encoded_label_padded: bytes = _pad_bytes_new(encoded_label, vl_len + 1)
                bio.write(encoded_label_padded)
            else:
                bio.write(blank)
        self._write_bytes(self._tag(bio.getvalue(), 'variable_labels'))

    def _write_characteristics(self) -> None:
        """No-op in dta 117+"""
        self._update_map('characteristics')
        self._write_bytes(self._tag(b'', 'characteristics'))

    def _write_data(self, records: np.recarray) -> None:
        self._update_map('data')
        self._write_bytes(b'<data>')
        self._write_bytes(records.tobytes())
        self._write_bytes(b'</data>')

    @staticmethod
    def _null_terminate_str(s: str) -> str:
        s += '\x00'
        return s

    def _null_terminate_bytes(self, s: str) -> bytes:
        return self._null_terminate_str(s).encode(self._encoding)


def _dtype_to_stata_type_117(
    dtype: np.dtype,
    column: Series,
    force_strl: bool = False,
) -> Union[int, str]:
    """
    Converts dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    1 - 2045 are strings of this length
                Pandas    Stata
    32768 - for object    strL
    65526 - for int8      byte
    65527 - for int16     int
    65528 - for int32     long
    65529 - for float32   float
    65530 - for double    double

    If there are dates to convert, then dtype will already have the correct
    type inserted.
    """
    if force_strl:
        return 32768
    if dtype.type is np.object_:
        itemsize = max_len_string_array(ensure_object(column._values))
        itemsize = max(itemsize, 1)
        if itemsize <= 2045:
            return itemsize
        return 32768
    elif dtype.type is np.float64:
        return 65526
    elif dtype.type is np.float32:
        return 65527
    elif dtype.type is np.int32:
        return 65528
    elif dtype.type is np.int16:
        return 65529
    elif dtype.type is np.int8:
        return 65530
    else:
        raise NotImplementedError(f'Data type {dtype} not supported.')


def _pad_bytes_new(name: bytes, length: int) -> bytes:
    """
    Takes a bytes instance and pads it with null bytes until it's length chars.
    """
    return name + b'\x00' * (length - len(name))


class StataStrLWriter:
    """
    Converter for Stata StrLs

    Stata StrLs map 8 byte values to strings which are stored using a
    dictionary-like format where strings are keyed to two values.

    Parameters
    ----------
    df : DataFrame
        DataFrame to convert
    columns : List[str]
        List of columns names to convert to StrL
    version : int, optional
        dta version.  Currently supports 117, 118 and 119
    byteorder : str, optional
        Can be ">", "<", "little", or "big". default is `sys.byteorder`

    Notes
    -----
    Supports creation of the StrL block of a dta file for dta versions
    117, 118 and 119.  These differ in how the GSO is stored.  118 and
    119 store the GSO lookup value as a uint32 and a uint64, while 117
    uses two uint32s. 118 and 119 also encode all strings as unicode
    which is required by the format.  117 uses 'latin-1' a fixed width
    encoding that extends the 7-bit ascii table with an additional 128
    characters.
    """

    def __init__(
        self,
        df: DataFrame,
        columns: List[str],
        version: int = 117,
        byteorder: Optional[str] = None,
    ) -> None:
        if version not in (117, 118, 119):
            raise ValueError('Only dta versions 117, 118 and 119 supported')
        self._dta_ver: int = version
        self.df: DataFrame = df
        self.columns: List[str] = columns
        self._gso_table: Dict[str, Tuple[int, int]] = {'': (0, 0)}
        if byteorder is None:
            byteorder = sys.byteorder
        self._byteorder: str = _set_endianness(byteorder)
        self._native_byteorder: bool = self._byteorder == _set_endianness(sys.byteorder)
        gso_v_type: str = 'I'
        gso_o_type: str = 'Q'
        self._encoding: str = 'utf-8'
        if version == 117:
            o_size: int = 4
            gso_o_type = 'I'
            self._encoding = 'latin-1'
        elif version == 118:
            o_size = 6
        else:
            o_size = 5
        if self._native_byteorder:
            self._o_offet: int = 2 ** (8 * (8 - o_size))
        else:
            self._o_offet = 2 ** (8 * o_size)
        self._gso_o_type: str = gso_o_type
        self._gso_v_type: str = gso_v_type

    def _convert_key(
        self,
        key: Tuple[int, int],
    ) -> int:
        v, o = key
        if self._native_byteorder:
            return v + self._o_offet * o
        else:
            return o + self._o_offet * v

    def generate_table(
        self,
    ) -> Tuple[Dict[str, Tuple[int, int]], DataFrame]:
        """
        Generates the GSO lookup table for the DataFrame

        Returns
        -------
        gso_table : dict
            Ordered dictionary using the string found as keys
            and their lookup position (v,o) as values
        gso_df : DataFrame
            DataFrame where strl columns have been converted to
            (v,o) values

        Notes
        -----
        Modifies the DataFrame in-place.

        The DataFrame returned encodes the (v,o) values as uint64s. The
        encoding depends on the dta version, and can be expressed as

        enc = v + o * 2 ** (o_size * 8)

        so that v is stored in the lower bits and o is in the upper
        bits. o_size is

          * 117: 4
          * 118: 6
          * 119: 5
        """
        gso_table = self._gso_table
        gso_df = self.df
        columns = list(gso_df.columns)
        selected = gso_df[self.columns]
        col_index: List[Tuple[str, int]] = [(col, columns.index(col)) for col in self.columns]
        keys: np.ndarray = np.empty(selected.shape, dtype=np.uint64)
        for o, (_, row) in enumerate(selected.iterrows()):
            for j, (col, v) in enumerate(col_index):
                val = row[col]
                val = '' if val is None else val
                key = gso_table.get(val, None)
                if key is None:
                    key = (v + 1, o + 1)
                    gso_table[val] = key
                keys[o, j] = self._convert_key(key)
        for i, col in enumerate(self.columns):
            gso_df[col] = keys[:, i]
        return (gso_table, gso_df)

    def generate_blob(
        self,
        gso_table: Dict[str, Tuple[int, int]],
    ) -> bytes:
        """
        Generates the binary blob of GSOs that is written to the dta file.

        Parameters
        ----------
        gso_table : dict
            Ordered dictionary (str, vo)

        Returns
        -------
        gso : bytes
            Binary content of dta file to be placed between strl tags

        Notes
        -----
        Output format depends on dta version.  117 uses two uint32s to
        express v and o while 118+ uses a uint32 for v and a uint64 for o.
        """
        bio = BytesIO()
        gso = bytes('GSO', 'ascii')
        gso_type = struct.pack(self._byteorder + 'B', 130)
        null = struct.pack(self._byteorder + 'B', 0)
        v_type = self._byteorder + self._gso_v_type
        o_type = self._byteorder + self._gso_o_type
        len_type = self._byteorder + 'I'
        for strl, vo in gso_table.items():
            if vo == (0, 0):
                continue
            v, o = vo
            bio.write(gso)
            bio.write(struct.pack(v_type, v))
            bio.write(struct.pack(o_type, o))
            bio.write(gso_type)
            utf8_string: bytes = bytes(strl, 'utf-8')
            bio.write(struct.pack(len_type, len(utf8_string) + 1))
            bio.write(utf8_string)
            bio.write(null)
        return bio.getvalue()


class StataWriterUTF8(StataWriter117):
    """
    Stata binary dta file writing in Stata 15 (118) and 16 (119) formats

    DTA 118 and 119 format files support unicode string data (both fixed
    and strL) format. Unicode is also supported in value labels, variable
    labels and the dataset label. Format 119 is automatically used if the
    file contains more than 32,767 variables.

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, pathlib.Path or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict, default None
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool, default True
        Write the index to Stata dataset.
    byteorder : str, default None
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime, default None
        A datetime to use as file creation date.  Default is the current time
    data_label : str, default None
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict, default None
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    convert_strl : list, default None
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
    version : int, default None
        The dta version to use. By default, uses the size of data to determine
        the version. 118 is used if data.shape[1] <= 32767, and 119 is used
        for storing larger DataFrames.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.

        .. versionadded:: 1.4.0

    Returns
    -------
    StataWriterUTF8
        The instance has a write_file method, which will write the file to the
        given `fname`.

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters

    Examples
    --------
    Using Unicode data and column names

    >>> from pandas.io.stata import StataWriterUTF8
    >>> data = pd.DataFrame([[1.0, 1, "ᴬ"]], columns=["a", "β", "ĉ"])
    >>> writer = StataWriterUTF8("./data_file.dta", data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = StataWriterUTF8("./data_file.zip", data, compression=compression)
    >>> writer.write_file()

    Or with long strings stored in strl format

    >>> data = pd.DataFrame(
    ...     [["ᴀ relatively long ŝtring"], [""], [""]], columns=["strls"]
    ... )
    >>> writer = StataWriterUTF8(
    ...     "./data_file_with_long_strings.dta", data, convert_strl=["strls"]
    ... )
    >>> writer.write_file()
    """
    _encoding: Final[str] = 'utf-8'

    def __init__(
        self,
        fname: Union[str, os.PathLike, IO[bytes]],
        data: DataFrame,
        convert_dates: Optional[Dict[Union[str, int], str]] = None,
        write_index: bool = True,
        byteorder: Optional[str] = None,
        time_stamp: Optional[datetime] = None,
        data_label: Optional[str] = None,
        variable_labels: Optional[Dict[str, str]] = None,
        convert_strl: Optional[List[str]] = None,
        version: Optional[int] = None,
        compression: str = 'infer',
        storage_options: Optional[Dict[str, Any]] = None,
        *,
        value_labels: Optional[Dict[str, Dict[Union[int, float], str]]] = None,
    ) -> None:
        if version is None:
            version = 118 if data.shape[1] <= 32767 else 119
        elif version not in (118, 119):
            raise ValueError('version must be either 118 or 119.')
        elif version == 118 and data.shape[1] > 32767:
            raise ValueError('You must use version 119 for data sets containing more than32,767 variables')
        super().__init__(
            fname,
            data,
            convert_dates=convert_dates,
            write_index=write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            value_labels=value_labels,
            convert_strl=convert_strl,
            compression=compression,
            storage_options=storage_options,
        )
        self._dta_version: int = version

    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 118+ support most unicode characters. The only limitation is in
        the ascii range where the characters supported are a-z, A-Z, 0-9 and _.
        """
        for c in name:
            if (ord(c) < 128 and (
                (c < 'A' or c > 'Z') and
                (c < 'a' or c > 'z') and
                (c < '0' or c > '9') and
                (c != '_')
            )) or 128 <= ord(c) < 192 or c in {'×', '÷'}:
                name = name.replace(c, '_')
        return name
