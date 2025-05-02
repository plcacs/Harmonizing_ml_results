from __future__ import annotations
from collections import abc
from datetime import datetime, timedelta
from io import BytesIO
import os
import struct
import sys
from typing import IO, TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union, cast
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
from pandas import Categorical, DatetimeIndex, NaT, Timestamp, isna, to_datetime
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator
    from types import TracebackType
    from typing import Literal
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
        Self,
        StorageOptions,
        WriteBuffer,
    )

_version_error: str = 'Version of given Stata file is {version}. pandas supports importing versions 102, 103, 104, 105, 108, 110 (Stata 7), 111 (Stata 7SE),  113 (Stata 8/9), 114 (Stata 10/11), 115 (Stata 12), 117 (Stata 13), 118 (Stata 14/15/16), and 119 (Stata 15/16, over 32,767 variables).'
_statafile_processing_params1: str = 'convert_dates : bool, default True\n    Convert date variables to DataFrame time values.\nconvert_categoricals : bool, default True\n    Read value labels and convert columns to Categorical/Factor variables.'
_statafile_processing_params2: str = 'index_col : str, optional\n    Column to set as index.\nconvert_missing : bool, default False\n    Flag indicating whether to convert missing values to their Stata\n    representations.  If False, missing values are replaced with nan.\n    If True, columns containing missing values are returned with\n    object data types and missing values are represented by\n    StataMissingValue objects.\npreserve_dtypes : bool, default True\n    Preserve Stata datatypes. If False, numeric data are upcast to pandas\n    default types for foreign data (float64 or int64).\ncolumns : list or None\n    Columns to retain.  Columns will be returned in the given order.  None\n    returns all columns.\norder_categoricals : bool, default True\n    Flag indicating whether converted categorical data are ordered.'
_chunksize_params: str = 'chunksize : int, default None\n    Return StataReader object for iterations, returns chunks with\n    given number of lines.'
_iterator_params: str = 'iterator : bool, default False\n    Return StataReader object.'
_reader_notes: str = 'Notes\n-----\nCategorical variables read through an iterator may not have the same\ncategories and dtype. This occurs when  a variable stored in a DTA\nfile is associated to an incomplete set of value labels that only\nlabel a strict subset of the values.'
_read_stata_doc: str = f"""\nRead Stata file into DataFrame.\n\nParameters\n----------\nfilepath_or_buffer : str, path object or file-like object\n    Any valid string path is acceptable. The string could be a URL. Valid\n    URL schemes include http, ftp, s3, and file. For file URLs, a host is\n    expected. A local file could be: ``file://localhost/path/to/table.dta``.\n\n    If you want to pass in a path object, pandas accepts any ``os.PathLike``.\n\n    By file-like object, we refer to objects with a ``read()`` method,\n    such as a file handle (e.g. via builtin ``open`` function)\n    or ``StringIO``.\n{_statafile_processing_params1}\n{_statafile_processing_params2}\n{_chunksize_params}\n{_iterator_params}\n{_shared_docs['decompression_options'] % 'filepath_or_buffer'}\n{_shared_docs['storage_options']}\n\nReturns\n-------\nDataFrame, pandas.api.typing.StataReader\n    If iterator or chunksize, returns StataReader, else DataFrame.\n\nSee Also\n--------\nio.stata.StataReader : Low-level reader for Stata data files.\nDataFrame.to_stata: Export Stata data files.\n\n{_reader_notes}\n\nExamples\n--------\n\nCreating a dummy stata for this example\n\n>>> df = pd.DataFrame({{'animal': ['falcon', 'parrot', 'falcon', 'parrot'],\n...                   'speed': [350, 18, 361, 15]}})  # doctest: +SKIP\n>>> df.to_stata('animals.dta')  # doctest: +SKIP\n\nRead a Stata dta file:\n\n>>> df = pd.read_stata('animals.dta')  # doctest: +SKIP\n\nRead a Stata dta file in 10,000 line chunks:\n\n>>> values = np.random.randint(0, 10, size=(20_000, 1), dtype="uint8")  # doctest: +SKIP\n>>> df = pd.DataFrame(values, columns=["i"])  # doctest: +SKIP\n>>> df.to_stata('filename.dta')  # doctest: +SKIP\n\n>>> with pd.read_stata('filename.dta', chunksize=10000) as itr:  # doctest: +SKIP\n>>>     for chunk in itr:\n...         # Operate on a single chunk, e.g., chunk.mean()\n...         pass  # doctest: +SKIP\n"""
_read_method_doc: str = f'Reads observations from Stata file, converting them into a dataframe\n\nParameters\n----------\nnrows : int\n    Number of lines to read from data file, if None read whole file.\n{_statafile_processing_params1}\n{_statafile_processing_params2}\n\nReturns\n-------\nDataFrame\n'
_stata_reader_doc: str = f'Class for reading Stata dta files.\n\nParameters\n----------\npath_or_buf : path (string), buffer or path object\n    string, pathlib.Path or object\n    implementing a binary read() functions.\n{_statafile_processing_params1}\n{_statafile_processing_params2}\n{_chunksize_params}\n{_shared_docs['decompression_options']}\n{_shared_docs['storage_options']}\n\n{_reader_notes}\n'
_date_formats: List[str] = ['%tc', '%tC', '%td', '%d', '%tw', '%tm', '%tq', '%th', '%ty']
stata_epoch: datetime = datetime(1960, 1, 1)
unix_epoch: datetime = datetime(1970, 1, 1)

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

def _datetime_to_stata_elapsed_vec(dates: Series, fmt: str) -> Series:
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
    US_PER_DAY: int = NS_PER_DAY // 1000
    MS_PER_DAY: int = NS_PER_DAY // 1000000

    def parse_dates_safe(dates: Series, delta: bool = False, year: bool = False, days: bool = False) -> DataFrame:
        d = {}
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

excessive_string_length_error: str = "\nFixed width strings in Stata .dta files are limited to 244 (or fewer)\ncharacters.  Column '{0}' does not satisfy this restriction. Use the\n'version=117' parameter to write the newer (Stata 13 and later) format.\n"
precision_loss_doc: str = '\nColumn converted from {0} to {1}, and some data are outside of the lossless\nconversion range. This may result in a loss of precision in the saved data.\n'
value_label_mism