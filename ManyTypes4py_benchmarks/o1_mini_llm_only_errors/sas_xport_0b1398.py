"""
Read a SAS XPort format file into a Pandas DataFrame.

Based on code from Jack Cushman (github.com/jcushman/xport).

The file format is defined here:

https://support.sas.com/content/dam/SAS/support/en/technical-papers/record-layout-of-a-sas-version-5-or-6-data-set-in-sas-transport-xport-format.pdf
"""
from __future__ import annotations
from datetime import datetime
import struct
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, Optional
import warnings
import numpy as np
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas.io.common import get_handle
from pandas.io.sas.sasreader import SASReader

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        DatetimeNaTType,
        FilePath,
        ReadBuffer,
    )

_correct_line1: str = 'HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000  '
_correct_header1: str = 'HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000'
_correct_header2: str = 'HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!000000000000000000000000000000  '
_correct_obs_header: str = 'HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000  '
_fieldkeys: List[str] = [
    'ntype',
    'nhfun',
    'field_length',
    'nvar0',
    'name',
    'label',
    'nform',
    'nfl',
    'num_decimals',
    'nfj',
    'nfill',
    'niform',
    'nifl',
    'nifd',
    'npos',
    '_',
]
_base_params_doc: str = (
    'Parameters\n----------\n'
    'filepath_or_buffer : str or file-like object\n'
    '    Path to SAS file or object implementing binary read method.'
)
_params2_doc: str = (
    'index : identifier of index column\n'
    '    Identifier of column that should be used as index of the DataFrame.\n'
    'encoding : str\n'
    '    Encoding for text data.\n'
    'chunksize : int\n'
    '    Read file `chunksize` lines at a time, returns iterator.'
)
_format_params_doc: str = 'format : str\n    File format, only `xport` is currently supported.'
_iterator_doc: str = (
    'iterator : bool, default False\n'
    '    Return XportReader object for reading file incrementally.'
)
_read_sas_doc: str = (
    f"Read a SAS file into a DataFrame.\n\n{_base_params_doc}\n{_format_params_doc}\n{_params2_doc}\n"
    f"{_iterator_doc}\n\nReturns\n-------\nDataFrame or XportReader\n\nExamples\n--------\n"
    "Read a SAS Xport file:\n\n>>> df = pd.read_sas('filename.XPT')\n\n"
    "Read a Xport file in 10,000 line chunks:\n\n>>> itr = pd.read_sas('filename.XPT', chunksize=10000)\n"
    ">>> for chunk in itr:\n>>>     do_something(chunk)\n\n"
)
_xport_reader_doc: str = (
    'Class for reading SAS Xport files.\n\n'
    f"{_base_params_doc}\n{_params2_doc}\n\nAttributes\n----------\n"
    'member_info : list\n    Contains information about the file\n'
    'fields : list\n    Contains information about the variables in the file\n'
)
_read_method_doc: str = (
    'Read observations from SAS Xport file, returning as data frame.\n\n'
    'Parameters\n----------\n'
    'nrows : int\n'
    '    Number of rows to read from data file; if None, read whole\n'
    '    file.\n\nReturns\n-------\nA DataFrame.\n'
)


def _parse_date(datestr: str) -> Optional[datetime]:
    """Given a date in xport format, return Python date."""
    try:
        return datetime.strptime(datestr, '%d%b%y:%H:%M:%S')
    except ValueError:
        return pd.NaT


def _split_line(s: str, parts: List[Tuple[str, int]]) -> Dict[str, str]:
    """
    Parameters
    ----------
    s: str
        Fixed-length string to split
    parts: list of (name, length) pairs
        Used to break up string, name '_' will be filtered from output.

    Returns
    -------
    Dict of name:contents of string at given location.
    """
    out: Dict[str, str] = {}
    start: int = 0
    for name, length in parts:
        out[name] = s[start : start + length].strip()
        start += length
    del out['_']
    return out


def _handle_truncated_float_vec(vec: np.ndarray, nbytes: int) -> np.ndarray:
    if nbytes != 8:
        vec1: np.ndarray = np.zeros(len(vec), dtype='S8')
        dtype: np.dtype = np.dtype(f'S{nbytes},S{8 - nbytes}')
        vec2: np.ndarray = vec1.view(dtype=dtype)
        vec2['f0'] = vec
        return vec2
    return vec


def _parse_float_vec(vec: np.ndarray) -> np.ndarray:
    """
    Parse a vector of float values representing IBM 8 byte floats into
    native 8 byte floats.
    """
    dtype: np.dtype = np.dtype('>u4,>u4')
    vec1: np.ndarray = vec.view(dtype=dtype)
    xport1: np.ndarray = vec1['f0']
    xport2: np.ndarray = vec1['f1']
    ieee1: np.ndarray = xport1 & 16777215
    shift: np.ndarray = np.zeros(len(vec), dtype=np.uint8)
    shift[np.where(xport1 & 2097152)] = 1
    shift[np.where(xport1 & 4194304)] = 2
    shift[np.where(xport1 & 8388608)] = 3
    ieee1 >>= shift
    ieee2: np.ndarray = (xport2 >> shift) | ((xport1 & 7) << (29 + (3 - shift)))
    ieee1 &= 4293918719
    ieee1 |= (((xport1 >> 24) & 127) - 65 << 2) + shift + (1023 << 20) | (xport1 & 2147483648)
    ieee: np.ndarray = np.empty((len(ieee1),), dtype='>u4,>u4')
    ieee['f0'] = ieee1
    ieee['f1'] = ieee2
    ieee = ieee.view(dtype='>f8')
    ieee = ieee.astype('f8')
    return ieee


class XportReader(SASReader):
    __doc__: str = _xport_reader_doc

    def __init__(
        self,
        filepath_or_buffer: Union[str, "FilePath", "ReadBuffer[bytes]"],
        index: Optional[str] = None,
        encoding: str = 'ISO-8859-1',
        chunksize: Optional[int] = None,
        compression: Union[str, bool, None] = 'infer',
    ) -> None:
        self._encoding: str = encoding
        self._lines_read: int = 0
        self._index: Optional[str] = index
        self._chunksize: Optional[int] = chunksize
        self.handles = get_handle(
            filepath_or_buffer,
            'rb',
            encoding=encoding,
            is_text=False,
            compression=compression,
        )
        self.filepath_or_buffer = self.handles.handle
        try:
            self._read_header()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.handles.close()

    def _get_row(self) -> str:
        return self.filepath_or_buffer.read(80).decode()

    def _read_header(self) -> None:
        self.filepath_or_buffer.seek(0)
        line1: str = self._get_row()
        if line1 != _correct_line1:
            if '**COMPRESSED**' in line1:
                raise ValueError('Header record indicates a CPORT file, which is not readable.')
            raise ValueError('Header record is not an XPORT file.')
        line2: str = self._get_row()
        fif: List[Tuple[str, int]] = [
            ('prefix', 24),
            ('version', 8),
            ('OS', 8),
            ('_', 24),
            ('created', 16),
        ]
        file_info: Dict[str, str] = _split_line(line2, fif)
        if file_info['prefix'] != 'SAS     SAS     SASLIB':
            raise ValueError('Header record has invalid prefix.')
        file_info['created'] = _parse_date(file_info['created']).isoformat() if _parse_date(file_info['created']) else ''
        line3: str = self._get_row()
        file_info['modified'] = _parse_date(line3[:16]).isoformat() if _parse_date(line3[:16]) else ''
        header1: str = self._get_row()
        header2: str = self._get_row()
        headflag1: bool = header1.startswith(_correct_header1)
        headflag2: bool = header2 == _correct_header2
        if not (headflag1 and headflag2):
            raise ValueError('Member header not found')
        fieldnamelength: int = int(header1[-5:-2])
        mem: List[Tuple[str, int]] = [
            ('prefix', 8),
            ('set_name', 8),
            ('sasdata', 8),
            ('version', 8),
            ('OS', 8),
            ('_', 24),
            ('created', 16),
        ]
        member_info: Dict[str, str] = _split_line(self._get_row(), mem)
        mem = [
            ('modified', 16),
            ('_', 16),
            ('label', 40),
            ('type', 8),
        ]
        member_info.update(_split_line(self._get_row(), mem))
        member_info['modified'] = _parse_date(member_info['modified']).isoformat() if _parse_date(member_info['modified']) else ''
        member_info['created'] = _parse_date(member_info['created']).isoformat() if _parse_date(member_info['created']) else ''
        self.member_info: Dict[str, str] = member_info
        types: Dict[int, str] = {1: 'numeric', 2: 'char'}
        fieldcount: int = int(self._get_row()[54:58])
        datalength: int = fieldnamelength * fieldcount
        if datalength % 80:
            datalength += 80 - (datalength % 80)
        fielddata: bytes = self.filepath_or_buffer.read(datalength)
        fields: List[Dict[str, Any]] = []
        obs_length: int = 0
        while len(fielddata) >= fieldnamelength:
            fieldbytes, fielddata = fielddata[:fieldnamelength], fielddata[fieldnamelength:]
            fieldbytes = fieldbytes.ljust(140)
            fieldstruct: Tuple[Any, ...] = struct.unpack('>hhhh8s40s8shhh2s8shhl52s', fieldbytes)
            field: Dict[str, Any] = dict(zip(_fieldkeys, fieldstruct))
            del field['_']
            field['ntype'] = types[field['ntype']]
            fl: int = field['field_length']
            if field['ntype'] == 'numeric' and (fl < 2 or fl > 8):
                msg: str = f'Floating field width {fl} is not between 2 and 8.'
                raise TypeError(msg)
            for k, v in field.items():
                try:
                    field[k] = v.strip()
                except AttributeError:
                    pass
            obs_length += field['field_length']
            fields.append(field)
        header: str = self._get_row()
        if header != _correct_obs_header:
            raise ValueError('Observation header not found.')
        self.fields: List[Dict[str, Any]] = fields
        self.record_length: int = obs_length
        self.record_start: int = self.filepath_or_buffer.tell()
        self.nobs: int = self._record_count()
        self.columns: List[str] = [x['name'].decode() for x in self.fields]
        dtypel: List[Tuple[str, str]] = [
            (f's{i}', f'S{field["field_length"]}') for i, field in enumerate(self.fields)
        ]
        dtype: np.dtype = np.dtype(dtypel)
        self._dtype: np.dtype = dtype

    def __next__(self) -> pd.DataFrame:
        return self.read(nrows=self._chunksize or 1)

    def _record_count(self) -> int:
        """
        Get number of records in file.

        This is maybe suboptimal because we have to seek to the end of
        the file.

        Side effect: returns file position to record_start.
        """
        self.filepath_or_buffer.seek(0, 2)
        total_records_length: int = self.filepath_or_buffer.tell() - self.record_start
        if total_records_length % 80 != 0:
            warnings.warn('xport file may be corrupted.', stacklevel=find_stack_level())
        if self.record_length > 80:
            self.filepath_or_buffer.seek(self.record_start)
            return total_records_length // self.record_length
        self.filepath_or_buffer.seek(-80, 2)
        last_card_bytes: bytes = self.filepath_or_buffer.read(80)
        last_card: np.ndarray = np.frombuffer(last_card_bytes, dtype=np.uint64)
        ix: np.ndarray = np.flatnonzero(last_card == 2314885530818453536)
        if len(ix) == 0:
            tail_pad: int = 0
        else:
            tail_pad = 8 * len(ix)
        self.filepath_or_buffer.seek(self.record_start)
        return (total_records_length - tail_pad) // self.record_length

    def get_chunk(self, size: Optional[int] = None) -> pd.DataFrame:
        """
        Reads lines from Xport file and returns as dataframe

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

    def _missing_double(self, vec: np.ndarray) -> np.ndarray:
        v: np.ndarray = vec.view(dtype='u1,u1,u2,u4')
        miss: np.ndarray = (v['f1'] == 0) & (v['f2'] == 0) & (v['f3'] == 0)
        miss1: np.ndarray = ((v['f0'] >= 65) & (v['f0'] <= 90)) | (v['f0'] == 95) | (v['f0'] == 46)
        miss &= miss1
        return miss

    @Appender(_read_method_doc)
    def read(self, nrows: Optional[int] = None) -> pd.DataFrame:
        if nrows is None:
            nrows = self.nobs
        read_lines: int = min(nrows, self.nobs - self._lines_read)
        read_len: int = read_lines * self.record_length
        if read_len <= 0:
            self.close()
            raise StopIteration
        raw: bytes = self.filepath_or_buffer.read(read_len)
        data: np.ndarray = np.frombuffer(raw, dtype=self._dtype, count=read_lines)
        df_data: Dict[str, Union[List[float], List[str]]] = {}
        for j, x in enumerate(self.columns):
            vec: np.ndarray = data[f's{j}']
            ntype: str = self.fields[j]['ntype']
            if ntype == 'numeric':
                vec = _handle_truncated_float_vec(vec, self.fields[j]['field_length'])
                miss: np.ndarray = self._missing_double(vec)
                v: np.ndarray = _parse_float_vec(vec)
                v[miss] = np.nan
            elif self.fields[j]['ntype'] == 'char':
                v: List[str] = [y.rstrip().decode(self._encoding) if self._encoding else y.rstrip() for y in vec]
            df_data[x] = v
        df: pd.DataFrame = pd.DataFrame(df_data)
        if self._index is None:
            df.index = pd.Index(range(self._lines_read, self._lines_read + read_lines))
        else:
            df = df.set_index(self._index)
        self._lines_read += read_lines
        return df
