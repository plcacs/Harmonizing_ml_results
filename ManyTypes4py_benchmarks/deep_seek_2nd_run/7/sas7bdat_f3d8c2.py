"""
Read SAS7BDAT files

Based on code written by Jared Hobbs:
  https://bitbucket.org/jaredhobbs/sas7bdat

See also:
  https://github.com/BioStatMatt/sas7bdat

Partial documentation of the file format:
  https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf

Reference for binary data compression:
  http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
"""
from __future__ import annotations
from datetime import datetime
import sys
from typing import TYPE_CHECKING, Optional, Union, List, Tuple, Dict, Any, cast
import numpy as np
from pandas._config import get_option
from pandas._libs.byteswap import read_double_with_byteswap, read_float_with_byteswap, read_uint16_with_byteswap, read_uint32_with_byteswap, read_uint64_with_byteswap
from pandas._libs.sas import Parser, get_subheader_index
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import DataFrame, Timestamp, Series
from pandas.io.common import get_handle
import pandas.io.sas.sas_constants as const
from pandas.io.sas.sasreader import SASReader
if TYPE_CHECKING:
    from pandas._typing import CompressionOptions, FilePath, ReadBuffer
    from io import BufferedIOBase
    from collections.abc import Sequence

_unix_origin = Timestamp('1970-01-01')
_sas_origin = Timestamp('1960-01-01')

def _convert_datetimes(sas_datetimes: Series, unit: str) -> Series:
    """
    Convert to Timestamp if possible, otherwise to datetime.datetime.
    SAS float64 lacks precision for more than ms resolution so the fit
    to datetime.datetime is ok.

    Parameters
    ----------
    sas_datetimes : {Series, Sequence[float]}
       Dates or datetimes in SAS
    unit : {'d', 's'}
       "d" if the floats represent dates, "s" for datetimes

    Returns
    -------
    Series
       Series of datetime64 dtype or datetime.datetime.
    """
    td = (_sas_origin - _unix_origin).as_unit('s')
    if unit == 's':
        millis = cast_from_unit_vectorized(sas_datetimes._values, unit='s', out_unit='ms')
        dt64ms = millis.view('M8[ms]') + td
        return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
    else:
        vals = np.array(sas_datetimes, dtype='M8[D]') + td
        return pd.Series(vals, dtype='M8[s]', index=sas_datetimes.index, copy=False)

class _Column:
    def __init__(
        self,
        col_id: int,
        name: Union[str, bytes],
        label: Union[str, bytes],
        format: Union[str, bytes],
        ctype: bytes,
        length: int
    ) -> None:
        self.col_id = col_id
        self.name = name
        self.label = label
        self.format = format
        self.ctype = ctype
        self.length = length

class SAS7BDATReader(SASReader):
    """
    Read SAS files in SAS7BDAT format.

    Parameters
    ----------
    path_or_buf : path name or buffer
        Name of SAS file or file-like object pointing to SAS file
        contents.
    index : column identifier, defaults to None
        Column to use as index.
    convert_dates : bool, defaults to True
        Attempt to convert dates to Pandas datetime values.  Note that
        some rarely used SAS date formats may be unsupported.
    blank_missing : bool, defaults to True
        Convert empty strings to missing values (SAS uses blanks to
        indicate missing character variables).
    chunksize : int, defaults to None
        Return SAS7BDATReader object for iterations, returns chunks
        with given number of lines.
    encoding : str, 'infer', defaults to None
        String encoding acc. to Python standard encodings,
        encoding='infer' tries to detect the encoding from the file header,
        encoding=None will leave the data in binary format.
    convert_text : bool, defaults to True
        If False, text variables are left as raw bytes.
    convert_header_text : bool, defaults to True
        If False, header text, including column names, are left as raw
        bytes.
    """
    def __init__(
        self,
        path_or_buf: Union[FilePath, ReadBuffer[bytes]],
        index: Optional[str] = None,
        convert_dates: bool = True,
        blank_missing: bool = True,
        chunksize: Optional[int] = None,
        encoding: Optional[str] = None,
        convert_text: bool = True,
        convert_header_text: bool = True,
        compression: Union[CompressionOptions, str] = 'infer'
    ) -> None:
        self.index: Optional[str] = index
        self.convert_dates: bool = convert_dates
        self.blank_missing: bool = blank_missing
        self.chunksize: Optional[int] = chunksize
        self.encoding: Optional[str] = encoding
        self.convert_text: bool = convert_text
        self.convert_header_text: bool = convert_header_text
        self.default_encoding: str = 'latin-1'
        self.compression: bytes = b''
        self.column_names_raw: List[bytes] = []
        self.column_names: List[Union[str, bytes]] = []
        self.column_formats: List[Union[str, bytes]] = []
        self.columns: List[_Column] = []
        self._current_page_data_subheader_pointers: List[Tuple[int, int]] = []
        self._cached_page: Optional[bytes] = None
        self._column_data_lengths: List[int] = []
        self._column_data_offsets: List[int] = []
        self._column_types: List[bytes] = []
        self._current_row_in_file_index: int = 0
        self._current_row_on_page_index: int = 0
        self._current_row_in_file_index: int = 0
        self.handles = get_handle(path_or_buf, 'rb', is_text=False, compression=compression)
        self._path_or_buf: BufferedIOBase = self.handles.handle
        self._subheader_processors: List[Optional[Any]] = [
            self._process_rowsize_subheader,
            self._process_columnsize_subheader,
            self._process_subheader_counts,
            self._process_columntext_subheader,
            self._process_columnname_subheader,
            self._process_columnattributes_subheader,
            self._process_format_subheader,
            self._process_columnlist_subheader,
            None
        ]
        try:
            self._get_properties()
            self._parse_metadata()
        except Exception:
            self.close()
            raise

    def column_data_lengths(self) -> np.ndarray:
        """Return a numpy int64 array of the column data lengths"""
        return np.asarray(self._column_data_lengths, dtype=np.int64)

    def column_data_offsets(self) -> np.ndarray:
        """Return a numpy int64 array of the column offsets"""
        return np.asarray(self._column_data_offsets, dtype=np.int64)

    def column_types(self) -> np.ndarray:
        """
        Returns a numpy character array of the column types:
           s (string) or d (double)
        """
        return np.asarray(self._column_types, dtype=np.dtype('S1'))

    def close(self) -> None:
        self.handles.close()

    def _get_properties(self) -> None:
        self._path_or_buf.seek(0)
        self._cached_page = self._path_or_buf.read(288)
        if self._cached_page[0:len(const.magic)] != const.magic:
            raise ValueError('magic number mismatch (not a SAS file?)')
        buf = self._read_bytes(const.align_1_offset, const.align_1_length)
        if buf == const.u64_byte_checker_value:
            self.U64: bool = True
            self._int_length: int = 8
            self._page_bit_offset: int = const.page_bit_offset_x64
            self._subheader_pointer_length: int = const.subheader_pointer_length_x64
        else:
            self.U64 = False
            self._page_bit_offset = const.page_bit_offset_x86
            self._subheader_pointer_length = const.subheader_pointer_length_x86
            self._int_length = 4
        buf = self._read_bytes(const.align_2_offset, const.align_2_length)
        if buf == const.align_1_checker_value:
            align1: int = const.align_2_value
        else:
            align1 = 0
        buf = self._read_bytes(const.endianness_offset, const.endianness_length)
        if buf == b'\x01':
            self.byte_order: str = '<'
            self.need_byteswap: bool = sys.byteorder == 'big'
        else:
            self.byte_order = '>'
            self.need_byteswap = sys.byteorder == 'little'
        buf = self._read_bytes(const.encoding_offset, const.encoding_length)[0]
        if buf in const.encoding_names:
            self.inferred_encoding: str = const.encoding_names[buf]
            if self.encoding == 'infer':
                self.encoding = self.inferred_encoding
        else:
            self.inferred_encoding = f'unknown (code={buf})'
        epoch = datetime(1960, 1, 1)
        x = self._read_float(const.date_created_offset + align1, const.date_created_length)
        self.date_created: datetime = epoch + pd.to_timedelta(x, unit='s')
        x = self._read_float(const.date_modified_offset + align1, const.date_modified_length)
        self.date_modified: datetime = epoch + pd.to_timedelta(x, unit='s')
        self.header_length: int = self._read_uint(const.header_size_offset + align1, const.header_size_length)
        buf = self._path_or_buf.read(self.header_length - 288)
        self._cached_page += buf
        if len(self._cached_page) != self.header_length:
            raise ValueError('The SAS7BDAT file appears to be truncated.')
        self._page_length: int = self._read_uint(const.page_size_offset + align1, const.page_size_length)

    def __next__(self) -> DataFrame:
        da = self.read(nrows=self.chunksize or 1)
        if da.empty:
            self.close()
            raise StopIteration
        return da

    def _read_float(self, offset: int, width: int) -> float:
        assert self._cached_page is not None
        if width == 4:
            return read_float_with_byteswap(self._cached_page, offset, self.need_byteswap)
        elif width == 8:
            return read_double_with_byteswap(self._cached_page, offset, self.need_byteswap)
        else:
            self.close()
            raise ValueError('invalid float width')

    def _read_uint(self, offset: int, width: int) -> int:
        assert self._cached_page is not None
        if width == 1:
            return self._read_bytes(offset, 1)[0]
        elif width == 2:
            return read_uint16_with_byteswap(self._cached_page, offset, self.need_byteswap)
        elif width == 4:
            return read_uint32_with_byteswap(self._cached_page, offset, self.need_byteswap)
        elif width == 8:
            return read_uint64_with_byteswap(self._cached_page, offset, self.need_byteswap)
        else:
            self.close()
            raise ValueError('invalid int width')

    def _read_bytes(self, offset: int, length: int) -> bytes:
        assert self._cached_page is not None
        if offset + length > len(self._cached_page):
            self.close()
            raise ValueError('The cached page is too small.')
        return self._cached_page[offset:offset + length]

    def _parse_metadata(self) -> None:
        done = False
        while not done:
            self._cached_page = self._path_or_buf.read(self._page_length)
            if len(self._cached_page) <= 0:
                break
            if len(self._cached_page) != self._page_length:
                raise ValueError('Failed to read a meta data page from the SAS file.')
            done = self._process_page_meta()

    def _process_page_meta(self) -> bool:
        self._read_page_header()
        pt = const.page_meta_types + [const.page_amd_type, const.page_mix_type]
        if self._current_page_type in pt:
            self._process_page_metadata()
        is_data_page = self._current_page_type == const.page_data_type
        is_mix_page = self._current_page_type == const.page_mix_type
        return bool(is_data_page or is_mix_page or self._current_page_data_subheader_pointers != [])

    def _read_page_header(self) -> None:
        bit_offset = self._page_bit_offset
        tx = const.page_type_offset + bit_offset
        self._current_page_type: int = self._read_uint(tx, const.page_type_length) & const.page_type_mask2
        tx = const.block_count_offset + bit_offset
        self._current_page_block_count: int = self._read_uint(tx, const.block_count_length)
        tx = const.subheader_count_offset + bit_offset
        self._current_page_subheaders_count: int = self._read_uint(tx, const.subheader_count_length)

    def _process_page_metadata(self) -> None:
        bit_offset = self._page_bit_offset
        for i in range(self._current_page_subheaders_count):
            offset = const.subheader_pointers_offset + bit_offset
            total_offset = offset + self._subheader_pointer_length * i
            subheader_offset = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length
            subheader_length = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length
            subheader_compression = self._read_uint(total_offset, 1)
            total_offset += 1
            subheader_type = self._read_uint(total_offset, 1)
            if subheader_length == 0 or subheader_compression == const.truncated_subheader_id:
                continue
            subheader_signature = self._read_bytes(subheader_offset, self._int_length)
            subheader_index = get_subheader_index(subheader_signature)
            subheader_processor = self._subheader_processors[subheader_index]
            if subheader_processor is None:
                f1 = subheader_compression in (const.compressed_subheader_id, 0)
                f2 = subheader_type == const.compressed_subheader_type
                if self.compression and f1 and f2:
                    self._current_page_data_subheader_pointers.append((subheader_offset, subheader_length))
                else:
                    self.close()
                    raise ValueError(f'Unknown subheader signature {subheader_signature}')
            else:
                subheader_processor(subheader_offset, subheader_length)

    def _process_rowsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        lcs_offset = offset
        lcp_offset = offset
        if self.U64:
            lcs_offset += 682
            lcp_offset += 706
        else:
            lcs_offset += 354
            lcp_offset += 378
        self.row_length: int = self._read_uint(offset + const.row_length_offset_multiplier * int_len, int_len)
        self.row_count: int = self._read_uint(offset + const.row_count_offset_multiplier * int_len, int_len)
        self.col_count_p1: int = self._read_uint(offset + const.col_count_p1_multiplier * int_len, int_len)
        self.col_count_p2: int = self._read_uint(offset + const.col_count_p2_multiplier * int_len, int_len)
        mx = const.row_count_on_mix_page_offset_multiplier * int_len
        self._mix_page_row_count: int = self._read_uint(offset + mx, int_len)
        self._lcs: int = self._read_uint(lcs_offset, 2)
        self._lcp: int = self._read_uint(lcp_offset, 2)

    def _process_columnsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        offset += int_len
        self.column_count: int = self._read_uint(offset, int_len)
        if self.col_count_p1 + self.col_count_p2 != self.column_count:
            print(f'Warning: column count mismatch ({self.col_count_p1} + {self.col_count_p2} != {self.column_count})\n')

    def _process_subheader_counts(self, offset: int, length: int) -> None:
        pass

    def _process_columntext_subheader(self, offset: int, length: int) -> None:
        offset += self._int_length
        text_block_size = self._read_uint(offset, const.text_block_size_length)
        buf = self._read_bytes(offset, text_block_size)
        cname_raw = buf[0:text_block_size].rstrip(b'\x00 ')
        self.column_names_raw.append(cname_raw)
        if len(self.column_names_raw) == 1:
            compression_literal = b''
            for cl in const.compression_literals:
                if cl in cname_raw:
                    compression_literal = cl
            self.compression = compression_literal
            offset -= self._int_length
            offset1 = offset + 16
            if self.U64:
                offset1 += 4
            buf = self._read_bytes(offset1, self._lcp)
            compression_literal = buf.rstrip(b'\x00')
            if