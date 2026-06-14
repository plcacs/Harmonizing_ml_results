from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import pandas as pd
from pandas import DataFrame, Series, Timestamp

if TYPE_CHECKING:
    from pandas._typing import CompressionOptions, FilePath, ReadBuffer

from pandas.io.sas.sasreader import SASReader

_unix_origin: Timestamp
_sas_origin: Timestamp

def _convert_datetimes(sas_datetimes: Series, unit: str) -> Series: ...

class _Column:
    col_id: int
    name: str | bytes
    label: str | bytes
    format: str | bytes
    ctype: bytes
    length: int
    def __init__(
        self,
        col_id: int,
        name: str | bytes,
        label: str | bytes,
        format: str | bytes,
        ctype: bytes,
        length: int,
    ) -> None: ...

class SAS7BDATReader(SASReader):
    index: str | None
    convert_dates: bool
    blank_missing: bool
    chunksize: int | None
    encoding: str | None
    convert_text: bool
    convert_header_text: bool
    default_encoding: str
    compression: bytes
    column_names_raw: list[bytes]
    column_names: list[str | bytes]
    column_formats: list[str | bytes]
    columns: list[_Column]
    _current_page_data_subheader_pointers: list[tuple[int, int]]
    _cached_page: bytes | None
    _column_data_lengths: list[int]
    _column_data_offsets: list[int]
    _column_types: list[bytes]
    _current_row_in_file_index: int
    _current_row_on_page_index: int
    _current_row_in_chunk_index: int
    handles: object
    _path_or_buf: object
    _subheader_processors: list[object]
    U64: bool
    _int_length: int
    _page_bit_offset: int
    _subheader_pointer_length: int
    byte_order: str
    need_byteswap: bool
    inferred_encoding: str
    date_created: datetime
    date_modified: datetime
    header_length: int
    _page_length: int
    row_length: int
    row_count: int
    col_count_p1: int
    col_count_p2: int
    _mix_page_row_count: int
    _lcs: int
    _lcp: int
    column_count: int
    _current_page_type: int
    _current_page_block_count: int
    _current_page_subheaders_count: int
    creator_proc: str | bytes
    _string_chunk: np.ndarray
    _byte_chunk: np.ndarray

    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        index: str | None = ...,
        convert_dates: bool = ...,
        blank_missing: bool = ...,
        chunksize: int | None = ...,
        encoding: str | None = ...,
        convert_text: bool = ...,
        convert_header_text: bool = ...,
        compression: CompressionOptions = ...,
    ) -> None: ...
    def column_data_lengths(self) -> np.ndarray: ...
    def column_data_offsets(self) -> np.ndarray: ...
    def column_types(self) -> np.ndarray: ...
    def close(self) -> None: ...
    def _get_properties(self) -> None: ...
    def __next__(self) -> DataFrame: ...
    def _read_float(self, offset: int, width: int) -> float: ...
    def _read_uint(self, offset: int, width: int) -> int: ...
    def _read_bytes(self, offset: int, length: int) -> bytes: ...
    def _parse_metadata(self) -> None: ...
    def _process_page_meta(self) -> bool: ...
    def _read_page_header(self) -> None: ...
    def _process_page_metadata(self) -> None: ...
    def _process_rowsize_subheader(self, offset: int, length: int) -> None: ...
    def _process_columnsize_subheader(self, offset: int, length: int) -> None: ...
    def _process_subheader_counts(self, offset: int, length: int) -> None: ...
    def _process_columntext_subheader(self, offset: int, length: int) -> None: ...
    def _process_columnname_subheader(self, offset: int, length: int) -> None: ...
    def _process_columnattributes_subheader(self, offset: int, length: int) -> None: ...
    def _process_columnlist_subheader(self, offset: int, length: int) -> None: ...
    def _process_format_subheader(self, offset: int, length: int) -> None: ...
    def read(self, nrows: int | None = ...) -> DataFrame: ...
    def _read_next_page(self) -> bool: ...
    def _chunk_to_dataframe(self) -> DataFrame: ...
    def _decode_string(self, b: pd.core.strings.StringMethods | bytes) -> str | Series: ...
    def _convert_header_text(self, b: bytes) -> str | bytes: ...