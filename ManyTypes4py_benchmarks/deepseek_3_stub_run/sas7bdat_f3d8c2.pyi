from __future__ import annotations
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Iterator,
    Optional,
    Sequence,
    Union,
)
import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Timestamp
from pandas.io.sas.sasreader import SASReader

if TYPE_CHECKING:
    from pandas._typing import CompressionOptions, FilePath, ReadBuffer

_unix_origin: Timestamp = ...
_sas_origin: Timestamp = ...

def _convert_datetimes(
    sas_datetimes: Union[Series, Sequence[float]],
    unit: str
) -> Series:
    ...

class _Column:
    col_id: int
    name: str
    label: str
    format: str
    ctype: bytes
    length: int
    
    def __init__(
        self,
        col_id: int,
        name: str,
        label: str,
        format: str,
        ctype: bytes,
        length: int
    ) -> None:
        ...

class SAS7BDATReader(SASReader):
    index: Optional[Any]
    convert_dates: bool
    blank_missing: bool
    chunksize: Optional[int]
    encoding: Optional[str]
    convert_text: bool
    convert_header_text: bool
    default_encoding: str
    compression: bytes
    column_names_raw: list[bytes]
    column_names: list[str]
    column_formats: list[str]
    columns: list[_Column]
    _current_page_data_subheader_pointers: list[tuple[int, int]]
    _cached_page: Optional[bytes]
    _column_data_lengths: list[int]
    _column_data_offsets: list[int]
    _column_types: list[bytes]
    _current_row_in_file_index: int
    _current_row_on_page_index: int
    _current_row_in_chunk_index: int
    _path_or_buf: BinaryIO
    _subheader_processors: list[Optional[Any]]
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
    creator_proc: Optional[bytes]
    _string_chunk: np.ndarray
    _byte_chunk: np.ndarray
    
    def __init__(
        self,
        path_or_buf: Union[FilePath, ReadBuffer[bytes]],
        index: Optional[Any] = None,
        convert_dates: bool = True,
        blank_missing: bool = True,
        chunksize: Optional[int] = None,
        encoding: Optional[str] = None,
        convert_text: bool = True,
        convert_header_text: bool = True,
        compression: CompressionOptions = "infer"
    ) -> None:
        ...
    
    def column_data_lengths(self) -> np.ndarray:
        ...
    
    def column_data_offsets(self) -> np.ndarray:
        ...
    
    def column_types(self) -> np.ndarray:
        ...
    
    def close(self) -> None:
        ...
    
    def __next__(self) -> DataFrame:
        ...
    
    def __iter__(self) -> Iterator[DataFrame]:
        ...
    
    def read(self, nrows: Optional[int] = None) -> DataFrame:
        ...
    
    def _get_properties(self) -> None:
        ...
    
    def _read_float(self, offset: int, width: int) -> float:
        ...
    
    def _read_uint(self, offset: int, width: int) -> int:
        ...
    
    def _read_bytes(self, offset: int, length: int) -> bytes:
        ...
    
    def _parse_metadata(self) -> None:
        ...
    
    def _process_page_meta(self) -> bool:
        ...
    
    def _read_page_header(self) -> None:
        ...
    
    def _process_page_metadata(self) -> None:
        ...
    
    def _process_rowsize_subheader(self, offset: int, length: int) -> None:
        ...
    
    def _process_columnsize_subheader(self, offset: int, length: int) -> None:
        ...
    
    def _process_subheader_counts(self, offset: int, length: int) -> None:
        ...
    
    def _process_columntext_subheader(self, offset: int, length: int) -> None:
        ...
    
    def _process_columnname_subheader(self, offset: int, length: int) -> None:
        ...
    
    def _process_columnattributes_subheader(self, offset: int, length: int) -> None:
        ...
    
    def _process_columnlist_subheader(self, offset: int, length: int) -> None:
        ...
    
    def _process_format_subheader(self, offset: int, length: int) -> None:
        ...
    
    def _read_next_page(self) -> bool:
        ...
    
    def _chunk_to_dataframe(self) -> DataFrame:
        ...
    
    def _decode_string(self, b: Any) -> str:
        ...
    
    def _convert_header_text(self, b: bytes) -> Union[str, bytes]:
        ...