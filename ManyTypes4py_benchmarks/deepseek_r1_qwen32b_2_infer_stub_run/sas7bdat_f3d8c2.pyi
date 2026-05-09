"""
Read SAS7BDAT files
"""

from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Hashable,
    overload,
)
import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp
from pandas._typing import (
    CompressionOptions,
    FilePath,
    ReadBuffer,
)
from pandas.io.sas.sasreader import SASReader

_unix_origin: Timestamp = ...
_sas_origin: Timestamp = ...

def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    ...

class _Column:
    def __init__(self, col_id: int, name: str, label: str, format: str, ctype: bytes, length: int) -> None:
        ...
    col_id: int
    name: str
    label: str
    format: str
    ctype: bytes
    length: int

class SAS7BDATReader(SASReader):
    def __init__(self, path_or_buf: Union[FilePath, ReadBuffer], index: Optional[Hashable] = None, convert_dates: bool = True, blank_missing: bool = True, chunksize: Optional[int] = None, encoding: Optional[str] = None, convert_text: bool = True, convert_header_text: bool = True, compression: CompressionOptions = 'infer') -> None:
        ...
    
    column_names_raw: List[bytes]
    column_names: List[str]
    column_formats: List[str]
    columns: List[_Column]
    row_length: int
    row_count: int
    column_count: int
    _column_data_lengths: List[int]
    _column_data_offsets: List[int]
    _column_types: List[bytes]
    
    def column_data_lengths(self) -> np.ndarray:
        ...
    
    def column_data_offsets(self) -> np.ndarray:
        ...
    
    def column_types(self) -> np.ndarray:
        ...
    
    def close(self) -> None:
        ...
    
    def read(self, nrows: Optional[int] = None) -> DataFrame:
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
    
    def _chunk_to_dataframe(self) -> DataFrame:
        ...
    
    def _decode_string(self, b: bytes) -> str:
        ...
    
    def _convert_header_text(self, b: bytes) -> Union[bytes, str]:
        ...