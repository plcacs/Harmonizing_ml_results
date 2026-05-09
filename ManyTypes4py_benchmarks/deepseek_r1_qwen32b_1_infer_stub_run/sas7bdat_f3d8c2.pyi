"""
Read SAS7BDAT files
"""

from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    List,
    Optional,
    Union,
    Tuple,
    Dict,
    Sequence,
    overload,
)
import numpy as np
from pandas import DataFrame, Timestamp
from pandas._typing import CompressionOptions
from pandas.io.common import ReadBuffer

_unix_origin: Timestamp = ...
_sas_origin: Timestamp = ...

class _Column:
    def __init__(self, col_id: int, name: str, label: str, format: str, ctype: str, length: int) -> None:
        ...

class SAS7BDATReader:
    def __init__(
        self,
        path_or_buf: Union[str, bytes, ReadBuffer],
        index: Optional[str] = None,
        convert_dates: bool = True,
        blank_missing: bool = True,
        chunksize: Optional[int] = None,
        encoding: Optional[str] = None,
        convert_text: bool = True,
        convert_header_text: bool = True,
        compression: CompressionOptions = 'infer',
    ) -> None:
        ...

    @property
    def index(self) -> Optional[str]:
        ...

    @property
    def convert_dates(self) -> bool:
        ...

    @property
    def blank_missing(self) -> bool:
        ...

    @property
    def chunksize(self) -> Optional[int]:
        ...

    @property
    def encoding(self) -> Optional[str]:
        ...

    @property
    def convert_text(self) -> bool:
        ...

    @property
    def convert_header_text(self) -> bool:
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

    def read(self, nrows: Optional[int] = None) -> DataFrame:
        ...

    def _read_float(self, offset: int, width: int) -> float:
        ...

    def _read_uint(self, offset: int, width: int) -> int:
        ...

    def _read_bytes(self, offset: int, length: int) -> bytes:
        ...

    def _process_page_meta(self) -> bool:
        ...

    def _chunk_to_dataframe(self) -> DataFrame:
        ...

    def _decode_string(self, b: bytes) -> str:
        ...

    def _convert_header_text(self, b: bytes) -> str:
        ...

def _convert_datetimes(sas_datetimes: Series, unit: str) -> Series:
    ...