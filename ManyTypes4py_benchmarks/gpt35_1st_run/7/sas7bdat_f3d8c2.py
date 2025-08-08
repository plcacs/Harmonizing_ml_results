from __future__ import annotations
from datetime import datetime
import sys
from typing import TYPE_CHECKING
import numpy as np
from pandas._config import get_option
from pandas._libs.byteswap import read_double_with_byteswap, read_float_with_byteswap, read_uint16_with_byteswap, read_uint32_with_byteswap, read_uint64_with_byteswap
from pandas._libs.sas import Parser, get_subheader_index
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import DataFrame, Timestamp
from pandas.io.common import get_handle
import pandas.io.sas.sas_constants as const
from pandas.io.sas.sasreader import SASReader

if TYPE_CHECKING:
    from pandas._typing import CompressionOptions, FilePath, ReadBuffer

_unix_origin = Timestamp('1970-01-01')
_sas_origin = Timestamp('1960-01-01')

def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    ...

class _Column:
    def __init__(self, col_id: int, name: str, label: str, format: str, ctype: bytes, length: int):
        ...

class SAS7BDATReader(SASReader):
    def __init__(self, path_or_buf: FilePath, index: str = None, convert_dates: bool = True, blank_missing: bool = True, chunksize: int = None, encoding: str = None, convert_text: bool = True, convert_header_text: bool = True, compression: str = 'infer'):
        ...

    def column_data_lengths(self) -> np.ndarray:
        ...

    def column_data_offsets(self) -> np.ndarray:
        ...

    def column_types(self) -> np.ndarray:
        ...

    def close(self):
        ...

    def _get_properties(self):
        ...

    def __next__(self) -> pd.DataFrame:
        ...

    def _read_float(self, offset: int, width: int) -> float:
        ...

    def _read_uint(self, offset: int, width: int) -> int:
        ...

    def _read_bytes(self, offset: int, length: int) -> bytes:
        ...

    def _parse_metadata(self):
        ...

    def _process_page_meta(self) -> bool:
        ...

    def _read_page_header(self):
        ...

    def _process_page_metadata(self):
        ...

    def _process_rowsize_subheader(self, offset: int, length: int):
        ...

    def _process_columnsize_subheader(self, offset: int, length: int):
        ...

    def _process_subheader_counts(self, offset: int, length: int):
        ...

    def _process_columntext_subheader(self, offset: int, length: int):
        ...

    def _process_columnname_subheader(self, offset: int, length: int):
        ...

    def _process_columnattributes_subheader(self, offset: int, length: int):
        ...

    def _process_columnlist_subheader(self, offset: int, length: int):
        ...

    def _process_format_subheader(self, offset: int, length: int):
        ...

    def read(self, nrows: int = None) -> pd.DataFrame:
        ...

    def _read_next_page(self) -> bool:
        ...

    def _chunk_to_dataframe(self) -> pd.DataFrame:
        ...

    def _decode_string(self, b: pd.Series) -> str:
        ...

    def _convert_header_text(self, b: bytes) -> str:
        ...
