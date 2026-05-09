"""
Read SAS7BDAT files
"""

from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union, List, Any, Tuple, Dict, Iterable, Iterator, Callable, overload, Literal
import sys
import numpy as np
from pandas import DataFrame, Timestamp
from pandas._typing import CompressionOptions, FilePath, ReadBuffer
from pandas.io.sas.sasreader import SASReader

class _Column:
    def __init__(self, col_id: int, name: str, label: str, format: str, ctype: str, length: int) -> None:
        ...

class SAS7BDATReader(SASReader):
    """
    Read SAS files in SAS7BDAT format.
    """
    def __init__(self, path_or_buf: Union[FilePath, ReadBuffer], index: Optional[str] = None, convert_dates: bool = True, blank_missing: bool = True, chunksize: Optional[int] = None, encoding: Optional[Union[str, Literal['infer']]] = None, convert_text: bool = True, convert_header_text: bool = True, compression: CompressionOptions = 'infer') -> None:
        ...

    def column_data_lengths(self) -> np.ndarray:
        """
        Return a numpy int64 array of the column data lengths
        """
        ...

    def column_data_offsets(self) -> np.ndarray:
        """
        Return a numpy int64 array of the column offsets
        """
        ...

    def column_types(self) -> np.ndarray:
        """
        Returns a numpy character array of the column types:
           s (string) or d (double)
        """
        ...

    def close(self) -> None:
        """
        Close the reader and any associated resources
        """
        ...

    def __next__(self) -> DataFrame:
        ...

    def read(self, nrows: Optional[int] = None) -> DataFrame:
        """
        Read data from SAS7BDAT file
        """
        ...

    def _chunk_to_dataframe(self) -> DataFrame:
        ...

    def _decode_string(self, b: bytes) -> str:
        ...

    def _convert_header_text(self, b: bytes) -> Union[bytes, str]:
        ...