```python
from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    BinaryIO,
    Iterator,
    Sequence,
    overload,
    TYPE_CHECKING
)
import numpy as np
import pandas as pd
from pandas import DataFrame, Timestamp
from pandas.io.sas.sasreader import SASReader

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer
    )

_unix_origin: Timestamp = ...
_sas_origin: Timestamp = ...

def _convert_datetimes(
    sas_datetimes: pd.Series | Sequence[float],
    unit: str
) -> pd.Series: ...

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
    ) -> None: ...

class SAS7BDATReader(SASReader):
    index: Any | None
    convert_dates: bool
    blank_missing: bool
    chunksize: int | None
    encoding: str | None
    convert_text: bool
    convert_header_text: bool
    default_encoding: str
    compression: bytes
    column_names_raw: list[bytes]
    column_names: list[str]
    column_formats: list[str]
    columns: list[_Column]
    U64: bool
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
    creator_proc: Any
    
    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        index: Any = None,
        convert_dates: bool = True,
        blank_missing: bool = True,
        chunksize: int | None = None,
        encoding: str | None = None,
        convert_text: bool = True,
        convert_header_text: bool = True,
        compression: CompressionOptions = "infer"
    ) -> None: ...
    
    def column_data_lengths(self) -> np.ndarray: ...
    
    def column_data_offsets(self) -> np.ndarray: ...
    
    def column_types(self) -> np.ndarray: ...
    
    def close(self) -> None: ...
    
    def __next__(self) -> DataFrame: ...
    
    def __iter__(self) -> Iterator[DataFrame]: ...
    
    def read(self, nrows: int | None = None) -> DataFrame: ...
```