```python
from __future__ import annotations
from collections.abc import Hashable, Iterable, Iterator, Sequence
import csv as csvlib
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    overload,
    cast,
)
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import (
    ABCDatetimeIndex,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
)
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        FloatFormatType,
        IndexLabel,
        StorageOptions,
        WriteBuffer,
        npt,
    )
    from pandas.io.formats.format import DataFrameFormatter

_DEFAULT_CHUNKSIZE_CELLS: int = ...

class CSVFormatter:
    fmt: DataFrameFormatter
    obj: Any
    filepath_or_buffer: str | WriteBuffer[bytes] | WriteBuffer[str]
    encoding: str | None
    compression: CompressionOptions
    mode: str
    storage_options: StorageOptions | None
    sep: str
    index_label: Any
    errors: str
    quoting: int
    quotechar: str | None
    doublequote: bool
    escapechar: str | None
    lineterminator: str
    date_format: str | None
    cols: Any
    chunksize: int
    writer: Any

    def __init__(
        self,
        formatter: DataFrameFormatter,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] = "",
        sep: str = ",",
        cols: Any = None,
        index_label: IndexLabel | None = None,
        mode: str = "w",
        encoding: str | None = None,
        errors: str = "strict",
        compression: CompressionOptions = "infer",
        quoting: int | None = None,
        lineterminator: str = "\n",
        chunksize: int | None = None,
        quotechar: str = '"',
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None: ...

    @property
    def na_rep(self) -> str: ...

    @property
    def float_format(self) -> FloatFormatType | None: ...

    @property
    def decimal(self) -> str: ...

    @property
    def header(self) -> bool | Sequence[Hashable]: ...

    @property
    def index(self) -> bool: ...

    def _initialize_index_label(self, index_label: Any) -> Any: ...

    def _get_index_label_from_obj(self) -> Any: ...

    def _get_index_label_multiindex(self) -> list[str]: ...

    def _get_index_label_flat(self) -> list[str]: ...

    def _initialize_quotechar(self, quotechar: str) -> str | None: ...

    @property
    def has_mi_columns(self) -> bool: ...

    def _initialize_columns(self, cols: Any) -> Any: ...

    def _initialize_chunksize(self, chunksize: int | None) -> int: ...

    @property
    def _number_format(self) -> dict[str, Any]: ...

    @cache_readonly
    def data_index(self) -> Any: ...

    @property
    def nlevels(self) -> int: ...

    @property
    def _has_aliases(self) -> bool: ...

    @property
    def _need_to_save_header(self) -> bool: ...

    @property
    def write_cols(self) -> SequenceNotStr[Hashable]: ...

    @property
    def encoded_labels(self) -> list[Any]: ...

    def save(self) -> None: ...

    def _save(self) -> None: ...

    def _save_header(self) -> None: ...

    def _generate_multiindex_header_rows(self) -> Iterator[list[Any]]: ...

    def _save_body(self) -> None: ...

    def _save_chunk(self, start_i: int, end_i: int) -> None: ...
```