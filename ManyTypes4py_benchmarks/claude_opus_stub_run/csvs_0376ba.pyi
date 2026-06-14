from __future__ import annotations

from collections.abc import Hashable, Sequence
import csv as _csvlib
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pandas._typing import SequenceNotStr
from pandas.core.indexes.api import Index

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        FloatFormatType,
        IndexLabel,
        StorageOptions,
        WriteBuffer,
    )
    from pandas.io.formats.format import DataFrameFormatter
    from pandas import DataFrame

_DEFAULT_CHUNKSIZE_CELLS: int = ...

class CSVFormatter:
    fmt: DataFrameFormatter
    obj: DataFrame
    filepath_or_buffer: FilePath | WriteBuffer[str] | WriteBuffer[bytes]
    encoding: str | None
    compression: CompressionOptions
    mode: str
    storage_options: StorageOptions
    sep: str
    index_label: Sequence[Hashable] | Literal[False]
    errors: str
    quoting: int
    quotechar: str | None
    doublequote: bool
    escapechar: str | None
    lineterminator: str
    date_format: str | None
    cols: np.ndarray
    chunksize: int
    writer: _csvlib.writer

    def __init__(
        self,
        formatter: DataFrameFormatter,
        path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] = ...,
        sep: str = ...,
        cols: Sequence[Hashable] | None = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        errors: str = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        lineterminator: str | None = ...,
        chunksize: int | None = ...,
        quotechar: str | None = ...,
        date_format: str | None = ...,
        doublequote: bool = ...,
        escapechar: str | None = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @property
    def na_rep(self) -> str: ...
    @property
    def float_format(self) -> FloatFormatType | None: ...
    @property
    def decimal(self) -> str: ...
    @property
    def header(self) -> bool | list[str]: ...
    @property
    def index(self) -> bool: ...
    def _initialize_index_label(
        self, index_label: IndexLabel | None
    ) -> Sequence[Hashable] | Literal[False]: ...
    def _get_index_label_from_obj(self) -> Sequence[Hashable]: ...
    def _get_index_label_multiindex(self) -> list[Hashable]: ...
    def _get_index_label_flat(self) -> list[Hashable]: ...
    def _initialize_quotechar(self, quotechar: str | None) -> str | None: ...
    @property
    def has_mi_columns(self) -> bool: ...
    def _initialize_columns(
        self, cols: Sequence[Hashable] | None
    ) -> np.ndarray: ...
    def _initialize_chunksize(self, chunksize: int | None) -> int: ...
    @property
    def _number_format(self) -> dict[str, Any]: ...
    @property
    def data_index(self) -> Index: ...
    @property
    def nlevels(self) -> int: ...
    @property
    def _has_aliases(self) -> bool: ...
    @property
    def _need_to_save_header(self) -> bool: ...
    @property
    def write_cols(self) -> Sequence[Hashable]: ...
    @property
    def encoded_labels(self) -> list[Hashable]: ...
    def save(self) -> None: ...
    def _save(self) -> None: ...
    def _save_header(self) -> None: ...
    def _generate_multiindex_header_rows(self) -> Iterator[list[Hashable]]: ...
    def _save_body(self) -> None: ...
    def _save_chunk(self, start_i: int, end_i: int) -> None: ...
```

Wait, I need to add the missing imports:

```python
from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence
import csv as _csvlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pandas.core.indexes.api import Index

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        FloatFormatType,
        IndexLabel,
        StorageOptions,
        WriteBuffer,
    )
    from pandas.io.formats.format import DataFrameFormatter
    from pandas import DataFrame

_DEFAULT_CHUNKSIZE_CELLS: int = ...

class CSVFormatter:
    fmt: DataFrameFormatter
    obj: DataFrame
    filepath_or_buffer: FilePath | WriteBuffer[str] | WriteBuffer[bytes]
    encoding: str | None
    compression: CompressionOptions
    mode: str
    storage_options: StorageOptions
    sep: str
    index_label: Sequence[Hashable] | bool
    errors: str
    quoting: int
    quotechar: str | None
    doublequote: bool
    escapechar: str | None
    lineterminator: str
    date_format: str | None
    cols: np.ndarray
    chunksize: int

    def __init__(
        self,
        formatter: DataFrameFormatter,
        path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] = ...,
        sep: str = ...,
        cols: Sequence[Hashable] | None = ...,
        index_label: IndexLabel | None = ...,
        mode: str = ...,
        encoding: str | None = ...,
        errors: str = ...,
        compression: CompressionOptions = ...,
        quoting: int | None = ...,
        lineterminator: str | None = ...,
        chunksize: int | None = ...,
        quotechar: str | None = ...,
        date_format: str | None = ...,
        doublequote: bool = ...,
        escapechar: str | None = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    @property
    def na_rep(self) -> str: ...
    @property
    def float_format(self) -> FloatFormatType | None: ...
    @property
    def decimal(self) -> str: ...
    @property
    def header(self) -> bool | list[str]: ...
    @property
    def index(self) -> bool: ...
    def _initialize_index_label(self, index_label: IndexLabel | None) -> Sequence[Hashable] | bool: ...
    def _get_index_label_from_obj(self) -> list[Hashable]: ...
    def _get_index_label_multiindex(self) -> list[Hashable]: ...
    def _get_index_label_flat(self) -> list[Hashable]: ...
    def _initialize_quotechar(self, quotechar: str | None) -> str | None: ...
    @property
    def has_mi_columns(self) -> bool: ...
    def _initialize_columns(self, cols: Sequence[Hashable] | None) -> np.ndarray: ...
    def _initialize_chunksize(self, chunksize: int | None) -> int: ...
    @property
    def _number_format(self) -> dict[str, Any]: ...
    @property
    def data_index(self) -> Index: ...
    @property
    def nlevels(self) -> int: ...
    @property
    def _has_aliases(self) -> bool: ...
    @property
    def _need_to_save_header(self) -> bool: ...
    @property
    def write_cols(self) -> Sequence[Hashable]: ...
    @property
    def encoded_labels(self) -> list[Hashable]: ...
    def save(self) -> None: ...
    def _save(self) -> None: ...
    def _save_header(self) -> None: ...
    def _generate_multiindex_header_rows(self) -> Iterator[list[Hashable]]: ...
    def _save_body(self) -> None: ...
    def _save_chunk(self, start_i: int, end_i: int) -> None: ...