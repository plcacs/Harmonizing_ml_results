```pyi
from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np

_DEFAULT_CHUNKSIZE_CELLS: int

class CSVFormatter:
    fmt: Any
    obj: Any
    filepath_or_buffer: str
    encoding: str | None
    compression: str
    mode: str
    storage_options: Any
    sep: str
    index_label: list[Any] | bool
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
        formatter: Any,
        path_or_buf: str = ...,
        sep: str = ...,
        cols: Any = ...,
        index_label: Any = ...,
        mode: str = ...,
        encoding: str | None = ...,
        errors: str = ...,
        compression: str = ...,
        quoting: int | None = ...,
        lineterminator: str = ...,
        chunksize: int | None = ...,
        quotechar: str = ...,
        date_format: str | None = ...,
        doublequote: bool = ...,
        escapechar: str | None = ...,
        storage_options: Any = ...,
    ) -> None: ...
    @property
    def na_rep(self) -> Any: ...
    @property
    def float_format(self) -> Any: ...
    @property
    def decimal(self) -> Any: ...
    @property
    def header(self) -> Any: ...
    @property
    def index(self) -> Any: ...
    def _initialize_index_label(self, index_label: Any) -> list[Any] | bool: ...
    def _get_index_label_from_obj(self) -> list[Any]: ...
    def _get_index_label_multiindex(self) -> list[Any]: ...
    def _get_index_label_flat(self) -> list[Any]: ...
    def _initialize_quotechar(self, quotechar: str) -> str | None: ...
    @property
    def has_mi_columns(self) -> bool: ...
    def _initialize_columns(self, cols: Any) -> Any: ...
    def _initialize_chunksize(self, chunksize: int | None) -> int: ...
    @property
    def _number_format(self) -> dict[str, Any]: ...
    @property
    def data_index(self) -> Any: ...
    @property
    def nlevels(self) -> int: ...
    @property
    def _has_aliases(self) -> bool: ...
    @property
    def _need_to_save_header(self) -> bool: ...
    @property
    def write_cols(self) -> Any: ...
    @property
    def encoded_labels(self) -> list[Any]: ...
    def save(self) -> None: ...
    def _save(self) -> None: ...
    def _save_header(self) -> None: ...
    def _generate_multiindex_header_rows(self) -> Iterator[list[Any]]: ...
    def _save_body(self) -> None: ...
    def _save_chunk(self, start_i: int, end_i: int) -> None: ...
```