"""
Module for formatting output data into CSV files.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
import csv as csvlib
import os
from typing import Optional, Union, Iterator, cast
import numpy as np
from pandas._typing import CompressionOptions, FilePath, FloatFormatType, IndexLabel, StorageOptions, WriteBuffer, npt
from pandas.io.formats.format import DataFrameFormatter
from pandas.core.indexes.api import Index

_DEFAULT_CHUNKSIZE_CELLS: int = 100000

class CSVFormatter:
    def __init__(
        self,
        formatter: DataFrameFormatter,
        path_or_buf: FilePath | WriteBuffer = '',
        sep: str = ',',
        cols: Optional[Union[Sequence[Hashable], Index]] = None,
        index_label: Optional[Union[str, Sequence[str], bool]] = None,
        mode: str = 'w',
        encoding: Optional[str] = None,
        errors: str = 'strict',
        compression: Optional[CompressionOptions] = 'infer',
        quoting: Optional[int] = None,
        lineterminator: str = '\n',
        chunksize: Optional[int] = None,
        quotechar: str = '"',
        date_format: Optional[str] = None,
        doublequote: bool = True,
        escapechar: Optional[str] = None,
        storage_options: Optional[StorageOptions] = None,
    ) -> None: ...

    @property
    def na_rep(self) -> str: ...

    @property
    def float_format(self) -> Optional[FloatFormatType]: ...

    @property
    def decimal(self) -> str: ...

    @property
    def header(self) -> Optional[Union[bool, Sequence[Hashable]]]: ...

    @property
    def index(self) -> bool: ...

    def _initialize_index_label(self, index_label: Optional[Union[str, Sequence[str], bool]]) -> Optional[Union[str, Sequence[str], bool]]: ...

    def _get_index_label_from_obj(self) -> Sequence[str]: ...

    def _get_index_label_multiindex(self) -> list[str]: ...

    def _get_index_label_flat(self) -> list[str]: ...

    def _initialize_quotechar(self, quotechar: str) -> Optional[str]: ...

    @property
    def has_mi_columns(self) -> bool: ...

    def _initialize_columns(self, cols: Optional[Union[Sequence[Hashable], Index]]) -> Sequence[Hashable]: ...

    def _initialize_chunksize(self, chunksize: Optional[int]) -> int: ...

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

    def _generate_multiindex_header_rows(self) -> Iterator[Sequence[Any]]: ...

    def _save_body(self) -> None: ...

    def _save_chunk(self, start_i: int, end_i: int) -> None: ...

    def __setattr__(self, name: str, value: Any) -> None: ...