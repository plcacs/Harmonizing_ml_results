"""
Module for formatting output data into CSV files.
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Optional, Union, List, Tuple, Dict, Callable, TypeVar, overload, Literal
import csv as csvlib
import os
from pathlib import Path
import numpy as np
from pandas._libs.writers import write_csv_rows as libwriters_write_csv_rows
from pandas._typing import (
    CompressionOptions,
    FilePath,
    FloatFormatType,
    IndexLabel,
    StorageOptions,
    WriteBuffer,
    npt,
)
from pandas.io.common import Handle

T = TypeVar("T")

class CSVFormatter:
    """
    A CSV formatter class for writing DataFrames to CSV files.
    """
    def __init__(
        self,
        formatter: DataFrameFormatter,
        path_or_buf: Union[FilePath, WriteBuffer] = '',
        sep: str = ',',
        cols: Optional[Sequence[Hashable]] = None,
        index_label: Optional[IndexLabel] = None,
        mode: str = 'w',
        encoding: Optional[str] = None,
        errors: str = 'strict',
        compression: Optional[CompressionOptions] = 'infer',
        quoting: Optional[int] = None,
        lineterminator: Optional[str] = '\n',
        chunksize: Optional[int] = None,
        quotechar: Optional[str] = '"',
        date_format: Optional[str] = None,
        doublequote: bool = True,
        escapechar: Optional[str] = None,
        storage_options: Optional[StorageOptions] = None,
    ) -> None:
        ...

    @property
    def na_rep(self) -> str:
        ...

    @property
    def float_format(self) -> Optional[FloatFormatType]:
        ...

    @property
    def decimal(self) -> str:
        ...

    @property
    def header(self) -> Union[bool, Sequence[Hashable]]:
        ...

    @property
    def index(self) -> bool:
        ...

    def _initialize_index_label(self, index_label: Optional[IndexLabel]) -> Optional[IndexLabel]:
        ...

    def _get_index_label_from_obj(self) -> List[str]:
        ...

    def _get_index_label_multiindex(self) -> List[str]:
        ...

    def _get_index_label_flat(self) -> List[str]:
        ...

    def _initialize_quotechar(self, quotechar: Optional[str]) -> Optional[str]:
        ...

    @property
    def has_mi_columns(self) -> bool:
        ...

    def _initialize_columns(self, cols: Optional[Sequence[Hashable]]) -> npt.NDArray[np.object_]:
        ...

    def _initialize_chunksize(self, chunksize: Optional[int]) -> int:
        ...

    @property
    def _number_format(self) -> Dict[str, Any]:
        ...

    @property
    def data_index(self) -> Index:
        ...

    @property
    def nlevels(self) -> int:
        ...

    @property
    def _has_aliases(self) -> bool:
        ...

    @property
    def _need_to_save_header(self) -> bool:
        ...

    @property
    def write_cols(self) -> Sequence[Hashable]:
        ...

    @property
    def encoded_labels(self) -> List[str]:
        ...

    def save(self) -> None:
        ...

    def _save(self) -> None:
        ...

    def _save_header(self) -> None:
        ...

    def _generate_multiindex_header_rows(self) -> Iterator[List[str]]:
        ...

    def _save_body(self) -> None:
        ...

    def _save_chunk(self, start_i: int, end_i: int) -> None:
        ...

def write_csv_rows(
    data: Sequence[Sequence[Any]],
    ix: Sequence[Any],
    nlevels: int,
    columns: Sequence[Hashable],
    writer: csvlib._csv.writer,
) -> None:
    ...

def get_handle(
    path_or_buf: Union[FilePath, WriteBuffer],
    mode: str,
    encoding: Optional[str] = None,
    errors: str = 'strict',
    compression: Optional[CompressionOptions] = 'infer',
    storage_options: Optional[StorageOptions] = None,
) -> Handle:
    ...