"""
Module for formatting output data into CSV files.
"""
from __future__ import annotations
from collections.abc import Iterator, Sequence
import csv as csvlib
import os
from typing import Any, Optional, Union, List, Sequence as Seq, Iterator as Iter, Tuple, Generator, Any, Dict, Optional as Opt, Union, List, Sequence, Iterable, Iterator, Any, cast, overload, TYPE_CHECKING
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr, FloatFormatType, IndexLabel, CompressionOptions, StorageOptions, WriteBuffer, npt
from pandas.io.common import get_handle
from pandas.io.formats.format import DataFrameFormatter
from pandas.core.dtypes.generic import ABCDatetimeIndex, ABCIndex, ABCMultiIndex, ABCPeriodIndex
from pandas.core.indexes.api import Index

_DEFAULT_CHUNKSIZE_CELLS: int = ...

class CSVFormatter:
    """
    Class for formatting DataFrame data into CSV files.
    """
    def __init__(self, formatter: DataFrameFormatter, path_or_buf: Union[str, bytes, os.PathLike] = '', sep: str = ',', cols: Optional[Seq[Hashable]] = None, index_label: Optional[Union[Seq[str], str, bool]] = None, mode: str = 'w', encoding: Optional[str] = None, errors: str = 'strict', compression: Union[str, CompressionOptions] = 'infer', quoting: Optional[int] = None, lineterminator: str = '\n', chunksize: Optional[int] = None, quotechar: str = '"', date_format: Optional[str] = None, doublequote: bool = True, escapechar: Optional[str] = None, storage_options: Optional[StorageOptions] = None) -> None:
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
    def header(self) -> Union[bool, Seq[str]]:
        ...

    @property
    def index(self) -> bool:
        ...

    def _initialize_index_label(self, index_label: Optional[Union[Seq[str], str, bool]]) -> Optional[Union[Seq[str], str, bool]]:
        ...

    def _get_index_label_from_obj(self) -> Union[List[str], List[Optional[str]]]:
        ...

    def _get_index_label_multiindex(self) -> List[str]:
        ...

    def _get_index_label_flat(self) -> List[str]:
        ...

    def _initialize_quotechar(self, quotechar: str) -> Optional[str]:
        ...

    @property
    def has_mi_columns(self) -> bool:
        ...

    def _initialize_columns(self, cols: Optional[Seq[Hashable]]) -> npt.NDArray[np.object_]:
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
    def write_cols(self) -> Union[Seq[Hashable], Seq[str]]:
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

    def _generate_multiindex_header_rows(self) -> Generator[List[str], None, None]:
        ...

    def _save_body(self) -> None:
        ...

    def _save_chunk(self, start_i: int, end_i: int) -> None:
        ...