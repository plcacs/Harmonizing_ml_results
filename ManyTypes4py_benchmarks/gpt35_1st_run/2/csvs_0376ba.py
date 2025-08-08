from __future__ import annotations
from collections.abc import Hashable, Iterable, Iterator, Sequence
import csv as csvlib
import os
from typing import TYPE_CHECKING, Any, cast, List, Tuple, Union
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import ABCDatetimeIndex, ABCIndex, ABCMultiIndex, ABCPeriodIndex
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle

if TYPE_CHECKING:
    from pandas._typing import CompressionOptions, FilePath, FloatFormatType, IndexLabel, StorageOptions, WriteBuffer, npt
    from pandas.io.formats.format import DataFrameFormatter

_DEFAULT_CHUNKSIZE_CELLS: int = 100000

class CSVFormatter:
    def __init__(self, formatter: DataFrameFormatter, path_or_buf: FilePath = '', sep: str = ',', cols: Union[None, List[str]] = None, index_label: Union[None, str, List[str], np.ndarray, ABCIndex] = None, mode: str = 'w', encoding: Union[None, str] = None, errors: str = 'strict', compression: str = 'infer', quoting: Union[None, int] = None, lineterminator: str = '\n', chunksize: Union[None, int] = None, quotechar: str = '"', date_format: Union[None, str] = None, doublequote: bool = True, escapechar: Union[None, str] = None, storage_options: Union[None, StorageOptions] = None) -> None:
    
    def _initialize_index_label(self, index_label: Union[None, str, List[str], np.ndarray, ABCIndex]) -> Union[None, List[str]]:
    
    def _get_index_label_from_obj(self) -> List[str]:
    
    def _get_index_label_multiindex(self) -> List[str]:
    
    def _get_index_label_flat(self) -> List[str]:
    
    def _initialize_quotechar(self, quotechar: Union[None, str]) -> Union[None, str]:
    
    @property
    def has_mi_columns(self) -> bool:
    
    def _initialize_columns(self, cols: Union[None, List[str]]) -> List[str]:
    
    def _initialize_chunksize(self, chunksize: Union[None, int]) -> int:
    
    @property
    def _number_format(self) -> dict[str, Any]:
    
    @cache_readonly
    def data_index(self) -> Index:
    
    @property
    def nlevels(self) -> int:
    
    @property
    def _has_aliases(self) -> bool:
    
    @property
    def _need_to_save_header(self) -> bool:
    
    @property
    def write_cols(self) -> SequenceNotStr[Hashable]:
    
    @property
    def encoded_labels(self) -> List[str]:
    
    def save(self) -> None:
    
    def _save(self) -> None:
    
    def _save_header(self) -> None:
    
    def _generate_multiindex_header_rows(self) -> Iterator[List[str]]:
    
    def _save_body(self) -> None:
    
    def _save_chunk(self, start_i: int, end_i: int) -> None:
