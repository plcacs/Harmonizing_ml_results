#!/usr/bin/env python3
"""
Module for formatting output data into CSV files.
"""

from __future__ import annotations
from collections.abc import Hashable, Iterable, Iterator, Sequence
import csv as csvlib
import os
from typing import Any, Optional, Union, List, Iterator as TypingIterator
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr, CompressionOptions, FilePath, WriteBuffer
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import ABCDatetimeIndex, ABCIndex, ABCMultiIndex, ABCPeriodIndex
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle

if False:
    # For type checking purposes
    from pandas.io.formats.format import DataFrameFormatter

_DEFAULT_CHUNKSIZE_CELLS = 100000


class CSVFormatter:
    def __init__(
        self,
        formatter: Any,  # Expected to be a DataFrameFormatter
        path_or_buf: Union[str, FilePath, WriteBuffer] = '',
        sep: str = ',',
        cols: Optional[Iterable[Hashable]] = None,
        index_label: Optional[Union[bool, Hashable, Sequence[Hashable]]] = None,
        mode: str = 'w',
        encoding: Optional[str] = None,
        errors: str = 'strict',
        compression: Union[CompressionOptions, str] = 'infer',
        quoting: Optional[int] = None,
        lineterminator: Optional[str] = '\n',
        chunksize: Optional[int] = None,
        quotechar: str = '"',
        date_format: Optional[str] = None,
        doublequote: bool = True,
        escapechar: Optional[str] = None,
        storage_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.fmt = formatter
        self.obj = self.fmt.frame
        self.filepath_or_buffer: Union[str, FilePath, WriteBuffer] = path_or_buf
        self.encoding: Optional[str] = encoding
        self.compression: Union[CompressionOptions, str] = compression
        self.mode: str = mode
        self.storage_options: Optional[dict[str, Any]] = storage_options
        self.sep: str = sep
        self.index_label: Optional[Union[bool, List[Hashable]]] = self._initialize_index_label(index_label)
        self.errors: str = errors
        self.quoting: int = quoting or csvlib.QUOTE_MINIMAL
        self.quotechar: Optional[str] = self._initialize_quotechar(quotechar)
        self.doublequote: bool = doublequote
        self.escapechar: Optional[str] = escapechar
        self.lineterminator: str = lineterminator or os.linesep
        self.date_format: Optional[str] = date_format
        self.cols: Sequence[Hashable] = self._initialize_columns(cols)
        self.chunksize: int = self._initialize_chunksize(chunksize)

    @property
    def na_rep(self) -> Any:
        return self.fmt.na_rep

    @property
    def float_format(self) -> Any:
        return self.fmt.float_format

    @property
    def decimal(self) -> Any:
        return self.fmt.decimal

    @property
    def header(self) -> Any:
        return self.fmt.header

    @property
    def index(self) -> bool:
        return self.fmt.index

    def _initialize_index_label(
        self, index_label: Optional[Union[bool, Hashable, Sequence[Hashable]]]
    ) -> Optional[Union[bool, List[Hashable]]]:
        if index_label is not False:
            if index_label is None:
                return self._get_index_label_from_obj()
            elif not isinstance(index_label, (list, tuple, np.ndarray, ABCIndex)):
                return [index_label]  # type: ignore
        return index_label

    def _get_index_label_from_obj(self) -> List[Hashable]:
        if isinstance(self.obj.index, ABCMultiIndex):
            return self._get_index_label_multiindex()
        else:
            return self._get_index_label_flat()

    def _get_index_label_multiindex(self) -> List[str]:
        return [name or '' for name in self.obj.index.names]

    def _get_index_label_flat(self) -> List[str]:
        index_label_val = self.obj.index.name
        return [''] if index_label_val is None else [index_label_val]  # type: ignore

    def _initialize_quotechar(self, quotechar: Optional[str]) -> Optional[str]:
        if self.quoting != csvlib.QUOTE_NONE:
            return quotechar
        return None

    @property
    def has_mi_columns(self) -> bool:
        return isinstance(self.obj.columns, ABCMultiIndex)

    def _initialize_columns(self, cols: Optional[Iterable[Hashable]]) -> Sequence[Hashable]:
        if self.has_mi_columns:
            if cols is not None:
                msg: str = 'cannot specify cols with a MultiIndex on the columns'
                raise TypeError(msg)
        if cols is not None:
            if isinstance(cols, ABCIndex):
                cols = cols._get_values_for_csv(**self._number_format)
            else:
                cols = list(cols)
            self.obj = self.obj.loc[:, cols]
        new_cols = self.obj.columns
        return new_cols._get_values_for_csv(**self._number_format)

    def _initialize_chunksize(self, chunksize: Optional[int]) -> int:
        if chunksize is None:
            return _DEFAULT_CHUNKSIZE_CELLS // (len(self.cols) or 1) or 1
        return int(chunksize)

    @property
    def _number_format(self) -> dict[str, Any]:
        """Dictionary used for storing number formatting settings."""
        return {
            'na_rep': self.na_rep,
            'float_format': self.float_format,
            'date_format': self.date_format,
            'quoting': self.quoting,
            'decimal': self.decimal,
        }

    @cache_readonly
    def data_index(self) -> Index:
        data_index: Index = self.obj.index
        if isinstance(data_index, (ABCDatetimeIndex, ABCPeriodIndex)) and self.date_format is not None:
            data_index = Index([x.strftime(self.date_format) if notna(x) else '' for x in data_index])
        elif isinstance(data_index, ABCMultiIndex):
            data_index = data_index.remove_unused_levels()
        return data_index

    @property
    def nlevels(self) -> int:
        if self.index:
            return getattr(self.data_index, 'nlevels', 1)
        else:
            return 0

    @property
    def _has_aliases(self) -> bool:
        return isinstance(self.header, (tuple, list, np.ndarray, ABCIndex))

    @property
    def _need_to_save_header(self) -> bool:
        return bool(self._has_aliases or self.header)

    @property
    def write_cols(self) -> SequenceNotStr[Hashable]:
        if self._has_aliases:
            assert not isinstance(self.header, bool)
            if len(self.header) != len(self.cols):
                raise ValueError(f'Writing {len(self.cols)} cols but got {len(self.header)} aliases')
            return self.header  # type: ignore
        else:
            return self.cols  # type: ignore

    @property
    def encoded_labels(self) -> List[Hashable]:
        encoded_labels: List[Hashable] = []
        if self.index and self.index_label:
            assert isinstance(self.index_label, Sequence)
            encoded_labels = list(self.index_label)
        if not self.has_mi_columns or self._has_aliases:
            encoded_labels += list(self.write_cols)
        return encoded_labels

    def save(self) -> None:
        """
        Create the writer & save.
        """
        with get_handle(
            self.filepath_or_buffer,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            compression=self.compression,
            storage_options=self.storage_options,
        ) as handles:
            self.writer = csvlib.writer(
                handles.handle,
                lineterminator=self.lineterminator,
                delimiter=self.sep,
                quoting=self.quoting,
                doublequote=self.doublequote,
                escapechar=self.escapechar,
                quotechar=self.quotechar,
            )
            self._save()

    def _save(self) -> None:
        if self._need_to_save_header:
            self._save_header()
        self._save_body()

    def _save_header(self) -> None:
        if not self.has_mi_columns or self._has_aliases:
            self.writer.writerow(self.encoded_labels)
        else:
            for row in self._generate_multiindex_header_rows():
                self.writer.writerow(row)

    def _generate_multiindex_header_rows(self) -> TypingIterator[List[Any]]:
        columns = self.obj.columns
        for i in range(columns.nlevels):
            col_line: List[Any] = []
            if self.index:
                col_line.append(columns.names[i])
                if isinstance(self.index_label, list) and len(self.index_label) > 1:
                    col_line.extend([''] * (len(self.index_label) - 1))
            col_line.extend(columns._get_level_values(i))
            yield col_line
        if self.encoded_labels and set(self.encoded_labels) != {''}:
            yield self.encoded_labels + [''] * len(columns)

    def _save_body(self) -> None:
        nrows: int = len(self.data_index)
        chunks: int = nrows // self.chunksize + 1
        for i in range(chunks):
            start_i: int = i * self.chunksize
            end_i: int = min(start_i + self.chunksize, nrows)
            if start_i >= end_i:
                break
            self._save_chunk(start_i, end_i)

    def _save_chunk(self, start_i: int, end_i: int) -> None:
        slicer = slice(start_i, end_i)
        df = self.obj.iloc[slicer]
        res = df._get_values_for_csv(**self._number_format)
        data = list(res._iter_column_arrays())
        ix = (
            self.data_index[slicer]._get_values_for_csv(**self._number_format)
            if self.nlevels != 0
            else np.empty(end_i - start_i)
        )
        libwriters.write_csv_rows(data, ix, self.nlevels, self.cols, self.writer)
