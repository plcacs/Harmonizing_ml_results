from __future__ import annotations
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
import functools
import itertools
import re
from typing import TYPE_CHECKING, Any, cast
import warnings
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Period, PeriodIndex
import pandas.core.common as com
from pandas.io.formats.css import CSSResolver, CSSWarning
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.style import Styler
from pandas.io.formats.excel import ExcelWriter, ExcelCell, CssExcelCell, CSSToExcelConverter

if TYPE_CHECKING:
    from pandas._typing import ExcelWriterMergeCells, FilePath, IndexLabel, StorageOptions, WriteExcelBuffer

class ExcelFormatter:
    max_rows: int = 2 ** 20
    max_cols: int = 2 ** 14

    def __init__(self, df: DataFrame, na_rep: str = '', float_format: str = None, cols: Sequence = None, header: bool = True, index: bool = True, index_label: Any = None, merge_cells: bool | str = False, inf_rep: str = 'inf', style_converter: Callable = None) -> None:
        self.rowcounter: int = 0
        self.na_rep: str = na_rep
        self.styler: Styler | None = None
        self.style_converter: CSSToExcelConverter | None = None
        self.df: DataFrame = df
        self.columns: Index = self.df.columns
        self.float_format: str | None = float_format
        self.index: bool = index
        self.index_label: Any = index_label
        self.header: bool = header
        self.merge_cells: bool | str = merge_cells
        self.inf_rep: str = inf_rep

    def _format_value(self, val: Any) -> Any:
        ...

    def _format_header_mi(self) -> Iterable[ExcelCell]:
        ...

    def _format_header_regular(self) -> Iterable[CssExcelCell]:
        ...

    def _format_header(self) -> Iterable[ExcelCell]:
        ...

    def _format_body(self) -> Iterable[CssExcelCell]:
        ...

    def _format_regular_rows(self) -> Iterable[CssExcelCell]:
        ...

    def _format_hierarchical_rows(self) -> Iterable[CssExcelCell]:
        ...

    @property
    def _has_aliases(self) -> bool:
        ...

    def _generate_body(self, coloffset: int) -> Iterable[CssExcelCell]:
        ...

    def get_formatted_cells(self) -> Iterable[ExcelCell]:
        ...

    def write(self, writer: ExcelWriter | FilePath | WriteExcelBuffer, sheet_name: str = 'Sheet1', startrow: int = 0, startcol: int = 0, freeze_panes: tuple[int, int] | None = None, engine: str | None = None, storage_options: StorageOptions | None = None, engine_kwargs: dict | None = None) -> None:
        ...
