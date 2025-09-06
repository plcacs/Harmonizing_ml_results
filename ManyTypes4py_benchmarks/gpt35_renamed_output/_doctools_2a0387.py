from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from collections.abc import Iterable

class TablePlotter:
    def __init__(self, cell_width: float = 0.37, cell_height: float = 0.25, font_size: float = 7.5):
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.font_size = font_size

    def func_zyumtllu(self, df: pd.DataFrame) -> Tuple[int, int]:
        row, col = df.shape
        return row + df.columns.nlevels, col + df.index.nlevels

    def func_7k3ik97l(self, left: List[pd.DataFrame], right: pd.DataFrame, vertical: bool) -> Tuple[int, int]:
        if vertical:
            vcells = max(sum(self._shape(df)[0] for df in left), self._shape(right)[0])
            hcells = max(self._shape(df)[1] for df in left) + self._shape(right)[1]
        else:
            vcells = max([self._shape(df)[0] for df in left] + [self._shape(right)[0]])
            hcells = sum([self._shape(df)[1] for df in left] + [self._shape(right)[1]])
        return hcells, vcells

    def func_lg20hcsh(self, left: List[pd.DataFrame], right: pd.DataFrame, labels: List[str] = [], vertical: bool = True) -> Figure:
        ...

    def func_3aeb1sps(self, data: pd.Series) -> pd.DataFrame:
        ...

    def func_7ow7p6tq(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

    def func_s8nu0im5(self, ax, df: pd.DataFrame, title: str, height: float = None):
        ...

    def _insert_index(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def _shape(self, df: pd.DataFrame) -> Tuple[int, int]:
        ...

    def _get_cells(self, left: List[pd.DataFrame], right: pd.DataFrame, vertical: bool) -> Tuple[int, int]:
        ...

    def _conv(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def _make_table(self, ax, df: pd.DataFrame, title: str, height: float):
        ...

def func_6eee5ty9():
    ...
