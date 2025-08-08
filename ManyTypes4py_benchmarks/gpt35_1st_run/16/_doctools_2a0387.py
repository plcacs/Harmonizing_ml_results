from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from collections.abc import Iterable
    from matplotlib.figure import Figure

class TablePlotter:
    def __init__(self, cell_width: float = 0.37, cell_height: float = 0.25, font_size: float = 7.5) -> None:
    def _shape(self, df: pd.DataFrame) -> Tuple[int, int]:
    def _get_cells(self, left: List[pd.DataFrame], right: pd.DataFrame, vertical: bool) -> Tuple[int, int]:
    def plot(self, left: List[pd.DataFrame], right: pd.DataFrame, labels: List[str] = (), vertical: bool = True) -> Figure:
    def _conv(self, data: pd.DataFrame) -> pd.DataFrame:
    def _insert_index(self, data: pd.DataFrame) -> pd.DataFrame:
    def _make_table(self, ax, df: pd.DataFrame, title: str, height: float = None) -> None:

def main() -> None:
