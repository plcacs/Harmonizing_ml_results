from __future__ import annotations
from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Index, Series
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

def scatter_matrix(frame: DataFrame, alpha: float = 0.5, figsize: Optional[Tuple[int, int]] = None, ax: Optional[Axes] = None, grid: bool = False, diagonal: str = 'hist', marker: str = '.', density_kwds: Optional[Dict[str, Any]] = None, hist_kwds: Optional[Dict[str, Any]] = None, range_padding: float = 0.05, **kwds: Any) -> ndarray[Axes]:
    ...

def _get_marker_compat(marker: str) -> str:
    ...

def radviz(frame: DataFrame, class_column: str, ax: Optional[Axes] = None, color: Optional[str] = None, colormap: Optional[str] = None, **kwds: Any) -> Axes:
    ...

def andrews_curves(frame: DataFrame, class_column: str, ax: Optional[Axes] = None, samples: int = 200, color: Optional[str] = None, colormap: Optional[str] = None, **kwds: Any) -> Axes:
    ...

def bootstrap_plot(series: Series, fig: Optional[Figure] = None, size: int = 50, samples: int = 500, **kwds: Any) -> Figure:
    ...

def parallel_coordinates(frame: DataFrame, class_column: str, cols: Optional[List[str]] = None, ax: Optional[Axes] = None, color: Optional[str] = None, use_columns: bool = False, xticks: Optional[List[float]] = None, colormap: Optional[str] = None, axvlines: bool = True, axvlines_kwds: Optional[Dict[str, Any]] = None, sort_labels: bool = False, **kwds: Any) -> Axes:
    ...

def lag_plot(series: Series, lag: int = 1, ax: Optional[Axes] = None, **kwds: Any) -> Axes:
    ...

def autocorrelation_plot(series: Series, ax: Optional[Axes] = None, **kwds: Any) -> Axes:
    ...

def unpack_single_str_list(keys: Union[List[str], str]) -> str:
    ...