from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Optional, Tuple, Dict, Any
import numpy as np
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import Series
import pandas._testing as tm
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.collections import Collection, LineCollection, PolyCollection
from matplotlib.colors import ColorConverter
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

def _check_legend_labels(axes: Sequence[Axes], labels: Optional[Sequence[str]], visible: bool) -> None:
    ...

def _check_legend_marker(ax: Axes, expected_markers: Optional[Sequence[str]], visible: bool) -> None:
    ...

def _check_data(xp: Axes, rs: Axes) -> None:
    ...

def _check_visible(collections: Sequence[Collection], visible: bool) -> None:
    ...

def _check_patches_all_filled(axes: Sequence[Axes], filled: bool) -> None:
    ...

def _get_colors_mapped(series: Series, colors: Sequence[str]) -> List[str]:
    ...

def _check_colors(collections: Sequence[Any], linecolors: Optional[Sequence[str]], facecolors: Optional[Sequence[str]], mapping: Optional[Series]) -> None:
    ...

def _check_text_labels(texts: Sequence[Any], expected: Optional[Sequence[str]]) -> None:
    ...

def _check_ticks_props(axes: Sequence[Axes], xlabelsize: Optional[float], xrot: Optional[float], ylabelsize: Optional[float], yrot: Optional[float]) -> None:
    ...

def _check_ax_scales(axes: Sequence[Axes], xaxis: str, yaxis: str) -> None:
    ...

def _check_axes_shape(axes: Sequence[Axes], axes_num: Optional[int], layout: Optional[Tuple[int, int]], figsize: Optional[Tuple[float, float]]) -> None:
    ...

def _flatten_visible(axes: Sequence[Axes]) -> List[Axes]:
    ...

def _check_has_errorbars(axes: Sequence[Axes], xerr: float, yerr: float) -> None:
    ...

def _check_box_return_type(returned: Any, return_type: str, expected_keys: Optional[Sequence[str]], check_ax_title: bool) -> None:
    ...

def _check_grid_settings(obj, kinds, kws=None) -> None:
    ...

def _unpack_cycler(rcParams, field='color') -> List[Any]:
    ...

def get_x_axis(ax: Axes) -> Axes:
    ...

def get_y_axis(ax: Axes) -> Axes:
    ...

def assert_is_valid_plot_return_object(objs: Any) -> None:
    ...

def _check_plot_works(f: Callable, default_axes: bool, **kwargs) -> Any:
    ...
