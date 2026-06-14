from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import matplotlib as mpl
import numpy as np

from pandas._libs import lib
from pandas.util._decorators import cache_readonly

import pandas as pd

from pandas.plotting._matplotlib.core import LinePlot, MPLPlot
from pandas.plotting._matplotlib.style import get_standard_colors

if TYPE_CHECKING:
    from collections.abc import Collection

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from pandas._typing import MatplotlibColor

def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs) -> None: ...

class BoxPlot(LinePlot):
    @property
    def _kind(self) -> str: ...

    _layout_type: str
    _valid_return_types: tuple[None, str, str, str]

    class BP(NamedTuple):
        ax: Axes
        lines: dict[str, list[Line2D]]

    return_type: str | None

    def __init__(
        self,
        data: pd.DataFrame,
        return_type: str = ...,
        **kwargs,
    ) -> None: ...

    @classmethod
    def _plot(
        cls,
        ax: Axes,
        y: np.ndarray,
        column_num: int | None = ...,
        return_type: str = ...,
        **kwds,
    ) -> tuple: ...

    def _validate_color_args(
        self,
        color: dict[str, MatplotlibColor] | MatplotlibColor | lib.NoDefault,
        colormap: str | None,
    ) -> dict[str, MatplotlibColor] | MatplotlibColor | None: ...

    @cache_readonly
    def _color_attrs(self) -> list: ...

    @cache_readonly
    def _boxes_c(self) -> MatplotlibColor: ...

    @cache_readonly
    def _whiskers_c(self) -> MatplotlibColor: ...

    @cache_readonly
    def _medians_c(self) -> MatplotlibColor: ...

    @cache_readonly
    def _caps_c(self) -> MatplotlibColor: ...

    def _get_colors(
        self,
        num_colors: int | None = ...,
        color_kwds: str = ...,
    ) -> None: ...

    def maybe_color_bp(self, bp: dict[str, list[Line2D]]) -> None: ...

    def _make_plot(self, fig: Figure) -> None: ...

    def _make_legend(self) -> None: ...

    def _post_plot_logic(self, ax: Axes, data: pd.DataFrame) -> None: ...

    @property
    def orientation(self) -> Literal['vertical', 'horizontal']: ...

    @property
    def result(self) -> pd.Series | np.ndarray | Axes | dict: ...

def maybe_color_bp(
    bp: dict[str, list[Line2D]],
    color_tup: tuple,
    **kwds,
) -> None: ...

def _grouped_plot_by_column(
    plotf,
    data: pd.DataFrame,
    columns: pd.Index | list | None = ...,
    by: str | list[str] | None = ...,
    numeric_only: bool = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    ax: Axes | None = ...,
    layout: tuple[int, int] | None = ...,
    return_type: str | None = ...,
    **kwargs,
) -> pd.Series | np.ndarray: ...

def boxplot(
    data: pd.DataFrame | pd.Series,
    column: str | list[str] | None = ...,
    by: str | list[str] | None = ...,
    ax: Axes | None = ...,
    fontsize: int | float | None = ...,
    rot: int = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: tuple[int, int] | None = ...,
    return_type: str | None = ...,
    **kwds,
) -> Axes | dict | BoxPlot.BP | pd.Series | np.ndarray: ...

def boxplot_frame(
    self: pd.DataFrame,
    column: str | list[str] | None = ...,
    by: str | list[str] | None = ...,
    ax: Axes | None = ...,
    fontsize: int | float | None = ...,
    rot: int = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: tuple[int, int] | None = ...,
    return_type: str | None = ...,
    **kwds,
) -> Axes | dict | BoxPlot.BP | pd.Series | np.ndarray: ...

def boxplot_frame_groupby(
    grouped: pd.core.groupby.DataFrameGroupBy,
    subplots: bool = ...,
    column: str | list[str] | None = ...,
    fontsize: int | float | None = ...,
    rot: int = ...,
    grid: bool = ...,
    ax: Axes | None = ...,
    figsize: tuple[float, float] | None = ...,
    layout: tuple[int, int] | None = ...,
    sharex: bool = ...,
    sharey: bool = ...,
    **kwds,
) -> pd.Series | Axes | dict | BoxPlot.BP | np.ndarray: ...