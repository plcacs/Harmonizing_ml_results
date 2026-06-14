from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

import pandas as pd
from pandas.plotting._matplotlib.core import LinePlot, MPLPlot

if TYPE_CHECKING:
    from collections.abc import Collection
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from pandas._typing import MatplotlibColor


def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs) -> None: ...


class BoxPlot(LinePlot):
    @property
    def _kind(self) -> Literal["box"]: ...

    _layout_type: str
    _valid_return_types: tuple[None, str, str, str]

    class BP(NamedTuple):
        ax: Axes
        lines: dict[str, list[Line2D]]

    return_type: str | None

    def __init__(self, data: pd.DataFrame, return_type: str | None = ..., **kwargs) -> None: ...

    @classmethod
    def _plot(
        cls,
        ax: Axes,
        y: np.ndarray,
        column_num: int | None = ...,
        return_type: str | None = ...,
        **kwds,
    ) -> tuple[Axes | dict | BoxPlot.BP, dict]: ...

    def _validate_color_args(
        self, color: object, colormap: object | None
    ) -> dict[str, MatplotlibColor] | MatplotlibColor | None: ...

    @property
    def _color_attrs(self) -> list: ...

    @property
    def _boxes_c(self) -> MatplotlibColor: ...

    @property
    def _whiskers_c(self) -> MatplotlibColor: ...

    @property
    def _medians_c(self) -> MatplotlibColor: ...

    @property
    def _caps_c(self) -> MatplotlibColor: ...

    def _get_colors(
        self, num_colors: int | None = ..., color_kwds: str = ...
    ) -> None: ...

    def maybe_color_bp(self, bp: dict) -> None: ...

    def _make_plot(self, fig: Figure) -> None: ...

    def _make_legend(self) -> None: ...

    def _post_plot_logic(self, ax: Axes, data: pd.DataFrame) -> None: ...

    @property
    def orientation(self) -> Literal["vertical", "horizontal"]: ...

    @property
    def result(self) -> pd.Series | object: ...


def maybe_color_bp(
    bp: dict,
    color_tup: tuple[MatplotlibColor, MatplotlibColor, MatplotlibColor, MatplotlibColor],
    **kwds,
) -> None: ...


def _grouped_plot_by_column(
    plotf: object,
    data: pd.DataFrame,
    columns: list | pd.Index | None = ...,
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