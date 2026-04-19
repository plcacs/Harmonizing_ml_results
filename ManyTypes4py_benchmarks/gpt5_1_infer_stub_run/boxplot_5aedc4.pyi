from __future__ import annotations

from typing import Any, Callable, Collection, Iterable, Literal, NamedTuple, Optional, TypeAlias, Union

import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas._typing import MatplotlibColor
from pandas.plotting._matplotlib.core import LinePlot, MPLPlot


BPDict: TypeAlias = dict[str, list[Line2D]]
BoxplotSimpleReturn: TypeAlias = Union[Axes, BPDict, "BoxPlot.BP"]
BoxplotReturn: TypeAlias = Union[BoxplotSimpleReturn, pd.Series, npt.NDArray[object]]


def _set_ticklabels(ax: Axes, labels: Collection[str], is_vertical: bool, **kwargs: Any) -> None: ...


class BoxPlot(LinePlot):
    _layout_type: str
    _valid_return_types: tuple[None | Literal["axes", "dict", "both"], ...]
    return_type: Optional[Literal["axes", "dict", "both"]]

    class BP(NamedTuple):
        ax: Axes
        lines: BPDict

    @property
    def _kind(self) -> str: ...
    def __init__(self, data: pd.DataFrame, return_type: Optional[Literal["axes", "dict", "both"]] = "axes", **kwargs: Any) -> None: ...
    @classmethod
    def _plot(
        cls,
        ax: Axes,
        y: Any,
        column_num: Optional[int] = ...,
        return_type: Optional[Literal["axes", "dict", "both"]] = "axes",
        **kwds: Any,
    ) -> tuple[Axes | BP | BPDict, BPDict]: ...
    def _validate_color_args(self, color: Any, colormap: Any | None) -> Any | None: ...
    @property
    def _color_attrs(self) -> list[MatplotlibColor]: ...
    @property
    def _boxes_c(self) -> MatplotlibColor: ...
    @property
    def _whiskers_c(self) -> MatplotlibColor: ...
    @property
    def _medians_c(self) -> MatplotlibColor: ...
    @property
    def _caps_c(self) -> MatplotlibColor: ...
    def _get_colors(self, num_colors: Optional[int] = ..., color_kwds: str = "color") -> Any: ...
    def maybe_color_bp(self, bp: BPDict) -> None: ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _make_legend(self) -> None: ...
    def _post_plot_logic(self, ax: Axes, data: Any) -> None: ...
    @property
    def orientation(self) -> Literal["vertical", "horizontal"]: ...
    @property
    def result(self) -> object: ...


def maybe_color_bp(
    bp: BPDict,
    color_tup: tuple[MatplotlibColor, MatplotlibColor, MatplotlibColor, MatplotlibColor],
    **kwds: Any,
) -> None: ...


def _grouped_plot_by_column(
    plotf: Callable[[Iterable[object], Iterable[object], Axes], Any],
    data: pd.DataFrame,
    columns: Optional[Collection[object]] = ...,
    by: Optional[Union[object, list[object], tuple[object, ...]]] = ...,
    numeric_only: bool = ...,
    grid: bool = ...,
    figsize: Optional[tuple[float, float]] = ...,
    ax: Optional[Axes] = ...,
    layout: Optional[tuple[int, int]] = ...,
    return_type: Optional[Literal["axes", "dict", "both"]] = ...,
    **kwargs: Any,
) -> Union[pd.Series, Axes, npt.NDArray[object]]: ...


def boxplot(
    data: Union[pd.DataFrame, pd.Series],
    column: Optional[Union[object, Collection[object]]] = ...,
    by: Optional[Union[object, list[object], tuple[object, ...]]] = ...,
    ax: Optional[Axes] = ...,
    fontsize: Optional[Union[int, float]] = ...,
    rot: Union[int, float] = ...,
    grid: bool = ...,
    figsize: Optional[tuple[float, float]] = ...,
    layout: Optional[tuple[int, int]] = ...,
    return_type: Optional[Literal["axes", "dict", "both"]] = ...,
    **kwds: Any,
) -> BoxplotReturn: ...


def boxplot_frame(
    self: pd.DataFrame,
    column: Optional[Union[object, Collection[object]]] = ...,
    by: Optional[Union[object, list[object], tuple[object, ...]]] = ...,
    ax: Optional[Axes] = ...,
    fontsize: Optional[Union[int, float]] = ...,
    rot: Union[int, float] = ...,
    grid: bool = ...,
    figsize: Optional[tuple[float, float]] = ...,
    layout: Optional[tuple[int, int]] = ...,
    return_type: Optional[Literal["axes", "dict", "both"]] = ...,
    **kwds: Any,
) -> BoxplotReturn: ...


def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = ...,
    column: Optional[Union[object, Collection[object]]] = ...,
    fontsize: Optional[Union[int, float]] = ...,
    rot: Union[int, float] = ...,
    grid: bool = ...,
    ax: Optional[Axes] = ...,
    figsize: Optional[tuple[float, float]] = ...,
    layout: Optional[tuple[int, int]] = ...,
    sharex: bool = ...,
    sharey: bool = ...,
    **kwds: Any,
) -> Union[pd.Series, BoxplotSimpleReturn, npt.NDArray[object]]: ...