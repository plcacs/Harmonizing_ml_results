from typing import Any, Callable, ClassVar, Iterable, Literal, NamedTuple, Optional, Sequence
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def _set_ticklabels(ax: Axes, labels: list[str], is_vertical: bool, **kwargs: Any) -> None: ...
class BoxPlot(LinePlot):
    _layout_type: ClassVar[str]
    _valid_return_types: ClassVar[tuple[None | Literal["axes", "dict", "both"], ...]]

    @property
    def _kind(self) -> Literal["box"]: ...

    class BP(NamedTuple):
        ax: Axes
        lines: dict[str, Any]

    def __init__(self, data: pd.DataFrame | pd.Series, return_type: Optional[Literal["axes", "dict", "both"]] = "axes", **kwargs: Any) -> None: ...
    @classmethod
    def _plot(
        cls,
        ax: Axes,
        y: np.ndarray,
        column_num: int | None = ...,
        return_type: Optional[Literal["axes", "dict", "both"]] = "axes",
        **kwds: Any
    ) -> tuple[Axes | BoxPlot.BP | dict[str, Any], dict[str, Any]]: ...
    def _validate_color_args(self, color: object, colormap: object | None) -> object | None: ...
    @cache_readonly
    def _color_attrs(self) -> list[object]: ...
    @cache_readonly
    def _boxes_c(self) -> object: ...
    @cache_readonly
    def _whiskers_c(self) -> object: ...
    @cache_readonly
    def _medians_c(self) -> object: ...
    @cache_readonly
    def _caps_c(self) -> object: ...
    def _get_colors(self, num_colors: int | None = ..., color_kwds: str = "color") -> list[object] | None: ...
    def maybe_color_bp(self, bp: dict[str, Any]) -> None: ...
    def _make_plot(self, fig: Figure) -> None: ...
    def _make_legend(self) -> None: ...
    def _post_plot_logic(self, ax: Axes, data: pd.DataFrame | pd.Series) -> None: ...
    @property
    def orientation(self) -> Literal["vertical", "horizontal"]: ...
    @property
    def result(self) -> object: ...

def maybe_color_bp(bp: dict[str, Any], color_tup: tuple[object, object, object, object], **kwds: Any) -> None: ...
def _grouped_plot_by_column(
    plotf: Callable[..., object],
    data: pd.DataFrame,
    columns: Sequence[object] | None = ...,
    by: object | Sequence[object] | None = ...,
    numeric_only: bool = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    ax: Axes | np.ndarray | None = ...,
    layout: object | None = ...,
    return_type: Optional[Literal["axes", "dict", "both"]] = ...,
    **kwargs: Any
) -> pd.Series | Axes | np.ndarray: ...
def boxplot(
    data: pd.DataFrame | pd.Series,
    column: object | Sequence[object] | None = ...,
    by: object | Sequence[object] | None = ...,
    ax: Axes | None = ...,
    fontsize: float | int | None = ...,
    rot: float | int = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: object | None = ...,
    return_type: Optional[Literal["axes", "dict", "both"]] = ...,
    **kwds: Any
) -> pd.Series | Axes | dict[str, Any] | BoxPlot.BP | np.ndarray: ...
def boxplot_frame(
    self: pd.DataFrame,
    column: object | Sequence[object] | None = ...,
    by: object | Sequence[object] | None = ...,
    ax: Axes | None = ...,
    fontsize: float | int | None = ...,
    rot: float | int = ...,
    grid: bool = ...,
    figsize: tuple[float, float] | None = ...,
    layout: object | None = ...,
    return_type: Optional[Literal["axes", "dict", "both"]] = ...,
    **kwds: Any
) -> pd.Series | Axes | dict[str, Any] | BoxPlot.BP | np.ndarray: ...
def boxplot_frame_groupby(
    grouped: Iterable[tuple[object, pd.DataFrame]],
    subplots: bool = ...,
    column: object | Sequence[object] | None = ...,
    fontsize: float | int | None = ...,
    rot: float | int = ...,
    grid: bool = ...,
    ax: Axes | None = ...,
    figsize: tuple[float, float] | None = ...,
    layout: object | None = ...,
    sharex: bool = ...,
    sharey: bool = ...,
    **kwds: Any
) -> pd.Series | Axes: ...