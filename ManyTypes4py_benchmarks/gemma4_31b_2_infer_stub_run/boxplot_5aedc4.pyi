from __future__ import annotations
from typing import TYPE_CHECKING, Literal, NamedTuple, Union, Optional, Any, overload
import numpy as np
import pandas as pd
from pandas.core.dtypes.generic import ABCSeries

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from pandas._typing import MatplotlibColor

def _set_ticklabels(ax: Axes, labels: Sequence[str], is_vertical: bool, **kwargs: Any) -> None: ...

class BoxPlot:
    _layout_type: Literal['horizontal']
    _valid_return_types: tuple[Optional[Any], Literal['axes', 'dict', 'both']]

    class BP(NamedTuple):
        ax: Axes
        lines: dict[str, Any]

    def __init__(self, data: Any, return_type: Literal['axes', 'dict', 'both'] = 'axes', **kwargs: Any) -> None: ...

    @property
    def _kind(self) -> Literal['box']: ...

    @classmethod
    def _plot(cls, ax: Axes, y: Any, column_num: Optional[int] = None, return_type: Literal['axes', 'dict', 'both'] = 'axes', **kwds: Any) -> tuple[Any, dict[str, Any]]: ...

    def _validate_color_args(self, color: Any, colormap: Optional[Any]) -> Optional[Union[MatplotlibColor, dict[str, MatplotlibColor]]]: ...

    @property
    def _color_attrs(self) -> np.ndarray: ...

    @property
    def _boxes_c(self) -> MatplotlibColor: ...

    @property
    def _whiskers_c(self) -> MatplotlibColor: ...

    @property
    def _medians_c(self) -> MatplotlibColor: ...

    @property
    def _caps_c(self) -> MatplotlibColor: ...

    def _get_colors(self, num_colors: Optional[int] = None, color_kwds: str = 'color') -> None: ...

    def maybe_color_bp(self, bp: dict[str, Any]) -> None: ...

    def _make_plot(self, fig: Figure) -> None: ...

    def _make_legend(self) -> None: ...

    def _post_plot_logic(self, ax: Axes, data: Any) -> None: ...

    @property
    def orientation(self) -> Literal['vertical', 'horizontal']: ...

    @property
    def result(self) -> Any: ...

def maybe_color_bp(bp: dict[str, Any], color_tup: tuple[MatplotlibColor, ...], **kwds: Any) -> None: ...

def _grouped_plot_by_column(
    plotf: Any,
    data: pd.DataFrame,
    columns: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[str]]] = None,
    numeric_only: bool = True,
    grid: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    layout: Optional[Any] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = None,
    **kwargs: Any
) -> Union[pd.Series, Axes]: ...

def boxplot(
    data: Union[pd.DataFrame, ABCSeries],
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[float] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[Any] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = None,
    **kwds: Any
) -> Any: ...

def boxplot_frame(
    self: pd.DataFrame,
    column: Optional[Union[str, Sequence[str]]] = None,
    by: Optional[Union[str, Sequence[str]]] = None,
    ax: Optional[Axes] = None,
    fontsize: Optional[float] = None,
    rot: float = 0,
    grid: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[Any] = None,
    return_type: Optional[Literal['axes', 'dict', 'both']] = None,
    **kwds: Any
) -> Any: ...

def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = True,
    column: Optional[Union[str, Sequence[str]]] = None,
    fontsize: Optional[float] = None,
    rot: float = 0,
    grid: bool = True,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    layout: Optional[Any] = None,
    sharex: bool = False,
    sharey: bool = True,
    **kwds: Any
) -> Any: ...