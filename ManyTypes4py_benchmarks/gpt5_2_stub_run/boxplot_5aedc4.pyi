from typing import Any, Literal, NamedTuple, Optional, Tuple

def _set_ticklabels(ax: Any, labels: Any, is_vertical: Any, **kwargs: Any) -> None: ...

class BoxPlot(object):
    _layout_type: str
    _valid_return_types: Tuple[Optional[str], ...]

    class BP(NamedTuple):
        ...

    @property
    def _kind(self) -> str: ...
    def __init__(self, data: Any, return_type: Optional[Literal['axes', 'dict', 'both']] = 'axes', **kwargs: Any) -> None: ...
    @classmethod
    def _plot(cls, ax: Any, y: Any, column_num: Any = ..., return_type: Optional[Literal['axes', 'dict', 'both']] = 'axes', **kwds: Any) -> Tuple[Any, Any]: ...
    def _validate_color_args(self, color: Any, colormap: Any) -> Any: ...
    @property
    def _color_attrs(self) -> Any: ...
    @property
    def _boxes_c(self) -> Any: ...
    @property
    def _whiskers_c(self) -> Any: ...
    @property
    def _medians_c(self) -> Any: ...
    @property
    def _caps_c(self) -> Any: ...
    def _get_colors(self, num_colors: Any = ..., color_kwds: Any = ...) -> Any: ...
    def maybe_color_bp(self, bp: Any) -> None: ...
    def _make_plot(self, fig: Any) -> None: ...
    def _make_legend(self) -> None: ...
    def _post_plot_logic(self, ax: Any, data: Any) -> None: ...
    @property
    def orientation(self) -> str: ...
    @property
    def result(self) -> Any: ...

def maybe_color_bp(bp: Any, color_tup: Any, **kwds: Any) -> None: ...
def _grouped_plot_by_column(
    plotf: Any,
    data: Any,
    columns: Any = ...,
    by: Any = ...,
    numeric_only: bool = ...,
    grid: bool = ...,
    figsize: Any = ...,
    ax: Any = ...,
    layout: Any = ...,
    return_type: Optional[Literal['axes', 'dict', 'both']] = ...,
    **kwargs: Any
) -> Any: ...
def boxplot(
    data: Any,
    column: Any = ...,
    by: Any = ...,
    ax: Any = ...,
    fontsize: Any = ...,
    rot: int = ...,
    grid: bool = ...,
    figsize: Any = ...,
    layout: Any = ...,
    return_type: Optional[Literal['axes', 'dict', 'both']] = ...,
    **kwds: Any
) -> Any: ...
def boxplot_frame(
    self: Any,
    column: Any = ...,
    by: Any = ...,
    ax: Any = ...,
    fontsize: Any = ...,
    rot: int = ...,
    grid: bool = ...,
    figsize: Any = ...,
    layout: Any = ...,
    return_type: Optional[Literal['axes', 'dict', 'both']] = ...,
    **kwds: Any
) -> Any: ...
def boxplot_frame_groupby(
    grouped: Any,
    subplots: bool = ...,
    column: Any = ...,
    fontsize: Any = ...,
    rot: int = ...,
    grid: bool = ...,
    ax: Any = ...,
    figsize: Any = ...,
    layout: Any = ...,
    sharex: bool = ...,
    sharey: bool = ...,
    **kwds: Any
) -> Any: ...