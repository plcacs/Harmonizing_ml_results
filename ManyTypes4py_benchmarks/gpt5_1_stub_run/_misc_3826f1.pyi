from typing import Any, ContextManager, Optional, Tuple

def table(ax: Any, data: Any, **kwargs: Any) -> Any: ...
def register() -> None: ...
def deregister() -> None: ...
def scatter_matrix(
    frame: Any,
    alpha: float = ...,
    figsize: Optional[Tuple[float, float]] = ...,
    ax: Optional[Any] = ...,
    grid: bool = ...,
    diagonal: str = ...,
    marker: str = ...,
    density_kwds: Optional[Any] = ...,
    hist_kwds: Optional[Any] = ...,
    range_padding: float = ...,
    **kwargs: Any
) -> Any: ...
def radviz(
    frame: Any,
    class_column: Any,
    ax: Optional[Any] = ...,
    color: Optional[Any] = ...,
    colormap: Optional[Any] = ...,
    **kwds: Any
) -> Any: ...
def andrews_curves(
    frame: Any,
    class_column: Any,
    ax: Optional[Any] = ...,
    samples: int = ...,
    color: Optional[Any] = ...,
    colormap: Optional[Any] = ...,
    **kwargs: Any
) -> Any: ...
def bootstrap_plot(
    series: Any,
    fig: Optional[Any] = ...,
    size: int = ...,
    samples: int = ...,
    **kwds: Any
) -> Any: ...
def parallel_coordinates(
    frame: Any,
    class_column: Any,
    cols: Optional[Any] = ...,
    ax: Optional[Any] = ...,
    color: Optional[Any] = ...,
    use_columns: bool = ...,
    xticks: Optional[Any] = ...,
    colormap: Optional[Any] = ...,
    axvlines: bool = ...,
    axvlines_kwds: Optional[Any] = ...,
    sort_labels: bool = ...,
    **kwargs: Any
) -> Any: ...
def lag_plot(series: Any, lag: int = ..., ax: Optional[Any] = ..., **kwds: Any) -> Any: ...
def autocorrelation_plot(series: Any, ax: Optional[Any] = ..., **kwargs: Any) -> Any: ...

class _Options(dict[str, Any]):
    _ALIASES: dict[str, str]
    _DEFAULT_KEYS: list[str]
    def __init__(self) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def reset(self) -> None: ...
    def _get_canonical_key(self, key: str) -> str: ...
    def use(self, key: str, value: Any) -> ContextManager["_Options"]: ...

plot_params: _Options ...