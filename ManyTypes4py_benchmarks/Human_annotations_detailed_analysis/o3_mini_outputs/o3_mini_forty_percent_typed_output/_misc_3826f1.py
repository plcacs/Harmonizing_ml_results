from __future__ import annotations

from contextlib import contextmanager
from typing import (
    Any,
    Generator,
    Mapping,
    Optional,
    Tuple,
    Union,
    List,
)
from pandas.plotting._core import _get_plot_backend

if False:
    from collections.abc import Mapping as MappingType  # type: ignore
else:
    pass

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.table import Table
import numpy as np

from pandas import DataFrame, Series


def table(ax: Axes, data: DataFrame | Series, **kwargs: Any) -> Table:
    """
    Helper function to convert DataFrame and Series to matplotlib.table.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.table(
        ax=ax, data=data, rowLabels=None, colLabels=None, **kwargs
    )


def register() -> None:
    """
    Register pandas formatters and converters with matplotlib.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    plot_backend.register()


def deregister() -> None:
    """
    Remove pandas formatters and converters.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    plot_backend.deregister()


def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds: Optional[Mapping[str, Any]] = None,
    hist_kwds: Optional[Mapping[str, Any]] = None,
    range_padding: float = 0.05,
    **kwargs: Any,
) -> np.ndarray:
    """
    Draw a matrix of scatter plots.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.scatter_matrix(
        frame=frame,
        alpha=alpha,
        figsize=figsize,
        ax=ax,
        grid=grid,
        diagonal=diagonal,
        marker=marker,
        density_kwds=density_kwds,
        hist_kwds=hist_kwds,
        range_padding=range_padding,
        **kwargs,
    )


def radviz(
    frame: DataFrame,
    class_column: str,
    ax: Optional[Axes] = None,
    color: Optional[Union[List[str], Tuple[str, ...]]] = None,
    colormap: Optional[Union[str, Colormap]] = None,
    **kwds: Any,
) -> Axes:
    """
    Plot a multidimensional dataset in 2D.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.radviz(
        frame=frame,
        class_column=class_column,
        ax=ax,
        color=color,
        colormap=colormap,
        **kwds,
    )


def andrews_curves(
    frame: DataFrame,
    class_column: str,
    ax: Optional[Axes] = None,
    samples: int = 200,
    color: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    colormap: Optional[Union[str, Colormap]] = None,
    **kwargs: Any,
) -> Axes:
    """
    Generate a matplotlib plot for visualizing clusters of multivariate data.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.andrews_curves(
        frame=frame,
        class_column=class_column,
        ax=ax,
        samples=samples,
        color=color,
        colormap=colormap,
        **kwargs,
    )


def bootstrap_plot(
    series: Series,
    fig: Optional[Figure] = None,
    size: int = 50,
    samples: int = 500,
    **kwds: Any,
) -> Figure:
    """
    Bootstrap plot on mean, median and mid-range statistics.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.bootstrap_plot(
        series=series, fig=fig, size=size, samples=samples, **kwds
    )


def parallel_coordinates(
    frame: DataFrame,
    class_column: str,
    cols: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    color: Optional[Union[List[str], Tuple[str, ...]]] = None,
    use_columns: bool = False,
    xticks: Optional[Union[List[Any], Tuple[Any, ...]]] = None,
    colormap: Optional[Union[str, Colormap]] = None,
    axvlines: bool = True,
    axvlines_kwds: Optional[Mapping[str, Any]] = None,
    sort_labels: bool = False,
    **kwargs: Any,
) -> Axes:
    """
    Parallel coordinates plotting.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.parallel_coordinates(
        frame=frame,
        class_column=class_column,
        cols=cols,
        ax=ax,
        color=color,
        use_columns=use_columns,
        xticks=xticks,
        colormap=colormap,
        axvlines=axvlines,
        axvlines_kwds=axvlines_kwds,
        sort_labels=sort_labels,
        **kwargs,
    )


def lag_plot(series: Series, lag: int = 1, ax: Optional[Axes] = None, **kwds: Any) -> Axes:
    """
    Lag plot for time series.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.lag_plot(series=series, lag=lag, ax=ax, **kwds)


def autocorrelation_plot(series: Series, ax: Optional[Axes] = None, **kwargs: Any) -> Axes:
    """
    Autocorrelation plot for time series.
    
    (Docstring unchanged.)
    """
    plot_backend = _get_plot_backend("matplotlib")
    return plot_backend.autocorrelation_plot(series=series, ax=ax, **kwargs)


class _Options(dict):
    """
    Stores pandas plotting options.
    
    (Docstring unchanged.)
    """

    # alias so the names are same as plotting method parameter names
    _ALIASES: Mapping[str, str] = {"x_compat": "xaxis.compat"}
    _DEFAULT_KEYS: List[str] = ["xaxis.compat"]

    def __init__(self) -> None:
        super().__setitem__("xaxis.compat", False)

    def __getitem__(self, key: str) -> Any:
        key = self._get_canonical_key(key)
        if key not in self:
            raise ValueError(f"{key} is not a valid pandas plotting option")
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        key = self._get_canonical_key(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        key = self._get_canonical_key(key)
        if key in self._DEFAULT_KEYS:
            raise ValueError(f"Cannot remove default parameter {key}")
        super().__delitem__(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        key = self._get_canonical_key(key)
        return super().__contains__(key)

    def reset(self) -> None:
        """
        Reset the option store to its initial state

        Returns
        -------
        None
        """
        self.__init__()  # type: ignore[misc]

    def _get_canonical_key(self, key: str) -> str:
        return self._ALIASES.get(key, key)

    @contextmanager
    def use(self, key: str, value: Any) -> Generator[_Options, None, None]:
        """
        Temporarily set a parameter value using the with statement.
        Aliasing allowed.
        """
        old_value = self[key]
        try:
            self[key] = value
            yield self
        finally:
            self[key] = old_value


plot_params: _Options = _Options()