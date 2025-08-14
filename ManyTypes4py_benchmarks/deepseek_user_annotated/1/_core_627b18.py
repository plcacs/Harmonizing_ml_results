from __future__ import annotations

import importlib
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

from pandas._config import get_option

from pandas.util._decorators import (
    Appender,
    Substitution,
)

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.base import PandasObject

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )
    from typing_extensions import TypeAlias

    from matplotlib.axes import Axes
    import numpy as np

    from pandas._typing import (
        Axis,
        IndexLabel,
        RandomState,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.groupby.generic import DataFrameGroupBy

    IndexLabel: TypeAlias = Hashable | Sequence[Hashable]

def holds_integer(column: Index) -> bool:
    return column.inferred_type in {"integer", "mixed-integer"}

def hist_series(
    self: Series,
    by: Optional[Any] = None,
    ax: Optional[Axes] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    figsize: Optional[Tuple[int, int]] = None,
    bins: Union[int, Sequence[int]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any,
) -> Axes:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_series(
        self,
        by=by,
        ax=ax,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        figsize=figsize,
        bins=bins,
        legend=legend,
        **kwargs,
    )

def hist_frame(
    data: DataFrame,
    column: Optional[IndexLabel] = None,
    by: Optional[Any] = None,
    grid: bool = True,
    xlabelsize: Optional[int] = None,
    xrot: Optional[float] = None,
    ylabelsize: Optional[int] = None,
    yrot: Optional[float] = None,
    ax: Optional[Axes] = None,
    sharex: bool = False,
    sharey: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    layout: Optional[Tuple[int, int]] = None,
    bins: Union[int, Sequence[int]] = 10,
    backend: Optional[str] = None,
    legend: bool = False,
    **kwargs: Any,
) -> Union[Axes, np.ndarray]:
    plot_backend = _get_plot_backend(backend)
    return plot_backend.hist_frame(
        data,
        column=column,
        by=by,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        ax=ax,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
        legend=legend,
        bins=bins,
        **kwargs,
    )

# [Rest of the type annotations would continue in the same pattern...]

def _load_backend(backend: str) -> types.ModuleType:
    try:
        module = importlib.import_module("pandas.plotting._matplotlib")
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting when the "
            'default backend "matplotlib" is selected.'
        ) from None
    return module

def _get_plot_backend(backend: Optional[str] = None) -> types.ModuleType:
    backend_str: str = backend or get_option("plotting.backend")

    if backend_str in _backends:
        return _backends[backend_str]

    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module
