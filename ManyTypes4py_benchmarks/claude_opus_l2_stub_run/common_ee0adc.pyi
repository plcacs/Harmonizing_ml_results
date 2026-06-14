from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pandas import Series


def _check_legend_labels(
    axes: Axes | Sequence[Axes] | np.ndarray,
    labels: Sequence[str] | None = ...,
    visible: bool = ...,
) -> None: ...

def _check_legend_marker(
    ax: Axes,
    expected_markers: list[Any] | None = ...,
    visible: bool = ...,
) -> None: ...

def _check_data(xp: Axes, rs: Axes) -> None: ...

def _check_visible(
    collections: Any,
    visible: bool = ...,
) -> None: ...

def _check_patches_all_filled(
    axes: Axes | Sequence[Axes] | np.ndarray,
    filled: bool = ...,
) -> None: ...

def _get_colors_mapped(
    series: Series,
    colors: Sequence[Any],
) -> list[Any]: ...

def _check_colors(
    collections: Sequence[Any],
    linecolors: Sequence[Any] | None = ...,
    facecolors: Sequence[Any] | None = ...,
    mapping: Series | None = ...,
) -> None: ...

def _check_text_labels(
    texts: Any,
    expected: str | Sequence[str],
) -> None: ...

def _check_ticks_props(
    axes: Axes | Sequence[Axes] | np.ndarray,
    xlabelsize: float | None = ...,
    xrot: float | None = ...,
    ylabelsize: float | None = ...,
    yrot: float | None = ...,
) -> None: ...

def _check_ax_scales(
    axes: Axes | Sequence[Axes] | np.ndarray,
    xaxis: str = ...,
    yaxis: str = ...,
) -> None: ...

def _check_axes_shape(
    axes: Axes | Sequence[Axes] | np.ndarray,
    axes_num: int | None = ...,
    layout: tuple[int, int] | None = ...,
    figsize: tuple[float, float] | None = ...,
) -> None: ...

def _flatten_visible(
    axes: Axes | Sequence[Axes] | np.ndarray,
) -> list[Axes]: ...

def _check_has_errorbars(
    axes: Axes | Sequence[Axes] | np.ndarray,
    xerr: int = ...,
    yerr: int = ...,
) -> None: ...

def _check_box_return_type(
    returned: Any,
    return_type: str | None,
    expected_keys: Sequence[str] | None = ...,
    check_ax_title: bool = ...,
) -> None: ...

def _check_grid_settings(
    obj: Any,
    kinds: Sequence[str],
    kws: dict[str, Any] | None = ...,
) -> None: ...

def _unpack_cycler(
    rcParams: dict[str, Any],
    field: str = ...,
) -> list[Any]: ...

def get_x_axis(ax: Axes) -> Any: ...

def get_y_axis(ax: Axes) -> Any: ...

def assert_is_valid_plot_return_object(objs: Any) -> None: ...

def _check_plot_works(
    f: Any,
    default_axes: bool = ...,
    **kwargs: Any,
) -> Any: ...

def _gen_default_plot(
    f: Any,
    fig: Figure,
    **kwargs: Any,
) -> Any: ...

def _gen_two_subplots(
    f: Any,
    fig: Figure,
    **kwargs: Any,
) -> Any: ...