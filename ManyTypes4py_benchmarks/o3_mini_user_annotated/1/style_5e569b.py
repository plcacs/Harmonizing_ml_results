from __future__ import annotations

import copy
import operator
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from matplotlib.colors import Colormap
import matplotlib

# -----------------------------------------------------------------------
# Helper functions with type annotations

def _validate_apply_axis_arg(
    arg: Union[pd.Series, pd.DataFrame, Sequence[Any], np.ndarray],
    arg_name: str,
    dtype: Optional[Any],
    data: Union[pd.Series, pd.DataFrame],
) -> np.ndarray:
    """
    For the apply-type methods, ``axis=None`` creates ``data`` as DataFrame, and for
    ``axis=[1,0]`` it creates a Series. Where ``arg`` is expected as an element
    of some operator with ``data`` we must make sure that the two are compatible shapes,
    or raise.
    """
    dtype_kwargs = {"dtype": dtype} if dtype else {}
    if isinstance(arg, pd.Series) and isinstance(data, pd.DataFrame):
        raise ValueError(
            f"'{arg_name}' is a Series but underlying data for operations "
            f"is a DataFrame since 'axis=None'"
        )
    if isinstance(arg, pd.DataFrame) and isinstance(data, pd.Series):
        raise ValueError(
            f"'{arg_name}' is a DataFrame but underlying data for "
            f"operations is a Series with 'axis in [0,1]'"
        )
    if isinstance(arg, (pd.Series, pd.DataFrame)):
        arg_aligned = arg.reindex_like(data)
        arg_array = arg_aligned.to_numpy(**dtype_kwargs)
    else:
        arg_array = np.asarray(arg, **dtype_kwargs)
        if arg_array.shape != data.shape:
            raise ValueError(
                f"supplied '{arg_name}' is not correct shape for data over "
                f"selected 'axis': got {arg_array.shape}, expected {data.shape}"
            )
    return arg_array


def _background_gradient(
    data: Union[pd.Series, pd.DataFrame],
    cmap: Union[str, Colormap] = "PuBu",
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gmap: Optional[Union[Sequence[Any], np.ndarray, pd.DataFrame, pd.Series]] = None,
    text_only: bool = False,
) -> Union[List[str], pd.DataFrame]:
    """
    Color background in a range according to the data or a gradient map.
    """
    if gmap is None:
        gmap_array = data.to_numpy(dtype=float, na_value=np.nan)
    else:
        gmap_array = _validate_apply_axis_arg(gmap, "gmap", float, data)
    smin: float = np.nanmin(gmap_array) if vmin is None else vmin
    smax: float = np.nanmax(gmap_array) if vmax is None else vmax
    rng: float = smax - smin
    norm = matplotlib.colors.Normalize(smin - (rng * low), smax + (rng * high))
    if cmap is None:
        rgbas = matplotlib.colormaps[matplotlib.rcParams["image.cmap"]](norm(gmap_array))
    else:
        if isinstance(cmap, str):
            rgbas = matplotlib.colormaps.get_cmap(cmap)(norm(gmap_array))
        else:
            rgbas = cmap(norm(gmap_array))

    def relative_luminance(rgba: Tuple[float, float, float, float]) -> float:
        r, g, b = (
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
            for x in rgba[:3]
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba: Tuple[float, float, float, float], text_only: bool) -> str:
        if not text_only:
            dark: bool = relative_luminance(rgba) < text_color_threshold
            text_color: str = "#f1f1f1" if dark else "#000000"
            return f"background-color: {matplotlib.colors.rgb2hex(rgba)};color: {text_color};"
        else:
            return f"color: {matplotlib.colors.rgb2hex(rgba)};"

    if data.ndim == 1:
        return [css(rgba, text_only) for rgba in rgbas]
    else:
        return pd.DataFrame(
            [[css(rgba, text_only) for rgba in row] for row in rgbas],
            index=data.index,
            columns=data.columns,
        )


def _highlight_between(
    data: Union[pd.Series, pd.DataFrame],
    props: str,
    left: Optional[Union[Any, Sequence[Any], np.ndarray, pd.DataFrame]] = None,
    right: Optional[Union[Any, Sequence[Any], np.ndarray, pd.DataFrame]] = None,
    inclusive: Union[bool, str] = True,
) -> np.ndarray:
    """
    Return an array of css props based on condition of data values within given range.
    """
    if np.iterable(left) and not isinstance(left, str):
        left_array = _validate_apply_axis_arg(left, "left", None, data)
    else:
        left_array = left
    if np.iterable(right) and not isinstance(right, str):
        right_array = _validate_apply_axis_arg(right, "right", None, data)
    else:
        right_array = right
    if inclusive == "both":
        ops = (operator.ge, operator.le)
    elif inclusive == "neither":
        ops = (operator.gt, operator.lt)
    elif inclusive == "left":
        ops = (operator.ge, operator.lt)
    elif inclusive == "right":
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(
            f"'inclusive' values can be 'both', 'left', 'right', or 'neither'; got {inclusive}"
        )
    g_left = ops[0](data, left_array) if left_array is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(g_left, (pd.DataFrame, pd.Series)):
        g_left = g_left.where(pd.notna(g_left), False)
    l_right = ops[1](data, right_array) if right_array is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(l_right, (pd.DataFrame, pd.Series)):
        l_right = l_right.where(pd.notna(l_right), False)
    return np.where(g_left & l_right, props, "")


def _highlight_value(
    data: Union[pd.Series, pd.DataFrame], op: str, props: str
) -> np.ndarray:
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(skipna=True)
    if isinstance(data, pd.DataFrame):
        value = getattr(value, op)(skipna=True)
    cond = data == value
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, "")


def _bar(
    data: Union[pd.Series, pd.DataFrame],
    align: Union[str, float, Callable[[np.ndarray], float]],
    colors: Union[str, List[str], Tuple[str, str]],
    cmap: Optional[Union[str, Colormap]],
    width: float,
    height: float,
    vmin: Optional[float],
    vmax: Optional[float],
    base_css: str,
) -> Union[List[str], np.ndarray]:
    """
    Draw bar chart in data cells using HTML CSS linear gradient.
    """
    def css_bar(start: float, end: float, color: str) -> str:
        cell_css: str = base_css
        if end > start:
            cell_css += "background: linear-gradient(90deg,"
            if start > 0:
                cell_css += f" transparent {start * 100:.1f}%, {color} {start * 100:.1f}%,"
            cell_css += f" {color} {end * 100:.1f}%, transparent {end * 100:.1f}%)"
        return cell_css

    def css_calc(
        x: float, left: float, right: float, align: str, color: Union[str, List[str], Tuple[str, str]]
    ) -> str:
        if pd.isna(x):
            return base_css
        if isinstance(color, (list, tuple)):
            color = color[0] if x < 0 else color[1]
        x = left if x < left else x
        x = right if x > right else x
        start: float = 0.0
        end: float = 1.0
        if align == "left":
            end = (x - left) / (right - left)
        elif align == "right":
            start = (x - left) / (right - left)
        else:
            z_frac: float = 0.5
            if align == "zero":
                limit: float = max(abs(left), abs(right))
                left, right = -limit, limit
            elif align == "mid":
                mid: float = (left + right) / 2
                z_frac = (-mid / (right - left) + 0.5) if mid < 0 else (-left / (right - left))
            if x < 0:
                start, end = (x - left) / (right - left), z_frac
            else:
                start, end = z_frac, (x - left) / (right - left)
        ret: str = css_bar(start * width, end * width, color)
        if height < 1 and "background: linear-gradient(" in ret:
            return ret + f" no-repeat center; background-size: 100% {height * 100:.1f}%;"
        else:
            return ret

    values: np.ndarray = data.to_numpy()
    left: float = np.nanmin(data.min(skipna=True)) if vmin is None else vmin
    right: float = np.nanmax(data.max(skipna=True)) if vmax is None else vmax
    z: float = 0.0
    if align == "mid":
        if left >= 0:
            align, left = "left", 0 if vmin is None else vmin
        elif right <= 0:
            align, right = "right", 0 if vmax is None else vmax
    elif align == "mean":
        z, align = float(np.nanmean(values)), "zero"
    elif callable(align):
        z, align = float(align(values)), "zero"
    elif isinstance(align, (float, int)):
        z, align = float(align), "zero"
    elif align not in ("left", "right", "zero"):
        raise ValueError(
            "`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or be a "
            "value defining the center line or a callable that returns a float"
        )
    rgbas: Optional[Union[List[str], List[List[str]]]] = None
    if cmap is not None:
        if isinstance(cmap, str):
            cmap_obj: Colormap = matplotlib.colormaps.get_cmap(cmap)
        else:
            cmap_obj = cmap
        norm = matplotlib.colors.Normalize(left, right)
        cmap_values = cmap_obj(norm(values))
        if data.ndim == 1:
            rgbas = [matplotlib.colors.rgb2hex(rgba) for rgba in cmap_values]
        else:
            rgbas = [[matplotlib.colors.rgb2hex(rgba) for rgba in row] for row in cmap_values]
    if data.ndim == 1:
        return [
            css_calc(
                x - z,
                left - z,
                right - z,
                align,
                colors if rgbas is None else rgbas[i],
            )
            for i, x in enumerate(values)
        ]
    else:
        return np.array(
            [
                [
                    css_calc(
                        x - z,
                        left - z,
                        right - z,
                        align,
                        colors if rgbas is None else rgbas[i][j],
                    )
                    for j, x in enumerate(row)
                ]
                for i, row in enumerate(values)
            ]
        )


# The rest of the module (e.g., class methods for Styler) remains unchanged.
# Only the helper functions above have been annotated with appropriate type hints.
