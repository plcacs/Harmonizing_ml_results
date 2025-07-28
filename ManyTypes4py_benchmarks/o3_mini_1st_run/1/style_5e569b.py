#!/usr/bin/env python3
import copy
import operator
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib  # type: ignore


def _validate_apply_axis_arg(
    arg: Any, arg_name: str, dtype: Optional[Any], data: Union[Series, DataFrame]
) -> np.ndarray:
    """
    For the apply-type methods, ``axis=None`` creates ``data`` as DataFrame, and for
    ``axis=[1,0]`` it creates a Series. Where ``arg`` is expected as an element
    of some operator with ``data`` we must make sure that the two are compatible shapes,
    or raise.

    Parameters
    ----------
    arg : sequence, Series or DataFrame
        the user input arg
    arg_name : string
        name of the arg for use in error messages
    dtype : numpy dtype, optional
        forced numpy dtype if given
    data : Series or DataFrame
        underlying subset of Styler data on which operations are performed

    Returns
    -------
    ndarray
    """
    dtype_kw = {'dtype': dtype} if dtype else {}
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(
            f"'{arg_name}' is a Series but underlying data for operations is a DataFrame since 'axis=None'"
        )
    if isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(
            f"'{arg_name}' is a DataFrame but underlying data for operations is a Series with 'axis in [0,1]'"
        )
    if isinstance(arg, (Series, DataFrame)):
        arg = arg.reindex_like(data).to_numpy(**dtype_kw)
    else:
        arg = np.asarray(arg, **dtype_kw)
        assert isinstance(arg, np.ndarray)
        if arg.shape != data.shape:
            raise ValueError(
                f"supplied '{arg_name}' is not correct shape for data over selected 'axis': got {arg.shape}, expected {data.shape}"
            )
    return arg


def _background_gradient(
    data: Union[Series, DataFrame],
    cmap: Optional[Union[str, Any]] = "PuBu",
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gmap: Optional[Any] = None,
    text_only: bool = False,
) -> Union[List[str], DataFrame]:
    """
    Color background in a range according to the data or a gradient map.
    """
    if gmap is None:
        gmap = data.to_numpy(dtype=float, na_value=np.nan)
    else:
        gmap = _validate_apply_axis_arg(gmap, 'gmap', float, data)
    smin: float = np.nanmin(gmap) if vmin is None else vmin
    smax: float = np.nanmax(gmap) if vmax is None else vmax
    rng: float = smax - smin
    _matplotlib = matplotlib  # using imported matplotlib
    norm = _matplotlib.colors.Normalize(smin - rng * low, smax + rng * high)
    if cmap is None:
        rgbas = _matplotlib.colormaps[_matplotlib.rcParams["image.cmap"]](norm(gmap))
    else:
        rgbas = _matplotlib.colormaps.get_cmap(cmap)(norm(gmap))

    def relative_luminance(rgba: Sequence[float]) -> float:
        """
        Calculate relative luminance of a color.

        The calculation adheres to the W3C standards
        (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

        Parameters
        ----------
        rgba : rgb or rgba tuple

        Returns
        -------
        float
            The relative luminance as a value from 0 to 1
        """
        def convert_channel(x: float) -> float:
            return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

        r, g, b = (convert_channel(x) for x in rgba[:3])
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba: Sequence[float], text_only_flag: bool) -> str:
        if not text_only_flag:
            dark: bool = relative_luminance(rgba) < text_color_threshold
            text_color: str = "#f1f1f1" if dark else "#000000"
            return f"background-color: {_matplotlib.colors.rgb2hex(rgba)};color: {text_color};"
        else:
            return f"color: {_matplotlib.colors.rgb2hex(rgba)};"

    if data.ndim == 1:
        return [css(rgba, text_only) for rgba in rgbas]
    else:
        return DataFrame(
            [[css(rgba, text_only) for rgba in row] for row in rgbas],
            index=data.index,
            columns=data.columns,
        )


def _highlight_between(
    data: Union[Series, DataFrame],
    props: str,
    left: Optional[Any] = None,
    right: Optional[Any] = None,
    inclusive: Union[str, bool] = True,
) -> np.ndarray:
    """
    Return an array of css props based on condition of data values within given range.
    """
    if np.iterable(left) and (not isinstance(left, str)):
        left = _validate_apply_axis_arg(left, 'left', None, data)
    if np.iterable(right) and (not isinstance(right, str)):
        right = _validate_apply_axis_arg(right, 'right', None, data)
    if inclusive == "both":
        ops = (operator.ge, operator.le)
    elif inclusive == "neither":
        ops = (operator.gt, operator.lt)
    elif inclusive == "left":
        ops = (operator.ge, operator.lt)
    elif inclusive == "right":
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(f"'inclusive' values can be 'both', 'left', 'right', or 'neither' got {inclusive}")
    g_left = ops[0](data, left) if left is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(g_left, (DataFrame, Series)):
        g_left = g_left.where(pd.notna(g_left), False)
    l_right = ops[1](data, right) if right is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(l_right, (DataFrame, Series)):
        l_right = l_right.where(pd.notna(l_right), False)
    return np.where(g_left & l_right, props, "")


def _highlight_value(
    data: Union[Series, DataFrame],
    op: str,
    props: str,
) -> np.ndarray:
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(skipna=True)
    if isinstance(data, DataFrame):
        value = getattr(value, op)(skipna=True)
    cond = data == value
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, "")


def _bar(
    data: Union[Series, DataFrame],
    align: Union[str, int, float, Callable[[np.ndarray], float]],
    colors: Union[str, Sequence[str]],
    cmap: Optional[Union[str, Any]],
    width: float,
    height: float,
    vmin: Optional[float],
    vmax: Optional[float],
    base_css: str,
) -> Union[List[str], np.ndarray]:
    """
    Draw bar chart in data cells using HTML CSS linear gradient.

    Parameters
    ----------
    data : Series or DataFrame
        Underlying subset of Styler data on which operations are performed.
    align : {"left", "right", "mid", "zero", "mean"} or numeric or callable
        Method for how bars are structured or scalar value of centre point.
    colors : str or sequence of str
        If a str is passed, the color is the same for both negative and positive numbers.
        If a sequence is used, the first element is for negative and the second for positive.
    cmap : str or colormap, optional
        A string name of a matplotlib Colormap, or a Colormap object. Cannot be
        used together with ``color``.
    width : float
        The percentage of the cell (as a fraction of 1) in which to draw the bars.
    height : float
        The percentage height of the bar in the cell (as a fraction of 1), centrally aligned.
    vmin : float, optional
        Minimum bar value.
    vmax : float, optional
        Maximum bar value.
    base_css : str
        Additional CSS that is included in the cell before bars are drawn.

    Returns
    -------
    list or ndarray of str
        The CSS strings for each cell.
    """
    def css_bar(start: float, end: float, color: str) -> str:
        """
        Generate CSS code to draw a bar from start to end in a table cell.
        """
        cell_css: str = base_css
        if end > start:
            cell_css += "background: linear-gradient(90deg,"
            if start > 0:
                cell_css += f" transparent {start * 100:.1f}%, {color} {start * 100:.1f}%,"
            cell_css += f" {color} {end * 100:.1f}%, transparent {end * 100:.1f}%)"
        return cell_css

    def css_calc(x: float, left_val: float, right_val: float, align_str: str, color_val: str) -> str:
        """
        Return the correct CSS for bar placement based on calculated values.
        """
        if pd.isna(x):
            return base_css
        if isinstance(color_val, (list, tuple)):
            # Should not happen because we convert earlier, but just in case.
            color_sel: str = color_val[0] if x < 0 else color_val[1]
        else:
            color_sel = color_val
        x = left_val if x < left_val else x
        x = right_val if x > right_val else x
        start: float = 0.0
        end: float = 1.0
        if align_str == "left":
            end = (x - left_val) / (right_val - left_val)
        elif align_str == "right":
            start = (x - left_val) / (right_val - left_val)
        else:
            z_frac: float = 0.5
            if align_str == "zero":
                limit = max(abs(left_val), abs(right_val))
                left_val, right_val = -limit, limit
            elif align_str == "mid":
                mid = (left_val + right_val) / 2
                z_frac = (-mid / (right_val - left_val) + 0.5) if mid < 0 else (-left_val / (right_val - left_val))
            if x < 0:
                start, end = ((x - left_val) / (right_val - left_val), z_frac)
            else:
                start, end = (z_frac, (x - left_val) / (right_val - left_val))
        ret: str = css_bar(start * width, end * width, color_sel)
        if height < 1 and "background: linear-gradient(" in ret:
            return ret + f" no-repeat center; background-size: 100% {height * 100:.1f}%;"
        else:
            return ret

    values = data.to_numpy()
    left: float = np.nanmin(data.min(skipna=True)) if vmin is None else vmin
    right: float = np.nanmax(data.max(skipna=True)) if vmax is None else vmax
    z: float = 0.0
    if align == "mid":
        if left >= 0:
            align, left = "left", (0 if vmin is None else vmin)
        elif right <= 0:
            align, right = "right", (0 if vmax is None else vmax)
    elif align == "mean":
        z, align = np.nanmean(values), "zero"
    elif callable(align):
        z, align = align(values), "zero"
    elif isinstance(align, (float, int)):
        z, align = float(align), "zero"
    elif align not in ("left", "right", "zero"):
        raise ValueError("`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or be a value defining the center line or a callable that returns a float")
    rgbas: Optional[Union[List[str], List[List[str]]]] = None
    if cmap is not None:
        _matplotlib = matplotlib
        if isinstance(cmap, str):
            cmap = _matplotlib.colormaps[cmap]
        norm = _matplotlib.colors.Normalize(left, right)
        rgb_array = cmap(norm(values))
        if data.ndim == 1:
            rgbas = [_matplotlib.colors.rgb2hex(rgba) for rgba in rgb_array]
        else:
            rgbas = [
                [_matplotlib.colors.rgb2hex(rgba) for rgba in row] for row in rgb_array
            ]
    # Now compute the CSS for each value.
    if data.ndim == 1:
        return [
            css_calc(x - z, left - z, right - z, align, colors if rgbas is None else (rgbas[i]))
            for i, x in enumerate(values)
        ]
    else:
        return np.array(
            [
                [
                    css_calc(x - z, left - z, right - z, align, colors if rgbas is None else rgbas[i][j])
                    for j, x in enumerate(row)
                ]
                for i, row in enumerate(values)
            ]
        )
