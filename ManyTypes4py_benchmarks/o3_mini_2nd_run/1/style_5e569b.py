from __future__ import annotations
import copy
import operator
import numpy as np
import pandas as pd
from typing import Any, Callable, List, Optional, Sequence, Union


def _validate_apply_axis_arg(arg: Any, arg_name: str, dtype: Optional[Any], data: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
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
        underling subset of Styler data on which operations are performed

    Returns
    -------
    ndarray
    """
    dtype_kwargs = {'dtype': dtype} if dtype else {}
    if isinstance(arg, pd.Series) and isinstance(data, pd.DataFrame):
        raise ValueError(f"'{arg_name}' is a Series but underlying data for operations is a DataFrame since 'axis=None'")
    if isinstance(arg, pd.DataFrame) and isinstance(data, pd.Series):
        raise ValueError(f"'{arg_name}' is a DataFrame but underlying data for operations is a Series with 'axis in [0,1]'")
    if isinstance(arg, (pd.Series, pd.DataFrame)):
        arg = arg.reindex_like(data).to_numpy(**dtype_kwargs)
    else:
        arg = np.asarray(arg, **dtype_kwargs)
        assert isinstance(arg, np.ndarray)
        if arg.shape != data.shape:
            raise ValueError(f"supplied '{arg_name}' is not correct shape for data over selected 'axis': got {arg.shape}, expected {data.shape}")
    return arg


def _background_gradient(
    data: Union[pd.Series, pd.DataFrame],
    cmap: Union[str, Any] = 'PuBu',
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gmap: Optional[Any] = None,
    text_only: bool = False
) -> Union[List[str], pd.DataFrame]:
    """
    Color background in a range according to the data or a gradient map
    """
    if gmap is None:
        gmap_array = data.to_numpy(dtype=float, na_value=np.nan)
    else:
        gmap_array = _validate_apply_axis_arg(gmap, 'gmap', float, data)
    smin = np.nanmin(gmap_array) if vmin is None else vmin
    smax = np.nanmax(gmap_array) if vmax is None else vmax
    rng = smax - smin
    _matplotlib = __import__('matplotlib')
    norm = _matplotlib.colors.Normalize(smin - rng * low, smax + rng * high)
    if cmap is None:
        rgbas = _matplotlib.colormaps[_matplotlib.rcParams['image.cmap']](norm(gmap_array))
    else:
        rgbas = _matplotlib.colormaps.get_cmap(cmap)(norm(gmap_array))

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
        r, g, b = (
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
            for x in rgba[:3]
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba: Sequence[float], text_only: bool) -> str:
        if not text_only:
            dark = relative_luminance(rgba) < text_color_threshold
            text_color = '#f1f1f1' if dark else '#000000'
            return f'background-color: {_matplotlib.colors.rgb2hex(rgba)};color: {text_color};'
        else:
            return f'color: {_matplotlib.colors.rgb2hex(rgba)};'

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
    left: Optional[Any] = None,
    right: Optional[Any] = None,
    inclusive: str = 'both'
) -> np.ndarray:
    """
    Return an array of css props based on condition of data values within given range.
    """
    if np.iterable(left) and (not isinstance(left, str)):
        left = _validate_apply_axis_arg(left, 'left', None, data)
    if np.iterable(right) and (not isinstance(right, str)):
        right = _validate_apply_axis_arg(right, 'right', None, data)
    if inclusive == 'both':
        ops = (operator.ge, operator.le)
    elif inclusive == 'neither':
        ops = (operator.gt, operator.lt)
    elif inclusive == 'left':
        ops = (operator.ge, operator.lt)
    elif inclusive == 'right':
        ops = (operator.gt, operator.le)
    else:
        raise ValueError(f"'inclusive' values can be 'both', 'left', 'right', or 'neither' got {inclusive}")
    g_left = ops[0](data, left) if left is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(g_left, (pd.DataFrame, pd.Series)):
        g_left = g_left.where(pd.notna(g_left), False)
    l_right = ops[1](data, right) if right is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(l_right, (pd.DataFrame, pd.Series)):
        l_right = l_right.where(pd.notna(l_right), False)
    return np.where(g_left & l_right, props, '')


def _highlight_value(
    data: Union[pd.Series, pd.DataFrame],
    op: str,
    props: str
) -> np.ndarray:
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(skipna=True)
    if isinstance(data, pd.DataFrame):
        value = getattr(value, op)(skipna=True)
    cond = data == value
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, '')


def _bar(
    data: Union[pd.Series, pd.DataFrame],
    align: Union[str, int, float, Callable[[np.ndarray], float]],
    colors: Union[str, Sequence[str]],
    cmap: Optional[Union[str, Any]],
    width: float,
    height: float,
    vmin: Optional[float],
    vmax: Optional[float],
    base_css: str
) -> Union[List[str], np.ndarray]:
    """
    Draw bar chart in data cells using HTML CSS linear gradient.

    Parameters
    ----------
    data : Series or DataFrame
        Underling subset of Styler data on which operations are performed.
    align : str in {"left", "right", "mid", "zero", "mean"}, int, float, callable
        Method for how bars are structured or scalar value of centre point.
    colors : list-like of str
        Either a string or a list/tuple of two strings representing colors.
    cmap : str, matplotlib.cm.ColorMap or None
        A string name of a matplotlib Colormap, or a Colormap object. Cannot be
        used together with ``color``.
    width : float
        The percentage of the cell, measured from left, where drawn bars will reside (as a fraction 0 to 1).
    height : float
        The percentage of the cell's height where drawn bars will reside, centrally aligned (as a fraction 0 to 1).
    vmin : float, optional
        Minimum bar value, defining the left hand limit of the bar drawing range.
    vmax : float, optional
        Maximum bar value, defining the right hand limit of the bar drawing range.
    base_css : str
        Additional CSS that is included in the cell before bars are drawn.
    """
    def css_bar(start: float, end: float, color: str) -> str:
        """
        Generate CSS code to draw a bar from start to end in a table cell.

        Uses linear-gradient.

        Parameters
        ----------
        start : float
            Relative positional start of bar coloring in [0,1]
        end : float
            Relative positional end of the bar coloring in [0,1]
        color : str
            CSS valid color to apply.

        Returns
        -------
        str : The CSS applicable to the cell.
        """
        cell_css: str = base_css
        if end > start:
            cell_css += 'background: linear-gradient(90deg,'
            if start > 0:
                cell_css += f' transparent {start * 100:.1f}%, {color} {start * 100:.1f}%,'
            cell_css += f' {color} {end * 100:.1f}%, transparent {end * 100:.1f}%)'
        return cell_css

    def css_calc(x: float, left_val: float, right_val: float, align_str: str, color: str) -> str:
        """
        Return the correct CSS for bar placement based on calculated values.

        Parameters
        ----------
        x : float
            Value which determines the bar placement.
        left_val : float
            Value marking the left side of calculation, usually minimum of data.
        right_val : float
            Value marking the right side of the calculation, usually maximum of data.
        align_str : str
            How the bars will be positioned. Can be "left", "right", "zero", or "mid".
        color : str
            The CSS color to apply.

        Returns
        -------
        str : Resultant CSS with linear gradient.
        """
        if pd.isna(x):
            return base_css
        if isinstance(colors, (list, tuple)):
            bar_color: str = colors[0] if x < 0 else colors[1]
        else:
            bar_color = colors  # type: ignore
        x = left_val if x < left_val else x
        x = right_val if x > right_val else x
        start: float = 0
        end: float = 1
        if align_str == 'left':
            end = (x - left_val) / (right_val - left_val)
        elif align_str == 'right':
            start = (x - left_val) / (right_val - left_val)
        else:
            z_frac: float = 0.5
            if align_str == 'zero':
                limit = max(abs(left_val), abs(right_val))
                left_val, right_val = (-limit, limit)
            elif align_str == 'mid':
                mid = (left_val + right_val) / 2
                z_frac = -mid / (right_val - left_val) + 0.5 if mid < 0 else -left_val / (right_val - left_val)
            if x < 0:
                start, end = ((x - left_val) / (right_val - left_val), z_frac)
            else:
                start, end = (z_frac, (x - left_val) / (right_val - left_val))
        ret: str = css_bar(start * width, end * width, bar_color)
        if height < 1 and 'background: linear-gradient(' in ret:
            return ret + f' no-repeat center; background-size: 100% {height * 100:.1f}%;'
        else:
            return ret

    values = data.to_numpy()
    left_val: float = np.nanmin(data.min(skipna=True)) if vmin is None else vmin
    right_val: float = np.nanmax(data.max(skipna=True)) if vmax is None else vmax
    z: float = 0
    if align == 'mid':
        if left_val >= 0:
            align = 'left'
            left_val = 0 if vmin is None else vmin
        elif right_val <= 0:
            align = 'right'
            right_val = 0 if vmax is None else vmax
    elif align == 'mean':
        z = float(np.nanmean(values))
        align = 'zero'
    elif callable(align):
        z = align(values)
        align = 'zero'
    elif isinstance(align, (float, int)):
        z = float(align)
        align = 'zero'
    elif align not in ('left', 'right', 'zero'):
        raise ValueError("`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or be a value defining the center line or a callable that returns a float")
    rgbas: Optional[Union[List[str], List[List[str]]]] = None
    if cmap is not None:
        _matplotlib = __import__('matplotlib')
        if isinstance(cmap, str):
            cmap_obj = _matplotlib.colormaps[cmap]
        else:
            cmap_obj = cmap
        norm = _matplotlib.colors.Normalize(left_val, right_val)
        cmap_array = cmap_obj(norm(values))
        if data.ndim == 1:
            rgbas = [_matplotlib.colors.rgb2hex(rgba) for rgba in cmap_array]
        else:
            rgbas = [[_matplotlib.colors.rgb2hex(rgba) for rgba in row] for row in cmap_array]
    if data.ndim == 1:
        return [css_calc(x - z, left_val - z, right_val - z, align, colors if rgbas is None else rgbas[i]) 
                for i, x in enumerate(values)]
    else:
        return np.array([[css_calc(x - z, left_val - z, right_val - z, align, colors if rgbas is None else rgbas[i][j])
                           for j, x in enumerate(row)] for i, row in enumerate(values)])