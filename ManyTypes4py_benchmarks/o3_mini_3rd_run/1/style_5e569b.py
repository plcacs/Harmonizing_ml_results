from __future__ import annotations
import copy
import operator
from typing import Any, Callable, Optional, Sequence, Union
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.common import _is_bool_indexer


def _validate_apply_axis_arg(arg: Any, arg_name: str, dtype: Optional[Any], data: Union[Series, DataFrame]) -> np.ndarray:
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
    if isinstance(arg, Series) and isinstance(data, DataFrame):
        raise ValueError(f"'{arg_name}' is a Series but underlying data for operations is a DataFrame since 'axis=None'")
    if isinstance(arg, DataFrame) and isinstance(data, Series):
        raise ValueError(f"'{arg_name}' is a DataFrame but underlying data for operations is a Series with 'axis in [0,1]'")
    if isinstance(arg, (Series, DataFrame)):
        arg = arg.reindex_like(data).to_numpy(**dtype_kwargs)
    else:
        arg = np.asarray(arg, **dtype_kwargs)
        if arg.shape != data.shape:
            raise ValueError(f"supplied '{arg_name}' is not correct shape for data over selected 'axis': got {arg.shape}, expected {data.shape}")
    return arg


def _background_gradient(
    data: Union[Series, DataFrame],
    cmap: Union[str, Any] = 'PuBu',
    low: float = 0,
    high: float = 0,
    text_color_threshold: float = 0.408,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    gmap: Optional[Any] = None,
    text_only: bool = False,
) -> Union[list[str], DataFrame]:
    """
    Color background in a range according to the data or a gradient map
    """
    if gmap is None:
        gmap = data.to_numpy(dtype=float, na_value=np.nan)
    else:
        gmap = _validate_apply_axis_arg(gmap, 'gmap', float, data)
    smin: float = np.nanmin(gmap) if vmin is None else vmin
    smax: float = np.nanmax(gmap) if vmax is None else vmax
    rng: float = smax - smin
    _matplotlib: Any = import_optional_dependency('matplotlib', extra='Styler.background_gradient requires matplotlib.')
    norm = _matplotlib.colors.Normalize(smin - rng * low, smax + rng * high)
    if cmap is None:
        rgbas = _matplotlib.colormaps[_matplotlib.rcParams['image.cmap']](norm(gmap))
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
        r, g, b = (
            x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 
            for x in rgba[:3]
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def css(rgba: Sequence[float], text_only: bool) -> str:
        if not text_only:
            dark: bool = relative_luminance(rgba) < text_color_threshold
            text_color: str = '#f1f1f1' if dark else '#000000'
            return f'background-color: {_matplotlib.colors.rgb2hex(rgba)};color: {text_color};'
        else:
            return f'color: {_matplotlib.colors.rgb2hex(rgba)};'

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
    left: Optional[Union[float, Sequence[float]]] = None,
    right: Optional[Union[float, Sequence[float]]] = None,
    inclusive: Union[str, bool] = True,
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
    if isinstance(g_left, (DataFrame, Series)):
        g_left = g_left.where(pd.notna(g_left), False)
    l_right = ops[1](data, right) if right is not None else np.full(data.shape, True, dtype=bool)
    if isinstance(l_right, (DataFrame, Series)):
        l_right = l_right.where(pd.notna(l_right), False)
    return np.where(g_left & l_right, props, '')


def _highlight_value(data: Any, op: str, props: str) -> np.ndarray:
    """
    Return an array of css strings based on the condition of values matching an op.
    """
    value = getattr(data, op)(skipna=True)
    if isinstance(data, DataFrame):
        value = getattr(value, op)(skipna=True)
    cond = data == value
    cond = cond.where(pd.notna(cond), False)
    return np.where(cond, props, '')


def _bar(
    data: Union[Series, DataFrame],
    align: Union[str, int, float, Callable[[Any], float]],
    colors: Union[str, Sequence[str]],
    cmap: Optional[Union[str, Any]],
    width: float,
    height: float,
    vmin: Optional[float],
    vmax: Optional[float],
    base_css: str,
) -> Union[list[str], np.ndarray]:
    """
    Draw bar chart in data cells using HTML CSS linear gradient.

    Parameters
    ----------
    data : Series or DataFrame
        Underling subset of Styler data on which operations are performed.
    align : {"left", "right", "mid", "zero", "mean"} or numeric or callable
        Method for how bars are structured or scalar value of centre point.
    colors : str or sequence of str
        If a str is passed, the color is the same for both
        negative and positive numbers. If 2-tuple/list is used, the
        first element is the color_negative and the second is the
        color_positive.
    cmap : str or colormap, optional
        A string name of a matplotlib Colormap, or a Colormap object. Cannot be
        used together with ``color``.
    width : float
        The percentage of the cell, measured from left, where drawn bars will reside (as a fraction of 1).
    height : float
        The percentage of the cell's height where drawn bars will reside, centrally aligned (as a fraction of 1).
    vmin : float, optional
        Minimum bar value, defining the left hand limit
        of the bar drawing range, lower values are clipped to `vmin`.
    vmax : float, optional
        Maximum bar value, defining the right hand limit
        of the bar drawing range, higher values are clipped to `vmax`.
    base_css : str
        Additional CSS that is included in the cell before bars are drawn.
    """
    def css_bar(start: float, end: float, color: str) -> str:
        """
        Generate CSS code to draw a bar from start to end in a table cell.

        Uses linear-gradient.
        """
        cell_css: str = base_css
        if end > start:
            cell_css += 'background: linear-gradient(90deg,'
            if start > 0:
                cell_css += f' transparent {start * 100:.1f}%, {color} {start * 100:.1f}%,'
            cell_css += f' {color} {end * 100:.1f}%, transparent {end * 100:.1f}%)'
        return cell_css

    def css_calc(x: float, left: float, right: float, align: str, color: str) -> str:
        """
        Return the correct CSS for bar placement based on calculated values.
        """
        if pd.isna(x):
            return base_css
        if isinstance(color, (list, tuple)):
            color = color[0] if x < 0 else color[1]
        assert isinstance(color, str)
        x = left if x < left else x
        x = right if x > right else x
        start: float = 0
        end: float = 1
        if align == 'left':
            end = (x - left) / (right - left)
        elif align == 'right':
            start = (x - left) / (right - left)
        else:
            z_frac: float = 0.5
            if align == 'zero':
                limit: float = max(abs(left), abs(right))
                left, right = (-limit, limit)
            elif align == 'mid':
                mid: float = (left + right) / 2
                z_frac = -mid / (right - left) + 0.5 if mid < 0 else -left / (right - left)
            if x < 0:
                start, end = ((x - left) / (right - left), z_frac)
            else:
                start, end = (z_frac, (x - left) / (right - left))
        ret: str = css_bar(start * width, end * width, color)
        if height < 1 and 'background: linear-gradient(' in ret:
            return ret + f' no-repeat center; background-size: 100% {height * 100:.1f}%;'
        else:
            return ret

    values = data.to_numpy()
    left_bound: float = np.nanmin(data.min(skipna=True)) if vmin is None else vmin
    right_bound: float = np.nanmax(data.max(skipna=True)) if vmax is None else vmax
    z: float = 0
    if align == 'mid':
        if left_bound >= 0:
            align, left_bound = ('left', 0 if vmin is None else vmin)
        elif right_bound <= 0:
            align, right_bound = ('right', 0 if vmax is None else vmax)
    elif align == 'mean':
        z, align = (np.nanmean(values), 'zero')
    elif callable(align):
        z, align = (align(values), 'zero')
    elif isinstance(align, (float, int)):
        z, align = (float(align), 'zero')
    elif align not in ('left', 'right', 'zero'):
        raise ValueError("`align` should be in {'left', 'right', 'mid', 'mean', 'zero'} or be a value defining the center line or a callable that returns a float")
    rgbas: Optional[Union[list[str], list[list[str]]]] = None
    if cmap is not None:
        _matplotlib: Any = import_optional_dependency('matplotlib', extra='Styler.bar requires matplotlib.')
        cmap_obj = _matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        norm = _matplotlib.colors.Normalize(left_bound, right_bound)
        rgb_array = cmap_obj(norm(values))
        if data.ndim == 1:
            rgbas = [_matplotlib.colors.rgb2hex(rgba) for rgba in rgb_array]
        else:
            rgbas = [[_matplotlib.colors.rgb2hex(rgba) for rgba in row] for row in rgb_array]
    if data.ndim == 1:
        return [css_calc(x - z, left_bound - z, right_bound - z, align, colors if rgbas is None else rgbas[i]) for i, x in enumerate(values)]
    else:
        return np.array([[css_calc(x - z, left_bound - z, right_bound - z, align, colors if rgbas is None else rgbas[i][j]) for j, x in enumerate(row)] for i, row in enumerate(values)]) 


def import_optional_dependency(package: str, extra: str = "") -> Any:
    try:
        return __import__(package)
    except ImportError as err:
        raise ImportError(f"{package} is required. {extra}") from err